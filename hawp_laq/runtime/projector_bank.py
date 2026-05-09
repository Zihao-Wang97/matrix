from __future__ import annotations

import warnings
from pathlib import Path

import torch

from hawp_laq.config import resolve_projector_ranks, ProjectorConfig
from hawp_laq.modeling.attention_hawp import HAWPAttention
from hawp_laq.utils.io import save_json, load_json


def save_projectors(model: torch.nn.Module, output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    layer_data = {}
    for name, module in model.named_modules():
        if isinstance(module, HAWPAttention):
            layer_dir = output_dir / f"layer_{module.layer_idx}"
            layer_dir.mkdir(parents=True, exist_ok=True)
            data = {
                "p_k": module.p_k.data.cpu(),
                "p_v": module.p_v.data.cpu(),
                "gamma": module.gamma.data.cpu(),
                "r_k": module.r_k,
                "r_v": module.r_v,
                "logit_scale_mode": module.logit_scale_mode,
            }
            if module.d_v is not None:
                data["d_v"] = module.d_v.data.cpu()
            torch.save(data, layer_dir / "projector.pt")
            layer_data[str(module.layer_idx)] = {
                "r_k": module.r_k,
                "r_v": module.r_v,
            }
    save_json(layer_data, output_dir / "ranks.json")
    return output_dir


def load_projectors(
    model: torch.nn.Module,
    projector_dir: str | Path,
    strict: bool = True,
    expected_logit_scale_mode: str | None = None,
) -> None:
    projector_dir = Path(projector_dir)
    for name, module in model.named_modules():
        if isinstance(module, HAWPAttention):
            pt_path = projector_dir / f"layer_{module.layer_idx}" / "projector.pt"
            if not pt_path.exists():
                continue
            data = torch.load(pt_path, map_location="cpu", weights_only=True)
            data = normalize_projector_data(data, module.layer_idx)
            artifact_scale = data.get("logit_scale_mode")
            if expected_logit_scale_mode is not None and artifact_scale is not None:
                if artifact_scale != expected_logit_scale_mode:
                    raise ValueError(
                        f"Layer {module.layer_idx}: projector logit_scale_mode={artifact_scale!r} "
                        f"does not match configured hawp.logit_scale_mode="
                        f"{expected_logit_scale_mode!r}. Retrain/refine projectors with the "
                        f"same config used for inference."
                    )
            if module.gamma_mode == "learned" and not data.get("causal_mask", False):
                import warnings
                warnings.warn(
                    f"Layer {module.layer_idx}: projector was trained without causal mask "
                    f"but gamma_mode='learned' will apply gamma during inference. "
                    f"This may cause quality regression. Consider retraining projectors.",
                    UserWarning,
                    stacklevel=2,
                )
            module.load_projector_data(data, strict=strict)


def load_ranks(ranks_path: str | Path) -> dict[int, tuple[int, int]]:
    ranks_path = Path(ranks_path)
    if ranks_path.is_dir():
        ranks_path = ranks_path / "ranks.json"
    if not ranks_path.exists():
        return {}
    raw = load_json(ranks_path)
    if "selected_ranks" in raw:
        raw = raw["selected_ranks"]
    return {int(k): (v["r_k"], v["r_v"]) for k, v in raw.items()}


def normalize_projector_data(data: dict, layer_idx: int) -> dict:
    """Normalize projector data dict for backward compatibility.

    - If 'gamma' is missing but 'gamma_v'/'gamma_k' exists, synthesizes
      'gamma' (preferring gamma_v) and emits a warning.
    - If 'r_k' is missing, infers it from p_k.shape[1] when p_k is not
      square, and emits a warning.  Same for 'r_v'.
    - Square p_k/p_v (shape [d_h, d_h]) means the rank cannot be inferred;
      in that case no key is added.
    """
    if "gamma" not in data:
        if "gamma_v" in data:
            warnings.warn(
                f"Layer {layer_idx}: projector.pt missing 'gamma', using "
                f"'gamma_v' as fallback. Consider retraining projectors.",
                UserWarning,
                stacklevel=2,
            )
            data["gamma"] = data["gamma_v"]
        elif "gamma_k" in data:
            warnings.warn(
                f"Layer {layer_idx}: projector.pt missing 'gamma', using "
                f"'gamma_k' as fallback. Consider retraining projectors.",
                UserWarning,
                stacklevel=2,
            )
            data["gamma"] = data["gamma_k"]

    p_k = data.get("p_k")
    p_v = data.get("p_v")

    if "r_k" not in data and p_k is not None and p_k.ndim == 2 and p_k.shape[0] != p_k.shape[1]:
        inferred = p_k.shape[1]
        warnings.warn(
            f"Layer {layer_idx}: projector.pt missing 'r_k', inferred "
            f"r_k={inferred} from p_k shape {tuple(p_k.shape)}. "
            f"Consider retraining projectors.",
            UserWarning,
            stacklevel=2,
        )
        data["r_k"] = inferred

    if "r_v" not in data and p_v is not None and p_v.ndim == 2 and p_v.shape[0] != p_v.shape[1]:
        inferred = p_v.shape[1]
        warnings.warn(
            f"Layer {layer_idx}: projector.pt missing 'r_v', inferred "
            f"r_v={inferred} from p_v shape {tuple(p_v.shape)}. "
            f"Consider retraining projectors.",
            UserWarning,
            stacklevel=2,
        )
        data["r_v"] = inferred

    return data


def get_available_projector_layers(projector_dir: str | Path) -> set[int]:
    """Return set of layer indices that have a projector.pt file."""
    projector_dir = Path(projector_dir)
    if not projector_dir.exists():
        return set()
    available: set[int] = set()
    for d in sorted(projector_dir.iterdir()):
        if d.is_dir() and d.name.startswith("layer_"):
            pt_path = d / "projector.pt"
            if pt_path.exists():
                try:
                    idx = int(d.name.split("_", 1)[1])
                    available.add(idx)
                except ValueError:
                    pass
    return available


def _iter_layer_dirs(projector_dir: Path) -> list[tuple[int, Path]]:
    """Return sorted (layer_idx, dir_path) pairs for layer_*/ directories."""
    results = []
    if not projector_dir.exists():
        return results
    for d in sorted(projector_dir.iterdir()):
        if d.is_dir() and d.name.startswith("layer_"):
            try:
                idx = int(d.name.split("_", 1)[1])
                results.append((idx, d))
            except ValueError:
                pass
    return results


def rebuild_ranks_json(projector_dir: str | Path) -> Path:
    """Scan layer_*/projector.pt files and write ranks.json.

    Skips files missing ``r_k`` / ``r_v`` keys and emits a warning for each.
    Preserves existing entries in ranks.json for layers that do not yet have
    a projector.pt file (e.g. from rank_search selected_ranks.json).
    """
    projector_dir = Path(projector_dir)
    ranks_data: dict[str, dict[str, int]] = {}

    existing_ranks = load_ranks(projector_dir)
    for k, (rk, rv) in existing_ranks.items():
        ranks_data[str(k)] = {"r_k": rk, "r_v": rv}

    for layer_idx, layer_dir in _iter_layer_dirs(projector_dir):
        pt_path = layer_dir / "projector.pt"
        if not pt_path.exists():
            continue
        data = torch.load(pt_path, map_location="cpu", weights_only=False)
        data = normalize_projector_data(data, layer_idx)
        if "r_k" not in data or "r_v" not in data:
            warnings.warn(
                f"Layer {layer_idx} projector.pt is missing r_k/r_v keys; "
                f"skipping this layer in ranks.json. "
                f"This usually indicates a legacy-format projector file.",
                UserWarning,
                stacklevel=2,
            )
        else:
            ranks_data[str(layer_idx)] = {"r_k": data["r_k"], "r_v": data["r_v"]}

    if ranks_data:
        save_json(ranks_data, projector_dir / "ranks.json")
    return projector_dir / "ranks.json"


def inspect_projector_dir(
    projector_dir: str | Path,
    expected_head_dim: int,
    default_r_k: int,
    default_r_v: int,
    ranks_per_layer: dict[int, tuple[int, int]] | None = None,
) -> dict:
    """Scan projector directory and classify each layer file.

    Returns a dict with keys:
      valid_layers: list[int]           — files with compatible shapes
      legacy_layers: list[int]          — files missing r_k/r_v (old format)
      missing_rank_layers: list[int]    — files with r_k/r_v but shape mismatch
      shape_mismatch_layers: list[int]  — files whose p_k/p_v shape is incompatible
    """
    projector_dir = Path(projector_dir)
    result: dict[str, list[int]] = {
        "valid_layers": [],
        "legacy_layers": [],
        "missing_rank_layers": [],
        "shape_mismatch_layers": [],
    }

    for layer_idx, layer_dir in _iter_layer_dirs(projector_dir):
        pt_path = layer_dir / "projector.pt"
        if not pt_path.exists():
            continue

        try:
            data = torch.load(pt_path, map_location="cpu", weights_only=False)
        except Exception:
            result["shape_mismatch_layers"].append(layer_idx)
            continue

        data = normalize_projector_data(data, layer_idx)

        if "r_k" not in data or "r_v" not in data:
            result["legacy_layers"].append(layer_idx)
            continue

        layer_r_k = data["r_k"]
        layer_r_v = data["r_v"]

        p_k = data.get("p_k")
        p_v = data.get("p_v")

        if p_k is None or p_v is None:
            result["shape_mismatch_layers"].append(layer_idx)
            continue

        pk_ok = (
            p_k.shape == (expected_head_dim, expected_head_dim)
            or p_k.shape == (expected_head_dim, layer_r_k)
        )
        pv_ok = (
            p_v.shape == (expected_head_dim, expected_head_dim)
            or p_v.shape == (expected_head_dim, layer_r_v)
        )
        d_v = data.get("d_v")
        dv_ok = d_v is None or d_v.shape == (layer_r_v, expected_head_dim)

        if pk_ok and pv_ok and dv_ok:
            result["valid_layers"].append(layer_idx)
        else:
            result["shape_mismatch_layers"].append(layer_idx)

    return result


def _resolve_layer_ranks(
    layer_idx: int,
    head_dim: int,
    r_k: int | None = None,
    r_v: int | None = None,
    rank: int | None = None,
    ranks_per_layer: dict[int, tuple[int, int]] | None = None,
) -> tuple[int, int]:
    if ranks_per_layer and layer_idx in ranks_per_layer:
        return ranks_per_layer[layer_idx]
    if r_k is not None and r_v is not None:
        return r_k, r_v
    if rank is not None:
        return rank, rank
    raise ValueError(
        f"layer {layer_idx}: must provide r_k/r_v, rank, or ranks_per_layer"
    )


def train_all_layers(
    calib_dir: str | Path,
    n_layers: int,
    n_heads: int,
    rank: int | None = None,
    r_k: int | None = None,
    r_v: int | None = None,
    ranks_per_layer: dict[int, tuple[int, int]] | None = None,
    lr: float = 1e-3,
    n_steps: int = 200,
    orthogonalize_every: int = 10,
    w_logits: float = 1.0,
    w_attn: float = 1.0,
    w_value: float = 0.5,
    device: str = "cpu",
    output_dir: str | Path = "artifacts/projectors",
) -> Path:
    """Train projectors for all layers (legacy entry point).

    .. deprecated::
        This function is kept for backward compatibility and does not
        consume the full ProjectorConfig / RankSearchConfig. Prefer
        scripts/02_train_projectors.py or direct ProjectorTrainer usage
        with explicit riemannian_adam config.
    """
    warnings.warn(
        "train_all_layers is a legacy entry point and does not consume the full "
        "projector config; prefer scripts/02_train_projectors.py / ProjectorTrainer "
        "with explicit riemannian_adam config.",
        DeprecationWarning,
        stacklevel=2,
    )
    from hawp_laq.offline.projector_trainer import ProjectorTrainer
    from hawp_laq.utils.io import load_pt

    calib_dir = Path(calib_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for layer_idx in range(n_layers):
        layer_path = calib_dir / f"layer_{layer_idx}.pt"
        if not layer_path.exists():
            print(f"[train] layer {layer_idx}: no calib data, skipping")
            continue

        data = load_pt(layer_path)
        q = data["q"].float()
        k = data["k"].float()
        v = data["v"].float()
        d_model = q.shape[-1]
        head_dim = d_model // n_heads

        layer_r_k, layer_r_v = _resolve_layer_ranks(
            layer_idx, head_dim, r_k=r_k, r_v=r_v, rank=rank,
            ranks_per_layer=ranks_per_layer,
        )

        print(f"[train] layer {layer_idx}: d_model={d_model} r_k={layer_r_k} r_v={layer_r_v}")

        trainer_k = ProjectorTrainer(
            d_model=d_model, rank_k=layer_r_k, rank_v=layer_r_v, n_heads=n_heads,
            lr=lr, orthogonalize_every=orthogonalize_every,
            w_logits=w_logits, w_attn=w_attn, w_value=w_value,
            device=device,
        )
        result = trainer_k.train_one_group(q, k, v, n_steps=n_steps)

        ProjectorTrainer.save_result(result, layer_idx, output_dir)

    ranks_data = {}
    for layer_idx in range(n_layers):
        pt_path = output_dir / f"layer_{layer_idx}" / "projector.pt"
        if pt_path.exists():
            d = torch.load(pt_path, map_location="cpu", weights_only=False)
            ranks_data[str(layer_idx)] = {"r_k": d["r_k"], "r_v": d["r_v"]}
    if ranks_data:
        save_json(ranks_data, output_dir / "ranks.json")

    return output_dir
