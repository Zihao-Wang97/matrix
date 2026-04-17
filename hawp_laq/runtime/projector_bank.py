from __future__ import annotations

from pathlib import Path

import torch

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
            }
            torch.save(data, layer_dir / "projector.pt")
            layer_data[str(module.layer_idx)] = {
                "r_k": module.r_k,
                "r_v": module.r_v,
            }
    save_json(layer_data, output_dir / "ranks.json")
    return output_dir


def load_projectors(model: torch.nn.Module, projector_dir: str | Path) -> None:
    projector_dir = Path(projector_dir)
    for name, module in model.named_modules():
        if isinstance(module, HAWPAttention):
            pt_path = projector_dir / f"layer_{module.layer_idx}" / "projector.pt"
            if not pt_path.exists():
                continue
            data = torch.load(pt_path, map_location="cpu", weights_only=True)
            p_k = data["p_k"].to(module.p_k.device, module.p_k.dtype)
            p_v = data["p_v"].to(module.p_v.device, module.p_v.dtype)
            if p_k.shape == module.p_k.shape:
                module.p_k.data.copy_(p_k)
            if p_v.shape == module.p_v.shape:
                module.p_v.data.copy_(p_v)
            if "gamma" in data:
                module.gamma.data.copy_(
                    data["gamma"].to(module.gamma.device, module.gamma.dtype),
                )


def load_ranks(ranks_path: str | Path) -> dict[int, tuple[int, int]]:
    ranks_path = Path(ranks_path)
    if ranks_path.is_dir():
        ranks_path = ranks_path / "ranks.json"
    if not ranks_path.exists():
        return {}
    raw = load_json(ranks_path)
    return {int(k): (v["r_k"], v["r_v"]) for k, v in raw.items()}


def train_all_layers(
    calib_dir: str | Path,
    n_layers: int,
    n_heads: int,
    rank: int = 64,
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

        if ranks_per_layer and layer_idx in ranks_per_layer:
            r_k, r_v = ranks_per_layer[layer_idx]
        else:
            r_k = rank
            r_v = rank

        print(f"[train] layer {layer_idx}: d_model={d_model} r_k={r_k} r_v={r_v}")

        trainer_k = ProjectorTrainer(
            d_model=d_model, rank=r_k, n_heads=n_heads,
            lr=lr, orthogonalize_every=orthogonalize_every,
            w_logits=w_logits, w_attn=w_attn, w_value=w_value,
            device=device,
        )
        result_k = trainer_k.train_one_group(q, k, v, n_steps=n_steps)

        trainer_v = ProjectorTrainer(
            d_model=d_model, rank=r_v, n_heads=n_heads,
            lr=lr, orthogonalize_every=orthogonalize_every,
            w_logits=w_logits, w_attn=w_attn, w_value=w_value,
            device=device,
        )
        result_v = trainer_v.train_one_group(q, k, v, n_steps=n_steps)

        combined = {
            "p_k": result_k["p_k"],
            "p_v": result_v["p_v"],
            "gamma_k": result_k["gamma"],
            "gamma_v": result_v["gamma"],
            "r_k": r_k,
            "r_v": r_v,
        }
        torch.save(combined, output_dir / f"layer_{layer_idx}" / "projector.pt")

    return output_dir
