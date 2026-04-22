from __future__ import annotations

import warnings
from pathlib import Path

import torch

from hawp_laq.offline.projector_trainer import ProjectorTrainer
from hawp_laq.utils.io import load_pt, save_json, save_pt


def _evaluate_rank(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    rank_k: int,
    rank_v: int,
    d_model: int,
    n_heads: int,
    n_steps: int,
    lr: float,
    orthogonalize_every: int,
    w_logits: float,
    w_attn: float,
    w_value: float,
    device: str,
) -> dict:
    trainer = ProjectorTrainer(
        d_model=d_model,
        rank_k=rank_k,
        rank_v=rank_v,
        n_heads=n_heads,
        lr=lr,
        orthogonalize_every=orthogonalize_every,
        w_logits=w_logits,
        w_attn=w_attn,
        w_value=w_value,
        device=device,
    )
    result = trainer.train_one_group(q, k, v, n_steps=n_steps)
    metrics = result["metrics"]
    return {
        "rank_k": rank_k,
        "rank_v": rank_v,
        "final_loss": metrics["total"][-1],
        "final_logits_loss": metrics["logits"][-1],
        "final_attn_loss": metrics["attn"][-1],
        "final_value_loss": metrics["value"][-1],
        "p_k_shape": tuple(result["p_k"].shape),
        "p_v_shape": tuple(result["p_v"].shape),
    }


def _selection_score(result: dict, value_weight: float) -> float:
    return result["final_attn_loss"] + value_weight * result["final_value_loss"]


def search_rank_per_layer(
    calib_dir: str | Path,
    rank_candidates: list[int],
    tolerance: float = 0.02,
    n_steps: int = 1500,
    lr: float = 1e-3,
    orthogonalize_every: int = 10,
    w_logits: float = 1.0,
    w_attn: float = 1.0,
    w_value: float = 0.5,
    device: str = "cpu",
    output_dir: str | Path | None = None,
    selection_value_weight: float = 0.25,
    selection_abs_tolerance: float = 0.04,
    selection_metric: str = "attn_value_abs",
) -> dict[int, tuple[int, int]]:
    if selection_metric != "attn_value_abs":
        raise ValueError(
            f"Unsupported selection_metric='{selection_metric}'. "
            f"Currently only 'attn_value_abs' is implemented."
        )

    calib_dir = Path(calib_dir)
    meta = load_pt(calib_dir / "meta.pt")
    n_layers = meta.get("n_layers", 0)
    n_heads = meta.get("n_heads")

    if n_layers == 0:
        for p in sorted(calib_dir.glob("layer_*.pt")):
            idx = int(p.stem.split("_")[1])
            n_layers = max(n_layers, idx + 1)

    selected_ranks: dict[int, tuple[int, int]] = {}

    for layer_idx in range(n_layers):
        layer_path = calib_dir / f"layer_{layer_idx}.pt"
        if not layer_path.exists():
            print(f"[rank_search] layer {layer_idx}: calib data not found, skipping")
            continue

        data = load_pt(layer_path)
        q = data["q"].float()
        k = data["k"].float()
        v = data["v"].float()
        d_model = q.shape[-1]

        if n_heads is None:
            from transformers import AutoConfig
            cfg_auto = AutoConfig.from_pretrained(meta.get("model_id", "facebook/opt-125m"))
            n_heads = cfg_auto.num_attention_heads

        print(f"\n[rank_search] layer {layer_idx}: d_model={d_model} n_heads={n_heads}")
        head_dim = d_model // n_heads

        valid_candidates = [r for r in rank_candidates if 1 <= r <= head_dim]
        removed = [r for r in rank_candidates if r not in valid_candidates]
        if removed:
            warnings.warn(
                f"[rank_search] layer {layer_idx}: head_dim={head_dim}, "
                f"filtering out rank candidates exceeding head_dim: {removed}. "
                f"Valid candidates: {valid_candidates}",
                UserWarning,
                stacklevel=2,
            )
        if not valid_candidates:
            raise ValueError(
                f"[rank_search] layer {layer_idx}: head_dim={head_dim}, "
                f"no valid rank candidates remain after filtering. "
                f"Original candidates: {rank_candidates}. "
                f"All candidates exceed head_dim."
            )

        print(
            f"[rank_search]   candidates={valid_candidates}"
            f"  n_steps={n_steps}"
            f"  selection=attn_value_abs"
            f"  value_weight={selection_value_weight}"
            f"  abs_tol={selection_abs_tolerance}"
        )

        sorted_candidates = sorted(valid_candidates)
        results = []
        baseline_result = None

        for rank in sorted_candidates:
            r = _evaluate_rank(
                q, k, v, rank, rank, d_model, n_heads,
                n_steps, lr, orthogonalize_every,
                w_logits, w_attn, w_value, device,
            )
            r["selection_score"] = _selection_score(r, selection_value_weight)
            results.append(r)
            print(
                f"  rank=({rank},{rank})"
                f"  total={r['final_loss']:.6f}"
                f"  logits={r['final_logits_loss']:.6f}"
                f"  attn={r['final_attn_loss']:.6f}"
                f"  value={r['final_value_loss']:.6f}"
                f"  select={r['selection_score']:.6f}"
            )

            if rank == sorted_candidates[-1]:
                baseline_result = r

        if baseline_result is None:
            continue

        baseline_selection_score = baseline_result["selection_score"]
        chosen = sorted_candidates[-1]
        for r in results:
            if r["selection_score"] <= baseline_selection_score + selection_abs_tolerance:
                chosen = r["rank_k"]
                break

        selected_ranks[layer_idx] = (chosen, chosen)
        print(
            f"[rank_search] layer {layer_idx}: selected (r_k, r_v)=({chosen}, {chosen})"
            f"  (baseline_select={baseline_selection_score:.6f}"
            f"  abs_tol={selection_abs_tolerance})"
        )

        if output_dir is not None:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            save_json(
                {
                    "layer_idx": layer_idx,
                    "selected_r_k": chosen,
                    "selected_r_v": chosen,
                    "baseline_selection_score": baseline_selection_score,
                    "selection_metric": "attn_value_abs",
                    "selection_value_weight": selection_value_weight,
                    "selection_abs_tolerance": selection_abs_tolerance,
                    "tolerance": tolerance,
                    "candidates": sorted_candidates,
                    "results": results,
                },
                out / f"layer_{layer_idx}_rank_search.json",
            )

    return selected_ranks


def run_rank_search_from_config(config) -> dict[int, tuple[int, int]]:
    from hawp_laq.config import HAWPLAQConfig

    calib_dir = config.calib.output_dir
    output_dir = config.rank_search.output_dir
    n_steps = getattr(config.rank_search, "n_steps", None)
    if n_steps is None:
        n_steps = config.projector.n_steps

    return search_rank_per_layer(
        calib_dir=calib_dir,
        rank_candidates=config.rank_search.rank_candidates,
        tolerance=config.rank_search.tolerance,
        n_steps=n_steps,
        lr=config.projector.lr,
        orthogonalize_every=config.projector.orthogonalize_every,
        w_logits=config.projector.w_logits,
        w_attn=config.projector.w_attn,
        w_value=config.projector.w_value,
        device=config.train.device,
        output_dir=output_dir,
        selection_value_weight=config.rank_search.selection_value_weight,
        selection_abs_tolerance=config.rank_search.selection_abs_tolerance,
        selection_metric=config.rank_search.selection_metric,
    )
