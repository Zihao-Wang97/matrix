from __future__ import annotations

from pathlib import Path

import torch

from hawp_laq.offline.projector_trainer import ProjectorTrainer
from hawp_laq.utils.io import load_pt, save_json, save_pt


def _evaluate_rank(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    rank: int,
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
        rank=rank,
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
        "rank": rank,
        "final_loss": metrics["total"][-1],
        "final_logits_loss": metrics["logits"][-1],
        "final_attn_loss": metrics["attn"][-1],
        "final_value_loss": metrics["value"][-1],
        "p_k_shape": tuple(result["p_k"].shape),
        "p_v_shape": tuple(result["p_v"].shape),
    }


def search_rank_per_layer(
    calib_dir: str | Path,
    rank_candidates: list[int],
    tolerance: float = 0.02,
    n_steps: int = 200,
    lr: float = 1e-3,
    orthogonalize_every: int = 10,
    w_logits: float = 1.0,
    w_attn: float = 1.0,
    w_value: float = 0.5,
    device: str = "cpu",
    output_dir: str | Path | None = None,
) -> dict[int, int]:
    calib_dir = Path(calib_dir)
    meta = load_pt(calib_dir / "meta.pt")
    n_layers = meta.get("n_layers", 0)
    n_heads = meta.get("n_heads")

    if n_layers == 0:
        for p in sorted(calib_dir.glob("layer_*.pt")):
            idx = int(p.stem.split("_")[1])
            n_layers = max(n_layers, idx + 1)

    selected_ranks: dict[int, int] = {}

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
        print(f"[rank_search]   candidates={rank_candidates}  tolerance={tolerance}")

        sorted_candidates = sorted(rank_candidates)
        results = []
        baseline_result = None

        for rank in sorted_candidates:
            r = _evaluate_rank(
                q, k, v, rank, d_model, n_heads,
                n_steps, lr, orthogonalize_every,
                w_logits, w_attn, w_value, device,
            )
            results.append(r)
            print(f"  rank={rank:>4d}  loss={r['final_loss']:.6f}")

            if rank == sorted_candidates[-1]:
                baseline_result = r

        if baseline_result is None:
            continue

        baseline_loss = baseline_result["final_loss"]
        chosen = sorted_candidates[-1]
        for r in results:
            if r["final_loss"] <= baseline_loss * (1.0 + tolerance):
                chosen = r["rank"]
                break

        selected_ranks[layer_idx] = chosen
        print(f"[rank_search] layer {layer_idx}: selected rank={chosen}  (baseline_loss={baseline_loss:.6f})")

        if output_dir is not None:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            save_json(
                {
                    "layer_idx": layer_idx,
                    "selected_rank": chosen,
                    "baseline_loss": baseline_loss,
                    "tolerance": tolerance,
                    "candidates": sorted_candidates,
                    "results": results,
                },
                out / f"layer_{layer_idx}_rank_search.json",
            )

    return selected_ranks


def run_rank_search_from_config(config) -> dict[int, int]:
    from hawp_laq.config import HAWPLAQConfig

    calib_dir = config.calib.output_dir
    output_dir = config.rank_search.output_dir

    return search_rank_per_layer(
        calib_dir=calib_dir,
        rank_candidates=config.rank_search.rank_candidates,
        tolerance=config.rank_search.tolerance,
        n_steps=config.projector.n_steps,
        lr=config.projector.lr,
        orthogonalize_every=config.projector.orthogonalize_every,
        w_logits=config.projector.w_logits,
        w_attn=config.projector.w_attn,
        w_value=config.projector.w_value,
        device=config.train.device,
        output_dir=output_dir,
    )
