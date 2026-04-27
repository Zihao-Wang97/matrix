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


def _component_pass(
    result_loss: float,
    baseline_loss: float,
    relative_tolerance: float,
    abs_tolerance: float,
    eps: float = 1e-8,
) -> bool:
    if baseline_loss >= eps:
        return result_loss <= baseline_loss * (1.0 + relative_tolerance)
    return result_loss <= abs_tolerance


def search_rank_per_layer(
    calib_dir: str | Path,
    rank_candidates: list[int],
    n_steps: int = 1500,
    lr: float = 1e-3,
    orthogonalize_every: int = 10,
    w_logits: float = 1.0,
    w_attn: float = 1.0,
    w_value: float = 0.5,
    device: str = "cpu",
    output_dir: str | Path | None = None,
    relative_tolerance: float = 0.10,
    logits_abs_tolerance: float = 1e-6,
    attn_abs_tolerance: float = 1e-5,
    value_abs_tolerance: float = 1e-4,
) -> dict[int, tuple[int, int]]:

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
            f"  selection=constraint"
            f"  rel_tol={relative_tolerance}"
            f"  logits_abs_tol={logits_abs_tolerance}"
            f"  attn_abs_tol={attn_abs_tolerance}"
            f"  value_abs_tol={value_abs_tolerance}"
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
            results.append(r)
            print(
                f"  rank=({rank},{rank})"
                f"  total={r['final_loss']:.6f}"
                f"  logits={r['final_logits_loss']:.6f}"
                f"  attn={r['final_attn_loss']:.6f}"
                f"  value={r['final_value_loss']:.6f}"
            )

            if rank == sorted_candidates[-1]:
                baseline_result = r

        if baseline_result is None:
            continue

        for r in results:
            r["logits_pass"] = _component_pass(
                r["final_logits_loss"], baseline_result["final_logits_loss"],
                relative_tolerance, logits_abs_tolerance,
            )
            r["attn_pass"] = _component_pass(
                r["final_attn_loss"], baseline_result["final_attn_loss"],
                relative_tolerance, attn_abs_tolerance,
            )
            r["value_pass"] = _component_pass(
                r["final_value_loss"], baseline_result["final_value_loss"],
                relative_tolerance, value_abs_tolerance,
            )
            r["all_pass"] = r["logits_pass"] and r["attn_pass"] and r["value_pass"]

        print(f"\n  --- constraint check (rel_tol={relative_tolerance}) ---")
        print(
            f"  {'rank':>10}"
            f"  {'logits':>10}"
            f"  {'attn':>10}"
            f"  {'value':>10}"
            f"  {'result':>8}"
        )
        for r in results:
            tag = "PASS" if r["all_pass"] else "REJECT"
            print(
                f"  ({r['rank_k']},{r['rank_v']:>2})"
                f"  {'PASS' if r['logits_pass'] else 'FAIL':>10}"
                f"  {'PASS' if r['attn_pass'] else 'FAIL':>10}"
                f"  {'PASS' if r['value_pass'] else 'FAIL':>10}"
                f"  {tag:>8}"
            )

        chosen = sorted_candidates[-1]
        for r in results:
            if r["all_pass"]:
                chosen = r["rank_k"]
                break

        selected_ranks[layer_idx] = (chosen, chosen)
        print(
            f"[rank_search] layer {layer_idx}: selected (r_k, r_v)=({chosen}, {chosen})"
            f"  (rel_tol={relative_tolerance})"
        )

        if output_dir is not None:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            save_json(
                {
                    "layer_idx": layer_idx,
                    "selected_r_k": chosen,
                    "selected_r_v": chosen,
                    "selection_method": "constraint",
                    "relative_tolerance": relative_tolerance,
                    "logits_abs_tolerance": logits_abs_tolerance,
                    "attn_abs_tolerance": attn_abs_tolerance,
                    "value_abs_tolerance": value_abs_tolerance,
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
        n_steps=n_steps,
        lr=config.projector.lr,
        orthogonalize_every=config.projector.orthogonalize_every,
        w_logits=config.projector.w_logits,
        w_attn=config.projector.w_attn,
        w_value=config.projector.w_value,
        device=config.train.device,
        output_dir=output_dir,
        relative_tolerance=config.rank_search.relative_tolerance,
        logits_abs_tolerance=config.rank_search.logits_abs_tolerance,
        attn_abs_tolerance=config.rank_search.attn_abs_tolerance,
        value_abs_tolerance=config.rank_search.value_abs_tolerance,
    )
