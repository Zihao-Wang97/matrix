#!/usr/bin/env python
"""Train projectors.

Modes:
  single_group  Train one layer's KV projector group (default)
  rank_search   Scan candidate ranks per layer, output optimal rank config

Usage:
  python scripts/02_train_projectors.py configs/dev_local.yaml --mode single_group --layer 0
  python scripts/02_train_projectors.py configs/dev_local.yaml --mode rank_search
"""

import argparse
import shutil
from pathlib import Path

import torch
from transformers import AutoConfig

from hawp_laq.config import load_config, resolve_projector_ranks
from hawp_laq.offline.projector_trainer import ProjectorTrainer
from hawp_laq.offline.rank_search import search_rank_per_layer, run_rank_search_from_config
from hawp_laq.runtime.projector_bank import load_ranks, rebuild_ranks_json
from hawp_laq.utils.io import load_pt, save_json


def _run_single_group(cfg, layer_idx: int, clean_output_dir: bool = False) -> None:
    output_dir = cfg.projector.output_dir
    if clean_output_dir and Path(output_dir).exists():
        print(f"[single_group] --clean-output-dir: removing {output_dir}")
        shutil.rmtree(output_dir)

    calib_dir = cfg.calib.output_dir
    meta = load_pt(Path(calib_dir) / "meta.pt")
    n_heads = meta.get("n_heads")
    if n_heads is None:
        model_cfg = AutoConfig.from_pretrained(cfg.model.model_id, local_files_only=Path(cfg.model.model_id).expanduser().is_dir())
        n_heads = model_cfg.num_attention_heads

    capture_mode = meta.get("capture_mode", "pre_rope")
    model_type = meta.get("model_type", "")
    _NON_ROPE_MODEL_TYPES = {"opt", "gpt_neox"}

    if not model_type:
        model_cfg = AutoConfig.from_pretrained(cfg.model.model_id, local_files_only=Path(cfg.model.model_id).expanduser().is_dir())
        model_type = getattr(model_cfg, "model_type", "")

    is_rope = model_type.lower() not in _NON_ROPE_MODEL_TYPES
    if is_rope and capture_mode != "post_rope":
        raise ValueError(
            f"RoPE model ({model_type}) requires post_rope calibration data, "
            f"but got capture_mode={capture_mode}. "
            f"Re-run calibration with calib.capture_mode=post_rope (or auto)."
        )

    layer_data = load_pt(Path(calib_dir) / f"layer_{layer_idx}.pt")
    q = layer_data["q"].float()
    k = layer_data["k"].float()
    v = layer_data["v"].float()
    d_model = q.shape[-1]
    head_dim = d_model // n_heads

    r_k, r_v = None, None

    existing_ranks = load_ranks(output_dir)
    if layer_idx in existing_ranks:
        r_k, r_v = existing_ranks[layer_idx]
        print(f"[single_group] layer {layer_idx}: using rank from ranks.json: r_k={r_k} r_v={r_v}")

    if r_k is None or r_v is None:
        r_k, r_v = resolve_projector_ranks(cfg.projector, head_dim=head_dim, mode="single_group")

    print("=" * 60)
    print(f"[single_group] layer={layer_idx}  d_model={d_model}  n_heads={n_heads}  head_dim={head_dim}")
    print(f"[single_group] r_k={r_k}  r_v={r_v}")
    print(f"[single_group] q={tuple(q.shape)} k={tuple(k.shape)} v={tuple(v.shape)}")
    print(f"[single_group] n_steps={cfg.projector.n_steps}  lr={cfg.projector.lr}")
    print("=" * 60)

    trainer = ProjectorTrainer(
        d_model=d_model,
        rank_k=r_k,
        rank_v=r_v,
        n_heads=n_heads,
        lr=cfg.projector.lr,
        orthogonalize_every=cfg.projector.orthogonalize_every,
        w_logits=cfg.projector.w_logits,
        w_attn=cfg.projector.w_attn,
        w_value=cfg.projector.w_value,
        device=cfg.train.device,
    )

    result = trainer.train_one_group(q, k, v, n_steps=cfg.projector.n_steps)
    first_loss = result["metrics"]["total"][0]
    last_loss = result["metrics"]["total"][-1]
    print(f"\n[result] loss: {first_loss:.6f} -> {last_loss:.6f}  (delta={last_loss - first_loss:.6f})")

    ProjectorTrainer.save_result(result, layer_idx, cfg.projector.output_dir)

    ranks_path = rebuild_ranks_json(cfg.projector.output_dir)
    print(f"[single_group] rebuilt ranks.json at {ranks_path}")


def _run_rank_search(cfg) -> None:
    rk_cands = getattr(cfg.rank_search, "r_k_candidates", None)
    rv_cands = getattr(cfg.rank_search, "r_v_candidates", None)
    pair_cands = getattr(cfg.rank_search, "rank_pair_candidates", None)
    legacy = cfg.rank_search.rank_candidates

    print("=" * 60)
    if pair_cands:
        print(f"[rank_search] rank_pair_candidates={pair_cands}")
    elif rk_cands and rv_cands:
        print(f"[rank_search] r_k_candidates={rk_cands}  "
              f"r_v_candidates={rv_cands}")
    else:
        print(f"[rank_search] rank_candidates (legacy symmetric)={legacy}")
    print(f"[rank_search] n_steps={cfg.rank_search.n_steps}  "
          f"lr={cfg.projector.lr}  device={cfg.train.device}")
    print(f"[rank_search] selection=constraint"
          f"  rel_tol={cfg.rank_search.relative_tolerance}"
          f"  logits_abs_tol={cfg.rank_search.logits_abs_tolerance}"
          f"  attn_abs_tol={cfg.rank_search.attn_abs_tolerance}"
          f"  value_abs_tol={cfg.rank_search.value_abs_tolerance}")
    print("=" * 60)

    selected = run_rank_search_from_config(cfg)

    print(f"\n{'='*60}")
    print("[rank_search] summary")
    print(f"  {'layer':>6} {'r_k':>6} {'r_v':>6}")
    print(f"  {'-'*20}")
    for idx in sorted(selected.keys()):
        rk, rv = selected[idx]
        print(f"  {idx:>6d} {rk:>6d} {rv:>6d}")

    out_dir = Path(cfg.rank_search.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(
        {"selected_ranks": {str(k): {"r_k": v[0], "r_v": v[1]} for k, v in selected.items()}},
        out_dir / "selected_ranks.json",
    )
    print(f"\n[rank_search] saved to {out_dir / 'selected_ranks.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="HAWP-LAQ projector training (r_k / r_v)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  single_group  Train one layer's KV projector group (default)
  rank_search   Scan candidate ranks per layer, output optimal (r_k, r_v) config

Examples:
  python scripts/02_train_projectors.py configs/dev_local.yaml
  python scripts/02_train_projectors.py configs/dev_local.yaml --mode single_group --layer 3
  python scripts/02_train_projectors.py configs/dev_local.yaml --mode rank_search
  python scripts/02_train_projectors.py configs/dev_local.yaml --mode rank_search --ranks 16 32 64
""",
    )
    parser.add_argument(
        "config",
        nargs="?",
        default=None,
        help="Path to yaml config (default: configs/dev_local.yaml)",
    )
    parser.add_argument(
        "--mode",
        choices=["single_group", "rank_search"],
        default="single_group",
        help="Training mode (default: single_group)",
    )
    parser.add_argument("--layer", type=int, default=None, help="Target layer index (single_group mode)")
    parser.add_argument("--ranks", nargs="+", type=int, default=None, help="Override rank candidates (rank_search mode)")
    parser.add_argument("--relative-tolerance", type=float, default=None, help="Override relative tolerance (rank_search mode)")
    parser.add_argument("--logits-abs-tolerance", type=float, default=None, help="Override logits abs tolerance (rank_search mode)")
    parser.add_argument("--attn-abs-tolerance", type=float, default=None, help="Override attn abs tolerance (rank_search mode)")
    parser.add_argument("--value-abs-tolerance", type=float, default=None, help="Override value abs tolerance (rank_search mode)")
    parser.add_argument(
        "--clean-output-dir",
        action="store_true",
        default=False,
        help="Delete projector output dir before training (avoid stale artifacts)",
    )
    args = parser.parse_args()

    if args.config is None:
        args.config = Path(__file__).resolve().parent.parent / "configs" / "dev_local.yaml"

    cfg = load_config(args.config)

    if args.ranks is not None:
        cfg.rank_search.rank_candidates = args.ranks
        cfg.rank_search.r_k_candidates = None
        cfg.rank_search.r_v_candidates = None
        cfg.rank_search.rank_pair_candidates = None
    if args.relative_tolerance is not None:
        cfg.rank_search.relative_tolerance = args.relative_tolerance
    if args.logits_abs_tolerance is not None:
        cfg.rank_search.logits_abs_tolerance = args.logits_abs_tolerance
    if args.attn_abs_tolerance is not None:
        cfg.rank_search.attn_abs_tolerance = args.attn_abs_tolerance
    if args.value_abs_tolerance is not None:
        cfg.rank_search.value_abs_tolerance = args.value_abs_tolerance

    if args.mode == "single_group":
        layer_idx = args.layer if args.layer is not None else cfg.projector.target_layer
        _run_single_group(cfg, layer_idx, clean_output_dir=args.clean_output_dir)
    elif args.mode == "rank_search":
        _run_rank_search(cfg)


if __name__ == "__main__":
    main()
