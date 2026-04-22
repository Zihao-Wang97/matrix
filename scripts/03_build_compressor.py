#!/usr/bin/env python
"""Build compressor package: python scripts/03_build_compressor.py [config]"""

import argparse
from pathlib import Path

from transformers import AutoConfig

from hawp_laq.config import load_config, load_projector_ranks_from_dir
from hawp_laq.runtime.compressor import CompressorPackage
from hawp_laq.utils.io import load_pt, load_json


def main() -> None:
    parser = argparse.ArgumentParser(description="HAWP-LAQ compressor package builder")
    parser.add_argument(
        "config",
        nargs="?",
        default=None,
        help="Path to yaml config (default: configs/run_server.yaml)",
    )
    parser.add_argument("--projector-dir", type=str, default=None, help="Override projector directory")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output directory")
    args = parser.parse_args()

    if args.config is None:
        args.config = Path(__file__).resolve().parent.parent / "configs" / "run_server.yaml"

    cfg = load_config(args.config)
    projector_dir = Path(args.projector_dir) if args.projector_dir else cfg.projector.output_dir
    output_dir = Path(args.output_dir) if args.output_dir else Path("artifacts/compressor")

    print("=" * 60)
    print("[compressor] building compressor package")
    print(f"[compressor] projector_dir = {projector_dir}")
    print(f"[compressor] output_dir    = {output_dir}")
    print("=" * 60)

    model_cfg = AutoConfig.from_pretrained(cfg.model.model_id)
    n_layers = getattr(model_cfg, "num_hidden_layers", 12)
    n_heads = getattr(model_cfg, "num_attention_heads", 12)
    head_dim = getattr(model_cfg, "hidden_size", 768) // n_heads

    pkg = CompressorPackage(
        projector_dir=projector_dir,
        n_layers=n_layers,
        n_heads=n_heads,
        head_dim=head_dim,
        k_group_size=cfg.quant.k_group_size,
        v_group_size=cfg.quant.v_group_size,
        use_rotation_for_k=cfg.quant.use_rotation_for_k,
        use_rotation_for_v=cfg.quant.use_rotation_for_v,
        outlier_threshold=cfg.quant.outlier_threshold,
        total_budget=cfg.sched.total_budget,
        recent_window=cfg.sched.recent_window,
        high_ratio=cfg.sched.high_ratio,
        low_ratio=cfg.sched.low_ratio,
    )

    ranks = pkg.ranks
    print(f"[compressor] loaded {len(ranks)} layer projectors")
    for idx, (r_k, r_v) in sorted(ranks.items()):
        print(f"  layer {idx}: r_k={r_k}  r_v={r_v}")
    for layer_idx in range(n_layers):
        if layer_idx not in ranks:
            print(f"  layer {layer_idx}: has_projector=False  (full-rank in profile only, low-rank NOT active)")

    per_layer_ranks = load_projector_ranks_from_dir(projector_dir)
    if per_layer_ranks:
        print(f"[compressor] per-layer ranks loaded from {projector_dir / 'ranks.json'}")
        for idx, (rk, rv) in sorted(per_layer_ranks.items()):
            print(f"  layer {idx}: r_k={rk}  r_v={rv}")

    saved_dir = pkg.save(output_dir)
    print(f"\n[compressor] package saved to {saved_dir}")

    missing_rank_layers = []
    for idx, data in pkg._projectors.items():
        if "r_k" not in data or "r_v" not in data:
            missing_rank_layers.append(idx)
    if missing_rank_layers:
        print(
            f"[compressor] NOTE: {len(missing_rank_layers)} layer(s) had missing r_k/r_v "
            f"keys in projector files; profiled with head_dim fallback. "
            f"Missing-rank fallback in compressor profiling is warning-backed "
            f"and profile-only, not reflecting actual low-rank configuration."
        )
    print(f"\n[compressor] package saved to {saved_dir}")
    print(f"[compressor] contents:")
    for f in sorted(saved_dir.rglob("*")):
        if f.is_file():
            print(f"  {f.relative_to(saved_dir)}")

    print("\n[compressor] KV memory profiles:")
    for seq_len in (512, 1024, 2048, 4096, 8192):
        summary = pkg.total_kv_bytes(seq_len)
        print(
            f"  seq_len={seq_len:>5d}: "
            f"baseline={summary['baseline_formatted']}  "
            f"latent={summary['latent_formatted']}  "
            f"quant={summary['quant_formatted']}  "
            f"saving(latent)={summary['latent_saving_ratio']:.1%}  "
            f"saving(quant)={summary['quant_saving_ratio']:.1%}"
        )


if __name__ == "__main__":
    main()
