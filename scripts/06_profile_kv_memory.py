#!/usr/bin/env python
"""KV memory profile: python scripts/06_profile_kv_memory.py [config]"""

import argparse
from pathlib import Path

from transformers import AutoConfig

from hawp_laq.config import load_config
from hawp_laq.runtime.compressor import CompressorPackage
from hawp_laq.utils.io import save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="HAWP-LAQ KV memory profiling")
    parser.add_argument(
        "config",
        nargs="?",
        default=None,
        help="Path to yaml config (default: configs/run_server.yaml)",
    )
    parser.add_argument("--projector-dir", type=str, default=None, help="Override projector directory")
    parser.add_argument("--seq-lens", nargs="+", type=int, default=[512, 1024, 2048, 4096, 8192, 16384])
    parser.add_argument("--output", type=str, default=None, help="Output json path")
    args = parser.parse_args()

    if args.config is None:
        args.config = Path(__file__).resolve().parent.parent / "configs" / "run_server.yaml"

    cfg = load_config(args.config)
    projector_dir = Path(args.projector_dir) if args.projector_dir else cfg.projector.output_dir

    model_cfg = AutoConfig.from_pretrained(cfg.model.model_id)
    n_layers = getattr(model_cfg, "num_hidden_layers", 12)
    n_heads = getattr(model_cfg, "num_attention_heads", 12)
    head_dim = getattr(model_cfg, "hidden_size", 768) // n_heads

    print("=" * 60)
    print("[profile] KV memory profiling")
    print(f"[profile] model={cfg.model.model_id}")
    print(f"[profile] n_layers={n_layers}  n_heads={n_heads}  head_dim={head_dim}")
    print(f"[profile] projector_dir={projector_dir}")
    print(f"[profile] k_group_size={cfg.quant.k_group_size}  v_group_size={cfg.quant.v_group_size}")
    print("=" * 60)

    pkg = CompressorPackage(
        projector_dir=projector_dir,
        n_layers=n_layers,
        n_heads=n_heads,
        head_dim=head_dim,
        k_group_size=cfg.quant.k_group_size,
        v_group_size=cfg.quant.v_group_size,
        use_rotation=cfg.quant.use_rotation,
        outlier_threshold=cfg.quant.outlier_threshold,
        total_budget=cfg.sched.total_budget,
        recent_window=cfg.sched.recent_window,
        high_ratio=cfg.sched.high_ratio,
        low_ratio=cfg.sched.low_ratio,
    )

    all_results = []
    for seq_len in args.seq_lens:
        summary = pkg.total_kv_bytes(seq_len)
        per_layer = summary.pop("per_layer")

        print(f"\n--- seq_len={seq_len} ---")
        print(f"  baseline: {summary['baseline_formatted']}")
        print(f"  latent:   {summary['latent_formatted']}  (saving {summary['latent_saving_ratio']:.1%})")
        print(f"  quant:    {summary['quant_formatted']}  (saving {summary['quant_saving_ratio']:.1%})")

        print(f"  {'layer':>6} {'r_k':>4} {'r_v':>4} {'baseline':>12} {'latent':>12} {'quant':>12}")
        print(f"  {'-'*52}")
        for idx in sorted(per_layer.keys()):
            d = per_layer[idx]
            print(f"  {idx:>6d} {d['r_k']:>4d} {d['r_v']:>4d} "
                  f"{d['baseline_formatted']:>12} {d['latent_formatted']:>12} {d['quant_formatted']:>12}")

        entry = {**summary, "per_layer": per_layer}
        all_results.append(entry)

    if args.output:
        out_path = Path(args.output)
    else:
        out_path = Path("artifacts/kv_memory_profile.json")
    save_json(all_results, out_path)
    print(f"\n[profile] results saved to {out_path}")


if __name__ == "__main__":
    main()
