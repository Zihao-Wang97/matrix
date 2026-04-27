#!/usr/bin/env python
"""Generation eval: python scripts/04_run_generation_eval.py [config] [--mode MODE]

All modes use the same real generation path (profile_generate_by_mode),
the same prompt / max_new_tokens, and the same measurement:
  torch.cuda.reset_peak_memory_stats() -> run -> torch.cuda.max_memory_allocated()

Returns unified CacheStats: cache_runtime_bytes, cache_compressed_bytes, peak_gpu_bytes.
"""

import argparse
from pathlib import Path

import torch

from hawp_laq.config import load_config
from hawp_laq.runtime.generate import (
    load_baseline_model,
    _resolve_device,
    _print_results,
)
from hawp_laq.runtime.mode_runner import setup_mode, make_reset_fn, profile_generate_by_mode


_MODES = ("baseline", "hawp_only", "quant_only", "pure_quant_only", "hawp_quant", "hawp_quant_all", "hawp_quant_sched")


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description="HAWP-LAQ generation eval")
    parser.add_argument(
        "config",
        nargs="?",
        default=None,
        help="Path to yaml config (default: configs/dev_local.yaml)",
    )
    parser.add_argument(
        "--mode",
        choices=list(_MODES),
        default="baseline",
        help="Generation mode (default: baseline)",
    )
    args = parser.parse_args()

    if args.config is None:
        script_dir = Path(__file__).resolve().parent
        args.config = script_dir.parent / "configs" / "dev_local.yaml"

    cfg = load_config(args.config)
    device = _resolve_device(cfg.train.device)

    print("=" * 60)
    print(f"[mode] {args.mode}")
    print("=" * 60)

    model, tokenizer, _ = load_baseline_model(cfg)
    model, coordinator, kv_manager = setup_mode(model, cfg, device, args.mode)
    model.eval()
    reset_fn = make_reset_fn(model, coordinator, kv_manager)

    prompts = cfg.generation.prompts
    print(f"[{args.mode}] running {len(prompts)} prompt(s) ...")

    outputs, stats, _ = profile_generate_by_mode(
        model, tokenizer, prompts, cfg, args.mode,
        coordinator=coordinator, kv_manager=kv_manager, reset_fn=reset_fn,
    )

    _print_results(prompts, outputs)

    print()
    print("=" * 60)
    print(stats.format_summary())
    print("=" * 60)


if __name__ == "__main__":
    main()
