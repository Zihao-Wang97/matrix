#!/usr/bin/env python
"""KV memory profiling: peak GPU memory for each mode.

All modes use the same real generation path (profile_generate_by_mode),
the same prompt / max_new_tokens, and the same measurement:
  torch.cuda.reset_peak_memory_stats() -> run -> torch.cuda.max_memory_allocated()

Returns unified CacheStats with derived metrics:
  kv_compression_ratio, model_bytes_per_token, recent_ratio, archive_ratio.

Usage:
  python scripts/06_profile_kv_memory.py configs/dev_local.yaml --mode hawp_quant
  python scripts/06_profile_kv_memory.py configs/run_server.yaml --mode hawp_quant_sched
"""

import argparse
from pathlib import Path

import torch
from transformers import AutoConfig

from hawp_laq.config import load_config
from hawp_laq.runtime.generate import (
    load_baseline_model,
    _resolve_device,
)
from hawp_laq.runtime.mode_runner import setup_mode, make_reset_fn, profile_generate_by_mode
from hawp_laq.utils.memory import format_nbytes
from hawp_laq.utils.io import save_json


def _build_prompt_for_profile(tokenizer, target_seq_len: int) -> tuple[str, int]:
    """Build a text prompt and report the token length actually profiled.

    The profiler accepts text prompts, so we decode token ids and then measure
    the length after the same re-tokenization path used by profile_generate_by_mode.
    """
    seed_text = "The " * target_seq_len
    enc = tokenizer(seed_text, return_tensors="pt")
    prompt_ids = enc["input_ids"][0][:target_seq_len]
    prompt = tokenizer.decode(
        prompt_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    actual_len = tokenizer(prompt, return_tensors="pt").input_ids.shape[1]
    return prompt, actual_len


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description="HAWP-LAQ KV memory profiling")
    parser.add_argument("config", nargs="?", default=None)
    parser.add_argument("--mode",
                        choices=["baseline", "hawp_only", "quant_only", "hawp_quant", "hawp_quant_all", "hawp_quant_sched", "pure_quant_only"],
                        default="baseline")
    parser.add_argument("--seq-lens", nargs="+", type=int,
                        default=[512, 1024, 2048, 4096])
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if args.config is None:
        args.config = Path(__file__).resolve().parent.parent / "configs" / "run_server.yaml"

    cfg = load_config(args.config)
    device = _resolve_device(cfg.train.device)

    model_cfg = AutoConfig.from_pretrained(cfg.model.model_id, local_files_only=Path(cfg.model.model_id).expanduser().is_dir())
    n_layers = getattr(model_cfg, "num_hidden_layers", 12)
    n_heads = getattr(model_cfg, "num_attention_heads", 12)
    n_kv_heads = getattr(model_cfg, "num_key_value_heads", n_heads)
    head_dim = getattr(model_cfg, "hidden_size", 768) // n_heads

    print("=" * 60)
    print(f"[profile] mode={args.mode}  model={cfg.model.model_id}")
    print(f"[profile] n_layers={n_layers}  n_kv_heads={n_kv_heads}  head_dim={head_dim}")
    print(f"[profile] metric = peak_gpu_bytes + cache_runtime_bytes (real measurement)")
    print("=" * 60)

    model, tokenizer, _ = load_baseline_model(cfg)
    model, coordinator, kv_manager = setup_mode(model, cfg, device, args.mode)
    model.eval()
    reset_fn = make_reset_fn(model, coordinator, kv_manager)

    all_results = []

    for requested_seq_len in args.seq_lens:
        prompt, actual_seq_len = _build_prompt_for_profile(tokenizer, requested_seq_len)

        _, stats, _ = profile_generate_by_mode(
            model, tokenizer, [prompt], cfg, args.mode,
            coordinator=coordinator, kv_manager=kv_manager, reset_fn=reset_fn,
        )

        entry = {
            "requested_seq_len": requested_seq_len,
            "seq_len": actual_seq_len,
            "mode": args.mode,
            "cache_tokens_total": stats.cache_tokens_total,
            "cache_runtime_bytes": stats.cache_runtime_bytes,
            "cache_runtime_formatted": format_nbytes(stats.cache_runtime_bytes),
            "cache_compressed_bytes": stats.cache_compressed_bytes,
            "cache_compressed_formatted": format_nbytes(stats.cache_compressed_bytes),
            "archive_quant_bytes": stats.cache_compressed_bytes,
            "archive_quant_formatted": format_nbytes(stats.cache_compressed_bytes),
            "baseline_kv_bytes": stats.baseline_kv_bytes,
            "baseline_kv_formatted": format_nbytes(stats.baseline_kv_bytes),
            "kv_compression_ratio": round(stats.kv_compression_ratio, 2),
            "bytes_per_token": round(stats.bytes_per_token, 1),
            "model_bytes_per_token": round(stats.bytes_per_token, 1),
            "recent_tokens": stats.recent_tokens,
            "recent_ratio": round(stats.recent_ratio, 3),
            "archive_tokens": stats.archive_tokens,
            "archive_ratio": round(stats.archive_ratio, 3),
            "peak_gpu_bytes": stats.peak_gpu_bytes,
            "peak_gpu_formatted": format_nbytes(stats.peak_gpu_bytes),
            "memory_overhead_ratio": round(stats.memory_overhead_ratio, 2),
            "impl": stats.impl,
        }
        all_results.append(entry)

        len_note = f"actual_seq_len={actual_seq_len}"
        if actual_seq_len != requested_seq_len:
            len_note += f" (requested={requested_seq_len})"
        print(f"\n--- {len_note} ---")
        print(f"  peak_gpu: {entry['peak_gpu_formatted']}  "
              f"cache_runtime: {entry['cache_runtime_formatted']}  "
              f"baseline_kv: {entry['baseline_kv_formatted']}")
        if stats.kv_compression_ratio > 0:
            print(f"  kv_compression_ratio={stats.kv_compression_ratio:.1f}x  "
                  f"model_bytes_per_token={stats.bytes_per_token:.1f} B  "
                  f"memory_overhead={stats.memory_overhead_ratio:.1f}x")
        if stats.recent_tokens > 0 or stats.archive_tokens > 0:
            print(f"  recent={stats.recent_tokens} ({stats.recent_ratio:.1%})  "
                  f"archive={stats.archive_tokens} ({stats.archive_ratio:.1%})  "
                  f"archive_quant={entry['archive_quant_formatted']}  impl={stats.impl}")

    print(f"\n{'='*100}")
    print(f"{'seq_len':>8} {'mode':>18} {'KV_ratio':>8} {'model_B/tok':>12} {'Overhead':>9} {'cache_rt':>12} {'peak_gpu':>12} {'recent%':>8} {'archive%':>8}")
    print("-" * 100)
    for r in all_results:
        cr = f"{r['kv_compression_ratio']:.1f}x" if r['kv_compression_ratio'] > 0 else "N/A"
        bpt = f"{r['model_bytes_per_token']:.1f}" if r['model_bytes_per_token'] > 0 else "N/A"
        oh = f"{r['memory_overhead_ratio']:.1f}x" if r['memory_overhead_ratio'] > 0 else "N/A"
        rr = f"{r['recent_ratio']:.0%}" if r['recent_ratio'] > 0 else "-"
        ar = f"{r['archive_ratio']:.0%}" if r['archive_ratio'] > 0 else "-"
        print(f"{r['seq_len']:>8d} {r['mode']:>18} {cr:>8} {bpt:>12} {oh:>9} "
              f"{r['cache_runtime_formatted']:>12} {r['peak_gpu_formatted']:>12} {rr:>8} {ar:>8}")

    if args.output:
        out_path = Path(args.output)
    else:
        out_path = Path("artifacts/kv_memory_profile.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(all_results, out_path)
    print(f"\n[profile] saved to {out_path}")


if __name__ == "__main__":
    main()
