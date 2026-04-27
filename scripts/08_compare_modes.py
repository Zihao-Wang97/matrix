#!/usr/bin/env python
"""Compare all modes side-by-side: python scripts/08_compare_modes.py [config]

All modes use the same real generation path (profile_generate_by_mode),
the same prompt / max_new_tokens, and the same measurement:
  torch.cuda.reset_peak_memory_stats() -> run -> torch.cuda.max_memory_allocated()

Returns unified CacheStats: cache_runtime_bytes, cache_compressed_bytes, peak_gpu_bytes.
Derived metrics: kv_compression_ratio, bytes_per_token, recent_ratio, archive_ratio,
  memory_overhead_ratio.
Quality metrics: token_consistency (from raw ids, no tokenizer round-trip),
  ΔPPL (stepwise, through quant cache), compression-quality efficiency ratio.
"""

import argparse
import copy
import math
import sys
import time
from pathlib import Path

import torch
from transformers import AutoConfig

from hawp_laq.config import load_config
from hawp_laq.runtime.generate import (
    _fmt_bytes,
    _resolve_device,
    load_baseline_model,
    stepwise_greedy_generate,
)
from hawp_laq.runtime.mode_runner import setup_mode, make_reset_fn, profile_generate_by_mode
from hawp_laq.runtime.cache_stats import CacheStats
from hawp_laq.utils.memory import format_nbytes

_MODES = ["baseline", "hawp_only", "quant_only", "pure_quant_only", "hawp_quant", "hawp_quant_all", "hawp_quant_sched"]


def _run_correctness(model, tokenizer, prompts, max_new_tokens, coordinator, kv_manager, mode, cfg=None):
    """Unified stepwise greedy for all modes — for correctness comparison.

    Always returns (texts, gen_ids_list) so consistency can be computed
    from raw token ids without tokenizer round-trip.
    """
    if mode == "pure_quant_only":
        if cfg is None:
            from hawp_laq.config import HAWPLAQConfig
            cfg = HAWPLAQConfig()
        cfg.generation.max_new_tokens = max_new_tokens
        from hawp_laq.runtime.generate import generate_pure_quant_only
        return generate_pure_quant_only(model, tokenizer, prompts, cfg, kv_manager, return_ids=True)

    if mode in ("baseline", "hawp_only"):
        reset_fn = None
        full_recompute = False
    else:
        reset_fn = make_reset_fn(model, coordinator, kv_manager)
        full_recompute = False

    return stepwise_greedy_generate(
        model, tokenizer, prompts, max_new_tokens,
        coordinator=coordinator if coordinator is not None else None,
        reset_cache_fn=reset_fn,
        full_recompute=full_recompute,
        use_external_past=mode in ("baseline", "hawp_only"),
        return_ids=True,
    )


def _safe_print(text):
    enc = sys.stdout.encoding or "utf-8"
    return text.encode(enc, errors="replace").decode(enc, errors="replace")


def _compute_token_consistency_from_ids(baseline_ids: torch.Tensor, mode_ids: torch.Tensor) -> float:
    n_cmp = min(len(baseline_ids), len(mode_ids))
    if n_cmp == 0:
        return 1.0
    matched = sum(1 for a, b in zip(baseline_ids[:n_cmp].tolist(), mode_ids[:n_cmp].tolist()) if a == b)
    total = max(len(baseline_ids), len(mode_ids))
    return matched / total


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description="Compare all HAWP-LAQ modes")
    parser.add_argument("config", nargs="?", default=None)
    parser.add_argument("--modes", nargs="+", default=_MODES, choices=_MODES)
    parser.add_argument("--skip-speed", action="store_true",
                        help="Skip speed comparison (only run correctness)")
    parser.add_argument("--skip-ppl", action="store_true",
                        help="Skip perplexity evaluation")
    args = parser.parse_args()

    if args.config is None:
        args.config = Path(__file__).resolve().parent.parent / "configs" / "dev_local.yaml"

    cfg = load_config(args.config)
    device = _resolve_device(cfg.train.device)

    correctness_results = {}
    profile_results = {}
    ppl_results = {}

    print(f"\n{'='*60}")
    print("Loading base model (once) ...")
    base_model, tokenizer, dev = load_baseline_model(cfg)
    if not cfg.model.load_in_4bit:
        base_model = base_model.cpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    for mode in args.modes:
        print(f"\n{'='*60}")
        print(f"[{mode}] setting up mode ...")

        try:
            if cfg.model.load_in_4bit:
                model, _, _ = load_baseline_model(cfg)
            else:
                model = copy.deepcopy(base_model)
                model = model.to(dev)

            model, coordinator, kv_manager = setup_mode(model, cfg, dev, mode)
        except (RuntimeError, ValueError, FileNotFoundError) as exc:
            print(f"[{mode}] SKIPPED — {exc}")
            if not cfg.model.load_in_4bit:
                del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
        model.eval()
        reset_fn = make_reset_fn(model, coordinator, kv_manager)
        prompts = cfg.generation.prompts
        max_new_tokens = cfg.generation.max_new_tokens

        # --- Perplexity (stepwise, through quant cache) ---
        if not args.skip_ppl:
            from hawp_laq.eval.perplexity import compute_stepwise_ppl
            seq_len = cfg.calib.seq_len
            nsamples = cfg.calib.nsamples if cfg.mode == "local" else None
            print(f"[{mode}] perplexity stepwise (seq_len={seq_len}, through cache) ...")
            ppl_result = compute_stepwise_ppl(
                model, tokenizer,
                coordinator=coordinator, kv_manager=kv_manager, reset_fn=reset_fn,
                seq_len=seq_len, nsamples=nsamples, device=dev,
            )
            ppl_results[mode] = ppl_result["perplexity"]
            print(f"[{mode}] PPL={ppl_result['perplexity']:.4f}  (stepwise, quant cache exercised)")

        # --- Correctness run (unified stepwise greedy) ---
        print(f"[{mode}] correctness: stepwise greedy ({max_new_tokens} tokens) ...")
        corr_start = time.perf_counter()
        corr_texts, corr_gen_ids = _run_correctness(model, tokenizer, prompts, max_new_tokens, coordinator, kv_manager, mode, cfg)
        corr_time = time.perf_counter() - corr_start
        total_new = max_new_tokens * len(prompts)
        correctness_results[mode] = {
            "texts": corr_texts,
            "gen_ids": corr_gen_ids,
            "time": corr_time,
            "tok_per_s": total_new / corr_time if corr_time > 0 else 0,
        }

        # --- Speed + cache + peak GPU run (profile_generate_by_mode) ---
        if not args.skip_speed:
            print(f"[{mode}] speed + cache + peak GPU: profile_generate_by_mode ...")
            speed_start = time.perf_counter()
            speed_texts, stats, gen_ids_list = profile_generate_by_mode(
                model, tokenizer, prompts, cfg, mode,
                coordinator=coordinator, kv_manager=kv_manager, reset_fn=reset_fn,
            )
            speed_time = time.perf_counter() - speed_start

            profile_results[mode] = {
                "texts": speed_texts,
                "time": speed_time,
                "tok_per_s": total_new / speed_time if speed_time > 0 else 0,
                "stats": stats,
                "gen_ids": gen_ids_list,
            }

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Determine reference mode for consistency
    if "baseline" in correctness_results:
        ref_mode = "baseline"
    elif "hawp_only" in correctness_results:
        ref_mode = "hawp_only"
    elif args.modes:
        ref_mode = args.modes[0]
    else:
        ref_mode = None

    if ref_mode is not None and ref_mode != "baseline":
        print(f"\n[warning] 'baseline' not in --modes; using '{ref_mode}' as reference for consistency/ΔPPL")

    # Pre-compute token consistency per mode vs reference (always from raw ids)
    consistency = {}
    if ref_mode and len(args.modes) > 1 and ref_mode in correctness_results:
        ref_ids_list = correctness_results[ref_mode].get("gen_ids")
        if ref_ids_list is None:
            print("\n[warning] no gen_ids available for consistency comparison; skipping")
        else:
            for mode in args.modes:
                if mode == ref_mode:
                    consistency[mode] = 1.0
                    continue
                if mode not in correctness_results:
                    continue
                mode_ids_list = correctness_results[mode].get("gen_ids")
                if mode_ids_list is None:
                    continue
                ratios = []
                for ref_ids, mode_ids in zip(ref_ids_list, mode_ids_list):
                    ratios.append(_compute_token_consistency_from_ids(ref_ids, mode_ids))
                consistency[mode] = sum(ratios) / len(ratios) if ratios else 0.0

    # ================================================================
    # Print: Correctness comparison
    # ================================================================
    print("\n" + "=" * 80)
    print("CORRECTNESS COMPARISON  (unified stepwise greedy, argmax)")
    print("  -> All modes use the same argmax token selection.")
    print("  -> All modes use incremental KV cache (past_key_values) for decoding.")
    print("  -> quant_*: internal KV cache (only new token per step)")
    print("  -> pure_quant_only: original attention + quantized KV cache (no HAWP)")
    print("  -> Token differences are due only to quantisation / low-rank.")
    print("=" * 80)

    for pi, prompt in enumerate(cfg.generation.prompts):
        print(f"\n--- Prompt {pi}: \"{prompt}\" ---")
        for mode in args.modes:
            text = correctness_results[mode]["texts"][pi]
            print(f"  [{mode:>20}] {_safe_print(text)}")

    # Token consistency + ΔPPL
    if consistency:
        print(f"\n{'='*80}")
        ref_label = ref_mode
        print(f"QUALITY COMPARISON  (vs {ref_label})")
        print(f"{'='*80}")
        header = f"{'Mode':<22} {'Consistency':>12}"
        if ppl_results:
            header += f" {'PPL':>10} {'ΔPPL':>10}"
        print(f"\n{header}")
        print("-" * len(header))
        ref_ppl = ppl_results.get(ref_mode, None) if ppl_results else None
        for mode in args.modes:
            line = f"{mode:<22} {consistency.get(mode, 0.0):>11.1%}"
            if ppl_results and mode in ppl_results:
                ppl = ppl_results[mode]
                dppl = ppl - ref_ppl if ref_ppl is not None and not math.isnan(ppl) else float("nan")
                line += f" {ppl:>10.2f} {dppl:>+10.2f}"
            print(line)

    # ================================================================
    # Print: Speed + Cache + Peak GPU comparison
    # ================================================================
    if profile_results:
        print("\n" + "=" * 80)
        print("SPEED + CACHE + PEAK GPU COMPARISON  (profile_generate_by_mode)")
        print("  -> All modes use the same stepwise generation + measurement flow.")
        print("  -> cache_runtime_bytes = real runtime cache footprint (not formula).")
        print("  -> peak_gpu_bytes = torch.cuda.max_memory_allocated().")
        print("=" * 80)

        print(f"\n{'Mode':<22} {'Speed':>12} {'KV压缩率':>10} {'model_B/tok':>12} {'Overhead':>9} {'Cache RT':>12} {'Peak GPU':>12}")
        print("-" * 90)
        for mode in args.modes:
            if mode not in profile_results:
                continue
            r = profile_results[mode]
            s = r["stats"]
            cr = f"{s.kv_compression_ratio:.1f}x" if s.kv_compression_ratio > 0 else "N/A"
            bpt = f"{s.bytes_per_token:.1f}" if s.bytes_per_token > 0 else "N/A"
            oh = f"{s.memory_overhead_ratio:.1f}x" if s.memory_overhead_ratio > 0 else "N/A"
            print(f"{mode:<22} {r['tok_per_s']:>10.1f} t/s "
                  f"{cr:>10} {bpt:>12} {oh:>9} "
                  f"{format_nbytes(s.cache_runtime_bytes):>12} "
                  f"{format_nbytes(s.peak_gpu_bytes):>12}")

        # Detailed cache stats
        print(f"\n{'='*80}")
        print("CACHE DETAIL")
        print(f"{'='*80}")
        for mode in args.modes:
            if mode not in profile_results:
                continue
            s = profile_results[mode]["stats"]
            print(f"\n  [{mode}]")
            print(f"    tokens_total={s.cache_tokens_total}  recent={s.recent_tokens} ({s.recent_ratio:.1%})  "
                  f"archive={s.archive_tokens} ({s.archive_ratio:.1%})")
            print(f"    runtime={format_nbytes(s.cache_runtime_bytes)}  "
                  f"archive_quant={format_nbytes(s.cache_compressed_bytes)}  "
                  f"baseline_kv={format_nbytes(s.baseline_kv_bytes)}  "
                  f"compression={s.kv_compression_ratio:.1f}x  "
                  f"overhead={s.memory_overhead_ratio:.1f}x  "
                  f"impl={s.impl}")

    # ================================================================
    # Print: Compression-Quality Efficiency Ratio
    # ================================================================
    if consistency and profile_results and ref_mode:
        print(f"\n{'='*80}")
        print(f"COMPRESSION-QUALITY EFFICIENCY  (compression_ratio-1)/(1-consistency)")
        print("  -> Higher = more compression gained per unit of quality lost.")
        print("  -> 'inf' means no quality loss at all (consistency=100%).")
        print(f"{'='*80}")
        print(f"\n{'Mode':<22} {'压缩率':>10} {'一致率':>10} {'效率比':>12}")
        print("-" * 58)
        for mode in args.modes:
            if mode == ref_mode:
                print(f"{mode:<22} {'1.0x':>10} {'100.0%':>10} {'—':>12}")
                continue
            if mode not in profile_results or mode not in consistency:
                continue
            cr = profile_results[mode]["stats"].kv_compression_ratio
            c = consistency[mode]
            if c >= 1.0:
                eff = float("inf")
                eff_str = "inf"
            else:
                eff = (cr - 1.0) / (1.0 - c)
                eff_str = f"{eff:.1f}"
            print(f"{mode:<22} {cr:>9.1f}x {c:>9.1%} {eff_str:>12}")


if __name__ == "__main__":
    main()
