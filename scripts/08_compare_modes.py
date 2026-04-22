#!/usr/bin/env python
"""Compare all modes side-by-side: python scripts/08_compare_modes.py [config]

Separates **correctness** comparison (unified stepwise greedy) from **speed**
comparison (production-path generation).  This ensures that token-level
differences are due only to quantisation, not to divergent outer generation
semantics (e.g. HF generate vs manual decode loop).
"""

import argparse
import sys
import time
from pathlib import Path

import torch
from transformers import AutoConfig

from hawp_laq.config import load_config
from hawp_laq.modeling.attention_hawp import HAWPAttention
from hawp_laq.modeling.modeling_llama_hawp import convert_llama_to_hawp
from hawp_laq.eval.metrics import collect_kv_metrics
from hawp_laq.runtime.generate import (
    _fmt_bytes,
    _resolve_device,
    load_baseline_model,
    generate_text,
    generate_hawp_quant,
    stepwise_greedy_generate,
    _setup_hawp_quant_on_model,
    _setup_hawp_quant_all_on_model,
    _setup_quant_only_on_model,
)

_MODES = ["baseline", "hawp_only", "quant_only", "hawp_quant", "hawp_quant_all", "hawp_quant_sched"]


def _setup(model, cfg, device, mode):
    if mode == "baseline":
        return model, None
    if mode == "hawp_only":
        from hawp_laq.runtime.projector_bank import load_projectors
        model = convert_llama_to_hawp(model, r_k=cfg.projector.r_k, r_v=cfg.projector.r_v)
        model = model.to(device).eval()
        if Path(cfg.projector.output_dir).exists():
            load_projectors(model, cfg.projector.output_dir)
        return model, None
    if mode == "quant_only":
        model, _ = _setup_quant_only_on_model(model, cfg, device)
        return model, None
    if mode == "hawp_quant":
        model = _setup_hawp_quant_on_model(model, cfg, device)
        return model, None
    if mode == "hawp_quant_all":
        model = _setup_hawp_quant_all_on_model(model, cfg, device)
        return model, None
    if mode == "hawp_quant_sched":
        from hawp_laq.runtime.scheduler import TokenBudgetScheduler
        from hawp_laq.runtime.cache_manager import ModelCacheCoordinator
        model = _setup_hawp_quant_on_model(model, cfg, device)
        sched = TokenBudgetScheduler(
            total_budget=cfg.sched.total_budget,
            recent_window=cfg.sched.recent_window,
            high_ratio=cfg.sched.high_ratio,
            low_ratio=cfg.sched.low_ratio,
            drop_strategy=getattr(cfg.sched, "drop_strategy", "position"),
        )
        coordinator = ModelCacheCoordinator.from_model(
            model, sched, drop_strategy=getattr(cfg.sched, "drop_strategy", "position"),
        )
        return model, coordinator
    raise ValueError(f"Unknown mode: {mode}")


def _make_reset_fn(model, coordinator):
    def _reset():
        for mod in model.modules():
            if isinstance(mod, HAWPAttention) and mod.use_cache_manager:
                mod.reset_quant_cache()
        if coordinator is not None:
            coordinator.reset()
    return _reset


def _run_correctness(model, tokenizer, prompts, max_new_tokens, coordinator, mode):
    """Unified stepwise greedy for all modes — for correctness comparison."""
    if mode in ("baseline", "hawp_only"):
        reset_fn = None
    else:
        reset_fn = _make_reset_fn(model, coordinator)

    return stepwise_greedy_generate(
        model, tokenizer, prompts, max_new_tokens,
        coordinator=coordinator if coordinator is not None else None,
        reset_cache_fn=reset_fn,
    )


def _run_speed(model, tokenizer, prompts, cfg, coordinator, mode):
    """Production-path generation — for speed comparison."""
    if mode in ("baseline", "hawp_only"):
        return generate_text(model, tokenizer, prompts, cfg)
    else:
        return generate_hawp_quant(model, tokenizer, prompts, cfg, coordinator=coordinator)


def _safe_print(text):
    enc = sys.stdout.encoding or "utf-8"
    return text.encode(enc, errors="replace").decode(enc, errors="replace")


def _compute_kv_summary(model, mode, cfg, tokenizer, prompts, model_config, coordinator=None):
    has_cache = any(isinstance(m, HAWPAttention) and m.use_cache_manager for m in model.modules())
    n_layers = getattr(model_config, "num_hidden_layers", 12)
    n_kv_heads = getattr(model_config, "num_key_value_heads", getattr(model_config, "num_attention_heads", 12))
    head_dim = getattr(model_config, "hidden_size", 768) // getattr(model_config, "num_attention_heads", 12)
    max_new_tokens = cfg.generation.max_new_tokens

    if has_cache:
        kv_info = collect_kv_metrics(model)
        total_tokens = kv_info["total_tokens"]
        baseline_bytes = total_tokens * n_layers * n_kv_heads * head_dim * 2 * 2
        runtime_bytes = kv_info.get("total_runtime_bytes", kv_info.get("total_bytes", baseline_bytes))
        compressed_bytes = kv_info.get("compressed_storage_bytes", kv_info.get("total_bytes", baseline_bytes))
    elif mode == "hawp_only":
        seq_len = max(len(tokenizer(p)["input_ids"]) for p in prompts) + max_new_tokens
        baseline_bytes = seq_len * n_layers * n_kv_heads * head_dim * 2 * 2
        r_k = cfg.projector.r_k
        r_v = cfg.projector.r_v
        runtime_bytes = compressed_bytes = seq_len * n_layers * n_kv_heads * (r_k + r_v) * 2
    else:
        seq_len = max(len(tokenizer(p)["input_ids"]) for p in prompts) + max_new_tokens
        baseline_bytes = seq_len * n_layers * n_kv_heads * head_dim * 2 * 2
        runtime_bytes = baseline_bytes
        compressed_bytes = baseline_bytes

    saving = 1.0 - compressed_bytes / baseline_bytes if baseline_bytes > 0 else 0.0

    sched_info = None
    if mode == "hawp_quant_sched" and coordinator is not None:
        try:
            d = coordinator.scheduler.rebalance()
            sched_info = {"HIGH": d.n_high, "LOW": d.n_low, "DROP": d.n_drop}
        except Exception:
            pass

    return {
        "baseline_bytes": baseline_bytes,
        "runtime_bytes": runtime_bytes,
        "compressed_bytes": compressed_bytes,
        "saving_ratio": saving,
        "sched_info": sched_info,
    }


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description="Compare all HAWP-LAQ modes")
    parser.add_argument("config", nargs="?", default=None)
    parser.add_argument("--modes", nargs="+", default=_MODES, choices=_MODES)
    parser.add_argument("--skip-speed", action="store_true",
                        help="Skip speed comparison (only run correctness)")
    args = parser.parse_args()

    if args.config is None:
        args.config = Path(__file__).resolve().parent.parent / "configs" / "dev_local.yaml"

    cfg = load_config(args.config)
    device = _resolve_device(cfg.train.device)
    model_config = AutoConfig.from_pretrained(cfg.model.model_id)

    correctness_results = {}
    speed_results = {}
    kv_results = {}

    for mode in args.modes:
        print(f"\n{'='*60}")
        print(f"[{mode}] loading model ...")
        model, tokenizer, dev = load_baseline_model(cfg)
        model, coordinator = _setup(model, cfg, dev, mode)
        model.eval()
        prompts = cfg.generation.prompts
        max_new_tokens = cfg.generation.max_new_tokens

        # --- Correctness run (unified stepwise greedy) ---
        print(f"[{mode}] correctness: stepwise greedy ({max_new_tokens} tokens) ...")
        corr_start = time.perf_counter()
        corr_texts = _run_correctness(model, tokenizer, prompts, max_new_tokens, coordinator, mode)
        corr_time = time.perf_counter() - corr_start
        total_new = max_new_tokens * len(prompts)
        correctness_results[mode] = {
            "texts": corr_texts,
            "time": corr_time,
            "tok_per_s": total_new / corr_time if corr_time > 0 else 0,
        }

        # --- Speed run (production path) ---
        if not args.skip_speed:
            if mode in ("baseline", "hawp_only"):
                for mod in model.modules():
                    if isinstance(mod, HAWPAttention) and mod.use_cache_manager:
                        mod.reset_quant_cache()
            elif coordinator is not None:
                coordinator.reset()
            else:
                for mod in model.modules():
                    if isinstance(mod, HAWPAttention) and mod.use_cache_manager:
                        mod.reset_quant_cache()

            print(f"[{mode}] speed: production path ...")
            speed_start = time.perf_counter()
            speed_texts = _run_speed(model, tokenizer, prompts, cfg, coordinator, mode)
            speed_time = time.perf_counter() - speed_start
            speed_results[mode] = {
                "texts": speed_texts,
                "time": speed_time,
                "tok_per_s": total_new / speed_time if speed_time > 0 else 0,
            }

        # --- KV summary ---
        kv_results[mode] = _compute_kv_summary(model, mode, cfg, tokenizer, prompts, model_config, coordinator)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ================================================================
    # Print: Correctness comparison
    # ================================================================
    print("\n" + "=" * 80)
    print("CORRECTNESS COMPARISON  (unified stepwise greedy, argmax)")
    print("  -> All modes use the same outer decode loop.")
    print("  -> Token differences are due only to quantisation / low-rank.")
    print("=" * 80)

    for pi, prompt in enumerate(cfg.generation.prompts):
        print(f"\n--- Prompt {pi}: \"{prompt}\" ---")
        for mode in args.modes:
            text = correctness_results[mode]["texts"][pi]
            print(f"  [{mode:>20}] {_safe_print(text)}")

    # ================================================================
    # Print: Speed comparison
    # ================================================================
    if speed_results:
        print("\n" + "=" * 80)
        print("SPEED COMPARISON  (production-path generation)")
        print("  -> baseline / hawp_only use HF model.generate().")
        print("  -> quant_* modes use manual stepwise decode loop.")
        print("  -> Speed numbers are NOT directly comparable across paths.")
        print("=" * 80)

        print(f"\n{'Mode':<22} {'Time (s)':>10} {'Speed':>12}")
        print("-" * 46)
        for mode in args.modes:
            r = speed_results[mode]
            print(f"{mode:<22} {r['time']:>10.2f} {r['tok_per_s']:>10.1f} t/s")
    else:
        print("\n  (speed comparison skipped)")

    # ================================================================
    # Print: KV memory comparison
    # ================================================================
    print("\n" + "=" * 80)
    print("KV MEMORY COMPARISON")
    print("=" * 80)

    print(f"\n{'Mode':<22} {'Baseline':>12} {'Runtime':>12} {'Compressed':>12} {'Saving':>8}")
    print("-" * 70)
    for mode in args.modes:
        kv = kv_results[mode]
        print(f"{mode:<22} {_fmt_bytes(kv['baseline_bytes']):>12} "
              f"{_fmt_bytes(kv['runtime_bytes']):>12} "
              f"{_fmt_bytes(kv['compressed_bytes']):>12} "
              f"{kv['saving_ratio']:>7.1%}")

    if any(kv_results[m].get("sched_info") for m in args.modes):
        print(f"\n{'Mode':<22} {'HIGH':>6} {'LOW':>6} {'DROP':>6}")
        print("-" * 42)
        for mode in args.modes:
            si = kv_results[mode].get("sched_info")
            if si:
                print(f"{mode:<22} {si['HIGH']:>6} {si['LOW']:>6} {si['DROP']:>6}")


if __name__ == "__main__":
    main()
