#!/usr/bin/env python
"""Generation eval: python scripts/04_run_generation_eval.py [config] [--mode MODE]"""

import argparse
from pathlib import Path

import torch
from transformers import AutoConfig

from hawp_laq.config import load_config
from hawp_laq.modeling.attention_hawp import HAWPAttention
from hawp_laq.runtime.generate import (
    _fmt_bytes,
    _print_results,
    _setup_hawp_quant_on_model,
    _setup_hawp_quant_all_on_model,
    _setup_quant_only_on_model,
    _convert_and_load_projectors,
    _resolve_hawp_ranks,
    generate_hawp_quant,
    generate_text,
    load_baseline_model,
    print_device_info,
)
from hawp_laq.modeling.modeling_llama_hawp import convert_llama_to_hawp


_MODES = ("baseline", "hawp_only", "quant_only", "hawp_quant", "hawp_quant_all", "hawp_quant_sched")


def _fmt_bytes_dual(n: int) -> str:
    return f"{n} ({_fmt_bytes(n)})"


def estimate_baseline_kv_bytes(model_config, seq_len: int, bytes_per_element: int = 2) -> int:
    n_layers = getattr(model_config, "num_hidden_layers", 12)
    n_heads = getattr(model_config, "num_attention_heads", 12)
    n_kv_heads = getattr(model_config, "num_key_value_heads", n_heads)
    head_dim = getattr(model_config, "hidden_size", 768) // n_heads
    return n_layers * n_kv_heads * seq_len * head_dim * 2 * bytes_per_element


def estimate_hawp_compressed_bytes_from_ranks(model, seq_len: int, bytes_per_element: int = 2) -> int:
    total = 0
    for mod in model.modules():
        if isinstance(mod, HAWPAttention):
            r_k = mod.r_k if hasattr(mod, 'r_k') else mod.head_dim
            r_v = mod.r_v if hasattr(mod, 'r_v') else mod.head_dim
            total += mod.num_key_value_heads * seq_len * (r_k + r_v) * bytes_per_element
    return total


def try_get_runtime_quant_summary(model) -> dict | None:
    summaries = []
    for mod in model.modules():
        if isinstance(mod, HAWPAttention) and mod.use_cache_manager:
            try:
                summaries.append(mod.quant_cache_summary())
            except Exception:
                pass
    if not summaries:
        return None
    total_runtime = sum(s.get("total_runtime_bytes", 0) for s in summaries)
    total_compressed = sum(s.get("compressed_storage_bytes", 0) for s in summaries)
    total_recent = sum(s.get("recent_tokens", 0) for s in summaries)
    total_archive = sum(s.get("archive_tokens", 0) for s in summaries)
    return {
        "total_runtime_bytes": total_runtime,
        "compressed_storage_bytes": total_compressed,
        "recent_tokens": total_recent,
        "archive_tokens": total_archive,
        "n_layers": len(summaries),
    }


def format_profile_result(mode: str, baseline_bytes: int, runtime_bytes: int | None,
                          compressed_bytes: int | None) -> list[str]:
    lines = []
    lines.append(f"[mode] {mode}")
    lines.append(f"[kv] baseline_bytes   = {_fmt_bytes_dual(baseline_bytes)}")

    if runtime_bytes is not None:
        lines.append(f"[kv] runtime_bytes   = {_fmt_bytes_dual(runtime_bytes)}")
    else:
        lines.append(f"[kv] runtime_bytes   = N/A (no runtime cache summary)")

    if compressed_bytes is not None:
        lines.append(f"[kv] compressed_bytes = {_fmt_bytes_dual(compressed_bytes)}")
    else:
        lines.append(f"[kv] compressed_bytes = N/A")

    rt_pct = None
    cmp_pct = None
    if baseline_bytes > 0:
        if runtime_bytes is not None:
            rt_pct = (1.0 - runtime_bytes / baseline_bytes) * 100
        if compressed_bytes is not None:
            cmp_pct = (1.0 - compressed_bytes / baseline_bytes) * 100

    parts = []
    if rt_pct is not None:
        parts.append(f"runtime {rt_pct:.1f}%")
    else:
        parts.append("runtime N/A")
    if cmp_pct is not None:
        parts.append(f"compressed {cmp_pct:.1f}%")
    else:
        parts.append("compressed N/A")
    lines.append(f"[kv] saving_vs_baseline = {'  '.join(parts)}")

    return lines


def print_profile_block(mode: str, model, model_config, cfg) -> None:
    seq_len = getattr(cfg.generation, "max_new_tokens", 512)
    baseline_bytes = estimate_baseline_kv_bytes(model_config, seq_len)

    runtime_bytes = None
    compressed_bytes = None

    if mode == "baseline":
        runtime_bytes = baseline_bytes
        compressed_bytes = baseline_bytes

    elif mode == "hawp_only":
        compressed_bytes = estimate_hawp_compressed_bytes_from_ranks(model, seq_len)
        quant_summary = try_get_runtime_quant_summary(model)
        if quant_summary is not None:
            runtime_bytes = quant_summary["total_runtime_bytes"]

    else:
        quant_summary = try_get_runtime_quant_summary(model)
        if quant_summary is not None:
            runtime_bytes = quant_summary["total_runtime_bytes"]
            compressed_bytes = quant_summary["compressed_storage_bytes"]
        else:
            compressed_bytes = estimate_hawp_compressed_bytes_from_ranks(model, seq_len)

    print()
    print("=" * 60)
    for line in format_profile_result(mode, baseline_bytes, runtime_bytes, compressed_bytes):
        print(line)

    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        alloc = torch.cuda.memory_allocated(idx)
        reserved = torch.cuda.memory_reserved(idx)
        print(f"[cuda] allocated = {_fmt_bytes_dual(alloc)}")
        print(f"[cuda] reserved  = {_fmt_bytes_dual(reserved)}")
    else:
        print("[cuda] unavailable")

    print("=" * 60)


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
    model_config = AutoConfig.from_pretrained(cfg.model.model_id)

    print("=" * 60)
    print(f"[mode] {args.mode}")
    print_device_info(cfg.train.device)
    print("=" * 60)

    model, tokenizer, device = load_baseline_model(cfg)
    coordinator = None

    if args.mode == "baseline":
        prompts = cfg.generation.prompts
        print(f"[baseline] running {len(prompts)} prompt(s) ...")
        outputs = generate_text(model, tokenizer, prompts, cfg)

    elif args.mode == "hawp_only":
        model, r_k, r_v = _convert_and_load_projectors(model, cfg, device, "hawp_only")
        print(f"[hawp_only] r_k={r_k}  r_v={r_v}")
        prompts = cfg.generation.prompts
        print(f"[hawp_only] running {len(prompts)} prompt(s) ...")
        outputs = generate_text(model, tokenizer, prompts, cfg)

    elif args.mode == "quant_only":
        model, head_dim = _setup_quant_only_on_model(model, cfg, device)
        prompts = cfg.generation.prompts
        print(f"[quant_only] running {len(prompts)} prompt(s) ...")
        outputs = generate_hawp_quant(model, tokenizer, prompts, cfg)

    elif args.mode == "hawp_quant":
        model = _setup_hawp_quant_on_model(model, cfg, device)
        r_k, r_v = _resolve_hawp_ranks(cfg, model, "hawp_quant")[:2]
        print(f"[hawp_quant] r_k={r_k}  r_v={r_v}  recent_window={cfg.sched.recent_window}")
        prompts = cfg.generation.prompts
        print(f"[hawp_quant] running {len(prompts)} prompt(s) ...")
        outputs = generate_hawp_quant(model, tokenizer, prompts, cfg)

    elif args.mode == "hawp_quant_all":
        model = _setup_hawp_quant_all_on_model(model, cfg, device)
        r_k, r_v = _resolve_hawp_ranks(cfg, model, "hawp_quant_all")[:2]
        print(f"[hawp_quant_all] r_k={r_k}  r_v={r_v}  recent_window=0 (all tokens quantized)")
        prompts = cfg.generation.prompts
        print(f"[hawp_quant_all] running {len(prompts)} prompt(s) ...")
        outputs = generate_hawp_quant(model, tokenizer, prompts, cfg)

    elif args.mode == "hawp_quant_sched":
        from hawp_laq.runtime.scheduler import TokenBudgetScheduler
        from hawp_laq.runtime.cache_manager import ModelCacheCoordinator

        model = _setup_hawp_quant_on_model(model, cfg, device)
        r_k, r_v = _resolve_hawp_ranks(cfg, model, "hawp_quant_sched")[:2]
        total_budget = cfg.sched.total_budget
        recent_window = cfg.sched.recent_window
        drop_strategy = getattr(cfg.sched, "drop_strategy", "position")

        scheduler = TokenBudgetScheduler(
            total_budget=total_budget,
            recent_window=recent_window,
            high_ratio=cfg.sched.high_ratio,
            low_ratio=cfg.sched.low_ratio,
            drop_strategy=drop_strategy,
        )
        coordinator = ModelCacheCoordinator.from_model(
            model, scheduler, drop_strategy=drop_strategy,
        )

        print(f"[hawp_quant_sched] r_k={r_k}  r_v={r_v}  "
              f"budget={total_budget}  recent_window={recent_window}  "
              f"drop_strategy={drop_strategy}")
        prompts = cfg.generation.prompts
        print(f"[hawp_quant_sched] running {len(prompts)} prompt(s) ...")
        outputs = generate_hawp_quant(model, tokenizer, prompts, cfg, coordinator=coordinator)

    _print_results(prompts, outputs)
    print_profile_block(args.mode, model, model_config, cfg)


if __name__ == "__main__":
    main()
