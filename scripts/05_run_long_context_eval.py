#!/usr/bin/env python
"""Long context + perplexity + needle eval.

Usage:
  python scripts/05_run_long_context_eval.py configs/dev_local.yaml --mode hawp_quant
  python scripts/05_run_long_context_eval.py configs/run_server.yaml --mode hawp_quant_sched
"""

import argparse
import time
from pathlib import Path

import torch

from hawp_laq.config import load_config
from hawp_laq.runtime.generate import (
    load_baseline_model,
    _resolve_device,
    _fmt_bytes,
)
from hawp_laq.runtime.mode_runner import setup_mode, make_reset_fn, generate_by_mode, profile_generate_by_mode
from hawp_laq.runtime.cache_stats import CacheStats
from hawp_laq.modeling.attention_hawp import HAWPAttention
from hawp_laq.utils.memory import format_nbytes


def _run_perplexity(model, tokenizer, cfg, mode, coordinator, kv_manager, device):
    from hawp_laq.eval.perplexity import compute_perplexity

    seq_len = cfg.calib.seq_len
    nsamples = cfg.calib.nsamples if cfg.mode == "local" else None

    print(f"\n{'='*60}")
    print(f"[ppl] mode={mode}  seq_len={seq_len}  nsamples={nsamples}")
    print(f"  NOTE: teacher-forcing PPL — does not exercise quant cache / scheduler path")
    print(f"{'='*60}")

    make_reset_fn(model, coordinator, kv_manager)()

    result = compute_perplexity(model, tokenizer, seq_len=seq_len, nsamples=nsamples, device=device)
    print(f"  perplexity={result['perplexity']:.4f}  nll={result['nll']:.4f}  "
          f"chunks={result['n_chunks']}  tokens={result['n_tokens']}")
    return result


def _run_needle(model, tokenizer, cfg, mode, coordinator, kv_manager, device):
    from hawp_laq.eval.needle import run_needle_test, needle_accuracy

    if cfg.mode == "local":
        context_lens = [256, 512]
        depths = [0, 50, 100]
    else:
        context_lens = [512, 1024, 2048, 4096]
        depths = [0, 25, 50, 75, 100]

    print(f"\n{'='*60}")
    print(f"[needle] mode={mode}  context_lens={context_lens}")
    print(f"{'='*60}")

    def generate_fn(prompt: str) -> str:
        outputs = generate_by_mode(
            model, tokenizer, [prompt], cfg, mode,
            coordinator=coordinator, kv_manager=kv_manager,
        )
        return outputs[0]

    results = run_needle_test(
        model, tokenizer,
        context_lens=context_lens, depths=depths,
        device=device,
        generate_fn=generate_fn,
        reset_fn=make_reset_fn(model, coordinator, kv_manager),
    )

    acc = needle_accuracy(results)
    for ctx_len in sorted(k for k in acc if isinstance(k, int)):
        a = acc[ctx_len]
        print(f"  ctx={ctx_len:>5d}: accuracy={a['accuracy']:.1%}  ({a['found']}/{a['n']})")
    print(f"  overall: accuracy={acc['overall']['accuracy']:.1%}")

    return {"accuracy_summary": acc, "details": results}


def _run_long_context_speed(model, tokenizer, cfg, mode, coordinator, kv_manager, device, seq_lens, max_new_tokens):
    print(f"\n{'='*60}")
    print(f"[long_ctx] mode={mode}  seq_lens={seq_lens}")
    print(f"{'='*60}")

    reset_fn = make_reset_fn(model, coordinator, kv_manager)

    results = []
    for seq_len in seq_lens:
        seed_text = "The "
        repeated = seed_text * ((seq_len // len(seed_text.split())) + 2)
        enc = tokenizer(repeated, return_tensors="pt")
        prompt_ids = enc["input_ids"][0][:seq_len]
        prompt = tokenizer.decode(prompt_ids)

        reset_fn()

        start = time.perf_counter()
        outputs, stats, _ = profile_generate_by_mode(
            model, tokenizer, [prompt], cfg, mode,
            coordinator=coordinator, kv_manager=kv_manager, reset_fn=reset_fn,
        )
        elapsed = time.perf_counter() - start

        input_len = len(tokenizer.encode(prompt))
        generated_text = outputs[0]
        new_tokens = len(tokenizer.encode(generated_text)) - input_len

        info = {
            "target_seq_len": seq_len,
            "input_len": input_len,
            "new_tokens": new_tokens,
            "elapsed_s": round(elapsed, 3),
            "tokens_per_s": round(max_new_tokens / elapsed, 2) if elapsed > 0 else 0,
            "peak_gpu_bytes": stats.peak_gpu_bytes,
            "peak_gpu_formatted": format_nbytes(stats.peak_gpu_bytes),
            "cache_runtime_bytes": stats.cache_runtime_bytes,
            "cache_runtime_formatted": format_nbytes(stats.cache_runtime_bytes),
        }
        results.append(info)
        print(f"  seq={seq_len:>5d}: time={info['elapsed_s']:>6.3f}s  "
              f"speed={info['tokens_per_s']:>6.1f} tok/s  peak_gpu={info['peak_gpu_formatted']}  "
              f"cache={info['cache_runtime_formatted']}")

    return results


def _print_kv_summary(model):
    from hawp_laq.eval.metrics import collect_kv_metrics, format_kv_metrics
    has_cache = any(isinstance(m, HAWPAttention) and m.use_cache_manager for m in model.modules())
    if not has_cache:
        return
    metrics = collect_kv_metrics(model)
    print(f"\n--- KV memory summary ---")
    print(format_kv_metrics(metrics))


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description="HAWP-LAQ long context eval")
    parser.add_argument("config", nargs="?", default=None)
    parser.add_argument("--mode",
                        choices=["baseline", "hawp_only", "quant_only", "hawp_quant", "hawp_quant_all", "hawp_quant_sched", "pure_quant_only"],
                        default="baseline")
    parser.add_argument("--seq-lens", nargs="+", type=int, default=[512, 1024, 2048])
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--skip-needle", action="store_true")
    parser.add_argument("--skip-ppl", action="store_true")
    args = parser.parse_args()

    if args.config is None:
        args.config = Path(__file__).resolve().parent.parent / "configs" / "run_server.yaml"

    cfg = load_config(args.config)
    device = _resolve_device(cfg.train.device)

    print("=" * 60)
    print(f"[eval] mode={args.mode}  model={cfg.model.model_id}  device={device}")
    print("=" * 60)

    model, tokenizer, _ = load_baseline_model(cfg)
    model, coordinator, kv_manager = setup_mode(model, cfg, device, args.mode)
    model.eval()

    if not args.skip_ppl:
        _run_perplexity(model, tokenizer, cfg, args.mode, coordinator, kv_manager, device)

    if not args.skip_needle:
        _run_needle(model, tokenizer, cfg, args.mode, coordinator, kv_manager, device)

    _run_long_context_speed(model, tokenizer, cfg, args.mode, coordinator, kv_manager, device, args.seq_lens, args.max_new_tokens)

    _print_kv_summary(model)


if __name__ == "__main__":
    main()
