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
    _setup_hawp_quant_on_model,
    _setup_quant_only_on_model,
    generate_text,
    generate_hawp_quant,
)
from hawp_laq.modeling.attention_hawp import HAWPAttention
from hawp_laq.modeling.modeling_llama_hawp import convert_llama_to_hawp


def _setup_mode(model, cfg, device, mode: str):
    if mode == "baseline":
        return model, None

    if mode == "hawp_only":
        from hawp_laq.runtime.projector_bank import load_projectors
        r_k, r_v = cfg.projector.r_k, cfg.projector.r_v
        model = convert_llama_to_hawp(model, r_k=r_k, r_v=r_v)
        model = model.to(device).eval()
        if Path(cfg.projector.output_dir).exists():
            load_projectors(model, cfg.projector.output_dir)
        return model, None

    if mode == "quant_only":
        model, _ = _setup_quant_only_on_model(model, cfg, device)
        return model, "quant"

    if mode in ("hawp_quant", "hawp_quant_all", "hawp_quant_sched"):
        if mode == "hawp_quant_all":
            from hawp_laq.runtime.generate import _setup_hawp_quant_all_on_model
            model = _setup_hawp_quant_all_on_model(model, cfg, device)
        else:
            model = _setup_hawp_quant_on_model(model, cfg, device)
        coordinator = None
        if mode == "hawp_quant_sched":
            from hawp_laq.runtime.scheduler import TokenBudgetScheduler
            from hawp_laq.runtime.cache_manager import ModelCacheCoordinator
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


def _run_perplexity(model, tokenizer, cfg, mode, coordinator, device):
    from hawp_laq.eval.perplexity import compute_perplexity

    seq_len = cfg.calib.seq_len
    nsamples = cfg.calib.nsamples if cfg.mode == "local" else None

    print(f"\n{'='*60}")
    print(f"[ppl] mode={mode}  seq_len={seq_len}  nsamples={nsamples}")
    print(f"{'='*60}")

    for mod in model.modules():
        if isinstance(mod, HAWPAttention) and mod.use_cache_manager:
            mod.reset_quant_cache()
    if coordinator is not None:
        coordinator.reset()

    result = compute_perplexity(model, tokenizer, seq_len=seq_len, nsamples=nsamples, device=device)
    print(f"  perplexity={result['perplexity']:.4f}  nll={result['nll']:.4f}  "
          f"chunks={result['n_chunks']}  tokens={result['n_tokens']}")
    return result


def _run_needle(model, tokenizer, cfg, mode, coordinator, device):
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

    results = run_needle_test(
        model, tokenizer,
        context_lens=context_lens, depths=depths,
        device=device,
    )

    acc = needle_accuracy(results)
    for ctx_len in sorted(k for k in acc if isinstance(k, int)):
        a = acc[ctx_len]
        print(f"  ctx={ctx_len:>5d}: accuracy={a['accuracy']:.1%}  ({a['found']}/{a['n']})")
    print(f"  overall: accuracy={acc['overall']['accuracy']:.1%}")

    return {"accuracy_summary": acc, "details": results}


def _run_long_context_speed(model, tokenizer, cfg, mode, coordinator, device, seq_lens, max_new_tokens):
    print(f"\n{'='*60}")
    print(f"[long_ctx] mode={mode}  seq_lens={seq_lens}")
    print(f"{'='*60}")

    results = []
    for seq_len in seq_lens:
        seed_text = "The "
        repeated = seed_text * ((seq_len // len(seed_text.split())) + 2)
        enc = tokenizer(repeated, return_tensors="pt")
        prompt_ids = enc["input_ids"][0][:seq_len]
        prompt = tokenizer.decode(prompt_ids)

        for mod in model.modules():
            if isinstance(mod, HAWPAttention) and mod.use_cache_manager:
                mod.reset_quant_cache()
        if coordinator is not None:
            coordinator.reset()

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()

        with torch.inference_mode():
            out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

        elapsed = time.perf_counter() - start
        peak_mem = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0

        info = {
            "target_seq_len": seq_len,
            "input_len": inputs["input_ids"].shape[1],
            "new_tokens": out_ids.shape[1] - inputs["input_ids"].shape[1],
            "elapsed_s": round(elapsed, 3),
            "tokens_per_s": round(max_new_tokens / elapsed, 2) if elapsed > 0 else 0,
            "peak_gpu_bytes": peak_mem,
            "peak_gpu_formatted": _fmt_bytes(peak_mem),
        }
        results.append(info)
        print(f"  seq={seq_len:>5d}: time={info['elapsed_s']:>6.3f}s  "
              f"speed={info['tokens_per_s']:>6.1f} tok/s  peak_gpu={info['peak_gpu_formatted']}")

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
                        choices=["baseline", "hawp_only", "quant_only", "hawp_quant", "hawp_quant_all", "hawp_quant_sched"],
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
    model, coordinator = _setup_mode(model, cfg, device, args.mode)
    model.eval()

    if not args.skip_ppl:
        _run_perplexity(model, tokenizer, cfg, args.mode, coordinator, device)

    if not args.skip_needle:
        _run_needle(model, tokenizer, cfg, args.mode, coordinator, device)

    _run_long_context_speed(model, tokenizer, cfg, args.mode, coordinator, device, args.seq_lens, args.max_new_tokens)

    _print_kv_summary(model)


if __name__ == "__main__":
    main()
