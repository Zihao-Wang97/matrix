#!/usr/bin/env python
"""Long context eval: python scripts/05_run_long_context_eval.py [config] [--mode MODE]"""

import argparse
import time
from pathlib import Path

import torch
from transformers import AutoConfig

from hawp_laq.config import load_config
from hawp_laq.runtime.generate import load_baseline_model, _resolve_device, generate_text, _fmt_bytes
from hawp_laq.modeling.modeling_llama_hawp import convert_llama_to_hawp


def _make_long_prompt(tokenizer, target_len: int, seed_text: str = "The ") -> str:
    tokens = tokenizer(seed_text, return_tensors="pt")["input_ids"][0]
    if len(tokens) >= target_len:
        return tokenizer.decode(tokens[:target_len])

    repeated = seed_text * ((target_len // len(tokens)) + 2)
    enc = tokenizer(repeated, return_tensors="pt")
    return tokenizer.decode(enc["input_ids"][0][:target_len])


def _profile_generation(model, tokenizer, prompt: str, max_new_tokens: int, device: str) -> dict:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    start = time.perf_counter()

    with torch.inference_mode():
        out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    elapsed = time.perf_counter() - start

    peak_mem = 0
    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated()

    return {
        "input_len": inputs["input_ids"].shape[1],
        "output_len": out_ids.shape[1],
        "new_tokens": out_ids.shape[1] - inputs["input_ids"].shape[1],
        "elapsed_s": round(elapsed, 3),
        "tokens_per_s": round(max_new_tokens / elapsed, 2) if elapsed > 0 else 0,
        "peak_gpu_bytes": peak_mem,
        "peak_gpu_formatted": _fmt_bytes(peak_mem),
    }


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description="HAWP-LAQ long context evaluation")
    parser.add_argument(
        "config",
        nargs="?",
        default=None,
        help="Path to yaml config (default: configs/run_server.yaml)",
    )
    parser.add_argument(
        "--mode",
        choices=["baseline", "hawp_only", "hawp_quant", "hawp_quant_sched"],
        default="baseline",
    )
    parser.add_argument("--seq-lens", nargs="+", type=int, default=[512, 1024, 2048])
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()

    if args.config is None:
        args.config = Path(__file__).resolve().parent.parent / "configs" / "run_server.yaml"

    cfg = load_config(args.config)
    device = _resolve_device(cfg.train.device)

    print("=" * 60)
    print(f"[long_ctx] mode={args.mode}  device={device}")
    print(f"[long_ctx] seq_lens={args.seq_lens}  max_new_tokens={args.max_new_tokens}")
    print("=" * 60)

    model, tokenizer, _ = load_baseline_model(cfg)

    if args.mode != "baseline":
        r_k = cfg.projector.r_k
        r_v = cfg.projector.r_v
        model = convert_llama_to_hawp(model, r_k=r_k, r_v=r_v)
        model = model.to(device)
        model.eval()

        if Path(cfg.projector.output_dir).exists():
            from hawp_laq.runtime.projector_bank import load_projectors
            load_projectors(model, cfg.projector.output_dir)
            print(f"[long_ctx] loaded projectors from {cfg.projector.output_dir}")

    results = []
    for seq_len in args.seq_lens:
        prompt = _make_long_prompt(tokenizer, seq_len)
        print(f"\n[long_ctx] seq_len={seq_len}  actual_input={len(tokenizer(prompt)['input_ids'])}")

        info = _profile_generation(model, tokenizer, prompt, args.max_new_tokens, device)
        info["mode"] = args.mode
        info["target_seq_len"] = seq_len
        results.append(info)

        print(f"  input={info['input_len']}  output={info['output_len']}  "
              f"time={info['elapsed_s']}s  speed={info['tokens_per_s']} tok/s  "
              f"peak_gpu={info['peak_gpu_formatted']}")

    print("\n" + "=" * 60)
    print("[long_ctx] summary")
    print(f"{'seq_len':>8} {'time(s)':>8} {'tok/s':>8} {'peak_gpu':>12}")
    print("-" * 40)
    for r in results:
        print(f"{r['target_seq_len']:>8d} {r['elapsed_s']:>8.3f} {r['tokens_per_s']:>8.2f} {r['peak_gpu_formatted']:>12}")


if __name__ == "__main__":
    main()
