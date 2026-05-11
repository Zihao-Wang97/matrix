#!/usr/bin/env python
"""Validate q_len=1 blockwise archive decode against the original full-cat path.

The script uses one model and runs the same prompt twice:

  1. disable HAWPAttention.use_decode_blockwise_attention (legacy full-cat)
  2. enable HAWPAttention.use_decode_blockwise_attention (blockwise)

It compares the first decode-step logits after prefill and reports max-abs
error, cosine similarity, KL(full || blockwise), top-token agreement, and
decode-only peak GPU memory.

Example:
  python scripts/06d_validate_decode_blockwise.py configs/new_rank.yaml --seq-len 8192
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from hawp_laq.config import load_config
from hawp_laq.modeling.attention_hawp import HAWPAttention
from hawp_laq.runtime.forward_utils import prefill_forward_last_logits
from hawp_laq.runtime.generate import _resolve_device, load_baseline_model
from hawp_laq.runtime.mode_runner import make_reset_fn, setup_mode
from hawp_laq.utils.memory import format_nbytes


def _format_signed_nbytes(nbytes: int) -> str:
    sign = "-" if nbytes < 0 else ""
    return sign + format_nbytes(abs(nbytes))


def _build_prompt_for_profile(tokenizer, target_seq_len: int) -> tuple[str, int]:
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


def _set_blockwise(model, enabled: bool, block_size: int | None = None) -> None:
    for mod in model.modules():
        if isinstance(mod, HAWPAttention):
            mod.use_decode_blockwise_attention = enabled
            if block_size is not None:
                mod.decode_archive_block_size = block_size


@torch.inference_mode()
def _first_decode_logits(
    *,
    model,
    tokenizer,
    prompt: str,
    reset_fn,
    blockwise: bool,
    block_size: int,
) -> dict[str, Any]:
    _set_blockwise(model, blockwise, block_size)
    reset_fn()

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    bsz, prompt_len = input_ids.shape
    prefill_mask = torch.ones(bsz, prompt_len, device=model.device, dtype=torch.long)
    prefill_pos = torch.arange(prompt_len, device=model.device, dtype=torch.long).unsqueeze(0)

    prefill = prefill_forward_last_logits(
        model,
        input_ids=input_ids,
        attention_mask=prefill_mask,
        position_ids=prefill_pos,
        use_cache=True,
    )
    next_token = torch.argmax(prefill.logits[:, -1, :], dim=-1, keepdim=True)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    alloc_before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

    attention_mask = torch.ones(1, prompt_len + 1, device=model.device, dtype=torch.long)
    position_ids = torch.tensor([[prompt_len]], device=model.device, dtype=torch.long)
    decode = model(
        input_ids=next_token,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=True,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    alloc_after = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

    logits = decode.logits[:, -1, :].detach().float().cpu()
    return {
        "logits": logits,
        "prefill_next_token": int(next_token.item()),
        "decode_top_token": int(torch.argmax(logits, dim=-1).item()),
        "decode_peak_bytes": int(peak),
        "decode_peak": format_nbytes(int(peak)),
        "decode_peak_delta_bytes": int(peak - alloc_before),
        "decode_peak_delta": _format_signed_nbytes(int(peak - alloc_before)),
        "allocated_before_decode": format_nbytes(int(alloc_before)),
        "allocated_after_decode": format_nbytes(int(alloc_after)),
    }


def _compare_logits(full_logits: torch.Tensor, block_logits: torch.Tensor) -> dict[str, Any]:
    diff = (full_logits - block_logits).abs()
    full_prob = F.softmax(full_logits, dim=-1)
    block_logprob = F.log_softmax(block_logits, dim=-1)
    full_logprob = F.log_softmax(full_logits, dim=-1)
    kl = F.kl_div(block_logprob, full_prob, reduction="batchmean")
    cosine = F.cosine_similarity(full_logits, block_logits, dim=-1).mean()
    return {
        "max_abs_err": float(diff.max().item()),
        "mean_abs_err": float(diff.mean().item()),
        "cosine": float(cosine.item()),
        "kl_full_to_blockwise": float(kl.item()),
        "kl_blockwise_to_full": float(F.kl_div(full_logprob, F.softmax(block_logits, dim=-1), reduction="batchmean").item()),
        "top_token_match": int(torch.argmax(full_logits, dim=-1).item()) == int(torch.argmax(block_logits, dim=-1).item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate q_len=1 blockwise archive decode")
    parser.add_argument("config", nargs="?", default="configs/new_rank.yaml")
    parser.add_argument("--seq-len", type=int, default=8192)
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = _resolve_device(cfg.train.device)
    cfg.generation.max_new_tokens = 2

    print("=" * 80)
    print(f"[decode-blockwise] config={args.config}")
    print(f"[decode-blockwise] seq_len={args.seq_len} block_size={args.block_size}")
    print("=" * 80)

    model, tokenizer, _ = load_baseline_model(cfg)
    model, coordinator, kv_manager = setup_mode(model, cfg, device, "hawp_quant")
    if coordinator is not None or kv_manager is not None:
        raise RuntimeError("This validator expects plain hawp_quant without scheduler/kv_manager")
    model.eval()
    reset_fn = make_reset_fn(model)
    prompt, actual_seq_len = _build_prompt_for_profile(tokenizer, args.seq_len)

    full = _first_decode_logits(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        reset_fn=reset_fn,
        blockwise=False,
        block_size=args.block_size,
    )
    block = _first_decode_logits(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        reset_fn=reset_fn,
        blockwise=True,
        block_size=args.block_size,
    )

    metrics = _compare_logits(full["logits"], block["logits"])
    result = {
        "config": args.config,
        "requested_seq_len": args.seq_len,
        "actual_seq_len": actual_seq_len,
        "block_size": args.block_size,
        "full_cat": {k: v for k, v in full.items() if k != "logits"},
        "blockwise": {k: v for k, v in block.items() if k != "logits"},
        "metrics": metrics,
        "decode_peak_delta_advantage_bytes": full["decode_peak_delta_bytes"] - block["decode_peak_delta_bytes"],
        "decode_peak_delta_advantage": _format_signed_nbytes(full["decode_peak_delta_bytes"] - block["decode_peak_delta_bytes"]),
    }

    print("\n[decode-blockwise] metrics")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    print("\n[decode-blockwise] memory")
    print(f"  full_cat decode_peak_delta: {full['decode_peak_delta']}")
    print(f"  blockwise decode_peak_delta: {block['decode_peak_delta']}")
    print(f"  advantage: {result['decode_peak_delta_advantage']}")
    print(f"  top tokens: full={full['decode_top_token']} blockwise={block['decode_top_token']}")

    output = Path(args.output) if args.output else Path("artifacts/peak_segments") / f"decode_blockwise_validate_{actual_seq_len}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n[decode-blockwise] saved to {output}")


if __name__ == "__main__":
    main()
