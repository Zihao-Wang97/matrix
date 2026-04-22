from __future__ import annotations

import json
import random
from pathlib import Path

import torch


_NEEDLE_TEMPLATES = [
    "The secret code is {needle}. Remember this code.",
    "The passkey for the vault is {needle}. Do not share it.",
    "The project identifier is {needle}. Write it down.",
]

_HAYSTACK_PARAGRAPH = (
    "The history of computing spans many centuries, from the abacus to modern supercomputers. "
    "Early mechanical calculators like the Pascaline and the Difference Engine laid the groundwork "
    "for electronic computation. The invention of the transistor in 1947 revolutionized the field, "
    "leading to integrated circuits and microprocessors. By the 1980s, personal computers became "
    "commonplace in homes and offices. The internet transformed communication and commerce in the "
    "1990s. Cloud computing and mobile devices further changed how we interact with technology. "
    "Artificial intelligence and machine learning represent the latest frontier, with neural networks "
    "achieving remarkable results in vision, language, and game playing. "
)


def _make_needle_haystack(
    needle: str,
    context_len: int,
    depth_percent: float,
    tokenizer,
) -> str:
    needle_text = random.choice(_NEEDLE_TEMPLATES).format(needle=needle)
    question = f"What is the secret code mentioned in the text?"

    needle_tokens = tokenizer.encode(needle_text, add_special_tokens=False)
    question_tokens = tokenizer.encode(question, add_special_tokens=False)
    needle_len = len(needle_tokens)
    question_len = len(question_tokens)
    target_haystack_len = context_len - needle_len - question_len - 10

    paragraph_tokens = tokenizer.encode(_HAYSTACK_PARAGRAPH, add_special_tokens=False)
    n_repeats = (target_haystack_len // len(paragraph_tokens)) + 2
    haystack_tokens = (paragraph_tokens * n_repeats)[:target_haystack_len]

    insert_pos = int(len(haystack_tokens) * depth_percent / 100.0)
    all_tokens = (
        haystack_tokens[:insert_pos]
        + needle_tokens
        + haystack_tokens[insert_pos:]
        + question_tokens
    )
    return tokenizer.decode(all_tokens[:context_len]), needle_text


@torch.inference_mode()
def run_needle_test(
    model,
    tokenizer,
    context_lens: list[int] | None = None,
    depths: list[int] | None = None,
    needle: str = "7294",
    device: str = "cpu",
    max_new_tokens: int = 32,
) -> list[dict]:
    if context_lens is None:
        context_lens = [512, 1024, 2048]
    if depths is None:
        depths = [0, 25, 50, 75, 100]

    results = []
    for ctx_len in context_lens:
        for depth in depths:
            prompt, needle_text = _make_needle_haystack(
                needle, ctx_len, depth, tokenizer,
            )
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

            outputs = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
            generated = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
            found = needle in generated

            results.append({
                "context_len": ctx_len,
                "depth_percent": depth,
                "needle": needle,
                "generated": generated,
                "found": found,
            })

    return results


def needle_accuracy(results: list[dict]) -> dict:
    from collections import defaultdict

    by_len = defaultdict(list)
    for r in results:
        by_len[r["context_len"]].append(r["found"])

    summary = {}
    for ctx_len, founds in sorted(by_len.items()):
        acc = sum(founds) / len(founds) if founds else 0.0
        summary[ctx_len] = {"accuracy": acc, "n": len(founds), "found": sum(founds)}

    overall = sum(r["found"] for r in results) / len(results) if results else 0.0
    summary["overall"] = {"accuracy": overall, "n": len(results)}

    return summary
