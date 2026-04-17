from __future__ import annotations

from typing import Iterator

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


def get_calib_dataloader(
    tokenizer: AutoTokenizer,
    nsamples: int = 8,
    seq_len: int = 128,
    dataset_name: str = "wikitext2",
    seed: int = 42,
) -> DataLoader:
    tokens = _load_dataset_tokens(tokenizer, dataset_name, seed)
    tokens = tokens[: nsamples * seq_len + 1]
    samples = []
    for i in range(0, len(tokens) - seq_len, seq_len):
        input_ids = torch.tensor(tokens[i : i + seq_len], dtype=torch.long)
        samples.append(input_ids)
        if len(samples) >= nsamples:
            break
    return DataLoader(samples, batch_size=1, shuffle=False)


def _load_dataset_tokens(
    tokenizer: AutoTokenizer,
    dataset_name: str,
    seed: int,
) -> list[int]:
    if dataset_name == "wikitext2":
        try:
            from datasets import load_dataset

            ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
            text = "\n\n".join(ds["text"])
        except Exception:
            text = _fallback_text()
    else:
        text = _fallback_text()

    enc = tokenizer(text, return_tensors="pt")
    return enc["input_ids"][0].tolist()


def _fallback_text() -> str:
    return (
        "The quick brown fox jumps over the lazy dog. "
        * 200
    )
