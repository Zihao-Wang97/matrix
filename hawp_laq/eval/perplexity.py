from __future__ import annotations

import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


def _load_wikitext2(tokenizer, seq_len: int, split: str = "test", nsamples: int | None = None):
    try:
        from datasets import load_dataset
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    except Exception:
        from datasets import load_dataset
        dataset = load_dataset("parquet", data_files="data/wikitext2/test.parquet", split="train")

    text = "\n\n".join(dataset["text"])
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids

    total_len = input_ids.shape[1]
    n_chunks = total_len // seq_len
    if n_chunks == 0:
        return []

    input_ids = input_ids[:, :n_chunks * seq_len]
    chunks = input_ids.reshape(n_chunks, seq_len)

    if nsamples is not None and nsamples < n_chunks:
        indices = torch.randperm(n_chunks)[:nsamples]
        chunks = chunks[indices]

    return chunks


@torch.inference_mode()
def compute_perplexity(
    model,
    tokenizer,
    seq_len: int = 2048,
    nsamples: int | None = None,
    device: str = "cpu",
) -> dict:
    chunks = _load_wikitext2(tokenizer, seq_len, nsamples=nsamples)
    if len(chunks) == 0:
        return {"perplexity": float("nan"), "nll": float("nan"), "n_chunks": 0}

    nll_total = 0.0
    n_tokens = 0

    for chunk in chunks:
        input_ids = chunk.unsqueeze(0).to(device)
        outputs = model(input_ids=input_ids, use_cache=False)
        logits = outputs.logits[:, :-1, :]
        targets = input_ids[:, 1:]

        loss_fct = torch.nn.CrossEntropyLoss(reduction="sum")
        nll = loss_fct(logits.reshape(-1, logits.size(-1)), targets.reshape(-1)).item()
        nll_total += nll
        n_tokens += targets.numel()

    avg_nll = nll_total / n_tokens if n_tokens > 0 else float("nan")
    ppl = math.exp(avg_nll) if not math.isnan(avg_nll) else float("nan")

    return {
        "perplexity": ppl,
        "nll": avg_nll,
        "n_chunks": len(chunks),
        "n_tokens": n_tokens,
        "seq_len": seq_len,
    }
