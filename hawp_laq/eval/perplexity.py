from __future__ import annotations

import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


def _tokenize_texts_in_segments(tokenizer, text_list: list[str], max_chars: int = 20000) -> torch.Tensor:
    token_ids: list[int] = []
    sep_ids = tokenizer.encode("\n\n", add_special_tokens=False)

    for text_idx, text in enumerate(text_list):
        if text_idx > 0:
            token_ids.extend(sep_ids)
        if not text:
            continue

        # Avoid tokenizer/model-max-length warnings by never encoding a huge
        # corpus as one sequence; the model still receives fixed-size chunks.
        for start in range(0, len(text), max_chars):
            segment = text[start:start + max_chars]
            if not segment:
                continue
            token_ids.extend(tokenizer.encode(segment, add_special_tokens=False))

    return torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)


def _load_wikitext2(tokenizer, seq_len: int, split: str = "test", nsamples: int | None = None):
    def _load_local_txt() -> list[str] | None:
        split_txt = Path(f"data/wikitext2_{split}.txt")
        train_txt = Path("data/wikitext2_train.txt")
        local_txt = split_txt if split_txt.exists() else train_txt
        if local_txt.exists():
            return [local_txt.read_text(encoding="utf-8")]
        return None

    local_txt = _load_local_txt()
    if local_txt is not None:
        dataset = local_txt
    else:
        dataset = None

    if dataset is None:
        try:
            from datasets import load_dataset
            try:
                dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
            except Exception:
                local_parquet = Path("data/wikitext2/test.parquet")
                if local_parquet.exists():
                    dataset = load_dataset("parquet", data_files=str(local_parquet), split="train")
                else:
                    raise RuntimeError(
                        "Cannot load wikitext-2: online fetch failed and no local "
                        "data/wikitext2_test.txt, data/wikitext2_train.txt, or "
                        "data/wikitext2/test.parquet found."
                    )
        except ImportError:
            local_parquet = Path("data/wikitext2/test.parquet")
            if local_parquet.exists():
                import pandas as pd
                df = pd.read_parquet(local_parquet)
                texts = df["text"].tolist() if "text" in df.columns else []
                dataset = type("_DS", (), {"__getitem__": lambda s, k: texts[k], "__len__": lambda s: len(texts)})()
            else:
                raise RuntimeError(
                    "Cannot load wikitext-2: 'datasets' package is not installed and no "
                    "local data/wikitext2_test.txt, data/wikitext2_train.txt, or "
                    "data/wikitext2/test.parquet found."
                )

    if isinstance(dataset, list):
        text_list = dataset
    else:
        text_list = dataset["text"]

    input_ids = _tokenize_texts_in_segments(tokenizer, text_list)

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
    """Compute teacher-forcing perplexity (NLL per token).

    NOTE: This uses ``use_cache=False`` single-forward-per-chunk, so it
    does **not** exercise the HAWP quant cache or scheduler path.  For
    modes like ``hawp_quant_sched``, this metric reflects model quality
    but not runtime cache behaviour.
    """
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


@torch.inference_mode()
def compute_stepwise_ppl(
    model,
    tokenizer,
    coordinator=None,
    kv_manager=None,
    reset_fn=None,
    seq_len: int = 2048,
    nsamples: int | None = None,
    device: str = "cpu",
    use_past_kv: bool = True,
) -> dict:
    """Compute perplexity via minimal prefill + token-by-token decode with ``use_cache=True``.

    Unlike ``compute_perplexity`` (single forward, ``use_cache=False``), this
    does a real prefill-then-decode loop so the HAWP quant cache, scheduler,
    and pure-quant past-KV paths are fully exercised.  The resulting PPL
    reflects the *actual runtime* quantisation behaviour.

    For each chunk:
      1. Reset caches.
      2. Prefill only the first token (warmup) → collect 1 NLL (position 0→1).
      3. Decode token-by-token for positions 1..T-1 → collect decode logits
         and compute teacher-forcing NLL at each step.
      4. Each decode step calls ``coordinator.on_new_token()``,
         ``kv_manager.on_forward_done()``, and ``kv_manager.get_past_kv()`` as
         appropriate.

    Because almost all NLL comes from the decode path, the metric is sensitive
    to quantisation: changing ``budget`` / ``recent_window`` in
    ``hawp_quant_sched`` will shift ΔPPL, and ``pure_quant_only`` will no
    longer match the baseline.

    Args:
        model: The model (may have HAWP / quant cache installed).
        tokenizer: The tokenizer.
        coordinator: Optional ModelCacheCoordinator for sched mode.
        kv_manager: Optional PureQuantKVManager for pure_quant_only.
        reset_fn: Optional callable to reset cache between chunks.
        seq_len: Chunk length.
        nsamples: Max number of chunks to evaluate.
        device: Device string.
        use_past_kv: If True, pass past_key_values back for baseline/hawp_only
            modes so decode sees the full context.

    Returns:
        dict with perplexity, nll, n_chunks, n_tokens, seq_len,
        nll_prefill, nll_decode, n_tokens_prefill, n_tokens_decode.
    """
    from hawp_laq.modeling.attention_hawp import HAWPAttention

    chunks = _load_wikitext2(tokenizer, seq_len, nsamples=nsamples)
    if len(chunks) == 0:
        return {"perplexity": float("nan"), "nll": float("nan"), "n_chunks": 0, "n_tokens": 0, "seq_len": seq_len}

    loss_fct = torch.nn.CrossEntropyLoss(reduction="sum")
    nll_total = 0.0
    n_tokens = 0
    nll_prefill_total = 0.0
    nll_decode_total = 0.0
    n_tokens_prefill = 0
    n_tokens_decode = 0

    for chunk in chunks:
        input_ids = chunk.unsqueeze(0).to(device)
        bsz, T = input_ids.shape

        if reset_fn is not None:
            reset_fn()
        elif coordinator is not None or kv_manager is not None:
            for mod in model.modules():
                if isinstance(mod, HAWPAttention) and mod.use_cache_manager:
                    mod.reset_quant_cache()
            if coordinator is not None:
                coordinator.reset()
            if kv_manager is not None:
                kv_manager.reset_caches()

        # --- Prefill: only the first token (warmup) ---
        prefill_ids = input_ids[:, :1]
        prefill_mask = torch.ones(bsz, 1, device=device, dtype=torch.long)
        prefill_pos = torch.tensor([[0]], device=device, dtype=torch.long)

        outputs = model(
            input_ids=prefill_ids,
            attention_mask=prefill_mask,
            position_ids=prefill_pos,
            use_cache=True,
        )

        if kv_manager is not None:
            kv_manager.on_forward_done_from_output(outputs.past_key_values, prev_seq_len=0)

        if coordinator is not None:
            coordinator.on_prefill(1)

        # Prefill NLL: logits at position 0 predict position 1
        if T > 1:
            pf_logits = outputs.logits[:, -1, :]
            pf_target = input_ids[:, 1]
            nll_pf = loss_fct(pf_logits, pf_target).item()
            nll_prefill_total += nll_pf
            n_tokens_prefill += bsz
            nll_total += nll_pf
            n_tokens += bsz

        # --- Decode loop: positions 1 through T-1 (teacher-forcing) ---
        past_kv = outputs.past_key_values if use_past_kv else None

        for t in range(1, T):
            cur_id = input_ids[:, t:t + 1]

            dec_mask = torch.ones(bsz, t + 1, device=device, dtype=torch.long)
            dec_pos = torch.tensor([[t]], device=device, dtype=torch.long)

            fwd_kw: dict = {
                "attention_mask": dec_mask,
                "position_ids": dec_pos,
                "use_cache": True,
            }

            if kv_manager is not None:
                fwd_kw["past_key_values"] = kv_manager.get_past_kv()
            elif use_past_kv and past_kv is not None:
                fwd_kw["past_key_values"] = past_kv

            dec_outputs = model(input_ids=cur_id, **fwd_kw)

            if kv_manager is not None:
                kv_manager.on_forward_done_from_output(dec_outputs.past_key_values, prev_seq_len=t)

            if use_past_kv and past_kv is not None and kv_manager is None:
                past_kv = dec_outputs.past_key_values

            if coordinator is not None:
                coordinator.on_new_token()

            # Decode NLL: logits at position t predict position t+1
            if t < T - 1:
                dec_logits = dec_outputs.logits[:, -1, :]
                dec_target = input_ids[:, t + 1]
                nll_dec = loss_fct(dec_logits, dec_target).item()
                nll_decode_total += nll_dec
                n_tokens_decode += bsz
                nll_total += nll_dec
                n_tokens += bsz

    avg_nll = nll_total / n_tokens if n_tokens > 0 else float("nan")
    ppl = math.exp(avg_nll) if not math.isnan(avg_nll) else float("nan")

    return {
        "perplexity": ppl,
        "nll": avg_nll,
        "n_chunks": len(chunks),
        "n_tokens": n_tokens,
        "seq_len": seq_len,
        "nll_prefill": nll_prefill_total / n_tokens_prefill if n_tokens_prefill > 0 else float("nan"),
        "nll_decode": nll_decode_total / n_tokens_decode if n_tokens_decode > 0 else float("nan"),
        "n_tokens_prefill": n_tokens_prefill,
        "n_tokens_decode": n_tokens_decode,
    }
