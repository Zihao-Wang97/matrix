from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F

from hawp_laq.eval.perplexity import _load_wikitext2


def _normalize_top_k(top_k: list[int] | tuple[int, ...] | None, vocab_size: int | None = None) -> list[int]:
    values = top_k or [1, 5, 10]
    cleaned = sorted({int(k) for k in values if int(k) > 0})
    if vocab_size is not None:
        cleaned = [min(k, vocab_size) for k in cleaned]
        cleaned = sorted(set(cleaned))
    return cleaned or [1]


def ideal_distribution_metrics(
    top_k: list[int] | tuple[int, ...] | None,
    seq_len: int,
    nsamples: int | None,
) -> dict[str, float | int | None]:
    metrics: dict[str, float | int | None] = {
        "distribution_seq_len": seq_len,
        "distribution_nsamples": nsamples,
        "distribution_n_chunks": 0,
        "distribution_n_tokens": 0,
        "kl_mean": 0.0,
        "kl_p95": 0.0,
        "kl_max": 0.0,
        "argmax_agreement": 1.0,
    }
    for k in _normalize_top_k(top_k):
        metrics[f"top{k}_overlap"] = 1.0
    return metrics


@torch.inference_mode()
def _collect_stepwise_logits(
    model,
    input_ids: torch.Tensor,
    *,
    device: str,
    reset_fn: Callable[[], None] | None = None,
    coordinator=None,
    kv_manager=None,
    use_past_kv: bool = True,
) -> torch.Tensor:
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)
    input_ids = input_ids.to(device)
    bsz, seq_len = input_ids.shape
    if bsz != 1:
        raise ValueError("distribution metrics currently require batch size 1")
    if seq_len <= 1:
        return torch.empty(0, 0, device=device)

    if reset_fn is not None:
        reset_fn()

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

    logits = [outputs.logits[:, -1, :].detach()]
    past_kv = outputs.past_key_values if use_past_kv else None

    for pos in range(1, seq_len - 1):
        cur_id = input_ids[:, pos:pos + 1]
        dec_mask = torch.ones(bsz, pos + 1, device=device, dtype=torch.long)
        dec_pos = torch.tensor([[pos]], device=device, dtype=torch.long)
        fwd_kw: dict = {
            "attention_mask": dec_mask,
            "position_ids": dec_pos,
            "use_cache": True,
        }

        if kv_manager is not None:
            fwd_kw["past_key_values"] = kv_manager.get_past_kv()
        elif use_past_kv and past_kv is not None:
            fwd_kw["past_key_values"] = past_kv

        outputs = model(input_ids=cur_id, **fwd_kw)

        if kv_manager is not None:
            kv_manager.on_forward_done_from_output(outputs.past_key_values, prev_seq_len=pos)
        if use_past_kv and past_kv is not None and kv_manager is None:
            past_kv = outputs.past_key_values
        if coordinator is not None:
            coordinator.on_new_token()

        logits.append(outputs.logits[:, -1, :].detach())

    return torch.cat(logits, dim=0)


def _compute_topk_overlap(
    base_logits: torch.Tensor,
    cand_logits: torch.Tensor,
    top_k: list[int],
) -> tuple[dict[int, float], int, int]:
    vocab_size = base_logits.shape[-1]
    top_k = _normalize_top_k(top_k, vocab_size)
    max_k = max(top_k)
    base_top = torch.topk(base_logits, max_k, dim=-1).indices
    cand_top = torch.topk(cand_logits, max_k, dim=-1).indices

    n_tokens = base_logits.shape[0]
    argmax_matches = int((base_top[:, 0] == cand_top[:, 0]).sum().item())

    overlaps: dict[int, float] = {}
    for k in top_k:
        matches = base_top[:, :k].unsqueeze(2).eq(cand_top[:, :k].unsqueeze(1))
        overlap_count = matches.any(dim=2).sum(dim=1).float()
        overlaps[k] = float(overlap_count.sum().item())
    return overlaps, argmax_matches, n_tokens


@torch.inference_mode()
def compute_distribution_metrics(
    baseline_model,
    candidate_model,
    tokenizer,
    *,
    seq_len: int = 512,
    nsamples: int | None = 8,
    top_k: list[int] | None = None,
    seed: int | None = 0,
    device: str = "cpu",
    baseline_reset_fn: Callable[[], None] | None = None,
    candidate_reset_fn: Callable[[], None] | None = None,
    baseline_use_past_kv: bool = True,
    candidate_use_past_kv: bool = False,
    baseline_coordinator=None,
    baseline_kv_manager=None,
    candidate_coordinator=None,
    candidate_kv_manager=None,
) -> dict[str, float | int | None]:
    chunks = _load_wikitext2(tokenizer, seq_len, nsamples=nsamples, seed=seed)
    if len(chunks) == 0:
        metrics = ideal_distribution_metrics(top_k, seq_len=seq_len, nsamples=nsamples)
        metrics["distribution_n_chunks"] = 0
        metrics["distribution_n_tokens"] = 0
        return metrics

    top_k = _normalize_top_k(top_k)
    kl_values: list[torch.Tensor] = []
    kl_sum = 0.0
    kl_max = 0.0
    total_tokens = 0
    argmax_matches = 0
    topk_overlap_sums = {k: 0.0 for k in top_k}

    for chunk in chunks:
        base_logits = _collect_stepwise_logits(
            baseline_model,
            chunk,
            device=device,
            reset_fn=baseline_reset_fn,
            coordinator=baseline_coordinator,
            kv_manager=baseline_kv_manager,
            use_past_kv=baseline_use_past_kv,
        )
        cand_logits = _collect_stepwise_logits(
            candidate_model,
            chunk,
            device=device,
            reset_fn=candidate_reset_fn,
            coordinator=candidate_coordinator,
            kv_manager=candidate_kv_manager,
            use_past_kv=candidate_use_past_kv,
        )

        if base_logits.shape != cand_logits.shape:
            raise ValueError(
                f"baseline/candidate logits shape mismatch: "
                f"{tuple(base_logits.shape)} vs {tuple(cand_logits.shape)}"
            )
        if base_logits.numel() == 0:
            continue

        base_float = base_logits.float()
        cand_float = cand_logits.float()
        base_logp = F.log_softmax(base_float, dim=-1)
        cand_logp = F.log_softmax(cand_float, dim=-1)
        base_prob = base_logp.exp()
        kl = (base_prob * (base_logp - cand_logp)).sum(dim=-1).clamp_min(0.0)

        kl_sum += float(kl.sum().item())
        kl_max = max(kl_max, float(kl.max().item()))
        kl_values.append(kl.detach().cpu())

        overlaps, matches, n_tokens = _compute_topk_overlap(base_float, cand_float, top_k)
        argmax_matches += matches
        total_tokens += n_tokens
        for k, value in overlaps.items():
            topk_overlap_sums[k] += value

        del base_logits, cand_logits, base_float, cand_float, base_logp, cand_logp, base_prob, kl
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    metrics: dict[str, float | int | None] = {
        "distribution_seq_len": seq_len,
        "distribution_nsamples": nsamples,
        "distribution_n_chunks": len(chunks),
        "distribution_n_tokens": total_tokens,
    }
    if total_tokens <= 0:
        metrics.update({
            "kl_mean": None,
            "kl_p95": None,
            "kl_max": None,
            "argmax_agreement": None,
        })
        for k in top_k:
            metrics[f"top{k}_overlap"] = None
        return metrics

    all_kl = torch.cat(kl_values) if kl_values else torch.empty(0)
    metrics["kl_mean"] = kl_sum / total_tokens
    metrics["kl_p95"] = float(torch.quantile(all_kl, 0.95).item()) if all_kl.numel() else None
    metrics["kl_max"] = kl_max
    metrics["argmax_agreement"] = argmax_matches / total_tokens
    for k in top_k:
        metrics[f"top{k}_overlap"] = topk_overlap_sums[k] / (total_tokens * k)
    return metrics
