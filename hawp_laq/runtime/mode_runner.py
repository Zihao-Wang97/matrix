from __future__ import annotations

from typing import Callable

import torch

from hawp_laq.config import HAWPLAQConfig
from hawp_laq.modeling.attention_hawp import HAWPAttention
from hawp_laq.runtime.generate import (
    _convert_and_load_projectors,
    _setup_hawp_quant_on_model,
    _setup_hawp_quant_all_on_model,
    _setup_quant_only_on_model,
    _setup_pure_quant_only_on_model,
    stepwise_greedy_generate,
    generate_hawp_quant,
    generate_pure_quant_only,
)
from hawp_laq.runtime.cache_stats import (
    CacheStats,
    aggregate_cache_stats,
    collect_cache_stats,
    collect_cache_stats_from_tracker,
    compute_baseline_kv_bytes,
)
from hawp_laq.runtime.forward_utils import prefill_forward_last_logits
from hawp_laq.runtime.past_kv_tracker import PastKVTracker


def setup_mode(model, cfg: HAWPLAQConfig, device: str, mode: str):
    if mode == "baseline":
        return model, None, None

    if mode == "hawp_only":
        model, r_k, r_v = _convert_and_load_projectors(model, cfg, device, "hawp_only")
        return model, None, None

    if mode == "quant_only":
        model, head_dim = _setup_quant_only_on_model(model, cfg, device)
        return model, None, None

    if mode == "hawp_quant":
        model = _setup_hawp_quant_on_model(model, cfg, device)
        return model, None, None

    if mode == "hawp_quant_all":
        model = _setup_hawp_quant_all_on_model(model, cfg, device)
        return model, None, None

    if mode == "hawp_quant_sched":
        from hawp_laq.runtime.scheduler import TokenBudgetScheduler
        from hawp_laq.runtime.cache_manager import ModelCacheCoordinator

        model = _setup_hawp_quant_on_model(model, cfg, device)
        sched = TokenBudgetScheduler(
            total_budget=cfg.sched.total_budget,
            recent_window=cfg.sched.recent_window,
            high_ratio=cfg.sched.high_ratio,
            low_ratio=cfg.sched.low_ratio,
            drop_strategy=getattr(cfg.sched, "drop_strategy", "position"),
        )
        coordinator = ModelCacheCoordinator.from_model(
            model, sched,
            drop_strategy=getattr(cfg.sched, "drop_strategy", "position"),
        )
        return model, coordinator, None

    if mode == "pure_quant_only":
        model, head_dim, kv_manager = _setup_pure_quant_only_on_model(model, cfg, device)
        return model, None, kv_manager

    raise ValueError(f"Unknown mode: {mode}")


def make_reset_fn(model, coordinator=None, kv_manager=None) -> Callable:
    def _reset():
        for mod in model.modules():
            if isinstance(mod, HAWPAttention) and mod.use_cache_manager:
                mod.reset_quant_cache()
        if coordinator is not None:
            coordinator.reset()
        if kv_manager is not None:
            kv_manager.reset_caches()

    return _reset


def generate_by_mode(
    model,
    tokenizer,
    prompts: list[str],
    cfg: HAWPLAQConfig,
    mode: str,
    coordinator=None,
    kv_manager=None,
) -> list[str]:
    if mode in ("baseline", "hawp_only"):
        return stepwise_greedy_generate(model, tokenizer, prompts, cfg.generation.max_new_tokens)

    if mode in ("hawp_quant", "hawp_quant_all", "hawp_quant_sched", "quant_only"):
        return generate_hawp_quant(model, tokenizer, prompts, cfg, coordinator=coordinator)

    if mode == "pure_quant_only":
        return generate_pure_quant_only(model, tokenizer, prompts, cfg, kv_manager)

    raise ValueError(f"Unknown mode: {mode}")


@torch.inference_mode()
def profile_generate_by_mode(
    model,
    tokenizer,
    prompts: list[str],
    cfg: HAWPLAQConfig,
    mode: str,
    coordinator=None,
    kv_manager=None,
    reset_fn: Callable | None = None,
) -> tuple[list[str], CacheStats, list[torch.Tensor]]:
    """Run generation and collect unified cache + peak-GPU statistics.

    All modes follow the same flow:
      1. reset cache
      2. ``torch.cuda.reset_peak_memory_stats()``
      3. stepwise generation (argmax), with PastKVTracker for baseline/hawp_only
      4. ``torch.cuda.max_memory_allocated()``
      5. collect ``CacheStats`` from the appropriate source

    Returns:
        (generated_texts, CacheStats, generated_ids_list)
        generated_ids_list: list of 1-D LongTensors (generated token ids only,
        excluding prompt ids) — used for exact token consistency without
        tokenizer round-trip.
    """
    if reset_fn is not None:
        reset_fn()
    elif coordinator is not None or kv_manager is not None:
        make_reset_fn(model, coordinator, kv_manager)()

    use_past_tracker = mode in ("baseline", "hawp_only")
    use_external_past = mode in ("baseline", "hawp_only")

    max_new_tokens = cfg.generation.max_new_tokens
    per_prompt_stats: list[CacheStats] = []
    results = []
    all_gen_ids = []

    for prompt in prompts:
        if reset_fn is not None:
            reset_fn()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        tracker = PastKVTracker() if use_past_tracker else None

        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        bsz, prompt_len = input_ids.shape
        total_seen_tokens = prompt_len + max_new_tokens

        prefill_mask = torch.ones(bsz, prompt_len, device=model.device, dtype=torch.long)
        prefill_pos = torch.arange(prompt_len, device=model.device, dtype=torch.long).unsqueeze(0)

        outputs = prefill_forward_last_logits(
            model,
            input_ids=input_ids,
            attention_mask=prefill_mask,
            position_ids=prefill_pos,
            use_cache=True,
        )

        if tracker is not None:
            tracker.update(outputs.past_key_values)

        if mode == "pure_quant_only" and kv_manager is not None:
            kv_manager.on_forward_done_from_output(outputs.past_key_values, prev_seq_len=0)

        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        generated_ids = next_token

        if coordinator is not None:
            coordinator.on_prefill(prompt_len)

        past_kv = outputs.past_key_values

        cur_pos = prompt_len
        for _ in range(max_new_tokens - 1):
            attention_mask = torch.ones(1, cur_pos + 1, device=model.device, dtype=torch.long)
            position_ids = torch.tensor([[cur_pos]], device=model.device, dtype=torch.long)

            fwd_kw: dict = {
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "use_cache": True,
            }

            if mode == "pure_quant_only" and kv_manager is not None:
                fwd_kw["past_key_values"] = kv_manager.get_past_kv()
            elif use_external_past and past_kv is not None:
                fwd_kw["past_key_values"] = past_kv

            outputs = model(input_ids=next_token, **fwd_kw)

            if tracker is not None:
                tracker.update(outputs.past_key_values)

            if mode == "pure_quant_only" and kv_manager is not None:
                kv_manager.on_forward_done_from_output(outputs.past_key_values, prev_seq_len=cur_pos)

            past_kv = outputs.past_key_values

            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            cur_pos += 1

            if coordinator is not None:
                coordinator.on_new_token()

        full_ids = torch.cat([input_ids, generated_ids], dim=1)
        text = tokenizer.decode(full_ids[0], skip_special_tokens=True)
        results.append(text)
        all_gen_ids.append(generated_ids[0].cpu())

        peak_gpu = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        if use_past_tracker:
            impl = "past_kv_baseline" if mode == "baseline" else "past_kv_hawp_only"
            prompt_stats = collect_cache_stats_from_tracker(tracker, peak_gpu, impl=impl)
        else:
            prompt_stats = collect_cache_stats(model, kv_manager, peak_gpu_bytes=peak_gpu)
        prompt_stats.baseline_kv_bytes = compute_baseline_kv_bytes(model, total_seen_tokens)
        per_prompt_stats.append(prompt_stats)

    stats = aggregate_cache_stats(per_prompt_stats)

    return results, stats, all_gen_ids
