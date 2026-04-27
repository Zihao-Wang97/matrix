from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from hawp_laq.config import HAWPLAQConfig, load_config, build_k_quantizer, build_v_quantizer, resolve_projector_ranks, load_projector_ranks_from_dir
from hawp_laq.modeling.attention_hawp import HAWPAttention
from hawp_laq.modeling.modeling_llama_hawp import convert_llama_to_hawp, _align_hawp_params_device_dtype
from hawp_laq.runtime.pure_quant_hook import PureQuantKVManager, install_pure_quant_hooks

_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def _has_real_past_key_values(past_kv) -> bool:
    if past_kv is None:
        return False
    if isinstance(past_kv, (tuple, list)):
        return any(kv is not None for kv in past_kv)
    key_cache = getattr(past_kv, "key_cache", None)
    if key_cache is not None:
        return any(k is not None and getattr(k, "numel", lambda: 0)() > 0 for k in key_cache)
    return True


def print_device_info(device: str) -> None:
    cuda_ok = torch.cuda.is_available()
    print(f"[device] cuda.is_available() = {cuda_ok}")
    print(f"[device] target device = {device}")
    if cuda_ok:
        idx = torch.cuda.current_device()
        print(f"[device] GPU = {torch.cuda.get_device_name(idx)}")
        allocated = torch.cuda.memory_allocated(idx)
        reserved = torch.cuda.memory_reserved(idx)
        print(f"[device] mem allocated = {_fmt_bytes(allocated)}")
        print(f"[device] mem reserved   = {_fmt_bytes(reserved)}")


def _fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def load_baseline_model(cfg: HAWPLAQConfig):
    model_id = cfg.model.model_id
    dtype = _DTYPE_MAP.get(cfg.model.torch_dtype, torch.float32)
    device = _resolve_device(cfg.train.device)

    print(f"[load] model_id = {model_id}")
    print(f"[load] dtype = {cfg.model.torch_dtype}")
    print(f"[load] load_in_4bit = {cfg.model.load_in_4bit}")
    print(f"[load] device = {device}")

    is_local = Path(model_id).expanduser().is_dir()
    if is_local:
        print(f"[load] detected local model directory: {model_id}")

    tok_kwargs: dict[str, Any] = {}
    model_kwargs: dict[str, Any] = {"torch_dtype": dtype, "device_map": device}

    if is_local:
        tok_kwargs["local_files_only"] = True
        model_kwargs["local_files_only"] = True

    if cfg.model.load_in_4bit:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
        )
        model_kwargs.pop("torch_dtype", None)
        model_kwargs.pop("device_map", None)

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, **tok_kwargs)
    except OSError as e:
        if "gated" in str(e).lower() or "access" in str(e).lower():
            raise OSError(
                f"Model '{model_id}' requires authentication.  "
                f"Run `huggingface-cli login` or use a local model path instead.\n"
                f"Original error: {e}"
            ) from e
        if is_local:
            raise OSError(
                f"Failed to load tokenizer from local path '{model_id}'.  "
                f"Ensure the directory contains tokenizer files (tokenizer.json, tokenizer_config.json, etc.).\n"
                f"Original error: {e}"
            ) from e
        raise OSError(
            f"Failed to load tokenizer for '{model_id}'.  "
            f"If the server has no internet, set model_id to a local directory path.\n"
            f"Original error: {e}"
        ) from e

    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    except OSError as e:
        if "gated" in str(e).lower() or "access" in str(e).lower():
            raise OSError(
                f"Model '{model_id}' requires authentication.  "
                f"Run `huggingface-cli login` or use a local model path instead.\n"
                f"Original error: {e}"
            ) from e
        if is_local:
            raise OSError(
                f"Failed to load model from local path '{model_id}'.  "
                f"Ensure the directory contains model files (config.json, model.safetensors, etc.).\n"
                f"Original error: {e}"
            ) from e
        raise OSError(
            f"Failed to load model '{model_id}'.  "
            f"If the server has no internet, download the model first and set model_id to a local directory path.\n"
            f"Original error: {e}"
        ) from e

    if not cfg.model.load_in_4bit:
        model = model.to(device)
    model.eval()
    return model, tokenizer, device


def _resolve_device(device: str) -> str:
    if device == "cuda" and not torch.cuda.is_available():
        print("[warn] cuda requested but not available, falling back to cpu")
        return "cpu"
    return device


@torch.inference_mode()
def stepwise_greedy_generate(
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int,
    coordinator=None,
    reset_cache_fn=None,
    full_recompute: bool = False,
    use_external_past: bool = True,
    return_ids: bool = False,
):
    """Unified stepwise greedy generation for fair correctness comparison.

    Every mode (baseline, hawp_only, quant_*) goes through the same
    prefill-then-decode loop with argmax selection, ensuring that token
    choices differ only due to the model's KV quantization, not due to
    different outer generation semantics.

    For modes that cannot use the HF KV cache correctly, set
    ``full_recompute=True`` so that each decode step reprocesses the
    full sequence.  This is slower but guarantees correctness.  For
    quant modes that manage KV internally, ``full_recompute=False``
    (default) relies on the internal cache.

    Args:
        model: The model (may be vanilla or HAWP-converted).
        tokenizer: The tokenizer.
        prompts: List of prompt strings.
        max_new_tokens: Number of new tokens to generate per prompt.
        coordinator: Optional ModelCacheCoordinator for sched mode.
        reset_cache_fn: Optional callable to reset quant cache between prompts.
        full_recompute: If True, pass the full input_ids each decode step
            (for models without internal KV cache).
        use_external_past: If True, feed ``outputs.past_key_values`` back into
            the next decode step. Quant-cache modes should set this to False
            because history is managed inside HAWPAttention.
        return_ids: If True, also return a list of 1-D LongTensors with
            generated token ids (excluding prompt ids).

    Returns:
        ``list[str]`` if *return_ids* is False; otherwise
        ``(list[str], list[Tensor])`` where each Tensor holds the generated
        ids for one prompt.
    """
    results = []
    all_gen_ids = []

    for prompt in prompts:
        if reset_cache_fn is not None:
            reset_cache_fn()
        elif coordinator is not None:
            for module in model.modules():
                if isinstance(module, HAWPAttention) and module.use_cache_manager:
                    module.reset_quant_cache()
            coordinator.reset()

        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        bsz, prompt_len = input_ids.shape

        prefill_mask = torch.ones(
            bsz, prompt_len, device=model.device, dtype=torch.long,
        )
        prefill_pos = torch.arange(
            prompt_len, device=model.device, dtype=torch.long,
        ).unsqueeze(0)

        outputs = model(
            input_ids=input_ids,
            attention_mask=prefill_mask,
            position_ids=prefill_pos,
            use_cache=True,
        )
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        generated_ids = next_token

        if coordinator is not None:
            coordinator.on_prefill(prompt_len)

        past_kv = None if full_recompute or not use_external_past else outputs.past_key_values

        cur_pos = prompt_len
        for _ in range(max_new_tokens - 1):
            if full_recompute:
                all_ids = torch.cat([input_ids, generated_ids], dim=1)
                seq_len = all_ids.shape[1]
                attention_mask = torch.ones(
                    1, seq_len, device=model.device, dtype=torch.long,
                )
                position_ids = torch.arange(
                    seq_len, device=model.device, dtype=torch.long,
                ).unsqueeze(0)
                outputs = model(
                    input_ids=all_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=False,
                )
            else:
                attention_mask = torch.ones(
                    1, cur_pos + 1, device=model.device, dtype=torch.long,
                )
                position_ids = torch.tensor(
                    [[cur_pos]], device=model.device, dtype=torch.long,
                )
                fwd_kw: dict = {
                    "attention_mask": attention_mask,
                    "position_ids": position_ids,
                    "use_cache": True,
                }
                if use_external_past and _has_real_past_key_values(past_kv):
                    fwd_kw["past_key_values"] = past_kv
                outputs = model(input_ids=next_token, **fwd_kw)
                past_kv = outputs.past_key_values if use_external_past else None

            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            cur_pos += 1

            if coordinator is not None:
                coordinator.on_new_token()

        full_ids = torch.cat([input_ids, generated_ids], dim=1)
        text = tokenizer.decode(full_ids[0], skip_special_tokens=True)
        results.append(text)
        if return_ids:
            all_gen_ids.append(generated_ids[0].cpu())

    if return_ids:
        return results, all_gen_ids
    return results


@torch.inference_mode()
def generate_hawp_quant(
    model,
    tokenizer,
    prompts: list[str],
    cfg: HAWPLAQConfig,
    coordinator=None,
) -> list[str]:
    max_new_tokens = cfg.generation.max_new_tokens
    results = []

    for prompt in prompts:
        for module in model.modules():
            if isinstance(module, HAWPAttention) and module.use_cache_manager:
                module.reset_quant_cache()

        if coordinator is not None:
            coordinator.reset()

        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        bsz, prompt_len = input_ids.shape

        prefill_mask = torch.ones(
            bsz, prompt_len, device=model.device, dtype=torch.long,
        )
        prefill_pos = torch.arange(
            prompt_len, device=model.device, dtype=torch.long,
        ).unsqueeze(0)

        outputs = model(
            input_ids=input_ids,
            attention_mask=prefill_mask,
            position_ids=prefill_pos,
            use_cache=True,
        )
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        generated_ids = next_token

        if coordinator is not None:
            coordinator.on_prefill(prompt_len)

        cur_pos = prompt_len
        for _ in range(max_new_tokens - 1):
            attention_mask = torch.ones(
                1, cur_pos + 1, device=model.device, dtype=torch.long,
            )
            position_ids = torch.tensor([[cur_pos]], device=model.device, dtype=torch.long)

            outputs = model(
                input_ids=next_token,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=True,
            )
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            cur_pos += 1

            if coordinator is not None:
                coordinator.on_new_token()

        full_ids = torch.cat([input_ids, generated_ids], dim=1)
        text = tokenizer.decode(full_ids[0], skip_special_tokens=True)
        results.append(text)

    return results


@torch.inference_mode()
def generate_pure_quant_only(
    model,
    tokenizer,
    prompts: list[str],
    cfg: HAWPLAQConfig,
    kv_manager: PureQuantKVManager,
    return_ids: bool = False,
):
    """Stepwise greedy generation for pure_quant_only mode.

    Uses original HF attention with quantized KV cache managed by
    PureQuantKVManager.  After each forward, captured K/V are stored
    in the quant cache.  On decode steps, dequantized K/V are fed
    back as past_key_value.

    Args:
        return_ids: If True, also return a list of 1-D LongTensors with
            generated token ids (excluding prompt ids).

    Returns:
        ``list[str]`` if *return_ids* is False; otherwise
        ``(list[str], list[Tensor])``.
    """
    max_new_tokens = cfg.generation.max_new_tokens
    results = []
    all_gen_ids = []

    for prompt in prompts:
        kv_manager.reset_caches()

        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        bsz, prompt_len = input_ids.shape

        prefill_mask = torch.ones(
            bsz, prompt_len, device=model.device, dtype=torch.long,
        )
        prefill_pos = torch.arange(
            prompt_len, device=model.device, dtype=torch.long,
        ).unsqueeze(0)

        outputs = model(
            input_ids=input_ids,
            attention_mask=prefill_mask,
            position_ids=prefill_pos,
            use_cache=True,
        )
        kv_manager.on_forward_done_from_output(outputs.past_key_values, prev_seq_len=0)
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        generated_ids = next_token

        cur_pos = prompt_len
        for _ in range(max_new_tokens - 1):
            attention_mask = torch.ones(
                1, cur_pos + 1, device=model.device, dtype=torch.long,
            )
            position_ids = torch.tensor([[cur_pos]], device=model.device, dtype=torch.long)

            past_kv = kv_manager.get_past_kv()

            outputs = model(
                input_ids=next_token,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_kv,
                use_cache=True,
            )
            kv_manager.on_forward_done_from_output(outputs.past_key_values, prev_seq_len=cur_pos)
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            cur_pos += 1

        full_ids = torch.cat([input_ids, generated_ids], dim=1)
        text = tokenizer.decode(full_ids[0], skip_special_tokens=True)
        results.append(text)
        if return_ids:
            all_gen_ids.append(generated_ids[0].cpu())

    if return_ids:
        return results, all_gen_ids
    return results


def _print_results(prompts: list[str], outputs: list[str]) -> None:
    import sys
    for i, (p, o) in enumerate(zip(prompts, outputs)):
        print(f"\n--- prompt {i} ---")
        print(f"IN:  {p}")
        safe = o.encode(sys.stdout.encoding or 'utf-8', errors='replace').decode(sys.stdout.encoding or 'utf-8', errors='replace')
        print(f"OUT: {safe}")


def _resolve_head_dim_from_model_or_attn(model) -> int:
    for _, mod in model.named_modules():
        if isinstance(mod, HAWPAttention):
            return mod.head_dim
    from hawp_laq.modeling.attention_hawp import _get_attn_config
    for _, mod in model.named_modules():
        for attr in ("self_attn", "attention"):
            if hasattr(mod, attr):
                attn_config = _get_attn_config(getattr(mod, attr), model=model)
                return attn_config.hidden_size // attn_config.num_attention_heads
    raise ValueError("No attention module found in model")


def _resolve_hawp_ranks(cfg: HAWPLAQConfig, model, mode: str) -> tuple[int, int, dict[int, tuple[int, int]] | None]:
    head_dim = _resolve_head_dim_from_model_or_attn(model)
    r_k, r_v = resolve_projector_ranks(cfg.projector, head_dim=head_dim, mode=mode)
    ranks_per_layer = load_projector_ranks_from_dir(cfg.projector.output_dir)
    if ranks_per_layer:
        print(f"[{mode}] loaded per-layer ranks from {cfg.projector.output_dir / 'ranks.json'}")
        for idx, (rk, rv) in sorted(ranks_per_layer.items()):
            print(f"  layer {idx}: r_k={rk}  r_v={rv}")
    return r_k, r_v, ranks_per_layer or None


def _setup_quant_cache_per_layer(model, cfg: HAWPLAQConfig, recent_window: int) -> None:
    for module in model.modules():
        if isinstance(module, HAWPAttention):
            k_q = build_k_quantizer(cfg, r_k=module.r_k)
            v_q = build_v_quantizer(cfg, r_v=module.r_v)
            module.setup_quant_cache(k_q, v_q, recent_window=recent_window)


def _count_model_layers(model) -> int:
    n = 0
    for _, mod in model.named_modules():
        cls_name = type(mod).__name__
        if "DecoderLayer" in cls_name:
            n += 1
    return n


def _read_ranks_from_projector_files(
    projector_dir: Path, available_layers: set[int], head_dim: int
) -> dict[int, tuple[int, int]]:
    """Read r_k/r_v from each available projector.pt file.

    Skips layers where r_k or r_v exceeds head_dim (incompatible artifact).
    """
    ranks_from_files: dict[int, tuple[int, int]] = {}
    for layer_idx in available_layers:
        pt_path = projector_dir / f"layer_{layer_idx}" / "projector.pt"
        if pt_path.exists():
            data = torch.load(pt_path, map_location="cpu", weights_only=False)
            if "r_k" in data and "r_v" in data:
                rk, rv = data["r_k"], data["r_v"]
                if rk <= head_dim and rv <= head_dim:
                    ranks_from_files[layer_idx] = (rk, rv)
    return ranks_from_files


def _convert_and_load_projectors(model, cfg, device, mode: str):
    from hawp_laq.runtime.projector_bank import get_available_projector_layers

    r_k, r_v, ranks_per_layer = _resolve_hawp_ranks(cfg, model, mode)
    head_dim = _resolve_head_dim_from_model_or_attn(model)
    allow_default = (mode == "quant_only")
    projector_dir = cfg.projector.output_dir

    effective_ranks_per_layer: dict[int, tuple[int, int]] | None = None
    available_layers: set[int] = set()

    if mode != "quant_only" and Path(projector_dir).exists():
        available_layers = get_available_projector_layers(projector_dir)

    if available_layers and mode != "quant_only":
        n_layers = _count_model_layers(model)
        n_layers = max(n_layers, max(available_layers) + 1)

        ranks_from_files = _read_ranks_from_projector_files(
            Path(projector_dir), available_layers, head_dim
        )

        compatible_layers = set(ranks_from_files.keys()) | (
            set(ranks_per_layer.keys()) if ranks_per_layer else set()
        )
        compatible_layers &= available_layers

        effective_ranks_per_layer = {}
        for layer_idx in range(n_layers):
            if layer_idx in compatible_layers:
                if ranks_per_layer and layer_idx in ranks_per_layer:
                    effective_ranks_per_layer[layer_idx] = ranks_per_layer[layer_idx]
                elif layer_idx in ranks_from_files:
                    effective_ranks_per_layer[layer_idx] = ranks_from_files[layer_idx]
                else:
                    effective_ranks_per_layer[layer_idx] = (r_k, r_v)
            else:
                effective_ranks_per_layer[layer_idx] = (head_dim, head_dim)

        low_rank_count = sum(
            1 for rk, rv in effective_ranks_per_layer.values()
            if rk < head_dim or rv < head_dim
        )
        print(f"[{mode}] configured default r_k={r_k}  r_v={r_v}")
        print(f"[{mode}] projector layers found: {sorted(available_layers)}")
        if low_rank_count < len(effective_ranks_per_layer):
            print(f"[{mode}] layers without projector fallback to full-rank")
        print(f"[{mode}] effective low-rank layers: {low_rank_count} / {len(effective_ranks_per_layer)}")

    model = convert_llama_to_hawp(
        model, r_k=r_k, r_v=r_v,
        ranks_per_layer=effective_ranks_per_layer,
        allow_default_full_rank=allow_default,
        logit_scale_mode=cfg.hawp.logit_scale_mode,
        gamma_mode=cfg.hawp.gamma_mode,
        gamma_value=cfg.hawp.gamma_value,
        use_archive_k_ip_approx=cfg.hawp.use_archive_k_ip_approx,
    )
    if cfg.model.load_in_4bit:
        _align_hawp_params_device_dtype(model)
    else:
        model = model.to(device)
    model.eval()

    if mode != "quant_only":
        from hawp_laq.runtime.projector_bank import load_projectors, inspect_projector_dir
        if Path(projector_dir).exists():
            if not available_layers:
                raise RuntimeError(
                    f"[{mode}] projector_dir exists but contains no projector.pt files: "
                    f"{projector_dir}. Mode '{mode}' requires trained projectors. "
                    f"Run `python scripts/02_train_projectors.py` to train projectors first."
                )

            ranks_json_path = Path(projector_dir) / "ranks.json"
            has_pt_files = bool(available_layers)
            if has_pt_files and not ranks_json_path.exists():
                print(
                    f"[{mode}] WARNING: projector_dir contains projector.pt files "
                    f"but no ranks.json; this often means partial single_group "
                    f"training or stale artifacts"
                )

            report = inspect_projector_dir(
                projector_dir,
                expected_head_dim=head_dim,
                default_r_k=r_k,
                default_r_v=r_v,
                ranks_per_layer=ranks_per_layer,
            )

            if report["missing_rank_layers"]:
                warnings.warn(
                    f"[{mode}] Layers with missing r_k/r_v in projector.pt: "
                    f"{report['missing_rank_layers']}. These layers will be skipped.",
                    UserWarning,
                    stacklevel=2,
                )

            problem_layers = report["legacy_layers"] + report["shape_mismatch_layers"]
            if report["legacy_layers"] or report["shape_mismatch_layers"]:
                legacy_detail = (
                    f"legacy (missing r_k/r_v): {report['legacy_layers']}"
                    if report["legacy_layers"]
                    else ""
                )
                mismatch_detail = (
                    f"shape mismatch: {report['shape_mismatch_layers']}"
                    if report["shape_mismatch_layers"]
                    else ""
                )
                parts = [p for p in [legacy_detail, mismatch_detail] if p]
                raise ValueError(
                    f"Incompatible projector files found in {projector_dir}. "
                    f"Problem layers: {'; '.join(parts)}. "
                    f"Expected head_dim={head_dim}, default r_k={r_k}, r_v={r_v}. "
                    f"Suggestions: (1) clear the projector output directory and retrain, "
                    f"(2) use a new empty output_dir, or (3) retrain all layers."
                )

            load_projectors(model, projector_dir, strict=True)
            n_hawp = sum(1 for m in model.modules() if isinstance(m, HAWPAttention))
            n_loaded = len(available_layers)
            print(f"[{mode}] loaded projectors from {projector_dir} ({n_loaded}/{n_hawp} layers)")
        else:
            raise RuntimeError(
                f"[{mode}] projector_dir not found: {projector_dir}. "
                f"Mode '{mode}' requires trained projectors. "
                f"Run `python scripts/02_train_projectors.py` to train projectors first."
            )

    if effective_ranks_per_layer:
        rk_vals = {rk for rk, _ in effective_ranks_per_layer.values()}
        rv_vals = {rv for _, rv in effective_ranks_per_layer.values()}
        lr_count = sum(1 for rk, rv in effective_ranks_per_layer.values() if rk < head_dim or rv < head_dim)
        print(f"[{mode}] effective ranks: r_k∈{sorted(rk_vals)}  r_v∈{sorted(rv_vals)}  "
              f"({lr_count} low-rank / {len(effective_ranks_per_layer)} total)")
    elif ranks_per_layer:
        print(f"[{mode}] r_k={r_k}  r_v={r_v}  per-layer ranks: {len(ranks_per_layer)} layers")
    else:
        print(f"[{mode}] r_k={r_k}  r_v={r_v}")

    return model, r_k, r_v


def _setup_hawp_quant_on_model(model, cfg, device):
    model, r_k, r_v = _convert_and_load_projectors(model, cfg, device, "hawp_quant")
    _setup_quant_cache_per_layer(model, cfg, recent_window=cfg.sched.recent_window)
    print(f"[hawp_quant] recent_window={cfg.sched.recent_window}")
    return model


def _setup_hawp_quant_all_on_model(model, cfg, device):
    model, r_k, r_v = _convert_and_load_projectors(model, cfg, device, "hawp_quant_all")
    _setup_quant_cache_per_layer(model, cfg, recent_window=0)
    print(f"[hawp_quant_all] recent_window=0 (all tokens quantized)")
    return model


def _setup_quant_only_on_model(model, cfg, device):
    head_dim = _resolve_head_dim_from_model_or_attn(model)
    model = convert_llama_to_hawp(
        model, r_k=head_dim, r_v=head_dim, allow_default_full_rank=True,
        logit_scale_mode=cfg.hawp.logit_scale_mode,
        gamma_mode=cfg.hawp.gamma_mode,
        gamma_value=cfg.hawp.gamma_value,
        use_archive_k_ip_approx=cfg.hawp.use_archive_k_ip_approx,
    )
    if cfg.model.load_in_4bit:
        _align_hawp_params_device_dtype(model)
    else:
        model = model.to(device)
    model.eval()

    _setup_quant_cache_per_layer(model, cfg, recent_window=cfg.sched.recent_window)

    print(f"[quant_only] head_dim={head_dim}  recent_window={cfg.sched.recent_window}  (explicit full-rank, no low-rank projection)")

    return model, head_dim


def _setup_pure_quant_only_on_model(model, cfg, device):
    """Set up pure_quant_only: original HF attention + quantized KV cache.

    Does NOT call convert_llama_to_hawp.  Does NOT create HAWPAttention.
    Installs forward hooks on k_proj/v_proj to capture K/V, then stores
    them in a quantized cache.  On decode steps, dequantized K/V are
    fed back as past_key_value so the original attention formula is
    preserved.
    """
    if not cfg.model.load_in_4bit:
        model = model.to(device)
    model.eval()

    manager = install_pure_quant_hooks(model, cfg)
    head_dim = manager.head_dim

    print(f"[pure_quant_only] head_dim={head_dim}  recent_window={cfg.sched.recent_window}  "
          f"(original HF attention, no HAWP conversion, quantized KV cache)")

    return model, head_dim, manager


def run_baseline(config_path: str | Path) -> None:
    cfg = load_config(config_path)
    print("=" * 60)
    print(f"[mode] baseline")
    print_device_info(cfg.train.device)
    print("=" * 60)

    model, tokenizer, device = load_baseline_model(cfg)
    prompts = cfg.generation.prompts
    print(f"[baseline] running {len(prompts)} prompt(s) ...")

    outputs = stepwise_greedy_generate(model, tokenizer, prompts, cfg.generation.max_new_tokens)
    _print_results(prompts, outputs)


@torch.inference_mode()
def run_hawp_only(config_path: str | Path) -> None:
    cfg = load_config(config_path)
    print("=" * 60)
    print("[mode] hawp_only")
    print_device_info(cfg.train.device)
    print("=" * 60)

    model, tokenizer, device = load_baseline_model(cfg)
    model, r_k, r_v = _convert_and_load_projectors(model, cfg, device, "hawp_only")

    print(f"[hawp_only] r_k={r_k}  r_v={r_v}")
    prompts = cfg.generation.prompts
    print(f"[hawp_only] running {len(prompts)} prompt(s) ...")

    outputs = stepwise_greedy_generate(model, tokenizer, prompts, cfg.generation.max_new_tokens)
    _print_results(prompts, outputs)


@torch.inference_mode()
def run_hawp_quant(config_path: str | Path) -> None:
    cfg = load_config(config_path)
    print("=" * 60)
    print("[mode] hawp_quant")
    print_device_info(cfg.train.device)
    print("=" * 60)

    model, tokenizer, device = load_baseline_model(cfg)
    model = _setup_hawp_quant_on_model(model, cfg, device)

    r_k, r_v = _resolve_hawp_ranks(cfg, model, "hawp_quant")[:2]
    recent_window = cfg.sched.recent_window
    print(f"[hawp_quant] r_k={r_k}  r_v={r_v}  recent_window={recent_window}")
    prompts = cfg.generation.prompts
    print(f"[hawp_quant] running {len(prompts)} prompt(s) ...")

    outputs = generate_hawp_quant(model, tokenizer, prompts, cfg)
    _print_results(prompts, outputs)

    print(f"\n[hawp_quant] --- cache summary ---")
    for module in model.modules():
        if isinstance(module, HAWPAttention) and module.use_cache_manager:
            s = module.quant_cache_summary()
            print(f"  layer {s['layer']:>2d}: recent={s['recent_tokens']}  archive={s['archive_tokens']}  "
                  f"runtime={_fmt_bytes(s['total_runtime_bytes'])}  compressed={_fmt_bytes(s['compressed_storage_bytes'])}")


@torch.inference_mode()
def run_quant_only(config_path: str | Path) -> None:
    cfg = load_config(config_path)
    print("=" * 60)
    print("[mode] quant_only")
    print_device_info(cfg.train.device)
    print("=" * 60)

    model, tokenizer, device = load_baseline_model(cfg)
    model, head_dim = _setup_quant_only_on_model(model, cfg, device)

    recent_window = cfg.sched.recent_window
    print(f"[quant_only] head_dim={head_dim}  recent_window={recent_window}  (no low-rank projection)")
    prompts = cfg.generation.prompts
    print(f"[quant_only] running {len(prompts)} prompt(s) ...")

    outputs = generate_hawp_quant(model, tokenizer, prompts, cfg)
    _print_results(prompts, outputs)

    print(f"\n[quant_only] --- cache summary ---")
    for module in model.modules():
        if isinstance(module, HAWPAttention) and module.use_cache_manager:
            s = module.quant_cache_summary()
            print(f"  layer {s['layer']:>2d}: recent={s['recent_tokens']}  archive={s['archive_tokens']}  "
                  f"runtime={_fmt_bytes(s['total_runtime_bytes'])}  compressed={_fmt_bytes(s['compressed_storage_bytes'])}")


@torch.inference_mode()
def run_pure_quant_only(config_path: str | Path) -> None:
    cfg = load_config(config_path)
    print("=" * 60)
    print("[mode] pure_quant_only")
    print_device_info(cfg.train.device)
    print("=" * 60)

    model, tokenizer, device = load_baseline_model(cfg)
    model, head_dim, kv_manager = _setup_pure_quant_only_on_model(model, cfg, device)

    recent_window = cfg.sched.recent_window
    print(f"[pure_quant_only] head_dim={head_dim}  recent_window={recent_window}  (original attention + quant KV cache)")
    prompts = cfg.generation.prompts
    print(f"[pure_quant_only] running {len(prompts)} prompt(s) ...")

    outputs = generate_pure_quant_only(model, tokenizer, prompts, cfg, kv_manager)
    _print_results(prompts, outputs)

    print(f"\n[pure_quant_only] --- cache summary ---")
    for s in kv_manager.cache_summaries():
        print(f"  layer {s['layer']:>2d}: recent={s['recent_tokens']}  archive={s['archive_tokens']}  "
              f"runtime={_fmt_bytes(s['total_runtime_bytes'])}  compressed={_fmt_bytes(s['compressed_storage_bytes'])}")


@torch.inference_mode()
def run_hawp_quant_all(config_path: str | Path) -> None:
    cfg = load_config(config_path)
    print("=" * 60)
    print("[mode] hawp_quant_all")
    print_device_info(cfg.train.device)
    print("=" * 60)

    model, tokenizer, device = load_baseline_model(cfg)
    model = _setup_hawp_quant_all_on_model(model, cfg, device)

    r_k, r_v = _resolve_hawp_ranks(cfg, model, "hawp_quant_all")[:2]
    print(f"[hawp_quant_all] r_k={r_k}  r_v={r_v}  recent_window=0 (all tokens quantized)")
    prompts = cfg.generation.prompts
    print(f"[hawp_quant_all] running {len(prompts)} prompt(s) ...")

    outputs = generate_hawp_quant(model, tokenizer, prompts, cfg)
    _print_results(prompts, outputs)

    from hawp_laq.eval.metrics import collect_kv_metrics, format_kv_metrics
    metrics = collect_kv_metrics(model)
    print(f"\n[hawp_quant_all] --- cache summary ---")
    print(format_kv_metrics(metrics))


@torch.inference_mode()
def run_hawp_quant_sched(config_path: str | Path) -> None:
    from hawp_laq.runtime.scheduler import TokenBudgetScheduler
    from hawp_laq.runtime.cache_manager import ModelCacheCoordinator

    cfg = load_config(config_path)
    print("=" * 60)
    print("[mode] hawp_quant_sched")
    print_device_info(cfg.train.device)
    print("=" * 60)

    model, tokenizer, device = load_baseline_model(cfg)
    model = _setup_hawp_quant_on_model(model, cfg, device)

    r_k, r_v = _resolve_hawp_ranks(cfg, model, "hawp_quant_sched")[:2]
    total_budget = cfg.sched.total_budget
    recent_window = cfg.sched.recent_window
    drop_strategy = getattr(cfg.sched, "drop_strategy", "position")

    scheduler = TokenBudgetScheduler(
        total_budget=total_budget,
        recent_window=recent_window,
        high_ratio=cfg.sched.high_ratio,
        low_ratio=cfg.sched.low_ratio,
        drop_strategy=drop_strategy,
    )
    coordinator = ModelCacheCoordinator.from_model(
        model, scheduler, drop_strategy=drop_strategy,
    )

    print(f"[hawp_quant_sched] r_k={r_k}  r_v={r_v}  "
          f"budget={total_budget}  recent_window={recent_window}  "
          f"drop_strategy={drop_strategy}")
    prompts = cfg.generation.prompts
    print(f"[hawp_quant_sched] running {len(prompts)} prompt(s) ...")

    outputs = generate_hawp_quant(model, tokenizer, prompts, cfg, coordinator=coordinator)
    _print_results(prompts, outputs)

    decision = scheduler.rebalance()
    print(f"\n[hawp_quant_sched] --- scheduler decision ---")
    print(f"  HIGH={decision.n_high}  LOW={decision.n_low}  DROP={decision.n_drop}")

    print(f"\n[hawp_quant_sched] --- cache summary ---")
    for module in model.modules():
        if isinstance(module, HAWPAttention) and module.use_cache_manager:
            s = module.quant_cache_summary()
            print(f"  layer {s['layer']:>2d}: recent={s['recent_tokens']}  archive={s['archive_tokens']}  "
                  f"dropped={decision.n_drop}  runtime={_fmt_bytes(s['total_runtime_bytes'])}  compressed={_fmt_bytes(s['compressed_storage_bytes'])}")
