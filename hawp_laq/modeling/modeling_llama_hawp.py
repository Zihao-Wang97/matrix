from __future__ import annotations

import inspect
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from hawp_laq.modeling.attention_hawp import HAWPAttention, _resolve_compute_dtype


def _find_layers_and_attn(model: nn.Module) -> list[tuple[str, nn.Module, nn.Module]]:
    results = []
    for name, module in model.named_modules():
        cls_name = type(module).__name__
        if cls_name in ("LlamaDecoderLayer", "OPTDecoderLayer", "MistralDecoderLayer", "Qwen2DecoderLayer", "GemmaDecoderLayer", "Gemma2DecoderLayer", "Phi3DecoderLayer"):
            attn_name = None
            for attr in ("self_attn", "attention"):
                if hasattr(module, attr):
                    attn_name = attr
                    break
            if attn_name is not None:
                results.append((name, module, getattr(module, attn_name)))
    return results


def _align_hawp_params_device_dtype(model: nn.Module, compute_dtype: torch.dtype | None = None) -> None:
    for module in model.modules():
        if not isinstance(module, HAWPAttention):
            continue
        ref_param = getattr(module, "q_proj", None)
        if ref_param is None:
            continue
        ref_weight = ref_param.weight
        target_device = ref_weight.device
        if compute_dtype is not None:
            target_dtype = compute_dtype
        else:
            target_dtype = _resolve_compute_dtype(ref_param)
        for param_name in ("p_k", "p_v", "gamma"):
            p = getattr(module, param_name, None)
            if p is not None:
                p.data = p.data.to(device=target_device, dtype=target_dtype)


def _verify_hawp_params_device_consistency(model: nn.Module) -> None:
    for module in model.modules():
        if not isinstance(module, HAWPAttention):
            continue
        ref_param = getattr(module, "q_proj", None)
        if ref_param is None:
            continue
        expected_device = ref_param.weight.device
        for param_name in ("p_k", "p_v", "gamma"):
            p = getattr(module, param_name, None)
            if p is not None and p.device != expected_device:
                raise RuntimeError(
                    f"HAWPAttention layer {module.layer_idx}: {param_name} on {p.device} "
                    f"but q_proj on {expected_device}. Device alignment failed."
                )


def _install_opt_use_cache_bridge(layer_mod: nn.Module, hawp_attn: HAWPAttention) -> None:
    if getattr(layer_mod, "_hawp_use_cache_bridge_installed", False):
        return

    def _sync_use_cache(module, args, kwargs):
        use_cache = False
        valid = False

        try:
            forward_sig = inspect.signature(module.forward)
        except (TypeError, ValueError):
            forward_sig = None

        if forward_sig is not None:
            try:
                bound = forward_sig.bind_partial(*args, **kwargs)
                bound.apply_defaults()
                if "use_cache" in bound.arguments:
                    use_cache = bound.arguments["use_cache"]
                    valid = True
            except TypeError:
                pass

        if not valid and "use_cache" in kwargs:
            use_cache = kwargs["use_cache"]
            valid = True

        hawp_attn._hawp_parent_use_cache = bool(use_cache) if valid else False
        hawp_attn._hawp_parent_use_cache_valid = valid

    layer_mod.register_forward_pre_hook(_sync_use_cache, with_kwargs=True)
    layer_mod._hawp_use_cache_bridge_installed = True


def convert_llama_to_hawp(
    model: nn.Module,
    r_k: int | None = None,
    r_v: int | None = None,
    ranks_per_layer: dict[int, tuple[int, int]] | None = None,
    allow_default_full_rank: bool = False,
    logit_scale_mode: str = "rk",
    gamma_mode: str = "learned",
    gamma_value: float | None = None,
    use_archive_k_ip_approx: bool = True,
) -> nn.Module:
    """Replace decoder-layer attention modules with HAWPAttention.

    Config metadata written to ``model.config``:

    * ``_hawp_r_k`` / ``_hawp_r_v`` — **global default** values passed to
      this function.  When ``ranks_per_layer`` is used these may differ from
      the actual per-layer ranks stored on each ``HAWPAttention`` module.
    * ``_hawp_ranks_per_layer`` — per-layer override dict (or ``None``).
    * ``_hawp_global_default_r_k`` / ``_hawp_global_default_r_v`` — same
      values as ``_hawp_r_k``/``_hawp_r_v``, named more explicitly.
    * ``_hawp_uses_per_layer_ranks`` — ``True`` when ``ranks_per_layer``
      is not ``None``.
    """
    layers_and_attn = _find_layers_and_attn(model)
    if not layers_and_attn:
        raise ValueError("No compatible decoder layers found. Supported: Llama, OPT, Mistral, Qwen2.")

    for layer_idx, (layer_name, layer_mod, orig_attn) in enumerate(layers_and_attn):
        if ranks_per_layer is not None and layer_idx in ranks_per_layer:
            layer_r_k, layer_r_v = ranks_per_layer[layer_idx]
        else:
            layer_r_k, layer_r_v = r_k, r_v
        hawp_attn = HAWPAttention.from_attention(
            orig_attn, model=model, layer_idx=layer_idx, r_k=layer_r_k, r_v=layer_r_v,
            allow_default_full_rank=allow_default_full_rank,
            logit_scale_mode=logit_scale_mode,
            gamma_mode=gamma_mode,
            gamma_value=gamma_value,
            use_archive_k_ip_approx=use_archive_k_ip_approx,
        )
        attn_attr = "self_attn" if hasattr(layer_mod, "self_attn") else "attention"
        setattr(layer_mod, attn_attr, hawp_attn)
        if type(layer_mod).__name__ == "OPTDecoderLayer":
            _install_opt_use_cache_bridge(layer_mod, hawp_attn)

    model.config._hawp_converted = True
    model.config._hawp_r_k = r_k
    model.config._hawp_r_v = r_v
    model.config._hawp_ranks_per_layer = ranks_per_layer
    model.config._hawp_global_default_r_k = r_k
    model.config._hawp_global_default_r_v = r_v
    model.config._hawp_uses_per_layer_ranks = ranks_per_layer is not None
    return model


def load_hawp_model(
    model_id: str,
    r_k: int | None = None,
    r_v: int | None = None,
    torch_dtype: torch.dtype = torch.float32,
    device: str = "cpu",
    load_in_4bit: bool = False,
    allow_default_full_rank: bool = False,
    logit_scale_mode: str = "rk",
    gamma_mode: str = "learned",
    gamma_value: float | None = None,
    use_archive_k_ip_approx: bool = True,
):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    is_local = Path(model_id).expanduser().is_dir()
    tok_kw = {"local_files_only": True} if is_local else {}
    mdl_kw: dict = {"torch_dtype": torch_dtype, "device_map": device}

    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        mdl_kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
        )
        mdl_kw.pop("torch_dtype", None)
        if isinstance(device, str) and device not in ("auto", "balanced", "balanced_low_0", "sequential"):
            mdl_kw["device_map"] = {"": device}
        else:
            mdl_kw["device_map"] = device

    if is_local:
        mdl_kw["local_files_only"] = True

    tokenizer = AutoTokenizer.from_pretrained(model_id, **tok_kw)
    model = AutoModelForCausalLM.from_pretrained(model_id, **mdl_kw)
    model = convert_llama_to_hawp(model, r_k=r_k, r_v=r_v, allow_default_full_rank=allow_default_full_rank,
                                   logit_scale_mode=logit_scale_mode, gamma_mode=gamma_mode, gamma_value=gamma_value,
                                   use_archive_k_ip_approx=use_archive_k_ip_approx)
    if load_in_4bit:
        _align_hawp_params_device_dtype(model, compute_dtype=torch_dtype)
        _verify_hawp_params_device_consistency(model)
    else:
        model = model.to(device)
    model.eval()
    return model, tokenizer
