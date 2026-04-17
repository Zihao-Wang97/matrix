from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from hawp_laq.modeling.attention_hawp import HAWPAttention


def _find_layers_and_attn(model: nn.Module) -> list[tuple[str, nn.Module, nn.Module]]:
    results = []
    for name, module in model.named_modules():
        cls_name = type(module).__name__
        if cls_name in ("LlamaDecoderLayer", "OPTDecoderLayer", "MistralDecoderLayer", "Qwen2DecoderLayer"):
            attn_name = None
            for attr in ("self_attn", "attention"):
                if hasattr(module, attr):
                    attn_name = attr
                    break
            if attn_name is not None:
                results.append((name, module, getattr(module, attn_name)))
    return results


def convert_llama_to_hawp(
    model: nn.Module,
    r_k: int | None = None,
    r_v: int | None = None,
) -> nn.Module:
    layers_and_attn = _find_layers_and_attn(model)
    if not layers_and_attn:
        raise ValueError("No compatible decoder layers found. Supported: Llama, OPT, Mistral, Qwen2.")

    for layer_name, layer_mod, orig_attn in layers_and_attn:
        hawp_attn = HAWPAttention.from_attention(orig_attn, model=model, layer_idx=None, r_k=r_k, r_v=r_v)
        attn_attr = "self_attn" if hasattr(layer_mod, "self_attn") else "attention"
        setattr(layer_mod, attn_attr, hawp_attn)

    model.config._hawp_converted = True
    model.config._hawp_r_k = r_k
    model.config._hawp_r_v = r_v
    return model


def load_hawp_model(
    model_id: str,
    r_k: int | None = None,
    r_v: int | None = None,
    torch_dtype: torch.dtype = torch.float32,
    device: str = "cpu",
):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype, device_map=device)
    model = convert_llama_to_hawp(model, r_k=r_k, r_v=r_v)
    if device == "cpu" or (device == "cuda" and not torch.cuda.is_available()):
        model = model.to(device)
    model.eval()
    return model, tokenizer
