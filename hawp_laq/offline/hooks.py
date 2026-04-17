from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn


HookHandle = torch.utils.hooks.RemovableHandle


def _find_attention_modules(model: nn.Module) -> list[tuple[int, nn.Module]]:
    attn_cls_names = {
        "Attention",
        "SdpaAttention",
        "OPTAttention",
        "LlamaAttention",
        "MistralAttention",
        "GPTNeoXAttention",
        "BloomAttention",
        "Phi3Attention",
        "Qwen2Attention",
    }
    results: list[tuple[int, nn.Module]] = []
    idx = 0
    for name, module in model.named_modules():
        cls_name = type(module).__name__
        if cls_name in attn_cls_names:
            results.append((idx, module))
            idx += 1
    return results


def _get_proj(attn: nn.Module, names: list[str]) -> nn.Module | None:
    for name in names:
        m = getattr(attn, name, None)
        if m is not None:
            return m
    return None


def register_qkv_hooks(
    model: nn.Module,
    on_qkv: Callable[[int, torch.Tensor, torch.Tensor, torch.Tensor], None],
) -> list[HookHandle]:
    handles: list[HookHandle] = []
    attn_modules = _find_attention_modules(model)

    for layer_idx, attn in attn_modules:
        q_proj = _get_proj(attn, ["q_proj", "query"])
        k_proj = _get_proj(attn, ["k_proj", "key"])
        v_proj = _get_proj(attn, ["v_proj", "value"])

        if q_proj is None or k_proj is None or v_proj is None:
            continue

        captured: dict[str, torch.Tensor] = {}
        idx = layer_idx

        def _make_hook(name: str, store: dict):
            def hook(mod, inp, out):
                t = out[0] if isinstance(out, tuple) else out
                store[name] = t.detach().clone()
            return hook

        h_q = q_proj.register_forward_hook(_make_hook("q", captured))
        h_k = k_proj.register_forward_hook(_make_hook("k", captured))
        h_v = v_proj.register_forward_hook(_make_hook("v", captured))

        def _make_attn_hook(store: dict, li: int):
            def hook(mod, inp, out):
                q = store.pop("q", None)
                k = store.pop("k", None)
                v = store.pop("v", None)
                if q is not None and k is not None and v is not None:
                    on_qkv(li, q, k, v)
            return hook

        h_post = attn.register_forward_hook(_make_attn_hook(captured, idx))
        handles.extend([h_q, h_k, h_v, h_post])

    return handles


def remove_hooks(handles: list[HookHandle]) -> None:
    for h in handles:
        h.remove()


def count_attention_layers(model: nn.Module) -> int:
    return len(_find_attention_modules(model))
