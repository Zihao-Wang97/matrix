from __future__ import annotations

from types import MethodType
from typing import Any

import torch


def _get_nested_attr(obj: Any, path: str) -> Any:
    cur = obj
    for part in path.split("."):
        if cur is None or not hasattr(cur, part):
            return None
        cur = getattr(cur, part)
    return cur


def find_transformer_layers(model) -> Any:
    for path in ("model.layers", "model.decoder.layers", "transformer.h", "gpt_neox.layers"):
        layers = _get_nested_attr(model, path)
        if layers is not None:
            return layers
    return None


def _mark_mlp(module, name: str, **meta: Any) -> None:
    callback = getattr(module, "_hawp_mlp_marker_callback", None)
    if callback is not None:
        callback(module, name, meta)


def _chunked_mlp_forward(module, hidden_states: torch.Tensor, *args, **kwargs):
    original_forward = module._hawp_original_forward
    chunk_size = int(getattr(module, "_hawp_mlp_chunk_size", 0) or 0)
    min_seq_len = int(getattr(module, "_hawp_mlp_min_seq_len", 0) or 0)

    if (
        not isinstance(hidden_states, torch.Tensor)
        or hidden_states.dim() < 3
        or chunk_size <= 0
        or hidden_states.shape[-2] <= chunk_size
        or hidden_states.shape[-2] < min_seq_len
    ):
        return original_forward(hidden_states, *args, **kwargs)
    if torch.is_grad_enabled() and hidden_states.requires_grad:
        return original_forward(hidden_states, *args, **kwargs)

    seq_len = int(hidden_states.shape[-2])
    _mark_mlp(
        module,
        "prefill_mlp_chunking.before",
        q_len=seq_len,
        chunk_size=chunk_size,
    )

    first_end = min(chunk_size, seq_len)
    first_hidden = hidden_states.narrow(-2, 0, first_end)
    first_out = original_forward(first_hidden, *args, **kwargs)
    if not isinstance(first_out, torch.Tensor) or first_out.dim() < 3:
        return original_forward(hidden_states, *args, **kwargs)

    out_shape = hidden_states.shape[:-2] + (seq_len, first_out.shape[-1])
    output = first_out.new_empty(out_shape)
    output.narrow(-2, 0, first_end).copy_(first_out)
    _mark_mlp(
        module,
        "prefill_mlp_chunking.output_alloc.after",
        q_len=seq_len,
        chunk_size=chunk_size,
        shape=tuple(output.shape),
    )
    _mark_mlp(
        module,
        "prefill_mlp_chunking.chunk.after",
        q_len=seq_len,
        chunk_size=chunk_size,
        chunk_idx=0,
        start=0,
        end=int(first_end),
    )
    del first_hidden, first_out

    chunk_idx = 1
    for start in range(first_end, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        chunk_len = end - start
        chunk_hidden = hidden_states.narrow(-2, start, chunk_len)
        chunk_out = original_forward(chunk_hidden, *args, **kwargs)
        output.narrow(-2, start, chunk_len).copy_(chunk_out)
        if end == seq_len:
            _mark_mlp(
                module,
                "prefill_mlp_chunking.chunk.after",
                q_len=seq_len,
                chunk_size=chunk_size,
                chunk_idx=int(chunk_idx),
                start=int(start),
                end=int(end),
            )
        del chunk_hidden, chunk_out
        chunk_idx += 1

    _mark_mlp(
        module,
        "prefill_mlp_chunking.after",
        q_len=seq_len,
        chunk_size=chunk_size,
        n_chunks=int(chunk_idx),
        shape=tuple(output.shape),
    )
    return output


def _patch_mlp(module, *, chunk_size: int, min_seq_len: int) -> bool:
    if getattr(module, "_hawp_mlp_chunking_installed", False):
        module._hawp_mlp_chunk_size = int(chunk_size)
        module._hawp_mlp_min_seq_len = int(min_seq_len)
        return False

    module._hawp_original_forward = module.forward
    module._hawp_mlp_chunk_size = int(chunk_size)
    module._hawp_mlp_min_seq_len = int(min_seq_len)
    module._hawp_mlp_marker_callback = None

    def forward(self, hidden_states, *args, **kwargs):
        return _chunked_mlp_forward(self, hidden_states, *args, **kwargs)

    module.forward = MethodType(forward, module)
    module._hawp_mlp_chunking_installed = True
    return True


def install_prefill_mlp_chunking(
    model,
    *,
    chunk_size: int,
    min_seq_len: int | None = None,
) -> int:
    if chunk_size <= 0:
        return 0
    if min_seq_len is None or min_seq_len <= 0:
        min_seq_len = chunk_size + 1

    layers = find_transformer_layers(model)
    if layers is None:
        return 0

    n_installed = 0
    for layer in layers:
        mlp = getattr(layer, "mlp", None) or getattr(layer, "feed_forward", None) or getattr(layer, "ffn", None)
        if mlp is None:
            continue
        if _patch_mlp(mlp, chunk_size=int(chunk_size), min_seq_len=int(min_seq_len)):
            n_installed += 1
    return n_installed
