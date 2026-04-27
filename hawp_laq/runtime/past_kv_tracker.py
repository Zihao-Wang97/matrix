"""PastKVTracker: measure real runtime KV cache bytes from ``past_key_values``.

For modes that use standard HF ``model.generate()`` or a stepwise loop with
``use_cache=True``, the returned ``past_key_values`` contains the actual K/V
tensors held in memory.  This tracker walks those tensors and sums
``nelement() * element_size()`` to get the **true runtime cache footprint** —
no formula estimation needed.
"""

from __future__ import annotations

import torch


class PastKVTracker:
    def __init__(self) -> None:
        self._total_tokens: int = 0
        self._total_bytes: int = 0
        self._n_updates: int = 0

    def update(self, past_key_values) -> None:
        if past_key_values is None:
            return
        total_bytes = 0
        total_tokens = 0
        counted_layers = 0

        if hasattr(past_key_values, "key_cache"):
            for layer_idx in range(len(past_key_values.key_cache)):
                k = past_key_values.key_cache[layer_idx]
                v = past_key_values.value_cache[layer_idx]
                if k is not None and k.numel() > 0:
                    total_bytes += k.nelement() * k.element_size()
                    total_bytes += v.nelement() * v.element_size()
                    total_tokens = k.shape[2] if k.dim() >= 3 else k.shape[1]
                    counted_layers += 1
        else:
            for layer_kv in past_key_values:
                if layer_kv is None:
                    continue
                k, v = layer_kv[0], layer_kv[1]
                if k is not None and k.numel() > 0:
                    total_bytes += k.nelement() * k.element_size()
                    total_bytes += v.nelement() * v.element_size()
                    total_tokens = k.shape[2] if k.dim() >= 3 else k.shape[1]
                    counted_layers += 1

        if counted_layers > 0:
            self._total_tokens = total_tokens
            self._total_bytes = total_bytes
        self._n_updates += 1

    @property
    def total_tokens(self) -> int:
        return self._total_tokens

    @property
    def total_bytes(self) -> int:
        return self._total_bytes

    def reset(self) -> None:
        self._total_tokens = 0
        self._total_bytes = 0
        self._n_updates = 0
