from __future__ import annotations

import torch

from hawp_laq.runtime.latent_cache import LayerKVCache
from hawp_laq.runtime.quantizer import KQuantizer, VQuantizer
from hawp_laq.runtime.scheduler import TokenBudgetScheduler, TokenState
from hawp_laq.utils.memory import format_nbytes


class CacheManager:
    def __init__(
        self,
        n_layers: int,
        n_heads: int,
        head_dim: int,
        scheduler: TokenBudgetScheduler,
        k_group_size: int = 128,
        v_group_size: int = 128,
        use_rotation: bool = False,
        outlier_threshold: float | None = None,
    ):
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.scheduler = scheduler

        self._caches: list[LayerKVCache] = []
        for _ in range(n_layers):
            kq = KQuantizer(group_size=k_group_size, use_rotation=use_rotation)
            vq = VQuantizer(group_size=v_group_size, outlier_threshold=outlier_threshold)
            self._caches.append(LayerKVCache(n_heads, head_dim, kq, vq))

    def append_token(self, k_per_layer: list[torch.Tensor], v_per_layer: list[torch.Tensor]) -> None:
        if len(k_per_layer) != self.n_layers or len(v_per_layer) != self.n_layers:
            raise ValueError(f"Expected {self.n_layers} layers, got {len(k_per_layer)} k and {len(v_per_layer)} v")

        self.scheduler.on_new_token()
        for i, (k, v) in enumerate(zip(k_per_layer, v_per_layer)):
            self._caches[i].append_high(k, v)

    def get_kv_for_attention(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        cache = self._caches[layer_idx]
        return cache.get_all_k(), cache.get_all_v()

    def total_nbytes(self) -> int:
        total = 0
        for c in self._caches:
            total += c.nbytes_high() + c.nbytes_low()
        return total

    def total_nbytes_formatted(self) -> str:
        return format_nbytes(self.total_nbytes())

    def demote_all(self) -> None:
        for c in self._caches:
            c.demote_to_low()

    def apply_scheduler(self) -> list[int]:
        drop_indices = self.scheduler.rebalance()
        if drop_indices:
            for c in self._caches:
                c.drop_token(drop_indices)
        return drop_indices

    def summary(self) -> dict:
        high_tokens = self._caches[0].n_high if self._caches else 0
        low_tokens = self._caches[0].n_low if self._caches else 0
        return {
            "seq_len": self.scheduler.seq_len,
            "high_tokens": high_tokens,
            "low_tokens": low_tokens,
            "total_nbytes": self.total_nbytes(),
            "total_nbytes_formatted": self.total_nbytes_formatted(),
        }

    def __getitem__(self, layer_idx: int) -> LayerKVCache:
        return self._caches[layer_idx]

    def __len__(self) -> int:
        return self.n_layers
