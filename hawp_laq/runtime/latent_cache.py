from __future__ import annotations

import torch

from hawp_laq.runtime.quantizer import KQuantizer, VQuantizer, KQuantizeResult, VQuantizeResult
from hawp_laq.utils.memory import tensor_nbytes


class LayerKVCache:
    def __init__(
        self,
        n_heads: int,
        head_dim: int,
        k_quantizer: KQuantizer,
        v_quantizer: VQuantizer,
    ):
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.d_model = n_heads * head_dim
        self.k_quantizer = k_quantizer
        self.v_quantizer = v_quantizer

        self._high_k: list[torch.Tensor] = []
        self._high_v: list[torch.Tensor] = []
        self._low_k: KQuantizeResult | None = None
        self._low_v: VQuantizeResult | None = None
        self._low_k_raw: torch.Tensor | None = None
        self._low_v_raw: torch.Tensor | None = None

    def append_high(self, k: torch.Tensor, v: torch.Tensor) -> None:
        self._high_k.append(k.detach().cpu())
        self._high_v.append(v.detach().cpu())

    def demote_to_low(self) -> None:
        if not self._high_k:
            return
        high_k = torch.cat(self._high_k, dim=0)
        high_v = torch.cat(self._high_v, dim=0)
        self._high_k.clear()
        self._high_v.clear()

        if self._low_k_raw is not None:
            high_k = torch.cat([self._low_k_raw, high_k], dim=0)
            high_v = torch.cat([self._low_v_raw, high_v], dim=0)

        self._low_k_raw = high_k
        self._low_v_raw = high_v
        self._low_k = self.k_quantizer.quantize(high_k)
        self._low_v = self.v_quantizer.quantize(high_v)

    def drop_token(self, indices: list[int] | torch.Tensor) -> None:
        if isinstance(indices, list):
            indices = torch.tensor(indices, dtype=torch.long)
        indices = indices.to(torch.long)

        if self._high_k:
            high_k = torch.cat(self._high_k, dim=0)
            high_v = torch.cat(self._high_v, dim=0)
            mask = torch.ones(high_k.shape[0], dtype=torch.bool)
            valid = indices[indices < high_k.shape[0]]
            mask[valid] = False
            self._high_k = [high_k[mask]]
            self._high_v = [high_v[mask]]
            self._high_k_raw = None
            self._high_v_raw = None

        if self._low_k_raw is not None:
            mask = torch.ones(self._low_k_raw.shape[0], dtype=torch.bool)
            offset = self._high_k[0].shape[0] if self._high_k else 0
            low_indices = indices - offset
            valid = (low_indices >= 0) & (low_indices < self._low_k_raw.shape[0])
            mask[low_indices[valid]] = False
            self._low_k_raw = self._low_k_raw[mask]
            self._low_v_raw = self._low_v_raw[mask]
            if self._low_k_raw.shape[0] > 0:
                self._low_k = self.k_quantizer.quantize(self._low_k_raw)
                self._low_v = self.v_quantizer.quantize(self._low_v_raw)
            else:
                self._low_k = None
                self._low_v = None
                self._low_k_raw = None
                self._low_v_raw = None

    def get_all_k(self) -> torch.Tensor:
        parts = []
        if self._low_k is not None:
            parts.append(KQuantizer.dequantize(self._low_k))
        if self._high_k:
            parts.append(torch.cat(self._high_k, dim=0))
        if not parts:
            return torch.empty(0, self.d_model)
        return torch.cat(parts, dim=0)

    def get_all_v(self) -> torch.Tensor:
        parts = []
        if self._low_v is not None:
            parts.append(VQuantizer.dequantize(self._low_v))
        if self._high_v:
            parts.append(torch.cat(self._high_v, dim=0))
        if not parts:
            return torch.empty(0, self.d_model)
        return torch.cat(parts, dim=0)

    @property
    def n_high(self) -> int:
        if not self._high_k:
            return 0
        return sum(t.shape[0] for t in self._high_k)

    @property
    def n_low(self) -> int:
        if self._low_k_raw is None:
            return 0
        return self._low_k_raw.shape[0]

    @property
    def total_tokens(self) -> int:
        return self.n_high + self.n_low

    def nbytes_high(self) -> int:
        total = 0
        for t in self._high_k:
            total += tensor_nbytes(t)
        for t in self._high_v:
            total += tensor_nbytes(t)
        return total

    def nbytes_low(self) -> int:
        total = 0
        if self._low_k is not None:
            total += tensor_nbytes(self._low_k.q) + tensor_nbytes(self._low_k.scale)
            if self._low_k.rotation is not None:
                total += tensor_nbytes(self._low_k.rotation)
        if self._low_v is not None:
            total += tensor_nbytes(self._low_v.q) + tensor_nbytes(self._low_v.scale) + tensor_nbytes(self._low_v.zero_point)
            if self._low_v.residual is not None:
                total += tensor_nbytes(self._low_v.residual)
        return total
