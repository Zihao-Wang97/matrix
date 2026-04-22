from __future__ import annotations

from typing import Any

import torch

from hawp_laq.runtime.turboquant import TurboQuantMSE, TurboQuantProd, TurboQuantizedTensor, TurboQuantProdResult
from hawp_laq.utils.memory import tensor_nbytes


class LayerKVCache:
    """Per-layer KV cache with recent/archive tiers and TurboQuant compression.

    * **Recent** tokens are stored as fp16 latent tensors.
    * **Archive** tokens are compressed:
        - K uses ``TurboQuantProd`` (MSE + 1-bit residual for inner-product fidelity)
        - V uses ``TurboQuantMSE`` (MSE-optimised reconstruction)

    Args:
        n_heads: Number of KV heads.
        head_dim: Dimension per head (latent dim = head_dim for full-rank,
            or r_k/r_v for low-rank).
        k_quantizer: TurboQuantProd instance for key compression.
        v_quantizer: TurboQuantMSE instance for value compression.
        dtype: Storage dtype for recent tokens.  Default float16.
    """

    def __init__(
        self,
        n_heads: int,
        head_dim: int,
        k_quantizer: TurboQuantProd,
        v_quantizer: TurboQuantMSE,
        dtype: torch.dtype = torch.float16,
    ) -> None:
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.d_model = n_heads * head_dim
        self.k_quantizer = k_quantizer
        self.v_quantizer = v_quantizer
        self.dtype = dtype

        self._recent_k: list[torch.Tensor] = []
        self._recent_v: list[torch.Tensor] = []

        self._archive_k: TurboQuantProdResult | None = None
        self._archive_v: TurboQuantizedTensor | None = None
        self._archive_k_raw: torch.Tensor | None = None
        self._archive_v_raw: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Append
    # ------------------------------------------------------------------

    def append_recent(self, k: torch.Tensor, v: torch.Tensor) -> None:
        """Append a token's latent KV to the recent (uncompressed) tier.

        Args:
            k: Latent key, shape ``[r_k]`` or ``[1, r_k]``.
            v: Latent value, shape ``[r_v]`` or ``[1, r_v]``.
        """
        if k.dim() == 1:
            k = k.unsqueeze(0)
        if v.dim() == 1:
            v = v.unsqueeze(0)
        self._recent_k.append(k.detach().to(self.dtype).cpu())
        self._recent_v.append(v.detach().to(self.dtype).cpu())

    # ------------------------------------------------------------------
    # Demote recent → archive
    # ------------------------------------------------------------------

    def demote_to_archive(self) -> None:
        """Move all recent tokens into the compressed archive tier."""
        if not self._recent_k:
            return

        recent_k = torch.cat(self._recent_k, dim=0).float()
        recent_v = torch.cat(self._recent_v, dim=0).float()
        self._recent_k.clear()
        self._recent_v.clear()

        if self._archive_k_raw is not None:
            recent_k = torch.cat([self._archive_k_raw, recent_k], dim=0)
            recent_v = torch.cat([self._archive_v_raw, recent_v], dim=0)

        self._archive_k_raw = recent_k
        self._archive_v_raw = recent_v
        self._archive_k = self.k_quantizer.quantize(recent_k)
        self._archive_v = self.v_quantizer.quantize(recent_v)

    # ------------------------------------------------------------------
    # Retrieve
    # ------------------------------------------------------------------

    def get_all_k(self) -> torch.Tensor:
        """Return all latent keys, dequantizing the archive on the fly.

        Returns:
            Float tensor of shape ``[T, r_k]``.
        """
        parts: list[torch.Tensor] = []
        if self._archive_k is not None:
            parts.append(self.k_quantizer.dequantize(self._archive_k))
        if self._recent_k:
            parts.append(torch.cat(self._recent_k, dim=0).float())
        if not parts:
            return torch.empty(0, self.k_quantizer.dim)
        return torch.cat(parts, dim=0)

    def get_all_v(self) -> torch.Tensor:
        """Return all latent values, dequantizing the archive on the fly.

        Returns:
            Float tensor of shape ``[T, r_v]``.
        """
        parts: list[torch.Tensor] = []
        if self._archive_v is not None:
            parts.append(self.v_quantizer.dequantize(self._archive_v))
        if self._recent_v:
            parts.append(torch.cat(self._recent_v, dim=0).float())
        if not parts:
            return torch.empty(0, self.v_quantizer.dim)
        return torch.cat(parts, dim=0)

    def drop_oldest(self, n: int) -> int:
        if self._archive_k_raw is None:
            return 0
        T = self._archive_k_raw.shape[0]
        drop_n = min(n, T)
        if drop_n == 0:
            return 0
        self._archive_k_raw = self._archive_k_raw[drop_n:]
        self._archive_v_raw = self._archive_v_raw[drop_n:]
        if self._archive_k_raw.shape[0] > 0:
            self._archive_k = self.k_quantizer.quantize(self._archive_k_raw)
            self._archive_v = self.v_quantizer.quantize(self._archive_v_raw)
        else:
            self._archive_k_raw = None
            self._archive_v_raw = None
            self._archive_k = None
            self._archive_v = None
        return drop_n

    # ------------------------------------------------------------------
    # Token counts
    # ------------------------------------------------------------------

    @property
    def n_recent(self) -> int:
        if not self._recent_k:
            return 0
        return sum(t.shape[0] for t in self._recent_k)

    @property
    def n_archive(self) -> int:
        if self._archive_k_raw is None:
            return 0
        return self._archive_k_raw.shape[0]

    @property
    def total_tokens(self) -> int:
        return self.n_recent + self.n_archive

    # ------------------------------------------------------------------
    # Memory estimation
    # ------------------------------------------------------------------

    def nbytes_recent(self) -> int:
        total = 0
        for t in self._recent_k:
            total += tensor_nbytes(t)
        for t in self._recent_v:
            total += tensor_nbytes(t)
        return total

    def nbytes_archive(self) -> int:
        total = 0
        if self._archive_k is not None:
            total += self.k_quantizer.estimate_num_bytes(self._archive_k)
        if self._archive_v is not None:
            total += self.v_quantizer.estimate_num_bytes(self._archive_v)
        return total

    def nbytes_total(self) -> int:
        return self.nbytes_recent() + self.nbytes_archive()
