from __future__ import annotations

from typing import Any

import torch

from hawp_laq.modeling.attention_hawp import _QuantChunk
from hawp_laq.runtime.turboquant import TurboQuantMSE, TurboQuantProd, TurboQuantizedTensor, TurboQuantProdResult
from hawp_laq.utils.memory import tensor_nbytes


class LayerKVCache:
    """Per-layer KV cache with recent/archive tiers and TurboQuant compression.

    * **Recent** tokens are stored in ``dtype`` latent tensors.
    * **Archive** tokens are compressed into per-demote chunks:
        - K uses ``TurboQuantProd`` (MSE + 1-bit residual for inner-product fidelity)
        - V uses ``TurboQuantMSE`` (MSE-optimised reconstruction)

    No raw (unquantized) archive data is retained — only quantized chunks,
    ensuring actual memory savings from compression.

    Args:
        n_heads: Number of KV heads.
        head_dim: Dimension per head (latent dim = head_dim for full-rank,
            or r_k/r_v for low-rank).
        k_quantizer: TurboQuantProd instance for key compression.
        v_quantizer: TurboQuantMSE instance for value compression.
        dtype: Storage dtype for recent tokens.  Must be passed explicitly;
            should match the model weight dtype for consistency across
            quantised and non-quantised attention paths.
    """

    def __init__(
        self,
        n_heads: int,
        head_dim: int,
        k_quantizer: TurboQuantProd,
        v_quantizer: TurboQuantMSE,
        dtype: torch.dtype,
        recent_window: int = 0,
    ) -> None:
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.d_model = n_heads * head_dim
        self.k_quantizer = k_quantizer
        self.v_quantizer = v_quantizer
        self.dtype = dtype
        self.recent_window = recent_window

        self._recent_k: list[torch.Tensor] = []
        self._recent_v: list[torch.Tensor] = []

        self._archive_chunks: list[_QuantChunk] = []

    # ------------------------------------------------------------------
    # Append
    # ------------------------------------------------------------------

    def append_recent(self, k: torch.Tensor, v: torch.Tensor) -> None:
        """Append a token's latent KV to the recent (uncompressed) tier.

        If ``recent_window > 0`` and the recent buffer exceeds it after
        the append, excess tokens are automatically demoted to the
        compressed archive tier — ensuring real memory savings.

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
        if self.recent_window > 0:
            self._demote_excess()

    # ------------------------------------------------------------------
    # Demote recent → archive
    # ------------------------------------------------------------------

    def _quantize_flat(self, k: torch.Tensor, v: torch.Tensor) -> _QuantChunk:
        """Quantize a [T, D] KV pair into a chunk."""
        T = k.shape[0]
        dk = k.shape[-1]
        dv = v.shape[-1]
        k_qx = self.k_quantizer.quantize(k.float())
        v_qx = self.v_quantizer.quantize(v.float())
        k_norms = None
        return _QuantChunk(k_qx, v_qx, T, k_norms)

    def _demote_excess(self) -> None:
        """Demote tokens from recent to archive when recent_window is exceeded.

        Keeps at most ``recent_window`` tokens in the recent tier;
        all older tokens are quantized and moved to the archive.
        This is the counterpart of ``HAWPAttention._quant_cache_demote``
        for the standalone ``LayerKVCache`` path.
        """
        if self.recent_window <= 0:
            return
        while self.n_recent > self.recent_window:
            n_demote = self.n_recent - self.recent_window
            k_all = torch.cat(self._recent_k, dim=0)
            v_all = torch.cat(self._recent_v, dim=0)
            self._recent_k.clear()
            self._recent_v.clear()
            k_demote = k_all[:n_demote]
            v_demote = v_all[:n_demote]
            k_keep = k_all[n_demote:]
            v_keep = v_all[n_demote:]
            chunk = self._quantize_flat(k_demote, v_demote)
            self._archive_chunks.append(chunk)
            if k_keep.shape[0] > 0:
                self._recent_k.append(k_keep)
                self._recent_v.append(v_keep)

    def demote_to_archive(self) -> None:
        """Move all recent tokens into the compressed archive tier.

        Creates a new quantized chunk from the recent buffer.  No raw
        archive data is retained — only the quantized chunk.
        """
        if not self._recent_k:
            return

        recent_k = torch.cat(self._recent_k, dim=0).float()
        recent_v = torch.cat(self._recent_v, dim=0).float()
        self._recent_k.clear()
        self._recent_v.clear()

        chunk = self._quantize_flat(recent_k, recent_v)
        self._archive_chunks.append(chunk)

    # ------------------------------------------------------------------
    # Retrieve
    # ------------------------------------------------------------------

    def get_all_k(self) -> torch.Tensor:
        """Return all latent keys, dequantizing the archive on the fly.

        Returns:
            Tensor of shape ``[T, r_k]`` in ``self.dtype``.
        """
        parts: list[torch.Tensor] = []
        for chunk in self._archive_chunks:
            deq = self.k_quantizer.dequantize(chunk.k_qx)
            parts.append(deq)
        if self._recent_k:
            parts.append(torch.cat(self._recent_k, dim=0).to(self.dtype))
        if not parts:
            return torch.empty(0, self.k_quantizer.dim)
        return torch.cat(parts, dim=0).to(self.dtype)

    def get_all_v(self) -> torch.Tensor:
        """Return all latent values, dequantizing the archive on the fly.

        Returns:
            Tensor of shape ``[T, r_v]`` in ``self.dtype``.
        """
        parts: list[torch.Tensor] = []
        for chunk in self._archive_chunks:
            deq = self.v_quantizer.dequantize(chunk.v_qx)
            parts.append(deq)
        if self._recent_v:
            parts.append(torch.cat(self._recent_v, dim=0).to(self.dtype))
        if not parts:
            return torch.empty(0, self.v_quantizer.dim)
        return torch.cat(parts, dim=0).to(self.dtype)

    def drop_oldest(self, n: int) -> int:
        """Drop the oldest *n* archive tokens.

        Processes chunks from oldest to newest.  Whole chunks are removed
        when possible.  When a partial chunk must be trimmed, the chunk
        is temporarily dequantized, sliced, and re-quantized (no raw data
        is kept).
        """
        if not self._archive_chunks:
            return 0
        dropped = 0
        while self._archive_chunks and dropped < n:
            first = self._archive_chunks[0]
            can_drop = min(first.n_tokens, n - dropped)
            if can_drop >= first.n_tokens:
                self._archive_chunks.pop(0)
                dropped += first.n_tokens
            else:
                remaining = first.n_tokens - can_drop
                k_deq = self.k_quantizer.dequantize(first.k_qx)[can_drop:]
                v_deq = self.v_quantizer.dequantize(first.v_qx)[can_drop:]
                new_chunk = self._quantize_flat(k_deq, v_deq)
                self._archive_chunks[0] = new_chunk
                dropped += can_drop
        return dropped

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
        return sum(c.n_tokens for c in self._archive_chunks)

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
        for chunk in self._archive_chunks:
            total += self.k_quantizer.estimate_num_bytes(chunk.k_qx)
            total += self.v_quantizer.estimate_num_bytes(chunk.v_qx)
        return total

    def nbytes_total(self) -> int:
        return self.nbytes_recent() + self.nbytes_archive()
