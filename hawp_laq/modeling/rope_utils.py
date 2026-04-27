from __future__ import annotations

import math
import warnings

import torch
import torch.nn as nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: float = 10000.0,
        rope_scaling: dict | None = None,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.rope_scaling = rope_scaling

        self._rope_type = "default"
        self._scaling_factor = 1.0

        if rope_scaling is not None:
            self._rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", "default"))
            self._scaling_factor = rope_scaling.get("factor", 1.0)

            supported = {"default", "linear", "dynamic", "llama3", "yarn", "longrope"}
            if self._rope_type not in supported:
                warnings.warn(
                    f"[rope_utils] Unsupported rope_scaling type '{self._rope_type}'. "
                    f"Supported: {sorted(supported)}. Falling back to 'default' (no scaling). "
                    f"This may cause incorrect positional encodings.",
                    UserWarning,
                    stacklevel=2,
                )
                self._rope_type = "default"
                self._scaling_factor = 1.0

        inv_freq = self._compute_inv_freq(seq_len=max_position_embeddings)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(seq_len=max_position_embeddings)

    def _compute_inv_freq(self, seq_len: int | None = None) -> torch.Tensor:
        base = self.base
        dim = self.dim

        if self._rope_type == "dynamic":
            if seq_len is not None and seq_len > self.max_position_embeddings:
                context_factor = self._scaling_factor * seq_len / self.max_position_embeddings - (self._scaling_factor - 1)
                base = base * (context_factor ** (dim / (dim - 2)))

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))

        if self._rope_type == "llama3":
            inv_freq = self._apply_llama3_scaling(inv_freq)

        if self._rope_type == "longrope":
            inv_freq = self._apply_longrope_scaling(inv_freq, seq_len)

        if self._rope_type == "yarn":
            inv_freq = self._apply_yarn_scaling(inv_freq, seq_len)

        return inv_freq

    def _apply_llama3_scaling(self, inv_freq: torch.Tensor) -> torch.Tensor:
        factor = self._scaling_factor
        low_freq_factor = self.rope_scaling.get("low_freq_factor", 1.0) if self.rope_scaling else 1.0
        high_freq_factor = self.rope_scaling.get("high_freq_factor", 4.0) if self.rope_scaling else 4.0
        original_max_pos = (
            self.rope_scaling.get("original_max_position_embeddings", self.max_position_embeddings)
            if self.rope_scaling
            else self.max_position_embeddings
        )

        low_freq_wavelen = original_max_pos / low_freq_factor
        high_freq_wavelen = original_max_pos / high_freq_factor

        new_freqs = []
        for freq in inv_freq:
            wavelen = 2 * math.pi / freq.item()
            if wavelen < high_freq_wavelen:
                new_freqs.append(freq)
            elif wavelen > low_freq_wavelen:
                new_freqs.append(freq / factor)
            else:
                smooth = (original_max_pos / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
                new_freqs.append((1 - smooth) * freq / factor + smooth * freq)

        return torch.stack(new_freqs)

    def _apply_longrope_scaling(self, inv_freq: torch.Tensor, seq_len: int | None) -> torch.Tensor:
        if not self.rope_scaling:
            return inv_freq
        short_factors = self.rope_scaling.get("short_factor", None)
        long_factors = self.rope_scaling.get("long_factor", None)
        if short_factors is None or long_factors is None:
            warnings.warn(
                "[rope_utils] longrope scaling requires 'short_factor' and 'long_factor' "
                "in rope_scaling config. Using default (no scaling).",
                UserWarning,
                stacklevel=3,
            )
            return inv_freq

        scale = self._scaling_factor
        if seq_len is not None and seq_len > self.max_position_embeddings:
            factors = torch.tensor(long_factors, dtype=torch.float32)
        else:
            factors = torch.tensor(short_factors, dtype=torch.float32)

        if factors.shape[0] != inv_freq.shape[0]:
            warnings.warn(
                f"[rope_utils] longrope factor length ({factors.shape[0]}) != "
                f"inv_freq length ({inv_freq.shape[0]}). Using default (no scaling).",
                UserWarning,
                stacklevel=3,
            )
            return inv_freq

        return inv_freq * factors.to(device=inv_freq.device)

    def _apply_yarn_scaling(self, inv_freq: torch.Tensor, seq_len: int | None) -> torch.Tensor:
        if not self.rope_scaling:
            return inv_freq
        factor = self._scaling_factor
        attention_factor = self.rope_scaling.get("attention_factor", None)
        if attention_factor is None:
            attention_factor = 1.0
            if factor > 1.0:
                sqrt_dim = math.sqrt(self.dim)
                attention_factor = 0.1 * sqrt_dim * math.log(factor) + 1.0
        self._yarn_attention_factor = attention_factor

        original_max_pos = (
            self.rope_scaling.get("original_max_position_embeddings", self.max_position_embeddings)
            if self.rope_scaling
            else self.max_position_embeddings
        )

        if seq_len is not None and seq_len > original_max_pos:
            scale = factor
        else:
            scale = 1.0

        return inv_freq / scale

    def _set_cos_sin_cache(self, seq_len: int):
        self.max_seq_len_cached = seq_len

        if self._rope_type == "dynamic" and seq_len > self.max_position_embeddings:
            new_inv_freq = self._compute_inv_freq(seq_len=seq_len)
            self.inv_freq = new_inv_freq.to(device=self.inv_freq.device, dtype=self.inv_freq.dtype)

        t = torch.arange(seq_len, dtype=torch.float32, device=self.inv_freq.device)

        if self._rope_type == "linear":
            t = t / self._scaling_factor

        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

        attn_factor = getattr(self, "_yarn_attention_factor", 1.0)
        if attn_factor != 1.0:
            emb = emb * attn_factor

        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int | None = None):
        if seq_len is None:
            seq_len = x.shape[1]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len)
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )
