from __future__ import annotations

import logging

import math
from types import SimpleNamespace
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from hawp_laq.modeling.rope_utils import LlamaRotaryEmbedding, apply_rotary_pos_emb
from hawp_laq.utils.packbits import unpack_bool


logger = logging.getLogger(__name__)


def _resolve_compute_dtype(ref_param: nn.Module, fallback_dtype: torch.dtype = torch.float32) -> torch.dtype:
    if hasattr(ref_param, "compute_dtype"):
        return ref_param.compute_dtype
    ref_weight = ref_param.weight
    if hasattr(ref_weight, "quant_state") and hasattr(ref_weight.quant_state, "dtype"):
        return ref_weight.quant_state.dtype
    dtype = getattr(ref_weight, "dtype", torch.float32)
    if not dtype.is_floating_point:
        return fallback_dtype
    return dtype


class _QuantChunk:
    __slots__ = ('k_qx', 'v_qx', 'n_tokens', 'k_norms')

    def __init__(self, k_qx, v_qx, n_tokens: int, k_norms: Optional[torch.Tensor] = None):
        self.k_qx = k_qx
        self.v_qx = v_qx
        self.n_tokens = n_tokens
        self.k_norms = k_norms


_DEFAULT_CONFIG = SimpleNamespace(
    hidden_size=768,
    num_attention_heads=12,
    num_key_value_heads=12,
    max_position_embeddings=2048,
    rope_theta=10000.0,
    rope_scaling=None,
    model_type="",
    enable_bias=False,
    attention_dropout=0.0,
)


def _get_attn_config(base_attn_module, model=None):
    if model is not None and hasattr(model, "config") and model.config is not None:
        return model.config
    if base_attn_module is not None:
        if hasattr(base_attn_module, "config") and base_attn_module.config is not None:
            return base_attn_module.config
        if hasattr(base_attn_module, "self_attn") and base_attn_module.self_attn is not None:
            inner = base_attn_module.self_attn
            if hasattr(inner, "config") and inner.config is not None:
                return inner.config
        src_q = getattr(base_attn_module, "q_proj", None) or getattr(base_attn_module, "query_proj", None)
        if src_q is not None:
            out_features = src_q.out_features
            in_features = src_q.in_features
            num_heads = max(1, out_features // (out_features // in_features * in_features // in_features))
            head_dim = out_features // num_heads
            num_heads = max(1, out_features // head_dim) if head_dim > 0 else 1
            return SimpleNamespace(
                hidden_size=in_features,
                num_attention_heads=num_heads,
                num_key_value_heads=num_heads,
                max_position_embeddings=2048,
                rope_theta=10000.0,
                rope_scaling=None,
                model_type="",
                enable_bias=hasattr(src_q, "bias") and src_q.bias is not None,
                attention_dropout=0.0,
            )
    return _DEFAULT_CONFIG


def _make_causal_mask(q_len: int, kv_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    mask = torch.full((q_len, kv_len), float("-inf"), device=device, dtype=dtype)
    mask = torch.triu(mask, diagonal=kv_len - q_len + 1)
    return mask.unsqueeze(0).unsqueeze(0)


def _prepare_attention_mask(
    attention_mask: Optional[torch.Tensor],
    q_len: int,
    kv_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    if attention_mask is None:
        return _make_causal_mask(q_len, kv_len, device, dtype) if q_len > 1 else None

    if attention_mask.dim() == 4:
        return attention_mask[:, :, :, :kv_len].to(device=device, dtype=dtype)

    if attention_mask.dim() == 2:
        padding_mask = attention_mask[:, :kv_len].to(device=device)
        padding_mask = (1.0 - padding_mask.float()).unsqueeze(1).unsqueeze(2)
        neg_inf = torch.finfo(torch.float32).min
        padding_mask = padding_mask * neg_inf
        if q_len > 1:
            causal = _make_causal_mask(q_len, kv_len, device, torch.float32)
            combined = torch.min(padding_mask, causal)
            return combined.to(dtype=dtype)
        return padding_mask.to(dtype=dtype)

    return attention_mask.to(device=device, dtype=dtype)


def _attention_mask_is_fully_visible(
    attention_mask: Optional[torch.Tensor],
    kv_len: int,
) -> bool:
    """Return True when a 2-D mask has no padding in the visible KV prefix."""
    if attention_mask is None:
        return True
    if attention_mask.dim() != 2 or attention_mask.shape[1] < kv_len:
        return False
    return bool(torch.all(attention_mask[:, :kv_len] != 0).item())


def _scaled_dot_product_attention_with_logit_scale(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    logit_scale: torch.Tensor,
    is_causal: bool,
) -> torch.Tensor:
    """SDPA wrapper for arbitrary low-rank logit scales.

    ``scaled_dot_product_attention`` applies ``1 / sqrt(dim)`` internally.
    Multiplying the query by ``logit_scale * sqrt(dim)`` preserves the existing
    HAWP logits while letting PyTorch use its memory-efficient attention kernel.
    """
    scale_factor = logit_scale * math.sqrt(query.shape[-1])
    return F.scaled_dot_product_attention(
        query * scale_factor,
        key,
        value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=is_causal,
    )


def _cache_passthrough(past_key_value):
    if past_key_value is None:
        return None
    if hasattr(past_key_value, "to_legacy_cache"):
        return past_key_value
    if isinstance(past_key_value, tuple):
        return past_key_value
    if isinstance(past_key_value, list):
        return tuple(past_key_value)
    return None


class HAWPAttention(nn.Module):
    def __init__(
        self,
        config,
        layer_idx: int | None = None,
        r_k: int | None = None,
        r_v: int | None = None,
        allow_default_full_rank: bool = False,
        logit_scale_mode: str = "rk",
        gamma_mode: str = "learned",
        gamma_value: float | None = None,
        use_archive_k_ip_approx: bool = True,
        _skip_linear_init: bool = False,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx if layer_idx is not None else 0

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = getattr(config, "num_key_value_heads", self.num_heads)
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = getattr(config, "max_position_embeddings", 2048)
        self.is_causal = True

        self.model_type = getattr(config, "model_type", "").lower()
        self.is_opt = "opt" in self.model_type
        self.is_gpt_neox = "gpt_neox" in self.model_type

        self.scaling = self.head_dim ** -0.5

        use_bias = self.is_opt and getattr(config, "enable_bias", True)
        q_out = self.num_heads * self.head_dim
        kv_out = self.num_key_value_heads * self.head_dim
        if _skip_linear_init:
            self.q_proj = None
            self.k_proj = None
            self.v_proj = None
            self.v_lat_proj = None
            self.o_proj = None
        else:
            self.q_proj = nn.Linear(self.hidden_size, q_out, bias=use_bias)
            self.k_proj = nn.Linear(self.hidden_size, kv_out, bias=use_bias)
            self.v_proj = nn.Linear(self.hidden_size, kv_out, bias=use_bias)
            self.v_lat_proj = None
            self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=use_bias)

        self._use_rope = not (self.is_opt or self.is_gpt_neox)
        if self._use_rope:
            rope_theta = getattr(config, "rope_theta", 10000.0)
            rope_scaling = getattr(config, "rope_scaling", None)
            self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=rope_theta,
                rope_scaling=rope_scaling,
            )

        has_partial = (r_k is None) != (r_v is None)
        if has_partial:
            raise ValueError(
                "HAWPAttention requires both r_k and r_v together. "
                f"Got r_k={r_k}, r_v={r_v}."
            )

        if r_k is None and r_v is None:
            if allow_default_full_rank:
                r_k = self.head_dim
                r_v = self.head_dim
            else:
                raise ValueError(
                    f"HAWPAttention requires explicit r_k and r_v. "
                    f"Got r_k={r_k}, r_v={r_v}. "
                    f"Use allow_default_full_rank=True only for quant_only mode."
                )
        self.r_k = r_k
        self.r_v = r_v

        if not (1 <= self.r_k <= self.head_dim):
            raise ValueError(f"r_k={self.r_k} must satisfy 1 <= r_k <= head_dim={self.head_dim}")
        if not (1 <= self.r_v <= self.head_dim):
            raise ValueError(f"r_v={self.r_v} must satisfy 1 <= r_v <= head_dim={self.head_dim}")

        self._dtype = dtype if dtype is not None else torch.float32

        self.p_k = nn.Parameter(torch.eye(self.head_dim, dtype=self._dtype), requires_grad=(r_k < self.head_dim))
        self.p_v = nn.Parameter(torch.eye(self.head_dim, dtype=self._dtype), requires_grad=(r_v < self.head_dim))
        self.gamma = nn.Parameter(torch.ones(1, dtype=self._dtype), requires_grad=False)
        self.register_buffer("d_v", None, persistent=True)

        self.logit_scale_mode = logit_scale_mode
        self.gamma_mode = gamma_mode
        self.gamma_value = gamma_value
        self.use_archive_k_ip_approx = use_archive_k_ip_approx
        self.use_decode_blockwise_attention = True
        self.decode_archive_block_size = 1024
        self.use_prefill_grouped_attention = True
        self.use_prefill_fused_value_o_proj = True
        self.use_prefill_fused_v_latent = True
        self.prefill_fused_o_proj_chunk_size = 2048
        self._full_v_proj_offloaded = False
        self._fused_v_lat_weight = None
        self._fused_v_lat_bias = None
        self._fused_v_lat_cache_key = None

        self.use_quantizer = False
        self.use_cache_manager = False
        self.recent_window = 64
        self._calib_callback = None
        self._memory_marker_callback = None
        self._tq_k_quantizer = None
        self._tq_v_quantizer = None
        self._recent_k_buffer = None
        self._recent_v_buffer = None
        self._recent_start = 0
        self._recent_count = 0
        self._quant_archive_chunks: list[_QuantChunk] = []
        self._hawp_parent_use_cache = False
        self._hawp_parent_use_cache_valid = False
        self._hawp_parent_expects_2_outputs = False

    @property
    def _is_low_rank(self) -> bool:
        return self.r_k < self.head_dim or self.r_v < self.head_dim

    def _consume_opt_parent_use_cache(self) -> bool:
        if not self.is_opt:
            return False
        valid = bool(getattr(self, "_hawp_parent_use_cache_valid", False))
        use_cache = valid and bool(getattr(self, "_hawp_parent_use_cache", False))
        self._hawp_parent_use_cache = False
        self._hawp_parent_use_cache_valid = False
        return use_cache

    def _format_attn_output(self, attn_output, attn_weights=None, past_key_value=None):
        if getattr(self, "_hawp_parent_expects_2_outputs", False):
            return attn_output, attn_weights
        return attn_output, attn_weights, past_key_value

    def _mark_memory(self, name: str, **meta) -> None:
        callback = getattr(self, "_memory_marker_callback", None)
        if callback is not None:
            callback(self, name, meta)

    def setup_quant_cache(self, k_quantizer, v_quantizer, recent_window: int = 64) -> None:
        self.use_cache_manager = True
        self.recent_window = recent_window
        self._tq_k_quantizer = k_quantizer
        self._tq_v_quantizer = v_quantizer
        self.reset_quant_cache()

    def materialize_v_lat_proj(self, *, offload_full_v_proj: bool = True) -> bool:
        """Build an inference-only V->latent projection and optionally move full V off GPU."""
        if self.r_v >= self.head_dim:
            return False
        if self.v_proj is None:
            return self.v_lat_proj is not None

        weight = self.v_proj.weight
        bias = getattr(self.v_proj, "bias", None)
        device = weight.device
        dtype = _resolve_compute_dtype(self.v_proj)
        use_bias = bias is not None

        with torch.no_grad():
            v_weight = weight.detach().to(device=device, dtype=dtype)
            v_weight = v_weight.view(self.num_key_value_heads, self.head_dim, self.hidden_size)
            pv_down = self.p_v.detach()[:, :self.r_v].to(device=device, dtype=dtype)
            fused_weight = torch.einsum("odh,dr->orh", v_weight, pv_down).reshape(
                self.num_key_value_heads * self.r_v,
                self.hidden_size,
            ).contiguous()

            fused_bias = None
            if use_bias:
                v_bias = bias.detach().to(device=device, dtype=dtype).view(
                    self.num_key_value_heads,
                    self.head_dim,
                )
                fused_bias = torch.einsum("od,dr->or", v_bias, pv_down).reshape(
                    self.num_key_value_heads * self.r_v,
                ).contiguous()

            v_lat_proj = nn.Linear(
                self.hidden_size,
                self.num_key_value_heads * self.r_v,
                bias=use_bias,
                device=device,
                dtype=dtype,
            )
            v_lat_proj.weight.copy_(fused_weight)
            if use_bias and fused_bias is not None:
                v_lat_proj.bias.copy_(fused_bias)

        self.v_lat_proj = v_lat_proj
        self._fused_v_lat_weight = None
        self._fused_v_lat_bias = None
        self._fused_v_lat_cache_key = None

        if offload_full_v_proj:
            try:
                self.v_proj.to("cpu")
                self._full_v_proj_offloaded = True
            except Exception as exc:
                logger.warning("Failed to offload full v_proj to CPU; keeping it on %s: %s", device, exc)
                self._full_v_proj_offloaded = False
        else:
            self._full_v_proj_offloaded = False

        return True

    def reset_quant_cache(self) -> None:
        self._recent_k_buffer = None
        self._recent_v_buffer = None
        self._recent_start = 0
        self._recent_count = 0
        self._quant_archive_chunks = []

    @property
    def _quant_recent_k(self) -> torch.Tensor | None:
        return self._get_recent_k()

    @property
    def _quant_recent_v(self) -> torch.Tensor | None:
        return self._get_recent_v()

    @property
    def _single_archive_k_qx(self):
        """Return single-chunk k_qx, or None for empty/multi-chunk.

        Only valid when there is exactly one archive chunk.  Returns the
        quantized key tensor for that chunk.  Returns None when the archive
        has 0 or >1 chunks.
        """
        if len(self._quant_archive_chunks) == 1:
            return self._quant_archive_chunks[0].k_qx
        return None

    @property
    def _single_archive_v_qx(self):
        """Return single-chunk v_qx, or None for empty/multi-chunk.

        Same semantics as ``_single_archive_k_qx`` for values.
        """
        if len(self._quant_archive_chunks) == 1:
            return self._quant_archive_chunks[0].v_qx
        return None

    @property
    def n_archive_tokens(self) -> int:
        return sum(c.n_tokens for c in self._quant_archive_chunks)

    def _quant_cache_seq_len(self) -> int:
        total = 0
        for chunk in self._quant_archive_chunks:
            total += chunk.n_tokens
        total += self._recent_count
        return total

    def _quantize_to_chunk(self, k_lat: torch.Tensor, v_lat: torch.Tensor) -> _QuantChunk:
        k_new = k_lat[0].detach()
        v_new = v_lat[0].detach()
        nkv, T, rk = k_new.shape
        _, _, rv = v_new.shape
        self._mark_memory(
            "quantize_to_chunk.detach.after",
            T=int(T),
            nkv=int(nkv),
            r_k=int(rk),
            r_v=int(rv),
        )
        k_flat = k_new.reshape(nkv * T, rk).float()
        self._mark_memory(
            "quantize_to_chunk.k_flat.after",
            T=int(T),
            shape=tuple(k_flat.shape),
            dtype=str(k_flat.dtype).replace("torch.", ""),
        )
        self._mark_memory("quantize_to_chunk.k_quant.before", T=int(T))
        k_qx = self._tq_k_quantizer.quantize(
            k_flat,
            logical_shape=(nkv, T, rk),
        )
        self._mark_memory("quantize_to_chunk.k_quant.after", T=int(T))
        del k_flat
        k_norms = k_new.float().norm(dim=2).to(torch.float16)
        self._mark_memory(
            "quantize_to_chunk.k_norms.after",
            T=int(T),
            shape=tuple(k_norms.shape),
            dtype=str(k_norms.dtype).replace("torch.", ""),
        )
        v_flat = v_new.reshape(nkv * T, rv).float()
        self._mark_memory(
            "quantize_to_chunk.v_flat.after",
            T=int(T),
            shape=tuple(v_flat.shape),
            dtype=str(v_flat.dtype).replace("torch.", ""),
        )
        self._mark_memory("quantize_to_chunk.v_quant.before", T=int(T))
        v_qx = self._tq_v_quantizer.quantize(
            v_flat,
            logical_shape=(nkv, T, rv),
        )
        self._mark_memory("quantize_to_chunk.v_quant.after", T=int(T))
        del v_flat
        return _QuantChunk(k_qx, v_qx, T, k_norms)

    def _ensure_recent_buffer(self, k_new: torch.Tensor, v_new: torch.Tensor) -> None:
        if self.recent_window <= 0:
            return

        k_shape = (self.num_key_value_heads, self.recent_window, self.r_k)
        v_shape = (self.num_key_value_heads, self.recent_window, self.r_v)
        needs_alloc = (
            self._recent_k_buffer is None
            or self._recent_v_buffer is None
            or tuple(self._recent_k_buffer.shape) != k_shape
            or tuple(self._recent_v_buffer.shape) != v_shape
            or self._recent_k_buffer.device != k_new.device
            or self._recent_v_buffer.device != v_new.device
            or self._recent_k_buffer.dtype != k_new.dtype
            or self._recent_v_buffer.dtype != v_new.dtype
        )
        if needs_alloc:
            self._recent_k_buffer = torch.empty(k_shape, device=k_new.device, dtype=k_new.dtype)
            self._recent_v_buffer = torch.empty(v_shape, device=v_new.device, dtype=v_new.dtype)
            self._recent_start = 0
            self._recent_count = 0

    def _get_recent_from_buffer(self, buf: torch.Tensor | None) -> torch.Tensor | None:
        if buf is None or self._recent_count == 0:
            return None
        end = self._recent_start + self._recent_count
        if end <= self.recent_window:
            return buf[:, self._recent_start:end, :]
        tail = buf[:, self._recent_start:, :]
        head = buf[:, :end % self.recent_window, :]
        return torch.cat([tail, head], dim=1)

    def _get_recent_k(self) -> torch.Tensor | None:
        return self._get_recent_from_buffer(self._recent_k_buffer)

    def _get_recent_v(self) -> torch.Tensor | None:
        return self._get_recent_from_buffer(self._recent_v_buffer)

    def _write_recent_contiguous(self, k_keep: torch.Tensor, v_keep: torch.Tensor) -> None:
        T_keep = k_keep.shape[1]
        if self.recent_window <= 0:
            raise RuntimeError("_write_recent_contiguous requires recent_window > 0")
        if T_keep > self.recent_window:
            raise RuntimeError(f"T_keep={T_keep} exceeds recent_window={self.recent_window}")

        self._ensure_recent_buffer(k_keep, v_keep)
        if T_keep > 0:
            self._recent_k_buffer[:, :T_keep, :].copy_(k_keep)
            self._recent_v_buffer[:, :T_keep, :].copy_(v_keep)
        self._recent_start = 0
        self._recent_count = T_keep

    def _append_demoted_token_to_archive(self, k_old: torch.Tensor, v_old: torch.Tensor) -> None:
        chunk = self._quantize_to_chunk(k_old.unsqueeze(0), v_old.unsqueeze(0))
        self._append_or_merge_archive_chunk(chunk)

    def _quant_cache_append(self, k_lat: torch.Tensor, v_lat: torch.Tensor) -> None:
        if self.recent_window <= 0:
            self._quant_cache_append_to_archive(k_lat, v_lat)
            return

        k_new = k_lat[0].detach()
        v_new = v_lat[0].detach()
        self._ensure_recent_buffer(k_new, v_new)

        T = k_new.shape[1]
        if self._recent_count == 0 and T > self.recent_window:
            n_archive = T - self.recent_window
            archive_chunk = self._quantize_to_chunk(
                k_new[:, :n_archive, :].unsqueeze(0),
                v_new[:, :n_archive, :].unsqueeze(0),
            )
            self._append_or_merge_archive_chunk(archive_chunk)
            self._write_recent_contiguous(
                k_new[:, n_archive:, :],
                v_new[:, n_archive:, :],
            )
            return

        for t in range(T):
            if self._recent_count == self.recent_window:
                old_pos = self._recent_start
                k_old = self._recent_k_buffer[:, old_pos:old_pos + 1, :]
                v_old = self._recent_v_buffer[:, old_pos:old_pos + 1, :]
                self._append_demoted_token_to_archive(k_old, v_old)
                self._recent_start = (self._recent_start + 1) % self.recent_window
                self._recent_count -= 1

            write_pos = (self._recent_start + self._recent_count) % self.recent_window
            self._recent_k_buffer[:, write_pos:write_pos + 1, :].copy_(k_new[:, t:t + 1, :])
            self._recent_v_buffer[:, write_pos:write_pos + 1, :].copy_(v_new[:, t:t + 1, :])
            self._recent_count += 1

    def _quant_cache_demote(self) -> None:
        # Ring-buffer append demotes one oldest token before overwrite.
        return

    def _quant_cache_append_to_archive(self, k_lat: torch.Tensor, v_lat: torch.Tensor) -> None:
        chunk = self._quantize_to_chunk(k_lat, v_lat)
        self._append_or_merge_archive_chunk(chunk)

    def _quant_cache_append_latent(self, k_lat: torch.Tensor, v_lat: torch.Tensor) -> None:
        if self.recent_window <= 0:
            self._quant_cache_append_to_archive(k_lat, v_lat)
            return
        self._quant_cache_append(k_lat, v_lat)

    @staticmethod
    def _merge_rows_by_head(
        old_rows: torch.Tensor,
        new_rows: torch.Tensor,
        nkv: int,
        old_T: int,
        new_T: int,
    ) -> torch.Tensor:
        if old_rows.shape[0] != nkv * old_T:
            raise RuntimeError(
                f"old_rows first dim {old_rows.shape[0]} != nkv*old_T ({nkv * old_T})"
            )
        if new_rows.shape[0] != nkv * new_T:
            raise RuntimeError(
                f"new_rows first dim {new_rows.shape[0]} != nkv*new_T ({nkv * new_T})"
            )
        if old_rows.shape[1:] != new_rows.shape[1:]:
            raise RuntimeError(
                f"row payload shape mismatch: old={tuple(old_rows.shape[1:])}, "
                f"new={tuple(new_rows.shape[1:])}"
            )

        payload_shape = old_rows.shape[1:]
        old_by_head = old_rows.reshape(nkv, old_T, *payload_shape)
        new_by_head = new_rows.reshape(nkv, new_T, *payload_shape)
        merged = torch.cat([old_by_head, new_by_head], dim=1)
        return merged.reshape(nkv * (old_T + new_T), *payload_shape)

    @staticmethod
    def _check_quantized_compatible(old_qx, new_qx, dim: int) -> None:
        if type(old_qx) is not type(new_qx):
            raise RuntimeError(
                f"quantized tensor type mismatch: {type(old_qx).__name__} vs {type(new_qx).__name__}"
            )
        if old_qx.shape_orig[-1] != dim or new_qx.shape_orig[-1] != dim:
            raise RuntimeError(
                f"quantized dim mismatch: old={old_qx.shape_orig[-1]}, "
                f"new={new_qx.shape_orig[-1]}, expected={dim}"
            )
        if old_qx.bits != new_qx.bits:
            raise RuntimeError(f"bits mismatch: old={old_qx.bits}, new={new_qx.bits}")
        if old_qx.group_size != new_qx.group_size:
            raise RuntimeError(
                f"group_size mismatch: old={old_qx.group_size}, new={new_qx.group_size}"
            )
        old_rot = old_qx.rotation
        new_rot = new_qx.rotation
        if (
            old_rot is not None
            and new_rot is not None
            and not torch.equal(old_rot, new_rot.to(old_rot.device, old_rot.dtype))
        ):
            raise RuntimeError("rotation mismatch between quantized tensors")

    @staticmethod
    def _quantized_row_count(qx) -> int:
        if hasattr(qx, "mse"):
            return qx.mse.q.shape[0]
        return qx.q.shape[0]

    @staticmethod
    def _token_block_row_indices(
        qx,
        nkv: int,
        dim: int,
        start: int,
        end: int,
    ) -> tuple[torch.Tensor, int]:
        _, n_tokens, _ = HAWPAttention._get_logical_nkv_T_dim(qx, nkv, dim)
        if not (0 <= start <= end <= n_tokens):
            raise RuntimeError(
                f"Invalid token block [{start}, {end}) for archive length {n_tokens}"
            )
        block_len = end - start
        if hasattr(qx, "mse"):
            device = qx.mse.q.device
        else:
            device = qx.q.device
        base = torch.arange(nkv, device=device).unsqueeze(1) * n_tokens
        offsets = torch.arange(start, end, device=device).unsqueeze(0)
        return (base + offsets).reshape(-1), block_len

    @staticmethod
    def _slice_quantized_token_block(qx, nkv: int, dim: int, start: int, end: int):
        from hawp_laq.runtime.turboquant import TurboQuantProdResult, TurboQuantizedTensor

        rows, block_len = HAWPAttention._token_block_row_indices(qx, nkv, dim, start, end)
        logical_shape = (nkv, block_len, dim)

        if isinstance(qx, TurboQuantProdResult):
            mse = qx.mse
            sliced_mse = TurboQuantizedTensor(
                q=mse.q.index_select(0, rows),
                scale=mse.scale.index_select(0, rows),
                zero_point=mse.zero_point.index_select(0, rows),
                rotation=mse.rotation,
                shape_orig=(nkv * block_len, dim),
                bits=mse.bits,
                group_size=mse.group_size,
                logical_shape=logical_shape,
            )
            return TurboQuantProdResult(
                mse=sliced_mse,
                residual_sign=qx.residual_sign.index_select(0, rows),
                residual_norm=qx.residual_norm.index_select(0, rows),
                shape_orig=(nkv * block_len, dim),
                dim=dim,
                logical_shape=logical_shape,
            )

        if isinstance(qx, TurboQuantizedTensor):
            return TurboQuantizedTensor(
                q=qx.q.index_select(0, rows),
                scale=qx.scale.index_select(0, rows),
                zero_point=qx.zero_point.index_select(0, rows),
                rotation=qx.rotation,
                shape_orig=(nkv * block_len, dim),
                bits=qx.bits,
                group_size=qx.group_size,
                logical_shape=logical_shape,
            )

        raise RuntimeError(f"Unsupported quantized tensor type: {type(qx).__name__}")

    @staticmethod
    def _get_logical_nkv_T_dim(qx, nkv: int, dim: int) -> tuple[int, int, int]:
        logical_shape = getattr(qx, "logical_shape", None)
        if logical_shape is not None:
            if len(logical_shape) != 3:
                raise RuntimeError(f"logical_shape must be 3-D [nkv,T,dim], got {logical_shape}")
            q_nkv, q_T, q_dim = logical_shape
            if q_nkv != nkv or q_dim != dim:
                raise RuntimeError(
                    f"logical_shape mismatch: got {logical_shape}, expected nkv={nkv}, dim={dim}"
                )
            if HAWPAttention._quantized_row_count(qx) != nkv * q_T:
                raise RuntimeError(
                    f"logical_shape rows {nkv * q_T} do not match quantized rows "
                    f"{HAWPAttention._quantized_row_count(qx)}"
                )
            return q_nkv, q_T, q_dim

        rows = HAWPAttention._quantized_row_count(qx)
        if rows % nkv != 0:
            raise RuntimeError(f"Cannot infer token count from rows={rows}, nkv={nkv}")
        if qx.shape_orig[-1] != dim:
            raise RuntimeError(
                f"shape_orig dim mismatch: got {qx.shape_orig[-1]}, expected={dim}"
            )
        return nkv, rows // nkv, dim

    @staticmethod
    def _merge_quantized_by_head(old_qx, new_qx, nkv: int, dim: int):
        from hawp_laq.runtime.turboquant import TurboQuantProdResult, TurboQuantizedTensor

        _, old_T, _ = HAWPAttention._get_logical_nkv_T_dim(old_qx, nkv, dim)
        _, new_T, _ = HAWPAttention._get_logical_nkv_T_dim(new_qx, nkv, dim)
        merged_logical_shape = (nkv, old_T + new_T, dim)

        if isinstance(old_qx, TurboQuantProdResult):
            if not isinstance(new_qx, TurboQuantProdResult):
                raise RuntimeError("quantized tensor type mismatch for TurboQuantProdResult merge")
            if old_qx.dim != dim or new_qx.dim != dim:
                raise RuntimeError(
                    f"TurboQuantProd dim mismatch: old={old_qx.dim}, new={new_qx.dim}, expected={dim}"
                )
            old_mse = old_qx.mse
            new_mse = new_qx.mse
            HAWPAttention._check_quantized_compatible(old_mse, new_mse, dim)
            merged_mse = TurboQuantizedTensor(
                q=HAWPAttention._merge_rows_by_head(old_mse.q, new_mse.q, nkv, old_T, new_T),
                scale=HAWPAttention._merge_rows_by_head(old_mse.scale, new_mse.scale, nkv, old_T, new_T),
                zero_point=HAWPAttention._merge_rows_by_head(old_mse.zero_point, new_mse.zero_point, nkv, old_T, new_T),
                shape_orig=(nkv * (old_T + new_T), dim),
                bits=old_mse.bits,
                group_size=old_mse.group_size,
                rotation=old_mse.rotation if old_mse.rotation is not None else new_mse.rotation,
                logical_shape=merged_logical_shape,
            )
            return TurboQuantProdResult(
                mse=merged_mse,
                residual_sign=HAWPAttention._merge_rows_by_head(
                    old_qx.residual_sign, new_qx.residual_sign, nkv, old_T, new_T,
                ),
                residual_norm=HAWPAttention._merge_rows_by_head(
                    old_qx.residual_norm, new_qx.residual_norm, nkv, old_T, new_T,
                ),
                dim=dim,
                shape_orig=(nkv * (old_T + new_T), dim),
                logical_shape=merged_logical_shape,
            )

        if isinstance(old_qx, TurboQuantizedTensor):
            if not isinstance(new_qx, TurboQuantizedTensor):
                raise RuntimeError("quantized tensor type mismatch for TurboQuantizedTensor merge")
            HAWPAttention._check_quantized_compatible(old_qx, new_qx, dim)
            return TurboQuantizedTensor(
                q=HAWPAttention._merge_rows_by_head(old_qx.q, new_qx.q, nkv, old_T, new_T),
                scale=HAWPAttention._merge_rows_by_head(old_qx.scale, new_qx.scale, nkv, old_T, new_T),
                zero_point=HAWPAttention._merge_rows_by_head(old_qx.zero_point, new_qx.zero_point, nkv, old_T, new_T),
                shape_orig=(nkv * (old_T + new_T), dim),
                bits=old_qx.bits,
                group_size=old_qx.group_size,
                rotation=old_qx.rotation if old_qx.rotation is not None else new_qx.rotation,
                logical_shape=merged_logical_shape,
            )

        raise RuntimeError(f"Unsupported quantized tensor type: {type(old_qx).__name__}")

    @staticmethod
    def _merge_quantized(old_qx, new_qx, nkv, dim):
        return HAWPAttention._merge_quantized_by_head(old_qx, new_qx, nkv, dim)

    def _merge_chunks_by_head(self, old_chunk: _QuantChunk, new_chunk: _QuantChunk) -> _QuantChunk:
        nkv = self.num_key_value_heads
        _, old_T, _ = self._get_logical_nkv_T_dim(old_chunk.k_qx, nkv, self.r_k)
        _, new_T, _ = self._get_logical_nkv_T_dim(new_chunk.k_qx, nkv, self.r_k)
        _, old_v_T, _ = self._get_logical_nkv_T_dim(old_chunk.v_qx, nkv, self.r_v)
        _, new_v_T, _ = self._get_logical_nkv_T_dim(new_chunk.v_qx, nkv, self.r_v)
        if old_T != old_v_T or new_T != new_v_T:
            raise RuntimeError(
                f"K/V token count mismatch while merging chunks: "
                f"K=({old_T},{new_T}) V=({old_v_T},{new_v_T})"
            )
        k_qx = self._merge_quantized_by_head(old_chunk.k_qx, new_chunk.k_qx, nkv, self.r_k)
        v_qx = self._merge_quantized_by_head(old_chunk.v_qx, new_chunk.v_qx, nkv, self.r_v)

        if old_chunk.k_norms is None and new_chunk.k_norms is None:
            k_norms = None
        elif old_chunk.k_norms is not None and new_chunk.k_norms is not None:
            k_norms = torch.cat([old_chunk.k_norms, new_chunk.k_norms], dim=1)
        else:
            raise RuntimeError("k_norms mismatch while merging archive chunks")

        return _QuantChunk(k_qx, v_qx, old_T + new_T, k_norms)

    def _append_or_merge_archive_chunk(self, new_chunk: _QuantChunk) -> None:
        if not self._quant_archive_chunks:
            self._quant_archive_chunks.append(new_chunk)
            return
        if len(self._quant_archive_chunks) == 1:
            self._quant_archive_chunks[0] = self._merge_chunks_by_head(
                self._quant_archive_chunks[0], new_chunk,
            )
            return

        # Scheduler/drop policies can mutate archive chunks; single-old-chunk
        # mode with hawp_quant_sched needs separate validation before use.
        raise RuntimeError(
            f"single archive chunk invariant violated: found {len(self._quant_archive_chunks)} chunks"
        )

    def _quant_cache_get_kv(self) -> tuple[torch.Tensor, torch.Tensor]:
        k_parts = []
        v_parts = []
        for chunk in self._quant_archive_chunks:
            k_deq = self._tq_k_quantizer.dequantize(chunk.k_qx)
            if getattr(chunk.k_qx, "logical_shape", None) is not None:
                k_deq = k_deq.reshape(chunk.k_qx.logical_shape)
            else:
                k_deq = k_deq.reshape(self.num_key_value_heads, chunk.n_tokens, self.r_k)
            v_deq = self._tq_v_quantizer.dequantize(chunk.v_qx)
            if getattr(chunk.v_qx, "logical_shape", None) is not None:
                v_deq = v_deq.reshape(chunk.v_qx.logical_shape)
            else:
                v_deq = v_deq.reshape(self.num_key_value_heads, chunk.n_tokens, self.r_v)
            k_parts.append(k_deq)
            v_parts.append(v_deq)
        recent_k = self._get_recent_k()
        recent_v = self._get_recent_v()
        if recent_k is not None:
            k_parts.append(recent_k.float())
            v_parts.append(recent_v.float())
        if not k_parts:
            return None, None
        return torch.cat(k_parts, dim=1), torch.cat(v_parts, dim=1)

    def drop_oldest_from_archive(self, n: int) -> int:
        if not self._quant_archive_chunks:
            return 0
        dropped = 0
        while self._quant_archive_chunks and dropped < n:
            first = self._quant_archive_chunks[0]
            can_drop = min(first.n_tokens, n - dropped)
            if can_drop >= first.n_tokens:
                self._quant_archive_chunks.pop(0)
                dropped += first.n_tokens
            else:
                remaining = first.n_tokens - can_drop
                k_deq = self._tq_k_quantizer.dequantize(first.k_qx).reshape(
                    self.num_key_value_heads, first.n_tokens, self.r_k,
                )[:, can_drop:, :]
                v_deq = self._tq_v_quantizer.dequantize(first.v_qx).reshape(
                    self.num_key_value_heads, first.n_tokens, self.r_v,
                )[:, can_drop:, :]
                nkv = self.num_key_value_heads
                k_flat = k_deq.reshape(nkv * remaining, self.r_k).float()
                v_flat = v_deq.reshape(nkv * remaining, self.r_v).float()
                new_k_qx = self._tq_k_quantizer.quantize(
                    k_flat,
                    logical_shape=(nkv, remaining, self.r_k),
                )
                new_v_qx = self._tq_v_quantizer.quantize(
                    v_flat,
                    logical_shape=(nkv, remaining, self.r_v),
                )
                new_k_norms = first.k_norms[:, can_drop:] if first.k_norms is not None else None
                self._quant_archive_chunks[0] = _QuantChunk(new_k_qx, new_v_qx, remaining, new_k_norms)
                dropped += can_drop
        return dropped

    def drop_least_important_from_archive(self, n: int) -> int:
        if not self._quant_archive_chunks:
            return 0
        total_archive = sum(c.n_tokens for c in self._quant_archive_chunks)
        drop_n = min(n, total_archive)
        if drop_n == 0:
            return 0
        all_norms = []
        for chunk in self._quant_archive_chunks:
            if chunk.k_norms is not None:
                all_norms.append(chunk.k_norms)
        if not all_norms:
            return self.drop_oldest_from_archive(drop_n)
        combined_norms = torch.cat(all_norms, dim=1)
        _, indices = combined_norms.sum(dim=0).sort()
        drop_positions = set(indices[:drop_n].tolist())
        new_chunks = []
        offset = 0
        for chunk in self._quant_archive_chunks:
            local_drop = []
            for i in range(chunk.n_tokens):
                if (offset + i) in drop_positions:
                    local_drop.append(i)
            offset += chunk.n_tokens
            if not local_drop:
                new_chunks.append(chunk)
                continue
            keep_indices = sorted(set(range(chunk.n_tokens)) - set(local_drop))
            if not keep_indices:
                continue
            keep_tensor = torch.tensor(keep_indices, device=chunk.k_norms.device if chunk.k_norms is not None else 'cpu')
            kept_k_norms = chunk.k_norms[:, keep_indices] if chunk.k_norms is not None else None
            k_raw_keep = self._tq_k_quantizer.dequantize(chunk.k_qx).reshape(
                self.num_key_value_heads, chunk.n_tokens, self.r_k,
            )[:, keep_indices, :]
            v_raw_keep = self._tq_v_quantizer.dequantize(chunk.v_qx).reshape(
                self.num_key_value_heads, chunk.n_tokens, self.r_v,
            )[:, keep_indices, :]
            nkv = self.num_key_value_heads
            new_T = len(keep_indices)
            k_flat = k_raw_keep.reshape(nkv * new_T, self.r_k).float()
            v_flat = v_raw_keep.reshape(nkv * new_T, self.r_v).float()
            new_k_qx = self._tq_k_quantizer.quantize(
                k_flat,
                logical_shape=(nkv, new_T, self.r_k),
            )
            new_v_qx = self._tq_v_quantizer.quantize(
                v_flat,
                logical_shape=(nkv, new_T, self.r_v),
            )
            new_chunks.append(_QuantChunk(new_k_qx, new_v_qx, new_T, kept_k_norms))
        self._quant_archive_chunks = new_chunks
        return drop_n

    def quant_cache_summary(self) -> dict:
        n_recent = self._recent_count
        n_archive = sum(c.n_tokens for c in self._quant_archive_chunks)

        recent_active_bytes = 0
        recent_alloc_bytes = 0
        if self._recent_k_buffer is not None:
            recent_active_bytes += n_recent * self.num_key_value_heads * self.r_k * self._recent_k_buffer.element_size()
            recent_alloc_bytes += self._recent_k_buffer.nelement() * self._recent_k_buffer.element_size()
        if self._recent_v_buffer is not None:
            recent_active_bytes += n_recent * self.num_key_value_heads * self.r_v * self._recent_v_buffer.element_size()
            recent_alloc_bytes += self._recent_v_buffer.nelement() * self._recent_v_buffer.element_size()

        archive_quant_bytes = 0
        for chunk in self._quant_archive_chunks:
            archive_quant_bytes += self._tq_k_quantizer.estimate_num_bytes(chunk.k_qx)
            archive_quant_bytes += self._tq_v_quantizer.estimate_num_bytes(chunk.v_qx)

        archive_meta_bytes = 0
        for chunk in self._quant_archive_chunks:
            if chunk.k_norms is not None:
                archive_meta_bytes += chunk.k_norms.nelement() * chunk.k_norms.element_size()

        total_runtime_bytes = recent_alloc_bytes + archive_quant_bytes + archive_meta_bytes

        return {
            "layer": self.layer_idx,
            "recent_tokens": n_recent,
            "archive_tokens": n_archive,
            "recent_fp_bytes": recent_active_bytes,
            "recent_active_bytes": recent_active_bytes,
            "recent_alloc_bytes": recent_alloc_bytes,
            "archive_quant_bytes": archive_quant_bytes,
            "archive_meta_bytes": archive_meta_bytes,
            "total_runtime_bytes": total_runtime_bytes,
            "compressed_storage_bytes": archive_quant_bytes,
        }

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        bsz, q_len, _ = hidden_states.size()
        parent_use_cache = self._consume_opt_parent_use_cache()
        effective_use_cache = use_cache or parent_use_cache
        low_rank_path = self._is_low_rank or self.use_cache_manager

        if self.is_opt:
            query_states = self.q_proj(hidden_states) * self.scaling
        else:
            query_states = self.q_proj(hidden_states)

        key_states = self.k_proj(hidden_states)
        value_states = None
        v_lat_fused = None
        if (
            low_rank_path
            and self._calib_callback is None
            and self.v_lat_proj is not None
            and getattr(self, "use_prefill_fused_v_latent", True)
            and not torch.is_grad_enabled()
        ):
            v_lat_fused = self._project_hidden_to_v_latent(hidden_states, q_len=q_len)
        else:
            value_states = self._project_hidden_to_full_v(hidden_states)
        self._mark_memory(
            "full_qkv.proj.after",
            q_len=int(q_len),
            q_shape=tuple(query_states.shape),
            k_shape=tuple(key_states.shape),
            v_shape=tuple(value_states.shape) if value_states is not None else None,
            v_lat_shape=tuple(v_lat_fused.shape) if v_lat_fused is not None else None,
            v_fused=bool(v_lat_fused is not None),
            is_prefill=bool(q_len > 1),
        )

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        if value_states is not None:
            value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        self._mark_memory(
            "full_qkv.view.after",
            q_len=int(q_len),
            q_shape=tuple(query_states.shape),
            k_shape=tuple(key_states.shape),
            v_shape=tuple(value_states.shape) if value_states is not None else None,
            v_lat_shape=tuple(v_lat_fused.shape) if v_lat_fused is not None else None,
            v_fused=bool(v_lat_fused is not None),
            is_prefill=bool(q_len > 1),
        )

        if self._use_rope:
            if position_embeddings is not None:
                cos, sin = position_embeddings
            else:
                if position_ids is not None:
                    seq_len_for_rope = position_ids.max().item() + 1
                else:
                    seq_len_for_rope = q_len
                cos, sin = self.rotary_emb(key_states, seq_len=seq_len_for_rope)
                if position_ids is not None:
                    cos = cos[position_ids].unsqueeze(1)
                    sin = sin[position_ids].unsqueeze(1)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
            self._mark_memory(
                "full_qkv.rope.after",
                q_len=int(q_len),
                q_shape=tuple(query_states.shape),
                k_shape=tuple(key_states.shape),
                v_shape=tuple(value_states.shape) if value_states is not None else None,
                v_lat_shape=tuple(v_lat_fused.shape) if v_lat_fused is not None else None,
                v_fused=bool(v_lat_fused is not None),
                is_prefill=bool(q_len > 1),
            )

        if self._calib_callback is not None:
            self._calib_callback(self.layer_idx, query_states, key_states, value_states)

        if low_rank_path:
            q_lat, k_lat, v_lat, pk_down, pv_down = self._project_full_qkv_to_latent(
                query_states,
                key_states,
                value_states,
                v_lat=v_lat_fused,
                q_len=q_len,
            )
            query_dtype = query_states.dtype
            kv_seq_len = key_states.shape[-2]
            del query_states, key_states, value_states, v_lat_fused
            self._mark_memory(
                "low_rank.full_qkv.release.after",
                q_len=int(q_len),
                is_prefill=bool(q_len > 1),
            )
            return self._forward_low_rank(
                None, None, None,
                attention_mask, past_key_value, effective_use_cache,
                cache_position,
                precomputed_latents=(
                    q_lat,
                    k_lat,
                    v_lat,
                    pk_down,
                    pv_down,
                    bsz,
                    q_len,
                    kv_seq_len,
                    query_dtype,
                ),
                **kwargs,
            )

        if past_key_value is not None:
            if hasattr(past_key_value, "update"):
                cache_kwargs = {}
                if cache_position is not None:
                    cache_kwargs["cache_position"] = cache_position
                key_states, value_states = past_key_value.update(
                    key_states, value_states, self.layer_idx, cache_kwargs
                )
            else:
                key_states = torch.cat([past_key_value[0], key_states], dim=2)
                value_states = torch.cat([past_key_value[1], value_states], dim=2)

        raw_key_for_cache = key_states
        raw_value_for_cache = value_states

        key_states = self._apply_pk(key_states)
        value_states = self._apply_pv(value_states)

        key_states = self._repeat_kv(key_states)
        value_states = self._repeat_kv(value_states)

        if self.is_opt:
            attn_output, attn_weights = self._opt_attn_forward(
                query_states, key_states, value_states, attention_mask, **kwargs,
            )
        else:
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
            causal_mask = _prepare_attention_mask(
                attention_mask, q_len, key_states.shape[-2], query_states.device, query_states.dtype,
            )
            if causal_mask is not None:
                attn_weights = attn_weights + causal_mask
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        if self.is_opt:
            if past_key_value is not None and hasattr(past_key_value, "update"):
                return attn_output, attn_weights, past_key_value
            if effective_use_cache:
                return attn_output, attn_weights, (raw_key_for_cache, raw_value_for_cache)
            if past_key_value is not None:
                return attn_output, attn_weights, past_key_value
            return attn_output, attn_weights, (raw_key_for_cache, raw_value_for_cache)
        if effective_use_cache:
            if past_key_value is not None and hasattr(past_key_value, "update"):
                past_kv = past_key_value
            else:
                past_kv = (raw_key_for_cache, raw_value_for_cache)
        else:
            past_kv = None
        return self._format_attn_output(attn_output, None, past_kv)

    def _opt_attn_forward(self, query_states, key_states, value_states, attention_mask, **kwargs):
        attn_output, attn_weights = self._eager_attn(
            self, query_states, key_states, value_states, attention_mask,
            dropout=0.0, scaling=1.0,
        )
        return attn_output, attn_weights

    @staticmethod
    def _eager_attn(module, query, key, value, attention_mask, dropout=0.0, scaling=1.0, **kwargs):
        attn_weights = torch.matmul(query, key.transpose(2, 3))
        if scaling != 1.0:
            attn_weights = attn_weights * scaling
        causal_mask = _prepare_attention_mask(
            attention_mask, query.shape[2], key.shape[2], query.device, query.dtype,
        )
        if causal_mask is not None:
            attn_weights = attn_weights + causal_mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        if dropout > 0.0:
            attn_weights = F.dropout(attn_weights, p=dropout)
        attn_output = torch.matmul(attn_weights, value)
        return attn_output, attn_weights

    def _project_full_qkv_to_latent(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: Optional[torch.Tensor],
        *,
        v_lat: Optional[torch.Tensor] = None,
        q_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self._mark_memory(
            "low_rank.full_qkv.entry",
            q_len=int(q_len),
            q_shape=tuple(query_states.shape),
            k_shape=tuple(key_states.shape),
            v_shape=tuple(value_states.shape) if value_states is not None else None,
            v_lat_shape=tuple(v_lat.shape) if v_lat is not None else None,
            v_fused=bool(v_lat is not None),
            is_prefill=bool(q_len > 1),
        )

        pk_down = self.p_k[:, :self.r_k].to(device=query_states.device, dtype=query_states.dtype)
        pv_down = self.p_v[:, :self.r_v].to(device=query_states.device, dtype=query_states.dtype)
        self._mark_memory(
            "low_rank.proj_weights.after",
            q_len=int(q_len),
            is_prefill=bool(q_len > 1),
        )

        q_lat = query_states @ pk_down
        self._mark_memory("low_rank.q_lat.after", q_len=int(q_len), shape=tuple(q_lat.shape))
        k_lat = key_states @ pk_down
        self._mark_memory("low_rank.k_lat.after", q_len=int(q_len), shape=tuple(k_lat.shape))
        if v_lat is None:
            if value_states is None:
                raise RuntimeError("full V tensor is required when fused v_lat is not provided")
            v_lat = value_states @ pv_down
            self._mark_memory("low_rank.v_lat.after", q_len=int(q_len), shape=tuple(v_lat.shape), fused=False)
        else:
            self._mark_memory("low_rank.v_lat.after", q_len=int(q_len), shape=tuple(v_lat.shape), fused=True)
        self._mark_memory(
            "low_rank.latent_qkv.after",
            q_len=int(q_len),
            q_shape=tuple(q_lat.shape),
            k_shape=tuple(k_lat.shape),
            v_shape=tuple(v_lat.shape),
            is_prefill=bool(q_len > 1),
        )
        return q_lat, k_lat, v_lat, pk_down, pv_down

    def _project_hidden_to_v_latent(
        self,
        hidden_states: torch.Tensor,
        *,
        q_len: int,
    ) -> torch.Tensor:
        if self.v_lat_proj is None:
            fused_weight, fused_bias = self._get_fused_v_latent_params(hidden_states)
            v_lat = F.linear(hidden_states, fused_weight, fused_bias)
            source = "cached_weight"
        else:
            if self.v_lat_proj.weight.device != hidden_states.device:
                self.v_lat_proj.to(device=hidden_states.device, dtype=hidden_states.dtype)
            v_lat = self.v_lat_proj(hidden_states)
            source = "v_lat_proj"
        v_lat = v_lat.view(
            hidden_states.shape[0],
            q_len,
            self.num_key_value_heads,
            self.r_v,
        ).transpose(1, 2)
        self._mark_memory(
            "low_rank.v_lat.fused.after",
            q_len=int(q_len),
            shape=tuple(v_lat.shape),
            source=source,
        )
        return v_lat

    def _project_hidden_to_full_v(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.v_proj is None:
            raise RuntimeError("full V projection is unavailable; v_lat_proj inference path is required")
        if self.v_proj.weight.device != hidden_states.device:
            self.v_proj.to(device=hidden_states.device, dtype=hidden_states.dtype)
            self._full_v_proj_offloaded = False
        return self.v_proj(hidden_states)

    def _get_fused_v_latent_params(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        weight = self.v_proj.weight
        bias = getattr(self.v_proj, "bias", None)
        cache_key = (
            weight.data_ptr(),
            getattr(weight, "_version", 0),
            None if bias is None else bias.data_ptr(),
            None if bias is None else getattr(bias, "_version", 0),
            self.p_v.data_ptr(),
            getattr(self.p_v, "_version", 0),
            hidden_states.device,
            hidden_states.dtype,
            self.num_key_value_heads,
            self.head_dim,
            self.hidden_size,
            self.r_v,
        )
        if (
            self._fused_v_lat_cache_key == cache_key
            and self._fused_v_lat_weight is not None
            and self._fused_v_lat_weight.device == hidden_states.device
            and self._fused_v_lat_weight.dtype == hidden_states.dtype
        ):
            return self._fused_v_lat_weight, self._fused_v_lat_bias

        with torch.no_grad():
            v_weight = weight.detach().to(device=hidden_states.device, dtype=hidden_states.dtype)
            v_weight = v_weight.view(self.num_key_value_heads, self.head_dim, self.hidden_size)
            pv_down = self.p_v.detach()[:, :self.r_v].to(
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
            fused_weight = torch.einsum("odh,dr->orh", v_weight, pv_down).reshape(
                self.num_key_value_heads * self.r_v,
                self.hidden_size,
            ).contiguous()

            fused_bias = None
            if bias is not None:
                v_bias = bias.detach().to(device=hidden_states.device, dtype=hidden_states.dtype).view(
                    self.num_key_value_heads,
                    self.head_dim,
                )
                fused_bias = torch.einsum("od,dr->or", v_bias, pv_down).reshape(
                    self.num_key_value_heads * self.r_v,
                ).contiguous()

        self._fused_v_lat_weight = fused_weight
        self._fused_v_lat_bias = fused_bias
        self._fused_v_lat_cache_key = cache_key
        self._mark_memory(
            "low_rank.v_lat.fused_weight.cache_update.after",
            shape=tuple(fused_weight.shape),
            has_bias=bool(fused_bias is not None),
        )
        return fused_weight, fused_bias

    def _forward_low_rank(
        self,
        query_states: Optional[torch.Tensor],
        key_states: Optional[torch.Tensor],
        value_states: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value,
        effective_use_cache: bool,
        cache_position: Optional[torch.Tensor],
        precomputed_latents: Optional[tuple] = None,
        **kwargs,
    ):
        if precomputed_latents is None:
            if query_states is None or key_states is None or value_states is None:
                raise RuntimeError("full Q/K/V tensors are required when latents are not precomputed")
            bsz = query_states.size(0)
            q_len = query_states.size(2)
            kv_seq_len = key_states.shape[-2]
            query_dtype = query_states.dtype
            q_lat, k_lat, v_lat, pk_down, pv_down = self._project_full_qkv_to_latent(
                query_states,
                key_states,
                value_states,
                q_len=q_len,
            )
            del query_states, key_states, value_states
            self._mark_memory(
                "low_rank.full_qkv.release.after",
                q_len=int(q_len),
                is_prefill=bool(q_len > 1),
            )
        else:
            (
                q_lat,
                k_lat,
                v_lat,
                pk_down,
                pv_down,
                bsz,
                q_len,
                kv_seq_len,
                query_dtype,
            ) = precomputed_latents

        use_internal_quant_cache = (
            self.use_cache_manager
            and self._tq_k_quantizer is not None
            and effective_use_cache
            and bsz == 1
        )

        if use_internal_quant_cache:
            has_archive = bool(self._quant_archive_chunks)
            recent_k = self._get_recent_k()
            recent_v = self._get_recent_v()
            has_recent = recent_k is not None

            logit_scale = self._compute_low_rank_logit_scale(q_lat)

            if (
                not self.is_opt
                and not has_archive
                and not has_recent
                and q_len > 1
                and _attention_mask_is_fully_visible(attention_mask, kv_seq_len)
            ):
                if (
                    getattr(self, "use_prefill_grouped_attention", True)
                    and self.num_key_value_groups > 1
                ):
                    self._mark_memory("prefill.fast_path.grouped_sdpa.before", q_len=int(q_len))
                    self._mark_memory("prefill.fast_path.sdpa.before", q_len=int(q_len))
                    attn_output_lat = self._compute_prefill_attention_grouped(
                        q_lat,
                        k_lat,
                        v_lat,
                        logit_scale,
                    )
                else:
                    self._mark_memory("prefill.fast_path.before_repeat", q_len=int(q_len))
                    k_lat_expanded = self._repeat_kv(k_lat)
                    self._mark_memory(
                        "prefill.fast_path.k_repeat.after",
                        q_len=int(q_len),
                        shape=tuple(k_lat_expanded.shape),
                    )
                    v_lat_expanded = self._repeat_kv(v_lat)
                    self._mark_memory(
                        "prefill.fast_path.v_repeat.after",
                        q_len=int(q_len),
                        shape=tuple(v_lat_expanded.shape),
                    )
                    self._mark_memory("prefill.fast_path.sdpa.before", q_len=int(q_len))
                    attn_output_lat = _scaled_dot_product_attention_with_logit_scale(
                        q_lat,
                        k_lat_expanded,
                        v_lat_expanded,
                        logit_scale=logit_scale,
                        is_causal=True,
                    )
                    del k_lat_expanded, v_lat_expanded
                    self._mark_memory("prefill.fast_path.repeat_kv.release.after", q_len=int(q_len))
                self._mark_memory(
                    "prefill.fast_path.sdpa.after",
                    q_len=int(q_len),
                    shape=tuple(attn_output_lat.shape),
                )

                if (
                    getattr(self, "use_prefill_fused_value_o_proj", True)
                    and self._can_use_fused_value_o_proj()
                ):
                    self._mark_memory("prefill.fast_path.fused_value_o_proj.before", q_len=int(q_len))
                    attn_output = self._decode_value_and_o_proj_chunked(
                        attn_output_lat,
                        pv_down,
                    )
                    self._mark_memory(
                        "prefill.fast_path.fused_value_o_proj.after",
                        q_len=int(q_len),
                        shape=tuple(attn_output.shape),
                    )
                    del attn_output_lat
                    self._mark_memory("prefill.fast_path.attn_output_lat.release.after", q_len=int(q_len))
                else:
                    attn_output = self._decode_value_latent(attn_output_lat, pv_down)
                    self._mark_memory(
                        "prefill.fast_path.value_decode.after",
                        q_len=int(q_len),
                        shape=tuple(attn_output.shape),
                    )
                    del attn_output_lat
                    self._mark_memory("prefill.fast_path.attn_output_lat.release.after", q_len=int(q_len))
                    attn_output = attn_output.transpose(1, 2).contiguous()
                    self._mark_memory(
                        "prefill.fast_path.transpose_contiguous.after",
                        q_len=int(q_len),
                        shape=tuple(attn_output.shape),
                    )
                    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
                    self._mark_memory(
                        "prefill.fast_path.reshape.after",
                        q_len=int(q_len),
                        shape=tuple(attn_output.shape),
                    )
                    attn_output = self.o_proj(attn_output)
                    self._mark_memory(
                        "prefill.fast_path.o_proj.after",
                        q_len=int(q_len),
                        shape=tuple(attn_output.shape),
                    )

                self._mark_memory("prefill.fast_path.cache_append.before", q_len=int(q_len))
                self._quant_cache_append_latent(k_lat, v_lat)
                self._mark_memory("prefill.fast_path.cache_append.after", q_len=int(q_len))
                del q_lat, k_lat, v_lat
                self._mark_memory("prefill.fast_path.latent_qkv.release.after", q_len=int(q_len))

                cache_passthrough = _cache_passthrough(past_key_value)
                return self._format_attn_output(attn_output, None, cache_passthrough)

            total_kv_len = self.n_archive_tokens + self._recent_count + q_len
            if (
                not self.is_opt
                and self.use_decode_blockwise_attention
                and has_archive
                and q_len == 1
                and _attention_mask_is_fully_visible(attention_mask, total_kv_len)
            ):
                attn_output_lat = self._compute_decode_attention_blockwise(
                    q_lat,
                    k_lat,
                    v_lat,
                    recent_k,
                    recent_v,
                    logit_scale,
                )

                attn_output = self._decode_value_latent(attn_output_lat.to(query_dtype), pv_down)
                attn_output = attn_output.transpose(1, 2).contiguous()
                attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
                attn_output = self.o_proj(attn_output)

                self._quant_cache_append_latent(k_lat, v_lat)

                cache_passthrough = _cache_passthrough(past_key_value)
                return self._format_attn_output(attn_output, None, cache_passthrough)

            logit_parts = []
            v_parts_for_attn = []

            if has_archive:
                if self._can_use_archive_k_ip_approx():
                    archive_logits = self._compute_archive_k_logits_approx(q_lat)
                else:
                    k_archive_deq = self._dequant_archive_k()
                    archive_logits = self._compute_archive_k_logits_dequant(q_lat, k_archive_deq)
                logit_parts.append(archive_logits.to(q_lat.dtype))
                v_parts_for_attn.append(self._dequant_archive_v().to(q_lat.dtype))

            if has_recent:
                recent_logits = self._compute_recent_k_logits(q_lat, recent_k)
                logit_parts.append(recent_logits)
                v_parts_for_attn.append(recent_v.to(q_lat.dtype))

            current_k_expanded = self._repeat_kv(k_lat)
            current_logits = torch.matmul(q_lat, current_k_expanded.transpose(2, 3))
            logit_parts.append(current_logits)
            v_parts_for_attn.append(v_lat[0].to(q_lat.dtype))

            attn_weights = torch.cat(logit_parts, dim=-1) * logit_scale
            v_full = torch.cat(v_parts_for_attn, dim=1)
            v_full_expanded = self._repeat_kv(v_full.unsqueeze(0).to(q_lat.device, q_lat.dtype))

            total_kv_len = attn_weights.shape[-1]
            causal_mask = _prepare_attention_mask(
                attention_mask, q_len, total_kv_len, q_lat.device, q_lat.dtype,
            )
            if causal_mask is not None:
                attn_weights = attn_weights + causal_mask

            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_dtype)
            attn_output_lat = torch.matmul(attn_weights, v_full_expanded)

            attn_output = self._decode_value_latent(attn_output_lat, pv_down)

            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
            attn_output = self.o_proj(attn_output)

            self._quant_cache_append_latent(k_lat, v_lat)

            cache_passthrough = _cache_passthrough(past_key_value)
            if self.is_opt:
                return attn_output, attn_weights, cache_passthrough
            return self._format_attn_output(attn_output, None, cache_passthrough)

        elif past_key_value is not None:
            if hasattr(past_key_value, "update"):
                cache_kwargs: dict = {}
                if cache_position is not None:
                    cache_kwargs["cache_position"] = cache_position
                k_lat, v_lat = past_key_value.update(
                    k_lat, v_lat, self.layer_idx, cache_kwargs,
                )
            else:
                past_k, past_v = past_key_value
                if past_k.shape[-1] == self.r_k:
                    k_lat = torch.cat([past_k, k_lat], dim=2)
                else:
                    if past_k.shape[-1] != self.head_dim:
                        raise ValueError(
                            f"past_k last dim {past_k.shape[-1]} != r_k ({self.r_k}) "
                            f"and != head_dim ({self.head_dim}), cannot auto-convert"
                        )
                    k_lat = torch.cat([past_k @ pk_down, k_lat], dim=2)
                if past_v.shape[-1] == self.r_v:
                    v_lat = torch.cat([past_v, v_lat], dim=2)
                else:
                    if past_v.shape[-1] != self.head_dim:
                        raise ValueError(
                            f"past_v last dim {past_v.shape[-1]} != r_v ({self.r_v}) "
                            f"and != head_dim ({self.head_dim}), cannot auto-convert"
                        )
                    v_lat = torch.cat([past_v @ pv_down, v_lat], dim=2)

        k_lat_expanded = self._repeat_kv(k_lat)
        v_lat_expanded = self._repeat_kv(v_lat)

        logit_scale = self._compute_low_rank_logit_scale(q_lat)
        attn_weights = torch.matmul(q_lat, k_lat_expanded.transpose(2, 3)) * logit_scale

        causal_mask = _prepare_attention_mask(
            attention_mask, q_len, k_lat_expanded.shape[-2], q_lat.device, q_lat.dtype,
        )
        if causal_mask is not None:
            attn_weights = attn_weights + causal_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_dtype)
        attn_output_lat = torch.matmul(attn_weights, v_lat_expanded)

        attn_output = self._decode_value_latent(attn_output_lat, pv_down)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        cache_passthrough = _cache_passthrough(past_key_value)
        if self.is_opt:
            if self.use_cache_manager and self._tq_k_quantizer is not None and effective_use_cache and bsz == 1:
                return attn_output, attn_weights, cache_passthrough
            if effective_use_cache:
                if past_key_value is not None and hasattr(past_key_value, "update"):
                    return attn_output, attn_weights, past_key_value
                return attn_output, attn_weights, (k_lat, v_lat)
            if past_key_value is not None:
                return attn_output, attn_weights, past_key_value
            return attn_output, attn_weights, (k_lat, v_lat)
        if self.use_cache_manager and self._tq_k_quantizer is not None and effective_use_cache and bsz == 1:
            return self._format_attn_output(attn_output, None, cache_passthrough)
        if effective_use_cache:
            if past_key_value is not None and hasattr(past_key_value, "update"):
                past_kv = past_key_value
            else:
                past_kv = (k_lat, v_lat)
        else:
            past_kv = None
        return self._format_attn_output(attn_output, None, past_kv)

    def _decode_value_latent(self, attn_output_lat: torch.Tensor, pv_down: torch.Tensor) -> torch.Tensor:
        if self.d_v is not None:
            d_v = self.d_v.to(device=attn_output_lat.device, dtype=attn_output_lat.dtype)
            return attn_output_lat @ d_v
        return attn_output_lat @ pv_down.T

    def _can_use_fused_value_o_proj(self) -> bool:
        weight = getattr(self.o_proj, "weight", None)
        if not isinstance(weight, torch.Tensor):
            return False
        if weight.dim() != 2 or tuple(weight.shape) != (self.hidden_size, self.hidden_size):
            return False
        return bool(weight.dtype.is_floating_point)

    def _decode_value_and_o_proj_chunked(
        self,
        attn_output_lat: torch.Tensor,
        pv_down: torch.Tensor,
    ) -> torch.Tensor:
        bsz, num_heads, q_len, rv = attn_output_lat.shape
        if num_heads != self.num_heads or rv != self.r_v:
            raise RuntimeError(
                "Fused value/o_proj shape mismatch: "
                f"attn_output_lat={tuple(attn_output_lat.shape)}, "
                f"num_heads={self.num_heads}, r_v={self.r_v}"
            )

        dtype = attn_output_lat.dtype
        device = attn_output_lat.device
        if self.d_v is not None:
            decode_weight = self.d_v.to(device=device, dtype=dtype)
        else:
            decode_weight = pv_down.T
            if decode_weight.device != device or decode_weight.dtype != dtype:
                decode_weight = decode_weight.to(device=device, dtype=dtype)

        o_weight = self.o_proj.weight
        if o_weight.device != device or o_weight.dtype != dtype:
            o_weight = o_weight.to(device=device, dtype=dtype)

        attn_output = attn_output_lat.new_empty((bsz, q_len, self.hidden_size))
        bias = getattr(self.o_proj, "bias", None)
        if bias is not None:
            bias = bias.to(device=device, dtype=dtype)
            attn_output.copy_(bias.view(1, 1, self.hidden_size).expand_as(attn_output))
        else:
            attn_output.zero_()
        self._mark_memory(
            "prefill.fused_value_o_proj.output_alloc.after",
            q_len=int(q_len),
            shape=tuple(attn_output.shape),
        )

        chunk_size = max(1, int(getattr(self, "prefill_fused_o_proj_chunk_size", 2048)))
        for head_idx in range(num_heads):
            head_start = head_idx * self.head_dim
            head_end = head_start + self.head_dim
            fused_weight = decode_weight @ o_weight[:, head_start:head_end].transpose(0, 1)
            fused_weight_t = fused_weight.transpose(0, 1).contiguous()
            for start in range(0, q_len, chunk_size):
                end = min(start + chunk_size, q_len)
                projected = F.linear(
                    attn_output_lat[:, head_idx, start:end, :],
                    fused_weight_t,
                    bias=None,
                )
                attn_output[:, start:end, :].add_(projected)
                del projected
            if head_idx == 0 or head_idx == num_heads - 1:
                self._mark_memory(
                    "prefill.fused_value_o_proj.head.after",
                    q_len=int(q_len),
                    head_idx=int(head_idx),
                    chunk_size=int(chunk_size),
                )
            del fused_weight, fused_weight_t

        return attn_output

    def _apply_pk(self, k: torch.Tensor) -> torch.Tensor:
        if self.r_k >= self.head_dim and not self.p_k.requires_grad:
            return k
        pk_down = self.p_k[:, :self.r_k].to(device=k.device, dtype=k.dtype)
        return k @ pk_down @ pk_down.T

    def _apply_pv(self, v: torch.Tensor) -> torch.Tensor:
        if self.r_v >= self.head_dim and not self.p_v.requires_grad:
            return v
        pv_down = self.p_v[:, :self.r_v].to(device=v.device, dtype=v.dtype)
        return v @ pv_down @ pv_down.T

    def load_projector_data(self, data: dict[str, torch.Tensor], strict: bool = True) -> None:
        p_k = data["p_k"]
        p_v = data["p_v"]

        if p_k.shape == (self.head_dim, self.head_dim):
            self.p_k.data.copy_(p_k.to(self.p_k.device, self.p_k.dtype))
        elif p_k.shape == (self.head_dim, self.r_k):
            self.p_k.data.zero_()
            self.p_k.data[:, :self.r_k].copy_(p_k.to(self.p_k.device, self.p_k.dtype))
        elif strict:
            raise ValueError(
                f"layer {self.layer_idx}: p_k shape {p_k.shape} incompatible "
                f"with expected ({self.head_dim},{self.head_dim}) or ({self.head_dim},{self.r_k})"
            )
        else:
            import warnings
            warnings.warn(
                f"layer {self.layer_idx}: p_k shape {tuple(p_k.shape)} incompatible "
                f"with expected ({self.head_dim},{self.head_dim}) or ({self.head_dim},{self.r_k}), skipping"
            )

        if p_v.shape == (self.head_dim, self.head_dim):
            self.p_v.data.copy_(p_v.to(self.p_v.device, self.p_v.dtype))
        elif p_v.shape == (self.head_dim, self.r_v):
            self.p_v.data.zero_()
            self.p_v.data[:, :self.r_v].copy_(p_v.to(self.p_v.device, self.p_v.dtype))
        elif strict:
            raise ValueError(
                f"layer {self.layer_idx}: p_v shape {p_v.shape} incompatible "
                f"with expected ({self.head_dim},{self.head_dim}) or ({self.head_dim},{self.r_v})"
            )
        else:
            import warnings
            warnings.warn(
                f"layer {self.layer_idx}: p_v shape {tuple(p_v.shape)} incompatible "
                f"with expected ({self.head_dim},{self.head_dim}) or ({self.head_dim},{self.r_v}), skipping"
            )

        d_v = data.get("d_v")
        if d_v is not None:
            if d_v.shape == (self.r_v, self.head_dim):
                self.d_v = d_v.to(self.p_v.device, self.p_v.dtype).contiguous()
            elif strict:
                raise ValueError(
                    f"layer {self.layer_idx}: d_v shape {d_v.shape} incompatible "
                    f"with expected ({self.r_v},{self.head_dim})"
                )
            else:
                import warnings
                warnings.warn(
                    f"layer {self.layer_idx}: d_v shape {tuple(d_v.shape)} incompatible "
                    f"with expected ({self.r_v},{self.head_dim}), falling back to p_v.T",
                    UserWarning,
                    stacklevel=2,
                )
                self.d_v = None
        else:
            self.d_v = None

        if "gamma" in data:
            self.gamma.data.copy_(data["gamma"].to(self.gamma.device, self.gamma.dtype))
        elif "gamma_v" in data:
            import warnings
            warnings.warn(
                f"layer {self.layer_idx}: projector.pt missing 'gamma', using "
                f"'gamma_v' as fallback. Consider retraining projectors.",
                UserWarning,
                stacklevel=2,
            )
            self.gamma.data.copy_(data["gamma_v"].to(self.gamma.device, self.gamma.dtype))
        elif "gamma_k" in data:
            import warnings
            warnings.warn(
                f"layer {self.layer_idx}: projector.pt missing 'gamma', using "
                f"'gamma_k' as fallback. Consider retraining projectors.",
                UserWarning,
                stacklevel=2,
            )
            self.gamma.data.copy_(data["gamma_k"].to(self.gamma.device, self.gamma.dtype))

    def _compute_low_rank_logit_scale(self, q_lat: torch.Tensor) -> torch.Tensor:
        """Compute the scale factor applied to low-rank logits.

        Handles three concerns:
          1. Temperature: sqrt(head_dim) vs sqrt(r_k)
          2. Gamma: off vs fixed vs learned
          3. OPT pre-scaling: query_states was already multiplied by
             1/sqrt(head_dim) in forward().  For ``logit_scale_mode="dh"``
             we want the standard 1/sqrt(d_h) temperature, so the net scale
             on q_lat should be 1 (undo the pre-scaling).  For
             ``logit_scale_mode="rk"`` we want 1/sqrt(r_k), so we undo the
             pre-scaling and apply sqrt(d_h)/sqrt(r_k).

        Returns:
            A scalar (0-dim) tensor representing the overall multiplier for
            ``q_lat @ k_lat^T``.
        """
        if self.logit_scale_mode == "dh":
            temp_scale = 1.0 / math.sqrt(self.head_dim)
        elif self.logit_scale_mode == "rk":
            temp_scale = 1.0 / math.sqrt(self.r_k)
        else:
            raise ValueError(
                f"Unknown logit_scale_mode='{self.logit_scale_mode}'. "
                f"Supported: 'dh', 'rk'"
            )

        if self.is_opt:
            opt_undo = math.sqrt(self.head_dim)
            scale_value = opt_undo * temp_scale
        else:
            scale_value = temp_scale

        scale = torch.as_tensor(scale_value, dtype=q_lat.dtype, device=q_lat.device)
        if self.gamma_mode == "learned":
            scale = scale * self.gamma.to(device=q_lat.device, dtype=q_lat.dtype).reshape(())
        elif self.gamma_mode == "fixed":
            if self.gamma_value is None:
                gamma = self.gamma.to(device=q_lat.device, dtype=q_lat.dtype).reshape(())
            else:
                gamma = torch.as_tensor(self.gamma_value, dtype=q_lat.dtype, device=q_lat.device)
            scale = scale * gamma

        return scale

    def _can_use_archive_k_ip_approx(self) -> bool:
        """Check whether the approx_inner_product fast path is available.

        Checks chunk count and quantizer type directly, without touching
        ``_single_archive_k_qx`` (which triggers dequantization for
        multi-chunk).
        """
        if not self.use_archive_k_ip_approx:
            return False
        if not self._quant_archive_chunks:
            return False
        if self._tq_k_quantizer is None:
            return False
        from hawp_laq.runtime.turboquant import TurboQuantProd
        return isinstance(self._tq_k_quantizer, TurboQuantProd)

    def _dequant_archive_k(self) -> torch.Tensor:
        parts = []
        for chunk in self._quant_archive_chunks:
            deq = self._tq_k_quantizer.dequantize(chunk.k_qx)
            if getattr(chunk.k_qx, "logical_shape", None) is not None:
                deq = deq.reshape(chunk.k_qx.logical_shape)
            else:
                deq = deq.reshape(self.num_key_value_heads, chunk.n_tokens, self.r_k)
            parts.append(deq)
        return torch.cat(parts, dim=1)

    def _dequant_archive_v(self) -> torch.Tensor:
        parts = []
        for chunk in self._quant_archive_chunks:
            deq = self._tq_v_quantizer.dequantize(chunk.v_qx)
            if getattr(chunk.v_qx, "logical_shape", None) is not None:
                deq = deq.reshape(chunk.v_qx.logical_shape)
            else:
                deq = deq.reshape(self.num_key_value_heads, chunk.n_tokens, self.r_v)
            parts.append(deq)
        return torch.cat(parts, dim=1)

    @staticmethod
    def _stream_softmax_block(
        logits: torch.Tensor,
        values: torch.Tensor,
        running_max: torch.Tensor,
        running_sum: torch.Tensor,
        running_out: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = logits.float()
        values = values.float()
        block_max = logits.max(dim=-1).values
        new_max = torch.maximum(running_max, block_max)
        old_scale = torch.exp(running_max - new_max)
        block_scale = torch.exp(block_max - new_max)
        exp_logits = torch.exp(logits - block_max.unsqueeze(-1))
        block_sum = exp_logits.sum(dim=-1)
        block_out = torch.einsum("ngt,ntr->ngr", exp_logits, values)
        running_out = running_out * old_scale.unsqueeze(-1) + block_out * block_scale.unsqueeze(-1)
        running_sum = running_sum * old_scale + block_sum * block_scale
        return new_max, running_sum, running_out

    def _compute_grouped_k_logits(
        self,
        q_grouped: torch.Tensor,
        k_block: torch.Tensor,
    ) -> torch.Tensor:
        return torch.einsum("ngr,ntr->ngt", q_grouped.float(), k_block.float())

    def _compute_prefill_attention_grouped(
        self,
        q_lat: torch.Tensor,
        k_lat: torch.Tensor,
        v_lat: torch.Tensor,
        logit_scale: torch.Tensor,
    ) -> torch.Tensor:
        """Run causal prefill attention per KV-head group without full K/V repeat."""
        if self.num_key_value_groups == 1:
            return _scaled_dot_product_attention_with_logit_scale(
                q_lat,
                k_lat,
                v_lat,
                logit_scale=logit_scale,
                is_causal=True,
            )

        bsz, num_heads, q_len, rk = q_lat.shape
        _, nkv, k_len, _ = k_lat.shape
        g = self.num_key_value_groups
        rv = v_lat.shape[-1]
        if nkv != self.num_key_value_heads or num_heads != nkv * g:
            raise RuntimeError(
                "Grouped prefill attention shape mismatch: "
                f"num_heads={num_heads}, num_key_value_heads={nkv}, groups={g}"
            )

        q_grouped = q_lat.reshape(bsz, nkv, g, q_len, rk)
        attn_output_lat = q_lat.new_empty((bsz, num_heads, q_len, rv))
        self._mark_memory(
            "prefill.grouped_sdpa.output_alloc.after",
            q_len=int(q_len),
            shape=tuple(attn_output_lat.shape),
        )

        for kv_idx in range(nkv):
            head_start = kv_idx * g
            head_end = head_start + g
            q_part = q_grouped[:, kv_idx, :, :, :]
            k_part = k_lat[:, kv_idx : kv_idx + 1, :, :].expand(bsz, g, k_len, rk)
            v_part = v_lat[:, kv_idx : kv_idx + 1, :, :].expand(bsz, g, k_len, rv)
            out_part = _scaled_dot_product_attention_with_logit_scale(
                q_part,
                k_part,
                v_part,
                logit_scale=logit_scale,
                is_causal=True,
            )
            attn_output_lat[:, head_start:head_end, :, :] = out_part
            if kv_idx == 0 or kv_idx == nkv - 1:
                self._mark_memory(
                    "prefill.grouped_sdpa.group.after",
                    q_len=int(q_len),
                    kv_idx=int(kv_idx),
                    shape=tuple(out_part.shape),
                )
            del q_part, k_part, v_part, out_part

        return attn_output_lat

    def _compute_archive_k_logits_block_approx_grouped(
        self,
        q_grouped: torch.Tensor,
        k_qx,
        block_len: int,
    ) -> torch.Tensor:
        nkv = self.num_key_value_heads
        g = self.num_key_value_groups
        rk = self.r_k
        x_hat = self._tq_k_quantizer.dequantize_mse(k_qx, logical=True)
        if x_hat.dim() == 2:
            x_hat = x_hat.reshape(nkv, block_len, rk)
        x_hat = x_hat.to(device=q_grouped.device, dtype=torch.float32)
        mse_ip = torch.einsum("ngr,ntr->ngt", q_grouped.float(), x_hat)

        sign_unpacked = unpack_bool(k_qx.residual_sign, rk)
        sign_unpacked = sign_unpacked.reshape(nkv, block_len, rk)
        sign_float = sign_unpacked.to(device=q_grouped.device, dtype=torch.float32)
        sign_float.mul_(2.0).sub_(1.0)
        scale = (k_qx.residual_norm / math.sqrt(rk)).reshape(nkv, block_len)
        scale = scale.to(device=q_grouped.device, dtype=torch.float32)
        sign_ip = torch.einsum("ngr,ntr->ngt", q_grouped.float(), sign_float)
        sign_ip = sign_ip * scale.unsqueeze(1)
        if g != q_grouped.shape[1]:
            raise RuntimeError("Grouped query shape mismatch while computing archive logits")
        return mse_ip + sign_ip

    def _compute_decode_attention_blockwise(
        self,
        q_lat: torch.Tensor,
        k_lat: torch.Tensor,
        v_lat: torch.Tensor,
        recent_k: torch.Tensor | None,
        recent_v: torch.Tensor | None,
        logit_scale: torch.Tensor,
    ) -> torch.Tensor:
        nkv = self.num_key_value_heads
        g = self.num_key_value_groups
        q_grouped = q_lat[0].reshape(nkv, g, 1, self.r_k)[:, :, 0, :]
        running_max = torch.full(
            (nkv, g), -float("inf"), device=q_lat.device, dtype=torch.float32,
        )
        running_sum = torch.zeros((nkv, g), device=q_lat.device, dtype=torch.float32)
        running_out = torch.zeros((nkv, g, self.r_v), device=q_lat.device, dtype=torch.float32)
        scale = logit_scale.to(device=q_lat.device, dtype=torch.float32)
        block_size = max(1, int(self.decode_archive_block_size))

        for chunk in self._quant_archive_chunks:
            _, n_tokens, _ = self._get_logical_nkv_T_dim(chunk.k_qx, nkv, self.r_k)
            for start in range(0, n_tokens, block_size):
                end = min(start + block_size, n_tokens)
                block_len = end - start
                k_qx = self._slice_quantized_token_block(chunk.k_qx, nkv, self.r_k, start, end)
                v_qx = self._slice_quantized_token_block(chunk.v_qx, nkv, self.r_v, start, end)

                if self._can_use_archive_k_ip_approx():
                    logits = self._compute_archive_k_logits_block_approx_grouped(
                        q_grouped, k_qx, block_len,
                    )
                else:
                    k_block = self._tq_k_quantizer.dequantize(k_qx).reshape(
                        nkv, block_len, self.r_k,
                    )
                    logits = self._compute_grouped_k_logits(q_grouped, k_block)

                v_block = self._tq_v_quantizer.dequantize(v_qx).reshape(
                    nkv, block_len, self.r_v,
                )
                running_max, running_sum, running_out = self._stream_softmax_block(
                    logits * scale,
                    v_block,
                    running_max,
                    running_sum,
                    running_out,
                )

        if recent_k is not None and recent_v is not None:
            recent_logits = self._compute_grouped_k_logits(q_grouped, recent_k)
            running_max, running_sum, running_out = self._stream_softmax_block(
                recent_logits * scale,
                recent_v,
                running_max,
                running_sum,
                running_out,
            )

        current_k = k_lat[0]
        current_v = v_lat[0]
        current_logits = self._compute_grouped_k_logits(q_grouped, current_k)
        running_max, running_sum, running_out = self._stream_softmax_block(
            current_logits * scale,
            current_v,
            running_max,
            running_sum,
            running_out,
        )

        attn_output = running_out / running_sum.clamp_min(1e-20).unsqueeze(-1)
        return attn_output.reshape(1, self.num_heads, 1, self.r_v).to(q_lat.dtype)

    def _compute_archive_k_logits_approx(
        self,
        q_lat: torch.Tensor,
    ) -> torch.Tensor:
        """Compute archive K logits via batch approx_inner_product.

        Eliminates the per-head Python loop by batch-dequantizing each chunk
        across all KV heads and using batched matrix multiply.

        Args:
            q_lat: [bsz, n_heads, q_len, r_k]

        Returns:
            archive_logits: [bsz, n_heads, q_len, T_archive]
        """
        if self.num_heads != self.num_key_value_heads * self.num_key_value_groups:
            raise RuntimeError(
                f"Head layout mismatch: num_heads={self.num_heads}, "
                f"num_key_value_heads={self.num_key_value_heads}, "
                f"num_key_value_groups={self.num_key_value_groups}"
            )
        if q_lat.shape[1] != self.num_heads:
            raise RuntimeError(
                f"q_lat head count mismatch: "
                f"expected {self.num_heads}, got {q_lat.shape[1]}"
            )
        if q_lat.shape[-1] != self.r_k:
            raise RuntimeError(
                f"q_lat latent dim mismatch: "
                f"expected r_k={self.r_k}, got {q_lat.shape[-1]}"
            )

        g = self.num_key_value_groups
        rk = self.r_k
        bsz = q_lat.shape[0]
        q_len = q_lat.shape[2]
        nkv = self.num_key_value_heads
        num_heads = self.num_heads

        if not self._quant_archive_chunks:
            return q_lat.new_empty(bsz, num_heads, q_len, 0)

        # ------------------------------------------------------------------
        # 1. 重组 query：从 [bsz, num_heads, q_len, rk]
        #    变为 [bsz, nkv, g*q_len, rk]
        # ------------------------------------------------------------------
        q_grouped = q_lat.reshape(bsz, nkv, g, q_len, rk)
        q_grouped = q_grouped.permute(0, 1, 3, 2, 4).contiguous()
        q_grouped = q_grouped.reshape(bsz, nkv, g * q_len, rk).float()

        # ------------------------------------------------------------------
        # 2. 逐个 chunk 处理，但在 chunk 内 batch 所有 heads
        # ------------------------------------------------------------------
        chunk_logits_list = []
        for chunk in self._quant_archive_chunks:
            T_chunk = chunk.n_tokens

            # --- Part A: MSE contribution ---
            x_hat = self._tq_k_quantizer.dequantize_mse(chunk.k_qx, logical=True)
            if x_hat.dim() == 2:
                x_hat = x_hat.reshape(nkv, T_chunk, rk)
            x_hat = x_hat.to(device=q_grouped.device, dtype=torch.float32)
            k_bmm = x_hat.unsqueeze(0)
            mse_ip = torch.matmul(q_grouped, k_bmm.transpose(-2, -1))
            # [bsz, nkv, g*q_len, T_chunk]

            # --- Part B: Residual contribution ---
            sign_unpacked = unpack_bool(chunk.k_qx.residual_sign, rk)
            sign_unpacked = sign_unpacked.reshape(nkv, T_chunk, rk)
            sign_float = sign_unpacked.to(device=q_grouped.device, dtype=torch.float32)
            sign_float.mul_(2.0).sub_(1.0)
            scale = (chunk.k_qx.residual_norm / math.sqrt(rk)).reshape(nkv, T_chunk)
            scale = scale.to(device=q_grouped.device, dtype=torch.float32)

            sign_bmm = sign_float.unsqueeze(0)
            sign_ip = torch.matmul(q_grouped, sign_bmm.transpose(-2, -1))
            sign_ip = sign_ip * scale.unsqueeze(0).unsqueeze(-2)

            # --- Combine MSE + Residual ---
            chunk_ip = mse_ip + sign_ip

            # ------------------------------------------------------------------
            # 3. 重组回标准 attention logits shape: [bsz, num_heads, q_len, T_chunk]
            # ------------------------------------------------------------------
            chunk_ip = chunk_ip.reshape(bsz, nkv, q_len, g, T_chunk)
            chunk_ip = chunk_ip.permute(0, 1, 3, 2, 4).contiguous()
            chunk_ip = chunk_ip.reshape(bsz, num_heads, q_len, T_chunk)

            chunk_logits_list.append(chunk_ip)

        return torch.cat(chunk_logits_list, dim=-1)

    def _compute_archive_k_logits_dequant(
        self,
        q_lat: torch.Tensor,
        k_archive_deq: torch.Tensor,
    ) -> torch.Tensor:
        """Compute archive K logits via dequantize-then-matmul fallback.

        Args:
            q_lat: [bsz, n_heads, q_len, r_k]
            k_archive_deq: [n_kv_heads, T_archive, r_k]  (already dequantized)

        Returns:
            archive_logits: [bsz, n_heads, q_len, T_archive]
        """
        k_archive = k_archive_deq.to(q_lat.dtype)
        k_expanded = self._repeat_kv(k_archive.unsqueeze(0))
        return torch.matmul(q_lat, k_expanded.transpose(2, 3))

    def _compute_recent_k_logits(
        self,
        q_lat: torch.Tensor,
        k_recent: torch.Tensor,
    ) -> torch.Tensor:
        """Compute recent K logits from high-precision recent keys.

        Args:
            q_lat: [bsz, n_heads, q_len, r_k]
            k_recent: [n_kv_heads, T_recent, r_k]

        Returns:
            recent_logits: [bsz, n_heads, q_len, T_recent]
        """
        k_expanded = self._repeat_kv(k_recent.unsqueeze(0))
        return torch.matmul(q_lat, k_expanded.transpose(2, 3))

    def _repeat_kv(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.num_key_value_groups == 1:
            return hidden_states
        bsz, num_kv_heads, seq_len, head_dim = hidden_states.shape
        hidden_states = hidden_states[:, :, None, :, :].expand(bsz, num_kv_heads, self.num_key_value_groups, seq_len, head_dim)
        return hidden_states.reshape(bsz, self.num_heads, seq_len, head_dim)

    @classmethod
    def from_attention(
        cls,
        attn_module,
        model=None,
        layer_idx: int | None = None,
        r_k: int | None = None,
        r_v: int | None = None,
        allow_default_full_rank: bool = False,
        logit_scale_mode: str = "rk",
        gamma_mode: str = "learned",
        gamma_value: float | None = None,
        use_archive_k_ip_approx: bool = True,
    ):
        config = _get_attn_config(attn_module, model)
        if layer_idx is None and attn_module is not None:
            layer_idx = getattr(attn_module, "layer_idx", None)
        instance = cls(config, layer_idx=layer_idx, r_k=r_k, r_v=r_v,
                       allow_default_full_rank=allow_default_full_rank,
                       logit_scale_mode=logit_scale_mode,
                       gamma_mode=gamma_mode,
                       gamma_value=gamma_value,
                       use_archive_k_ip_approx=use_archive_k_ip_approx,
                       _skip_linear_init=True)

        _proj_name_map = {
            "q_proj": ("q_proj", "query_proj"),
            "k_proj": ("k_proj", "key_proj"),
            "v_proj": ("v_proj", "value_proj"),
            "o_proj": ("o_proj", "out_proj"),
        }
        _src_modules = {}
        for std_name, alts in _proj_name_map.items():
            for alt in alts:
                mod = getattr(attn_module, alt, None)
                if mod is not None:
                    _src_modules[std_name] = mod
                    break

        ref_dtype = torch.float32
        ref_device = torch.device("cpu")
        if _src_modules:
            _ref_linear = next(iter(_src_modules.values()))
            ref_dtype = _resolve_compute_dtype(_ref_linear)
            ref_device = _ref_linear.weight.device

        for std_name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            if std_name in _src_modules:
                setattr(instance, std_name, _src_modules[std_name])
            else:
                use_bias = instance.is_opt and getattr(config, "enable_bias", True)
                logger.warning(
                    "HAWPAttention.from_attention: '%s' not found in source module, "
                    "falling back to randomly initialized nn.Linear. "
                    "This will produce incorrect outputs until trained.",
                    std_name,
                )
                if std_name in ("q_proj",):
                    fallback = nn.Linear(instance.hidden_size, instance.num_heads * instance.head_dim, bias=use_bias)
                elif std_name == "k_proj":
                    fallback = nn.Linear(instance.hidden_size, instance.num_key_value_heads * instance.head_dim, bias=use_bias)
                elif std_name == "v_proj":
                    fallback = nn.Linear(instance.hidden_size, instance.num_key_value_heads * instance.head_dim, bias=use_bias)
                elif std_name == "o_proj":
                    fallback = nn.Linear(instance.num_heads * instance.head_dim, instance.hidden_size, bias=use_bias)
                fallback = fallback.to(dtype=ref_dtype, device=ref_device)
                setattr(instance, std_name, fallback)

        if "q_proj" in _src_modules:
            target_device = _src_modules["q_proj"].weight.device
            target_dtype = _resolve_compute_dtype(_src_modules["q_proj"])

            instance.p_k.data.copy_(
                torch.eye(instance.head_dim, device=target_device, dtype=target_dtype)
            )
            instance.p_v.data.copy_(
                torch.eye(instance.head_dim, device=target_device, dtype=target_dtype)
            )
            instance.gamma.data.fill_(1.0)

            instance.p_k.data = instance.p_k.data.to(device=target_device, dtype=target_dtype)
            instance.p_v.data = instance.p_v.data.to(device=target_device, dtype=target_dtype)
            instance.gamma.data = instance.gamma.data.to(device=target_device, dtype=target_dtype)

        if hasattr(instance, "rotary_emb") and attn_module is not None and hasattr(attn_module, "rotary_emb") and hasattr(attn_module.rotary_emb, "inv_freq"):
            instance.rotary_emb.inv_freq.data.copy_(attn_module.rotary_emb.inv_freq.data)
            instance.rotary_emb._set_cos_sin_cache(0)

        return instance

    @classmethod
    def from_llama_attention(cls, attn_module, layer_idx=None, r_k=None, r_v=None, allow_default_full_rank=False):
        return cls.from_attention(attn_module, model=None, layer_idx=layer_idx, r_k=r_k, r_v=r_v, allow_default_full_rank=allow_default_full_rank)
