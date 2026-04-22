from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from hawp_laq.modeling.rope_utils import LlamaRotaryEmbedding, apply_rotary_pos_emb


_DEFAULT_CONFIG = SimpleNamespace(
    hidden_size=768,
    num_attention_heads=12,
    num_key_value_heads=12,
    max_position_embeddings=2048,
    rope_theta=10000.0,
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
                model_type="",
                enable_bias=hasattr(src_q, "bias") and src_q.bias is not None,
                attention_dropout=0.0,
            )
    return _DEFAULT_CONFIG


def _make_causal_mask(q_len: int, kv_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    mask = torch.full((q_len, kv_len), float("-inf"), device=device, dtype=dtype)
    mask = torch.triu(mask, diagonal=kv_len - q_len + 1)
    return mask.unsqueeze(0).unsqueeze(0)


class HAWPAttention(nn.Module):
    def __init__(
        self,
        config,
        layer_idx: int | None = None,
        r_k: int | None = None,
        r_v: int | None = None,
        allow_default_full_rank: bool = False,
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
        self.q_proj = nn.Linear(self.hidden_size, q_out, bias=use_bias)
        self.k_proj = nn.Linear(self.hidden_size, kv_out, bias=use_bias)
        self.v_proj = nn.Linear(self.hidden_size, kv_out, bias=use_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=use_bias)

        self._use_rope = not (self.is_opt or self.is_gpt_neox)
        if self._use_rope:
            rope_theta = getattr(config, "rope_theta", 10000.0)
            self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=rope_theta,
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

        self.p_k = nn.Parameter(torch.eye(self.head_dim), requires_grad=(r_k < self.head_dim))
        self.p_v = nn.Parameter(torch.eye(self.head_dim), requires_grad=(r_v < self.head_dim))
        self.gamma = nn.Parameter(torch.ones(1), requires_grad=False)

        self._src_weights: dict[str, torch.Tensor] = {}
        self.use_quantizer = False
        self.use_cache_manager = False
        self.recent_window = 64
        self._tq_k_quantizer = None
        self._tq_v_quantizer = None
        self._quant_recent_k = None
        self._quant_recent_v = None
        self._quant_archive_k_qx = None
        self._quant_archive_v_qx = None
        self._quant_archive_k_raw = None
        self._quant_archive_v_raw = None

    @property
    def _is_low_rank(self) -> bool:
        return self.r_k < self.head_dim or self.r_v < self.head_dim

    def setup_quant_cache(self, k_quantizer, v_quantizer, recent_window: int = 64) -> None:
        self.use_cache_manager = True
        self.recent_window = recent_window
        self._tq_k_quantizer = k_quantizer
        self._tq_v_quantizer = v_quantizer
        self.reset_quant_cache()

    def reset_quant_cache(self) -> None:
        self._quant_recent_k = None
        self._quant_recent_v = None
        self._quant_archive_k_qx = None
        self._quant_archive_v_qx = None
        self._quant_archive_k_raw = None
        self._quant_archive_v_raw = None

    def _quant_cache_seq_len(self) -> int:
        total = 0
        if self._quant_archive_k_raw is not None:
            total += self._quant_archive_k_raw.shape[1]
        if self._quant_recent_k is not None:
            total += self._quant_recent_k.shape[1]
        return total

    def _quant_cache_append(self, k_lat: torch.Tensor, v_lat: torch.Tensor) -> None:
        k_new = k_lat[0].detach()
        v_new = v_lat[0].detach()
        if self._quant_recent_k is None:
            self._quant_recent_k = k_new
            self._quant_recent_v = v_new
        else:
            self._quant_recent_k = torch.cat([self._quant_recent_k, k_new], dim=1)
            self._quant_recent_v = torch.cat([self._quant_recent_v, v_new], dim=1)

    def _quant_cache_demote(self) -> None:
        if self._quant_recent_k is None:
            return
        k_raw = self._quant_recent_k
        v_raw = self._quant_recent_v
        if self._quant_archive_k_raw is not None:
            k_raw = torch.cat([self._quant_archive_k_raw, k_raw], dim=1)
            v_raw = torch.cat([self._quant_archive_v_raw, v_raw], dim=1)
        self._quant_archive_k_raw = k_raw
        self._quant_archive_v_raw = v_raw
        nkv, T, rk = k_raw.shape
        _, _, rv = v_raw.shape
        k_flat = k_raw.reshape(nkv * T, rk).float()
        v_flat = v_raw.reshape(nkv * T, rv).float()
        self._quant_archive_k_qx = self._tq_k_quantizer.quantize(k_flat)
        self._quant_archive_v_qx = self._tq_v_quantizer.quantize(v_flat)
        self._quant_recent_k = None
        self._quant_recent_v = None

    def _quant_cache_append_to_archive(self, k_lat: torch.Tensor, v_lat: torch.Tensor) -> None:
        # Correctness-first: full requantize of the entire archive after every
        # append.  This guarantees scale/zero-point are computed over the full
        # token range rather than incrementally merged, avoiding quantization
        # drift.  Speed optimisations (incremental merge) can be re-introduced
        # later as an optional fast path once correctness is validated end-to-end.
        k_new = k_lat[0].detach()
        v_new = v_lat[0].detach()
        if self._quant_archive_k_raw is not None:
            self._quant_archive_k_raw = torch.cat([self._quant_archive_k_raw, k_new], dim=1)
            self._quant_archive_v_raw = torch.cat([self._quant_archive_v_raw, v_new], dim=1)
        else:
            self._quant_archive_k_raw = k_new
            self._quant_archive_v_raw = v_new
        nkv, T, rk = self._quant_archive_k_raw.shape
        _, _, rv = self._quant_archive_v_raw.shape
        k_flat = self._quant_archive_k_raw.reshape(nkv * T, rk).float()
        v_flat = self._quant_archive_v_raw.reshape(nkv * T, rv).float()
        self._quant_archive_k_qx = self._tq_k_quantizer.quantize(k_flat)
        self._quant_archive_v_qx = self._tq_v_quantizer.quantize(v_flat)

    # TODO(optional fast-path): _merge_quantized is preserved for a future
    # incremental-merge fast path.  Currently unused because the default
    # _quant_cache_append_to_archive uses correctness-first full requantize.
    @staticmethod
    def _merge_quantized(old_qx, new_qx, nkv, dim):
        from hawp_laq.runtime.turboquant import TurboQuantProdResult, TurboQuantizedTensor
        if isinstance(new_qx, TurboQuantProdResult):
            old_mse = old_qx.mse
            new_mse = new_qx.mse
            merged_mse = TurboQuantizedTensor(
                q=torch.cat([old_mse.q, new_mse.q], dim=0),
                scale=torch.cat([old_mse.scale, new_mse.scale], dim=0),
                zero_point=torch.cat([old_mse.zero_point, new_mse.zero_point], dim=0),
                shape_orig=(old_mse.shape_orig[0] + new_mse.shape_orig[0], dim),
                bits=old_mse.bits,
                group_size=old_mse.group_size,
                rotation=old_mse.rotation,
            )
            return TurboQuantProdResult(
                mse=merged_mse,
                residual_sign=torch.cat([old_qx.residual_sign, new_qx.residual_sign], dim=0),
                residual_norm=torch.cat([old_qx.residual_norm, new_qx.residual_norm], dim=0),
                dim=dim,
                shape_orig=(old_qx.shape_orig[0] + new_qx.shape_orig[0], dim),
            )
        else:
            return TurboQuantizedTensor(
                q=torch.cat([old_qx.q, new_qx.q], dim=0),
                scale=torch.cat([old_qx.scale, new_qx.scale], dim=0),
                zero_point=torch.cat([old_qx.zero_point, new_qx.zero_point], dim=0),
                shape_orig=(old_qx.shape_orig[0] + new_qx.shape_orig[0], dim),
                bits=old_qx.bits,
                group_size=old_qx.group_size,
                rotation=old_qx.rotation,
            )

    def _quant_cache_get_kv(self) -> tuple[torch.Tensor, torch.Tensor]:
        k_parts = []
        v_parts = []
        if self._quant_archive_k_qx is not None:
            nkv, T, rk = self._quant_archive_k_raw.shape
            _, _, rv = self._quant_archive_v_raw.shape
            k_deq = self._tq_k_quantizer.dequantize(self._quant_archive_k_qx).reshape(nkv, T, rk)
            v_deq = self._tq_v_quantizer.dequantize(self._quant_archive_v_qx).reshape(nkv, T, rv)
            k_parts.append(k_deq)
            v_parts.append(v_deq)
        if self._quant_recent_k is not None:
            k_parts.append(self._quant_recent_k.float())
            v_parts.append(self._quant_recent_v.float())
        if not k_parts:
            return None, None
        return torch.cat(k_parts, dim=1), torch.cat(v_parts, dim=1)

    def drop_oldest_from_archive(self, n: int) -> int:
        if self._quant_archive_k_raw is None:
            return 0
        T = self._quant_archive_k_raw.shape[1]
        drop_n = min(n, T)
        if drop_n == 0:
            return 0
        self._quant_archive_k_raw = self._quant_archive_k_raw[:, drop_n:, :]
        self._quant_archive_v_raw = self._quant_archive_v_raw[:, drop_n:, :]
        if self._quant_archive_k_raw.shape[1] > 0:
            nkv, T_new, rk = self._quant_archive_k_raw.shape
            _, _, rv = self._quant_archive_v_raw.shape
            k_flat = self._quant_archive_k_raw.reshape(nkv * T_new, rk).float()
            v_flat = self._quant_archive_v_raw.reshape(nkv * T_new, rv).float()
            self._quant_archive_k_qx = self._tq_k_quantizer.quantize(k_flat)
            self._quant_archive_v_qx = self._tq_v_quantizer.quantize(v_flat)
        else:
            self._quant_archive_k_raw = None
            self._quant_archive_v_raw = None
            self._quant_archive_k_qx = None
            self._quant_archive_v_qx = None
        return drop_n

    def drop_least_important_from_archive(self, n: int) -> int:
        if self._quant_archive_k_raw is None:
            return 0
        T = self._quant_archive_k_raw.shape[1]
        drop_n = min(n, T)
        if drop_n == 0:
            return 0
        norms = self._quant_archive_k_raw.norm(dim=(0, 2))
        _, indices = norms.sort()
        keep_indices = indices[drop_n:].sort()[0]
        self._quant_archive_k_raw = self._quant_archive_k_raw[:, keep_indices, :]
        self._quant_archive_v_raw = self._quant_archive_v_raw[:, keep_indices, :]
        if self._quant_archive_k_raw.shape[1] > 0:
            nkv, T_new, rk = self._quant_archive_k_raw.shape
            _, _, rv = self._quant_archive_v_raw.shape
            k_flat = self._quant_archive_k_raw.reshape(nkv * T_new, rk).float()
            v_flat = self._quant_archive_v_raw.reshape(nkv * T_new, rv).float()
            self._quant_archive_k_qx = self._tq_k_quantizer.quantize(k_flat)
            self._quant_archive_v_qx = self._tq_v_quantizer.quantize(v_flat)
        else:
            self._quant_archive_k_raw = None
            self._quant_archive_v_raw = None
            self._quant_archive_k_qx = None
            self._quant_archive_v_qx = None
        return drop_n

    def quant_cache_summary(self) -> dict:
        n_recent = self._quant_recent_k.shape[1] if self._quant_recent_k is not None else 0
        n_archive = self._quant_archive_k_raw.shape[1] if self._quant_archive_k_raw is not None else 0

        recent_fp_bytes = 0
        if self._quant_recent_k is not None:
            recent_fp_bytes += self._quant_recent_k.nelement() * self._quant_recent_k.element_size()
            recent_fp_bytes += self._quant_recent_v.nelement() * self._quant_recent_v.element_size()

        archive_raw_bytes = 0
        if self._quant_archive_k_raw is not None:
            archive_raw_bytes += self._quant_archive_k_raw.nelement() * self._quant_archive_k_raw.element_size()
            archive_raw_bytes += self._quant_archive_v_raw.nelement() * self._quant_archive_v_raw.element_size()

        archive_quant_bytes = 0
        if self._quant_archive_k_qx is not None:
            archive_quant_bytes += self._tq_k_quantizer.estimate_num_bytes(self._quant_archive_k_qx)
        if self._quant_archive_v_qx is not None:
            archive_quant_bytes += self._tq_v_quantizer.estimate_num_bytes(self._quant_archive_v_qx)

        compressed_storage_bytes = archive_quant_bytes
        total_runtime_bytes = recent_fp_bytes + archive_raw_bytes + archive_quant_bytes

        return {
            "layer": self.layer_idx,
            "recent_tokens": n_recent,
            "archive_tokens": n_archive,
            "recent_fp_bytes": recent_fp_bytes,
            "archive_raw_bytes": archive_raw_bytes,
            "archive_quant_bytes": archive_quant_bytes,
            "total_runtime_bytes": total_runtime_bytes,
            "compressed_storage_bytes": compressed_storage_bytes,
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

        if self.is_opt:
            query_states = self.q_proj(hidden_states) * self.scaling
        else:
            query_states = self.q_proj(hidden_states)

        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if self._use_rope:
            if position_embeddings is not None:
                cos, sin = position_embeddings
            else:
                cos, sin = self.rotary_emb(value_states, seq_len=q_len)
                if position_ids is not None:
                    cos = cos[position_ids].unsqueeze(1)
                    sin = sin[position_ids].unsqueeze(1)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if self._is_low_rank or self.use_cache_manager:
            return self._forward_low_rank(
                query_states, key_states, value_states,
                attention_mask, past_key_value, use_cache,
                cache_position, **kwargs,
            )

        if self.is_opt and past_key_value is not None:
            cache_kwargs = {}
            if cache_position is not None:
                cache_kwargs["cache_position"] = cache_position
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )
        elif past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

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
            if attention_mask is not None:
                causal_mask = attention_mask[:, :, :, :key_states.shape[-2]]
                attn_weights = attn_weights + causal_mask
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        if self.is_opt:
            return attn_output, attn_weights, past_key_value
        past_kv = (key_states, value_states) if use_cache else None
        return attn_output, None, past_kv

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
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, :key.shape[-2]]
            attn_weights = attn_weights + causal_mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        if dropout > 0.0:
            attn_weights = F.dropout(attn_weights, p=dropout)
        attn_output = torch.matmul(attn_weights, value)
        return attn_output, attn_weights

    def _forward_low_rank(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        past_key_value,
        use_cache: bool,
        cache_position: Optional[torch.Tensor],
        **kwargs,
    ):
        bsz = query_states.size(0)
        q_len = query_states.size(2)

        pk_down = self.p_k[:, :self.r_k]
        pv_down = self.p_v[:, :self.r_v]

        q_lat = query_states @ pk_down
        k_lat = key_states @ pk_down
        v_lat = value_states @ pv_down

        kv_from_cache = False

        if self.use_cache_manager and self._tq_k_quantizer is not None:
            cache_was_empty = (
                self._quant_recent_k is None and self._quant_archive_k_qx is None
            )
            if self.recent_window == 0:
                self._quant_cache_append_to_archive(k_lat, v_lat)
                k_cached, v_cached = self._quant_cache_get_kv()
                if k_cached is not None:
                    k_lat = k_cached.unsqueeze(0).to(q_lat.device, q_lat.dtype)
                    v_lat = v_cached.unsqueeze(0).to(q_lat.device, q_lat.dtype)
                    kv_from_cache = True
            else:
                self._quant_cache_append(k_lat, v_lat)
                if self._quant_recent_k is not None and self._quant_recent_k.shape[1] > self.recent_window:
                    self._quant_cache_demote()
                if not cache_was_empty:
                    k_cached, v_cached = self._quant_cache_get_kv()
                    k_lat = k_cached.unsqueeze(0).to(q_lat.device, q_lat.dtype)
                    v_lat = v_cached.unsqueeze(0).to(q_lat.device, q_lat.dtype)
                    kv_from_cache = True
        elif self.is_opt and past_key_value is not None:
            cache_kwargs: dict = {}
            if cache_position is not None:
                cache_kwargs["cache_position"] = cache_position
            k_lat, v_lat = past_key_value.update(
                k_lat, v_lat, self.layer_idx, cache_kwargs,
            )
        elif past_key_value is not None:
            k_lat = torch.cat([past_key_value[0], k_lat], dim=2)
            v_lat = torch.cat([past_key_value[1], v_lat], dim=2)

        cache_k_lat = k_lat
        cache_v_lat = v_lat

        k_lat_expanded = self._repeat_kv(k_lat)
        v_lat_expanded = self._repeat_kv(v_lat)

        if self.is_opt:
            q_lat = q_lat * (self.head_dim ** 0.5)

        lr_scale = self.gamma.item() / math.sqrt(self.r_k)
        attn_weights = torch.matmul(q_lat, k_lat_expanded.transpose(2, 3)) * lr_scale

        causal_mask = None
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, :k_lat_expanded.shape[-2]]
        elif q_len > 1:
            causal_mask = _make_causal_mask(q_len, k_lat_expanded.shape[-2], q_lat.device, q_lat.dtype)

        if causal_mask is not None:
            attn_weights = attn_weights + causal_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output_lat = torch.matmul(attn_weights, v_lat_expanded)

        attn_output = attn_output_lat @ pv_down.T

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        if self.is_opt:
            if self.use_cache_manager and self._tq_k_quantizer is not None:
                return attn_output, attn_weights, past_key_value
            return attn_output, attn_weights, past_key_value
        if self.use_cache_manager and self._tq_k_quantizer is not None:
            return attn_output, None, None
        past_kv = (cache_k_lat, cache_v_lat) if use_cache else None
        return attn_output, None, past_kv

    def _apply_pk(self, k: torch.Tensor) -> torch.Tensor:
        if self.r_k >= self.head_dim and not self.p_k.requires_grad:
            return k
        pk_down = self.p_k[:, :self.r_k]
        return k @ pk_down @ pk_down.T

    def _apply_pv(self, v: torch.Tensor) -> torch.Tensor:
        if self.r_v >= self.head_dim and not self.p_v.requires_grad:
            return v
        pv_down = self.p_v[:, :self.r_v]
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

        if "gamma" in data:
            self.gamma.data.copy_(data["gamma"].to(self.gamma.device, self.gamma.dtype))
        elif "gamma_k" in data and "gamma_v" in data:
            self.gamma.data.copy_(data["gamma_v"].to(self.gamma.device, self.gamma.dtype))

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
    ):
        config = _get_attn_config(attn_module, model)
        if layer_idx is None and attn_module is not None:
            layer_idx = getattr(attn_module, "layer_idx", None)
        instance = cls(config, layer_idx=layer_idx, r_k=r_k, r_v=r_v,
                       allow_default_full_rank=allow_default_full_rank)

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

        _src_weights = {}
        for std_name, src in _src_modules.items():
            if hasattr(src, "weight"):
                _src_weights[std_name] = src.weight.data.detach().clone()

        for std_name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            if std_name in _src_weights:
                getattr(instance, std_name).weight.data.copy_(_src_weights[std_name])
            else:
                _src_weights[std_name] = getattr(instance, std_name).weight.data.detach().clone()

        for std_name, src in _src_modules.items():
            if std_name != "o_proj" and hasattr(src, "bias") and src.bias is not None:
                getattr(instance, std_name).bias.data.copy_(src.bias.data)
        if "o_proj" in _src_modules and hasattr(_src_modules["o_proj"], "bias") and _src_modules["o_proj"].bias is not None:
            instance.o_proj.bias.data.copy_(_src_modules["o_proj"].bias.data)

        if "q_proj" in _src_modules:
            param_dtype = _src_modules["q_proj"].weight.dtype
            instance = instance.to(param_dtype)
            instance.p_k.data.copy_(torch.eye(instance.head_dim, device=instance.p_k.device, dtype=param_dtype))
            instance.p_v.data.copy_(torch.eye(instance.head_dim, device=instance.p_v.device, dtype=param_dtype))
            instance.gamma.data.fill_(1.0)

        if hasattr(instance, "rotary_emb") and attn_module is not None and hasattr(attn_module, "rotary_emb") and hasattr(attn_module.rotary_emb, "inv_freq"):
            instance.rotary_emb.inv_freq.data.copy_(attn_module.rotary_emb.inv_freq.data)

        instance._src_weights = _src_weights
        return instance

    @classmethod
    def from_llama_attention(cls, attn_module, layer_idx=None, r_k=None, r_v=None, allow_default_full_rank=False):
        return cls.from_attention(attn_module, model=None, layer_idx=layer_idx, r_k=r_k, r_v=r_v, allow_default_full_rank=allow_default_full_rank)
