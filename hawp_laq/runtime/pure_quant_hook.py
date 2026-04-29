from __future__ import annotations

import torch
import torch.nn as nn

from hawp_laq.config import HAWPLAQConfig, build_k_quantizer, build_v_quantizer
from hawp_laq.modeling.attention_hawp import HAWPAttention, _QuantChunk
from hawp_laq.modeling.modeling_llama_hawp import _resolve_compute_dtype


try:
    from transformers import DynamicCache
    _HAS_DYNAMIC_CACHE = True
except ImportError:
    _HAS_DYNAMIC_CACHE = False


class LayerKVQuantCache:
    """Per-layer quantized KV cache for original (non-HAWP) attention.

    Stores recent tokens in ``dtype`` and archive tokens as quantized chunks
    (no raw副本).  On read-back the archive is dequantized per-chunk
    and concatenated with recent so the original attention formula
    ``softmax(Q @ K^T / sqrt(d)) @ V`` is preserved exactly.
    """

    def __init__(self, k_quantizer, v_quantizer, recent_window: int = 64, n_kv_heads: int = 1, head_dim: int = 64, dtype: torch.dtype = torch.float32) -> None:
        self.k_quantizer = k_quantizer
        self.v_quantizer = v_quantizer
        self.recent_window = recent_window
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self._recent_k = None
        self._recent_v = None
        self._archive_chunks: list[_QuantChunk] = []

    def reset(self) -> None:
        self._recent_k = None
        self._recent_v = None
        self._archive_chunks = []

    def update(self, k: torch.Tensor, v: torch.Tensor) -> None:
        k_new = k.detach()
        v_new = v.detach()
        if self.recent_window == 0:
            self._append_to_archive(k_new, v_new)
        else:
            self._append_recent(k_new, v_new)
            while self._recent_k.shape[1] > self.recent_window:
                self._demote()

    def _quantize_to_chunk(self, k: torch.Tensor, v: torch.Tensor) -> _QuantChunk:
        nkv, T, dk = k.shape
        _, _, dv = v.shape
        k_flat = k.reshape(nkv * T, dk).float()
        v_flat = v.reshape(nkv * T, dv).float()
        k_qx = self.k_quantizer.quantize(k_flat, logical_shape=(nkv, T, dk))
        v_qx = self.v_quantizer.quantize(v_flat, logical_shape=(nkv, T, dv))
        k_norms = k.float().norm(dim=2).to(torch.float16)
        return _QuantChunk(k_qx, v_qx, T, k_norms)

    def _append_recent(self, k_new: torch.Tensor, v_new: torch.Tensor) -> None:
        if self._recent_k is None:
            self._recent_k = k_new
            self._recent_v = v_new
        else:
            self._recent_k = torch.cat([self._recent_k, k_new], dim=1)
            self._recent_v = torch.cat([self._recent_v, v_new], dim=1)

    def _demote(self) -> None:
        if self._recent_k is None:
            return
        n_recent = self._recent_k.shape[1]
        if n_recent <= self.recent_window:
            return
        n_demote = n_recent - self.recent_window
        k_demote = self._recent_k[:, :n_demote, :]
        v_demote = self._recent_v[:, :n_demote, :]
        chunk = self._quantize_to_chunk(k_demote, v_demote)
        self._append_or_merge_archive_chunk(chunk)
        self._recent_k = self._recent_k[:, n_demote:, :]
        self._recent_v = self._recent_v[:, n_demote:, :]

    def _append_to_archive(self, k_new: torch.Tensor, v_new: torch.Tensor) -> None:
        chunk = self._quantize_to_chunk(k_new, v_new)
        self._append_or_merge_archive_chunk(chunk)

    def _merge_chunks_by_head(self, old_chunk: _QuantChunk, new_chunk: _QuantChunk) -> _QuantChunk:
        nkv = self.n_kv_heads
        _, old_T, _ = HAWPAttention._get_logical_nkv_T_dim(old_chunk.k_qx, nkv, self.head_dim)
        _, new_T, _ = HAWPAttention._get_logical_nkv_T_dim(new_chunk.k_qx, nkv, self.head_dim)
        _, old_v_T, _ = HAWPAttention._get_logical_nkv_T_dim(old_chunk.v_qx, nkv, self.head_dim)
        _, new_v_T, _ = HAWPAttention._get_logical_nkv_T_dim(new_chunk.v_qx, nkv, self.head_dim)
        if old_T != old_v_T or new_T != new_v_T:
            raise RuntimeError(
                f"K/V token count mismatch while merging pure quant chunks: "
                f"K=({old_T},{new_T}) V=({old_v_T},{new_v_T})"
            )

        k_qx = HAWPAttention._merge_quantized_by_head(
            old_chunk.k_qx, new_chunk.k_qx, nkv, self.head_dim,
        )
        v_qx = HAWPAttention._merge_quantized_by_head(
            old_chunk.v_qx, new_chunk.v_qx, nkv, self.head_dim,
        )

        if old_chunk.k_norms is None and new_chunk.k_norms is None:
            k_norms = None
        elif old_chunk.k_norms is not None and new_chunk.k_norms is not None:
            k_norms = torch.cat([old_chunk.k_norms, new_chunk.k_norms], dim=1)
        else:
            raise RuntimeError("k_norms mismatch while merging pure quant archive chunks")

        return _QuantChunk(k_qx, v_qx, old_T + new_T, k_norms)

    def _append_or_merge_archive_chunk(self, new_chunk: _QuantChunk) -> None:
        if not self._archive_chunks:
            self._archive_chunks.append(new_chunk)
            return
        if len(self._archive_chunks) == 1:
            self._archive_chunks[0] = self._merge_chunks_by_head(
                self._archive_chunks[0], new_chunk,
            )
            return
        raise RuntimeError(
            f"single archive chunk invariant violated: found {len(self._archive_chunks)} chunks"
        )

    def get_kv(self) -> tuple[torch.Tensor, torch.Tensor]:
        k_parts = []
        v_parts = []
        for chunk in self._archive_chunks:
            k_deq = self.k_quantizer.dequantize(chunk.k_qx).reshape(
                self.n_kv_heads, chunk.n_tokens, self.head_dim,
            )
            v_deq = self.v_quantizer.dequantize(chunk.v_qx).reshape(
                self.n_kv_heads, chunk.n_tokens, self.head_dim,
            )
            k_parts.append(k_deq)
            v_parts.append(v_deq)
        if self._recent_k is not None:
            k_parts.append(self._recent_k.to(self.dtype))
            v_parts.append(self._recent_v.to(self.dtype))
        if not k_parts:
            raise RuntimeError("LayerKVQuantCache.get_kv called on empty cache")
        return torch.cat(k_parts, dim=1).to(self.dtype), torch.cat(v_parts, dim=1).to(self.dtype)

    @property
    def seq_len(self) -> int:
        total = 0
        for chunk in self._archive_chunks:
            total += chunk.n_tokens
        if self._recent_k is not None:
            total += self._recent_k.shape[1]
        return total

    def summary(self) -> dict:
        n_recent = self._recent_k.shape[1] if self._recent_k is not None else 0
        n_archive = sum(c.n_tokens for c in self._archive_chunks)
        recent_fp_bytes = 0
        if self._recent_k is not None:
            recent_fp_bytes += self._recent_k.nelement() * self._recent_k.element_size()
            recent_fp_bytes += self._recent_v.nelement() * self._recent_v.element_size()
        archive_quant_bytes = 0
        for chunk in self._archive_chunks:
            archive_quant_bytes += self.k_quantizer.estimate_num_bytes(chunk.k_qx)
            archive_quant_bytes += self.v_quantizer.estimate_num_bytes(chunk.v_qx)
        archive_meta_bytes = 0
        for chunk in self._archive_chunks:
            if chunk.k_norms is not None:
                archive_meta_bytes += chunk.k_norms.nelement() * chunk.k_norms.element_size()
        return {
            "recent_tokens": n_recent,
            "archive_tokens": n_archive,
            "recent_fp_bytes": recent_fp_bytes,
            "archive_quant_bytes": archive_quant_bytes,
            "archive_meta_bytes": archive_meta_bytes,
            "total_runtime_bytes": recent_fp_bytes + archive_quant_bytes + archive_meta_bytes,
            "compressed_storage_bytes": archive_quant_bytes,
        }


class PureQuantKVManager:
    """Manages quantized KV caches for a model with **original HF attention**.

    Key invariant: the model's attention modules are **never replaced** with
    HAWPAttention.  Instead, after each forward pass we extract K/V via
    PyTorch hooks, quantize them, and on subsequent steps we reconstruct
    the ``past_key_value`` tuples from the dequantized cache so the
    original attention forward sees standard K/V tensors.

    Call chain:
        1. ``install_hooks()`` — register forward hooks on each attention
           module's K/V projection outputs.
        2. ``model(...)`` — during forward, hooks capture K/V per layer.
        3. ``on_forward_done()`` — move captured K/V into quantized cache.
        4. ``get_past_kv()`` — return dequantized K/V as ``past_key_value``
           tuples for the next forward pass.
    """

    def __init__(self, model: nn.Module, cfg: HAWPLAQConfig) -> None:
        self.model = model
        self.cfg = cfg
        self._caches: dict[int, LayerKVQuantCache] = {}
        self._captured_kv: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self._hooks: list[torch.utils.hooks.RemovableHook] = []
        self._layer_attn_map: dict[int, nn.Module] = {}
        self._installed = False
        self._head_dim = self._resolve_head_dim()
        self._n_kv_heads = self._resolve_n_kv_heads()

    def _resolve_head_dim(self) -> int:
        for _, attn_mod in self._find_attention_modules():
            if hasattr(attn_mod, "head_dim"):
                return attn_mod.head_dim
            q_proj = getattr(attn_mod, "q_proj", None) or getattr(attn_mod, "query_proj", None)
            if q_proj is not None:
                out_features = q_proj.out_features
                num_heads = getattr(attn_mod, "num_heads", None)
                if num_heads is None and hasattr(attn_mod, "config"):
                    num_heads = getattr(attn_mod.config, "num_attention_heads", None)
                if num_heads is not None and num_heads > 0:
                    return out_features // num_heads
        config = getattr(self.model, "config", None)
        if config is not None:
            hidden_size = getattr(config, "hidden_size", 0)
            num_heads = getattr(config, "num_attention_heads", 0)
            if hidden_size > 0 and num_heads > 0:
                return hidden_size // num_heads
        raise ValueError("Cannot resolve head_dim from model attention modules")

    def _resolve_n_kv_heads(self) -> int:
        for _, attn_mod in self._find_attention_modules():
            if hasattr(attn_mod, "num_kv_heads"):
                return attn_mod.num_kv_heads
            if hasattr(attn_mod, "num_key_value_heads"):
                return attn_mod.num_key_value_heads
            k_proj = getattr(attn_mod, "k_proj", None) or getattr(attn_mod, "key_proj", None)
            if k_proj is not None:
                return k_proj.out_features // self._head_dim
        config = getattr(self.model, "config", None)
        if config is not None:
            n_kv = getattr(config, "num_key_value_heads", None)
            if n_kv is not None:
                return n_kv
            return getattr(config, "num_attention_heads", 1)
        return 1

    @property
    def head_dim(self) -> int:
        return self._head_dim

    def _find_attention_modules(self) -> list[tuple[int, nn.Module]]:
        results = []
        for name, module in self.model.named_modules():
            cls_name = type(module).__name__
            if "Attention" in cls_name and not isinstance(module, HAWPAttention):
                if any(kw in cls_name for kw in (
                    "Llama", "OPT", "Mistral", "Qwen2", "Phi3", "Gemma",
                )):
                    layer_idx = self._extract_layer_idx(name)
                    results.append((layer_idx, module))
        return results

    @staticmethod
    def _extract_layer_idx(name: str) -> int:
        parts = name.split(".")
        for p in parts:
            if p.isdigit():
                return int(p)
        return 0

    def _resolve_model_compute_dtype(self) -> torch.dtype:
        for _, attn_mod in self._find_attention_modules():
            q_proj = getattr(attn_mod, "q_proj", None) or getattr(attn_mod, "query_proj", None)
            if q_proj is not None:
                return _resolve_compute_dtype(q_proj)
        for p in self.model.parameters():
            if p.dtype.is_floating_point:
                return p.dtype
        return torch.float32

    def install_hooks(self) -> None:
        if self._installed:
            return
        recent_window = self.cfg.sched.recent_window
        param_dtype = self._resolve_model_compute_dtype()
        for layer_idx, attn_mod in self._find_attention_modules():
            k_q = build_k_quantizer(self.cfg, r_k=self._head_dim)
            v_q = build_v_quantizer(self.cfg, r_v=self._head_dim)
            self._caches[layer_idx] = LayerKVQuantCache(
                k_q, v_q, recent_window=recent_window,
                n_kv_heads=self._n_kv_heads, head_dim=self._head_dim,
                dtype=param_dtype,
            )
            self._layer_attn_map[layer_idx] = attn_mod

            k_proj = getattr(attn_mod, "k_proj", None) or getattr(attn_mod, "key_proj", None)
            v_proj = getattr(attn_mod, "v_proj", None) or getattr(attn_mod, "value_proj", None)
            if k_proj is None or v_proj is None:
                raise ValueError(
                    f"Cannot find k_proj/v_proj on {type(attn_mod).__name__} at layer {layer_idx}"
                )

            idx = layer_idx

            def make_k_hook(li):
                def hook_fn(module, input, output):
                    self._captured_kv.setdefault(li, (None, None))
                    k_val = output[0] if isinstance(output, tuple) else output
                    old_v = self._captured_kv[li][1]
                    self._captured_kv[li] = (k_val.detach(), old_v)
                return hook_fn

            def make_v_hook(li):
                def hook_fn(module, input, output):
                    self._captured_kv.setdefault(li, (None, None))
                    old_k = self._captured_kv[li][0]
                    v_val = output[0] if isinstance(output, tuple) else output
                    self._captured_kv[li] = (old_k, v_val.detach())
                return hook_fn

            h1 = k_proj.register_forward_hook(make_k_hook(idx))
            h2 = v_proj.register_forward_hook(make_v_hook(idx))
            self._hooks.extend([h1, h2])

        self._installed = True

        if not self._caches:
            model_type = getattr(self.model.config, "model_type", "")
            raise RuntimeError(
                f"[pure_quant_only] No attention modules found in model "
                f"(model_type={model_type!r}). Cannot install KV capture hooks. "
                f"Check that your model's attention class name is recognized "
                f"(see PureQuantKVManager._find_attention_modules)."
            )

    def reset_caches(self) -> None:
        for cache in self._caches.values():
            cache.reset()
        self._captured_kv.clear()

    def on_forward_done(self) -> None:
        """Move captured K/V from hooks into the quantized cache.

        Should be called after each ``model(...)`` forward pass.
        The hooks capture the raw k_proj/v_proj output which has shape
        ``[bsz, seq_len, n_kv_heads * head_dim]``.  We reshape to
        ``[n_kv_heads, seq_len, head_dim]`` before storing.
        """
        for layer_idx, (k, v) in self._captured_kv.items():
            if k is None or v is None:
                continue
            bsz = k.shape[0]
            seq_len_k = k.shape[1]
            seq_len_v = v.shape[1]
            k_3d = k.view(bsz, seq_len_k, self._n_kv_heads, self._head_dim) \
                     .permute(0, 2, 1, 3).squeeze(0)
            v_3d = v.view(bsz, seq_len_v, self._n_kv_heads, self._head_dim) \
                     .permute(0, 2, 1, 3).squeeze(0)
            self._caches[layer_idx].update(k_3d, v_3d)
        self._captured_kv.clear()

    def on_forward_done_from_output(self, past_key_values, prev_seq_len: int) -> None:
        """Extract **new** K/V from model output and store into quant cache.

        This is the correct way to capture K/V after a forward pass when the
        model uses RoPE (e.g. Llama, Mistral, Qwen2, Gemma).  The hooks on
        ``k_proj`` / ``v_proj`` capture **pre-RoPE** K/V, but the HF
        attention expects **post-RoPE** K/V in the cache.  Using hook-captured
        K/V causes a position-encoding mismatch that shifts generation results.

        This method reads the full K/V from ``outputs.past_key_values``
        (which is post-RoPE) and extracts only the newly computed tokens
        (positions ``prev_seq_len:``), then stores them in the quant cache.

        Args:
            past_key_values: The ``outputs.past_key_values`` returned by the
                model forward.  Can be a ``DynamicCache`` or a list of
                ``(key, value)`` tuples.
            prev_seq_len: Number of K/V tokens that were already in the cache
                *before* this forward pass.  Tokens from this index onward
                are the newly computed ones.
        """
        if _HAS_DYNAMIC_CACHE and isinstance(past_key_values, DynamicCache):
            for layer_idx in range(len(past_key_values.key_cache)):
                k_full = past_key_values.key_cache[layer_idx]
                v_full = past_key_values.value_cache[layer_idx]
                if k_full is None or v_full is None:
                    continue
                if k_full.shape[2] <= prev_seq_len:
                    continue
                k_new = k_full[:, :, prev_seq_len:, :]
                v_new = v_full[:, :, prev_seq_len:, :]
                k_3d = k_new.squeeze(0)
                v_3d = v_new.squeeze(0)
                if layer_idx in self._caches:
                    self._caches[layer_idx].update(k_3d, v_3d)
        else:
            for layer_idx, kv in enumerate(past_key_values):
                if kv is None:
                    continue
                k_full, v_full = kv
                if k_full.shape[2] <= prev_seq_len:
                    continue
                k_new = k_full[:, :, prev_seq_len:, :]
                v_new = v_full[:, :, prev_seq_len:, :]
                k_3d = k_new.squeeze(0)
                v_3d = v_new.squeeze(0)
                if layer_idx in self._caches:
                    self._caches[layer_idx].update(k_3d, v_3d)
        self._captured_kv.clear()

    def get_past_kv(self):
        """Return dequantized K/V as a DynamicCache for the model's next forward.

        For each layer, the cache is pre-filled with dequantized K/V tensors
        of shape ``[bsz, n_kv_heads, T, head_dim]``.  When the model calls
        ``past_key_value.update(key, value, layer_idx)``, the new token's
        K/V will be concatenated onto the existing cache, producing the full
        sequence K/V for that step's attention computation.

        Returns:
            A ``DynamicCache`` (or tuple list for old transformers versions)
            ready to pass as ``past_key_values``.
        """
        param_dtype = self._resolve_model_compute_dtype()
        if _HAS_DYNAMIC_CACHE:
            cache = DynamicCache()
            for layer_idx in sorted(self._caches.keys()):
                lc = self._caches[layer_idx]
                if lc.seq_len == 0:
                    continue
                k, v = lc.get_kv()
                k_4d = k.unsqueeze(0).to(self.model.device, dtype=param_dtype)
                v_4d = v.unsqueeze(0).to(self.model.device, dtype=param_dtype)
                cache.update(k_4d, v_4d, layer_idx)
            return cache
        else:
            past = []
            for layer_idx in sorted(self._caches.keys()):
                lc = self._caches[layer_idx]
                if lc.seq_len == 0:
                    past.append(None)
                    continue
                k, v = lc.get_kv()
                k_4d = k.unsqueeze(0).to(self.model.device, dtype=param_dtype)
                v_4d = v.unsqueeze(0).to(self.model.device, dtype=param_dtype)
                past.append((k_4d, v_4d))
            return past

    def cache_summaries(self) -> list[dict]:
        summaries = []
        for layer_idx in sorted(self._caches.keys()):
            s = self._caches[layer_idx].summary()
            s["layer"] = layer_idx
            summaries.append(s)
        return summaries

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._installed = False


def install_pure_quant_hooks(model: nn.Module, cfg: HAWPLAQConfig) -> PureQuantKVManager:
    """Create and install a PureQuantKVManager on a vanilla HF model.

    Returns the manager for cache reset / past_kv retrieval.
    """
    manager = PureQuantKVManager(model, cfg)
    manager.install_hooks()
    return manager
