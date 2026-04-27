"""Tests for hawp_only low-rank KV cache correctness.

These tests verify that hawp_only mode stores low-rank k_lat/v_lat in the
HF past_key_values cache (instead of full-rank reconstructions), which:
  1. Saves KV cache memory (r_k/head_dim ratio)
  2. Avoids round-trip projection error (k_lat @ pk_down.T @ pk_down)
  3. Produces output consistent with full-recompute (no cache)
"""
from __future__ import annotations

import pytest
import torch
from types import SimpleNamespace

from hawp_laq.modeling.attention_hawp import HAWPAttention, _make_causal_mask


def _make_config(model_type="opt"):
    return SimpleNamespace(
        hidden_size=256,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=512,
        rope_theta=10000.0,
        rope_scaling=None,
        model_type=model_type,
        enable_bias=True,
        attention_dropout=0.0,
    )


def _make_attn(r_k=32, r_v=32, model_type="opt"):
    cfg = _make_config(model_type)
    attn = HAWPAttention(cfg, layer_idx=0, r_k=r_k, r_v=r_v)
    attn.eval()
    return attn


_MODEL_TYPES = ["opt", "llama"]


class TestHawpOnlyLowRankKVCache:
    @pytest.mark.parametrize("model_type", _MODEL_TYPES)
    def test_tuple_past_kv_stores_low_rank(self, model_type):
        attn = _make_attn(r_k=32, r_v=32, model_type=model_type)
        head_dim = attn.head_dim
        assert attn.r_k < head_dim

        bsz, seq_len = 1, 5
        hidden = torch.randn(bsz, seq_len, attn.hidden_size)

        with torch.no_grad():
            out = attn(hidden_states=hidden, use_cache=True)

        attn_out, _, past_kv = out
        k, v = past_kv
        assert k.shape[-1] == attn.r_k, f"[{model_type}] K last dim should be r_k={attn.r_k}, got {k.shape[-1]}"
        assert v.shape[-1] == attn.r_v, f"[{model_type}] V last dim should be r_v={attn.r_v}, got {v.shape[-1]}"

    @pytest.mark.parametrize("model_type", _MODEL_TYPES)
    def test_dynamic_cache_stores_low_rank(self, model_type):
        from transformers import DynamicCache
        attn = _make_attn(r_k=32, r_v=32, model_type=model_type)

        bsz, seq_len = 1, 5
        hidden = torch.randn(bsz, seq_len, attn.hidden_size)

        cache = DynamicCache()
        with torch.no_grad():
            out = attn(hidden_states=hidden, use_cache=True, past_key_value=cache)

        assert len(cache.key_cache) > 0
        k = cache.key_cache[0]
        v = cache.value_cache[0]
        assert k.shape[-1] == attn.r_k, f"[{model_type}] K last dim should be r_k={attn.r_k}, got {k.shape[-1]}"
        assert v.shape[-1] == attn.r_v, f"[{model_type}] V last dim should be r_v={attn.r_v}, got {v.shape[-1]}"

    def test_opt_incremental_decode_matches_full_recompute(self):
        attn = _make_attn(r_k=32, r_v=32, model_type="opt")

        prompt_len = 5
        hidden = torch.randn(1, prompt_len, attn.hidden_size)

        with torch.no_grad():
            prefill_out = attn(hidden_states=hidden, use_cache=True)

            new_hidden = torch.randn(1, 1, attn.hidden_size)

            incremental_out = attn(
                hidden_states=new_hidden,
                use_cache=True,
                past_key_value=prefill_out[2],
            )
            incremental_attn = incremental_out[0]

            full_hidden = torch.cat([hidden, new_hidden], dim=1)
            full_out = attn(hidden_states=full_hidden, use_cache=False)
            full_attn = full_out[0][:, -1:, :]

        max_diff = (incremental_attn - full_attn).abs().max().item()
        assert max_diff < 1e-4, f"[opt] Incremental vs full-recompute diff: {max_diff}"

    def test_llama_incremental_decode_matches_full_recompute(self):
        attn = _make_attn(r_k=32, r_v=32, model_type="llama")

        prompt_len = 5
        hidden = torch.randn(1, prompt_len, attn.hidden_size)

        with torch.no_grad():
            prefill_out = attn(hidden_states=hidden, use_cache=True)

            new_hidden = torch.randn(1, 1, attn.hidden_size)
            pos_ids = torch.tensor([[prompt_len]])

            incremental_out = attn(
                hidden_states=new_hidden,
                use_cache=True,
                past_key_value=prefill_out[2],
                position_ids=pos_ids,
            )
            incremental_attn = incremental_out[0]

            full_hidden = torch.cat([hidden, new_hidden], dim=1)
            full_out = attn(hidden_states=full_hidden, use_cache=False)
            full_attn = full_out[0][:, -1:, :]

        max_diff = (incremental_attn - full_attn).abs().max().item()
        assert max_diff < 1e-4, f"[llama] Incremental vs full-recompute diff: {max_diff}"

    @pytest.mark.parametrize("model_type", _MODEL_TYPES)
    def test_multi_step_decode_consistency(self, model_type):
        attn = _make_attn(r_k=24, r_v=24, model_type=model_type)

        prompt_len = 4
        n_decode = 3
        hidden = torch.randn(1, prompt_len, attn.hidden_size)

        with torch.no_grad():
            out = attn(hidden_states=hidden, use_cache=True)
            past_kv = out[2]

            for step in range(n_decode):
                new_hidden = torch.randn(1, 1, attn.hidden_size)
                kwargs = dict(hidden_states=new_hidden, use_cache=True, past_key_value=past_kv)
                if attn._use_rope:
                    kwargs["position_ids"] = torch.tensor([[prompt_len + step]])
                out = attn(**kwargs)
                past_kv = out[2]

    @pytest.mark.parametrize("model_type", _MODEL_TYPES)
    def test_kv_cache_memory_is_reduced(self, model_type):
        attn = _make_attn(r_k=32, r_v=32, model_type=model_type)
        head_dim = attn.head_dim

        seq_len = 10
        hidden = torch.randn(1, seq_len, attn.hidden_size)

        with torch.no_grad():
            out = attn(hidden_states=hidden, use_cache=True)
            k, v = out[2]

        low_rank_bytes = k.nelement() * k.element_size() + v.nelement() * v.element_size()
        n_kv = k.shape[1]
        elem_size = k.element_size()
        full_rank_bytes = 2 * n_kv * seq_len * head_dim * elem_size
        ratio = low_rank_bytes / full_rank_bytes
        expected_ratio = (attn.r_k + attn.r_v) / (2 * head_dim)
        assert abs(ratio - expected_ratio) < 0.05, f"[{model_type}] Expected ratio ~{expected_ratio:.2f}, got {ratio:.2f}"

    @pytest.mark.parametrize("model_type", _MODEL_TYPES)
    def test_full_rank_still_uses_head_dim(self, model_type):
        attn = _make_attn(r_k=64, r_v=64, model_type=model_type)
        assert attn.r_k == attn.head_dim

        seq_len = 5
        hidden = torch.randn(1, seq_len, attn.hidden_size)

        with torch.no_grad():
            out = attn(hidden_states=hidden, use_cache=True)

        k, v = out[2]
        assert k.shape[-1] == attn.head_dim
        assert v.shape[-1] == attn.head_dim

    def test_opt_model_returns_valid_cache_without_use_cache(self):
        attn = _make_attn(r_k=32, r_v=32, model_type="opt")
        from transformers import DynamicCache

        cache = DynamicCache()
        seq_len = 5
        hidden = torch.randn(1, seq_len, attn.hidden_size)

        with torch.no_grad():
            out = attn(hidden_states=hidden, past_key_value=cache)

        present_kv = out[2]
        assert present_kv is not None, "OPT model must return non-None past_key_value for decoder compatibility"


class TestRoPEPathSpecific:
    def test_llama_uses_rotary_emb(self):
        attn = _make_attn(r_k=32, r_v=32, model_type="llama")
        assert attn._use_rope is True
        assert hasattr(attn, "rotary_emb")

    def test_opt_does_not_use_rotary_emb(self):
        attn = _make_attn(r_k=32, r_v=32, model_type="opt")
        assert attn._use_rope is False
        assert not hasattr(attn, "rotary_emb")

    def test_llama_incremental_decode_rope_position_increments(self):
        attn = _make_attn(r_k=32, r_v=32, model_type="llama")

        prompt_len = 4
        hidden = torch.randn(1, prompt_len, attn.hidden_size)

        with torch.no_grad():
            prefill_out = attn(hidden_states=hidden, use_cache=True)
            past_kv = prefill_out[2]

            pos_ids_5 = torch.tensor([[prompt_len]])
            step1_out = attn(
                hidden_states=torch.randn(1, 1, attn.hidden_size),
                use_cache=True,
                past_key_value=past_kv,
                position_ids=pos_ids_5,
            )
            past_kv = step1_out[2]

            pos_ids_6 = torch.tensor([[prompt_len + 1]])
            step2_out = attn(
                hidden_states=torch.randn(1, 1, attn.hidden_size),
                use_cache=True,
                past_key_value=past_kv,
                position_ids=pos_ids_6,
            )

        assert step1_out[0].shape == (1, 1, attn.hidden_size)
        assert step2_out[0].shape == (1, 1, attn.hidden_size)
        assert not torch.allclose(step1_out[0], step2_out[0], atol=1e-6), (
            "Different position_ids should produce different outputs under RoPE"
        )

    def test_llama_multi_step_decode_vs_full_recompute(self):
        attn = _make_attn(r_k=24, r_v=24, model_type="llama")

        prompt_len = 4
        n_decode = 3
        torch.manual_seed(42)
        hidden = torch.randn(1, prompt_len, attn.hidden_size)
        decode_hiddens = [torch.randn(1, 1, attn.hidden_size) for _ in range(n_decode)]

        with torch.no_grad():
            prefill_out = attn(hidden_states=hidden, use_cache=True)
            past_kv = prefill_out[2]
            incremental_outputs = []

            for step in range(n_decode):
                pos_ids = torch.tensor([[prompt_len + step]])
                step_out = attn(
                    hidden_states=decode_hiddens[step],
                    use_cache=True,
                    past_key_value=past_kv,
                    position_ids=pos_ids,
                )
                past_kv = step_out[2]
                incremental_outputs.append(step_out[0])

            full_hidden = torch.cat([hidden] + decode_hiddens, dim=1)
            full_out = attn(hidden_states=full_hidden, use_cache=False)
            full_attn_last = full_out[0][:, -1:, :]

        max_diff = (incremental_outputs[-1] - full_attn_last).abs().max().item()
        assert max_diff < 1e-4, f"[llama] Multi-step decode vs full-recompute diff: {max_diff}"


class TestMixedRankCacheCompat:
    def test_full_rank_k_low_rank_v_past_handled_correctly(self):
        attn = _make_attn(r_k=32, r_v=32, model_type="opt")

        seq_len = 3
        hidden = torch.randn(1, seq_len, attn.hidden_size)

        full_rank_k = torch.randn(1, attn.num_key_value_heads, seq_len, attn.head_dim)
        low_rank_v = torch.randn(1, attn.num_key_value_heads, seq_len, attn.r_v)

        past_kv = (full_rank_k, low_rank_v)

        with torch.no_grad():
            out = attn(hidden_states=hidden, use_cache=True, past_key_value=past_kv)

        k, v = out[2]
        assert k.shape[-1] == attn.r_k
        assert v.shape[-1] == attn.r_v
        assert k.shape[2] == seq_len + seq_len

    def test_low_rank_k_full_rank_v_past_handled_correctly(self):
        attn = _make_attn(r_k=32, r_v=32, model_type="opt")

        seq_len = 3
        hidden = torch.randn(1, seq_len, attn.hidden_size)

        low_rank_k = torch.randn(1, attn.num_key_value_heads, seq_len, attn.r_k)
        full_rank_v = torch.randn(1, attn.num_key_value_heads, seq_len, attn.head_dim)

        past_kv = (low_rank_k, full_rank_v)

        with torch.no_grad():
            out = attn(hidden_states=hidden, use_cache=True, past_key_value=past_kv)

        k, v = out[2]
        assert k.shape[-1] == attn.r_k
        assert v.shape[-1] == attn.r_v
        assert k.shape[2] == seq_len + seq_len

    def test_invalid_k_dim_past_raises_clear_error(self):
        attn = _make_attn(r_k=32, r_v=32, model_type="opt")

        seq_len = 3
        hidden = torch.randn(1, seq_len, attn.hidden_size)

        invalid_k = torch.randn(1, attn.num_key_value_heads, seq_len, 48)
        low_rank_v = torch.randn(1, attn.num_key_value_heads, seq_len, attn.r_v)

        past_kv = (invalid_k, low_rank_v)

        with pytest.raises(ValueError, match="past_k last dim"):
            with torch.no_grad():
                attn(hidden_states=hidden, use_cache=True, past_key_value=past_kv)

    def test_invalid_v_dim_past_raises_clear_error(self):
        attn = _make_attn(r_k=32, r_v=32, model_type="opt")

        seq_len = 3
        hidden = torch.randn(1, seq_len, attn.hidden_size)

        low_rank_k = torch.randn(1, attn.num_key_value_heads, seq_len, attn.r_k)
        invalid_v = torch.randn(1, attn.num_key_value_heads, seq_len, 48)

        past_kv = (low_rank_k, invalid_v)

        with pytest.raises(ValueError, match="past_v last dim"):
            with torch.no_grad():
                attn(hidden_states=hidden, use_cache=True, past_key_value=past_kv)
