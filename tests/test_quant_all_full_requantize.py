from __future__ import annotations

import pytest
import torch

from hawp_laq.modeling.attention_hawp import HAWPAttention


def _make_hawp_quant_all(n_heads=4, head_dim=16, r_k=8, r_v=8):
    from types import SimpleNamespace
    from hawp_laq.runtime.turboquant import TurboQuantMSE

    config = SimpleNamespace(
        hidden_size=n_heads * head_dim,
        num_attention_heads=n_heads,
        num_key_value_heads=n_heads,
        max_position_embeddings=2048,
        rope_theta=10000.0,
        model_type="opt",
        enable_bias=False,
        attention_dropout=0.0,
    )
    attn = HAWPAttention(config, r_k=r_k, r_v=r_v)
    k_q = TurboQuantMSE(dim=r_k, bits=4, group_size=16, use_rotation=False)
    v_q = TurboQuantMSE(dim=r_v, bits=4, group_size=16, use_rotation=False)
    attn.setup_quant_cache(k_q, v_q, recent_window=0)
    return attn


def test_quant_all_chunked_archive_after_append():
    attn = _make_hawp_quant_all(n_heads=2, head_dim=16, r_k=8, r_v=8)

    nkv, seq, rk, rv = 2, 10, 8, 8
    k_lat = torch.randn(1, nkv, seq, rk)
    v_lat = torch.randn(1, nkv, seq, rv)

    attn._quant_cache_append_to_archive(k_lat, v_lat)

    assert len(attn._quant_archive_chunks) == 1
    chunk = attn._quant_archive_chunks[0]
    assert chunk.n_tokens == seq
    assert chunk.k_qx is not None
    assert chunk.v_qx is not None
    assert chunk.k_norms is not None

    k_deq_1, v_deq_1 = attn._quant_cache_get_kv()
    assert k_deq_1.shape == (nkv, seq, rk)
    assert v_deq_1.shape == (nkv, seq, rv)

    k_lat_2 = torch.randn(1, nkv, 5, rk)
    v_lat_2 = torch.randn(1, nkv, 5, rv)
    attn._quant_cache_append_to_archive(k_lat_2, v_lat_2)

    assert len(attn._quant_archive_chunks) == 1
    assert attn._quant_archive_chunks[0].n_tokens == 15

    k_deq_2, v_deq_2 = attn._quant_cache_get_kv()
    assert k_deq_2.shape == (nkv, 15, rk)
    assert v_deq_2.shape == (nkv, 15, rv)

    assert torch.isfinite(k_deq_2).all()
    assert torch.isfinite(v_deq_2).all()


def test_archive_chunked_roundtrip_after_multiple_appends():
    attn = _make_hawp_quant_all(n_heads=2, head_dim=16, r_k=8, r_v=8)

    nkv, rk, rv = 2, 8, 8
    total_tokens = 0
    append_sizes = [8, 4, 3, 2, 1, 5]

    for n in append_sizes:
        k_lat = torch.randn(1, nkv, n, rk)
        v_lat = torch.randn(1, nkv, n, rv)
        attn._quant_cache_append_to_archive(k_lat, v_lat)
        total_tokens += n

        k_deq, v_deq = attn._quant_cache_get_kv()
        assert k_deq.shape == (nkv, total_tokens, rk)
        assert v_deq.shape == (nkv, total_tokens, rv)

    assert len(attn._quant_archive_chunks) == 1
    assert attn._quant_archive_chunks[0].n_tokens == sum(append_sizes)

    k_deq, v_deq = attn._quant_cache_get_kv()
    assert k_deq.shape[1] == sum(append_sizes)
    assert v_deq.shape[1] == sum(append_sizes)

    assert torch.isfinite(k_deq).all()
    assert torch.isfinite(v_deq).all()
