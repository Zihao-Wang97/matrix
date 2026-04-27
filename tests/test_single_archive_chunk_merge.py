from __future__ import annotations

from types import SimpleNamespace

import torch

from hawp_laq.modeling.attention_hawp import HAWPAttention
from hawp_laq.runtime.turboquant import TurboQuantProd


def _make_attn(n_heads=2, n_kv_heads=2, head_dim=4, r_k=4, r_v=4, recent_window=0):
    config = SimpleNamespace(
        hidden_size=n_heads * head_dim,
        num_attention_heads=n_heads,
        num_key_value_heads=n_kv_heads,
        max_position_embeddings=128,
        rope_theta=10000.0,
        model_type="llama",
        enable_bias=False,
        attention_dropout=0.0,
    )
    attn = HAWPAttention(
        config,
        layer_idx=0,
        r_k=r_k,
        r_v=r_v,
        use_archive_k_ip_approx=True,
    )
    kq = TurboQuantProd(dim=r_k, bits=8, use_rotation=False, group_size=r_k)
    vq = TurboQuantProd(dim=r_v, bits=8, use_rotation=False, group_size=r_v)
    attn.setup_quant_cache(kq, vq, recent_window=recent_window)
    return attn


def _constant_latents(nkv: int, n_tokens: int, dim: int, token_offset: int = 0, base: float = 1.0):
    x = torch.empty(1, nkv, n_tokens, dim, dtype=torch.float32)
    for h in range(nkv):
        for t in range(n_tokens):
            value = base + h * 1000.0 + token_offset + t
            x[0, h, t, :] = value
    return x


def test_recent_window_zero_multiple_appends_keep_single_archive_chunk():
    attn = _make_attn(recent_window=0)

    total = 0
    for n_tokens in [2, 3, 1, 4]:
        k_lat = torch.randn(1, attn.num_key_value_heads, n_tokens, attn.r_k)
        v_lat = torch.randn(1, attn.num_key_value_heads, n_tokens, attn.r_v)
        attn._quant_cache_append_latent(k_lat, v_lat)
        total += n_tokens
        assert len(attn._quant_archive_chunks) == 1
        assert attn._quant_archive_chunks[0].n_tokens == total
        chunk = attn._quant_archive_chunks[0]
        assert chunk.k_qx.logical_shape == (attn.num_key_value_heads, total, attn.r_k)
        assert chunk.v_qx.logical_shape == (attn.num_key_value_heads, total, attn.r_v)


def test_recent_window_demotes_keep_single_archive_chunk():
    attn = _make_attn(recent_window=2)

    for _ in range(4):
        k_lat = torch.randn(1, attn.num_key_value_heads, 3, attn.r_k)
        v_lat = torch.randn(1, attn.num_key_value_heads, 3, attn.r_v)
        attn._quant_cache_append_latent(k_lat, v_lat)

    assert len(attn._quant_archive_chunks) == 1
    chunk = attn._quant_archive_chunks[0]
    assert chunk.n_tokens == 10
    assert chunk.k_qx.logical_shape == (attn.num_key_value_heads, 10, attn.r_k)
    assert chunk.v_qx.logical_shape == (attn.num_key_value_heads, 10, attn.r_v)
    assert attn._quant_recent_k.shape[1] == 2


def test_head_wise_merge_preserves_dequant_archive_k_order():
    attn = _make_attn(n_heads=2, n_kv_heads=2, recent_window=0)
    nkv = attn.num_key_value_heads

    k1 = _constant_latents(nkv, 2, attn.r_k, token_offset=0, base=1.0)
    v1 = _constant_latents(nkv, 2, attn.r_v, token_offset=0, base=100.0)
    k2 = _constant_latents(nkv, 3, attn.r_k, token_offset=2, base=1.0)
    v2 = _constant_latents(nkv, 3, attn.r_v, token_offset=2, base=100.0)

    attn._quant_cache_append_to_archive(k1, v1)
    attn._quant_cache_append_to_archive(k2, v2)

    assert len(attn._quant_archive_chunks) == 1
    chunk = attn._quant_archive_chunks[0]
    assert chunk.k_qx.logical_shape == (nkv, 5, attn.r_k)
    assert chunk.v_qx.logical_shape == (nkv, 5, attn.r_v)
    expected_k = torch.cat([k1[0], k2[0]], dim=1)
    actual_k = attn._dequant_archive_k()
    assert actual_k.shape == expected_k.shape
    assert torch.allclose(actual_k, expected_k, atol=1e-5)


def test_quant_cache_get_kv_returns_archive_then_recent():
    attn = _make_attn(n_heads=2, n_kv_heads=2, recent_window=4)
    nkv = attn.num_key_value_heads

    k_archive = _constant_latents(nkv, 2, attn.r_k, token_offset=0, base=1.0)
    v_archive = _constant_latents(nkv, 2, attn.r_v, token_offset=0, base=100.0)
    k_recent = _constant_latents(nkv, 3, attn.r_k, token_offset=2, base=1.0)
    v_recent = _constant_latents(nkv, 3, attn.r_v, token_offset=2, base=100.0)

    attn._quant_cache_append_to_archive(k_archive, v_archive)
    attn._quant_cache_append(k_recent, v_recent)

    k_full, v_full = attn._quant_cache_get_kv()
    assert torch.allclose(k_full, torch.cat([k_archive[0], k_recent[0]], dim=1), atol=1e-5)
    assert torch.allclose(v_full, torch.cat([v_archive[0], v_recent[0]], dim=1), atol=1e-5)


def test_archive_k_logits_approx_runs_with_single_merged_chunk():
    attn = _make_attn(n_heads=4, n_kv_heads=2, recent_window=0)
    nkv = attn.num_key_value_heads

    attn._quant_cache_append_to_archive(
        torch.randn(1, nkv, 2, attn.r_k),
        torch.randn(1, nkv, 2, attn.r_v),
    )
    attn._quant_cache_append_to_archive(
        torch.randn(1, nkv, 3, attn.r_k),
        torch.randn(1, nkv, 3, attn.r_v),
    )

    assert len(attn._quant_archive_chunks) == 1
    q_lat = torch.randn(1, attn.num_heads, 2, attn.r_k)
    logits = attn._compute_archive_k_logits_approx(q_lat)
    assert logits.shape == (1, attn.num_heads, 2, 5)


def test_drop_paths_preserve_logical_shape_metadata():
    attn = _make_attn(n_heads=2, n_kv_heads=2, recent_window=0)
    nkv = attn.num_key_value_heads

    attn._quant_cache_append_to_archive(
        torch.randn(1, nkv, 6, attn.r_k),
        torch.randn(1, nkv, 6, attn.r_v),
    )

    assert attn.drop_oldest_from_archive(2) == 2
    chunk = attn._quant_archive_chunks[0]
    assert chunk.k_qx.logical_shape == (nkv, 4, attn.r_k)
    assert chunk.v_qx.logical_shape == (nkv, 4, attn.r_v)

    assert attn.drop_least_important_from_archive(1) == 1
    chunk = attn._quant_archive_chunks[0]
    assert chunk.k_qx.logical_shape == (nkv, 3, attn.r_k)
    assert chunk.v_qx.logical_shape == (nkv, 3, attn.r_v)
