from __future__ import annotations

from types import SimpleNamespace

import torch

from hawp_laq.modeling.attention_hawp import HAWPAttention
from hawp_laq.runtime.turboquant import TurboQuantProd


def _make_attn(
    n_heads: int = 2,
    n_kv_heads: int = 2,
    head_dim: int = 4,
    r_k: int = 4,
    r_v: int = 4,
    recent_window: int = 3,
    model_type: str = "llama",
) -> HAWPAttention:
    config = SimpleNamespace(
        hidden_size=n_heads * head_dim,
        num_attention_heads=n_heads,
        num_key_value_heads=n_kv_heads,
        max_position_embeddings=128,
        rope_theta=10000.0,
        model_type=model_type,
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
            x[0, h, t, :] = base + h * 1000.0 + token_offset + t
    return x


def test_recent_window_zero_all_tokens_go_directly_to_archive():
    attn = _make_attn(recent_window=0)
    nkv = attn.num_key_value_heads

    k_lat = _constant_latents(nkv, 5, attn.r_k)
    v_lat = _constant_latents(nkv, 5, attn.r_v, base=100.0)
    attn._quant_cache_append_latent(k_lat, v_lat)

    assert attn._recent_count == 0
    assert attn._get_recent_k() is None
    assert len(attn._quant_archive_chunks) == 1
    assert attn.n_archive_tokens == 5


def test_recent_less_than_window_keeps_time_order_without_archive():
    attn = _make_attn(recent_window=5)
    nkv = attn.num_key_value_heads

    k_lat = _constant_latents(nkv, 3, attn.r_k)
    v_lat = _constant_latents(nkv, 3, attn.r_v, base=100.0)
    attn._quant_cache_append_latent(k_lat, v_lat)

    assert not attn._quant_archive_chunks
    assert attn._recent_count == 3
    assert torch.allclose(attn._get_recent_k(), k_lat[0])
    assert torch.allclose(attn._get_recent_v(), v_lat[0])
    summary = attn.quant_cache_summary()
    assert summary["recent_active_bytes"] < summary["recent_alloc_bytes"]


def test_recent_equal_window_is_full_without_archive():
    attn = _make_attn(recent_window=3)
    nkv = attn.num_key_value_heads

    k_lat = _constant_latents(nkv, 3, attn.r_k)
    v_lat = _constant_latents(nkv, 3, attn.r_v, base=100.0)
    attn._quant_cache_append_latent(k_lat, v_lat)

    assert not attn._quant_archive_chunks
    assert attn._recent_count == 3
    assert torch.allclose(attn._get_recent_k(), k_lat[0])
    assert torch.allclose(attn._get_recent_v(), v_lat[0])
    summary = attn.quant_cache_summary()
    assert summary["recent_active_bytes"] == summary["recent_alloc_bytes"]


def test_more_than_window_archives_old_tokens_and_keeps_recent_tail():
    attn = _make_attn(recent_window=3)
    nkv = attn.num_key_value_heads

    k_lat = _constant_latents(nkv, 7, attn.r_k)
    v_lat = _constant_latents(nkv, 7, attn.r_v, base=100.0)
    attn._quant_cache_append_latent(k_lat, v_lat)

    assert len(attn._quant_archive_chunks) == 1
    assert attn.n_archive_tokens == 4
    assert attn._recent_count == 3
    assert torch.allclose(attn._dequant_archive_k(), k_lat[0, :, :4, :], atol=1e-5)
    assert torch.allclose(attn._get_recent_k(), k_lat[0, :, 4:, :])
    assert torch.allclose(attn._get_recent_v(), v_lat[0, :, 4:, :])


def test_prefill_fast_path_archives_prefix_and_keeps_recent_tail():
    attn = _make_attn(recent_window=4)
    nkv = attn.num_key_value_heads

    k_lat = _constant_latents(nkv, 10, attn.r_k)
    v_lat = _constant_latents(nkv, 10, attn.r_v, base=100.0)
    attn._quant_cache_append_latent(k_lat, v_lat)

    assert attn._recent_count == 4
    assert attn.n_archive_tokens == 6
    assert len(attn._quant_archive_chunks) == 1
    assert torch.allclose(attn._get_recent_k(), k_lat[0, :, 6:, :])
    assert torch.allclose(attn._get_recent_v(), v_lat[0, :, 6:, :])
    assert torch.allclose(attn._dequant_archive_k(), k_lat[0, :, :6, :], atol=1e-5)


def test_prefill_fast_path_not_used_when_recent_non_empty():
    attn = _make_attn(recent_window=4)
    nkv = attn.num_key_value_heads

    k_first = _constant_latents(nkv, 2, attn.r_k)
    v_first = _constant_latents(nkv, 2, attn.r_v, base=100.0)
    k_next = _constant_latents(nkv, 6, attn.r_k, token_offset=2)
    v_next = _constant_latents(nkv, 6, attn.r_v, token_offset=2, base=100.0)

    attn._quant_cache_append_latent(k_first, v_first)
    attn._quant_cache_append_latent(k_next, v_next)

    expected_k = torch.cat([k_first[0], k_next[0]], dim=1)
    expected_v = torch.cat([v_first[0], v_next[0]], dim=1)

    assert attn.n_archive_tokens == 4
    assert attn._recent_count == 4
    assert len(attn._quant_archive_chunks) <= 1
    assert torch.allclose(attn._get_recent_k(), expected_k[:, -4:, :])
    assert torch.allclose(attn._get_recent_v(), expected_v[:, -4:, :])


def test_prefill_equal_window_does_not_archive():
    attn = _make_attn(recent_window=4)
    nkv = attn.num_key_value_heads

    k_lat = _constant_latents(nkv, 4, attn.r_k)
    v_lat = _constant_latents(nkv, 4, attn.r_v, base=100.0)
    attn._quant_cache_append_latent(k_lat, v_lat)

    assert attn.n_archive_tokens == 0
    assert attn._recent_count == 4
    assert torch.allclose(attn._get_recent_k(), k_lat[0])
    assert torch.allclose(attn._get_recent_v(), v_lat[0])


def test_prefill_less_than_window_does_not_archive():
    attn = _make_attn(recent_window=4)
    nkv = attn.num_key_value_heads

    k_lat = _constant_latents(nkv, 3, attn.r_k)
    v_lat = _constant_latents(nkv, 3, attn.r_v, base=100.0)
    attn._quant_cache_append_latent(k_lat, v_lat)

    assert attn.n_archive_tokens == 0
    assert attn._recent_count == 3
    assert torch.allclose(attn._get_recent_k(), k_lat[0])
    assert torch.allclose(attn._get_recent_v(), v_lat[0])


def test_multi_step_decode_keeps_last_window_and_single_archive_chunk():
    attn = _make_attn(recent_window=3)
    nkv = attn.num_key_value_heads

    all_k = []
    all_v = []
    for t in range(8):
        k_lat = _constant_latents(nkv, 1, attn.r_k, token_offset=t)
        v_lat = _constant_latents(nkv, 1, attn.r_v, token_offset=t, base=100.0)
        all_k.append(k_lat)
        all_v.append(v_lat)
        attn._quant_cache_append_latent(k_lat, v_lat)

    expected_k = torch.cat([x[0] for x in all_k], dim=1)
    expected_v = torch.cat([x[0] for x in all_v], dim=1)

    assert len(attn._quant_archive_chunks) == 1
    assert attn.n_archive_tokens == 5
    assert attn._recent_count == 3
    assert torch.allclose(attn._dequant_archive_k(), expected_k[:, :5, :], atol=1e-5)
    assert torch.allclose(attn._get_recent_k(), expected_k[:, 5:, :])
    assert torch.allclose(attn._get_recent_v(), expected_v[:, 5:, :])


def test_ring_wrap_returns_recent_in_time_order():
    attn = _make_attn(recent_window=3)
    nkv = attn.num_key_value_heads

    k_first = _constant_latents(nkv, 3, attn.r_k)
    v_first = _constant_latents(nkv, 3, attn.r_v, base=100.0)
    attn._quant_cache_append_latent(k_first, v_first)

    k_next = _constant_latents(nkv, 2, attn.r_k, token_offset=3)
    v_next = _constant_latents(nkv, 2, attn.r_v, token_offset=3, base=100.0)
    attn._quant_cache_append_latent(k_next, v_next)

    expected_k = torch.cat([k_first[0], k_next[0]], dim=1)[:, -3:, :]
    expected_v = torch.cat([v_first[0], v_next[0]], dim=1)[:, -3:, :]

    assert attn._recent_start != 0
    assert torch.allclose(attn._get_recent_k(), expected_k)
    assert torch.allclose(attn._get_recent_v(), expected_v)


def test_internal_quant_cache_forward_attention_length_archive_recent_current():
    attn = _make_attn(n_heads=2, n_kv_heads=2, recent_window=3, model_type="opt")
    nkv = attn.num_key_value_heads

    k_lat = _constant_latents(nkv, 5, attn.r_k)
    v_lat = _constant_latents(nkv, 5, attn.r_v, base=100.0)
    attn._quant_cache_append_latent(k_lat, v_lat)

    assert attn.n_archive_tokens == 2
    assert attn._recent_count == 3

    x = torch.randn(1, 1, attn.hidden_size)
    with torch.no_grad():
        _, attn_weights, _ = attn(x, attention_mask=None, use_cache=True)

    assert attn_weights.shape[-1] == 2 + 3 + 1
