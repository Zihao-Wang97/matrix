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


def test_quant_cache_summary_counts_quant_archive():
    attn = _make_hawp_quant_all(n_heads=2, head_dim=16, r_k=8, r_v=8)

    nkv, seq, rk, rv = 2, 10, 8, 8
    k_lat = torch.randn(1, nkv, seq, rk)
    v_lat = torch.randn(1, nkv, seq, rv)

    s0 = attn.quant_cache_summary()
    assert s0["recent_fp_bytes"] == 0
    assert s0["archive_quant_bytes"] == 0
    assert s0["archive_meta_bytes"] == 0
    assert s0["total_runtime_bytes"] == 0
    assert s0["compressed_storage_bytes"] == 0
    assert "archive_raw_bytes" not in s0

    attn._quant_cache_append_to_archive(k_lat, v_lat)
    s1 = attn.quant_cache_summary()

    assert s1["archive_quant_bytes"] > 0
    assert s1["archive_meta_bytes"] > 0

    assert s1["recent_fp_bytes"] == 0
    assert s1["total_runtime_bytes"] == s1["recent_fp_bytes"] + s1["archive_quant_bytes"] + s1["archive_meta_bytes"]
    assert s1["compressed_storage_bytes"] == s1["archive_quant_bytes"]

    k_lat_2 = torch.randn(1, nkv, 5, rk)
    v_lat_2 = torch.randn(1, nkv, 5, rv)
    attn._quant_cache_append_to_archive(k_lat_2, v_lat_2)
    s2 = attn.quant_cache_summary()

    assert s2["archive_quant_bytes"] > 0
    assert s2["archive_meta_bytes"] > 0
    assert s2["total_runtime_bytes"] == s2["recent_fp_bytes"] + s2["archive_quant_bytes"] + s2["archive_meta_bytes"]
    assert s2["compressed_storage_bytes"] == s2["archive_quant_bytes"]

    assert "archive_raw_bytes" not in s2
    assert "total_bytes" not in s2
    assert "recent_bytes" not in s2
    assert "archive_bytes" not in s2
