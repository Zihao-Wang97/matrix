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


def test_profile_reports_runtime_and_compressed_bytes():
    from hawp_laq.eval.metrics import collect_kv_metrics, format_kv_metrics

    attn = _make_hawp_quant_all(n_heads=2, head_dim=16, r_k=8, r_v=8)
    nkv, seq, rk, rv = 2, 10, 8, 8
    k_lat = torch.randn(1, nkv, seq, rk)
    v_lat = torch.randn(1, nkv, seq, rv)
    attn._quant_cache_append_to_archive(k_lat, v_lat)

    class _FakeModel:
        def modules(self):
            yield attn

    attn.use_cache_manager = True
    metrics = collect_kv_metrics(_FakeModel())

    assert "total_runtime_bytes" in metrics
    assert "compressed_storage_bytes" in metrics
    assert "archive_quant_bytes" in metrics
    assert "archive_meta_bytes" in metrics
    assert "recent_fp_bytes" in metrics
    assert "runtime_saving_ratio" in metrics
    assert "compressed_saving_ratio" in metrics
    assert metrics["total_runtime_bytes"] > 0
    assert metrics["compressed_storage_bytes"] > 0
    assert metrics["compressed_storage_bytes"] <= metrics["total_runtime_bytes"]
    assert "archive_raw_bytes" not in metrics

    assert "total_bytes" not in metrics
    assert "recent_bytes" not in metrics
    assert "archive_bytes" not in metrics

    text = format_kv_metrics(metrics)
    assert "[runtime]" in text
    assert "[compressed storage]" in text
    assert "runtime=" in text
    assert "compressed=" in text
