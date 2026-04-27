from __future__ import annotations

import math

import pytest
import torch

from hawp_laq.modeling.attention_hawp import HAWPAttention


def _make_attn(r_k=8, r_v=8, n_heads=2, head_dim=16, **kwargs):
    from types import SimpleNamespace

    config = SimpleNamespace(
        hidden_size=n_heads * head_dim,
        num_attention_heads=n_heads,
        num_key_value_heads=n_heads,
        max_position_embeddings=2048,
        rope_theta=10000.0,
        model_type="llama",
        enable_bias=False,
        attention_dropout=0.0,
    )
    return HAWPAttention(config, r_k=r_k, r_v=r_v, **kwargs)


class TestLogitScaleModeDhVsRk:
    def test_dh_and_rk_produce_different_logits(self):
        n_heads, head_dim, r_k = 2, 16, 8
        attn_dh = _make_attn(r_k=r_k, r_v=8, logit_scale_mode="dh", gamma_mode="off")
        attn_rk = _make_attn(r_k=r_k, r_v=8, logit_scale_mode="rk", gamma_mode="off")

        q_lat = torch.randn(1, n_heads, 1, r_k)
        k_lat = torch.randn(1, n_heads, 5, r_k)

        scale_dh = attn_dh._compute_low_rank_logit_scale(q_lat)
        scale_rk = attn_rk._compute_low_rank_logit_scale(q_lat)

        expected_dh = 1.0 / math.sqrt(head_dim)
        expected_rk = 1.0 / math.sqrt(r_k)

        assert torch.isclose(scale_dh, torch.tensor(expected_dh), atol=1e-6)
        assert torch.isclose(scale_rk, torch.tensor(expected_rk), atol=1e-6)

        logits_dh = (q_lat @ k_lat.transpose(-2, -1)) * scale_dh
        logits_rk = (q_lat @ k_lat.transpose(-2, -1)) * scale_rk

        assert not torch.allclose(logits_dh, logits_rk)

    def test_dh_equals_standard_attention_temperature(self):
        attn = _make_attn(r_k=8, r_v=8, logit_scale_mode="dh", gamma_mode="off")
        q_lat = torch.randn(1, 2, 1, 8)
        scale = attn._compute_low_rank_logit_scale(q_lat)
        assert torch.isclose(scale, torch.tensor(1.0 / math.sqrt(16)), atol=1e-6)

    def test_rk_scale_depends_on_rank(self):
        attn_r4 = _make_attn(r_k=4, r_v=4, logit_scale_mode="rk", gamma_mode="off")
        attn_r8 = _make_attn(r_k=8, r_v=8, logit_scale_mode="rk", gamma_mode="off")

        q = torch.randn(1, 2, 1, 4)
        scale_4 = attn_r4._compute_low_rank_logit_scale(q)
        q2 = torch.randn(1, 2, 1, 8)
        scale_8 = attn_r8._compute_low_rank_logit_scale(q2)

        assert torch.isclose(scale_4, torch.tensor(1.0 / math.sqrt(4)), atol=1e-6)
        assert torch.isclose(scale_8, torch.tensor(1.0 / math.sqrt(8)), atol=1e-6)

    def test_full_rank_dh_equals_standard_temperature(self):
        attn = _make_attn(r_k=16, r_v=16, head_dim=16, logit_scale_mode="dh", gamma_mode="off")
        q_lat = torch.randn(1, 2, 1, 16)
        scale = attn._compute_low_rank_logit_scale(q_lat)
        assert torch.isclose(scale, torch.tensor(1.0 / math.sqrt(16)), atol=1e-6)

    def test_full_rank_rk_equals_dh_when_rk_equals_dh(self):
        attn = _make_attn(r_k=16, r_v=16, head_dim=16, logit_scale_mode="rk", gamma_mode="off")
        q_lat = torch.randn(1, 2, 1, 16)
        scale = attn._compute_low_rank_logit_scale(q_lat)
        assert torch.isclose(scale, torch.tensor(1.0 / math.sqrt(16)), atol=1e-6)

    def test_invalid_mode_raises(self):
        attn = _make_attn(r_k=8, r_v=8, logit_scale_mode="bogus", gamma_mode="off")
        q_lat = torch.randn(1, 2, 1, 8)
        with pytest.raises(ValueError, match="logit_scale_mode"):
            attn._compute_low_rank_logit_scale(q_lat)
