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


class TestGammaOffVsFixed:
    def test_gamma_off_no_gamma_in_scale(self):
        attn = _make_attn(r_k=8, r_v=8, logit_scale_mode="dh", gamma_mode="off")
        q_lat = torch.randn(1, 2, 1, 8)
        scale = attn._compute_low_rank_logit_scale(q_lat)
        expected = 1.0 / math.sqrt(16)
        assert torch.isclose(scale, torch.tensor(expected), atol=1e-6)

    def test_gamma_fixed_multiplies_gamma_value(self):
        gamma_val = 2.5
        attn = _make_attn(r_k=8, r_v=8, logit_scale_mode="dh", gamma_mode="fixed",
                          gamma_value=gamma_val)
        q_lat = torch.randn(1, 2, 1, 8)
        scale = attn._compute_low_rank_logit_scale(q_lat)
        expected = gamma_val / math.sqrt(16)
        assert torch.isclose(scale, torch.tensor(expected), atol=1e-5)

    def test_gamma_fixed_uses_module_gamma_when_value_is_none(self):
        attn = _make_attn(r_k=8, r_v=8, logit_scale_mode="dh", gamma_mode="fixed",
                          gamma_value=None)
        attn.gamma.data.fill_(3.0)
        q_lat = torch.randn(1, 2, 1, 8)
        scale = attn._compute_low_rank_logit_scale(q_lat)
        expected = 3.0 / math.sqrt(16)
        assert torch.isclose(scale, torch.tensor(expected), atol=1e-6)

    def test_gamma_off_and_fixed_produce_different_logits(self):
        attn_off = _make_attn(r_k=8, r_v=8, logit_scale_mode="dh", gamma_mode="off")
        attn_fixed = _make_attn(r_k=8, r_v=8, logit_scale_mode="dh", gamma_mode="fixed",
                                 gamma_value=2.0)

        q_lat = torch.randn(1, 2, 1, 8)
        k_lat = torch.randn(1, 2, 5, 8)

        raw = q_lat @ k_lat.transpose(-2, -1)
        logits_off = raw * attn_off._compute_low_rank_logit_scale(q_lat)
        logits_fixed = raw * attn_fixed._compute_low_rank_logit_scale(q_lat)

        assert not torch.allclose(logits_off, logits_fixed)
        assert torch.allclose(logits_fixed, logits_off * 2.0, atol=1e-5)

    def test_gamma_value_one_equals_off(self):
        attn_off = _make_attn(r_k=8, r_v=8, logit_scale_mode="dh", gamma_mode="off")
        attn_fixed1 = _make_attn(r_k=8, r_v=8, logit_scale_mode="dh", gamma_mode="fixed",
                                  gamma_value=1.0)

        q_lat = torch.randn(1, 2, 1, 8)
        scale_off = attn_off._compute_low_rank_logit_scale(q_lat)
        scale_fixed = attn_fixed1._compute_low_rank_logit_scale(q_lat)
        assert torch.isclose(scale_off, scale_fixed, atol=1e-6)


class TestGammaLearned:
    def test_gamma_learned_applies_module_gamma(self):
        attn = _make_attn(r_k=8, r_v=8, logit_scale_mode="dh", gamma_mode="learned")
        attn.gamma.data.fill_(2.5)
        q_lat = torch.randn(1, 2, 1, 8)
        scale = attn._compute_low_rank_logit_scale(q_lat)
        expected = 2.5 / math.sqrt(16)
        assert torch.isclose(scale, torch.tensor(expected), atol=1e-5)

    def test_gamma_learned_with_gamma_one_equals_off(self):
        attn_off = _make_attn(r_k=8, r_v=8, logit_scale_mode="dh", gamma_mode="off")
        attn_learned = _make_attn(r_k=8, r_v=8, logit_scale_mode="dh", gamma_mode="learned")
        attn_learned.gamma.data.fill_(1.0)
        q_lat = torch.randn(1, 2, 1, 8)
        scale_off = attn_off._compute_low_rank_logit_scale(q_lat)
        scale_learned = attn_learned._compute_low_rank_logit_scale(q_lat)
        assert torch.isclose(scale_off, scale_learned, atol=1e-6)

    def test_gamma_learned_equals_fixed_with_same_value(self):
        gamma_val = 1.8
        attn_learned = _make_attn(r_k=8, r_v=8, logit_scale_mode="dh", gamma_mode="learned")
        attn_learned.gamma.data.fill_(gamma_val)
        attn_fixed = _make_attn(r_k=8, r_v=8, logit_scale_mode="dh", gamma_mode="fixed",
                                gamma_value=gamma_val)
        q_lat = torch.randn(1, 2, 1, 8)
        scale_learned = attn_learned._compute_low_rank_logit_scale(q_lat)
        scale_fixed = attn_fixed._compute_low_rank_logit_scale(q_lat)
        assert torch.isclose(scale_learned, scale_fixed, atol=1e-6)
