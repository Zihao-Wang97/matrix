from __future__ import annotations

import math

import pytest
import torch

from hawp_laq.modeling.attention_hawp import HAWPAttention


def _make_attn(r_k=8, r_v=8, n_heads=2, head_dim=16, model_type="opt", **kwargs):
    from types import SimpleNamespace

    config = SimpleNamespace(
        hidden_size=n_heads * head_dim,
        num_attention_heads=n_heads,
        num_key_value_heads=n_heads,
        max_position_embeddings=2048,
        rope_theta=10000.0,
        model_type=model_type,
        enable_bias=False,
        attention_dropout=0.0,
    )
    return HAWPAttention(config, r_k=r_k, r_v=r_v, **kwargs)


class TestOptNoDoubleScaling:
    def test_opt_dh_mode_undoes_pre_scaling(self):
        attn = _make_attn(r_k=8, r_v=8, model_type="opt",
                          logit_scale_mode="dh", gamma_mode="off")
        q_lat = torch.randn(1, 2, 1, 8)

        scale = attn._compute_low_rank_logit_scale(q_lat)

        expected = math.sqrt(16) * (1.0 / math.sqrt(16))
        assert torch.isclose(scale, torch.tensor(expected), atol=1e-6)
        assert torch.isclose(scale, torch.tensor(1.0), atol=1e-6)

    def test_opt_rk_mode_applies_sqrt_dh_over_sqrt_rk(self):
        r_k = 8
        attn = _make_attn(r_k=r_k, r_v=8, model_type="opt",
                          logit_scale_mode="rk", gamma_mode="off")
        q_lat = torch.randn(1, 2, 1, r_k)

        scale = attn._compute_low_rank_logit_scale(q_lat)

        expected = math.sqrt(16) / math.sqrt(r_k)
        assert torch.isclose(scale, torch.tensor(expected), atol=1e-6)

    def test_non_opt_dh_no_extra_undo(self):
        attn = _make_attn(r_k=8, r_v=8, model_type="llama",
                          logit_scale_mode="dh", gamma_mode="off")
        q_lat = torch.randn(1, 2, 1, 8)

        scale = attn._compute_low_rank_logit_scale(q_lat)

        expected = 1.0 / math.sqrt(16)
        assert torch.isclose(scale, torch.tensor(expected), atol=1e-6)

    def test_non_opt_rk_no_extra_undo(self):
        r_k = 8
        attn = _make_attn(r_k=r_k, r_v=8, model_type="llama",
                          logit_scale_mode="rk", gamma_mode="off")
        q_lat = torch.randn(1, 2, 1, r_k)

        scale = attn._compute_low_rank_logit_scale(q_lat)

        expected = 1.0 / math.sqrt(r_k)
        assert torch.isclose(scale, torch.tensor(expected), atol=1e-6)

    def test_opt_dh_net_effect_is_1_over_sqrt_dh_on_original_q(self):
        attn = _make_attn(r_k=8, r_v=8, model_type="opt",
                          logit_scale_mode="dh", gamma_mode="off")

        q_orig = torch.randn(1, 2, 1, 16)
        q_prescaled = q_orig * (1.0 / math.sqrt(16))

        pk_down = attn.p_k[:, :8]
        q_lat = q_prescaled @ pk_down
        k_lat = torch.randn(1, 2, 5, 8)

        scale = attn._compute_low_rank_logit_scale(q_lat)
        logits = (q_lat @ k_lat.transpose(-2, -1)) * scale

        q_full_attn = q_orig @ pk_down
        logits_ref = q_full_attn @ k_lat.transpose(-2, -1) / math.sqrt(16)

        assert torch.allclose(logits, logits_ref, atol=1e-5)
