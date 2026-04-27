from __future__ import annotations

import math
import torch
import pytest
from types import SimpleNamespace

from hawp_laq.modeling.attention_hawp import HAWPAttention, _make_causal_mask


def _make_config():
    return SimpleNamespace(
        hidden_size=256,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=512,
        rope_theta=10000.0,
        model_type="opt",
        enable_bias=True,
        attention_dropout=0.0,
    )


def _make_attn(r_k=None, r_v=None):
    cfg = _make_config()
    return HAWPAttention(cfg, layer_idx=0, r_k=r_k, r_v=r_v)


class TestLowRankProjectionReconstructionFormula:
    def test_apply_pk_uses_pk_down_transpose(self):
        attn = _make_attn(r_k=16, r_v=16)
        attn.p_k.data.normal_()
        attn.eval()

        x = torch.randn(1, 4, 4, 64)
        pk_down = attn.p_k[:, :attn.r_k]
        expected = x @ pk_down @ pk_down.T
        result = attn._apply_pk(x)
        assert torch.allclose(result, expected, atol=1e-5), "_apply_pk must use P_down @ P_down^T"

    def test_apply_pv_uses_pv_down_transpose(self):
        attn = _make_attn(r_k=16, r_v=16)
        attn.p_v.data.normal_()
        attn.eval()

        x = torch.randn(1, 4, 4, 64)
        pv_down = attn.p_v[:, :attn.r_v]
        expected = x @ pv_down @ pv_down.T
        result = attn._apply_pv(x)
        assert torch.allclose(result, expected, atol=1e-5), "_apply_pv must use P_down @ P_down^T"

    def test_low_rank_output_uses_pv_down_transpose(self):
        attn = _make_attn(r_k=16, r_v=16)
        attn.p_k.data.normal_()
        attn.p_v.data.normal_()
        attn.gamma.data.fill_(1.0)
        attn.eval()

        x = torch.randn(1, 4, 256)
        with torch.no_grad():
            out = attn(x, attention_mask=None)

        pv_down = attn.p_v[:, :attn.r_v]
        assert pv_down.shape == (64, 16)
        assert pv_down.T.shape == (16, 64)


class TestGammaAppliedOnLogitsNotOutput:
    def test_gamma_does_not_multiply_output(self):
        attn = _make_attn(r_k=16, r_v=16)
        attn.logit_scale_mode = "rk"
        attn.gamma_mode = "fixed"
        attn.gamma_value = None
        attn.p_k.data.normal_()
        attn.p_v.data.normal_()
        attn.gamma.data.fill_(2.5)
        attn.eval()

        x = torch.randn(1, 4, 256)
        with torch.no_grad():
            attn.gamma.data.fill_(1.0)
            out_gamma1 = attn(x, attention_mask=None)[0].clone()

            attn.gamma.data.fill_(2.0)
            out_gamma2 = attn(x, attention_mask=None)[0].clone()

        diff = (out_gamma2 - out_gamma1).abs().max().item()
        same = torch.allclose(out_gamma1, out_gamma2, atol=1e-5)
        assert not same, f"gamma=2.0 should differ from gamma=1.0, but max diff={diff}"

    def test_gamma_scales_logits_not_value_path(self):
        attn = _make_attn(r_k=16, r_v=16)
        attn.logit_scale_mode = "rk"
        attn.gamma_mode = "fixed"
        attn.gamma_value = None
        attn.p_k.data.normal_()
        attn.p_v.data.normal_()
        attn.eval()

        x = torch.randn(1, 4, 256)
        with torch.no_grad():
            attn.gamma.data.fill_(1.0)
            out_g1 = attn(x, attention_mask=None)[0].clone()

            attn.gamma.data.fill_(3.0)
            out_g3 = attn(x, attention_mask=None)[0].clone()

        diff = (out_g3 - out_g1).abs().max().item()
        assert diff > 1e-3, f"gamma=3 should produce different output than gamma=1, max_diff={diff}"

    def test_apply_pv_has_no_gamma(self):
        attn = _make_attn(r_k=16, r_v=16)
        attn.p_v.data.normal_()
        attn.gamma.data.fill_(5.0)
        attn.eval()

        x = torch.randn(1, 4, 4, 64)
        pv_down = attn.p_v[:, :attn.r_v]
        expected = x @ pv_down @ pv_down.T
        result = attn._apply_pv(x)
        assert torch.allclose(result, expected, atol=1e-5), "_apply_pv must NOT include gamma"


class TestLowRankScalingUsesRk:
    def test_logits_scaled_by_sqrt_rk_not_head_dim(self):
        attn = _make_attn(r_k=16, r_v=16)
        attn.logit_scale_mode = "rk"
        attn.gamma_mode = "fixed"
        attn.gamma_value = None
        attn.p_k.data.normal_()
        attn.p_v.data.normal_()
        attn.gamma.data.fill_(1.0)
        attn.eval()

        x = torch.randn(1, 4, 256)
        head_dim = attn.head_dim  # 64
        r_k = attn.r_k            # 16

        pk_down = attn.p_k[:, :r_k]
        pv_down = attn.p_v[:, :attn.r_v]

        q = attn.q_proj(x) * attn.scaling
        q = q.view(1, 4, 4, head_dim).transpose(1, 2)
        k = attn.k_proj(x).view(1, 4, 4, head_dim).transpose(1, 2)
        v = attn.v_proj(x).view(1, 4, 4, head_dim).transpose(1, 2)

        q_lat = q * (head_dim ** 0.5) @ pk_down
        k_lat = k @ pk_down
        v_lat = v @ pv_down

        manual_logits = (q_lat @ k_lat.transpose(2, 3)) / math.sqrt(r_k)
        causal_mask = _make_causal_mask(4, 4, x.device, x.dtype)
        manual_logits = manual_logits + causal_mask
        manual_weights = torch.softmax(manual_logits, dim=-1)
        manual_out_lat = manual_weights @ v_lat
        manual_out = manual_out_lat @ pv_down.T

        with torch.no_grad():
            model_out = attn(x, attention_mask=None)[0]

        manual_out_2d = manual_out.transpose(1, 2).reshape(1, 4, 256)
        manual_final = attn.o_proj(manual_out_2d)

        assert torch.allclose(model_out, manual_final, atol=1e-4), \
            "Model output must match manual computation with 1/sqrt(r_k) scaling"

    def test_different_r_k_gives_different_scaling(self):
        attn16 = _make_attn(r_k=16, r_v=16)
        attn32 = _make_attn(r_k=32, r_v=16)

        torch.manual_seed(42)
        x = torch.randn(1, 4, 256)

        attn16.p_k.data.normal_()
        attn16.p_v.data.normal_()
        attn16.gamma.data.fill_(1.0)
        attn16.eval()

        attn32.p_k.data.normal_()
        attn32.p_v.data.normal_()
        attn32.gamma.data.fill_(1.0)
        attn32.eval()

        with torch.no_grad():
            out16 = attn16(x, attention_mask=None)[0]
            out32 = attn32(x, attention_mask=None)[0]

        assert not torch.allclose(out16, out32, atol=1e-3), \
            "Different r_k should produce different outputs due to different scaling"


class TestFullRankIdentityEquivalence:
    def test_full_rank_identity_close_to_no_projection(self):
        attn = _make_attn(r_k=64, r_v=64)
        assert attn.p_k.shape == (64, 64)
        assert attn.gamma.item() == pytest.approx(1.0)
        attn.eval()

        x = torch.randn(1, 4, 256)
        with torch.no_grad():
            out = attn(x, attention_mask=None)[0]

        q = attn.q_proj(x) * attn.scaling
        k = attn.k_proj(x)
        v = attn.v_proj(x)
        q = q.view(1, 4, 4, 64).transpose(1, 2)
        k = k.view(1, 4, 4, 64).transpose(1, 2)
        v = v.view(1, 4, 4, 64).transpose(1, 2)

        logits = q @ k.transpose(2, 3)
        causal_mask = torch.triu(
            torch.full((4, 4), float("-inf"), dtype=logits.dtype, device=logits.device),
            diagonal=1,
        ).unsqueeze(0).unsqueeze(0)
        logits = logits + causal_mask
        weights = torch.softmax(logits, dim=-1)
        attn_out = (weights @ v).transpose(1, 2).reshape(1, 4, 256)
        baseline_out = attn.o_proj(attn_out)

        assert torch.allclose(out, baseline_out, atol=1e-4), \
            f"Full-rank (r_k=r_v=head_dim, P=I, gamma=1) must match baseline, max_diff={(out - baseline_out).abs().max().item()}"

    def test_full_rank_no_gamma_effect(self):
        attn = _make_attn(r_k=64, r_v=64)
        attn.eval()

        x = torch.randn(1, 4, 256)
        with torch.no_grad():
            attn.gamma.data.fill_(1.0)
            out1 = attn(x, attention_mask=None)[0].clone()
            attn.gamma.data.fill_(2.0)
            out2 = attn(x, attention_mask=None)[0].clone()

        assert torch.allclose(out1, out2, atol=1e-5), \
            "Full-rank path should not use gamma"
