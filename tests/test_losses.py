import torch
import pytest
from hawp_laq.offline.losses import (
    compute_logits_fp,
    compute_logits_hat,
    logits_mse_loss,
    attention_output_mse_loss,
    value_reconstruction_loss,
    total_projector_loss,
)


class TestComputeLogitsFp:
    def test_shape(self):
        q = torch.randn(2, 12, 8, 64)
        k = torch.randn(2, 12, 8, 64)
        out = compute_logits_fp(q, k)
        assert out.shape == (2, 12, 8, 8)

    def test_scale(self):
        q = torch.randn(1, 1, 4, 16)
        k = torch.randn(1, 1, 4, 16)
        out = compute_logits_fp(q, k, scale=0.25)
        expected = (q @ k.transpose(-2, -1)) * 0.25
        assert torch.allclose(out, expected)


class TestComputeLogitsHat:
    def test_shape(self):
        q = torch.randn(2, 8, 64)
        k = torch.randn(2, 8, 64)
        p_k = torch.randn(64, 16)
        out = compute_logits_hat(q, k, p_k)
        assert out.shape == (2, 8, 8)

    def test_equivalent_to_reconstruction(self):
        q = torch.randn(1, 4, 32)
        k = torch.randn(1, 4, 32)
        p_k = torch.randn(32, 8)
        k_recon = k @ p_k @ p_k.T
        direct = (q @ k_recon.transpose(-2, -1)) * (32 ** -0.5)
        via_hat = compute_logits_hat(q, k, p_k)
        assert torch.allclose(direct, via_hat, atol=1e-5)


class TestLogitsMseLoss:
    def test_zero_loss(self):
        x = torch.randn(2, 4, 4, 4)
        assert logits_mse_loss(x, x).item() == pytest.approx(0.0, abs=1e-7)

    def test_nonzero_loss(self):
        a = torch.zeros(2, 4, 4, 4)
        b = torch.ones(2, 4, 4, 4)
        assert logits_mse_loss(a, b).item() == pytest.approx(1.0)


class TestAttentionOutputMseLoss:
    def test_zero_loss(self):
        x = torch.randn(2, 8, 4, 32)
        assert attention_output_mse_loss(x, x).item() == pytest.approx(0.0, abs=1e-7)


class TestValueReconstructionLoss:
    def test_perfect_projection(self):
        v = torch.randn(2, 8, 32)
        p_v = torch.eye(32)
        loss = value_reconstruction_loss(v, p_v)
        assert loss.item() == pytest.approx(0.0, abs=1e-5)

    def test_with_gamma(self):
        v = torch.randn(2, 8, 32)
        p_v = torch.eye(32)
        gamma = torch.tensor([2.0])
        loss = value_reconstruction_loss(v, p_v, gamma)
        expected = torch.nn.functional.mse_loss(2.0 * v, v)
        assert loss.item() == pytest.approx(expected.item(), abs=1e-5)


class TestTotalProjectorLoss:
    def test_weighted_sum(self):
        logits_fp = torch.randn(2, 4, 8, 8)
        logits_hat = torch.randn(2, 4, 8, 8)
        attn_out = torch.randn(2, 4, 8, 32)
        attn_out_hat = torch.randn(2, 4, 8, 32)
        v = torch.randn(2, 8, 32)
        p_v = torch.randn(32, 16)
        loss = total_projector_loss(logits_fp, logits_hat, attn_out, attn_out_hat, v, p_v, w_logits=1.0, w_attn=2.0, w_value=3.0)
        assert loss.item() > 0.0
