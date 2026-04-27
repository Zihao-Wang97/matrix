import torch
import pytest
from pathlib import Path
from hawp_laq.offline.projector_trainer import ProjectorModule, ProjectorTrainer
import torch.nn.functional as F


class TestProjectorModule:
    def test_forward_shapes(self):
        d_model, rank_k, rank_v, n_heads = 64, 16, 12, 4
        mod = ProjectorModule(d_model, rank_k, rank_v, n_heads)
        q = torch.randn(2, 8, d_model)
        k = torch.randn(2, 8, d_model)
        v = torch.randn(2, 8, d_model)
        logits_fp, logits_hat, causal_valid, attn_out, attn_out_hat, k_recon, v_recon = mod(q, k, v)
        head_dim = d_model // n_heads
        assert logits_fp.shape == (2, n_heads, 8, 8)
        assert logits_hat.shape == (2, n_heads, 8, 8)
        assert causal_valid.shape == (1, 1, 8, 8)
        assert attn_out.shape == (2, n_heads, 8, head_dim)
        assert attn_out_hat.shape == (2, n_heads, 8, head_dim)
        assert k_recon.shape == (2, 8, d_model)
        assert v_recon.shape == (2, 8, d_model)

    def test_orthogonalize(self):
        mod = ProjectorModule(32, 8, 8, 4)
        mod.p_k_basis.data = torch.randn(8, 8)
        mod.orthogonalize_projectors()
        product = mod.p_k_basis.T @ mod.p_k_basis
        assert torch.allclose(product, torch.eye(8), atol=1e-4)

    def test_gradient_flows(self):
        mod = ProjectorModule(32, 8, 8, 4)
        q = torch.randn(1, 4, 32)
        k = torch.randn(1, 4, 32)
        v = torch.randn(1, 4, 32)
        logits_fp, logits_hat, causal_valid, attn_out, attn_out_hat, k_recon, v_recon = mod(q, k, v)
        mask = causal_valid.expand_as(logits_hat)
        loss = F.mse_loss(logits_hat[mask], logits_fp[mask]) + F.mse_loss(attn_out_hat, attn_out) + F.mse_loss(v_recon, v)
        loss.backward()
        assert mod.p_k_basis.grad is not None
        assert mod.p_v_basis.grad is not None
        assert mod.gamma.grad is not None

    def test_asymmetric_ranks(self):
        d_model, rank_k, rank_v, n_heads = 64, 16, 8, 4
        mod = ProjectorModule(d_model, rank_k, rank_v, n_heads)
        head_dim = d_model // n_heads
        assert mod.p_k_basis.shape == (head_dim, rank_k)
        assert mod.p_v_basis.shape == (head_dim, rank_v)
        q = torch.randn(1, 4, d_model)
        k = torch.randn(1, 4, d_model)
        v = torch.randn(1, 4, d_model)
        logits_fp, logits_hat, causal_valid, attn_out, attn_out_hat, k_recon, v_recon = mod(q, k, v)
        assert logits_hat.shape == logits_fp.shape
        assert attn_out_hat.shape == attn_out.shape


class TestProjectorTrainer:
    def test_train_one_group_converges(self):
        torch.manual_seed(42)
        d_model, rank_k, rank_v, n_heads = 32, 8, 8, 4
        q = torch.randn(2, 8, d_model)
        k = torch.randn(2, 8, d_model)
        v = torch.randn(2, 8, d_model)
        trainer = ProjectorTrainer(d_model, rank_k, rank_v, n_heads, lr=1e-2, orthogonalize_every=5, device="cpu")
        result = trainer.train_one_group(q, k, v, n_steps=50)
        losses = result["metrics"]["total"]
        assert losses[-1] < 1e-3, f"loss did not converge: {losses[-1]:.6f}"

    def test_result_keys_and_shapes(self):
        d_model, rank_k, rank_v, n_heads = 32, 8, 6, 4
        q = torch.randn(1, 4, d_model)
        k = torch.randn(1, 4, d_model)
        v = torch.randn(1, 4, d_model)
        trainer = ProjectorTrainer(d_model, rank_k, rank_v, n_heads, lr=1e-3, device="cpu")
        result = trainer.train_one_group(q, k, v, n_steps=5)
        assert "p_k" in result
        assert "p_v" in result
        assert "gamma" in result
        assert "r_k" in result
        assert "r_v" in result
        assert "metrics" in result
        head_dim = d_model // n_heads
        assert result["p_k"].shape == (head_dim, head_dim)
        assert result["p_v"].shape == (head_dim, head_dim)
        assert result["r_k"] == rank_k
        assert result["r_v"] == rank_v
        assert result["gamma"].numel() == 1

    def test_save_result(self, tmp_path):
        d_model, rank_k, rank_v, n_heads = 32, 8, 8, 4
        trainer = ProjectorTrainer(d_model, rank_k, rank_v, n_heads, device="cpu")
        q = torch.randn(1, 4, d_model)
        k = torch.randn(1, 4, d_model)
        v = torch.randn(1, 4, d_model)
        result = trainer.train_one_group(q, k, v, n_steps=3)
        out = ProjectorTrainer.save_result(result, 0, tmp_path / "projectors")
        assert (out / "projector.pt").exists()
        loaded = torch.load(out / "projector.pt", map_location="cpu", weights_only=False)
        assert "p_k" in loaded
        assert "p_v" in loaded
        assert "r_k" in loaded
        assert "r_v" in loaded

    def test_qr_columns_orthogonal(self):
        d_model, rank_k, rank_v, n_heads = 32, 8, 8, 4
        q = torch.randn(2, 8, d_model)
        k = torch.randn(2, 8, d_model)
        v = torch.randn(2, 8, d_model)
        trainer = ProjectorTrainer(d_model, rank_k, rank_v, n_heads, lr=1e-3, orthogonalize_every=1, device="cpu")
        result = trainer.train_one_group(q, k, v, n_steps=10)
        p_k = result["p_k"]
        head_dim = d_model // n_heads
        first_cols = p_k[:, :rank_k]
        ortho = first_cols.T @ first_cols
        assert torch.allclose(ortho, torch.eye(rank_k), atol=1e-4)
