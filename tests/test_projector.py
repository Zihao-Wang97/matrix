import torch
import pytest
from pathlib import Path
from hawp_laq.offline.projector_trainer import ProjectorModule, ProjectorTrainer
from hawp_laq.offline.losses import total_projector_loss


class TestProjectorModule:
    def test_forward_shapes(self):
        d_model, rank, n_heads = 64, 16, 4
        mod = ProjectorModule(d_model, rank, n_heads)
        q = torch.randn(2, 8, d_model)
        k = torch.randn(2, 8, d_model)
        v = torch.randn(2, 8, d_model)
        logits_fp, logits_hat, attn_out, attn_out_hat, k_recon, v_recon = mod(q, k, v)
        assert logits_fp.shape == (2, n_heads, 8, 8)
        assert logits_hat.shape == (2, n_heads, 8, 8)
        assert attn_out.shape == (2, n_heads, 8, d_model // n_heads)
        assert attn_out_hat.shape == (2, n_heads, 8, d_model // n_heads)
        assert k_recon.shape == (2, 8, d_model)
        assert v_recon.shape == (2, 8, d_model)

    def test_orthogonalize(self):
        mod = ProjectorModule(32, 8, 4)
        mod.p_k.data = torch.randn(32, 8)
        mod.orthogonalize_projectors()
        product = mod.p_k.T @ mod.p_k
        assert torch.allclose(product, torch.eye(8), atol=1e-4)

    def test_gradient_flows(self):
        mod = ProjectorModule(32, 8, 4)
        q = torch.randn(1, 4, 32)
        k = torch.randn(1, 4, 32)
        v = torch.randn(1, 4, 32)
        logits_fp, logits_hat, attn_out, attn_out_hat, k_recon, v_recon = mod(q, k, v)
        loss = total_projector_loss(logits_fp, logits_hat, attn_out, attn_out_hat, v, mod.p_v, mod.gamma)
        loss.backward()
        assert mod.p_k.grad is not None
        assert mod.p_v.grad is not None
        assert mod.gamma.grad is not None


class TestProjectorTrainer:
    def test_train_one_group_loss_decreases(self):
        torch.manual_seed(42)
        d_model, rank, n_heads = 32, 8, 4
        q = torch.randn(2, 8, d_model)
        k = torch.randn(2, 8, d_model)
        v = torch.randn(2, 8, d_model)
        trainer = ProjectorTrainer(d_model, rank, n_heads, lr=1e-2, orthogonalize_every=5, device="cpu")
        result = trainer.train_one_group(q, k, v, n_steps=50)
        losses = result["metrics"]["total"]
        assert losses[-1] < losses[0], f"loss did not decrease: {losses[0]:.6f} -> {losses[-1]:.6f}"

    def test_result_keys(self):
        d_model, rank, n_heads = 32, 8, 4
        q = torch.randn(1, 4, d_model)
        k = torch.randn(1, 4, d_model)
        v = torch.randn(1, 4, d_model)
        trainer = ProjectorTrainer(d_model, rank, n_heads, lr=1e-3, device="cpu")
        result = trainer.train_one_group(q, k, v, n_steps=5)
        assert "p_k" in result
        assert "p_v" in result
        assert "gamma" in result
        assert "metrics" in result
        assert result["p_k"].shape == (d_model, rank)
        assert result["p_v"].shape == (d_model, rank)
        assert result["gamma"].numel() == 1

    def test_save_result(self, tmp_path):
        d_model, rank, n_heads = 32, 8, 4
        trainer = ProjectorTrainer(d_model, rank, n_heads, device="cpu")
        q = torch.randn(1, 4, d_model)
        k = torch.randn(1, 4, d_model)
        v = torch.randn(1, 4, d_model)
        result = trainer.train_one_group(q, k, v, n_steps=3)
        out = ProjectorTrainer.save_result(result, 0, tmp_path / "projectors")
        assert (out / "projector.pt").exists()
        loaded = torch.load(out / "projector.pt", map_location="cpu", weights_only=False)
        assert "p_k" in loaded
        assert "p_v" in loaded

    def test_qr_columns_orthogonal(self):
        d_model, rank, n_heads = 32, 8, 4
        q = torch.randn(2, 8, d_model)
        k = torch.randn(2, 8, d_model)
        v = torch.randn(2, 8, d_model)
        trainer = ProjectorTrainer(d_model, rank, n_heads, lr=1e-3, orthogonalize_every=1, device="cpu")
        result = trainer.train_one_group(q, k, v, n_steps=10)
        p_k = result["p_k"]
        ortho = p_k.T @ p_k
        assert torch.allclose(ortho, torch.eye(rank), atol=1e-4)
