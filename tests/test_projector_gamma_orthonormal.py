import torch
import pytest

from hawp_laq.offline.projector_trainer import ProjectorModule, ProjectorTrainer, _complete_to_orthonormal_basis
from hawp_laq.modeling.attention_hawp import HAWPAttention
from types import SimpleNamespace


def test_gamma_receives_gradient_from_logits_path():
    mod = ProjectorModule(64, 8, 8, 4)
    q = torch.randn(1, 4, 64)
    k = torch.randn(1, 4, 64)
    v = torch.randn(1, 4, 64)
    logits_fp, logits_hat, attn_out, attn_out_hat, k_recon, v_recon = mod(q, k, v)
    import torch.nn.functional as F
    loss = F.mse_loss(logits_hat, logits_fp)
    loss.backward()
    assert mod.gamma.grad is not None, "gamma must receive gradient from logits path"
    assert mod.gamma.grad.abs().sum() > 0, "gamma gradient must be nonzero"


def test_projector_full_matrix_is_orthonormal():
    basis = torch.randn(16, 6)
    full = _complete_to_orthonormal_basis(basis, 16)
    assert full.shape == (16, 16)
    product = full.T @ full
    assert torch.allclose(product, torch.eye(16), atol=1e-4), (
        f"Full matrix is not orthonormal, max deviation: {(product - torch.eye(16)).abs().max()}"
    )


def test_trainer_output_matches_inference_gamma_semantics():
    d_model, rank_k, rank_v, n_heads = 64, 8, 6, 4
    head_dim = d_model // n_heads

    trainer = ProjectorTrainer(d_model, rank_k, rank_v, n_heads, lr=1e-3, device="cpu")
    q = torch.randn(1, 4, d_model)
    k = torch.randn(1, 4, d_model)
    v = torch.randn(1, 4, d_model)
    result = trainer.train_one_group(q, k, v, n_steps=5)

    p_k = result["p_k"]
    p_v = result["p_v"]
    assert p_k.shape == (head_dim, head_dim)
    assert p_v.shape == (head_dim, head_dim)

    p_k_ortho = p_k.T @ p_k
    assert torch.allclose(p_k_ortho, torch.eye(head_dim), atol=1e-4)

    p_v_ortho = p_v.T @ p_v
    assert torch.allclose(p_v_ortho, torch.eye(head_dim), atol=1e-4)

    config = SimpleNamespace(
        hidden_size=d_model, num_attention_heads=n_heads,
        num_key_value_heads=n_heads, max_position_embeddings=2048,
        rope_theta=10000.0, model_type="opt",
        enable_bias=False, attention_dropout=0.0,
    )
    attn = HAWPAttention(config, layer_idx=0, r_k=rank_k, r_v=rank_v)

    data = {"p_k": result["p_k"], "p_v": result["p_v"], "gamma": result["gamma"]}
    attn.load_projector_data(data, strict=True)

    lr_scale_inference = attn.gamma.item() / (rank_k ** 0.5)
    lr_scale_trainer = result["gamma"].item() / (rank_k ** 0.5)
    assert abs(lr_scale_inference - lr_scale_trainer) < 1e-6
