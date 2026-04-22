import torch
import pytest
from pathlib import Path

from hawp_laq.offline.projector_trainer import ProjectorTrainer
from hawp_laq.modeling.attention_hawp import HAWPAttention
from types import SimpleNamespace


def test_projector_trainer_output_matches_hawp_contract():
    d_model, rank_k, rank_v, n_heads = 64, 8, 6, 4
    head_dim = d_model // n_heads

    trainer = ProjectorTrainer(d_model, rank_k, rank_v, n_heads, lr=1e-3, device="cpu")
    q = torch.randn(1, 4, d_model)
    k = torch.randn(1, 4, d_model)
    v = torch.randn(1, 4, d_model)
    result = trainer.train_one_group(q, k, v, n_steps=5)

    assert result["p_k"].shape == (head_dim, head_dim), f"p_k shape: {result['p_k'].shape}"
    assert result["p_v"].shape == (head_dim, head_dim), f"p_v shape: {result['p_v'].shape}"
    assert result["r_k"] == rank_k
    assert result["r_v"] == rank_v
    assert result["gamma"].numel() == 1

    config = SimpleNamespace(
        hidden_size=d_model,
        num_attention_heads=n_heads,
        num_key_value_heads=n_heads,
        max_position_embeddings=2048,
        rope_theta=10000.0,
        model_type="opt",
        enable_bias=False,
        attention_dropout=0.0,
    )
    attn = HAWPAttention(config, layer_idx=0, r_k=rank_k, r_v=rank_v)

    data = {"p_k": result["p_k"], "p_v": result["p_v"], "gamma": result["gamma"]}
    attn.load_projector_data(data, strict=True)

    assert torch.allclose(attn.p_k.data, result["p_k"])
    assert torch.allclose(attn.p_v.data, result["p_v"])
    assert torch.allclose(attn.gamma.data, result["gamma"])
