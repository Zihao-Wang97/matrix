import torch
import pytest
from pathlib import Path

from hawp_laq.config import load_config, resolve_projector_ranks, ProjectorConfig
from hawp_laq.offline.projector_trainer import ProjectorTrainer


def test_single_group_training_uses_resolved_rk_rv():
    cfg = ProjectorConfig(r_k=16, r_v=12)
    head_dim = 64
    r_k, r_v = resolve_projector_ranks(cfg, head_dim=head_dim, mode="single_group")

    assert r_k == 16
    assert r_v == 12

    d_model = 64
    n_heads = 1
    trainer = ProjectorTrainer(d_model=d_model, rank_k=r_k, rank_v=r_v, n_heads=n_heads, lr=1e-3, device="cpu")
    q = torch.randn(1, 4, d_model)
    k = torch.randn(1, 4, d_model)
    v = torch.randn(1, 4, d_model)
    result = trainer.train_one_group(q, k, v, n_steps=3)

    assert result["r_k"] == 16
    assert result["r_v"] == 12
    assert result["p_k"].shape == (head_dim, head_dim)
    assert result["p_v"].shape == (head_dim, head_dim)
