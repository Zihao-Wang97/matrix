import torch
import pytest
from pathlib import Path

from hawp_laq.config import ProjectorConfig, resolve_projector_ranks


def test_resolve_projector_ranks_from_rk_rv():
    cfg = ProjectorConfig(r_k=16, r_v=32)
    r_k, r_v = resolve_projector_ranks(cfg, head_dim=64, mode="hawp_quant")
    assert r_k == 16
    assert r_v == 32


def test_resolve_projector_ranks_from_rank_alias():
    cfg = ProjectorConfig(rank=24)
    r_k, r_v = resolve_projector_ranks(cfg, head_dim=64, mode="hawp_quant")
    assert r_k == 24
    assert r_v == 24


def test_resolve_projector_ranks_quant_only_is_full_rank():
    cfg = ProjectorConfig()
    r_k, r_v = resolve_projector_ranks(cfg, head_dim=64, mode="quant_only")
    assert r_k == 64
    assert r_v == 64


def test_resolve_projector_ranks_missing_values_raise():
    cfg = ProjectorConfig()
    with pytest.raises(ValueError, match="Cannot resolve projector ranks"):
        resolve_projector_ranks(cfg, head_dim=64, mode="hawp_only")


def test_resolve_projector_ranks_partial_raises():
    cfg = ProjectorConfig(r_k=16)
    with pytest.raises(ValueError):
        resolve_projector_ranks(cfg, head_dim=64, mode="hawp_quant")


def test_resolve_projector_ranks_out_of_range():
    cfg = ProjectorConfig(r_k=128, r_v=16)
    with pytest.raises(ValueError, match="r_k must satisfy"):
        resolve_projector_ranks(cfg, head_dim=64, mode="hawp_quant")


def test_resolve_projector_ranks_zero_raises():
    cfg = ProjectorConfig(r_k=0, r_v=16)
    with pytest.raises(ValueError, match="r_k must satisfy"):
        resolve_projector_ranks(cfg, head_dim=64, mode="hawp_quant")
