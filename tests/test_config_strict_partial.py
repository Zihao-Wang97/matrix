import pytest

from hawp_laq.config import ProjectorConfig, resolve_projector_ranks


def test_resolve_projector_ranks_quant_only_partial_still_raises():
    cfg = ProjectorConfig(r_k=16, r_v=None)
    with pytest.raises(ValueError, match="Must provide both r_k and r_v"):
        resolve_projector_ranks(cfg, head_dim=64, mode="quant_only")


def test_resolve_projector_ranks_partial_rv_only_raises():
    cfg = ProjectorConfig(r_k=None, r_v=16)
    with pytest.raises(ValueError, match="Must provide both r_k and r_v"):
        resolve_projector_ranks(cfg, head_dim=64, mode="hawp_quant")
