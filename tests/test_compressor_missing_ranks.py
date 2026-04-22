import pytest
import torch
import warnings
from pathlib import Path

from hawp_laq.runtime.compressor import CompressorPackage


def test_compressor_warns_on_missing_rank_keys_and_falls_back(tmp_path):
    projector_dir = tmp_path / "projectors"
    layer_dir = projector_dir / "layer_0"
    layer_dir.mkdir(parents=True)

    data_without_ranks = {
        "p_k": torch.eye(16),
        "p_v": torch.eye(16),
        "gamma": torch.ones(1),
    }
    torch.save(data_without_ranks, layer_dir / "projector.pt")

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        pkg = CompressorPackage(
            projector_dir=projector_dir,
            n_layers=1,
            n_heads=4,
            head_dim=16,
        )
        rank_warnings = [x for x in w if issubclass(x.category, UserWarning) and "missing rank key" in str(x.message).lower()]
        assert len(rank_warnings) > 0
        assert "r_k" in str(rank_warnings[0].message)
        assert "r_v" in str(rank_warnings[0].message)

    ranks = pkg.ranks
    assert 0 in ranks
    assert ranks[0] == (16, 16)
