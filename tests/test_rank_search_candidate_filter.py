from __future__ import annotations

import warnings

import pytest
import torch

from hawp_laq.offline.projector_trainer import ProjectorTrainer, ProjectorModule


def test_rank_search_filters_candidates_above_head_dim(tmp_path):
    from hawp_laq.offline.rank_search import search_rank_per_layer
    from hawp_laq.utils.io import save_pt

    calib_dir = tmp_path / "calib"
    calib_dir.mkdir()
    n_layers = 1
    n_heads = 4
    d_model = 64
    head_dim = d_model // n_heads

    save_pt({"n_layers": n_layers, "n_heads": n_heads}, calib_dir / "meta.pt")

    layer_data = {
        "q": torch.randn(2, 8, d_model),
        "k": torch.randn(2, 8, d_model),
        "v": torch.randn(2, 8, d_model),
    }
    save_pt(layer_data, calib_dir / "layer_0.pt")

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = search_rank_per_layer(
            calib_dir=calib_dir,
            rank_candidates=[8, 16, 32, 128, 256],
            n_steps=2,
            device="cpu",
        )
        filter_warnings = [x for x in w if "filtering out rank candidates" in str(x.message)]
        assert len(filter_warnings) == 1
        assert "128" in str(filter_warnings[0].message)
        assert "256" in str(filter_warnings[0].message)

    assert 0 in result
    assert result[0][0] <= head_dim
    assert result[0][1] <= head_dim


def test_rank_search_raises_if_no_valid_candidates(tmp_path):
    from hawp_laq.offline.rank_search import search_rank_per_layer
    from hawp_laq.utils.io import save_pt

    calib_dir = tmp_path / "calib"
    calib_dir.mkdir()
    n_layers = 1
    n_heads = 4
    d_model = 64

    save_pt({"n_layers": n_layers, "n_heads": n_heads}, calib_dir / "meta.pt")

    layer_data = {
        "q": torch.randn(2, 8, d_model),
        "k": torch.randn(2, 8, d_model),
        "v": torch.randn(2, 8, d_model),
    }
    save_pt(layer_data, calib_dir / "layer_0.pt")

    with pytest.raises(ValueError, match="no valid rank candidates"):
        search_rank_per_layer(
            calib_dir=calib_dir,
            rank_candidates=[128, 256],
            n_steps=2,
            device="cpu",
        )


def test_projector_trainer_rejects_rank_above_head_dim():
    with pytest.raises(ValueError, match="rank_k=128 must satisfy 1 <= rank_k <= head_dim=16"):
        ProjectorTrainer(d_model=64, rank_k=128, rank_v=8, n_heads=4)

    with pytest.raises(ValueError, match="rank_v=128 must satisfy 1 <= rank_v <= head_dim=16"):
        ProjectorTrainer(d_model=64, rank_k=8, rank_v=128, n_heads=4)

    with pytest.raises(ValueError, match="rank_k=128 must satisfy 1 <= rank_k <= head_dim=16"):
        ProjectorModule(d_model=64, rank_k=128, rank_v=8, n_heads=4)
