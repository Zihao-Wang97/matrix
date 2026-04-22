from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest
import torch

from hawp_laq.offline.rank_search import (
    _selection_score,
    search_rank_per_layer,
)
from hawp_laq.utils.io import save_pt


def _make_calib_dir(tmp_path, n_layers=1, n_heads=4, d_model=64):
    calib_dir = tmp_path / "calib"
    calib_dir.mkdir()
    save_pt({"n_layers": n_layers, "n_heads": n_heads}, calib_dir / "meta.pt")
    for i in range(n_layers):
        save_pt(
            {
                "q": torch.randn(2, 8, d_model),
                "k": torch.randn(2, 8, d_model),
                "v": torch.randn(2, 8, d_model),
            },
            calib_dir / f"layer_{i}.pt",
        )
    return calib_dir


def test_rank_search_selection_uses_attn_value_score_not_total_loss():
    result_small = {
        "final_attn_loss": 0.05,
        "final_value_loss": 0.10,
        "final_loss": 1.0,
        "final_logits_loss": 0.925,
    }
    result_large = {
        "final_attn_loss": 0.005,
        "final_value_loss": 0.01,
        "final_loss": 0.5,
        "final_logits_loss": 0.485,
    }
    value_weight = 0.25
    score_small = _selection_score(result_small, value_weight)
    score_large = _selection_score(result_large, value_weight)

    assert result_small["final_loss"] > result_large["final_loss"]
    assert score_small > score_large

    baseline = score_large
    abs_tol = 0.04
    assert score_small > baseline + abs_tol

    result_medium = {
        "final_attn_loss": 0.008,
        "final_value_loss": 0.015,
        "final_loss": 0.8,
        "final_logits_loss": 0.777,
    }
    score_medium = _selection_score(result_medium, value_weight)
    assert score_medium <= baseline + abs_tol
    assert result_medium["final_loss"] > result_large["final_loss"]


def test_rank_search_absolute_tolerance_with_near_zero_baseline(tmp_path):
    calib_dir = _make_calib_dir(tmp_path, d_model=64, n_heads=4)
    head_dim = 16

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        result = search_rank_per_layer(
            calib_dir=calib_dir,
            rank_candidates=[8, head_dim],
            n_steps=2,
            device="cpu",
            selection_abs_tolerance=0.04,
        )

    assert 0 in result
    chosen = result[0][0]
    assert chosen <= head_dim

    if chosen < head_dim:
        pass
    else:
        assert chosen == head_dim


def test_rank_search_prefers_rank_search_n_steps_over_projector_n_steps(tmp_path):
    from hawp_laq.config import HAWPLAQConfig, ProjectorConfig, RankSearchConfig
    from hawp_laq.offline.rank_search import run_rank_search_from_config

    calib_dir = _make_calib_dir(tmp_path, d_model=64, n_heads=4)

    cfg = HAWPLAQConfig()
    cfg.calib.output_dir = calib_dir
    cfg.projector = ProjectorConfig(r_k=8, r_v=8, n_steps=5)
    cfg.rank_search = RankSearchConfig(
        rank_candidates=[8, 16],
        n_steps=2,
        output_dir=str(tmp_path / "rank_search_out"),
    )

    n_steps_used = []

    import hawp_laq.offline.rank_search as rs_mod
    original = rs_mod._evaluate_rank

    def _capturing_evaluate(q, k, v, rank_k, rank_v, d_model, n_heads,
                            n_steps, lr, orthogonalize_every, w_logits, w_attn,
                            w_value, device):
        n_steps_used.append(n_steps)
        return {
            "rank_k": rank_k,
            "rank_v": rank_v,
            "final_loss": 0.1,
            "final_logits_loss": 0.08,
            "final_attn_loss": 0.01,
            "final_value_loss": 0.02,
            "p_k_shape": (16, 8),
            "p_v_shape": (16, 8),
        }

    rs_mod._evaluate_rank = _capturing_evaluate
    try:
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            run_rank_search_from_config(cfg)
        assert all(s == 2 for s in n_steps_used), f"Expected n_steps=2, got {n_steps_used}"
    finally:
        rs_mod._evaluate_rank = original


def test_rank_search_json_contains_selection_fields(tmp_path):
    calib_dir = _make_calib_dir(tmp_path, d_model=64, n_heads=4)
    output_dir = tmp_path / "rank_search_out"

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        search_rank_per_layer(
            calib_dir=calib_dir,
            rank_candidates=[8, 16],
            n_steps=2,
            device="cpu",
            selection_value_weight=0.25,
            selection_abs_tolerance=0.04,
            output_dir=output_dir,
        )

    json_path = output_dir / "layer_0_rank_search.json"
    assert json_path.exists()
    with open(json_path) as f:
        data = json.load(f)

    assert "selection_metric" in data
    assert data["selection_metric"] == "attn_value_abs"
    assert "selection_value_weight" in data
    assert data["selection_value_weight"] == 0.25
    assert "selection_abs_tolerance" in data
    assert data["selection_abs_tolerance"] == 0.04
    assert "baseline_selection_score" in data
    for r in data["results"]:
        assert "selection_score" in r


def test_rank_search_rejects_unsupported_selection_metric(tmp_path):
    calib_dir = _make_calib_dir(tmp_path, d_model=64, n_heads=4)

    with pytest.raises(ValueError, match="Unsupported selection_metric='total_relative'"):
        search_rank_per_layer(
            calib_dir=calib_dir,
            rank_candidates=[8, 16],
            n_steps=2,
            device="cpu",
            selection_metric="total_relative",
        )
