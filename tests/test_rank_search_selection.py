from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest
import torch

from hawp_laq.offline.rank_search import (
    _component_pass,
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


def test_component_pass_relative_mode():
    assert _component_pass(0.011, 0.01, 0.10, 1e-6) is True
    assert _component_pass(0.012, 0.01, 0.10, 1e-6) is False


def test_component_pass_absolute_mode():
    assert _component_pass(5e-7, 0.0, 0.10, 1e-6) is True
    assert _component_pass(5e-6, 0.0, 0.10, 1e-6) is False


def test_component_pass_near_zero_baseline():
    assert _component_pass(5e-7, 1e-9, 0.10, 1e-6) is True
    assert _component_pass(5e-6, 1e-9, 0.10, 1e-6) is False


def test_rank_search_constraint_selection(tmp_path):
    calib_dir = _make_calib_dir(tmp_path, d_model=64, n_heads=4)
    output_dir = tmp_path / "rank_search_out"

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        result = search_rank_per_layer(
            calib_dir=calib_dir,
            rank_candidates=[8, 16],
            n_steps=2,
            device="cpu",
            relative_tolerance=0.10,
            output_dir=output_dir,
        )

    assert 0 in result
    chosen = result[0][0]
    assert chosen in (8, 16)

    json_path = output_dir / "layer_0_rank_search.json"
    assert json_path.exists()
    with open(json_path) as f:
        data = json.load(f)

    assert data["selection_method"] == "constraint"
    for r in data["results"]:
        assert "logits_pass" in r
        assert "attn_pass" in r
        assert "value_pass" in r
        assert "all_pass" in r


def test_rank_search_near_zero_baseline_uses_abs_tolerance(tmp_path):
    calib_dir = _make_calib_dir(tmp_path, d_model=64, n_heads=4)
    output_dir = tmp_path / "rank_search_out"

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        result = search_rank_per_layer(
            calib_dir=calib_dir,
            rank_candidates=[8, 16],
            n_steps=2,
            device="cpu",
            relative_tolerance=0.10,
            logits_abs_tolerance=1e-6,
            attn_abs_tolerance=1e-5,
            value_abs_tolerance=1e-4,
            output_dir=output_dir,
        )

    assert 0 in result
    json_path = output_dir / "layer_0_rank_search.json"
    with open(json_path) as f:
        data = json.load(f)

    assert data["logits_abs_tolerance"] == 1e-6
    assert data["attn_abs_tolerance"] == 1e-5
    assert data["value_abs_tolerance"] == 1e-4


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


def test_rank_search_json_no_stale_fields(tmp_path):
    calib_dir = _make_calib_dir(tmp_path, d_model=64, n_heads=4)
    output_dir = tmp_path / "rank_search_out"

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        search_rank_per_layer(
            calib_dir=calib_dir,
            rank_candidates=[8, 16],
            n_steps=2,
            device="cpu",
            relative_tolerance=0.10,
            output_dir=output_dir,
        )

    json_path = output_dir / "layer_0_rank_search.json"
    assert json_path.exists()
    with open(json_path) as f:
        data = json.load(f)

    assert "tolerance" not in data
    assert "selection_score" not in data["results"][0]
    assert "baseline_score" not in data
    assert "threshold" not in data


def test_rank_search_all_pass_selects_smallest_rank(tmp_path):
    import hawp_laq.offline.rank_search as rs_mod
    original = rs_mod._evaluate_rank

    def _mock_evaluate(q, k, v, rank_k, rank_v, d_model, n_heads,
                       n_steps, lr, orthogonalize_every, w_logits, w_attn,
                       w_value, device):
        return {
            "rank_k": rank_k,
            "rank_v": rank_v,
            "final_loss": 0.01,
            "final_logits_loss": 0.008,
            "final_attn_loss": 0.001,
            "final_value_loss": 0.001,
            "p_k_shape": (16, rank_k),
            "p_v_shape": (16, rank_v),
        }

    rs_mod._evaluate_rank = _mock_evaluate
    try:
        calib_dir = _make_calib_dir(tmp_path, d_model=64, n_heads=4)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = search_rank_per_layer(
                calib_dir=calib_dir,
                rank_candidates=[8, 16, 32, 64],
                n_steps=2,
                device="cpu",
                relative_tolerance=0.10,
            )
        assert result[0] == (8, 8)
    finally:
        rs_mod._evaluate_rank = original
