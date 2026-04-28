"""Tests for asymmetric rank search (r_k != r_v pairs)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from hawp_laq.offline.rank_search import (
    build_rank_pairs,
    _evaluate_rank,
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


# ------------------------------------------------------------------


class TestBuildRankPairs:
    def test_legacy_rank_candidates_produce_symmetric_pairs(self):
        pairs = build_rank_pairs(rank_candidates=[8, 16], head_dim=64)
        assert pairs == [(8, 8), (16, 16)]

    def test_k_v_candidates_produce_cartesian_pairs(self):
        pairs = build_rank_pairs(
            r_k_candidates=[32, 64],
            r_v_candidates=[16, 32],
            head_dim=64,
        )
        expected = [(32, 16), (32, 32), (64, 16), (64, 32)]
        assert pairs == expected

    def test_rank_pair_candidates_take_priority(self):
        pairs = build_rank_pairs(
            rank_pair_candidates=[[64, 48], [48, 32]],
            r_k_candidates=[32, 64],   # should be ignored
            rank_candidates=[8, 16],   # should be ignored
            head_dim=64,
        )
        assert pairs == [(64, 48), (48, 32)]

    def test_deduplication(self):
        pairs = build_rank_pairs(
            rank_pair_candidates=[[64, 48], [64, 48], [48, 32]],
            head_dim=64,
        )
        assert pairs == [(64, 48), (48, 32)]

    def test_empty_returns_empty(self):
        assert build_rank_pairs(head_dim=64) == []

    def test_invalid_rk_raises(self):
        with pytest.raises(ValueError, match="r_k=65"):
            build_rank_pairs(rank_pair_candidates=[[65, 48]], head_dim=64)

    def test_invalid_rv_raises(self):
        with pytest.raises(ValueError, match="r_v=65"):
            build_rank_pairs(rank_pair_candidates=[[48, 65]], head_dim=64)

    def test_rank_pair_candidates_sets_priority(self):
        # rank_pair_candidates takes priority over r_k/r_v_candidates
        pairs = build_rank_pairs(
            rank_pair_candidates=[[64, 64], [48, 48]],
            r_k_candidates=[32],
            r_v_candidates=[16],
            head_dim=64,
        )
        assert pairs == [(64, 64), (48, 48)]


class TestEvaluateRankAsymmetric:
    def test_evaluate_rank_accepts_asymmetric_pair(self):
        import torch
        d_model, n_heads = 64, 4
        q = torch.randn(2, 8, d_model)
        k = torch.randn(2, 8, d_model)
        v = torch.randn(2, 8, d_model)

        result = _evaluate_rank(
            q, k, v, r_k=16, r_v=8,  # asymmetric!
            d_model=d_model, n_heads=n_heads,
            n_steps=2, lr=1e-3, orthogonalize_every=5,
            w_logits=1.0, w_attn=1.0, w_value=0.5,
            device="cpu",
        )
        assert result["r_k"] == 16
        assert result["r_v"] == 8
        assert "final_loss" in result
        assert "rank_cost" in result
        assert result["rank_cost"] == 24


class TestRankSearchAsymmetricIntegration:
    def test_rank_search_writes_asymmetric_ranks_json(self, tmp_path):
        calib_dir = _make_calib_dir(tmp_path, d_model=64, n_heads=4)
        output_dir = tmp_path / "rank_search_out"

        result = search_rank_per_layer(
            calib_dir=calib_dir,
            rank_pairs=[(16, 8), (16, 16), (8, 8)],
            n_steps=2,
            device="cpu",
            relative_tolerance=0.10,
            output_dir=output_dir,
        )
        assert 0 in result
        rk, rv = result[0]
        assert isinstance(rk, int)
        assert isinstance(rv, int)

        json_path = output_dir / "layer_0_rank_search.json"
        assert json_path.exists()
        with open(json_path) as f:
            data = json.load(f)
        assert "selected_r_k" in data
        assert "selected_r_v" in data
        assert "results" in data
        for r in data["results"]:
            assert "r_k" in r
            assert "r_v" in r
            assert "rank_cost" in r
            assert "all_pass" in r

    def test_old_config_still_works(self, tmp_path):
        """Legacy rank_candidates=[int] should still produce valid results."""
        calib_dir = _make_calib_dir(tmp_path, d_model=64, n_heads=4)
        output_dir = tmp_path / "rank_search_out"

        result = search_rank_per_layer(
            calib_dir=calib_dir,
            rank_candidates=[8, 16],
            n_steps=2,
            device="cpu",
            relative_tolerance=0.10,
            output_dir=output_dir,
        )
        assert 0 in result
        rk, rv = result[0]
        assert rk == rv  # legacy always symmetric
        assert rk in (8, 16)

    def test_asymmetric_pairs_in_search_result(self, tmp_path):
        calib_dir = _make_calib_dir(tmp_path, d_model=64, n_heads=4)
        result = search_rank_per_layer(
            calib_dir=calib_dir,
            rank_pairs=[(16, 8)],
            n_steps=2,
            device="cpu",
            relative_tolerance=0.10,
        )
        assert 0 in result
        rk, rv = result[0]
        assert rk == 16
        assert rv == 8


class TestCliRanksOverride:
    """--ranks CLI flag must clear new config fields so legacy symmetric
    search is guaranteed."""

    @staticmethod
    def _apply_cli_overrides(cfg, args_ranks):
        if args_ranks is not None:
            cfg.rank_search.rank_candidates = args_ranks
            cfg.rank_search.r_k_candidates = None
            cfg.rank_search.r_v_candidates = None
            cfg.rank_search.rank_pair_candidates = None

    def test_ranks_override_clears_asymmetric_candidates(self):
        from types import SimpleNamespace
        cfg = SimpleNamespace()
        cfg.rank_search = SimpleNamespace()
        cfg.rank_search.rank_pair_candidates = [[64, 64], [64, 48]]
        cfg.rank_search.r_k_candidates = [32, 48, 64]
        cfg.rank_search.r_v_candidates = [16, 32, 48]
        cfg.rank_search.rank_candidates = [8]

        self._apply_cli_overrides(cfg, args_ranks=[16, 32])

        assert cfg.rank_search.rank_candidates == [16, 32]
        assert cfg.rank_search.r_k_candidates is None
        assert cfg.rank_search.r_v_candidates is None
        assert cfg.rank_search.rank_pair_candidates is None

        pairs = build_rank_pairs(
            rank_candidates=cfg.rank_search.rank_candidates,
            r_k_candidates=cfg.rank_search.r_k_candidates,
            r_v_candidates=cfg.rank_search.r_v_candidates,
            rank_pair_candidates=cfg.rank_search.rank_pair_candidates,
            head_dim=64,
        )
        assert pairs == [(16, 16), (32, 32)]


class TestBaselineUsesMinLoss:
    """Constraint baseline must use min final_loss, not max rank_cost."""

    def test_baseline_is_min_final_loss_not_max_cost(self):
        results = [
            {
                "r_k": 64, "r_v": 64, "rank_cost": 128, "final_loss": 10.0,
                "final_logits_loss": 8.0, "final_attn_loss": 1.0,
                "final_value_loss": 1.0,
            },
            {
                "r_k": 48, "r_v": 48, "rank_cost": 96, "final_loss": 1.0,
                "final_logits_loss": 0.8, "final_attn_loss": 0.1,
                "final_value_loss": 0.1,
            },
        ]
        baseline = min(results, key=lambda x: x["final_loss"])
        assert baseline["r_k"] == 48
        assert baseline["r_v"] == 48
        assert baseline["final_loss"] == 1.0
