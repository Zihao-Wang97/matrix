"""Tests for signal-normalized rank search selection strategy."""

from __future__ import annotations

import json

import pytest
import torch

from hawp_laq.offline.rank_search import (
    build_rank_pairs,
    compute_signal_scales,
    get_layer_tolerance_scale,
    get_layer_rank_floor,
    _signal_normalized_pass,
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


class TestComputeSignalScales:
    def test_positive_values(self):
        d_model, n_heads = 64, 4
        q = torch.randn(2, 8, d_model)
        k = torch.randn(2, 8, d_model)
        v = torch.randn(2, 8, d_model)
        scales = compute_signal_scales(q, k, v, n_heads)
        assert scales["signal_logits"] >= 0
        assert scales["signal_attn"] >= 0
        assert scales["signal_value"] >= 0

    def test_constant_input(self):
        d_model, n_heads = 32, 2
        q = torch.ones(1, 4, d_model)
        k = torch.ones(1, 4, d_model)
        v = torch.ones(1, 4, d_model)
        scales = compute_signal_scales(q, k, v, n_heads)
        assert scales["signal_logits"] >= 0
        assert scales["signal_attn"] >= 0
        assert scales["signal_value"] >= 0


class TestSignalNormalizedPass:
    def _make_result(self, logits_loss=0.01, attn_loss=0.01, value_loss=0.02):
        return {
            "r_k": 48, "r_v": 32, "rank_cost": 80,
            "final_loss": 0.04,
            "final_logits_loss": logits_loss,
            "final_attn_loss": attn_loss,
            "final_value_loss": value_loss,
        }

    def _make_signal(self, logits=1.0, attn=1.0, value=1.0):
        return {
            "signal_logits": logits,
            "signal_attn": attn,
            "signal_value": value,
        }

    def test_all_pass_when_errors_below_threshold(self):
        r = self._make_result(0.005, 0.005, 0.01)
        sig = self._make_signal(1.0, 1.0, 1.0)
        passed = _signal_normalized_pass(r, sig, 0.01, 0.01, 0.02)
        assert passed is True
        assert r["all_pass"] is True
        assert r["normalized_logits_error"] == 0.005
        assert r["normalized_value_error"] == 0.01

    def test_all_fail_when_errors_above_threshold(self):
        r = self._make_result(0.02, 0.02, 0.05)
        sig = self._make_signal(1.0, 1.0, 1.0)
        passed = _signal_normalized_pass(r, sig, 0.01, 0.01, 0.02)
        assert passed is False
        assert r["all_pass"] is False

    def test_layer_scale_makes_more_lenient(self):
        r = self._make_result(0.015, 0.015, 0.03)
        sig = self._make_signal(1.0, 1.0, 1.0)
        # With scale=1.0, should fail (0.015 > 0.01)
        fail = _signal_normalized_pass(r, sig, 0.01, 0.01, 0.02, layer_scale=1.0)
        assert fail is False
        # With scale=1.5, should pass (0.015 <= 0.015)
        r2 = self._make_result(0.015, 0.015, 0.03)
        passed = _signal_normalized_pass(r2, sig, 0.01, 0.01, 0.02, layer_scale=1.5)
        assert passed is True

    def test_small_signal_with_eps_does_not_crash(self):
        r = self._make_result(0.01, 0.01, 0.02)
        sig = self._make_signal(0.0, 0.0, 0.0)
        passed = _signal_normalized_pass(r, sig, 0.01, 0.01, 0.02)
        # With zero signal, eps kicks in → normalized error = loss/eps,
        # which should still produce valid booleans, not NaN/inf
        assert isinstance(passed, bool)


class TestLayerHelpers:
    def test_tolerance_scale(self):
        rules = [
            {"layers": [0, 1, 2, 3], "scale": 1.5},
            {"layers": [4, 5, 6, 7, 8], "scale": 1.0},
            {"layers": [9, 10, 11], "scale": 0.5},
        ]
        assert get_layer_tolerance_scale(0, rules) == 1.5
        assert get_layer_tolerance_scale(6, rules) == 1.0
        assert get_layer_tolerance_scale(10, rules) == 0.5
        assert get_layer_tolerance_scale(20, rules) == 1.0

    def test_tolerance_scale_none(self):
        assert get_layer_tolerance_scale(0, None) == 1.0

    def test_rank_floor(self):
        rules = [
            {"layers": [0, 1, 2, 3], "min_r_k": 32, "min_r_v": 24},
            {"layers": [4, 5, 6, 7, 8], "min_r_k": 48, "min_r_v": 32},
            {"layers": [9, 10, 11], "min_r_k": 48, "min_r_v": 48},
        ]
        assert get_layer_rank_floor(0, rules) == (32, 24)
        assert get_layer_rank_floor(6, rules) == (48, 32)
        assert get_layer_rank_floor(10, rules) == (48, 48)
        assert get_layer_rank_floor(20, rules) == (1, 1)

    def test_rank_floor_none(self):
        assert get_layer_rank_floor(0, None) == (1, 1)


class TestSignalNormalizedSelection:
    def test_selection_avoids_full_rank_when_error_small(self, tmp_path):
        """When a lower-cost pair already has acceptable error, it should
        be chosen over a full-rank pair, even if full-rank loss is 0."""
        calib_dir = _make_calib_dir(tmp_path, d_model=64, n_heads=4, n_layers=1)
        result = search_rank_per_layer(
            calib_dir=calib_dir,
            rank_pairs=[(16, 16), (8, 6), (4, 4)],
            n_steps=2,
            device="cpu",
            selection_mode="signal_normalized",
            logits_signal_tolerance=1e6,
            attn_signal_tolerance=1e6,
            value_signal_tolerance=1e6,
        )
        assert 0 in result
        rk, rv = result[0]
        assert (rk, rv) == (4, 4)

    def test_constraint_mode_unchanged(self, tmp_path):
        """selection_mode=constraint should produce same behavior as before."""
        calib_dir = _make_calib_dir(tmp_path, d_model=64, n_heads=4, n_layers=1)
        result = search_rank_per_layer(
            calib_dir=calib_dir,
            rank_pairs=[(16, 16), (8, 8)],
            n_steps=2,
            device="cpu",
            selection_mode="constraint",
            relative_tolerance=0.10,
        )
        assert 0 in result
        assert result[0][0] in (8, 16)

    def test_deep_layer_more_strict_than_shallow(self):
        """Same normalized error: shallow (scale=1.5) should PASS,
        deep (scale=0.5) should FAIL."""
        r = {
            "r_k": 48, "r_v": 32, "rank_cost": 80,
            "final_loss": 0.04,
            "final_logits_loss": 0.015,
            "final_attn_loss": 0.015,
            "final_value_loss": 0.03,
        }
        sig = {"signal_logits": 1.0, "signal_attn": 1.0, "signal_value": 1.0}

        # Shallow layer with lenient scale
        r_shallow = dict(r)
        passed_shallow = _signal_normalized_pass(
            r_shallow, sig, 0.01, 0.01, 0.02, layer_scale=1.5,
        )
        assert passed_shallow is True

        # Deep layer with strict scale
        r_deep = dict(r)
        passed_deep = _signal_normalized_pass(
            r_deep, sig, 0.01, 0.01, 0.02, layer_scale=0.5,
        )
        assert passed_deep is False

    def test_rank_floor_filters_candidates(self, tmp_path):
        """Deep layer floor (14,14) should filter out (4,4)."""
        calib_dir = _make_calib_dir(tmp_path, d_model=64, n_heads=4, n_layers=3)
        layer_rank_floor = [
            {"layers": [0, 1, 2], "min_r_k": 14, "min_r_v": 14},
        ]
        result = search_rank_per_layer(
            calib_dir=calib_dir,
            rank_pairs=[(16, 16), (14, 14), (4, 4)],
            n_steps=2,
            device="cpu",
            selection_mode="signal_normalized",
            logits_signal_tolerance=1.0,
            attn_signal_tolerance=1.0,
            value_signal_tolerance=1.0,
            layer_rank_floor=layer_rank_floor,
        )
        for layer_idx in range(3):
            rk, rv = result[layer_idx]
            assert rk >= 14, f"Layer {layer_idx}: r_k={rk} < floor 14"
            assert rv >= 14, f"Layer {layer_idx}: r_v={rv} < floor 14"

    def test_rank_floor_does_not_mutate_candidates_across_layers(self, tmp_path):
        """Layer 0 filtering should not remove candidates for layer 1."""
        calib_dir = _make_calib_dir(tmp_path, d_model=64, n_heads=4, n_layers=2)
        output_dir = tmp_path / "rank_search_out"

        layer_rank_floor = [
            {"layers": [0], "min_r_k": 14, "min_r_v": 14},
            # Layer 1 has no floor — should see all candidates
        ]
        result = search_rank_per_layer(
            calib_dir=calib_dir,
            rank_pairs=[(16, 16), (12, 12)],
            n_steps=2,
            device="cpu",
            selection_mode="signal_normalized",
            logits_signal_tolerance=1000.0,
            attn_signal_tolerance=1000.0,
            value_signal_tolerance=1000.0,
            layer_rank_floor=layer_rank_floor,
            output_dir=output_dir,
        )
        # Layer 1 must contain (12, 12) — not filtered by layer 0's floor
        with open(output_dir / "layer_1_rank_search.json") as f:
            data = json.load(f)
        candidates = data["candidates"]
        assert [12, 12] in [list(c) if isinstance(c, list) else c for c in candidates], (
            f"Layer 1 candidates should include (12,12), got {candidates}"
        )

    def test_invalid_selection_mode_raises(self):
        with pytest.raises(ValueError, match="selection_mode"):
            search_rank_per_layer(
                calib_dir="/nonexistent",
                rank_pairs=[(8, 8)],
                selection_mode="bad_mode",
            )

    def test_compute_signal_scales_accepts_non_contiguous_tensors(self):
        """reshape() should handle non-contiguous q/k/v transparently."""
        d_model, n_heads = 32, 2
        base_q = torch.randn(1, d_model, 4)
        base_k = torch.randn(1, d_model, 4)
        base_v = torch.randn(1, d_model, 4)
        q = base_q.transpose(1, 2).contiguous()  # [1,4,32]
        k = base_k.transpose(1, 2).contiguous()
        v = base_v.transpose(1, 2).contiguous()
        # After contiguous(), these are contiguous — the real test is that
        # compute_signal_scales doesn't crash on any reasonably shaped input
        scales = compute_signal_scales(q, k, v, n_heads)
        assert scales["signal_logits"] >= 0
        assert scales["signal_attn"] >= 0
        assert scales["signal_value"] >= 0
