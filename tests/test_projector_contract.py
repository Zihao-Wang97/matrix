from __future__ import annotations

import warnings
from pathlib import Path

import pytest
import torch

from hawp_laq.runtime.projector_bank import (
    normalize_projector_data,
    rebuild_ranks_json,
    inspect_projector_dir,
)
from hawp_laq.offline.projector_trainer import ProjectorTrainer
from hawp_laq.utils.io import load_json

CANONICAL_KEYS = (
    "p_k", "p_v", "gamma", "r_k", "r_v",
    "best_step", "best_calib_total", "actual_steps", "stopped_early", "metrics",
)


class TestNormalizeProjectorData:
    def test_gamma_v_fallback_with_warning(self):
        data = {
            "p_k": torch.randn(8, 4),
            "p_v": torch.randn(8, 4),
            "gamma_k": torch.tensor([1.5]),
            "gamma_v": torch.tensor([2.7]),
        }
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = normalize_projector_data(data, layer_idx=0)
            gamma_warns = [x for x in w if "gamma_v" in str(x.message)]
            assert len(gamma_warns) == 1
        assert "gamma" in result
        assert torch.allclose(result["gamma"], torch.tensor([2.7]))

    def test_gamma_k_fallback_when_no_gamma_v(self):
        data = {
            "p_k": torch.randn(8, 4),
            "p_v": torch.randn(8, 4),
            "gamma_k": torch.tensor([1.5]),
        }
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = normalize_projector_data(data, layer_idx=0)
            gamma_warns = [x for x in w if "gamma_k" in str(x.message)]
            assert len(gamma_warns) == 1
        assert "gamma" in result
        assert torch.allclose(result["gamma"], torch.tensor([1.5]))

    def test_gamma_preferred_over_legacy(self):
        data = {
            "p_k": torch.randn(8, 4),
            "p_v": torch.randn(8, 4),
            "gamma": torch.tensor([3.3]),
            "gamma_k": torch.tensor([1.5]),
            "gamma_v": torch.tensor([2.7]),
        }
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = normalize_projector_data(data, layer_idx=0)
            gamma_warns = [x for x in w if "gamma" in str(x.message).lower() and "fallback" in str(x.message)]
            assert len(gamma_warns) == 0
        assert torch.allclose(result["gamma"], torch.tensor([3.3]))

    def test_r_k_r_v_inferred_from_non_square_shape(self):
        data = {
            "p_k": torch.randn(8, 4),
            "p_v": torch.randn(8, 3),
            "gamma": torch.tensor([1.0]),
        }
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = normalize_projector_data(data, layer_idx=5)
            rk_warns = [x for x in w if "r_k" in str(x.message)]
            rv_warns = [x for x in w if "r_v" in str(x.message)]
            assert len(rk_warns) == 1
            assert len(rv_warns) == 1
        assert result["r_k"] == 4
        assert result["r_v"] == 3

    def test_no_inference_for_square_p_k(self):
        data = {
            "p_k": torch.randn(8, 8),
            "p_v": torch.randn(8, 8),
            "gamma": torch.tensor([1.0]),
        }
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = normalize_projector_data(data, layer_idx=0)
            rk_warns = [x for x in w if "r_k" in str(x.message)]
            assert len(rk_warns) == 0
        assert "r_k" not in result
        assert "r_v" not in result

    def test_no_op_on_canonical_data(self):
        data = {
            "p_k": torch.randn(8, 4),
            "p_v": torch.randn(8, 3),
            "gamma": torch.tensor([1.5]),
            "r_k": 4,
            "r_v": 3,
        }
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = normalize_projector_data(data, layer_idx=0)
            fallback_warns = [x for x in w if "fallback" in str(x.message) or "inferred" in str(x.message)]
            assert len(fallback_warns) == 0
        assert result["r_k"] == 4
        assert result["r_v"] == 3
        assert torch.allclose(result["gamma"], torch.tensor([1.5]))


class TestRoundtrip:
    def test_save_result_roundtrip_preserves_canonical_fields(self, tmp_path):
        d_model, r_k, r_v, n_heads = 32, 8, 6, 4
        trainer = ProjectorTrainer(d_model, r_k, r_v, n_heads, device="cpu")
        q = torch.randn(1, 4, d_model)
        k = torch.randn(1, 4, d_model)
        v = torch.randn(1, 4, d_model)
        result = trainer.train_one_group(q, k, v, n_steps=3, optimizer="riemannian_adam")

        ProjectorTrainer.save_result(result, 0, tmp_path)

        pt_path = tmp_path / "layer_0" / "projector.pt"
        loaded = torch.load(pt_path, map_location="cpu", weights_only=False)

        for key in CANONICAL_KEYS:
            assert key in loaded, f"Missing canonical key: {key}"

        assert torch.allclose(result["p_k"], loaded["p_k"])
        assert torch.allclose(result["p_v"], loaded["p_v"])
        assert torch.allclose(result["gamma"].float(), loaded["gamma"].float())
        assert loaded["r_k"] == r_k
        assert loaded["r_v"] == r_v

    def test_legacy_format_normalizes_correctly(self, tmp_path):
        head_dim, r_k, r_v = 16, 4, 3
        layer_dir = tmp_path / "proj" / "layer_0"
        layer_dir.mkdir(parents=True)
        legacy = {
            "p_k": torch.randn(head_dim, r_k),
            "p_v": torch.randn(head_dim, r_v),
            "gamma_k": torch.tensor([1.5]),
            "gamma_v": torch.tensor([2.7]),
        }
        torch.save(legacy, layer_dir / "projector.pt")

        data = torch.load(layer_dir / "projector.pt", map_location="cpu", weights_only=False)
        with pytest.warns(UserWarning) as record:
            data = normalize_projector_data(data, layer_idx=0)

        messages = [str(w.message) for w in record]
        assert any("missing 'gamma'" in m for m in messages), f"Expected 'missing gamma' warning, got {messages}"
        assert any("missing 'r_k'" in m for m in messages), f"Expected 'missing r_k' warning, got {messages}"
        assert any("missing 'r_v'" in m for m in messages), f"Expected 'missing r_v' warning, got {messages}"

        assert "gamma" in data
        assert torch.allclose(data["gamma"], torch.tensor([2.7]))
        assert data["r_k"] == r_k
        assert data["r_v"] == r_v


class TestRanksJsonNoConflict:
    def test_ranks_json_only_has_r_k_r_v(self, tmp_path):
        d_model, r_k, r_v, n_heads = 32, 8, 6, 4
        trainer = ProjectorTrainer(d_model, r_k, r_v, n_heads, device="cpu")
        q = torch.randn(1, 4, d_model)
        k = torch.randn(1, 4, d_model)
        v = torch.randn(1, 4, d_model)
        result = trainer.train_one_group(q, k, v, n_steps=3, optimizer="riemannian_adam")

        ProjectorTrainer.save_result(result, 0, tmp_path)
        rebuild_ranks_json(tmp_path)

        ranks = load_json(tmp_path / "ranks.json")
        assert set(ranks["0"].keys()) == {"r_k", "r_v"}


class TestInspectProjectorDirLegacy:
    def test_legacy_non_square_projector_classified_as_valid(self, tmp_path):
        projector_dir = tmp_path / "projectors"
        layer_dir = projector_dir / "layer_0"
        layer_dir.mkdir(parents=True)
        legacy = {
            "p_k": torch.randn(16, 8),
            "p_v": torch.randn(16, 6),
            "gamma": torch.ones(1),
        }
        torch.save(legacy, layer_dir / "projector.pt")

        with pytest.warns(UserWarning) as record:
            report = inspect_projector_dir(
                projector_dir,
                expected_head_dim=16,
                default_r_k=8,
                default_r_v=6,
            )

        messages = [str(w.message) for w in record]
        assert any("missing 'r_k'" in m for m in messages), f"Expected 'missing r_k' warning, got {messages}"
        assert any("missing 'r_v'" in m for m in messages), f"Expected 'missing r_v' warning, got {messages}"

        assert 0 in report["valid_layers"]
        assert 0 not in report["legacy_layers"]
        assert 0 not in report["shape_mismatch_layers"]

    def test_legacy_square_projector_classified_as_legacy(self, tmp_path):
        projector_dir = tmp_path / "projectors"
        layer_dir = projector_dir / "layer_0"
        layer_dir.mkdir(parents=True)
        legacy = {
            "p_k": torch.randn(16, 16),
            "p_v": torch.randn(16, 16),
            "gamma": torch.ones(1),
        }
        torch.save(legacy, layer_dir / "projector.pt")

        report = inspect_projector_dir(
            projector_dir,
            expected_head_dim=16,
            default_r_k=8,
            default_r_v=6,
        )

        assert 0 in report["legacy_layers"]
        assert 0 not in report["valid_layers"]
