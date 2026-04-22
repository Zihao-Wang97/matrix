from __future__ import annotations

import warnings
import json
from pathlib import Path

import pytest
import torch

from hawp_laq.runtime.projector_bank import rebuild_ranks_json, inspect_projector_dir


def _write_projector(
    layer_dir: Path,
    head_dim: int,
    r_k: int,
    r_v: int,
    include_ranks: bool = True,
    p_k_shape: tuple[int, int] | None = None,
    p_v_shape: tuple[int, int] | None = None,
) -> None:
    layer_dir.mkdir(parents=True, exist_ok=True)
    if p_k_shape is None:
        p_k_shape = (head_dim, head_dim)
    if p_v_shape is None:
        p_v_shape = (head_dim, head_dim)
    data: dict = {
        "p_k": torch.randn(*p_k_shape),
        "p_v": torch.randn(*p_v_shape),
        "gamma": torch.ones(1),
    }
    if include_ranks:
        data["r_k"] = r_k
        data["r_v"] = r_v
    torch.save(data, layer_dir / "projector.pt")


def test_single_group_rebuilds_ranks_json(tmp_path):
    projector_dir = tmp_path / "projectors"
    _write_projector(projector_dir / "layer_0", head_dim=64, r_k=16, r_v=16)
    _write_projector(projector_dir / "layer_1", head_dim=64, r_k=8, r_v=8)

    ranks_path = rebuild_ranks_json(projector_dir)

    assert ranks_path.exists()
    with open(ranks_path) as f:
        ranks = json.load(f)
    assert ranks["0"] == {"r_k": 16, "r_v": 16}
    assert ranks["1"] == {"r_k": 8, "r_v": 8}


def test_rebuild_ranks_json_skips_legacy_files_with_warning(tmp_path):
    projector_dir = tmp_path / "projectors"
    _write_projector(projector_dir / "layer_0", head_dim=64, r_k=16, r_v=16)
    _write_projector(projector_dir / "layer_1", head_dim=768, r_k=50, r_v=50, include_ranks=False)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ranks_path = rebuild_ranks_json(projector_dir)
        legacy_warnings = [x for x in w if "missing r_k/r_v" in str(x.message)]
        assert len(legacy_warnings) == 1
        assert "layer 1" in str(legacy_warnings[0].message).lower() or "Layer 1" in str(legacy_warnings[0].message)

    with open(ranks_path) as f:
        ranks = json.load(f)
    assert "0" in ranks
    assert "1" not in ranks


def test_inspect_projector_dir_detects_legacy_shape_mismatch(tmp_path):
    projector_dir = tmp_path / "projectors"

    _write_projector(projector_dir / "layer_0", head_dim=64, r_k=16, r_v=16)
    _write_projector(
        projector_dir / "layer_1",
        head_dim=768, r_k=50, r_v=50,
        p_k_shape=(768, 50), p_v_shape=(768, 50),
    )
    _write_projector(projector_dir / "layer_2", head_dim=64, r_k=16, r_v=16, include_ranks=False)

    report = inspect_projector_dir(
        projector_dir,
        expected_head_dim=64,
        default_r_k=16,
        default_r_v=16,
    )

    assert 0 in report["valid_layers"]
    assert 1 in report["shape_mismatch_layers"]
    assert 2 in report["legacy_layers"]


def test_generate_preflight_fails_fast_on_legacy_projectors(tmp_path):
    from hawp_laq.runtime.generate import _convert_and_load_projectors
    from types import SimpleNamespace
    import torch.nn as nn
    from hawp_laq.config import HAWPLAQConfig, ProjectorConfig

    projector_dir = tmp_path / "projectors"
    _write_projector(
        projector_dir / "layer_0",
        head_dim=768, r_k=50, r_v=50,
        p_k_shape=(768, 50), p_v_shape=(768, 50),
    )

    class FakeAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(
                hidden_size=64, num_attention_heads=4,
                num_key_value_heads=4, max_position_embeddings=2048,
                rope_theta=10000.0, model_type="opt",
                enable_bias=False, attention_dropout=0.0,
            )
            self.q_proj = nn.Linear(64, 64, bias=False)
            self.k_proj = nn.Linear(64, 64, bias=False)
            self.v_proj = nn.Linear(64, 64, bias=False)
            self.o_proj = nn.Linear(64, 64, bias=False)

    class OPTDecoderLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = FakeAttn()

    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(
                hidden_size=64, num_attention_heads=4,
                num_key_value_heads=4, max_position_embeddings=2048,
                rope_theta=10000.0, model_type="opt",
                enable_bias=False, attention_dropout=0.0,
            )
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([OPTDecoderLayer()])

        def to(self, *a, **kw):
            return self

        def eval(self):
            return self

    cfg = HAWPLAQConfig()
    cfg.projector = ProjectorConfig(r_k=16, r_v=16, output_dir=str(projector_dir))

    model = FakeModel()

    with pytest.raises(ValueError, match="Incompatible projector files found"):
        _convert_and_load_projectors(model, cfg, device="cpu", mode="hawp_only")
