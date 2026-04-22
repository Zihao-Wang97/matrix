from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from hawp_laq.config import HAWPLAQConfig, ProjectorConfig
from hawp_laq.modeling.attention_hawp import HAWPAttention
from hawp_laq.modeling.modeling_llama_hawp import convert_llama_to_hawp
from hawp_laq.runtime.generate import _convert_and_load_projectors


def _make_config():
    return SimpleNamespace(
        hidden_size=64, num_attention_heads=4,
        num_key_value_heads=4, max_position_embeddings=2048,
        rope_theta=10000.0, model_type="opt",
        enable_bias=False, attention_dropout=0.0,
    )


class _FakeOPTAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = _make_config()
        self.q_proj = nn.Linear(64, 64, bias=False)
        self.k_proj = nn.Linear(64, 64, bias=False)
        self.v_proj = nn.Linear(64, 64, bias=False)
        self.o_proj = nn.Linear(64, 64, bias=False)


class OPTDecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _FakeOPTAttention()


class FakeModel(nn.Module):
    def __init__(self, n_layers=3):
        super().__init__()
        self.config = _make_config()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(
            [OPTDecoderLayer() for _ in range(n_layers)]
        )

    def to(self, *a, **kw):
        return self

    def eval(self):
        return self


def _write_projector(layer_dir: Path, head_dim: int = 16, r_k: int = 8, r_v: int = 8):
    layer_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "p_k": torch.eye(head_dim),
            "p_v": torch.eye(head_dim),
            "gamma": torch.ones(1),
            "r_k": r_k,
            "r_v": r_v,
        },
        layer_dir / "projector.pt",
    )


def _make_cfg(projector_dir: str, r_k: int = 8, r_v: int = 8) -> HAWPLAQConfig:
    cfg = HAWPLAQConfig()
    cfg.projector = ProjectorConfig(r_k=r_k, r_v=r_v, output_dir=Path(projector_dir))
    return cfg


def test_layers_without_projector_fallback_to_full_rank(tmp_path):
    projector_dir = tmp_path / "projectors"
    _write_projector(projector_dir / "layer_0", head_dim=16, r_k=8, r_v=8)

    model = FakeModel(n_layers=3)
    cfg = _make_cfg(str(projector_dir), r_k=8, r_v=8)

    model, _, _ = _convert_and_load_projectors(model, cfg, device="cpu", mode="hawp_only")

    hawp_layers = [m for m in model.modules() if isinstance(m, HAWPAttention)]
    assert len(hawp_layers) == 3

    assert hawp_layers[0].r_k == 8
    assert hawp_layers[0].r_v == 8

    assert hawp_layers[1].r_k == 16
    assert hawp_layers[1].r_v == 16

    assert hawp_layers[2].r_k == 16
    assert hawp_layers[2].r_v == 16


def test_effective_ranks_per_layer_prefers_projector_layers(tmp_path):
    projector_dir = tmp_path / "projectors"
    _write_projector(projector_dir / "layer_0", head_dim=16, r_k=4, r_v=4)
    _write_projector(projector_dir / "layer_2", head_dim=16, r_k=8, r_v=8)

    model = FakeModel(n_layers=3)
    cfg = _make_cfg(str(projector_dir), r_k=6, r_v=6)

    model, _, _ = _convert_and_load_projectors(model, cfg, device="cpu", mode="hawp_only")

    hawp_layers = [m for m in model.modules() if isinstance(m, HAWPAttention)]
    assert len(hawp_layers) == 3

    assert hawp_layers[0].r_k == 4
    assert hawp_layers[0].r_v == 4

    assert hawp_layers[1].r_k == 16
    assert hawp_layers[1].r_v == 16

    assert hawp_layers[2].r_k == 8
    assert hawp_layers[2].r_v == 8


def test_hawp_only_single_projector_layer_does_not_low_rank_other_layers(tmp_path):
    projector_dir = tmp_path / "projectors"
    _write_projector(projector_dir / "layer_0", head_dim=16, r_k=16, r_v=16)

    model = FakeModel(n_layers=3)
    cfg = _make_cfg(str(projector_dir), r_k=16, r_v=16)

    model, _, _ = _convert_and_load_projectors(model, cfg, device="cpu", mode="hawp_only")

    hawp_layers = [m for m in model.modules() if isinstance(m, HAWPAttention)]
    assert len(hawp_layers) == 3

    assert hawp_layers[0].r_k == 16
    assert hawp_layers[0].r_v == 16

    assert hawp_layers[1].r_k == 16
    assert hawp_layers[1].r_v == 16

    assert hawp_layers[2].r_k == 16
    assert hawp_layers[2].r_v == 16

    assert hawp_layers[0].p_k.requires_grad is False
    assert hawp_layers[1].p_k.requires_grad is False
    assert hawp_layers[2].p_k.requires_grad is False
