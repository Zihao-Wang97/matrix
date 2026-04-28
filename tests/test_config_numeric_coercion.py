from __future__ import annotations

import pytest
import yaml

from hawp_laq.config import HAWPLAQConfig, _coerce_scalar, _to_dataclass, ProjectorConfig, QuantConfig


def _write_yaml(tmp_path, raw: dict) -> str:
    p = tmp_path / "test_cfg.yaml"
    with p.open("w", encoding="utf-8") as f:
        yaml.dump(raw, f)
    return str(p)


def test_projector_gamma_min_string_to_float(tmp_path):
    proj = _to_dataclass(ProjectorConfig, {"gamma_min": "1e-4"})
    assert isinstance(proj.gamma_min, float)
    assert proj.gamma_min == 1e-4


def test_projector_eps_loss_string_to_float(tmp_path):
    proj = _to_dataclass(ProjectorConfig, {"eps_loss": "1e-8"})
    assert isinstance(proj.eps_loss, float)
    assert proj.eps_loss == 1e-8


def test_projector_r_k_string_to_int(tmp_path):
    proj = _to_dataclass(ProjectorConfig, {"r_k": "48"})
    assert isinstance(proj.r_k, int)
    assert proj.r_k == 48


def test_quant_outlier_threshold_string_to_float(tmp_path):
    quant = _to_dataclass(QuantConfig, {"outlier_threshold": "1e-3"})
    assert isinstance(quant.outlier_threshold, float)
    assert quant.outlier_threshold == 1e-3


def test_projector_early_stopping_bool_not_coerced_to_int(tmp_path):
    proj = _to_dataclass(ProjectorConfig, {"early_stopping": False})
    assert isinstance(proj.early_stopping, bool)
    assert proj.early_stopping is False


def test_full_config_load_with_string_coercion(tmp_path):
    p = _write_yaml(tmp_path, {
        "projector": {
            "gamma_min": "1e-4",
            "eps_loss": "1e-8",
            "r_k": "48",
            "early_stopping": False,
        },
        "quant": {
            "outlier_threshold": "1e-3",
        },
    })
    from hawp_laq.config import load_config
    cfg = load_config(p)

    assert isinstance(cfg.projector.gamma_min, float)
    assert cfg.projector.gamma_min == 1e-4

    assert isinstance(cfg.projector.eps_loss, float)
    assert cfg.projector.eps_loss == 1e-8

    assert isinstance(cfg.projector.r_k, int)
    assert cfg.projector.r_k == 48

    assert isinstance(cfg.projector.early_stopping, bool)
    assert cfg.projector.early_stopping is False

    assert isinstance(cfg.quant.outlier_threshold, float)
    assert cfg.quant.outlier_threshold == 1e-3
