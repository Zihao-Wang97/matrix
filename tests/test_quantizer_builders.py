from __future__ import annotations

import torch
import pytest
from pathlib import Path

from hawp_laq.config import (
    HAWPLAQConfig,
    QuantConfig,
    load_config,
    build_k_quantizer,
    build_v_quantizer,
)
from hawp_laq.runtime.turboquant import TurboQuantMSE


class TestQuantConfig:
    def test_defaults(self):
        cfg = QuantConfig()
        assert cfg.enabled is False
        assert cfg.k_method == "turbo_prod"
        assert cfg.v_method == "turbo_mse"
        assert cfg.k_bits == 4
        assert cfg.v_bits == 8
        assert cfg.use_rotation_for_k is True
        assert cfg.use_rotation_for_v is True
        assert cfg.k_group_size == 128
        assert cfg.v_group_size == 128

    def test_custom(self):
        cfg = QuantConfig(
            enabled=True, k_method="turbo_mse", v_method="turbo_mse",
            k_bits=8, v_bits=4,
            use_rotation_for_k=False, use_rotation_for_v=True,
        )
        assert cfg.k_bits == 8
        assert cfg.v_bits == 4
        assert cfg.use_rotation_for_k is False


class TestQuantConfigFromYaml:
    def test_dev_local(self):
        script_dir = Path(__file__).resolve().parent.parent / "configs"
        cfg = load_config(script_dir / "dev_local.yaml")
        assert cfg.quant.enabled is True
        assert cfg.quant.k_method == "turbo_prod"
        assert cfg.quant.v_method == "turbo_mse"
        assert cfg.quant.k_bits == 4
        assert cfg.quant.v_bits == 8
        assert cfg.quant.use_rotation_for_k is False
        assert cfg.quant.use_rotation_for_v is False

    def test_run_server(self):
        script_dir = Path(__file__).resolve().parent.parent / "configs"
        cfg = load_config(script_dir / "run_server.yaml")
        assert cfg.quant.enabled is True
        assert cfg.quant.k_bits == 4
        assert cfg.quant.v_bits == 8


class TestBuildKQuantizer:
    def test_returns_turbo_mse(self):
        cfg = HAWPLAQConfig()
        cfg.quant.k_method = "turbo_mse"
        cfg.quant.k_bits = 4
        cfg.quant.use_rotation_for_k = True
        cfg.quant.k_group_size = 64
        q = build_k_quantizer(cfg, r_k=32)
        assert isinstance(q, TurboQuantMSE)
        assert q.dim == 32
        assert q.bits == 4
        assert q.use_rotation is True
        assert q.group_size == 64

    def test_no_rotation(self):
        cfg = HAWPLAQConfig()
        cfg.quant.k_method = "turbo_mse"
        cfg.quant.use_rotation_for_k = False
        q = build_k_quantizer(cfg, r_k=64)
        assert q._rotation is None

    def test_unsupported_method_raises(self):
        cfg = HAWPLAQConfig()
        cfg.quant.k_method = "unknown"
        with pytest.raises(ValueError, match="Unsupported quant method"):
            build_k_quantizer(cfg, r_k=32)

    def test_quantize_roundtrip(self):
        cfg = HAWPLAQConfig()
        cfg.quant.k_method = "turbo_mse"
        cfg.quant.k_bits = 4
        cfg.quant.use_rotation_for_k = True
        q = build_k_quantizer(cfg, r_k=32)
        x = torch.randn(50, 32)
        qx = q.quantize(x)
        x_hat = q.dequantize(qx)
        assert x_hat.shape == x.shape
        mse = (x - x_hat).pow(2).mean().item()
        assert mse < 1.0


class TestBuildVQuantizer:
    def test_returns_turbo_mse(self):
        cfg = HAWPLAQConfig()
        cfg.quant.v_method = "turbo_mse"
        cfg.quant.v_bits = 8
        cfg.quant.use_rotation_for_v = True
        cfg.quant.v_group_size = 128
        q = build_v_quantizer(cfg, r_v=48)
        assert isinstance(q, TurboQuantMSE)
        assert q.dim == 48
        assert q.bits == 8
        assert q.use_rotation is True

    def test_no_rotation(self):
        cfg = HAWPLAQConfig()
        cfg.quant.v_method = "turbo_mse"
        cfg.quant.use_rotation_for_v = False
        q = build_v_quantizer(cfg, r_v=64)
        assert q._rotation is None

    def test_unsupported_method_raises(self):
        cfg = HAWPLAQConfig()
        cfg.quant.v_method = "unknown"
        with pytest.raises(ValueError, match="Unsupported quant method"):
            build_v_quantizer(cfg, r_v=32)

    def test_quantize_roundtrip(self):
        cfg = HAWPLAQConfig()
        cfg.quant.v_method = "turbo_mse"
        cfg.quant.v_bits = 8
        cfg.quant.use_rotation_for_v = True
        q = build_v_quantizer(cfg, r_v=48)
        x = torch.randn(50, 48)
        qx = q.quantize(x)
        x_hat = q.dequantize(qx)
        assert x_hat.shape == x.shape
        mse = (x - x_hat).pow(2).mean().item()
        assert mse < 0.5


class TestBuildKVPair:
    def test_different_bits_and_rotation(self):
        cfg = HAWPLAQConfig()
        cfg.quant.k_method = "turbo_mse"
        cfg.quant.v_method = "turbo_mse"
        cfg.quant.k_bits = 4
        cfg.quant.v_bits = 8
        cfg.quant.use_rotation_for_k = True
        cfg.quant.use_rotation_for_v = False
        kq = build_k_quantizer(cfg, r_k=32)
        vq = build_v_quantizer(cfg, r_v=64)
        assert kq.bits == 4
        assert vq.bits == 8
        assert kq.use_rotation is True
        assert vq.use_rotation is False

    def test_from_yaml_roundtrip(self):
        script_dir = Path(__file__).resolve().parent.parent / "configs"
        cfg = load_config(script_dir / "dev_local.yaml")
        kq = build_k_quantizer(cfg, r_k=cfg.projector.r_k or 64)
        vq = build_v_quantizer(cfg, r_v=cfg.projector.r_v or 64)
        k_lat = torch.randn(30, kq.dim)
        v_lat = torch.randn(30, vq.dim)
        k_hat = kq.dequantize(kq.quantize(k_lat))
        v_hat = vq.dequantize(vq.quantize(v_lat))
        assert k_hat.shape == k_lat.shape
        assert v_hat.shape == v_lat.shape
