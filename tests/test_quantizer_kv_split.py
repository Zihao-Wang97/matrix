from __future__ import annotations

import torch
import pytest
from pathlib import Path

from hawp_laq.config import HAWPLAQConfig, QuantConfig, load_config, build_k_quantizer, build_v_quantizer
from hawp_laq.runtime.turboquant import TurboQuantMSE, TurboQuantProd


class TestQuantConfigDefaults:
    def test_k_method_is_turbo_prod(self):
        cfg = QuantConfig()
        assert cfg.k_method == "turbo_prod"

    def test_v_method_is_turbo_mse(self):
        cfg = QuantConfig()
        assert cfg.v_method == "turbo_mse"


class TestYamlKvSplit:
    def test_dev_local_k_prod_v_mse(self):
        cfg_dir = Path(__file__).resolve().parent.parent / "configs"
        cfg = load_config(cfg_dir / "dev_local.yaml")
        assert cfg.quant.k_method == "turbo_prod"
        assert cfg.quant.v_method == "turbo_mse"

    def test_run_server_k_prod_v_mse(self):
        cfg_dir = Path(__file__).resolve().parent.parent / "configs"
        cfg = load_config(cfg_dir / "run_server.yaml")
        assert cfg.quant.k_method == "turbo_prod"
        assert cfg.quant.v_method == "turbo_mse"


class TestBuildKQuantizerProd:
    def test_returns_turbo_prod(self):
        cfg = HAWPLAQConfig()
        cfg.quant.k_method = "turbo_prod"
        q = build_k_quantizer(cfg, r_k=32)
        assert isinstance(q, TurboQuantProd)

    def test_bits_propagated(self):
        cfg = HAWPLAQConfig()
        cfg.quant.k_method = "turbo_prod"
        cfg.quant.k_bits = 4
        q = build_k_quantizer(cfg, r_k=32)
        assert q.bits == 4

    def test_rotation_propagated(self):
        cfg = HAWPLAQConfig()
        cfg.quant.k_method = "turbo_prod"
        cfg.quant.use_rotation_for_k = True
        q = build_k_quantizer(cfg, r_k=32)
        assert q._rotation is not None

    def test_no_rotation(self):
        cfg = HAWPLAQConfig()
        cfg.quant.k_method = "turbo_prod"
        cfg.quant.use_rotation_for_k = False
        q = build_k_quantizer(cfg, r_k=32)
        assert q._rotation is None

    def test_group_size_propagated(self):
        cfg = HAWPLAQConfig()
        cfg.quant.k_method = "turbo_prod"
        cfg.quant.k_group_size = 64
        q = build_k_quantizer(cfg, r_k=32)
        assert q.group_size == 64

    def test_quantize_dequantize(self):
        cfg = HAWPLAQConfig()
        cfg.quant.k_method = "turbo_prod"
        cfg.quant.k_bits = 4
        q = build_k_quantizer(cfg, r_k=32)
        x = torch.randn(30, 32)
        qx = q.quantize(x)
        x_hat = q.dequantize(qx)
        assert x_hat.shape == x.shape

    def test_approx_inner_product(self):
        cfg = HAWPLAQConfig()
        cfg.quant.k_method = "turbo_prod"
        q = build_k_quantizer(cfg, r_k=32)
        x = torch.randn(20, 32)
        qx = q.quantize(x)
        query = torch.randn(5, 32)
        ip = q.approx_inner_product(query, qx)
        assert ip.shape == (5, 20)


class TestBuildKQuantizerMse:
    def test_returns_turbo_mse_when_requested(self):
        cfg = HAWPLAQConfig()
        cfg.quant.k_method = "turbo_mse"
        q = build_k_quantizer(cfg, r_k=32)
        assert isinstance(q, TurboQuantMSE)

    def test_bits_propagated(self):
        cfg = HAWPLAQConfig()
        cfg.quant.k_method = "turbo_mse"
        cfg.quant.k_bits = 8
        q = build_k_quantizer(cfg, r_k=32)
        assert q.bits == 8


class TestBuildVQuantizerMse:
    def test_returns_turbo_mse(self):
        cfg = HAWPLAQConfig()
        cfg.quant.v_method = "turbo_mse"
        q = build_v_quantizer(cfg, r_v=48)
        assert isinstance(q, TurboQuantMSE)

    def test_bits_propagated(self):
        cfg = HAWPLAQConfig()
        cfg.quant.v_method = "turbo_mse"
        cfg.quant.v_bits = 8
        q = build_v_quantizer(cfg, r_v=48)
        assert q.bits == 8

    def test_rotation_propagated(self):
        cfg = HAWPLAQConfig()
        cfg.quant.v_method = "turbo_mse"
        cfg.quant.use_rotation_for_v = True
        q = build_v_quantizer(cfg, r_v=32)
        assert q._rotation is not None

    def test_no_rotation(self):
        cfg = HAWPLAQConfig()
        cfg.quant.v_method = "turbo_mse"
        cfg.quant.use_rotation_for_v = False
        q = build_v_quantizer(cfg, r_v=32)
        assert q._rotation is None


class TestBuildVQuantizerProd:
    def test_returns_turbo_prod_when_requested(self):
        cfg = HAWPLAQConfig()
        cfg.quant.v_method = "turbo_prod"
        q = build_v_quantizer(cfg, r_v=32)
        assert isinstance(q, TurboQuantProd)


class TestKVPairFromYaml:
    def test_dev_local_builds_correct_types(self):
        cfg_dir = Path(__file__).resolve().parent.parent / "configs"
        cfg = load_config(cfg_dir / "dev_local.yaml")
        kq = build_k_quantizer(cfg, r_k=64)
        vq = build_v_quantizer(cfg, r_v=64)
        assert isinstance(kq, TurboQuantProd)
        assert isinstance(vq, TurboQuantMSE)

    def test_dev_local_bits_differ(self):
        cfg_dir = Path(__file__).resolve().parent.parent / "configs"
        cfg = load_config(cfg_dir / "dev_local.yaml")
        kq = build_k_quantizer(cfg, r_k=64)
        vq = build_v_quantizer(cfg, r_v=64)
        assert kq.bits == cfg.quant.k_bits
        assert vq.bits == cfg.quant.v_bits

    def test_roundtrip_from_yaml(self):
        cfg_dir = Path(__file__).resolve().parent.parent / "configs"
        cfg = load_config(cfg_dir / "dev_local.yaml")
        kq = build_k_quantizer(cfg, r_k=64)
        vq = build_v_quantizer(cfg, r_v=64)
        k_lat = torch.randn(20, 64)
        v_lat = torch.randn(20, 64)
        k_qx = kq.quantize(k_lat)
        v_qx = vq.quantize(v_lat)
        k_hat = kq.dequantize(k_qx)
        v_hat = vq.dequantize(v_qx)
        assert k_hat.shape == k_lat.shape
        assert v_hat.shape == v_lat.shape


class TestUnsupportedMethod:
    def test_k_unsupported(self):
        cfg = HAWPLAQConfig()
        cfg.quant.k_method = "unknown"
        with pytest.raises(ValueError, match="Unsupported quant method"):
            build_k_quantizer(cfg, r_k=32)

    def test_v_unsupported(self):
        cfg = HAWPLAQConfig()
        cfg.quant.v_method = "unknown"
        with pytest.raises(ValueError, match="Unsupported quant method"):
            build_v_quantizer(cfg, r_v=32)
