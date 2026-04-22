from __future__ import annotations

import torch
import pytest

from hawp_laq.runtime.turboquant import TurboQuantMSE
from hawp_laq.runtime.latent_quant_bridge import (
    QuantizedLatentKV,
    create_kv_quantizers,
    quantize_kv_latents,
    dequantize_kv_latents,
    latent_kv_bytes,
    baseline_kv_bytes,
    saving_ratio,
)


class TestCreateKVQuantizers:
    def test_creates_matched_pair(self):
        kq, vq = create_kv_quantizers(r_k=32, r_v=64)
        assert kq.dim == 32
        assert vq.dim == 64

    def test_custom_bits(self):
        kq, vq = create_kv_quantizers(r_k=32, r_v=32, k_bits=8, v_bits=4)
        assert kq.bits == 8
        assert vq.bits == 4

    def test_no_rotation(self):
        kq, vq = create_kv_quantizers(r_k=32, r_v=32, use_rotation=False)
        assert kq._rotation is None
        assert vq._rotation is None


class TestQuantizeKVLatents:
    def test_2d_input(self):
        kq, vq = create_kv_quantizers(r_k=32, r_v=48, use_rotation=False)
        k_lat = torch.randn(20, 32)
        v_lat = torch.randn(20, 48)
        qkv = quantize_kv_latents(k_lat, v_lat, kq, vq)
        assert isinstance(qkv, QuantizedLatentKV)
        assert qkv.k_q.q.shape == (20, 32)
        assert qkv.v_q.q.shape == (20, 48)

    def test_3d_input(self):
        kq, vq = create_kv_quantizers(r_k=32, r_v=48, use_rotation=False)
        k_lat = torch.randn(2, 20, 32)
        v_lat = torch.randn(2, 20, 48)
        qkv = quantize_kv_latents(k_lat, v_lat, kq, vq)
        assert qkv.k_q.shape_orig == (2, 20, 32)
        assert qkv.v_q.shape_orig == (2, 20, 48)

    def test_dim_mismatch_raises(self):
        kq, vq = create_kv_quantizers(r_k=32, r_v=32, use_rotation=False)
        with pytest.raises(ValueError, match="k_lat last dim"):
            quantize_kv_latents(torch.randn(10, 16), torch.randn(10, 32), kq, vq)

    def test_v_dim_mismatch_raises(self):
        kq, vq = create_kv_quantizers(r_k=32, r_v=32, use_rotation=False)
        with pytest.raises(ValueError, match="v_lat last dim"):
            quantize_kv_latents(torch.randn(10, 32), torch.randn(10, 16), kq, vq)

    def test_unsupported_ndim_raises(self):
        kq, vq = create_kv_quantizers(r_k=32, r_v=32)
        with pytest.raises(ValueError, match="2-D"):
            quantize_kv_latents(torch.randn(10), torch.randn(10, 32), kq, vq)


class TestDequantizeKVLatents:
    def test_roundtrip_2d(self):
        kq, vq = create_kv_quantizers(r_k=32, r_v=48, use_rotation=True)
        k_lat = torch.randn(50, 32)
        v_lat = torch.randn(50, 48)
        qkv = quantize_kv_latents(k_lat, v_lat, kq, vq)
        k_hat, v_hat = dequantize_kv_latents(qkv, kq, vq)
        assert k_hat.shape == k_lat.shape
        assert v_hat.shape == v_lat.shape
        k_mse = (k_lat - k_hat).pow(2).mean().item()
        v_mse = (v_lat - v_hat).pow(2).mean().item()
        assert k_mse < 1.0
        assert v_mse < 1.0

    def test_roundtrip_3d(self):
        kq, vq = create_kv_quantizers(r_k=32, r_v=32, use_rotation=True)
        k_lat = torch.randn(2, 30, 32)
        v_lat = torch.randn(2, 30, 32)
        qkv = quantize_kv_latents(k_lat, v_lat, kq, vq)
        k_hat, v_hat = dequantize_kv_latents(qkv, kq, vq)
        assert k_hat.shape == k_lat.shape
        assert v_hat.shape == v_lat.shape

    def test_8bit_better_than_4bit(self):
        torch.manual_seed(0)
        k_lat = torch.randn(100, 32)
        v_lat = torch.randn(100, 32)
        kq4, vq4 = create_kv_quantizers(r_k=32, r_v=32, k_bits=4, v_bits=4, use_rotation=True)
        kq8, vq8 = create_kv_quantizers(r_k=32, r_v=32, k_bits=8, v_bits=8, use_rotation=True)
        qkv4 = quantize_kv_latents(k_lat, v_lat, kq4, vq4)
        qkv8 = quantize_kv_latents(k_lat, v_lat, kq8, vq8)
        k4_mse = (k_lat - kq4.dequantize(qkv4.k_q)).pow(2).mean().item()
        k8_mse = (k_lat - kq8.dequantize(qkv8.k_q)).pow(2).mean().item()
        assert k8_mse < k4_mse

    def test_rotation_vs_no_rotation(self):
        torch.manual_seed(1)
        # Skewed latent where rotation helps
        k_lat = torch.randn(100, 32)
        k_lat[:, :4] *= 10.0
        kq_rot, vq_rot = create_kv_quantizers(r_k=32, r_v=32, use_rotation=True)
        kq_norot, vq_norot = create_kv_quantizers(r_k=32, r_v=32, use_rotation=False)
        qkv_rot = quantize_kv_latents(k_lat, k_lat, kq_rot, vq_rot)
        qkv_norot = quantize_kv_latents(k_lat, k_lat, kq_norot, vq_norot)
        k_rot_mse = (k_lat - kq_rot.dequantize(qkv_rot.k_q)).pow(2).mean().item()
        k_norot_mse = (k_lat - kq_norot.dequantize(qkv_norot.k_q)).pow(2).mean().item()
        assert k_rot_mse < k_norot_mse

    def test_different_rk_rv(self):
        kq, vq = create_kv_quantizers(r_k=16, r_v=64, use_rotation=True)
        k_lat = torch.randn(30, 16)
        v_lat = torch.randn(30, 64)
        qkv = quantize_kv_latents(k_lat, v_lat, kq, vq)
        k_hat, v_hat = dequantize_kv_latents(qkv, kq, vq)
        assert k_hat.shape == (30, 16)
        assert v_hat.shape == (30, 64)


class TestLatentKVBytes:
    def test_bytes_positive(self):
        kq, vq = create_kv_quantizers(r_k=32, r_v=32, use_rotation=False)
        k_lat = torch.randn(100, 32)
        v_lat = torch.randn(100, 32)
        qkv = quantize_kv_latents(k_lat, v_lat, kq, vq)
        info = latent_kv_bytes(qkv, kq, vq)
        assert info["k_bytes"] > 0
        assert info["v_bytes"] > 0
        assert info["total_bytes"] == info["k_bytes"] + info["v_bytes"]

    def test_4bit_smaller_than_fp16(self):
        kq, vq = create_kv_quantizers(r_k=32, r_v=32, use_rotation=False, k_bits=4, v_bits=4)
        k_lat = torch.randn(100, 32)
        v_lat = torch.randn(100, 32)
        qkv = quantize_kv_latents(k_lat, v_lat, kq, vq)
        quant = latent_kv_bytes(qkv, kq, vq)
        base = baseline_kv_bytes(100, 32, 32)
        assert quant["total_bytes"] < base["total_bytes"]


class TestBaselineKVBytes:
    def test_float16(self):
        info = baseline_kv_bytes(100, 32, 48, dtype=torch.float16)
        assert info["k_bytes"] == 100 * 32 * 2
        assert info["v_bytes"] == 100 * 48 * 2

    def test_float32(self):
        info = baseline_kv_bytes(100, 32, 32, dtype=torch.float32)
        assert info["total_bytes"] == 100 * 32 * 2 * 4


class TestSavingRatio:
    def test_ratio_between_0_and_1(self):
        kq, vq = create_kv_quantizers(r_k=32, r_v=32, use_rotation=False, k_bits=4, v_bits=4)
        k_lat = torch.randn(100, 32)
        v_lat = torch.randn(100, 32)
        qkv = quantize_kv_latents(k_lat, v_lat, kq, vq)
        r = saving_ratio(qkv, kq, vq)
        assert 0.0 < r < 1.0

    def test_4bit_better_ratio_than_8bit(self):
        k_lat = torch.randn(100, 32)
        v_lat = torch.randn(100, 32)
        kq4, vq4 = create_kv_quantizers(r_k=32, r_v=32, k_bits=4, v_bits=4, use_rotation=False)
        kq8, vq8 = create_kv_quantizers(r_k=32, r_v=32, k_bits=8, v_bits=8, use_rotation=False)
        qkv4 = quantize_kv_latents(k_lat, v_lat, kq4, vq4)
        qkv8 = quantize_kv_latents(k_lat, v_lat, kq8, vq8)
        r4 = saving_ratio(qkv4, kq4, vq4)
        r8 = saving_ratio(qkv8, kq8, vq8)
        assert r4 > r8
