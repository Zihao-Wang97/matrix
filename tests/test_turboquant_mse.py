from __future__ import annotations

import torch
import pytest

from hawp_laq.runtime.turboquant import TurboQuantMSE, TurboQuantizedTensor, TurboQuantState


class TestTurboQuantMSEConstruction:
    def test_default_construction(self):
        tq = TurboQuantMSE(dim=64)
        assert tq.dim == 64
        assert tq.bits == 4
        assert tq.use_rotation is True
        assert tq.group_size == 128
        assert tq._rotation is not None
        assert tq._rotation.shape == (64, 64)

    def test_no_rotation(self):
        tq = TurboQuantMSE(dim=64, use_rotation=False)
        assert tq._rotation is None

    def test_custom_group_size(self):
        tq = TurboQuantMSE(dim=64, group_size=32)
        assert tq.group_size == 32

    def test_invalid_bits(self):
        with pytest.raises(ValueError, match="bits"):
            TurboQuantMSE(dim=64, bits=5)

    def test_invalid_dim(self):
        with pytest.raises(ValueError, match="dim"):
            TurboQuantMSE(dim=0)

    def test_invalid_group_size(self):
        with pytest.raises(ValueError, match="group_size"):
            TurboQuantMSE(dim=64, group_size=-1)


class TestTurboQuantShape:
    def test_quantize_2d_q_shape(self):
        tq = TurboQuantMSE(dim=32, bits=4, use_rotation=False)
        x = torch.randn(10, 32)
        qx = tq.quantize(x)
        assert qx.q.shape == (10, 16)
        assert qx.shape_orig == (10, 32)

    def test_quantize_3d_q_shape(self):
        tq = TurboQuantMSE(dim=32, bits=4, use_rotation=False)
        x = torch.randn(2, 10, 32)
        qx = tq.quantize(x)
        assert qx.q.shape == (20, 16)
        assert qx.shape_orig == (2, 10, 32)

    def test_dequantize_2d_shape(self):
        tq = TurboQuantMSE(dim=32, bits=4, use_rotation=False)
        x = torch.randn(10, 32)
        qx = tq.quantize(x)
        x_hat = tq.dequantize(qx)
        assert x_hat.shape == x.shape

    def test_dequantize_3d_shape(self):
        tq = TurboQuantMSE(dim=32, bits=4, use_rotation=False)
        x = torch.randn(2, 10, 32)
        qx = tq.quantize(x)
        x_hat = tq.dequantize(qx)
        assert x_hat.shape == x.shape

    def test_scale_n_groups(self):
        tq = TurboQuantMSE(dim=64, bits=4, use_rotation=False, group_size=32)
        x = torch.randn(5, 64)
        qx = tq.quantize(x)
        assert qx.scale.shape == (5, 2)

    def test_invalid_input_dim(self):
        tq = TurboQuantMSE(dim=32, bits=4)
        with pytest.raises(ValueError, match="last dim"):
            tq.quantize(torch.randn(10, 16))

    def test_invalid_input_ndim(self):
        tq = TurboQuantMSE(dim=32, bits=4)
        with pytest.raises(ValueError, match="2-D"):
            tq.quantize(torch.randn(10))


class TestTurboQuantRoundtrip:
    def test_roundtrip_with_rotation(self):
        tq = TurboQuantMSE(dim=32, bits=4, use_rotation=True)
        x = torch.randn(10, 32)
        qx = tq.quantize(x)
        x_hat = tq.dequantize(qx)
        mse = (x - x_hat).pow(2).mean().item()
        assert mse < 1.0, f"MSE too large: {mse}"

    def test_roundtrip_without_rotation(self):
        tq = TurboQuantMSE(dim=32, bits=4, use_rotation=False)
        x = torch.randn(10, 32)
        qx = tq.quantize(x)
        x_hat = tq.dequantize(qx)
        mse = (x - x_hat).pow(2).mean().item()
        assert mse < 1.0, f"MSE too large: {mse}"

    def test_roundtrip_8bit_better_than_4bit(self):
        x = torch.randn(50, 32)
        tq4 = TurboQuantMSE(dim=32, bits=4, use_rotation=True)
        tq8 = TurboQuantMSE(dim=32, bits=8, use_rotation=True)
        mse4 = (x - tq4.dequantize(tq4.quantize(x))).pow(2).mean().item()
        mse8 = (x - tq8.dequantize(tq8.quantize(x))).pow(2).mean().item()
        assert mse8 < mse4, f"8-bit MSE ({mse8}) should be < 4-bit MSE ({mse4})"

    def test_roundtrip_3d(self):
        tq = TurboQuantMSE(dim=32, bits=4, use_rotation=True)
        x = torch.randn(2, 10, 32)
        qx = tq.quantize(x)
        x_hat = tq.dequantize(qx)
        assert x_hat.shape == x.shape
        mse = (x - x_hat).pow(2).mean().item()
        assert mse < 1.0

    def test_rotation_improves_quality(self):
        torch.manual_seed(42)
        # Non-uniform input: first dims dominate — rotation spreads energy
        x = torch.randn(100, 64)
        x[:, :8] *= 10.0
        tq_no_rot = TurboQuantMSE(dim=64, bits=4, use_rotation=False)
        tq_rot = TurboQuantMSE(dim=64, bits=4, use_rotation=True)
        mse_no_rot = (x - tq_no_rot.dequantize(tq_no_rot.quantize(x))).pow(2).mean().item()
        mse_rot = (x - tq_rot.dequantize(tq_rot.quantize(x))).pow(2).mean().item()
        assert mse_rot < mse_no_rot, (
            f"Rotation should help on skewed input: rot MSE={mse_rot} vs no-rot MSE={mse_no_rot}"
        )

    def test_dim_not_multiple_of_group_size(self):
        tq = TurboQuantMSE(dim=48, bits=4, use_rotation=False, group_size=32)
        x = torch.randn(10, 48)
        qx = tq.quantize(x)
        x_hat = tq.dequantize(qx)
        assert x_hat.shape == x.shape
        mse = (x - x_hat).pow(2).mean().item()
        assert mse < 1.0

    def test_dim_smaller_than_group_size(self):
        tq = TurboQuantMSE(dim=16, bits=4, use_rotation=False, group_size=128)
        x = torch.randn(10, 16)
        qx = tq.quantize(x)
        x_hat = tq.dequantize(qx)
        assert x_hat.shape == x.shape
        mse = (x - x_hat).pow(2).mean().item()
        assert mse < 1.0

    def test_2bit_roundtrip(self):
        tq = TurboQuantMSE(dim=32, bits=2, use_rotation=True)
        x = torch.randn(10, 32)
        qx = tq.quantize(x)
        x_hat = tq.dequantize(qx)
        mse = (x - x_hat).pow(2).mean().item()
        assert mse < 5.0

    def test_3bit_roundtrip(self):
        tq = TurboQuantMSE(dim=32, bits=3, use_rotation=True)
        x = torch.randn(10, 32)
        qx = tq.quantize(x)
        x_hat = tq.dequantize(qx)
        mse = (x - x_hat).pow(2).mean().item()
        assert mse < 2.0


class TestTurboQuantBytes:
    def test_estimate_positive(self):
        tq = TurboQuantMSE(dim=64, bits=4, use_rotation=False)
        x = torch.randn(100, 64)
        qx = tq.quantize(x)
        nbytes = tq.estimate_num_bytes(qx)
        assert nbytes > 0

    def test_4bit_smaller_than_fp32(self):
        tq = TurboQuantMSE(dim=64, bits=4, use_rotation=False)
        x = torch.randn(100, 64)
        qx = tq.quantize(x)
        nbytes = tq.estimate_num_bytes(qx)
        fp32_bytes = 100 * 64 * 4
        assert nbytes < fp32_bytes

    def test_8bit_larger_than_4bit(self):
        x = torch.randn(50, 32)
        tq4 = TurboQuantMSE(dim=32, bits=4, use_rotation=False)
        tq8 = TurboQuantMSE(dim=32, bits=8, use_rotation=False)
        qx4 = tq4.quantize(x)
        qx8 = tq8.quantize(x)
        assert tq8.estimate_num_bytes(qx8) > tq4.estimate_num_bytes(qx4)

    def test_rotation_adds_bytes(self):
        x = torch.randn(10, 32)
        tq_no_rot = TurboQuantMSE(dim=32, bits=4, use_rotation=False)
        tq_rot = TurboQuantMSE(dim=32, bits=4, use_rotation=True)
        qx_no_rot = tq_no_rot.quantize(x)
        qx_rot = tq_rot.quantize(x)
        assert tq_rot.estimate_num_bytes(qx_rot) > tq_no_rot.estimate_num_bytes(qx_no_rot)


class TestTurboQuantState:
    def test_get_state(self):
        tq = TurboQuantMSE(dim=32, bits=4, use_rotation=True)
        state = tq.get_state()
        assert state.bits == 4
        assert state.dim == 32
        assert state.use_rotation is True
        assert state.rotation is not None

    def test_from_state_roundtrip(self):
        tq = TurboQuantMSE(dim=32, bits=4, use_rotation=True)
        state = tq.get_state()
        tq2 = TurboQuantMSE.from_state(state)
        assert tq2.dim == tq.dim
        assert tq2.bits == tq.bits
        assert tq2.use_rotation == tq.use_rotation
        assert torch.allclose(tq._rotation.float(), tq2._rotation.float())

    def test_state_no_rotation(self):
        tq = TurboQuantMSE(dim=32, bits=8, use_rotation=False)
        state = tq.get_state()
        assert state.rotation is None
        tq2 = TurboQuantMSE.from_state(state)
        assert tq2._rotation is None


class TestRotationOrthogonal:
    def test_rotation_is_orthogonal(self):
        tq = TurboQuantMSE(dim=32, bits=4, use_rotation=True)
        R = tq._rotation.float()
        I = R @ R.T
        assert torch.allclose(I, torch.eye(32), atol=1e-5)

    def test_rebuild_rotation(self):
        tq = TurboQuantMSE(dim=16, bits=4, use_rotation=True)
        R1 = tq._rotation.clone()
        tq.build_rotation()
        R2 = tq._rotation
        assert not torch.allclose(R1, R2)
