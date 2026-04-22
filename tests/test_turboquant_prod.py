from __future__ import annotations

import torch
import pytest

from hawp_laq.runtime.turboquant import TurboQuantMSE, TurboQuantProd, TurboQuantProdResult


class TestTurboQuantProdConstruction:
    def test_default(self):
        tq = TurboQuantProd(dim=64)
        assert tq.dim == 64
        assert tq.bits == 4
        assert tq.use_rotation is True
        assert tq._mse_quantizer is not None
        assert tq._rotation is not None

    def test_no_rotation(self):
        tq = TurboQuantProd(dim=32, use_rotation=False)
        assert tq._rotation is None

    def test_invalid_dim(self):
        with pytest.raises(ValueError, match="dim"):
            TurboQuantProd(dim=0)


class TestTurboQuantProdQuantize:
    def test_2d_output_structure(self):
        tq = TurboQuantProd(dim=32, bits=4, use_rotation=False)
        x = torch.randn(20, 32)
        qx = tq.quantize(x)
        assert isinstance(qx, TurboQuantProdResult)
        assert qx.residual_sign.shape == (20, 32)
        assert qx.residual_norm.shape == (20,)
        assert qx.shape_orig == (20, 32)

    def test_3d_output_structure(self):
        tq = TurboQuantProd(dim=32, bits=4, use_rotation=False)
        x = torch.randn(2, 10, 32)
        qx = tq.quantize(x)
        assert qx.shape_orig == (2, 10, 32)
        assert qx.residual_sign.shape == (20, 32)

    def test_dim_mismatch_raises(self):
        tq = TurboQuantProd(dim=32, bits=4)
        with pytest.raises(ValueError, match="last dim"):
            tq.quantize(torch.randn(10, 16))

    def test_residual_sign_is_bool(self):
        tq = TurboQuantProd(dim=32, bits=4, use_rotation=False)
        x = torch.randn(10, 32)
        qx = tq.quantize(x)
        assert qx.residual_sign.dtype == torch.bool

    def test_residual_norm_nonnegative(self):
        tq = TurboQuantProd(dim=32, bits=4, use_rotation=False)
        x = torch.randn(10, 32)
        qx = tq.quantize(x)
        assert (qx.residual_norm >= 0).all()


class TestTurboQuantProdDequantize:
    def test_roundtrip_shape_2d(self):
        tq = TurboQuantProd(dim=32, bits=4, use_rotation=True)
        x = torch.randn(20, 32)
        qx = tq.quantize(x)
        x_hat = tq.dequantize(qx)
        assert x_hat.shape == x.shape

    def test_roundtrip_shape_3d(self):
        tq = TurboQuantProd(dim=32, bits=4, use_rotation=True)
        x = torch.randn(2, 10, 32)
        qx = tq.quantize(x)
        x_hat = tq.dequantize(qx)
        assert x_hat.shape == x.shape

    def test_prod_better_than_pure_mse(self):
        """TurboQuantProd (MSE + residual) should reconstruct better than pure MSE."""
        torch.manual_seed(0)
        x = torch.randn(100, 32)
        tq_mse = TurboQuantMSE(dim=32, bits=4, use_rotation=True)
        tq_prod = TurboQuantProd(dim=32, bits=4, use_rotation=True)
        tq_prod._mse_quantizer._rotation = tq_mse._rotation.clone()

        mse_only = tq_mse.dequantize(tq_mse.quantize(x))
        prod_result = tq_prod.quantize(x)
        prod_hat = tq_prod.dequantize(prod_result)

        mse_err = (x - mse_only).pow(2).mean().item()
        prod_err = (x - prod_hat).pow(2).mean().item()
        assert prod_err <= mse_err * 1.01, (
            f"Prod MSE ({prod_err}) should be <= MSE-only ({mse_err})"
        )


class TestTurboQuantProdInnerProduct:
    def test_approx_ip_runs_1d_query(self):
        tq = TurboQuantProd(dim=32, bits=4, use_rotation=True)
        x = torch.randn(20, 32)
        qx = tq.quantize(x)
        q = torch.randn(32)
        ip = tq.approx_inner_product(q, qx)
        assert ip.shape == (20,)

    def test_approx_ip_runs_2d_query(self):
        tq = TurboQuantProd(dim=32, bits=4, use_rotation=True)
        x = torch.randn(20, 32)
        qx = tq.quantize(x)
        q = torch.randn(5, 32)
        ip = tq.approx_inner_product(q, qx)
        assert ip.shape == (5, 20)

    def test_approx_ip_close_to_full_dequant(self):
        """approx_inner_product should approximate q @ dequant(x).T."""
        torch.manual_seed(42)
        tq = TurboQuantProd(dim=32, bits=4, use_rotation=True)
        x = torch.randn(30, 32)
        qx = tq.quantize(x)
        q = torch.randn(5, 32)

        ip_approx = tq.approx_inner_product(q, qx)
        x_hat = tq.dequantize(qx)
        ip_full = q @ x_hat.T
        max_diff = (ip_approx - ip_full).abs().max().item()
        assert max_diff < 1e-4, f"approx_ip vs full dequant diff = {max_diff}"

    def test_ip_prod_better_than_mse_only(self):
        """Inner product via TurboQuantProd should be more accurate than pure MSE."""
        torch.manual_seed(7)
        dim = 32
        x = torch.randn(50, dim)
        q = torch.randn(10, dim)

        # Compute ground truth
        gt = q @ x.T

        # MSE-only path
        tq_mse = TurboQuantMSE(dim=dim, bits=4, use_rotation=True)
        x_mse = tq_mse.dequantize(tq_mse.quantize(x))
        ip_mse = q @ x_mse.T
        mse_ip_err = (gt - ip_mse).pow(2).mean().item()

        # Prod path (same rotation)
        tq_prod = TurboQuantProd(dim=dim, bits=4, use_rotation=True)
        tq_prod._mse_quantizer._rotation = tq_mse._rotation.clone()
        qx_prod = tq_prod.quantize(x)
        ip_prod = tq_prod.approx_inner_product(q, qx_prod)
        prod_ip_err = (gt - ip_prod).pow(2).mean().item()

        assert prod_ip_err <= mse_ip_err * 1.01, (
            f"Prod IP error ({prod_ip_err}) should be <= MSE IP error ({mse_ip_err})"
        )


class TestTurboQuantProdBitsTrend:
    def test_higher_bits_lower_reconstruction_error(self):
        torch.manual_seed(3)
        x = torch.randn(50, 32)
        errors = {}
        for bits in (2, 4, 8):
            tq = TurboQuantProd(dim=32, bits=bits, use_rotation=True)
            qx = tq.quantize(x)
            x_hat = tq.dequantize(qx)
            errors[bits] = (x - x_hat).pow(2).mean().item()
        assert errors[8] < errors[4] < errors[2], (
            f"Error should decrease with bits: {errors}"
        )

    def test_higher_bits_lower_ip_error(self):
        torch.manual_seed(5)
        x = torch.randn(50, 32)
        q = torch.randn(5, 32)
        gt = q @ x.T
        errors = {}
        for bits in (2, 4, 8):
            tq = TurboQuantProd(dim=32, bits=bits, use_rotation=True)
            qx = tq.quantize(x)
            ip = tq.approx_inner_product(q, qx)
            errors[bits] = (gt - ip).pow(2).mean().item()
        assert errors[8] < errors[4] < errors[2], (
            f"IP error should decrease with bits: {errors}"
        )


class TestTurboQuantProdBytes:
    def test_larger_than_mse_only(self):
        x = torch.randn(50, 32)
        tq_mse = TurboQuantMSE(dim=32, bits=4, use_rotation=True)
        tq_prod = TurboQuantProd(dim=32, bits=4, use_rotation=True)
        mse_qx = tq_mse.quantize(x)
        prod_qx = tq_prod.quantize(x)
        mse_bytes = tq_mse.estimate_num_bytes(mse_qx)
        prod_bytes = tq_prod.estimate_num_bytes(prod_qx)
        assert prod_bytes > mse_bytes

    def test_positive(self):
        tq = TurboQuantProd(dim=32, bits=4, use_rotation=False)
        x = torch.randn(10, 32)
        qx = tq.quantize(x)
        assert tq.estimate_num_bytes(qx) > 0


class TestTurboQuantProdRotation:
    def test_build_rotation(self):
        tq = TurboQuantProd(dim=16, bits=4, use_rotation=True)
        R1 = tq._rotation.clone()
        tq.build_rotation()
        R2 = tq._rotation
        assert not torch.allclose(R1, R2)

    def test_rotation_is_orthogonal(self):
        tq = TurboQuantProd(dim=32, bits=4, use_rotation=True)
        R = tq._rotation.float()
        assert torch.allclose(R @ R.T, torch.eye(32), atol=1e-5)
