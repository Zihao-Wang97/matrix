from __future__ import annotations

import torch
import pytest

from hawp_laq.runtime.turboquant import TurboQuantMSE, TurboQuantProd


def test_quantized_tensor_saves_logical_shape():
    nkv, T, r = 2, 3, 4
    tq = TurboQuantMSE(dim=r, bits=8, use_rotation=False, group_size=r)
    x = torch.randn(nkv * T, r)

    qx = tq.quantize(x, logical_shape=(nkv, T, r))

    assert qx.shape_orig == (nkv * T, r)
    assert qx.logical_shape == (nkv, T, r)


def test_prod_result_saves_logical_shape():
    nkv, T, r = 2, 3, 4
    tq = TurboQuantProd(dim=r, bits=8, use_rotation=False, group_size=r)
    x = torch.randn(nkv * T, r)

    qx = tq.quantize(x, logical_shape=(nkv, T, r))

    assert qx.shape_orig == (nkv * T, r)
    assert qx.logical_shape == (nkv, T, r)
    assert qx.mse.logical_shape == (nkv, T, r)


def test_mse_dequantize_logical_shape():
    nkv, T, r = 2, 3, 4
    tq = TurboQuantMSE(dim=r, bits=8, use_rotation=False, group_size=r)
    x = torch.randn(nkv * T, r)

    qx = tq.quantize(x, logical_shape=(nkv, T, r))
    x_hat = tq.dequantize_logical(qx)

    assert x_hat.shape == (nkv, T, r)
    assert torch.allclose(x_hat.reshape(nkv * T, r), tq.dequantize(qx))


def test_prod_dequantize_logical_shape():
    nkv, T, r = 2, 3, 4
    tq = TurboQuantProd(dim=r, bits=8, use_rotation=False, group_size=r)
    x = torch.randn(nkv * T, r)

    qx = tq.quantize(x, logical_shape=(nkv, T, r))
    x_hat = tq.dequantize_logical(qx)
    mse_hat = tq.dequantize_mse(qx, logical=True)

    assert x_hat.shape == (nkv, T, r)
    assert mse_hat.shape == (nkv, T, r)
    assert torch.allclose(x_hat.reshape(nkv * T, r), tq.dequantize(qx))
    assert torch.allclose(mse_hat.reshape(nkv * T, r), tq.dequantize_mse(qx))


def test_mse_quantize_rejects_invalid_logical_shape():
    tq = TurboQuantMSE(dim=4, bits=8, use_rotation=False, group_size=4)
    x = torch.randn(6, 4)

    with pytest.raises(ValueError, match="last dim"):
        tq.quantize(x, logical_shape=(2, 3, 5))
    with pytest.raises(ValueError, match="row product"):
        tq.quantize(x, logical_shape=(2, 4, 4))


def test_prod_quantize_rejects_invalid_logical_shape():
    tq = TurboQuantProd(dim=4, bits=8, use_rotation=False, group_size=4)
    x = torch.randn(6, 4)

    with pytest.raises(ValueError, match="last dim"):
        tq.quantize(x, logical_shape=(2, 3, 5))
    with pytest.raises(ValueError, match="row product"):
        tq.quantize(x, logical_shape=(2, 4, 4))


def test_prod_dequantize_mse_falls_back_to_mse_logical_shape():
    nkv, T, r = 2, 3, 4
    tq = TurboQuantProd(dim=r, bits=8, use_rotation=False, group_size=r)
    x = torch.randn(nkv * T, r)

    qx = tq.quantize(x, logical_shape=(nkv, T, r))
    qx.logical_shape = None

    assert qx.mse.logical_shape == (nkv, T, r)
    assert tq.dequantize_mse(qx, logical=True).shape == (nkv, T, r)
    assert tq.dequantize_logical(qx).shape == (nkv, T, r)
