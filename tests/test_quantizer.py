import torch
import pytest
from hawp_laq.utils.packbits import pack_int4, unpack_int4
from hawp_laq.utils.math_utils import orthogonalize, topk_recall, pairwise_hinge_ranking_loss
from hawp_laq.runtime.quantizer import KQuantizer, VQuantizer


class TestPackInt4:
    def test_roundtrip_even(self):
        t = torch.tensor([1, -2, 3, -4, 5, -6, 7, -8], dtype=torch.int8)
        packed = pack_int4(t)
        unpacked = unpack_int4(packed, t.numel())
        assert torch.equal(t, unpacked)

    def test_roundtrip_odd(self):
        t = torch.tensor([1, -1, 3], dtype=torch.int8)
        packed = pack_int4(t)
        unpacked = unpack_int4(packed, t.numel())
        assert torch.equal(t, unpacked)

    def test_clamp_overflow(self):
        t = torch.tensor([15, -20, 0], dtype=torch.int8)
        packed = pack_int4(t)
        unpacked = unpack_int4(packed, t.numel())
        assert unpacked[0].item() == 7
        assert unpacked[1].item() == -8

    def test_packed_dtype_and_size(self):
        t = torch.randint(-8, 8, (100,), dtype=torch.int8)
        packed = pack_int4(t)
        assert packed.dtype == torch.uint8
        assert packed.numel() == 50

    def test_zero_tensor(self):
        t = torch.zeros(10, dtype=torch.int8)
        packed = pack_int4(t)
        unpacked = unpack_int4(packed, t.numel())
        assert torch.equal(t, unpacked)


class TestOrthogonalize:
    def test_output_is_orthogonal(self):
        w = torch.randn(32, 64)
        o = orthogonalize(w)
        product = o.float() @ o.float().T
        assert torch.allclose(product, torch.eye(32), atol=1e-4)

    def test_preserves_dtype(self):
        w = torch.randn(16, 32, dtype=torch.float16)
        o = orthogonalize(w)
        assert o.dtype == torch.float16


class TestTopkRecall:
    def test_perfect_recall(self):
        scores = torch.tensor([0.9, 0.8, 0.1, 0.0])
        targets = torch.tensor([1, 1, 0, 0])
        assert topk_recall(scores, targets, k=2) == 1.0

    def test_zero_recall(self):
        scores = torch.tensor([0.1, 0.0, 0.9, 0.8])
        targets = torch.tensor([1, 1, 0, 0])
        assert topk_recall(scores, targets, k=2) == 0.0

    def test_partial_recall(self):
        scores = torch.tensor([0.9, 0.1, 0.8, 0.0])
        targets = torch.tensor([1, 1, 0, 0])
        assert topk_recall(scores, targets, k=2) == 0.5


class TestPairwiseHingeRankingLoss:
    def test_zero_loss_when_neg_far(self):
        anchor = torch.tensor([[1.0, 0.0]])
        positive = torch.tensor([[1.0, 0.0]])
        negative = torch.tensor([[0.0, 10.0]])
        loss = pairwise_hinge_ranking_loss(anchor, positive, negative)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_nonzero_loss_when_neg_close(self):
        anchor = torch.tensor([[1.0, 0.0]])
        positive = torch.tensor([[0.0, 0.0]])
        negative = torch.tensor([[0.0, 0.0]])
        loss = pairwise_hinge_ranking_loss(anchor, positive, negative)
        assert loss.item() > 0.0


class TestKQuantizer:
    def test_dequantize_shape(self):
        x = torch.randn(10, 128)
        kq = KQuantizer(group_size=128)
        result = kq.quantize(x)
        out = KQuantizer.dequantize(result)
        assert out.shape == x.shape

    def test_int4_range(self):
        x = torch.randn(5, 64)
        kq = KQuantizer(group_size=64)
        result = kq.quantize(x)
        assert result.q.min() >= -8
        assert result.q.max() <= 7
        assert result.q.dtype == torch.int8

    def test_zero_input(self):
        x = torch.zeros(4, 128)
        kq = KQuantizer(group_size=128)
        result = kq.quantize(x)
        out = KQuantizer.dequantize(result)
        assert torch.allclose(out, x, atol=1e-5)

    def test_with_rotation(self):
        torch.manual_seed(0)
        x = torch.randn(8, 64)
        kq = KQuantizer(group_size=64, use_rotation=True)
        result = kq.quantize(x)
        assert result.rotation is not None
        assert result.rotation.shape == (64, 64)
        out = KQuantizer.dequantize(result)
        assert out.shape == x.shape

    def test_scale_positive(self):
        x = torch.randn(4, 128)
        kq = KQuantizer(group_size=128)
        result = kq.quantize(x)
        assert (result.scale > 0).all()

    def test_approximate_reconstruction(self):
        torch.manual_seed(42)
        x = torch.randn(16, 256)
        kq = KQuantizer(group_size=128)
        result = kq.quantize(x)
        out = KQuantizer.dequantize(result)
        rel_err = (out - x).abs().mean() / x.abs().mean()
        assert rel_err < 0.5


class TestVQuantizer:
    def test_dequantize_shape(self):
        x = torch.randn(10, 128)
        vq = VQuantizer(group_size=128)
        result = vq.quantize(x)
        out = VQuantizer.dequantize(result)
        assert out.shape == x.shape

    def test_int8_range(self):
        x = torch.randn(5, 64)
        vq = VQuantizer(group_size=64)
        result = vq.quantize(x)
        assert result.q.min() >= -128
        assert result.q.max() <= 127
        assert result.q.dtype == torch.int8

    def test_zero_input(self):
        x = torch.zeros(4, 128)
        vq = VQuantizer(group_size=128)
        result = vq.quantize(x)
        out = VQuantizer.dequantize(result)
        assert torch.allclose(out, x, atol=1e-4)

    def test_with_outlier_residual(self):
        x = torch.randn(8, 64)
        x[0, 0] = 100.0
        x[1, 5] = -50.0
        vq = VQuantizer(group_size=64, outlier_threshold=5.0)
        result = vq.quantize(x)
        assert result.residual is not None
        assert result.residual[0, 0].item() == pytest.approx(100.0)
        out = VQuantizer.dequantize(result)
        assert out.shape == x.shape

    def test_no_outlier_when_threshold_none(self):
        x = torch.randn(4, 128)
        vq = VQuantizer(group_size=128, outlier_threshold=None)
        result = vq.quantize(x)
        assert result.residual is None

    def test_approximate_reconstruction(self):
        torch.manual_seed(42)
        x = torch.randn(16, 256)
        vq = VQuantizer(group_size=128)
        result = vq.quantize(x)
        out = VQuantizer.dequantize(result)
        rel_err = (out - x).abs().mean() / x.abs().mean()
        assert rel_err < 0.3
