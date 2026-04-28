"""Test CacheManager with asymmetric K/V latent dimensions (k_dim != v_dim)."""
from __future__ import annotations

import torch
import pytest

from hawp_laq.runtime.cache_manager import CacheManager
from hawp_laq.runtime.turboquant import TurboQuantProd, TurboQuantMSE
from hawp_laq.runtime.latent_cache import LayerKVCache


def _make_quantizers(k_dim=48, v_dim=32):
    kq = TurboQuantProd(dim=k_dim, bits=4, use_rotation=False, group_size=16)
    vq = TurboQuantMSE(dim=v_dim, bits=8, use_rotation=False, group_size=16)
    return kq, vq


class TestCacheManagerAsymmetricDims:
    """CacheManager with k_dim != v_dim should work end-to-end."""

    def test_init_asymmetric_no_error(self):
        kq, vq = _make_quantizers(48, 32)
        cm = CacheManager(
            n_layers=2, n_heads=4, head_dim=64, dtype=torch.float32,
            k_dim=48, v_dim=32, k_quantizer=kq, v_quantizer=vq,
        )
        assert cm.k_dim == 48
        assert cm.v_dim == 32
        assert cm.k_dim != cm.v_dim

    def test_init_symmetric_no_error(self):
        """Symmetric dims should still work (regression check)."""
        kq, vq = _make_quantizers(48, 48)
        cm = CacheManager(
            n_layers=2, n_heads=4, head_dim=64, dtype=torch.float32,
            k_dim=48, v_dim=48, k_quantizer=kq, v_quantizer=vq,
        )
        assert cm.k_dim == 48
        assert cm.v_dim == 48

    def test_init_defaults_to_head_dim(self):
        """When k_dim/v_dim are not provided, defaults to head_dim."""
        kq = TurboQuantProd(dim=64, bits=4, use_rotation=False, group_size=16)
        vq = TurboQuantMSE(dim=64, bits=8, use_rotation=False, group_size=16)
        cm = CacheManager(
            n_layers=2, n_heads=4, head_dim=64, dtype=torch.float32,
            k_quantizer=kq, v_quantizer=vq,
        )
        assert cm.k_dim == 64
        assert cm.v_dim == 64

    def test_append_and_retrieve_asymmetric(self):
        kq, vq = _make_quantizers(48, 32)
        cm = CacheManager(
            n_layers=2, n_heads=4, head_dim=64, dtype=torch.float32,
            k_dim=48, v_dim=32, k_quantizer=kq, v_quantizer=vq,
        )
        # Append tokens: K shape [1, 48], V shape [1, 32]
        k_layer0 = torch.randn(1, 48)
        v_layer0 = torch.randn(1, 32)
        k_layer1 = torch.randn(1, 48)
        v_layer1 = torch.randn(1, 32)
        cm.append_token([k_layer0, k_layer1], [v_layer0, v_layer1])

        k, v = cm.get_kv_for_attention(0)
        assert k.shape[-1] == 48, f"K last dim should be 48, got {k.shape[-1]}"
        assert v.shape[-1] == 32, f"V last dim should be 32, got {v.shape[-1]}"

    def test_append_and_retrieve_multiple_tokens(self):
        kq, vq = _make_quantizers(48, 32)
        cm = CacheManager(
            n_layers=1, n_heads=4, head_dim=64, dtype=torch.float32,
            k_dim=48, v_dim=32, k_quantizer=kq, v_quantizer=vq,
        )
        for _ in range(5):
            cm.append_token([torch.randn(1, 48)], [torch.randn(1, 32)])

        k, v = cm.get_kv_for_attention(0)
        assert k.shape[-1] == 48
        assert v.shape[-1] == 32
        assert k.shape[0] == 5, f"Expected 5 tokens, got {k.shape[0]}"

    def test_demote_and_retrieve_still_correct(self):
        kq, vq = _make_quantizers(48, 32)
        cm = CacheManager(
            n_layers=1, n_heads=4, head_dim=64, dtype=torch.float32,
            k_dim=48, v_dim=32, k_quantizer=kq, v_quantizer=vq,
            recent_window=3,
        )
        for _ in range(10):
            cm.append_token([torch.randn(1, 48)], [torch.randn(1, 32)])

        # After 10 appends with recent_window=3, archive should have 7 tokens
        assert cm._caches[0].n_archive == 7
        assert cm._caches[0].n_recent == 3

        k, v = cm.get_kv_for_attention(0)
        assert k.shape[-1] == 48
        assert v.shape[-1] == 32
        assert k.shape[0] == 10


class TestLayerKVCacheAsymmetricDims:
    """LayerKVCache should store and retrieve asymmetric K/V."""

    def test_explicit_kdim_vdim(self):
        kq, vq = _make_quantizers(48, 32)
        cache = LayerKVCache(
            n_heads=4, k_quantizer=kq, v_quantizer=vq, dtype=torch.float32,
            k_dim=48, v_dim=32,
        )
        assert cache.k_dim == 48
        assert cache.v_dim == 32

    def test_infer_from_quantizers(self):
        kq, vq = _make_quantizers(48, 32)
        cache = LayerKVCache(
            n_heads=4, k_quantizer=kq, v_quantizer=vq, dtype=torch.float32,
        )
        assert cache.k_dim == 48
        assert cache.v_dim == 32

    def test_fallback_to_head_dim(self):
        kq = TurboQuantProd(dim=72, bits=4, use_rotation=False, group_size=16)
        vq = TurboQuantMSE(dim=72, bits=8, use_rotation=False, group_size=16)
        cache = LayerKVCache(
            n_heads=4, head_dim=72, k_quantizer=kq, v_quantizer=vq, dtype=torch.float32,
        )
        assert cache.k_dim == 72
        assert cache.v_dim == 72

    def test_append_retrieve_asymmetric(self):
        kq, vq = _make_quantizers(48, 32)
        cache = LayerKVCache(
            n_heads=4, k_quantizer=kq, v_quantizer=vq, dtype=torch.float32,
            k_dim=48, v_dim=32,
        )
        cache.append_recent(torch.randn(48), torch.randn(32))
        cache.append_recent(torch.randn(48), torch.randn(32))

        k = cache.get_all_k()
        v = cache.get_all_v()
        assert k.shape[-1] == 48
        assert v.shape[-1] == 32
        assert k.shape[0] == 2

    def test_demote_archive_retrieve_correct(self):
        kq, vq = _make_quantizers(48, 32)
        cache = LayerKVCache(
            n_heads=4, k_quantizer=kq, v_quantizer=vq, dtype=torch.float32,
            k_dim=48, v_dim=32, recent_window=3,
        )
        for _ in range(8):
            cache.append_recent(torch.randn(48), torch.randn(32))

        assert cache.n_archive == 5
        assert cache.n_recent == 3

        k = cache.get_all_k()
        v = cache.get_all_v()
        assert k.shape[-1] == 48
        assert v.shape[-1] == 32
        assert k.shape[0] == 8

    def test_drop_oldest_asymmetric(self):
        kq, vq = _make_quantizers(48, 32)
        cache = LayerKVCache(
            n_heads=4, k_quantizer=kq, v_quantizer=vq, dtype=torch.float32,
            k_dim=48, v_dim=32, recent_window=2,
        )
        for _ in range(8):
            cache.append_recent(torch.randn(48), torch.randn(32))

        # recent_window=2 with 8 appends → auto-demote → archive has 6 tokens
        assert cache.n_archive >= 5
        dropped = cache.drop_oldest(2)
        assert dropped == 2

        k = cache.get_all_k()
        v = cache.get_all_v()
        assert k.shape[-1] == 48
        assert v.shape[-1] == 32

    def test_memory_estimate_per_dim(self):
        kq, vq = _make_quantizers(48, 32)
        cache = LayerKVCache(
            n_heads=4, k_quantizer=kq, v_quantizer=vq, dtype=torch.float32,
            k_dim=48, v_dim=32, recent_window=2,
        )
        for _ in range(10):
            cache.append_recent(torch.randn(48), torch.randn(32))

        # recent_window=2 → archive should have 8 tokens
        assert cache.n_archive > 0, "Archive should be populated after auto-demote"
        nbytes = cache.nbytes_archive()
        assert nbytes > 0, "Archive should consume some bytes"

        k_bytes = sum(
            cache.k_quantizer.estimate_num_bytes(c.k_qx)
            for c in cache._archive_chunks
        )
        v_bytes = sum(
            cache.v_quantizer.estimate_num_bytes(c.v_qx)
            for c in cache._archive_chunks
        )
        assert k_bytes > 0
        assert v_bytes > 0


class TestCacheManagerDimInference:
    """CacheManager resolves k_dim/v_dim from quantizer when not explicitly given."""

    def test_infer_from_quantizer_k_dim_neq_v_dim(self):
        kq = TurboQuantProd(dim=8, bits=4, use_rotation=False, group_size=4)
        vq = TurboQuantMSE(dim=6, bits=8, use_rotation=False, group_size=3)
        cm = CacheManager(
            n_layers=1, n_heads=1, head_dim=16, dtype=torch.float16,
            k_quantizer=kq, v_quantizer=vq,
        )
        assert cm.k_dim == 8
        assert cm.v_dim == 6
        cm.append_token([torch.randn(8)], [torch.randn(6)])
        k, v = cm.get_kv_for_attention(0)
        assert k.shape == (1, 8)
        assert v.shape == (1, 6)

    def test_infer_head_dim_fallback(self):
        cm = CacheManager(n_layers=1, n_heads=1, head_dim=16, dtype=torch.float16)
        assert cm.k_dim == 16
        assert cm.v_dim == 16

    def test_raise_when_kdim_mismatches_quantizer(self):
        kq = TurboQuantProd(dim=8, bits=4, use_rotation=False, group_size=4)
        vq = TurboQuantMSE(dim=6, bits=8, use_rotation=False, group_size=3)
        with pytest.raises(ValueError, match="k_quantizer.dim"):
            CacheManager(
                n_layers=1, n_heads=1, head_dim=16, dtype=torch.float16,
                k_dim=7, v_dim=6, k_quantizer=kq, v_quantizer=vq,
            )

    def test_raise_when_vdim_mismatches_quantizer(self):
        kq = TurboQuantProd(dim=8, bits=4, use_rotation=False, group_size=4)
        vq = TurboQuantMSE(dim=6, bits=8, use_rotation=False, group_size=3)
        with pytest.raises(ValueError, match="v_quantizer.dim"):
            CacheManager(
                n_layers=1, n_heads=1, head_dim=16, dtype=torch.float16,
                k_dim=8, v_dim=7, k_quantizer=kq, v_quantizer=vq,
            )


class TestLayerKVCacheShapeValidation:
    """append_recent should fail immediately on shape mismatches."""

    def _cache(self):
        kq, vq = _make_quantizers(8, 6)
        return LayerKVCache(
            n_heads=2, k_quantizer=kq, v_quantizer=vq, dtype=torch.float32,
            k_dim=8, v_dim=6, recent_window=4,
        )

    def test_wrong_k_dim_raises(self):
        cache = self._cache()
        with pytest.raises(ValueError, match="Expected k dim 8"):
            cache.append_recent(torch.randn(9), torch.randn(6))

    def test_wrong_v_dim_raises(self):
        cache = self._cache()
        with pytest.raises(ValueError, match="Expected v dim 6"):
            cache.append_recent(torch.randn(8), torch.randn(7))

    def test_token_count_mismatch_raises(self):
        cache = self._cache()
        with pytest.raises(ValueError, match="token count mismatch"):
            cache.append_recent(torch.randn(2, 8), torch.randn(1, 6))

    def test_empty_cache_returns_correct_dtype_and_shape(self):
        kq, vq = _make_quantizers(8, 6)
        cache = LayerKVCache(
            n_heads=2, k_quantizer=kq, v_quantizer=vq, dtype=torch.float16,
            k_dim=8, v_dim=6,
        )
        k = cache.get_all_k()
        v = cache.get_all_v()
        assert k.shape == (0, 8), f"empty k shape {k.shape}"
        assert v.shape == (0, 6), f"empty v shape {v.shape}"
        assert k.dtype == torch.float16, f"empty k dtype {k.dtype}"
        assert v.dtype == torch.float16, f"empty v dtype {v.dtype}"

    def test_dtype_none_raises(self):
        kq, vq = _make_quantizers(8, 6)
        with pytest.raises(ValueError, match="dtype"):
            LayerKVCache(
                n_heads=2, k_quantizer=kq, v_quantizer=vq, dtype=None,
                k_dim=8, v_dim=6,
            )
