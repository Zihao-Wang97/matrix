from __future__ import annotations

import torch
import pytest

from hawp_laq.runtime.cache_manager import CacheManager
from hawp_laq.runtime.latent_cache import LayerKVCache
from hawp_laq.runtime.turboquant import TurboQuantMSE, TurboQuantProd
from hawp_laq.runtime.scheduler import TokenBudgetScheduler


def _make_cache(n_layers=3, n_heads=4, head_dim=16):
    kq = TurboQuantProd(dim=head_dim, bits=4, use_rotation=True)
    vq = TurboQuantMSE(dim=head_dim, bits=8, use_rotation=True)
    sched = TokenBudgetScheduler(total_budget=999999)
    return CacheManager(
        n_layers=n_layers, n_heads=n_heads, head_dim=head_dim,
        scheduler=sched, k_quantizer=kq, v_quantizer=vq,
    )


class TestLayerKVCache:
    def test_append_recent(self):
        kq = TurboQuantProd(dim=16, bits=4, use_rotation=True)
        vq = TurboQuantMSE(dim=16, bits=8, use_rotation=True)
        cache = LayerKVCache(4, 16, kq, vq)
        cache.append_recent(torch.randn(16), torch.randn(16))
        assert cache.n_recent == 1

    def test_get_all_empty(self):
        kq = TurboQuantProd(dim=16, bits=4, use_rotation=True)
        vq = TurboQuantMSE(dim=16, bits=8, use_rotation=True)
        cache = LayerKVCache(4, 16, kq, vq)
        k = cache.get_all_k()
        v = cache.get_all_v()
        assert k.shape == (0, 16)
        assert v.shape == (0, 16)

    def test_recent_only(self):
        kq = TurboQuantProd(dim=16, bits=4, use_rotation=True)
        vq = TurboQuantMSE(dim=16, bits=8, use_rotation=True)
        cache = LayerKVCache(4, 16, kq, vq)
        for _ in range(5):
            cache.append_recent(torch.randn(16), torch.randn(16))
        k = cache.get_all_k()
        v = cache.get_all_v()
        assert k.shape == (5, 16)
        assert v.shape == (5, 16)

    def test_demote_to_archive(self):
        kq = TurboQuantProd(dim=16, bits=4, use_rotation=True)
        vq = TurboQuantMSE(dim=16, bits=8, use_rotation=True)
        cache = LayerKVCache(4, 16, kq, vq)
        for _ in range(10):
            cache.append_recent(torch.randn(16), torch.randn(16))
        cache.demote_to_archive()
        assert cache.n_recent == 0
        assert cache.n_archive == 10

    def test_archive_dequant_shape(self):
        kq = TurboQuantProd(dim=16, bits=4, use_rotation=True)
        vq = TurboQuantMSE(dim=16, bits=8, use_rotation=True)
        cache = LayerKVCache(4, 16, kq, vq)
        for _ in range(10):
            cache.append_recent(torch.randn(16), torch.randn(16))
        cache.demote_to_archive()
        k = cache.get_all_k()
        v = cache.get_all_v()
        assert k.shape == (10, 16)
        assert v.shape == (10, 16)

    def test_recent_plus_archive(self):
        kq = TurboQuantProd(dim=16, bits=4, use_rotation=True)
        vq = TurboQuantMSE(dim=16, bits=8, use_rotation=True)
        cache = LayerKVCache(4, 16, kq, vq)
        for _ in range(8):
            cache.append_recent(torch.randn(16), torch.randn(16))
        cache.demote_to_archive()
        for _ in range(3):
            cache.append_recent(torch.randn(16), torch.randn(16))
        assert cache.n_recent == 3
        assert cache.n_archive == 8
        k = cache.get_all_k()
        v = cache.get_all_v()
        assert k.shape == (11, 16)
        assert v.shape == (11, 16)

    def test_nbytes_recent(self):
        kq = TurboQuantProd(dim=16, bits=4, use_rotation=True)
        vq = TurboQuantMSE(dim=16, bits=8, use_rotation=True)
        cache = LayerKVCache(4, 16, kq, vq)
        for _ in range(5):
            cache.append_recent(torch.randn(16), torch.randn(16))
        assert cache.nbytes_recent() > 0

    def test_nbytes_archive(self):
        kq = TurboQuantProd(dim=16, bits=4, use_rotation=True)
        vq = TurboQuantMSE(dim=16, bits=8, use_rotation=True)
        cache = LayerKVCache(4, 16, kq, vq)
        for _ in range(5):
            cache.append_recent(torch.randn(16), torch.randn(16))
        cache.demote_to_archive()
        assert cache.nbytes_archive() > 0

    def test_archive_smaller_than_fp16(self):
        kq = TurboQuantProd(dim=16, bits=4, use_rotation=False)
        vq = TurboQuantMSE(dim=16, bits=4, use_rotation=False)
        cache = LayerKVCache(4, 16, kq, vq)
        for _ in range(50):
            cache.append_recent(torch.randn(16), torch.randn(16))
        fp16_bytes = cache.nbytes_recent()
        cache.demote_to_archive()
        archive_bytes = cache.nbytes_archive()
        assert archive_bytes < fp16_bytes * 0.8

    def test_incremental_demote(self):
        kq = TurboQuantProd(dim=16, bits=4, use_rotation=True)
        vq = TurboQuantMSE(dim=16, bits=8, use_rotation=True)
        cache = LayerKVCache(4, 16, kq, vq)
        for _ in range(5):
            cache.append_recent(torch.randn(16), torch.randn(16))
        cache.demote_to_archive()
        for _ in range(5):
            cache.append_recent(torch.randn(16), torch.randn(16))
        cache.demote_to_archive()
        assert cache.n_archive == 10
        assert cache.n_recent == 0
        k = cache.get_all_k()
        assert k.shape == (10, 16)


class TestCacheManager:
    def test_create(self):
        cm = _make_cache()
        assert len(cm) == 3

    def test_append_and_retrieve(self):
        cm = _make_cache(n_layers=2, head_dim=16)
        k_per_layer = [torch.randn(16) for _ in range(2)]
        v_per_layer = [torch.randn(16) for _ in range(2)]
        cm.append_token(k_per_layer, v_per_layer)
        for i in range(2):
            k, v = cm.get_kv_for_attention(i)
            assert k.shape == (1, 16)
            assert v.shape == (1, 16)

    def test_layer_mismatch_raises(self):
        cm = _make_cache(n_layers=2)
        with pytest.raises(ValueError, match="Expected 2"):
            cm.append_token([torch.randn(16)], [torch.randn(16)])

    def test_demote_all(self):
        cm = _make_cache(n_layers=2, head_dim=16)
        for _ in range(10):
            cm.append_token(
                [torch.randn(16) for _ in range(2)],
                [torch.randn(16) for _ in range(2)],
            )
        cm.demote_all()
        summary = cm.summary()
        assert summary["recent_tokens"] == 0
        assert summary["archive_tokens"] == 10

    def test_recent_then_demote_then_append(self):
        cm = _make_cache(n_layers=1, head_dim=16)
        for _ in range(5):
            cm.append_token([torch.randn(16)], [torch.randn(16)])
        cm.demote_all()
        for _ in range(3):
            cm.append_token([torch.randn(16)], [torch.randn(16)])
        k, v = cm.get_kv_for_attention(0)
        assert k.shape == (8, 16)
        assert v.shape == (8, 16)
        summary = cm.summary()
        assert summary["recent_tokens"] == 3
        assert summary["archive_tokens"] == 5

    def test_nbytes_breakdown(self):
        kq = TurboQuantProd(dim=16, bits=4, use_rotation=False)
        vq = TurboQuantMSE(dim=16, bits=4, use_rotation=False)
        cm = CacheManager(
            n_layers=1, n_heads=4, head_dim=16,
            scheduler=TokenBudgetScheduler(total_budget=999999),
            k_quantizer=kq, v_quantizer=vq,
        )
        for _ in range(50):
            cm.append_token([torch.randn(16)], [torch.randn(16)])
        pre_demote_recent = cm.nbytes_recent()
        assert cm.nbytes_archive() == 0
        cm.demote_all()
        assert cm.nbytes_recent() == 0
        assert cm.nbytes_archive() > 0
        assert cm.nbytes_archive() < pre_demote_recent

    def test_summary_keys(self):
        cm = _make_cache(n_layers=1, head_dim=16)
        s = cm.summary()
        assert "recent_tokens" in s
        assert "archive_tokens" in s
        assert "recent_nbytes" in s
        assert "archive_nbytes" in s
        assert "total_nbytes" in s

    def test_from_config(self):
        from hawp_laq.config import HAWPLAQConfig
        cfg = HAWPLAQConfig()
        cfg.quant.enabled = True
        cfg.quant.k_method = "turbo_prod"
        cfg.quant.v_method = "turbo_mse"
        sched = TokenBudgetScheduler(total_budget=999999)
        cm = CacheManager(
            n_layers=2, n_heads=4, head_dim=16,
            scheduler=sched, cfg=cfg,
        )
        assert isinstance(cm._caches[0].k_quantizer, TurboQuantProd)
        assert isinstance(cm._caches[0].v_quantizer, TurboQuantMSE)
