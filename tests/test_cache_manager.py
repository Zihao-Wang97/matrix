import torch
import pytest
from hawp_laq.runtime.cache_manager import CacheManager
from hawp_laq.runtime.latent_cache import LayerKVCache
from hawp_laq.runtime.turboquant import TurboQuantMSE, TurboQuantProd
from hawp_laq.runtime.scheduler import TokenBudgetScheduler


def _make_cache(n_heads=4, head_dim=32):
    kq = TurboQuantProd(dim=head_dim, bits=4, use_rotation=True, group_size=128)
    vq = TurboQuantMSE(dim=head_dim, bits=8, use_rotation=True, group_size=128)
    return LayerKVCache(n_heads=n_heads, head_dim=head_dim, k_quantizer=kq, v_quantizer=vq)


def _make_cache_manager(n_layers=2, total_budget=256, recent_window=16):
    sched = TokenBudgetScheduler(total_budget=total_budget, recent_window=recent_window)
    return CacheManager(
        n_layers=n_layers,
        n_heads=4,
        head_dim=32,
        scheduler=sched,
    )


class TestLayerKVCache:
    def test_append_recent_and_count(self):
        cache = _make_cache()
        k = torch.randn(4, 32)
        v = torch.randn(4, 32)
        cache.append_recent(k, v)
        cache.append_recent(k, v)
        assert cache.n_recent == 8
        assert cache.total_tokens == 8

    def test_demote_to_archive(self):
        cache = _make_cache()
        for _ in range(5):
            cache.append_recent(torch.randn(1, 32), torch.randn(1, 32))
        assert cache.n_recent == 5
        cache.demote_to_archive()
        assert cache.n_recent == 0
        assert cache.n_archive == 5

    def test_get_all_k(self):
        cache = _make_cache()
        k1 = torch.randn(2, 32)
        k2 = torch.randn(3, 32)
        v1 = torch.randn(2, 32)
        v2 = torch.randn(3, 32)
        cache.append_recent(k1, v1)
        cache.append_recent(k2, v2)
        result = cache.get_all_k()
        assert result.shape == (5, 32)

    def test_get_all_after_demote(self):
        cache = _make_cache()
        cache.append_recent(torch.randn(3, 32), torch.randn(3, 32))
        cache.demote_to_archive()
        cache.append_recent(torch.randn(2, 32), torch.randn(2, 32))
        k = cache.get_all_k()
        v = cache.get_all_v()
        assert k.shape[0] == 5
        assert v.shape[0] == 5

    def test_drop_oldest(self):
        cache = _make_cache()
        for _ in range(5):
            cache.append_recent(torch.randn(1, 32), torch.randn(1, 32))
        cache.demote_to_archive()
        dropped = cache.drop_oldest(2)
        assert dropped == 2
        assert cache.n_archive == 3

    def test_nbytes(self):
        cache = _make_cache()
        cache.append_recent(torch.randn(4, 32), torch.randn(4, 32))
        nb_recent = cache.nbytes_recent()
        assert nb_recent > 0
        cache.demote_to_archive()
        nb_archive = cache.nbytes_archive()
        assert nb_archive > 0

    def test_empty_get_all(self):
        cache = _make_cache()
        k = cache.get_all_k()
        v = cache.get_all_v()
        assert k.shape[0] == 0
        assert v.shape[0] == 0


class TestCacheManager:
    def test_append_and_len(self):
        mgr = _make_cache_manager(n_layers=3)
        assert len(mgr) == 3
        k_list = [torch.randn(1, 32) for _ in range(3)]
        v_list = [torch.randn(1, 32) for _ in range(3)]
        mgr.append_token(k_list, v_list)
        assert mgr.scheduler.seq_len == 1

    def test_get_kv(self):
        mgr = _make_cache_manager(n_layers=2)
        k_list = [torch.randn(2, 32), torch.randn(2, 32)]
        v_list = [torch.randn(2, 32), torch.randn(2, 32)]
        mgr.append_token(k_list, v_list)
        k, v = mgr.get_kv_for_attention(0)
        assert k.shape[0] == 2
        assert v.shape[0] == 2

    def test_demote_all(self):
        mgr = _make_cache_manager()
        for _ in range(5):
            mgr.append_token([torch.randn(1, 32)] * 2, [torch.randn(1, 32)] * 2)
        mgr.demote_all()
        for i in range(2):
            assert mgr[i].n_recent == 0
            assert mgr[i].n_archive == 5

    def test_total_nbytes(self):
        mgr = _make_cache_manager()
        for _ in range(3):
            mgr.append_token([torch.randn(1, 32)] * 2, [torch.randn(1, 32)] * 2)
        nb = mgr.total_nbytes()
        assert nb > 0
        assert isinstance(mgr.total_nbytes_formatted(), str)

    def test_apply_scheduler(self):
        mgr = _make_cache_manager(n_layers=2, total_budget=16, recent_window=4)
        for _ in range(32):
            mgr.append_token([torch.randn(1, 32)] * 2, [torch.randn(1, 32)] * 2)
        drop_count = mgr.apply_scheduler()
        assert drop_count > 0

    def test_summary(self):
        mgr = _make_cache_manager()
        mgr.append_token([torch.randn(1, 32)] * 2, [torch.randn(1, 32)] * 2)
        s = mgr.summary()
        assert "seq_len" in s
        assert "total_nbytes" in s
        assert s["seq_len"] == 1

    def test_layer_mismatch_raises(self):
        mgr = _make_cache_manager(n_layers=2)
        with pytest.raises(ValueError):
            mgr.append_token([torch.randn(1, 32)], [torch.randn(1, 32)])

    def test_getitem(self):
        mgr = _make_cache_manager(n_layers=4)
        assert isinstance(mgr[0], LayerKVCache)
        assert isinstance(mgr[3], LayerKVCache)
