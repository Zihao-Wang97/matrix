import torch
import pytest
from hawp_laq.runtime.cache_manager import CacheManager
from hawp_laq.runtime.latent_cache import LayerKVCache
from hawp_laq.runtime.quantizer import KQuantizer, VQuantizer
from hawp_laq.runtime.scheduler import TokenBudgetScheduler


def _make_cache_manager(n_layers=2, total_budget=256, recent_window=16):
    sched = TokenBudgetScheduler(total_budget=total_budget, recent_window=recent_window)
    return CacheManager(
        n_layers=n_layers,
        n_heads=4,
        head_dim=32,
        scheduler=sched,
        k_group_size=128,
        v_group_size=128,
    )


class TestLayerKVCache:
    def _make_cache(self):
        kq = KQuantizer(group_size=128)
        vq = VQuantizer(group_size=128)
        return LayerKVCache(n_heads=4, head_dim=32, k_quantizer=kq, v_quantizer=vq)

    def test_append_and_count(self):
        cache = self._make_cache()
        k = torch.randn(1, 128)
        v = torch.randn(1, 128)
        cache.append_high(k, v)
        cache.append_high(k, v)
        assert cache.n_high == 2
        assert cache.total_tokens == 2

    def test_demote_to_low(self):
        cache = self._make_cache()
        for _ in range(5):
            cache.append_high(torch.randn(1, 128), torch.randn(1, 128))
        assert cache.n_high == 5
        cache.demote_to_low()
        assert cache.n_high == 0
        assert cache.n_low == 5

    def test_get_all_k(self):
        cache = self._make_cache()
        k1 = torch.randn(2, 128)
        k2 = torch.randn(3, 128)
        cache.append_high(k1, torch.randn(2, 128))
        cache.append_high(k2, torch.randn(3, 128))
        result = cache.get_all_k()
        assert result.shape == (5, 128)

    def test_get_all_after_demote(self):
        cache = self._make_cache()
        cache.append_high(torch.randn(3, 128), torch.randn(3, 128))
        cache.demote_to_low()
        cache.append_high(torch.randn(2, 128), torch.randn(2, 128))
        k = cache.get_all_k()
        v = cache.get_all_v()
        assert k.shape[0] == 5
        assert v.shape[0] == 5

    def test_drop_token(self):
        cache = self._make_cache()
        for i in range(5):
            cache.append_high(torch.full((1, 128), float(i)), torch.randn(1, 128))
        cache.drop_token([1, 3])
        k = cache.get_all_k()
        assert k.shape[0] == 3
        assert k[0, 0].item() == pytest.approx(0.0)
        assert k[1, 0].item() == pytest.approx(2.0)
        assert k[2, 0].item() == pytest.approx(4.0)

    def test_nbytes(self):
        cache = self._make_cache()
        cache.append_high(torch.randn(4, 128), torch.randn(4, 128))
        hb = cache.nbytes_high()
        assert hb > 0
        cache.demote_to_low()
        lb = cache.nbytes_low()
        assert lb > 0
        assert lb < hb

    def test_empty_get_all(self):
        cache = self._make_cache()
        k = cache.get_all_k()
        v = cache.get_all_v()
        assert k.shape == (0, 128)
        assert v.shape == (0, 128)


class TestCacheManager:
    def test_append_and_len(self):
        mgr = _make_cache_manager(n_layers=3)
        assert len(mgr) == 3
        k_list = [torch.randn(1, 128) for _ in range(3)]
        v_list = [torch.randn(1, 128) for _ in range(3)]
        mgr.append_token(k_list, v_list)
        assert mgr.scheduler.seq_len == 1

    def test_get_kv(self):
        mgr = _make_cache_manager(n_layers=2)
        k_list = [torch.randn(2, 128), torch.randn(2, 128)]
        v_list = [torch.randn(2, 128), torch.randn(2, 128)]
        mgr.append_token(k_list, v_list)
        k, v = mgr.get_kv_for_attention(0)
        assert k.shape[0] == 2
        assert v.shape[0] == 2

    def test_demote_all(self):
        mgr = _make_cache_manager()
        for _ in range(5):
            mgr.append_token([torch.randn(1, 128)] * 2, [torch.randn(1, 128)] * 2)
        mgr.demote_all()
        for i in range(2):
            assert mgr[i].n_high == 0
            assert mgr[i].n_low == 5

    def test_total_nbytes(self):
        mgr = _make_cache_manager()
        for _ in range(3):
            mgr.append_token([torch.randn(1, 128)] * 2, [torch.randn(1, 128)] * 2)
        nb = mgr.total_nbytes()
        assert nb > 0
        assert isinstance(mgr.total_nbytes_formatted(), str)

    def test_apply_scheduler(self):
        mgr = _make_cache_manager(n_layers=2, total_budget=16, recent_window=4)
        for _ in range(32):
            mgr.append_token([torch.randn(1, 128)] * 2, [torch.randn(1, 128)] * 2)
        drops = mgr.apply_scheduler()
        assert len(drops) > 0

    def test_summary(self):
        mgr = _make_cache_manager()
        mgr.append_token([torch.randn(1, 128)] * 2, [torch.randn(1, 128)] * 2)
        s = mgr.summary()
        assert "seq_len" in s
        assert "total_nbytes" in s
        assert s["seq_len"] == 1

    def test_layer_mismatch_raises(self):
        mgr = _make_cache_manager(n_layers=2)
        with pytest.raises(ValueError):
            mgr.append_token([torch.randn(1, 128)], [torch.randn(1, 128)])

    def test_getitem(self):
        mgr = _make_cache_manager(n_layers=4)
        assert isinstance(mgr[0], LayerKVCache)
        assert isinstance(mgr[3], LayerKVCache)
