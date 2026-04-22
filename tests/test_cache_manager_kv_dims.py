import pytest

from hawp_laq.runtime.cache_manager import CacheManager
from hawp_laq.config import HAWPLAQConfig


def test_cache_manager_rejects_asymmetric_dims_if_layer_cache_not_supported():
    with pytest.raises(NotImplementedError, match="k_dim == v_dim"):
        CacheManager(
            n_layers=2,
            n_heads=4,
            head_dim=16,
            k_dim=8,
            v_dim=6,
        )


def test_cache_manager_accepts_symmetric_kv_dims():
    mgr = CacheManager(
        n_layers=2,
        n_heads=4,
        head_dim=16,
        k_dim=8,
        v_dim=8,
    )
    assert mgr.k_dim == 8
    assert mgr.v_dim == 8
