import torch

from hawp_laq.runtime.cache_manager import CacheManager
from hawp_laq.runtime.turboquant import TurboQuantProd, TurboQuantMSE


def test_cache_manager_accepts_asymmetric_dims():
    """Asymmetric k_dim != v_dim should now be supported."""
    kq = TurboQuantProd(dim=8, bits=4, use_rotation=False, group_size=8)
    vq = TurboQuantMSE(dim=6, bits=8, use_rotation=False, group_size=6)
    mgr = CacheManager(
        n_layers=2, n_heads=4, head_dim=16, dtype=torch.float32,
        k_dim=8, v_dim=6, k_quantizer=kq, v_quantizer=vq,
    )
    assert mgr.k_dim == 8
    assert mgr.v_dim == 6
    assert mgr.k_dim != mgr.v_dim


def test_cache_manager_accepts_symmetric_kv_dims():
    """Symmetric k_dim == v_dim still works."""
    kq = TurboQuantProd(dim=8, bits=4, use_rotation=False, group_size=8)
    vq = TurboQuantMSE(dim=8, bits=8, use_rotation=False, group_size=8)
    mgr = CacheManager(
        n_layers=2, n_heads=4, head_dim=16, dtype=torch.float32,
        k_dim=8, v_dim=8, k_quantizer=kq, v_quantizer=vq,
    )
    assert mgr.k_dim == 8
    assert mgr.v_dim == 8
    assert mgr.k_dim == mgr.v_dim
