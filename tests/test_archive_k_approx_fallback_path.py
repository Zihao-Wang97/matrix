from __future__ import annotations

import torch

from hawp_laq.modeling.attention_hawp import HAWPAttention
from hawp_laq.runtime.turboquant import TurboQuantProd, TurboQuantMSE


def _make_config():
    from types import SimpleNamespace
    return SimpleNamespace(
        hidden_size=256,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=512,
        rope_theta=10000.0,
        model_type="llama",
        enable_bias=False,
        attention_dropout=0.0,
    )


class TestArchiveKApproxFallbackPath:
    def test_switch_off_uses_dequant_path(self):
        attn = HAWPAttention(
            _make_config(), layer_idx=0, r_k=16, r_v=16,
            use_archive_k_ip_approx=False,
        )
        k_quantizer = TurboQuantProd(dim=16, bits=4, use_rotation=False)
        v_quantizer = TurboQuantProd(dim=16, bits=4, use_rotation=False)
        attn.setup_quant_cache(k_quantizer, v_quantizer, recent_window=2)

        n_kv = 4
        k_lat = torch.randn(1, n_kv, 6, 16)
        v_lat = torch.randn(1, n_kv, 6, 16)
        attn._quant_cache_append(k_lat, v_lat)
        attn._quant_cache_demote()

        assert not attn._can_use_archive_k_ip_approx()

        q_lat = torch.randn(1, 4, 1, 16)
        k_deq = attn._dequant_archive_k()
        logits = attn._compute_archive_k_logits_dequant(q_lat, k_deq)
        assert logits.shape == (1, 4, 1, 4)

    def test_mse_quantizer_uses_dequant_path(self):
        attn = HAWPAttention(
            _make_config(), layer_idx=0, r_k=16, r_v=16,
            use_archive_k_ip_approx=True,
        )
        k_quantizer = TurboQuantMSE(dim=16, bits=4, use_rotation=False)
        v_quantizer = TurboQuantMSE(dim=16, bits=4, use_rotation=False)
        attn.setup_quant_cache(k_quantizer, v_quantizer, recent_window=2)

        n_kv = 4
        k_lat = torch.randn(1, n_kv, 6, 16)
        v_lat = torch.randn(1, n_kv, 6, 16)
        attn._quant_cache_append(k_lat, v_lat)
        attn._quant_cache_demote()

        assert not attn._can_use_archive_k_ip_approx()

    def test_no_archive_returns_false(self):
        attn = HAWPAttention(
            _make_config(), layer_idx=0, r_k=16, r_v=16,
            use_archive_k_ip_approx=True,
        )
        assert not attn._can_use_archive_k_ip_approx()

    def test_full_forward_with_switch_off(self):
        attn = HAWPAttention(
            _make_config(), layer_idx=0, r_k=16, r_v=16,
            use_archive_k_ip_approx=False,
        )
        k_quantizer = TurboQuantProd(dim=16, bits=4, use_rotation=False)
        v_quantizer = TurboQuantProd(dim=16, bits=4, use_rotation=False)
        attn.setup_quant_cache(k_quantizer, v_quantizer, recent_window=4)
        attn.eval()

        x1 = torch.randn(1, 6, 256)
        with torch.no_grad():
            out1 = attn(x1, attention_mask=None, use_cache=True)[0]
        assert out1.shape == (1, 6, 256)
