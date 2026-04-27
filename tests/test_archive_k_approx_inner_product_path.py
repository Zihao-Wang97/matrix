from __future__ import annotations

import torch

from hawp_laq.modeling.attention_hawp import HAWPAttention
from hawp_laq.runtime.turboquant import TurboQuantProd


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


class TestArchiveKApproxInnerProductPath:
    def test_approx_path_called_when_turboquant_prod_and_switch_on(self):
        attn = HAWPAttention(
            _make_config(), layer_idx=0, r_k=16, r_v=16,
            use_archive_k_ip_approx=True,
        )
        k_quantizer = TurboQuantProd(dim=16, bits=4, use_rotation=False)
        v_quantizer = TurboQuantProd(dim=16, bits=4, use_rotation=False)
        attn.setup_quant_cache(k_quantizer, v_quantizer, recent_window=2)

        n_kv = 4
        k_lat = torch.randn(1, n_kv, 6, 16)
        v_lat = torch.randn(1, n_kv, 6, 16)
        attn._quant_cache_append(k_lat, v_lat)
        attn._quant_cache_demote()

        T_archive = 4
        assert bool(attn._quant_archive_chunks)
        assert attn._can_use_archive_k_ip_approx()

        q_lat = torch.randn(1, 4, 1, 16)
        archive_logits = attn._compute_archive_k_logits_approx(q_lat)
        assert archive_logits.shape == (1, 4, 1, T_archive)

    def test_approx_path_result_close_to_dequant_path(self):
        attn = HAWPAttention(
            _make_config(), layer_idx=0, r_k=16, r_v=16,
            use_archive_k_ip_approx=True,
        )
        k_quantizer = TurboQuantProd(dim=16, bits=4, use_rotation=False)
        v_quantizer = TurboQuantProd(dim=16, bits=4, use_rotation=False)
        attn.setup_quant_cache(k_quantizer, v_quantizer, recent_window=2)

        torch.manual_seed(42)
        n_kv = 4
        k_lat = torch.randn(1, n_kv, 8, 16)
        v_lat = torch.randn(1, n_kv, 8, 16)
        attn._quant_cache_append(k_lat, v_lat)
        attn._quant_cache_demote()

        q_lat = torch.randn(1, 4, 1, 16)
        logits_approx = attn._compute_archive_k_logits_approx(q_lat)

        k_deq = attn._dequant_archive_k()
        logits_deq = attn._compute_archive_k_logits_dequant(q_lat, k_deq)

        assert logits_approx.shape == logits_deq.shape
        assert torch.allclose(logits_approx, logits_deq, atol=0.5), (
            f"approx and dequant paths should produce similar logits, "
            f"max_diff={(logits_approx - logits_deq).abs().max().item():.4f}"
        )

    def test_prefill_approx_path(self):
        attn = HAWPAttention(
            _make_config(), layer_idx=0, r_k=16, r_v=16,
            use_archive_k_ip_approx=True,
        )
        k_quantizer = TurboQuantProd(dim=16, bits=4, use_rotation=False)
        v_quantizer = TurboQuantProd(dim=16, bits=4, use_rotation=False)
        attn.setup_quant_cache(k_quantizer, v_quantizer, recent_window=2)

        n_kv = 4
        k_lat = torch.randn(1, n_kv, 8, 16)
        v_lat = torch.randn(1, n_kv, 8, 16)
        attn._quant_cache_append(k_lat, v_lat)
        attn._quant_cache_demote()

        q_lat = torch.randn(1, 4, 4, 16)
        archive_logits = attn._compute_archive_k_logits_approx(q_lat)
        assert archive_logits.shape == (1, 4, 4, 6)

    def test_full_forward_with_approx_path(self):
        attn = HAWPAttention(
            _make_config(), layer_idx=0, r_k=16, r_v=16,
            use_archive_k_ip_approx=True,
        )
        k_quantizer = TurboQuantProd(dim=16, bits=4, use_rotation=False)
        v_quantizer = TurboQuantProd(dim=16, bits=4, use_rotation=False)
        attn.setup_quant_cache(k_quantizer, v_quantizer, recent_window=4)
        attn.eval()

        x1 = torch.randn(1, 6, 256)
        with torch.no_grad():
            out1 = attn(x1, attention_mask=None, use_cache=True)[0]

        assert out1.shape == (1, 6, 256)

    def test_gqa_approx_path(self):
        from types import SimpleNamespace
        config = SimpleNamespace(
            hidden_size=256,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=512,
            rope_theta=10000.0,
            model_type="llama",
            enable_bias=False,
            attention_dropout=0.0,
        )
        attn = HAWPAttention(
            config, layer_idx=0, r_k=16, r_v=16,
            use_archive_k_ip_approx=True,
        )
        assert attn.num_key_value_groups == 2

        k_quantizer = TurboQuantProd(dim=16, bits=4, use_rotation=False)
        v_quantizer = TurboQuantProd(dim=16, bits=4, use_rotation=False)
        attn.setup_quant_cache(k_quantizer, v_quantizer, recent_window=2)

        n_kv = 2
        k_lat = torch.randn(1, n_kv, 6, 16)
        v_lat = torch.randn(1, n_kv, 6, 16)
        attn._quant_cache_append(k_lat, v_lat)
        attn._quant_cache_demote()

        q_lat = torch.randn(1, 4, 1, 16)
        archive_logits = attn._compute_archive_k_logits_approx(q_lat)
        assert archive_logits.shape == (1, 4, 1, 4)
