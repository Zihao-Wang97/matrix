from __future__ import annotations

import torch
import torch.nn.functional as F

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


class TestArchiveRecentLogitConcatSoftmax:
    def test_archive_recent_logits_concatenated_before_softmax(self):
        attn = HAWPAttention(
            _make_config(), layer_idx=0, r_k=16, r_v=16,
            use_archive_k_ip_approx=True,
        )
        k_quantizer = TurboQuantProd(dim=16, bits=4, use_rotation=False)
        v_quantizer = TurboQuantProd(dim=16, bits=4, use_rotation=False)
        attn.setup_quant_cache(k_quantizer, v_quantizer, recent_window=3)
        attn.eval()

        torch.manual_seed(0)
        n_kv = 4
        k_lat = torch.randn(1, n_kv, 8, 16)
        v_lat = torch.randn(1, n_kv, 8, 16)
        attn._quant_cache_append(k_lat, v_lat)
        attn._quant_cache_demote()

        k_lat2 = torch.randn(1, n_kv, 3, 16)
        v_lat2 = torch.randn(1, n_kv, 3, 16)
        attn._quant_cache_append(k_lat2, v_lat2)

        assert bool(attn._quant_archive_chunks)
        assert attn._quant_recent_k is not None

        q_lat = torch.randn(1, 4, 1, 16)
        logit_scale = attn._compute_low_rank_logit_scale(q_lat)

        archive_logits = attn._compute_archive_k_logits_approx(q_lat)
        recent_logits = attn._compute_recent_k_logits(q_lat, attn._quant_recent_k)

        T_archive = archive_logits.shape[-1]
        T_recent = recent_logits.shape[-1]

        concat_logits = torch.cat([archive_logits, recent_logits], dim=-1) * logit_scale
        concat_weights = F.softmax(concat_logits, dim=-1, dtype=torch.float32).to(q_lat.dtype)

        assert concat_weights.shape == (1, 4, 1, T_archive + T_recent)
        assert torch.allclose(concat_weights.sum(dim=-1, keepdim=True), torch.ones(1, 1, 1, 1), atol=1e-5)

    def test_separate_softmax_differs_from_unified_softmax(self):
        attn = HAWPAttention(
            _make_config(), layer_idx=0, r_k=16, r_v=16,
            use_archive_k_ip_approx=True,
        )
        k_quantizer = TurboQuantProd(dim=16, bits=4, use_rotation=False)
        v_quantizer = TurboQuantProd(dim=16, bits=4, use_rotation=False)
        attn.setup_quant_cache(k_quantizer, v_quantizer, recent_window=3)
        attn.eval()

        torch.manual_seed(99)
        n_kv = 4
        k_lat = torch.randn(1, n_kv, 8, 16)
        v_lat = torch.randn(1, n_kv, 8, 16)
        attn._quant_cache_append(k_lat, v_lat)
        attn._quant_cache_demote()
        k_lat2 = torch.randn(1, n_kv, 3, 16)
        v_lat2 = torch.randn(1, n_kv, 3, 16)
        attn._quant_cache_append(k_lat2, v_lat2)

        q_lat = torch.randn(1, 4, 1, 16)
        logit_scale = attn._compute_low_rank_logit_scale(q_lat)

        archive_logits = attn._compute_archive_k_logits_approx(q_lat) * logit_scale
        recent_logits = attn._compute_recent_k_logits(q_lat, attn._quant_recent_k) * logit_scale

        unified = F.softmax(
            torch.cat([archive_logits, recent_logits], dim=-1),
            dim=-1, dtype=torch.float32,
        ).to(q_lat.dtype)

        arch_w = F.softmax(archive_logits, dim=-1, dtype=torch.float32).to(q_lat.dtype)
        rec_w = F.softmax(recent_logits, dim=-1, dtype=torch.float32).to(q_lat.dtype)

        assert not torch.allclose(unified[:, :, :, :archive_logits.shape[-1]], arch_w, atol=1e-3)
        assert not torch.allclose(unified[:, :, :, archive_logits.shape[-1]:], rec_w, atol=1e-3)

    def test_result_shape_correct_with_archive_and_recent(self):
        attn = HAWPAttention(
            _make_config(), layer_idx=0, r_k=16, r_v=16,
            use_archive_k_ip_approx=True,
        )
        k_quantizer = TurboQuantProd(dim=16, bits=4, use_rotation=False)
        v_quantizer = TurboQuantProd(dim=16, bits=4, use_rotation=False)
        attn.setup_quant_cache(k_quantizer, v_quantizer, recent_window=3)
        attn.eval()

        n_kv = 4
        k_lat = torch.randn(1, n_kv, 6, 16)
        v_lat = torch.randn(1, n_kv, 6, 16)
        attn._quant_cache_append(k_lat, v_lat)
        attn._quant_cache_demote()
        k_lat2 = torch.randn(1, n_kv, 3, 16)
        v_lat2 = torch.randn(1, n_kv, 3, 16)
        attn._quant_cache_append(k_lat2, v_lat2)

        x = torch.randn(1, 1, 256)
        with torch.no_grad():
            out = attn(x, attention_mask=None, use_cache=True)[0]
        assert out.shape == (1, 1, 256)

    def test_decode_and_prefill_both_work(self):
        attn = HAWPAttention(
            _make_config(), layer_idx=0, r_k=16, r_v=16,
            use_archive_k_ip_approx=True,
        )
        k_quantizer = TurboQuantProd(dim=16, bits=4, use_rotation=False)
        v_quantizer = TurboQuantProd(dim=16, bits=4, use_rotation=False)
        attn.setup_quant_cache(k_quantizer, v_quantizer, recent_window=3)
        attn.eval()

        n_kv = 4
        k_lat = torch.randn(1, n_kv, 6, 16)
        v_lat = torch.randn(1, n_kv, 6, 16)
        attn._quant_cache_append(k_lat, v_lat)
        attn._quant_cache_demote()
        k_lat2 = torch.randn(1, n_kv, 3, 16)
        v_lat2 = torch.randn(1, n_kv, 3, 16)
        attn._quant_cache_append(k_lat2, v_lat2)

        x_decode = torch.randn(1, 1, 256)
        with torch.no_grad():
            out_decode = attn(x_decode, attention_mask=None, use_cache=True)[0]
        assert out_decode.shape == (1, 1, 256)

        x_prefill = torch.randn(1, 4, 256)
        with torch.no_grad():
            out_prefill = attn(x_prefill, attention_mask=None, use_cache=True)[0]
        assert out_prefill.shape == (1, 4, 256)

    def test_archive_only_no_recent(self):
        attn = HAWPAttention(
            _make_config(), layer_idx=0, r_k=16, r_v=16,
            use_archive_k_ip_approx=True,
        )
        k_quantizer = TurboQuantProd(dim=16, bits=4, use_rotation=False)
        v_quantizer = TurboQuantProd(dim=16, bits=4, use_rotation=False)
        attn.setup_quant_cache(k_quantizer, v_quantizer, recent_window=0)
        attn.eval()

        n_kv = 4
        k_lat = torch.randn(1, n_kv, 8, 16)
        v_lat = torch.randn(1, n_kv, 8, 16)
        attn._quant_cache_append_to_archive(k_lat, v_lat)

        x = torch.randn(1, 1, 256)
        with torch.no_grad():
            out = attn(x, attention_mask=None, use_cache=True)[0]
        assert out.shape == (1, 1, 256)
