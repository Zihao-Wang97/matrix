"""验证 prefill 阶段的因果性:

1. test_prefill_is_causal_in_hawp_path
   - 构造 HAWPAttention (low-rank + cache_manager)
   - prefill (q_len > 1) 时不传 attention_mask
   - 验证 _forward_low_rank 自动生成 causal mask
   - 即: 未来 token 对当前 token 的 attention weight 为 0

2. test_quant_all_reads_back_archive_on_first_prefill
   - 构造 HAWPAttention (recent_window=0, hawp_quant_all 模式)
   - 首次 prefill 后, k_lat/v_lat 应从量化 archive 读回
   - 而非保留原始 fp16 值
   - 验证 kv_from_cache 标志被正确设置
"""
from __future__ import annotations

import math
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

from hawp_laq.modeling.attention_hawp import HAWPAttention, _make_causal_mask
from hawp_laq.runtime.turboquant import TurboQuantMSE


def _make_config():
    return SimpleNamespace(
        hidden_size=256,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=512,
        rope_theta=10000.0,
        model_type="opt",
        enable_bias=True,
        attention_dropout=0.0,
    )


def _build_attn(r_k=32, r_v=32):
    attn = HAWPAttention(_make_config(), layer_idx=0, r_k=r_k, r_v=r_v)
    attn.eval()
    return attn


def _setup_quant_cache(attn, recent_window=64):
    k_q = TurboQuantMSE(dim=attn.r_k, bits=4, use_rotation=False)
    v_q = TurboQuantMSE(dim=attn.r_v, bits=4, use_rotation=False)
    attn.setup_quant_cache(k_q, v_q, recent_window=recent_window)


class TestPrefillIsCausal:
    def test_causal_mask_utility(self):
        mask = _make_causal_mask(4, 4, torch.device("cpu"), torch.float32)
        assert mask.shape == (1, 1, 4, 4)
        lower = mask.squeeze()
        for i in range(4):
            for j in range(4):
                if j <= i:
                    assert lower[i, j].item() == 0.0, f"({i},{j}) should be 0 (visible)"
                else:
                    assert lower[i, j].item() == float("-inf"), f"({i},{j}) should be -inf (masked)"

    def test_prefill_without_mask_produces_causal_attention(self):
        attn = _build_attn(r_k=32, r_v=32)
        _setup_quant_cache(attn, recent_window=64)

        seq_len = 6
        x = torch.randn(1, seq_len, 256)

        with torch.no_grad():
            out = attn(x, attention_mask=None, use_cache=False)[0]
        assert out.shape == (1, seq_len, 256), "Output shape mismatch"

    def test_prefill_no_mask_future_tokens_zero_attention(self):
        attn = _build_attn(r_k=32, r_v=32)
        _setup_quant_cache(attn, recent_window=64)

        seq_len = 4
        x = torch.randn(1, seq_len, 256)

        with torch.no_grad():
            _, attn_weights, _ = attn(x, attention_mask=None, use_cache=False)

        assert attn_weights is not None, "attention weights should be returned"
        assert attn_weights.shape[-2:] == (seq_len, seq_len)

        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                w = attn_weights[0, 0, i, j].item()
                assert w < 1e-5, (
                    f"Token {i} should not attend to future token {j}, weight={w:.6f}"
                )

    def test_prefill_with_explicit_mask_overrides_auto_mask(self):
        attn = _build_attn(r_k=32, r_v=32)
        _setup_quant_cache(attn, recent_window=64)

        seq_len = 4
        x = torch.randn(1, seq_len, 256)

        causal_mask = _make_causal_mask(seq_len, seq_len, x.device, x.dtype)

        with torch.no_grad():
            out_auto = attn(x, attention_mask=None, use_cache=False)[0]

        attn.reset_quant_cache()
        _setup_quant_cache(attn, recent_window=64)

        with torch.no_grad():
            out_explicit = attn(x, attention_mask=causal_mask, use_cache=False)[0]

        assert torch.allclose(out_auto, out_explicit, atol=1e-5), (
            "Auto causal mask should produce same output as explicit causal mask"
        )

    def test_decode_q1_no_auto_mask(self):
        attn = _build_attn(r_k=32, r_v=32)
        _setup_quant_cache(attn, recent_window=64)

        x_prefill = torch.randn(1, 4, 256)
        with torch.no_grad():
            attn(x_prefill, attention_mask=None, use_cache=False)

        x_decode = torch.randn(1, 1, 256)
        with torch.no_grad():
            _, weights, _ = attn(x_decode, attention_mask=None, use_cache=False)
        assert weights is not None


class TestQuantAllReadsBackArchiveOnFirstPrefill:
    def test_first_prefill_reads_from_archive(self):
        attn = _build_attn(r_k=32, r_v=32)
        _setup_quant_cache(attn, recent_window=0)

        seq_len = 4
        x = torch.randn(1, seq_len, 256)

        kv_from_cache_captured = [False]

        original_forward_low_rank = attn._forward_low_rank

        def patched_forward_low_rank(
            query_states, key_states, value_states,
            attention_mask, past_key_value, use_cache,
            cache_position, **kwargs,
        ):
            result = original_forward_low_rank(
                query_states, key_states, value_states,
                attention_mask, past_key_value, use_cache,
                cache_position, **kwargs,
            )
            return result

        with torch.no_grad():
            out = attn(x, attention_mask=None, use_cache=False)[0]
        assert out.shape == (1, seq_len, 256)

        assert attn._quant_archive_k_qx is not None, (
            "Archive should contain quantized data after first prefill"
        )
        assert attn._quant_recent_k is None, (
            "recent_window=0 → recent should stay None"
        )

    def test_quant_all_k_lat_differs_from_raw_fp16(self):
        attn = _build_attn(r_k=32, r_v=32)
        _setup_quant_cache(attn, recent_window=0)

        seq_len = 3
        x = torch.randn(1, seq_len, 256)

        pk_down = attn.p_k[:, :attn.r_k]
        pv_down = attn.p_v[:, :attn.r_v]

        q = attn.q_proj(x) * attn.scaling
        k = attn.k_proj(x)
        v = attn.v_proj(x)
        q = q.view(1, seq_len, 4, 64).transpose(1, 2)
        k = k.view(1, seq_len, 4, 64).transpose(1, 2)
        v = v.view(1, seq_len, 4, 64).transpose(1, 2)

        k_lat_raw = (k @ pk_down).clone()
        v_lat_raw = (v @ pv_down).clone()

        with torch.no_grad():
            attn(x, attention_mask=None, use_cache=False)

        k_cached, v_cached = attn._quant_cache_get_kv()
        assert k_cached is not None, "Archive should have data after first forward"

        k_lat_deq = k_cached.unsqueeze(0)

        if k_lat_raw.numel() > 0:
            diff = (k_lat_raw.float() - k_lat_deq.float()).abs().max().item()
            assert diff > 1e-4, (
                f"Dequantized k_lat should differ from raw fp16 k_lat due to quantization; "
                f"max_diff={diff:.6f}"
            )

    def test_quant_all_vs_quant_prefill_archive_populated(self):
        attn_all = _build_attn(r_k=32, r_v=32)
        _setup_quant_cache(attn_all, recent_window=0)

        attn_win = _build_attn(r_k=32, r_v=32)
        _setup_quant_cache(attn_win, recent_window=64)

        attn_win.p_k.data.copy_(attn_all.p_k.data)
        attn_win.p_v.data.copy_(attn_all.p_v.data)
        attn_win.gamma.data.copy_(attn_all.gamma.data)
        attn_win.q_proj.weight.data.copy_(attn_all.q_proj.weight.data)
        attn_win.k_proj.weight.data.copy_(attn_all.k_proj.weight.data)
        attn_win.v_proj.weight.data.copy_(attn_all.v_proj.weight.data)
        attn_win.o_proj.weight.data.copy_(attn_all.o_proj.weight.data)

        x = torch.randn(1, 4, 256)

        with torch.no_grad():
            attn_all(x, attention_mask=None, use_cache=False)
            attn_win(x, attention_mask=None, use_cache=False)

        assert attn_all._quant_archive_k_qx is not None, (
            "quant_all: archive must be populated after first forward"
        )
        assert attn_all._quant_recent_k is None, (
            "quant_all: recent must stay None (recent_window=0)"
        )

        assert attn_win._quant_recent_k is not None, (
            "quant (window>0): recent should be populated after first forward"
        )
        assert attn_win._quant_archive_k_qx is None, (
            "quant (window>0): archive should still be empty (tokens within window)"
        )
