"""验证 hawp_quant_all (recent_window==0) 模式下，第一次 prefill 时：
- token 被写入 quantized archive (_quant_cache_append_to_archive)
- 同一轮 forward 从 archive 读回 (_quant_cache_get_kv)
- attention 使用的 K/V 是 archive 读回后的量化/反量化结果，而非原始 fp16 旁路

为什么需要 monkeypatch:
  _quant_cache_get_kv / _quant_cache_append_to_archive 是 HAWPAttention 的内部方法，
  没有公开返回值标记 "kv_from_cache"，所以通过 mock 来观测调用是否发生。
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import torch

from hawp_laq.modeling.attention_hawp import HAWPAttention
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


def _build_attn(r_k=32, r_v=32, recent_window=0):
    attn = HAWPAttention(_make_config(), layer_idx=0, r_k=r_k, r_v=r_v)
    k_q = TurboQuantMSE(dim=r_k, bits=4, use_rotation=False)
    v_q = TurboQuantMSE(dim=r_v, bits=4, use_rotation=False)
    attn.setup_quant_cache(k_q, v_q, recent_window=recent_window)
    attn.eval()
    return attn


# ---------- 核心验证: append + get_kv 都被调用 ----------

def test_quant_all_first_prefill_reads_back_archive():
    attn = _build_attn(r_k=32, r_v=32, recent_window=0)

    with patch.object(
        attn, "_quant_cache_append_to_archive", wraps=attn._quant_cache_append_to_archive
    ) as mock_append, patch.object(
        attn, "_quant_cache_get_kv", wraps=attn._quant_cache_get_kv
    ) as mock_get_kv:
        torch.manual_seed(0)
        x = torch.randn(1, 4, 256)
        with torch.no_grad():
            out = attn(x, attention_mask=None, use_cache=False)[0]

        mock_append.assert_called_once(), (
            "_quant_cache_append_to_archive must be called on first prefill"
        )
        mock_get_kv.assert_called_once(), (
            "_quant_cache_get_kv must be called after append_to_archive "
            "on first prefill (quant_all must read back from archive)"
        )

    assert out.shape == (1, 4, 256), "Output shape mismatch"


# ---------- 验证: 不能只 append 不 read-back ----------

def test_quant_all_first_prefill_does_not_bypass_archive_kv():
    attn = _build_attn(r_k=32, r_v=32, recent_window=0)

    torch.manual_seed(1)
    x = torch.randn(1, 4, 256)

    with torch.no_grad():
        attn(x, attention_mask=None, use_cache=False)

    assert attn._quant_archive_k_qx is not None, (
        "Archive must contain quantized data after first prefill"
    )
    assert attn._quant_recent_k is None, (
        "recent_window=0 → recent must stay None"
    )

    k_cached, v_cached = attn._quant_cache_get_kv()
    assert k_cached is not None, (
        "_quant_cache_get_kv must return data after first prefill"
    )
    assert k_cached.shape[1] == 4, (
        f"Archive should have 4 tokens after prefill, got {k_cached.shape[1]}"
    )


# ---------- 验证: K/V 来自 archive 反量化，而非原始 fp16 旁路 ----------

def test_quant_all_uses_quantized_archive_when_recent_window_zero_if_accessible():
    attn = _build_attn(r_k=32, r_v=32, recent_window=0)

    torch.manual_seed(2)
    x = torch.randn(1, 3, 256)

    pk_down = attn.p_k[:, :attn.r_k]

    q = attn.q_proj(x) * attn.scaling
    k = attn.k_proj(x)
    q = q.view(1, 3, 4, 64).transpose(1, 2)
    k = k.view(1, 3, 4, 64).transpose(1, 2)
    k_lat_raw = (k @ pk_down).clone()

    with torch.no_grad():
        attn(x, attention_mask=None, use_cache=False)

    k_cached, _ = attn._quant_cache_get_kv()
    k_lat_deq = k_cached

    diff = (k_lat_raw[0].float() - k_lat_deq.float()).abs().max().item()
    assert diff > 1e-4, (
        f"Dequantized k_lat should differ from raw fp16 k_lat due to quantization; "
        f"max_diff={diff:.6f}. If they are identical, the forward path bypassed the archive."
    )


# ---------- 对比: recent_window>0 时 archive 为空 ----------

def test_quant_all_vs_quant_window_behavior():
    attn_all = _build_attn(r_k=32, r_v=32, recent_window=0)
    attn_win = _build_attn(r_k=32, r_v=32, recent_window=64)

    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        getattr(attn_win, name).weight.data.copy_(getattr(attn_all, name).weight.data)
    attn_win.p_k.data.copy_(attn_all.p_k.data)
    attn_win.p_v.data.copy_(attn_all.p_v.data)
    attn_win.gamma.data.copy_(attn_all.gamma.data)

    torch.manual_seed(3)
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


# ---------- 验证: 第二次 forward 继续走 archive 路径 ----------

def test_quant_all_second_decode_continues_archive_path():
    attn = _build_attn(r_k=32, r_v=32, recent_window=0)

    torch.manual_seed(4)
    x_prefill = torch.randn(1, 3, 256)
    with torch.no_grad():
        attn(x_prefill, attention_mask=None, use_cache=False)

    archive_qx_after_prefill = attn._quant_archive_k_qx

    with patch.object(
        attn, "_quant_cache_append_to_archive", wraps=attn._quant_cache_append_to_archive
    ) as mock_append, patch.object(
        attn, "_quant_cache_get_kv", wraps=attn._quant_cache_get_kv
    ) as mock_get_kv:
        x_decode = torch.randn(1, 1, 256)
        with torch.no_grad():
            attn(x_decode, attention_mask=None, use_cache=False)

        mock_append.assert_called_once(), (
            "Decode step must also append to archive"
        )
        mock_get_kv.assert_called_once(), (
            "Decode step must also read back from archive"
        )

    k_cached, _ = attn._quant_cache_get_kv()
    assert k_cached.shape[1] == 4, (
        f"After prefill(3)+decode(1), archive should have 4 tokens, got {k_cached.shape[1]}"
    )
