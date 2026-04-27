"""验证 HAWP 自定义 attention 路径在 prefill 阶段是因果注意力，而非双向注意力。

覆盖:
1. prefill (q_len > 1) — 未来 token 的 attention prob 必须为 0 或极小
2. decode  (q_len = 1) — 单 token 路径正常，无需 causal mask
3. attention_mask=None 时自动构造 causal mask 的兜底逻辑
4. generate_hawp_quant() prefill 是否显式传入 attention_mask
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import torch

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


# ---------- _make_causal_mask 单元测试 ----------

def test_make_causal_mask_shape_and_values():
    seq_len = 5
    mask = _make_causal_mask(seq_len, seq_len, torch.device("cpu"), torch.float32)
    assert mask.shape == (1, 1, seq_len, seq_len), f"Expected (1,1,{seq_len},{seq_len}), got {mask.shape}"

    m = mask.squeeze()
    for i in range(seq_len):
        for j in range(seq_len):
            if j <= i:
                assert m[i, j].item() == 0.0, f"({i},{j}) visible but got {m[i,j].item()}"
            else:
                assert m[i, j].item() == float("-inf"), f"({i},{j}) should be masked but got {m[i,j].item()}"


def test_make_causal_mask_rectangular():
    mask = _make_causal_mask(2, 5, torch.device("cpu"), torch.float32)
    assert mask.shape == (1, 1, 2, 5)
    m = mask.squeeze()
    assert m[0, 0].item() == 0.0
    assert m[0, 4].item() == float("-inf")
    assert m[1, 0].item() == 0.0
    assert m[1, 4].item() == 0.0


# ---------- prefill 因果性 ----------

def test_hawp_prefill_attention_is_causal():
    attn = _build_attn(r_k=32, r_v=32)
    _setup_quant_cache(attn, recent_window=64)

    seq_len = 6
    torch.manual_seed(0)
    x = torch.randn(1, seq_len, 256)

    with torch.no_grad():
        _, attn_weights, _ = attn(x, attention_mask=None, use_cache=True)

    assert attn_weights is not None
    assert attn_weights.shape[-2:] == (seq_len, seq_len)

    for i in range(seq_len):
        for j in range(i + 1, seq_len):
            w = attn_weights[0, 0, i, j].item()
            assert w < 1e-5, (
                f"Prefill: token {i} should not attend to future token {j}, "
                f"but got weight={w:.6f}"
            )


def test_hawp_prefill_with_explicit_mask_matches_auto_mask():
    attn = _build_attn(r_k=32, r_v=32)
    _setup_quant_cache(attn, recent_window=64)

    seq_len = 4
    torch.manual_seed(1)
    x = torch.randn(1, seq_len, 256)

    causal_mask = _make_causal_mask(seq_len, seq_len, x.device, x.dtype)

    with torch.no_grad():
        out_auto = attn(x, attention_mask=None, use_cache=True)[0].clone()

    attn.reset_quant_cache()
    _setup_quant_cache(attn, recent_window=64)

    with torch.no_grad():
        out_explicit = attn(x, attention_mask=causal_mask, use_cache=True)[0].clone()

    assert torch.allclose(out_auto, out_explicit, atol=1e-5), (
        "Auto causal mask should produce same result as explicit causal mask"
    )


def test_hawp_prefill_no_mask_differs_from_bidirectional():
    attn = _build_attn(r_k=32, r_v=32)
    _setup_quant_cache(attn, recent_window=64)

    seq_len = 4
    torch.manual_seed(2)
    x = torch.randn(1, seq_len, 256)

    with torch.no_grad():
        out_causal, weights_causal, _ = attn(x, attention_mask=None, use_cache=True)

    n_heads = attn.num_heads
    head_dim = attn.head_dim
    r_k = attn.r_k

    q = attn.q_proj(x) * attn.scaling
    k = attn.k_proj(x)
    v = attn.v_proj(x)
    q = q.view(1, seq_len, n_heads, head_dim).transpose(1, 2)
    k = k.view(1, seq_len, n_heads, head_dim).transpose(1, 2)
    v = v.view(1, seq_len, n_heads, head_dim).transpose(1, 2)

    pk_down = attn.p_k[:, :r_k]
    pv_down = attn.p_v[:, :attn.r_v]

    q_lat = q * (head_dim ** 0.5) @ pk_down
    k_lat = k @ pk_down
    v_lat = v @ pv_down

    import math
    import torch.nn.functional as F
    logits_bidir = torch.matmul(q_lat, k_lat.transpose(2, 3)) / math.sqrt(r_k)
    weights_bidir = F.softmax(logits_bidir, dim=-1)
    out_bidir_lat = torch.matmul(weights_bidir, v_lat)
    out_bidir = attn.o_proj(
        (out_bidir_lat @ pv_down.T).transpose(1, 2).reshape(1, seq_len, 256)
    )

    assert not torch.allclose(out_causal, out_bidir, atol=1e-3), (
        "Causal attention output should differ from bidirectional attention output"
    )


# ---------- decode 单 token 路径 ----------

def test_hawp_decode_single_token_path_is_valid():
    attn = _build_attn(r_k=32, r_v=32)
    _setup_quant_cache(attn, recent_window=64)

    torch.manual_seed(3)
    x_prefill = torch.randn(1, 4, 256)
    with torch.no_grad():
        attn(x_prefill, attention_mask=None, use_cache=True)

    x_decode = torch.randn(1, 1, 256)
    with torch.no_grad():
        out, weights, _ = attn(x_decode, attention_mask=None, use_cache=True)

    assert out.shape == (1, 1, 256), f"Decode output shape should be (1,1,256), got {out.shape}"
    assert weights.shape[-2] == 1, f"Decode should have q_len=1, got {weights.shape[-2]}"


# ---------- 兜底逻辑: attention_mask=None + q_len>1 → 自动 causal ----------

def test_hawp_builds_causal_mask_when_needed_if_accessible():
    attn = _build_attn(r_k=32, r_v=32)
    _setup_quant_cache(attn, recent_window=64)

    seq_len = 4
    torch.manual_seed(4)
    x = torch.randn(1, seq_len, 256)

    with patch(
        "hawp_laq.modeling.attention_hawp._make_causal_mask",
        wraps=_make_causal_mask,
    ) as mock_mask:
        with torch.no_grad():
            attn(x, attention_mask=None, use_cache=True)
        mock_mask.assert_called_once_with(
            seq_len, seq_len, torch.device("cpu"), torch.float32,
        )


# ---------- generate_hawp_quant prefill 传入 attention_mask ----------

def test_generate_hawp_quant_prefill_passes_attention_mask_if_accessible():
    from hawp_laq.runtime.generate import generate_hawp_quant
    from hawp_laq.config import HAWPLAQConfig

    cfg = HAWPLAQConfig()
    cfg.generation.max_new_tokens = 1

    fake_model = MagicMock()
    fake_model.device = torch.device("cpu")

    logits = torch.zeros(1, 5, 50257)
    logits[0, -1, 42] = 1.0
    fake_model.return_value = MagicMock(logits=logits)
    fake_model.modules.return_value = []

    fake_tokenizer = MagicMock()
    fake_tokenizer.return_value.input_ids = torch.tensor([[1, 2, 3, 4, 5]])

    generate_hawp_quant(fake_model, fake_tokenizer, ["hello"], cfg)

    first_call = fake_model.call_args
    assert first_call is not None, "model() should have been called"

    call_kwargs = first_call[1] if first_call[1] else {}
    assert "attention_mask" in call_kwargs, (
        "generate_hawp_quant must pass attention_mask during prefill"
    )
    mask = call_kwargs["attention_mask"]
    assert mask is not None, "attention_mask must not be None"
    assert mask.shape[1] == 5, f"attention_mask length should match prompt_len=5, got {mask.shape[1]}"

    assert "position_ids" in call_kwargs, (
        "generate_hawp_quant must pass position_ids during prefill"
    )
