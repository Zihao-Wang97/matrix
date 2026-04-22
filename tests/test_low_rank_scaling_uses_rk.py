"""验证 low-rank attention 的 logits 缩放使用 sqrt(r_k)，
而不是 sqrt(head_dim)。

数学要求:
  low-rank logits = gamma * (q_lat @ k_lat^T) / sqrt(r_k)

当 r_k < head_dim 时, 1/sqrt(r_k) > 1/sqrt(head_dim),
二者产生的 logits 和 softmax 分布不同。
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from types import SimpleNamespace

from hawp_laq.modeling.attention_hawp import HAWPAttention, _make_causal_mask
from hawp_laq.utils.math_utils import orthogonalize


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


def test_low_rank_logits_use_sqrt_rk_not_head_dim():
    torch.manual_seed(0)
    r_k, head_dim = 16, 64
    n_heads, seq_len = 4, 8

    q_lat = torch.randn(n_heads, seq_len, r_k)
    k_lat = torch.randn(n_heads, seq_len, r_k)

    logits_rk = torch.matmul(q_lat, k_lat.transpose(1, 2)) / math.sqrt(r_k)
    logits_head = torch.matmul(q_lat, k_lat.transpose(1, 2)) / math.sqrt(head_dim)

    assert not torch.allclose(logits_rk, logits_head, atol=1e-4), (
        f"logits scaled by 1/sqrt(r_k={r_k}) must differ from 1/sqrt(head_dim={head_dim}); "
        "if they match, the test is trivial"
    )

    scale_ratio = math.sqrt(head_dim) / math.sqrt(r_k)
    assert torch.allclose(logits_rk, logits_head * scale_ratio, atol=1e-5), (
        "logits_rk should equal logits_head * sqrt(head_dim/r_k)"
    )


def test_real_low_rank_logits_scaling_matches_rk_if_accessible():
    d, r_k, r_v = 64, 16, 16

    attn = HAWPAttention(_make_config(), layer_idx=0, r_k=r_k, r_v=r_v)
    torch.manual_seed(7)
    P_k = orthogonalize(torch.randn(d, d))
    P_v = orthogonalize(torch.randn(d, d))
    attn.p_k.data.copy_(P_k)
    attn.p_v.data.copy_(P_v)
    attn.gamma.data.fill_(1.0)
    attn.eval()

    x = torch.randn(1, 4, 256)

    q = attn.q_proj(x) * attn.scaling
    k = attn.k_proj(x)
    v = attn.v_proj(x)

    q = q.view(1, 4, 4, d).transpose(1, 2)
    k = k.view(1, 4, 4, d).transpose(1, 2)
    v = v.view(1, 4, 4, d).transpose(1, 2)

    pk_down = attn.p_k[:, :r_k]
    pv_down = attn.p_v[:, :r_v]

    q_lat = q * (d ** 0.5) @ pk_down
    k_lat = k @ pk_down
    v_lat = v @ pv_down

    manual_logits = torch.matmul(q_lat, k_lat.transpose(2, 3)) / math.sqrt(r_k)
    causal_mask = _make_causal_mask(4, 4, x.device, x.dtype)
    manual_logits = manual_logits + causal_mask
    manual_weights = F.softmax(manual_logits, dim=-1, dtype=torch.float32).to(q.dtype)
    manual_out_lat = torch.matmul(manual_weights, v_lat)
    manual_out = (manual_out_lat @ pv_down.T).transpose(1, 2).reshape(1, 4, 256)
    manual_final = attn.o_proj(manual_out)

    with torch.no_grad():
        model_out = attn(x, attention_mask=None)[0]

    assert torch.allclose(model_out, manual_final, atol=1e-4), (
        f"Model output must match manual computation with 1/sqrt(r_k) scaling; "
        f"max_diff={(model_out - manual_final).abs().max().item():.2e}"
    )

    wrong_logits = torch.matmul(q_lat, k_lat.transpose(2, 3)) / math.sqrt(d)
    wrong_logits = wrong_logits + causal_mask
    wrong_weights = F.softmax(wrong_logits, dim=-1, dtype=torch.float32).to(q.dtype)
    wrong_out_lat = torch.matmul(wrong_weights, v_lat)
    wrong_out = (wrong_out_lat @ pv_down.T).transpose(1, 2).reshape(1, 4, 256)
    wrong_final = attn.o_proj(wrong_out)

    assert not torch.allclose(model_out, wrong_final, atol=1e-3), (
        "Model output must NOT match 1/sqrt(head_dim) scaling; "
        "if it does, the implementation is using the wrong scale"
    )
