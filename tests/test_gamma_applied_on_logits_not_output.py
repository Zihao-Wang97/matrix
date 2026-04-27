"""验证 gamma 作用在 logits 上 (softmax 之前)，
而不是乘在 softmax 之后的输出上。

正确路径:  logits = gamma * (q @ k^T) / sqrt(r_k)
           attn   = softmax(logits)
           out    = attn @ v

错误路径:  logits = (q @ k^T) / sqrt(r_k)
           attn   = softmax(logits)
           out    = gamma * (attn @ v)

gamma 作用在 logits 上会改变 softmax 的分布 (温度缩放)，
而 gamma 乘在输出上只做线性缩放，二者在 gamma != 1 时不等价。
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from types import SimpleNamespace

from hawp_laq.modeling.attention_hawp import HAWPAttention
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


def test_gamma_on_logits_differs_from_gamma_on_output():
    torch.manual_seed(42)
    d, r = 64, 16
    n_heads, seq_len = 4, 8

    q_lat = torch.randn(n_heads, seq_len, r)
    k_lat = torch.randn(n_heads, seq_len, r)
    v_lat = torch.randn(n_heads, seq_len, r)
    gamma = 2.5

    logits_raw = torch.matmul(q_lat, k_lat.transpose(1, 2)) / math.sqrt(r)

    logits_correct = gamma * logits_raw
    attn_correct = F.softmax(logits_correct, dim=-1)
    out_correct = torch.matmul(attn_correct, v_lat)

    attn_wrong = F.softmax(logits_raw, dim=-1)
    out_wrong = gamma * torch.matmul(attn_wrong, v_lat)

    assert not torch.allclose(out_correct, out_wrong, atol=1e-4), (
        "gamma on logits (temperature scaling) should differ from gamma on output (linear scaling) "
        "when gamma != 1"
    )


def test_real_attention_path_uses_gamma_in_logits_if_accessible():
    d, r_k, r_v = 64, 16, 16

    attn = HAWPAttention(_make_config(), layer_idx=0, r_k=r_k, r_v=r_v,
                         gamma_mode="fixed", gamma_value=None)
    torch.manual_seed(99)
    P = orthogonalize(torch.randn(d, d))
    attn.p_k.data.copy_(P)
    attn.p_v.data.copy_(P)
    attn.eval()

    x = torch.randn(1, 4, 256)

    with torch.no_grad():
        attn.gamma.data.fill_(1.0)
        out_g1 = attn(x, attention_mask=None)[0].clone()

        attn.gamma.data.fill_(2.0)
        out_g2 = attn(x, attention_mask=None)[0].clone()

    diff = (out_g2 - out_g1).abs().max().item()
    assert diff > 1e-3, (
        f"Changing gamma from 1→2 must change output (gamma is on logits, acts as temperature), "
        f"but max_diff={diff:.6f}"
    )

    attn.gamma.data.fill_(1.0)
    with torch.no_grad():
        out_g1_again = attn(x, attention_mask=None)[0].clone()
    assert torch.allclose(out_g1, out_g1_again, atol=1e-6), "Determinism check"


def test_apply_pv_has_no_gamma():
    d, r_v = 64, 16
    attn = HAWPAttention(_make_config(), layer_idx=0, r_k=r_v, r_v=r_v)
    torch.manual_seed(10)
    P = orthogonalize(torch.randn(d, d))
    attn.p_v.data.copy_(P)
    attn.gamma.data.fill_(5.0)
    attn.eval()

    x = torch.randn(1, 4, 4, d)
    pv_down = attn.p_v[:, :attn.r_v]
    expected = x @ pv_down @ pv_down.T
    result = attn._apply_pv(x)

    assert torch.allclose(result, expected, atol=1e-5), (
        "_apply_pv must NOT contain gamma; "
        f"max_diff={(result - expected).abs().max().item():.2e}"
    )
