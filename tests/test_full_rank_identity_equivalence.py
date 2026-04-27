"""验证 full-rank + identity projector + gamma=1 条件下，
HAWP low-rank 路径与 baseline attention 数学等价。

条件: r_k = r_v = head_dim, P_k = I, P_v = I, gamma = 1

此时:
  q_lat = q @ I = q
  k_lat = k @ I = k
  v_lat = v @ I = v
  logits = 1 * (q @ k^T) / sqrt(head_dim) = baseline logits
  out    = (softmax(logits) @ v) @ I^T = baseline output
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from types import SimpleNamespace

from hawp_laq.modeling.attention_hawp import HAWPAttention


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


def test_full_rank_identity_matches_baseline_math():
    torch.manual_seed(0)
    d = 64
    n_heads, seq_len = 4, 8

    q = torch.randn(n_heads, seq_len, d)
    k = torch.randn(n_heads, seq_len, d)
    v = torch.randn(n_heads, seq_len, d)

    P_k = torch.eye(d)
    P_v = torch.eye(d)
    r_k = d
    r_v = d
    gamma = 1.0

    q_lat = q @ P_k[:, :r_k]
    k_lat = k @ P_k[:, :r_k]
    v_lat = v @ P_v[:, :r_v]

    logits_base = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(d)
    logits_hawp = gamma * torch.matmul(q_lat, k_lat.transpose(1, 2)) / math.sqrt(r_k)

    assert torch.allclose(logits_base, logits_hawp, atol=1e-6), (
        "Logits must match: gamma * q_lat @ k_lat^T / sqrt(r_k) == q @ k^T / sqrt(d) "
        "when P=I, r_k=d, gamma=1"
    )

    attn_base = F.softmax(logits_base, dim=-1)
    attn_hawp = F.softmax(logits_hawp, dim=-1)

    out_base = torch.matmul(attn_base, v)
    out_hawp = torch.matmul(attn_hawp, v_lat) @ P_v[:r_v, :].T

    assert torch.allclose(out_base, out_hawp, atol=1e-5), (
        f"Output must match when P=I, r_k=r_v=d, gamma=1; "
        f"max_diff={(out_base - out_hawp).abs().max().item():.2e}"
    )


def test_real_hawp_attention_matches_baseline_in_identity_mode():
    d = 64

    attn = HAWPAttention(_make_config(), layer_idx=0, r_k=d, r_v=d)
    assert torch.allclose(attn.p_k.float(), torch.eye(d), atol=1e-6), "P_k must be I"
    assert torch.allclose(attn.p_v.float(), torch.eye(d), atol=1e-6), "P_v must be I"
    assert abs(attn.gamma.item() - 1.0) < 1e-6, "gamma must be 1.0"
    attn.eval()

    torch.manual_seed(1)
    x = torch.randn(1, 4, 256)

    with torch.no_grad():
        model_out = attn(x, attention_mask=None)[0]

    q = attn.q_proj(x) * attn.scaling
    k = attn.k_proj(x)
    v = attn.v_proj(x)
    q = q.view(1, 4, 4, d).transpose(1, 2)
    k = k.view(1, 4, 4, d).transpose(1, 2)
    v = v.view(1, 4, 4, d).transpose(1, 2)

    logits = torch.matmul(q, k.transpose(2, 3))
    causal_mask = torch.triu(
        torch.full((4, 4), float("-inf"), dtype=logits.dtype, device=logits.device),
        diagonal=1,
    ).unsqueeze(0).unsqueeze(0)
    logits = logits + causal_mask
    weights = F.softmax(logits, dim=-1, dtype=torch.float32).to(q.dtype)
    attn_out = (weights @ v).transpose(1, 2).reshape(1, 4, 256)
    baseline_out = attn.o_proj(attn_out)

    assert torch.allclose(model_out, baseline_out, atol=1e-4), (
        f"Full-rank identity HAWPAttention must match baseline; "
        f"max_diff={(model_out - baseline_out).abs().max().item():.2e}"
    )
