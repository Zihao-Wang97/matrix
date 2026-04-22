"""验证 low-rank 投影/回投公式使用正确的列子空间 P_down @ P_down.T，
而不是错误地用前 r 行替代转置 P[:, :r] @ P[:r, :]。

正确:  x_hat = x @ P_down @ P_down.T   (P_down = P[:, :r], 列子空间投影)
错误:  x_hat = x @ P[:, :r] @ P[:r, :] (前 r 行 ≠ 列子空间的转置)
"""
from __future__ import annotations

import torch

from hawp_laq.modeling.attention_hawp import HAWPAttention
from hawp_laq.utils.math_utils import orthogonalize


def _random_orthogonal(d: int, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    return orthogonalize(torch.randn(d, d))


def test_low_rank_projection_reconstruction_uses_column_subspace():
    d, r = 64, 16
    P = _random_orthogonal(d, seed=7)
    P_down = P[:, :r]

    torch.manual_seed(1)
    x = torch.randn(2, 8, d)

    x_hat_correct = x @ P_down @ P_down.T
    x_hat_wrong = x @ P[:, :r] @ P[:r, :]

    assert not torch.allclose(x_hat_correct, x_hat_wrong, atol=1e-4), (
        "P_down @ P_down.T should differ from P[:,:r] @ P[:r,:] for general P; "
        "if they match, the test is not exercising a non-trivial case"
    )


def test_apply_pk_matches_pr_prt():
    d, r_k = 64, 16
    attn = HAWPAttention(
        _make_config(), layer_idx=0, r_k=r_k, r_v=r_k,
    )
    P = _random_orthogonal(d, seed=3)
    attn.p_k.data.copy_(P)
    attn.eval()

    torch.manual_seed(2)
    x = torch.randn(1, 4, 4, d)

    P_down = attn.p_k[:, :attn.r_k]
    expected = x @ P_down @ P_down.T
    result = attn._apply_pk(x)

    assert torch.allclose(result, expected, atol=1e-5), (
        f"_apply_pk must equal x @ P_down @ P_down.T, "
        f"max_diff={(result - expected).abs().max().item():.2e}"
    )


def test_apply_pv_matches_pr_prt():
    d, r_v = 64, 16
    attn = HAWPAttention(
        _make_config(), layer_idx=0, r_k=r_v, r_v=r_v,
    )
    P = _random_orthogonal(d, seed=5)
    attn.p_v.data.copy_(P)
    attn.gamma.data.fill_(3.0)
    attn.eval()

    torch.manual_seed(4)
    x = torch.randn(1, 4, 4, d)

    pv_down = attn.p_v[:, :attn.r_v]
    expected = x @ pv_down @ pv_down.T
    result = attn._apply_pv(x)

    assert torch.allclose(result, expected, atol=1e-5), (
        f"_apply_pv must equal x @ P_down @ P_down.T (no gamma), "
        f"max_diff={(result - expected).abs().max().item():.2e}"
    )


def _make_config():
    from types import SimpleNamespace
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
