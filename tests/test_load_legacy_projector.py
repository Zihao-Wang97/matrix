import torch
import pytest

from hawp_laq.modeling.attention_hawp import HAWPAttention
from types import SimpleNamespace


def _make_attn(r_k=8, r_v=8, layer_idx=0):
    config = SimpleNamespace(
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=2048,
        rope_theta=10000.0,
        model_type="opt",
        enable_bias=False,
        attention_dropout=0.0,
    )
    return HAWPAttention(config, layer_idx=layer_idx, r_k=r_k, r_v=r_v)


def test_load_legacy_rectangular_projector_format():
    attn = _make_attn(r_k=8, r_v=6, layer_idx=0)
    head_dim = 16

    legacy_p_k = torch.randn(head_dim, 8)
    legacy_p_v = torch.randn(head_dim, 6)
    data = {"p_k": legacy_p_k, "p_v": legacy_p_v, "gamma": torch.tensor([1.5])}

    attn.load_projector_data(data, strict=True)

    assert torch.allclose(attn.p_k.data[:, :8], legacy_p_k)
    assert torch.allclose(attn.p_v.data[:, :6], legacy_p_v)
    assert attn.p_k.data[:, 8:].abs().max() == 0.0
    assert attn.p_v.data[:, 6:].abs().max() == 0.0
    assert torch.allclose(attn.gamma.data, torch.tensor([1.5]))
