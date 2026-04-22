import torch
import pytest

from hawp_laq.modeling.attention_hawp import HAWPAttention
from types import SimpleNamespace


def _make_config():
    return SimpleNamespace(
        hidden_size=64, num_attention_heads=4,
        num_key_value_heads=4, max_position_embeddings=2048,
        rope_theta=10000.0, model_type="opt",
        enable_bias=False, attention_dropout=0.0,
    )


def _make_attn_module():
    import torch.nn as nn
    class DummyAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = _make_config()
            self.q_proj = nn.Linear(64, 64, bias=False)
            self.k_proj = nn.Linear(64, 64, bias=False)
            self.v_proj = nn.Linear(64, 64, bias=False)
            self.o_proj = nn.Linear(64, 64, bias=False)
    return DummyAttn()


def test_from_attention_missing_one_rank_raises_without_allow_default():
    attn = _make_attn_module()
    with pytest.raises(ValueError, match="HAWPAttention requires explicit r_k and r_v"):
        HAWPAttention.from_attention(attn, layer_idx=0)


def test_from_attention_quant_only_style_can_allow_default_full_rank():
    attn = _make_attn_module()
    hawp = HAWPAttention.from_attention(attn, layer_idx=0, allow_default_full_rank=True)
    assert hawp.r_k == 16
    assert hawp.r_v == 16


def test_from_attention_explicit_rk_rv_no_allow_needed():
    attn = _make_attn_module()
    hawp = HAWPAttention.from_attention(attn, layer_idx=0, r_k=8, r_v=6)
    assert hawp.r_k == 8
    assert hawp.r_v == 6


def test_attention_init_partial_ranks_raise_even_with_allow_default():
    config = _make_config()
    with pytest.raises(ValueError, match="both r_k and r_v together"):
        HAWPAttention(config, r_k=None, r_v=8, allow_default_full_rank=True)
    with pytest.raises(ValueError, match="both r_k and r_v together"):
        HAWPAttention(config, r_k=8, r_v=None, allow_default_full_rank=True)


def test_attention_init_both_none_allow_default_still_full_rank():
    config = _make_config()
    hawp = HAWPAttention(config, r_k=None, r_v=None, allow_default_full_rank=True)
    assert hawp.r_k == 16
    assert hawp.r_v == 16
