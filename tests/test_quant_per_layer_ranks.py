import torch
import pytest
from pathlib import Path

from hawp_laq.runtime.projector_bank import save_projectors, load_projectors
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


def test_hawp_quant_uses_module_local_ranks_for_quantizer():
    import torch.nn as nn

    class _FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn0 = _make_attn(r_k=8, r_v=6, layer_idx=0)
            self.attn1 = _make_attn(r_k=16, r_v=12, layer_idx=1)

    model = _FakeModel()

    from hawp_laq.config import build_k_quantizer, build_v_quantizer, HAWPLAQConfig
    cfg = HAWPLAQConfig()

    for module in model.modules():
        if isinstance(module, HAWPAttention):
            k_q = build_k_quantizer(cfg, r_k=module.r_k)
            v_q = build_v_quantizer(cfg, r_v=module.r_v)
            assert k_q.dim == module.r_k, f"K quantizer dim {k_q.dim} != r_k {module.r_k}"
            assert v_q.dim == module.r_v, f"V quantizer dim {v_q.dim} != r_v {module.r_v}"
