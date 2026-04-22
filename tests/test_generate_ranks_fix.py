import torch
import pytest

from hawp_laq.runtime.generate import (
    _resolve_head_dim_from_model_or_attn,
    _setup_quant_cache_per_layer,
    _convert_and_load_projectors,
)
from hawp_laq.modeling.attention_hawp import HAWPAttention
from hawp_laq.config import HAWPLAQConfig, build_k_quantizer, build_v_quantizer
from types import SimpleNamespace
import torch.nn as nn


def _make_dummy_opt_model(n_layers=2):
    config = SimpleNamespace(
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=2048,
        rope_theta=10000.0,
        model_type="opt",
        enable_bias=False,
        attention_dropout=0.0,
        _hawp_converted=False,
    )

    class DummyAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = config
            self.q_proj = nn.Linear(64, 64, bias=False)
            self.k_proj = nn.Linear(64, 64, bias=False)
            self.v_proj = nn.Linear(64, 64, bias=False)
            self.o_proj = nn.Linear(64, 64, bias=False)

    class OPTDecoderLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = DummyAttention()

    class LayerList(nn.Module):
        def __init__(self, layers):
            super().__init__()
            self.layers = nn.ModuleList(layers)

    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = config
            self.model = LayerList([OPTDecoderLayer() for _ in range(n_layers)])

    return DummyModel()


def test_setup_hawp_quant_on_model_resolves_head_dim_before_conversion():
    model = _make_dummy_opt_model()
    head_dim = _resolve_head_dim_from_model_or_attn(model)
    assert head_dim == 16


def test_setup_quant_cache_per_layer_calls_setup_quant_cache():
    model = _make_dummy_opt_model()
    from hawp_laq.modeling.modeling_llama_hawp import convert_llama_to_hawp
    model = convert_llama_to_hawp(model, r_k=8, r_v=8)

    cfg = HAWPLAQConfig()
    _setup_quant_cache_per_layer(model, cfg, recent_window=8)

    for module in model.modules():
        if isinstance(module, HAWPAttention):
            assert module.use_cache_manager is True
            assert module._tq_k_quantizer is not None
            assert module._tq_v_quantizer is not None


def test_hawp_quant_main_path_enables_use_cache_manager():
    model = _make_dummy_opt_model()
    from hawp_laq.modeling.modeling_llama_hawp import convert_llama_to_hawp
    model = convert_llama_to_hawp(model, r_k=8, r_v=8)

    cfg = HAWPLAQConfig()
    _setup_quant_cache_per_layer(model, cfg, recent_window=8)

    hawp_layers = [m for m in model.modules() if isinstance(m, HAWPAttention)]
    assert len(hawp_layers) == 2
    for layer in hawp_layers:
        assert layer.use_cache_manager is True
