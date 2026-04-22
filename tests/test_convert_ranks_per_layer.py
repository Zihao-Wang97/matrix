import torch
import pytest

from hawp_laq.modeling.modeling_llama_hawp import convert_llama_to_hawp
from hawp_laq.modeling.attention_hawp import HAWPAttention
import torch.nn as nn
from types import SimpleNamespace


def _make_dummy_opt_model():
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
            self.model = LayerList([OPTDecoderLayer(), OPTDecoderLayer(), OPTDecoderLayer()])

    return DummyModel()


def test_convert_llama_to_hawp_supports_ranks_per_layer():
    model = _make_dummy_opt_model()
    ranks_per_layer = {0: (8, 4), 2: (16, 16)}
    model = convert_llama_to_hawp(model, r_k=12, r_v=12, ranks_per_layer=ranks_per_layer)

    hawp_layers = [m for m in model.modules() if isinstance(m, HAWPAttention)]
    assert len(hawp_layers) == 3

    assert hawp_layers[0].r_k == 8
    assert hawp_layers[0].r_v == 4

    assert hawp_layers[1].r_k == 12
    assert hawp_layers[1].r_v == 12

    assert hawp_layers[2].r_k == 16
    assert hawp_layers[2].r_v == 16

    assert model.config._hawp_converted is True
    assert model.config._hawp_r_k == 12
    assert model.config._hawp_r_v == 12
    assert model.config._hawp_ranks_per_layer == ranks_per_layer


def test_convert_llama_to_hawp_global_default_metadata_when_per_layer_ranks_used():
    model = _make_dummy_opt_model()
    ranks_per_layer = {0: (8, 4), 2: (16, 16)}
    model = convert_llama_to_hawp(model, r_k=12, r_v=12, ranks_per_layer=ranks_per_layer)

    assert model.config._hawp_ranks_per_layer == ranks_per_layer
    assert model.config._hawp_global_default_r_k == 12
    assert model.config._hawp_global_default_r_v == 12
    assert model.config._hawp_uses_per_layer_ranks is True


def test_convert_llama_to_hawp_no_per_layer_ranks_metadata():
    model = _make_dummy_opt_model()
    model = convert_llama_to_hawp(model, r_k=8, r_v=8)

    assert model.config._hawp_global_default_r_k == 8
    assert model.config._hawp_global_default_r_v == 8
    assert model.config._hawp_uses_per_layer_ranks is False
