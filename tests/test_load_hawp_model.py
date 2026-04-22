import pytest
from unittest.mock import patch, MagicMock
import torch.nn as nn
from types import SimpleNamespace


def _make_fake_config(hidden_size=64, num_heads=4):
    return SimpleNamespace(
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        num_key_value_heads=num_heads,
        max_position_embeddings=2048,
        rope_theta=10000.0,
        model_type="opt",
        enable_bias=False,
        attention_dropout=0.0,
    )


class OPTDecoderLayer(nn.Module):
    def __init__(self, hidden_size=64, num_heads=4):
        super().__init__()
        self.self_attn = _FakeOPTAttention(hidden_size, num_heads)


class _FakeOPTAttention(nn.Module):
    def __init__(self, hidden_size=64, num_heads=4):
        super().__init__()
        self.config = _make_fake_config(hidden_size, num_heads)
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)


class _FakeModel(nn.Module):
    def __init__(self, hidden_size=64, num_heads=4, n_layers=1):
        super().__init__()
        self.config = _make_fake_config(hidden_size, num_heads)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(
            [OPTDecoderLayer(hidden_size, num_heads) for _ in range(n_layers)]
        )

    def to(self, *args, **kwargs):
        return self

    def eval(self):
        return self


def test_load_hawp_model_missing_ranks_raises_without_allow_default():
    from hawp_laq.modeling.modeling_llama_hawp import load_hawp_model

    with pytest.raises(ValueError, match="HAWPAttention requires explicit r_k and r_v"):
        with patch("transformers.AutoModelForCausalLM") as mock_model_cls, \
             patch("transformers.AutoTokenizer") as mock_tok_cls:
            mock_model_cls.from_pretrained.return_value = _FakeModel()
            mock_tok_cls.from_pretrained.return_value = MagicMock()
            load_hawp_model("fake-model", r_k=None, r_v=None, allow_default_full_rank=False)


def test_load_hawp_model_allow_default_full_rank():
    from hawp_laq.modeling.modeling_llama_hawp import load_hawp_model

    with patch("transformers.AutoModelForCausalLM") as mock_model_cls, \
         patch("transformers.AutoTokenizer") as mock_tok_cls:
        mock_model_cls.from_pretrained.return_value = _FakeModel()
        mock_tok_cls.from_pretrained.return_value = MagicMock()

        model, tokenizer = load_hawp_model("fake-model", allow_default_full_rank=True)
        assert tokenizer is not None

        from hawp_laq.modeling.attention_hawp import HAWPAttention
        hawp_layers = [m for m in model.modules() if isinstance(m, HAWPAttention)]
        assert len(hawp_layers) == 1
        assert hawp_layers[0].r_k == 16
        assert hawp_layers[0].r_v == 16
