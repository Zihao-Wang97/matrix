from __future__ import annotations

import pytest

from hawp_laq.config import load_config
from hawp_laq.modeling.attention_hawp import HAWPAttention
from hawp_laq.modeling.modeling_llama_hawp import convert_llama_to_hawp

_CONFIG_PATH = "configs/dev_local.yaml"


@pytest.fixture(scope="module")
def converted_model():
    from transformers import AutoModelForCausalLM

    cfg = load_config(_CONFIG_PATH)
    model_id = cfg.model.model_id
    r_k = cfg.projector.r_k or cfg.projector.rank
    r_v = cfg.projector.r_v or cfg.projector.rank

    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto")
    model = convert_llama_to_hawp(model, r_k=r_k, r_v=r_v)
    return model


def test_at_least_one_hawp_attention(converted_model):
    found = any(isinstance(m, HAWPAttention) for _, m in converted_model.named_modules())
    assert found, "No HAWPAttention module found after conversion"


def test_layer_idx_unique(converted_model):
    indices = [
        m.layer_idx for _, m in converted_model.named_modules()
        if isinstance(m, HAWPAttention)
    ]
    assert len(indices) == len(set(indices)), f"Duplicated layer_idx: {indices}"


def test_layer_idx_sequential(converted_model):
    indices = [
        m.layer_idx for _, m in converted_model.named_modules()
        if isinstance(m, HAWPAttention)
    ]
    expected = list(range(len(indices)))
    assert sorted(indices) == expected, f"Expected {expected}, got {sorted(indices)}"
