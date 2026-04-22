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


def test_load_projectors_strict_shape_check():
    import torch.nn as nn

    class _FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn0 = _make_attn(r_k=8, r_v=8, layer_idx=0)
            self.attn1 = _make_attn(r_k=8, r_v=8, layer_idx=1)

    model = _FakeModel()
    tmp = Path("/tmp/test_strict_projector")
    tmp.mkdir(parents=True, exist_ok=True)
    save_projectors(model, tmp)

    model2 = _FakeModel()

    bad_dir = tmp / "bad"
    bad_dir.mkdir(parents=True, exist_ok=True)
    layer_dir = bad_dir / "layer_0"
    layer_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"p_k": torch.randn(64, 32), "p_v": torch.randn(64, 32)}, layer_dir / "projector.pt")

    with pytest.raises(ValueError, match="p_k shape"):
        load_projectors(model2, bad_dir, strict=True)

    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
