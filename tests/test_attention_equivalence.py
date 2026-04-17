import torch
import pytest
from types import SimpleNamespace
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from hawp_laq.modeling.attention_hawp import HAWPAttention, _get_attn_config
from hawp_laq.modeling.modeling_llama_hawp import convert_llama_to_hawp
from hawp_laq.modeling.rope_utils import rotate_half, apply_rotary_pos_emb

_MODEL_ID = "facebook/opt-125m"


def _get_first_attn(model):
    for name, module in model.named_modules():
        cls_name = type(module).__name__
        if "Attention" in cls_name and "Sdpa" not in cls_name:
            return module
    return None


class TestRopeUtils:
    def test_rotate_half_shape(self):
        x = torch.randn(2, 4, 8, 32)
        out = rotate_half(x)
        assert out.shape == x.shape

    def test_rotate_half_value(self):
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        out = rotate_half(x)
        expected = torch.tensor([[-3.0, -4.0, 1.0, 2.0]])
        assert torch.allclose(out, expected)

    def test_apply_rotary_pos_emb_shape(self):
        q = torch.randn(1, 4, 8, 16)
        k = torch.randn(1, 4, 8, 16)
        cos = torch.randn(1, 1, 8, 16)
        sin = torch.randn(1, 1, 8, 16)
        q_out, k_out = apply_rotary_pos_emb(q, k, cos, sin)
        assert q_out.shape == q.shape
        assert k_out.shape == k.shape


class TestGetAttnConfig:
    def test_from_attn_module_config(self):
        model = AutoModelForCausalLM.from_pretrained(_MODEL_ID)
        orig = _get_first_attn(model)
        cfg = _get_attn_config(orig, model=None)
        assert cfg.hidden_size == 768
        assert cfg.num_attention_heads == 12

    def test_from_model_config(self):
        model = AutoModelForCausalLM.from_pretrained(_MODEL_ID)
        cfg = _get_attn_config(None, model=model)
        assert cfg.hidden_size == 768

    def test_fallback_default(self):
        cfg = _get_attn_config(None, model=None)
        assert cfg.hidden_size == 768
        assert cfg.num_attention_heads == 12

    def test_infer_from_proj_shapes(self):
        class FakeAttn(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(512, 512, bias=True)
                self.k_proj = nn.Linear(512, 512)
                self.v_proj = nn.Linear(512, 512)
                self.out_proj = nn.Linear(512, 512)
        cfg = _get_attn_config(FakeAttn(), model=None)
        assert cfg.hidden_size == 512
        assert cfg.enable_bias is True


class TestHAWPAttentionConstruction:
    def test_from_attention_identity_projectors(self):
        model = AutoModelForCausalLM.from_pretrained(_MODEL_ID)
        orig = _get_first_attn(model)
        hawp = HAWPAttention.from_llama_attention(orig, layer_idx=0)
        assert hawp.p_k.shape == (hawp.head_dim, hawp.head_dim)
        assert hawp.p_v.shape == (hawp.head_dim, hawp.head_dim)
        assert torch.allclose(hawp.p_k.float(), torch.eye(hawp.head_dim))
        assert torch.allclose(hawp.p_v.float(), torch.eye(hawp.head_dim))
        assert hawp.gamma.item() == pytest.approx(1.0)

    def test_weights_copied(self):
        model = AutoModelForCausalLM.from_pretrained(_MODEL_ID)
        orig = _get_first_attn(model)
        hawp = HAWPAttention.from_llama_attention(orig, layer_idx=0)
        assert torch.equal(hawp.q_proj.weight.data, hawp._src_weights["q_proj"])
        assert torch.equal(hawp.o_proj.weight.data, hawp._src_weights["o_proj"])

    def test_forward_shape(self):
        model = AutoModelForCausalLM.from_pretrained(_MODEL_ID)
        orig = _get_first_attn(model)
        hawp = HAWPAttention.from_llama_attention(orig, layer_idx=0)
        hawp.eval()
        dtype = hawp.q_proj.weight.dtype
        x = torch.randn(1, 4, hawp.hidden_size, dtype=dtype)
        with torch.no_grad():
            out = hawp(x, attention_mask=None)
        attn_out = out[0]
        assert attn_out.shape == (1, 4, hawp.hidden_size)

    def test_from_attention_model_none(self):
        model = AutoModelForCausalLM.from_pretrained(_MODEL_ID)
        orig = _get_first_attn(model)
        hawp = HAWPAttention.from_attention(orig, model=None, layer_idx=0)
        assert hawp.hidden_size == 768
        assert hawp.num_heads == 12
        assert torch.allclose(hawp.p_k.float(), torch.eye(hawp.head_dim))

    def test_from_attention_with_model(self):
        model = AutoModelForCausalLM.from_pretrained(_MODEL_ID)
        orig = _get_first_attn(model)
        hawp = HAWPAttention.from_attention(orig, model=model, layer_idx=0)
        assert hawp.hidden_size == 768

    def test_construct_without_real_attn(self):
        cfg = SimpleNamespace(
            hidden_size=256, num_attention_heads=4, num_key_value_heads=4,
            max_position_embeddings=512, rope_theta=10000.0, model_type="",
            enable_bias=False, attention_dropout=0.0,
        )
        hawp = HAWPAttention(cfg)
        assert hawp.head_dim == 64
        assert hawp.r_k == 64
        assert hawp.r_v == 64


class TestFullRankForwardEquivalence:
    @pytest.fixture(scope="class")
    def models(self):
        model_orig = AutoModelForCausalLM.from_pretrained(_MODEL_ID, attn_implementation="eager")
        model_orig.eval()

        model_hawp = AutoModelForCausalLM.from_pretrained(_MODEL_ID, attn_implementation="eager")
        model_hawp = convert_llama_to_hawp(model_hawp)
        model_hawp.eval()
        return model_orig, model_hawp

    @pytest.fixture(scope="class")
    def tokenizer(self):
        return AutoTokenizer.from_pretrained(_MODEL_ID)

    def test_logits_close(self, models, tokenizer):
        model_orig, model_hawp = models
        inputs = tokenizer("Hello world", return_tensors="pt")

        with torch.no_grad():
            logits_orig = model_orig(**inputs).logits.float()
            logits_hawp = model_hawp(**inputs).logits.float()

        max_diff = (logits_orig - logits_hawp).abs().max().item()
        rel_diff = (logits_orig - logits_hawp).abs().mean() / logits_orig.abs().mean()
        assert rel_diff < 0.01, f"logits relative diff = {rel_diff}, max diff = {max_diff}"

    def test_generation_same_first_token(self, models, tokenizer):
        model_orig, model_hawp = models
        inputs = tokenizer("The capital of France is", return_tensors="pt")

        with torch.no_grad():
            out_orig = model_orig.generate(**inputs, max_new_tokens=1, do_sample=False)
            out_hawp = model_hawp.generate(**inputs, max_new_tokens=1, do_sample=False)

        assert out_orig[0, -1].item() == out_hawp[0, -1].item()


class TestConversionMarks:
    def test_config_marked_after_conversion(self):
        model = AutoModelForCausalLM.from_pretrained(_MODEL_ID)
        model = convert_llama_to_hawp(model)
        assert model.config._hawp_converted is True

    def test_all_layers_replaced(self):
        model = AutoModelForCausalLM.from_pretrained(_MODEL_ID)
        model = convert_llama_to_hawp(model)
        from hawp_laq.offline.hooks import _find_attention_modules
        attns = _find_attention_modules(model)
        for _, attn in attns:
            assert isinstance(attn, HAWPAttention)
