import torch
import pytest
import torch.nn as nn
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

from hawp_laq.modeling.attention_hawp import HAWPAttention, _resolve_compute_dtype
from hawp_laq.modeling.modeling_llama_hawp import _align_hawp_params_device_dtype, convert_llama_to_hawp


def _make_config(hidden_size=64, num_heads=4, model_type="opt"):
    return SimpleNamespace(
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        num_key_value_heads=num_heads,
        max_position_embeddings=2048,
        rope_theta=10000.0,
        model_type=model_type,
        enable_bias=False,
        attention_dropout=0.0,
    )


class _FakeAttention(nn.Module):
    def __init__(self, hidden_size=64, num_heads=4, dtype=torch.float32):
        super().__init__()
        self.config = _make_config(hidden_size, num_heads)
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False).to(dtype=dtype)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False).to(dtype=dtype)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False).to(dtype=dtype)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False).to(dtype=dtype)


class OPTDecoderLayer(nn.Module):
    def __init__(self, hidden_size=64, num_heads=4, dtype=torch.float32):
        super().__init__()
        self.self_attn = _FakeAttention(hidden_size, num_heads, dtype)


class _FakeModel(nn.Module):
    def __init__(self, hidden_size=64, num_heads=4, n_layers=2, dtype=torch.float32):
        super().__init__()
        self.config = _make_config(hidden_size, num_heads)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(
            [OPTDecoderLayer(hidden_size, num_heads, dtype) for _ in range(n_layers)]
        )

    def to(self, *args, **kwargs):
        return super().to(*args, **kwargs)

    def eval(self):
        return self


def test_from_attention_p_k_p_v_inherit_source_dtype_float16():
    attn = _FakeAttention(dtype=torch.float16)
    hawp = HAWPAttention.from_attention(attn, layer_idx=0, r_k=8, r_v=8)
    assert hawp.p_k.dtype == torch.float16, f"p_k dtype {hawp.p_k.dtype} != float16"
    assert hawp.p_v.dtype == torch.float16, f"p_v dtype {hawp.p_v.dtype} != float16"
    assert hawp.gamma.dtype == torch.float16, f"gamma dtype {hawp.gamma.dtype} != float16"


def test_from_attention_p_k_p_v_inherit_source_dtype_bfloat16():
    attn = _FakeAttention(dtype=torch.bfloat16)
    hawp = HAWPAttention.from_attention(attn, layer_idx=0, r_k=8, r_v=8)
    assert hawp.p_k.dtype == torch.bfloat16, f"p_k dtype {hawp.p_k.dtype} != bfloat16"
    assert hawp.p_v.dtype == torch.bfloat16, f"p_v dtype {hawp.p_v.dtype} != bfloat16"
    assert hawp.gamma.dtype == torch.bfloat16, f"gamma dtype {hawp.gamma.dtype} != bfloat16"


def test_from_attention_p_k_p_v_inherit_source_dtype_float32():
    attn = _FakeAttention(dtype=torch.float32)
    hawp = HAWPAttention.from_attention(attn, layer_idx=0, r_k=8, r_v=8)
    assert hawp.p_k.dtype == torch.float32
    assert hawp.p_v.dtype == torch.float32
    assert hawp.gamma.dtype == torch.float32


def test_init_dtype_parameter_respected():
    config = _make_config()
    hawp = HAWPAttention(config, r_k=8, r_v=8, dtype=torch.float16)
    assert hawp.p_k.dtype == torch.float16
    assert hawp.p_v.dtype == torch.float16
    assert hawp.gamma.dtype == torch.float16


def test_init_dtype_default_is_float32():
    config = _make_config()
    hawp = HAWPAttention(config, r_k=8, r_v=8)
    assert hawp.p_k.dtype == torch.float32
    assert hawp.p_v.dtype == torch.float32
    assert hawp.gamma.dtype == torch.float32


def test_low_rank_forward_no_dtype_mismatch_fp16():
    attn = _FakeAttention(dtype=torch.float16)
    hawp = HAWPAttention.from_attention(attn, layer_idx=0, r_k=8, r_v=8)
    bsz, n_heads, q_len, head_dim = 1, 4, 3, 16
    q = torch.randn(bsz, n_heads, q_len, head_dim, dtype=torch.float16)
    k = torch.randn(bsz, n_heads, q_len, head_dim, dtype=torch.float16)
    v = torch.randn(bsz, n_heads, q_len, head_dim, dtype=torch.float16)
    q_lat = q @ hawp.p_k[:, :8].to(q.dtype)
    k_lat = k @ hawp.p_k[:, :8].to(k.dtype)
    v_lat = v @ hawp.p_v[:, :8].to(v.dtype)
    assert q_lat.dtype == torch.float16
    assert k_lat.dtype == torch.float16
    assert v_lat.dtype == torch.float16


def test_apply_pk_preserves_input_dtype():
    attn = _FakeAttention(dtype=torch.float16)
    hawp = HAWPAttention.from_attention(attn, layer_idx=0, r_k=8, r_v=8)
    k = torch.randn(1, 4, 3, 16, dtype=torch.float16)
    result = hawp._apply_pk(k)
    assert result.dtype == torch.float16, f"_apply_pk returned {result.dtype}"


def test_apply_pv_preserves_input_dtype():
    attn = _FakeAttention(dtype=torch.float16)
    hawp = HAWPAttention.from_attention(attn, layer_idx=0, r_k=8, r_v=8)
    v = torch.randn(1, 4, 3, 16, dtype=torch.float16)
    result = hawp._apply_pv(v)
    assert result.dtype == torch.float16, f"_apply_pv returned {result.dtype}"


def test_align_hawp_params_device_dtype_per_layer():
    model = _FakeModel(n_layers=2, dtype=torch.float16)
    from hawp_laq.modeling.modeling_llama_hawp import convert_llama_to_hawp
    model = convert_llama_to_hawp(model, r_k=8, r_v=8)
    _align_hawp_params_device_dtype(model)
    for module in model.modules():
        if isinstance(module, HAWPAttention):
            assert module.p_k.dtype == torch.float16, f"p_k dtype {module.p_k.dtype}"
            assert module.p_v.dtype == torch.float16, f"p_v dtype {module.p_v.dtype}"
            assert module.gamma.dtype == torch.float16, f"gamma dtype {module.gamma.dtype}"


def test_align_hawp_params_with_explicit_compute_dtype():
    model = _Fake4bitModel(compute_dtype=torch.float16)
    model = _convert_4bit_model_manually(model, r_k=8, r_v=8)
    _align_hawp_params_device_dtype(model, compute_dtype=torch.bfloat16)
    for module in model.modules():
        if isinstance(module, HAWPAttention):
            assert module.p_k.dtype == torch.bfloat16, \
                f"p_k dtype {module.p_k.dtype} != bfloat16 when compute_dtype override provided"
            assert module.p_v.dtype == torch.bfloat16, \
                f"p_v dtype {module.p_v.dtype} != bfloat16 when compute_dtype override provided"
            assert module.gamma.dtype == torch.bfloat16, \
                f"gamma dtype {module.gamma.dtype} != bfloat16 when compute_dtype override provided"


def test_resolve_compute_dtype_from_linear4bit():
    proj = _FakeLinear4bit(64, 64, compute_dtype=torch.float16)
    assert _resolve_compute_dtype(proj) == torch.float16, \
        "Should resolve compute_dtype from Linear4bit.compute_dtype"


def test_resolve_compute_dtype_from_quant_state():
    proj = _FakeLinear4bit(64, 64, compute_dtype=torch.bfloat16)
    resolved = _resolve_compute_dtype(proj)
    assert resolved == torch.bfloat16, \
        f"Should resolve dtype from quant_state, got {resolved}"


def test_resolve_compute_dtype_fallback_for_uint8():
    class _FakeUint8Proj(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(
                torch.randint(0, 255, (64, 64), dtype=torch.uint8),
                requires_grad=False,
            )
    proj = _FakeUint8Proj()
    resolved = _resolve_compute_dtype(proj, fallback_dtype=torch.float16)
    assert resolved == torch.float16, \
        f"Should fallback to fallback_dtype for uint8 weight, got {resolved}"


def test_align_hawp_params_uses_per_layer_device():
    model = _FakeModel(n_layers=2, dtype=torch.float32)
    from hawp_laq.modeling.modeling_llama_hawp import convert_llama_to_hawp
    model = convert_llama_to_hawp(model, r_k=8, r_v=8)
    _align_hawp_params_device_dtype(model)
    for module in model.modules():
        if isinstance(module, HAWPAttention):
            q_device = module.q_proj.weight.device
            assert module.p_k.device == q_device, f"p_k device {module.p_k.device} != q_proj {q_device}"
            assert module.p_v.device == q_device, f"p_v device {module.p_v.device} != q_proj {q_device}"
            assert module.gamma.device == q_device, f"gamma device {module.gamma.device} != q_proj {q_device}"


def test_load_hawp_model_fp16_aligns_dtype_and_device():
    from hawp_laq.modeling.modeling_llama_hawp import load_hawp_model

    fake_model = _FakeModel(dtype=torch.float16)

    with patch("transformers.AutoModelForCausalLM") as mock_model_cls, \
         patch("transformers.AutoTokenizer") as mock_tok_cls:
        mock_model_cls.from_pretrained.return_value = fake_model
        mock_tok_cls.from_pretrained.return_value = MagicMock()

        model, tokenizer = load_hawp_model(
            "fake-model",
            r_k=8,
            r_v=8,
            torch_dtype=torch.float16,
            device="cpu",
            load_in_4bit=False,
        )
        for module in model.modules():
            if isinstance(module, HAWPAttention):
                assert module.p_k.dtype == torch.float16, f"p_k dtype {module.p_k.dtype}"
                assert module.p_v.dtype == torch.float16, f"p_v dtype {module.p_v.dtype}"
                assert module.gamma.dtype == torch.float16, f"gamma dtype {module.gamma.dtype}"


def test_full_rank_identity_path_no_dtype_cast():
    attn = _FakeAttention(dtype=torch.float16)
    hawp = HAWPAttention.from_attention(attn, layer_idx=0, r_k=16, r_v=16)
    k = torch.randn(1, 4, 3, 16, dtype=torch.float16)
    result = hawp._apply_pk(k)
    assert result is k, "Full-rank identity path should return input unchanged"


def test_forward_low_rank_preserves_dtype_fp16():
    attn = _FakeAttention(dtype=torch.float16)
    hawp = HAWPAttention.from_attention(attn, layer_idx=0, r_k=8, r_v=8)
    bsz, seq_len = 1, 4
    hidden = torch.randn(bsz, seq_len, 64, dtype=torch.float16)
    attention_mask = None
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    output, _, _ = hawp(
        hidden_states=hidden,
        attention_mask=attention_mask,
        position_ids=position_ids,
    )
    assert output.dtype == torch.float16, f"forward output dtype {output.dtype} != float16"


def test_forward_low_rank_preserves_dtype_bf16():
    attn = _FakeAttention(dtype=torch.bfloat16)
    hawp = HAWPAttention.from_attention(attn, layer_idx=0, r_k=8, r_v=8)
    bsz, seq_len = 1, 4
    hidden = torch.randn(bsz, seq_len, 64, dtype=torch.bfloat16)
    attention_mask = None
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    output, _, _ = hawp(
        hidden_states=hidden,
        attention_mask=attention_mask,
        position_ids=position_ids,
    )
    assert output.dtype == torch.bfloat16, f"forward output dtype {output.dtype} != bfloat16"


class _FakeLinear4bit(nn.Module):
    def __init__(self, in_features, out_features, compute_dtype=torch.float16):
        super().__init__()
        self.compute_dtype = compute_dtype
        self.weight = nn.Parameter(
            torch.randint(0, 255, (out_features, in_features), dtype=torch.uint8),
            requires_grad=False,
        )
        self.weight.quant_state = SimpleNamespace(dtype=compute_dtype)


class _Fake4bitAttention(nn.Module):
    def __init__(self, hidden_size=64, num_heads=4, compute_dtype=torch.float16):
        super().__init__()
        self.config = _make_config(hidden_size, num_heads)
        self.q_proj = _FakeLinear4bit(hidden_size, hidden_size, compute_dtype)
        self.k_proj = _FakeLinear4bit(hidden_size, hidden_size, compute_dtype)
        self.v_proj = _FakeLinear4bit(hidden_size, hidden_size, compute_dtype)
        self.o_proj = _FakeLinear4bit(hidden_size, hidden_size, compute_dtype)


class _Fake4bitDecoderLayer(nn.Module):
    def __init__(self, hidden_size=64, num_heads=4, compute_dtype=torch.float16):
        super().__init__()
        self.self_attn = _Fake4bitAttention(hidden_size, num_heads, compute_dtype)


class _Fake4bitModel(nn.Module):
    def __init__(self, hidden_size=64, num_heads=4, n_layers=2, compute_dtype=torch.float16):
        super().__init__()
        self.config = _make_config(hidden_size, num_heads)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([
            _Fake4bitDecoderLayer(hidden_size, num_heads, compute_dtype)
            for _ in range(n_layers)
        ])

    def eval(self):
        return self


def _convert_4bit_model_manually(model, r_k, r_v):
    for name, module in model.named_modules():
        cls_name = type(module).__name__
        if cls_name == "_Fake4bitDecoderLayer":
            hawp_attn = HAWPAttention.from_attention(
                module.self_attn, model=model, layer_idx=0, r_k=r_k, r_v=r_v,
            )
            module.self_attn = hawp_attn
    return model


def test_align_4bit_uses_compute_dtype_not_storage_dtype():
    model = _Fake4bitModel(compute_dtype=torch.float16)
    model = _convert_4bit_model_manually(model, r_k=8, r_v=8)
    _align_hawp_params_device_dtype(model)
    for module in model.modules():
        if isinstance(module, HAWPAttention):
            assert module.p_k.dtype == torch.float16, \
                f"p_k dtype {module.p_k.dtype} != float16, likely used uint8 storage dtype"
            assert module.p_v.dtype == torch.float16, \
                f"p_v dtype {module.p_v.dtype} != float16, likely used uint8 storage dtype"
            assert module.gamma.dtype == torch.float16, \
                f"gamma dtype {module.gamma.dtype} != float16, likely used uint8 storage dtype"


def test_align_4bit_bfloat16_compute_dtype():
    model = _Fake4bitModel(compute_dtype=torch.bfloat16)
    model = _convert_4bit_model_manually(model, r_k=8, r_v=8)
    _align_hawp_params_device_dtype(model)
    for module in model.modules():
        if isinstance(module, HAWPAttention):
            assert module.p_k.dtype == torch.bfloat16, \
                f"p_k dtype {module.p_k.dtype} != bfloat16"
            assert module.p_v.dtype == torch.bfloat16, \
                f"p_v dtype {module.p_v.dtype} != bfloat16"
            assert module.gamma.dtype == torch.bfloat16, \
                f"gamma dtype {module.gamma.dtype} != bfloat16"


def test_from_attention_4bit_uses_compute_dtype():
    attn = _Fake4bitAttention(compute_dtype=torch.float16)
    hawp = HAWPAttention.from_attention(attn, layer_idx=0, r_k=8, r_v=8)
    assert hawp.p_k.dtype == torch.float16, \
        f"p_k dtype {hawp.p_k.dtype} != float16, likely used uint8 storage dtype"
    assert hawp.p_v.dtype == torch.float16, \
        f"p_v dtype {hawp.p_v.dtype} != float16, likely used uint8 storage dtype"
    assert hawp.gamma.dtype == torch.float16, \
        f"gamma dtype {hawp.gamma.dtype} != float16, likely used uint8 storage dtype"


def test_from_attention_4bit_bfloat16_compute_dtype():
    attn = _Fake4bitAttention(compute_dtype=torch.bfloat16)
    hawp = HAWPAttention.from_attention(attn, layer_idx=0, r_k=8, r_v=8)
    assert hawp.p_k.dtype == torch.bfloat16, \
        f"p_k dtype {hawp.p_k.dtype} != bfloat16"
    assert hawp.p_v.dtype == torch.bfloat16, \
        f"p_v dtype {hawp.p_v.dtype} != bfloat16"
    assert hawp.gamma.dtype == torch.bfloat16, \
        f"gamma dtype {hawp.gamma.dtype} != bfloat16"


def test_load_hawp_model_4bit_true_aligns_to_compute_dtype():
    def _fake_convert(model, *a, **kw):
        return _convert_4bit_model_manually(model, r_k=kw.get('r_k', 8), r_v=kw.get('r_v', 8))

    with patch("hawp_laq.modeling.modeling_llama_hawp.convert_llama_to_hawp", side_effect=_fake_convert), \
         patch("transformers.AutoModelForCausalLM") as mock_model_cls, \
         patch("transformers.AutoTokenizer") as mock_tok_cls, \
         patch("transformers.BitsAndBytesConfig", return_value=MagicMock()):
        mock_model_cls.from_pretrained.return_value = _Fake4bitModel(compute_dtype=torch.float16)
        mock_tok_cls.from_pretrained.return_value = MagicMock()

        from hawp_laq.modeling.modeling_llama_hawp import load_hawp_model
        model, tokenizer = load_hawp_model(
            "fake-model",
            r_k=8,
            r_v=8,
            torch_dtype=torch.float16,
            device="cpu",
            load_in_4bit=True,
        )
        for module in model.modules():
            if isinstance(module, HAWPAttention):
                assert module.p_k.dtype == torch.float16, \
                    f"p_k dtype {module.p_k.dtype} != float16 in 4-bit load path"
                assert module.p_v.dtype == torch.float16, \
                    f"p_v dtype {module.p_v.dtype} != float16 in 4-bit load path"
                assert module.gamma.dtype == torch.float16, \
                    f"gamma dtype {module.gamma.dtype} != float16 in 4-bit load path"
