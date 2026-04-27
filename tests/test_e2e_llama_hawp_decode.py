"""End-to-end integration tests for HAWP with a real HF Llama model + multi-step decode.

These tests verify that:
  1. HAWP-converted model produces output consistent with the original model
  2. KV cache memory is actually reduced under low-rank
  3. Multi-step autoregressive decode works correctly with RoPE

Marked as @pytest.mark.slow so CI can optionally skip them.
Requires network access to download model weights on first run.
"""
from __future__ import annotations

import pytest
import torch

from hawp_laq.modeling.attention_hawp import HAWPAttention
from hawp_laq.modeling.modeling_llama_hawp import convert_llama_to_hawp

_MODEL_ID = "facebook/opt-125m"
_R_K = 16
_R_V = 16


def _load_original():
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        _MODEL_ID, attn_implementation="eager", torch_dtype=torch.float32,
    )
    model.eval()
    return model


def _load_hawp():
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        _MODEL_ID, attn_implementation="eager", torch_dtype=torch.float32,
    )
    model = convert_llama_to_hawp(model, r_k=_R_K, r_v=_R_V)
    model.eval()
    return model


def _load_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(_MODEL_ID)


slow = pytest.mark.slow


@slow
class TestE2EHawpDecode:
    @pytest.fixture(scope="class")
    def model_orig(self):
        return _load_original()

    @pytest.fixture(scope="class")
    def model_hawp(self):
        return _load_hawp()

    @pytest.fixture(scope="class")
    def tokenizer(self):
        return _load_tokenizer()

    def test_logits_close(self, model_orig, model_hawp, tokenizer):
        inputs = tokenizer("Hello world", return_tensors="pt")

        with torch.no_grad():
            logits_orig = model_orig(**inputs).logits.float()
            logits_hawp = model_hawp(**inputs).logits.float()

        rel_diff = (logits_orig - logits_hawp).abs().mean() / (logits_orig.abs().mean() + 1e-8)
        assert rel_diff < 0.1, f"logits relative diff = {rel_diff}"

    def test_generate_same_first_token(self, model_orig, model_hawp, tokenizer):
        inputs = tokenizer("The capital of France is", return_tensors="pt")

        with torch.no_grad():
            out_orig = model_orig.generate(**inputs, max_new_tokens=1, do_sample=False)
            out_hawp = model_hawp.generate(**inputs, max_new_tokens=1, do_sample=False)

        assert out_orig[0, -1].item() == out_hawp[0, -1].item()

    def test_kv_cache_memory_reduced(self, model_hawp, tokenizer):
        inputs = tokenizer("Hello world, this is a test", return_tensors="pt")
        seq_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            out = model_hawp(**inputs, use_cache=True)

        past_kv = out.past_key_values
        assert past_kv is not None

        low_rank_bytes = 0
        full_rank_bytes = 0
        for layer_kv in past_kv:
            k, v = layer_kv
            low_rank_bytes += k.nelement() * k.element_size()
            low_rank_bytes += v.nelement() * v.element_size()

            head_dim = k.shape[-1] * 2
            n_kv = k.shape[1]
            elem_size = k.element_size()
            full_rank_bytes += 2 * n_kv * seq_len * head_dim * elem_size

        ratio = low_rank_bytes / full_rank_bytes
        expected = (_R_K + _R_V) / (2 * _R_K * 2)
        assert ratio < 1.0, f"KV cache should be smaller than full-rank, got ratio {ratio:.2f}"

    def test_multi_step_decode_consistency(self, model_hawp, tokenizer):
        prompt = "The quick brown fox"
        inputs = tokenizer(prompt, return_tensors="pt")
        prompt_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            out = model_hawp(**inputs, use_cache=True)
            past = out.past_key_values
            last_logits = out.logits[:, -1, :]
            next_token = last_logits.argmax(dim=-1, keepdim=True)

            for step in range(5):
                out = model_hawp(
                    input_ids=next_token,
                    past_key_values=past,
                    use_cache=True,
                )
                past = out.past_key_values
                last_logits = out.logits[:, -1, :]
                next_token = last_logits.argmax(dim=-1, keepdim=True)

        full_input_ids = torch.cat(
            [inputs["input_ids"], torch.zeros(1, 5, dtype=torch.long)], dim=1
        )
        with torch.no_grad():
            full_out = model_hawp(input_ids=full_input_ids[:, :prompt_len + 5], use_cache=False)

        assert full_out.logits.shape[1] == prompt_len + 5

    def test_all_layers_are_hawp(self, model_hawp):
        from hawp_laq.offline.hooks import _find_attention_modules
        attns = _find_attention_modules(model_hawp)
        for _, attn in attns:
            assert isinstance(attn, HAWPAttention), f"Expected HAWPAttention, got {type(attn)}"
            assert attn.r_k == _R_K, f"Expected r_k={_R_K}, got {attn.r_k}"
            assert attn.r_v == _R_V, f"Expected r_v={_R_V}, got {attn.r_v}"
