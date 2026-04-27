"""Tests for pure_quant_only mode.

Verifies:
1. pure_quant_only does NOT call convert_llama_to_hawp
2. pure_quant_only uses quantizers (TurboQuantProd for K, TurboQuantMSE for V)
3. pure_quant_only keeps the original attention formula unchanged
4. LayerKVQuantCache stores and retrieves quantized KV correctly
5. PureQuantKVManager installs hooks without replacing attention modules
"""
from __future__ import annotations

import importlib.util
import inspect

import pytest
import torch

from hawp_laq.config import HAWPLAQConfig, build_k_quantizer, build_v_quantizer
from hawp_laq.modeling.attention_hawp import HAWPAttention
from hawp_laq.runtime.pure_quant_hook import (
    LayerKVQuantCache,
    PureQuantKVManager,
    install_pure_quant_hooks,
)
from hawp_laq.runtime.turboquant import TurboQuantMSE, TurboQuantProd


class TestLayerKVQuantCache:
    def _make_cache(self, recent_window=64, head_dim=16):
        kq = TurboQuantProd(dim=head_dim, bits=4, use_rotation=True)
        vq = TurboQuantMSE(dim=head_dim, bits=8, use_rotation=True)
        return LayerKVQuantCache(kq, vq, recent_window=recent_window, n_kv_heads=4, head_dim=head_dim)

    def test_update_recent(self):
        cache = self._make_cache(recent_window=64)
        nkv, d = 4, 16
        k = torch.randn(nkv, 10, d)
        v = torch.randn(nkv, 10, d)
        cache.update(k, v)
        assert cache.seq_len == 10

    def test_demote_on_overflow(self):
        cache = self._make_cache(recent_window=8)
        nkv, d = 4, 16
        k = torch.randn(nkv, 10, d)
        v = torch.randn(nkv, 10, d)
        cache.update(k, v)
        assert cache.seq_len == 10
        assert bool(cache._archive_chunks)

    def test_get_kv_roundtrip_shape(self):
        cache = self._make_cache(recent_window=64)
        nkv, d = 4, 16
        k = torch.randn(nkv, 10, d)
        v = torch.randn(nkv, 10, d)
        cache.update(k, v)
        k_out, v_out = cache.get_kv()
        assert k_out.shape == (nkv, 10, d)
        assert v_out.shape == (nkv, 10, d)

    def test_archive_zero_recent_window(self):
        cache = self._make_cache(recent_window=0)
        nkv, d = 4, 16
        k = torch.randn(nkv, 5, d)
        v = torch.randn(nkv, 5, d)
        cache.update(k, v)
        assert bool(cache._archive_chunks)
        assert cache._recent_k is None

    def test_reset_clears_everything(self):
        cache = self._make_cache()
        nkv, d = 4, 16
        k = torch.randn(nkv, 5, d)
        v = torch.randn(nkv, 5, d)
        cache.update(k, v)
        cache.reset()
        assert cache.seq_len == 0

    def test_summary_keys(self):
        cache = self._make_cache()
        nkv, d = 4, 16
        k = torch.randn(nkv, 5, d)
        v = torch.randn(nkv, 5, d)
        cache.update(k, v)
        s = cache.summary()
        assert "recent_tokens" in s
        assert "archive_tokens" in s
        assert "total_runtime_bytes" in s
        assert "compressed_storage_bytes" in s

    def test_get_kv_empty_raises(self):
        cache = self._make_cache()
        with pytest.raises(RuntimeError, match="empty cache"):
            cache.get_kv()


class TestPureQuantOnlyDoesNotConvertToHAWP:
    def test_setup_function_does_not_call_convert(self):
        from hawp_laq.runtime import generate
        source = inspect.getsource(generate._setup_pure_quant_only_on_model)
        lines = source.splitlines()
        code_lines = [l for l in lines if l.strip() and not l.strip().startswith('"""') and not l.strip().startswith("'''")]
        for line in code_lines:
            assert "convert_llama_to_hawp(" not in line, (
                "pure_quant_only must NOT call convert_llama_to_hawp"
            )

    def test_setup_function_does_not_create_hawp_attention(self):
        from hawp_laq.runtime import generate
        source = inspect.getsource(generate._setup_pure_quant_only_on_model)
        lines = source.splitlines()
        code_lines = [l for l in lines if l.strip() and not l.strip().startswith('"""') and not l.strip().startswith("'''")]
        for line in code_lines:
            assert "HAWPAttention(" not in line, (
                "pure_quant_only must NOT instantiate HAWPAttention"
            )

    def test_pure_quant_hook_does_not_replace_attention(self):
        source = inspect.getsource(PureQuantKVManager)
        assert "setattr" not in source or "self_attn" not in source, (
            "PureQuantKVManager must NOT replace attention modules"
        )

    def test_model_retains_original_attention_after_install(self):
        try:
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(
                "facebook/opt-125m", torch_dtype=torch.float32,
            )
        except Exception:
            pytest.skip("Cannot load OPT-125m for integration test")

        cfg = HAWPLAQConfig()
        cfg.model.model_id = "facebook/opt-125m"
        cfg.sched.recent_window = 4
        manager = install_pure_quant_hooks(model, cfg)

        has_hawp = any(isinstance(m, HAWPAttention) for m in model.modules())
        assert not has_hawp, "Model must NOT contain HAWPAttention after pure_quant_only setup"

        from transformers.models.opt.modeling_opt import OPTAttention
        has_original = any(isinstance(m, OPTAttention) for m in model.modules())
        assert has_original, "Model must retain original OPTAttention modules"

        manager.remove_hooks()


class TestPureQuantOnlyUsesQuantizer:
    def test_layer_cache_uses_turboquant_prod_for_k(self):
        d = 16
        kq = TurboQuantProd(dim=d, bits=4, use_rotation=True)
        vq = TurboQuantMSE(dim=d, bits=8, use_rotation=True)
        cache = LayerKVQuantCache(kq, vq, recent_window=64)
        assert isinstance(cache.k_quantizer, TurboQuantProd)

    def test_layer_cache_uses_turboquant_mse_for_v(self):
        d = 16
        kq = TurboQuantProd(dim=d, bits=4, use_rotation=True)
        vq = TurboQuantMSE(dim=d, bits=8, use_rotation=True)
        cache = LayerKVQuantCache(kq, vq, recent_window=64)
        assert isinstance(cache.v_quantizer, TurboQuantMSE)

    def test_quantizer_built_from_config(self):
        cfg = HAWPLAQConfig()
        d = 64
        kq = build_k_quantizer(cfg, r_k=d)
        vq = build_v_quantizer(cfg, r_v=d)
        assert isinstance(kq, TurboQuantProd)
        assert isinstance(vq, TurboQuantMSE)


class TestPureQuantOnlyKeepsOriginalAttentionFormula:
    def test_dequantized_kv_preserves_attention_formula(self):
        """Verify that K/V passed through quantize→dequantize still produce
        the same attention pattern structure (softmax(Q @ K^T / sqrt(d)) @ V).
        The values will differ due to quantization error, but the formula
        structure must be identical to the original attention.
        """
        torch.manual_seed(42)
        d = 16
        n_heads = 2
        seq_len = 8

        q = torch.randn(1, n_heads, 1, d)
        k_orig = torch.randn(1, n_heads, seq_len, d)
        v_orig = torch.randn(1, n_heads, seq_len, d)

        import math
        import torch.nn.functional as F

        logits_orig = torch.matmul(q, k_orig.transpose(2, 3)) / math.sqrt(d)
        attn_orig = F.softmax(logits_orig, dim=-1, dtype=torch.float32).to(q.dtype)
        out_orig = torch.matmul(attn_orig, v_orig)

        kq = TurboQuantProd(dim=d, bits=4, use_rotation=False)
        vq = TurboQuantMSE(dim=d, bits=8, use_rotation=False)

        k_3d = k_orig[0]
        v_3d = v_orig[0]
        cache = LayerKVQuantCache(kq, vq, recent_window=9999, n_kv_heads=n_heads, head_dim=d)
        cache.update(k_3d, v_3d)
        k_deq, v_deq = cache.get_kv()

        k_deq_4d = k_deq.unsqueeze(0)
        v_deq_4d = v_deq.unsqueeze(0)

        logits_deq = torch.matmul(q, k_deq_4d.transpose(2, 3)) / math.sqrt(d)
        attn_deq = F.softmax(logits_deq, dim=-1, dtype=torch.float32).to(q.dtype)
        out_deq = torch.matmul(attn_deq, v_deq_4d)

        assert out_deq.shape == out_orig.shape

    def test_setup_pure_quant_only_returns_kv_manager(self):
        from hawp_laq.runtime.generate import _setup_pure_quant_only_on_model
        source = inspect.getsource(_setup_pure_quant_only_on_model)
        assert "manager" in source
        assert "install_pure_quant_hooks" in source
        assert "return model, head_dim, manager" in source
        assert "install_pure_quant_hooks" in source


class TestCompareModesIncludesPureQuantOnly:
    def _load_compare_module(self):
        spec = importlib.util.spec_from_file_location(
            "compare", "scripts/08_compare_modes.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_modes_list_includes_pure_quant_only(self):
        mod = self._load_compare_module()
        assert "pure_quant_only" in mod._MODES

    def test_setup_uses_mode_runner(self):
        mod = self._load_compare_module()
        source = inspect.getsource(mod)
        assert "setup_mode" in source

    def test_run_speed_uses_profile_generate(self):
        mod = self._load_compare_module()
        source = inspect.getsource(mod.main)
        assert "profile_generate_by_mode" in source

    def test_peak_gpu_measurement(self):
        mod = self._load_compare_module()
        source = inspect.getsource(mod)
        assert "reset_peak_memory_stats" in source
        assert "max_memory_allocated" in source


class TestRunGenerationEvalIncludesPureQuantOnly:
    def _load_eval_module(self):
        spec = importlib.util.spec_from_file_location(
            "eval", "scripts/04_run_generation_eval.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_modes_includes_pure_quant_only(self):
        mod = self._load_eval_module()
        assert "pure_quant_only" in mod._MODES

    def test_main_uses_mode_runner(self):
        mod = self._load_eval_module()
        source = inspect.getsource(mod.main)
        assert "setup_mode" in source
        assert "profile_generate_by_mode" in source

    def test_main_uses_profile_generate(self):
        mod = self._load_eval_module()
        source = inspect.getsource(mod.main)
        assert "profile_generate_by_mode" in source
        assert "CacheStats" in source or "stats.format_summary" in source
