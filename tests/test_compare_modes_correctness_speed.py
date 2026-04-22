from __future__ import annotations

import pytest
import torch

from hawp_laq.modeling.attention_hawp import HAWPAttention
from hawp_laq.runtime.generate import stepwise_greedy_generate


def _make_hawp_quant_all(n_heads=2, head_dim=16, r_k=8, r_v=8):
    from types import SimpleNamespace
    from hawp_laq.runtime.turboquant import TurboQuantMSE

    config = SimpleNamespace(
        hidden_size=n_heads * head_dim,
        num_attention_heads=n_heads,
        num_key_value_heads=n_heads,
        max_position_embeddings=2048,
        rope_theta=10000.0,
        model_type="opt",
        enable_bias=False,
        attention_dropout=0.0,
    )
    attn = HAWPAttention(config, r_k=r_k, r_v=r_v)
    k_q = TurboQuantMSE(dim=r_k, bits=4, group_size=16, use_rotation=False)
    v_q = TurboQuantMSE(dim=r_v, bits=4, group_size=16, use_rotation=False)
    attn.setup_quant_cache(k_q, v_q, recent_window=0)
    return attn


class TestStepwiseGreedyGenerate:
    def test_returns_list_of_strings(self):
        from unittest.mock import MagicMock
        from types import SimpleNamespace

        n_vocab = 100
        seq_len = 5
        max_new = 3

        model = MagicMock()
        logits = torch.zeros(1, 1, n_vocab)
        logits[0, 0, 0] = 1.0
        model.device = torch.device("cpu")
        model.return_value = MagicMock(logits=logits)

        tokenizer = MagicMock()
        tokenizer.return_value = SimpleNamespace(
            input_ids=torch.ones(1, seq_len, dtype=torch.long)
        )
        tokenizer.decode.return_value = "hello world"

        results = stepwise_greedy_generate(model, tokenizer, ["test"], max_new)
        assert isinstance(results, list)
        assert len(results) == 1
        assert isinstance(results[0], str)

    def test_reset_cache_fn_called_between_prompts(self):
        from unittest.mock import MagicMock
        from types import SimpleNamespace

        model = MagicMock()
        logits = torch.zeros(1, 1, 50)
        logits[0, 0, 0] = 1.0
        model.device = torch.device("cpu")
        model.return_value = MagicMock(logits=logits)

        tokenizer = MagicMock()
        tokenizer.return_value = SimpleNamespace(
            input_ids=torch.ones(1, 3, dtype=torch.long)
        )
        tokenizer.decode.return_value = "out"

        reset_fn = MagicMock()
        stepwise_greedy_generate(model, tokenizer, ["a", "b"], 2, reset_cache_fn=reset_fn)
        assert reset_fn.call_count == 2


class TestCompareScriptSeparatesCorrectnessAndSpeed:
    def test_correctness_and_speed_helpers_exist(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "compare", "scripts/08_compare_modes.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        assert hasattr(mod, "_run_correctness")
        assert hasattr(mod, "_run_speed")

    def test_correctness_uses_stepwise_greedy(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "compare", "scripts/08_compare_modes.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        import inspect
        source = inspect.getsource(mod._run_correctness)
        assert "stepwise_greedy_generate" in source

    def test_speed_uses_production_path(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "compare", "scripts/08_compare_modes.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        import inspect
        source = inspect.getsource(mod._run_speed)
        assert "generate_text" in source or "generate_hawp_quant" in source

    def test_output_has_correctness_and_speed_labels(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "compare", "scripts/08_compare_modes.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        import inspect
        source = inspect.getsource(mod.main)
        assert "CORRECTNESS COMPARISON" in source
        assert "SPEED COMPARISON" in source
