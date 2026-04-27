from __future__ import annotations

import torch
import pytest

from hawp_laq.runtime.scheduler import TokenBudgetScheduler, TokenState, SchedulerDecision
from hawp_laq.runtime.cache_manager import ModelCacheCoordinator
from hawp_laq.modeling.attention_hawp import HAWPAttention


# ======================================================================
# SchedulerDecision
# ======================================================================


class TestSchedulerDecision:
    def test_within_budget_no_drop(self):
        sched = TokenBudgetScheduler(total_budget=64, recent_window=8)
        sched.on_tokens(30)
        d = sched.rebalance()
        assert d.n_high == 8
        assert d.n_low == 22
        assert d.n_drop == 0

    def test_exceeds_budget_has_drop(self):
        sched = TokenBudgetScheduler(total_budget=32, recent_window=8)
        sched.on_tokens(50)
        d = sched.rebalance()
        assert d.n_high == 8
        assert d.n_low == 24
        assert d.n_drop == 50 - 8 - 24

    def test_small_seq_all_high(self):
        sched = TokenBudgetScheduler(total_budget=32, recent_window=8)
        sched.on_tokens(5)
        d = sched.rebalance()
        assert d.n_high == 5
        assert d.n_low == 0
        assert d.n_drop == 0

    def test_budget_equals_seq(self):
        sched = TokenBudgetScheduler(total_budget=32, recent_window=8)
        sched.on_tokens(32)
        d = sched.rebalance()
        assert d.n_high == 8
        assert d.n_low == 24
        assert d.n_drop == 0


class TestComputeDropCount:
    def test_incremental_drop(self):
        sched = TokenBudgetScheduler(total_budget=20, recent_window=4)
        sched.on_tokens(20)
        assert sched.compute_drop_count() == 0

        sched.on_new_token()
        assert sched.compute_drop_count() == 1
        sched.acknowledge_drop(1)

        sched.on_new_token()
        assert sched.compute_drop_count() == 1
        sched.acknowledge_drop(1)

    def test_incremental_drop_partial_ack(self):
        sched = TokenBudgetScheduler(total_budget=20, recent_window=4)
        sched.on_tokens(22)
        assert sched.compute_drop_count() == 2
        sched.acknowledge_drop(1)
        assert sched.compute_drop_count() == 1
        sched.acknowledge_drop(1)
        assert sched.compute_drop_count() == 0

    def test_batch_on_tokens(self):
        sched = TokenBudgetScheduler(total_budget=20, recent_window=4)
        sched.on_tokens(30)
        assert sched.compute_drop_count() == 10
        sched.acknowledge_drop(10)

    def test_reset_clears_drop_tracking(self):
        sched = TokenBudgetScheduler(total_budget=10, recent_window=2)
        sched.on_tokens(20)
        _ = sched.compute_drop_count()
        sched.acknowledge_drop(10)
        sched.reset()
        assert sched.seq_len == 0
        sched.on_tokens(5)
        assert sched.compute_drop_count() == 0


class TestGetState:
    def test_high_within_window(self):
        sched = TokenBudgetScheduler(total_budget=64, recent_window=8)
        sched.on_tokens(20)
        for pos in range(12, 20):
            assert sched.get_state(pos) == TokenState.HIGH

    def test_low_older_tokens(self):
        sched = TokenBudgetScheduler(total_budget=64, recent_window=8)
        sched.on_tokens(20)
        assert sched.get_state(0) == TokenState.LOW

    def test_drop_over_budget(self):
        sched = TokenBudgetScheduler(total_budget=16, recent_window=4)
        sched.on_tokens(30)
        assert sched.get_state(0) == TokenState.DROP


# ======================================================================
# HAWPAttention DROP methods
# ======================================================================


def _make_hawp_attn(r_k=8, r_v=8, recent_window=4):
    from types import SimpleNamespace
    config = SimpleNamespace(
        hidden_size=64,
        num_attention_heads=8,
        num_key_value_heads=8,
        max_position_embeddings=2048,
        rope_theta=10000.0,
        model_type="",
        enable_bias=False,
        attention_dropout=0.0,
    )
    attn = HAWPAttention(config, layer_idx=0, r_k=r_k, r_v=r_v)

    from hawp_laq.runtime.turboquant import TurboQuantProd, TurboQuantMSE
    kq = TurboQuantProd(dim=r_k, bits=4, use_rotation=False, group_size=8)
    vq = TurboQuantMSE(dim=r_v, bits=4, use_rotation=False, group_size=8)
    attn.setup_quant_cache(kq, vq, recent_window=recent_window)
    return attn


class TestDropOldestFromArchive:
    def test_no_archive_returns_zero(self):
        attn = _make_hawp_attn()
        assert attn.drop_oldest_from_archive(5) == 0

    def test_drop_from_archive(self):
        attn = _make_hawp_attn(recent_window=4)
        nkv = 8
        k_lat = torch.randn(1, nkv, 10, 8)
        v_lat = torch.randn(1, nkv, 10, 8)
        attn._quant_cache_append_to_archive(k_lat, v_lat)

        dropped = attn.drop_oldest_from_archive(3)
        assert dropped == 3
        remaining = sum(c.n_tokens for c in attn._quant_archive_chunks)
        assert remaining == 7
        assert bool(attn._quant_archive_chunks)

    def test_drop_more_than_archive(self):
        attn = _make_hawp_attn(recent_window=4)
        nkv = 8
        k_lat = torch.randn(1, nkv, 5, 8)
        v_lat = torch.randn(1, nkv, 5, 8)
        attn._quant_cache_append_to_archive(k_lat, v_lat)

        dropped = attn.drop_oldest_from_archive(10)
        assert dropped == 5
        assert not attn._quant_archive_chunks

    def test_drop_then_get_kv(self):
        attn = _make_hawp_attn(recent_window=4)
        nkv = 8
        k_lat = torch.randn(1, nkv, 10, 8)
        v_lat = torch.randn(1, nkv, 10, 8)
        attn._quant_cache_append_to_archive(k_lat, v_lat)

        attn.drop_oldest_from_archive(4)

        attn._quant_recent_k = torch.randn(nkv, 2, 8)
        attn._quant_recent_v = torch.randn(nkv, 2, 8)

        k, v = attn._quant_cache_get_kv()
        assert k is not None
        assert v is not None
        assert k.shape[1] == 6 + 2
        assert v.shape[1] == 6 + 2


class TestDropLeastImportantFromArchive:
    def test_no_archive_returns_zero(self):
        attn = _make_hawp_attn()
        assert attn.drop_least_important_from_archive(3) == 0

    def test_drop_by_norm(self):
        attn = _make_hawp_attn(recent_window=4)
        nkv = 8
        k_lat = torch.randn(1, nkv, 10, 8)
        v_lat = torch.randn(1, nkv, 10, 8)
        k_lat[0, :, 0, :] = 0.001
        k_lat[0, :, 1, :] = 0.001
        attn._quant_cache_append_to_archive(k_lat, v_lat)

        dropped = attn.drop_least_important_from_archive(2)
        assert dropped == 2
        remaining = sum(c.n_tokens for c in attn._quant_archive_chunks)
        assert remaining == 8


# ======================================================================
# ModelCacheCoordinator
# ======================================================================


class TestModelCacheCoordinator:
    def test_from_model(self):
        from types import SimpleNamespace
        config = SimpleNamespace(
            hidden_size=64,
            num_attention_heads=8,
            num_key_value_heads=8,
            max_position_embeddings=2048,
            rope_theta=10000.0,
            model_type="",
            enable_bias=False,
            attention_dropout=0.0,
        )
        model = torch.nn.Module()
        model.attn = HAWPAttention(config, layer_idx=0, r_k=8, r_v=8)

        from hawp_laq.runtime.turboquant import TurboQuantProd, TurboQuantMSE
        kq = TurboQuantProd(dim=8, bits=4, use_rotation=False, group_size=8)
        vq = TurboQuantMSE(dim=8, bits=4, use_rotation=False, group_size=8)
        model.attn.setup_quant_cache(kq, vq, recent_window=4)

        sched = TokenBudgetScheduler(total_budget=20, recent_window=4)
        coord = ModelCacheCoordinator.from_model(model, sched, drop_strategy="position")
        assert len(coord._layers) == 1

    def test_on_prefill_and_new_token(self):
        from types import SimpleNamespace
        config = SimpleNamespace(
            hidden_size=64,
            num_attention_heads=8,
            num_key_value_heads=8,
            max_position_embeddings=2048,
            rope_theta=10000.0,
            model_type="",
            enable_bias=False,
            attention_dropout=0.0,
        )
        attn = HAWPAttention(config, layer_idx=0, r_k=8, r_v=8)

        from hawp_laq.runtime.turboquant import TurboQuantProd, TurboQuantMSE
        kq = TurboQuantProd(dim=8, bits=4, use_rotation=False, group_size=8)
        vq = TurboQuantMSE(dim=8, bits=4, use_rotation=False, group_size=8)
        attn.setup_quant_cache(kq, vq, recent_window=4)

        nkv = 8
        k_lat = torch.randn(1, nkv, 30, 8)
        v_lat = torch.randn(1, nkv, 30, 8)
        attn._quant_cache_append_to_archive(k_lat, v_lat)

        class _FakeModel(torch.nn.Module):
            def __init__(self, attn):
                super().__init__()
                self.attn = attn

        sched = TokenBudgetScheduler(total_budget=20, recent_window=4)
        coord = ModelCacheCoordinator.from_model(
            _FakeModel(attn), sched, drop_strategy="position",
        )

        coord.on_prefill(50)
        d = sched.rebalance()
        assert d.n_drop > 0
        remaining_archive = sum(c.n_tokens for c in attn._quant_archive_chunks)
        assert remaining_archive <= d.n_low

    def test_deficit_carried_when_archive_empty(self):
        from types import SimpleNamespace
        config = SimpleNamespace(
            hidden_size=64,
            num_attention_heads=8,
            num_key_value_heads=8,
            max_position_embeddings=2048,
            rope_theta=10000.0,
            model_type="",
            enable_bias=False,
            attention_dropout=0.0,
        )
        attn = HAWPAttention(config, layer_idx=0, r_k=8, r_v=8)

        from hawp_laq.runtime.turboquant import TurboQuantProd, TurboQuantMSE
        kq = TurboQuantProd(dim=8, bits=4, use_rotation=False, group_size=8)
        vq = TurboQuantMSE(dim=8, bits=4, use_rotation=False, group_size=8)
        attn.setup_quant_cache(kq, vq, recent_window=4)

        class _FakeModel(torch.nn.Module):
            def __init__(self, attn):
                super().__init__()
                self.attn = attn

        sched = TokenBudgetScheduler(total_budget=10, recent_window=4)
        coord = ModelCacheCoordinator.from_model(
            _FakeModel(attn), sched, drop_strategy="position",
        )

        sched.on_tokens(30)
        coord._apply_drop()
        assert sched._prev_n_drop == 0
        assert sched.compute_drop_count() == 20

        nkv = 8
        k_lat = torch.randn(1, nkv, 10, 8)
        v_lat = torch.randn(1, nkv, 10, 8)
        attn._quant_cache_append_to_archive(k_lat, v_lat)

        coord._apply_drop()
        assert sched._prev_n_drop == 10
        remaining_archive = sum(c.n_tokens for c in attn._quant_archive_chunks)
        assert remaining_archive == 0

    def test_no_over_drop_with_asymmetric_archives(self):
        from types import SimpleNamespace
        config = SimpleNamespace(
            hidden_size=64,
            num_attention_heads=8,
            num_key_value_heads=8,
            max_position_embeddings=2048,
            rope_theta=10000.0,
            model_type="",
            enable_bias=False,
            attention_dropout=0.0,
        )
        attn0 = HAWPAttention(config, layer_idx=0, r_k=8, r_v=8)
        attn1 = HAWPAttention(config, layer_idx=1, r_k=8, r_v=8)

        from hawp_laq.runtime.turboquant import TurboQuantProd, TurboQuantMSE
        kq = TurboQuantProd(dim=8, bits=4, use_rotation=False, group_size=8)
        vq = TurboQuantMSE(dim=8, bits=4, use_rotation=False, group_size=8)
        attn0.setup_quant_cache(kq, vq, recent_window=4)
        attn1.setup_quant_cache(kq, vq, recent_window=4)

        nkv = 8
        attn0._quant_cache_append_to_archive(
            torch.randn(1, nkv, 20, 8), torch.randn(1, nkv, 20, 8),
        )
        attn1._quant_cache_append_to_archive(
            torch.randn(1, nkv, 5, 8), torch.randn(1, nkv, 5, 8),
        )

        class _FakeModel(torch.nn.Module):
            def __init__(self, a0, a1):
                super().__init__()
                self.attn0 = a0
                self.attn1 = a1

        sched = TokenBudgetScheduler(total_budget=10, recent_window=4)
        coord = ModelCacheCoordinator.from_model(
            _FakeModel(attn0, attn1), sched, drop_strategy="position",
        )

        sched.on_tokens(50)
        coord._apply_drop()

        remaining0 = sum(c.n_tokens for c in attn0._quant_archive_chunks)
        remaining1 = sum(c.n_tokens for c in attn1._quant_archive_chunks)
        assert remaining1 == 0
        assert remaining0 == 20 - 5
        assert sched._prev_n_drop == 5

    def test_invalid_drop_strategy_raises(self):
        sched = TokenBudgetScheduler(total_budget=20)
        with pytest.raises(ValueError, match="drop_strategy"):
            ModelCacheCoordinator(sched, drop_strategy="invalid")

    def test_summary(self):
        sched = TokenBudgetScheduler(total_budget=20, recent_window=4)
        coord = ModelCacheCoordinator(sched)
        sched.on_tokens(10)
        s = coord.summary()
        assert "seq_len" in s
        assert "drop_strategy" in s
        assert "scheduler_decision" in s


# ======================================================================
# End-to-end: scheduler + HAWPAttention + TurboQuant
# ======================================================================


class TestEndToEnd:
    @torch.inference_mode()
    def test_sched_drop_with_opt(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from hawp_laq.modeling.modeling_llama_hawp import convert_llama_to_hawp
        from hawp_laq.config import build_k_quantizer, build_v_quantizer, HAWPLAQConfig

        cfg = HAWPLAQConfig()
        cfg.model.model_id = "facebook/opt-125m"
        cfg.model.torch_dtype = "float32"
        cfg.train.device = "cpu"
        cfg.projector.r_k = 64
        cfg.projector.r_v = 64
        cfg.sched.total_budget = 20
        cfg.sched.recent_window = 8

        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
        model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m", torch_dtype=torch.float32)
        model = convert_llama_to_hawp(model, r_k=64, r_v=64)
        model.eval()

        k_quantizer = build_k_quantizer(cfg, r_k=64)
        v_quantizer = build_v_quantizer(cfg, r_v=64)
        for mod in model.modules():
            if isinstance(mod, HAWPAttention):
                mod.setup_quant_cache(k_quantizer, v_quantizer, recent_window=8)

        sched = TokenBudgetScheduler(total_budget=20, recent_window=8)
        coord = ModelCacheCoordinator.from_model(model, sched, drop_strategy="position")

        prompt = "Hello, my name is"
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids

        outputs = model(input_ids=input_ids, use_cache=True)
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        coord.on_prefill(input_ids.shape[1])

        generated = [next_token]
        for _ in range(30):
            attention_mask = torch.ones(1, input_ids.shape[1] + len(generated), dtype=torch.long)
            position_ids = torch.tensor([[input_ids.shape[1] + len(generated) - 1]], dtype=torch.long)
            outputs = model(input_ids=next_token, attention_mask=attention_mask, position_ids=position_ids, use_cache=True)
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
            generated.append(next_token)
            coord.on_new_token()

        first_attn = None
        for mod in model.modules():
            if isinstance(mod, HAWPAttention) and mod.use_cache_manager:
                first_attn = mod
                break
        assert first_attn is not None
        s = first_attn.quant_cache_summary()
        total_kept = s["recent_tokens"] + s["archive_tokens"]
        assert total_kept <= 20 + 1

        d = sched.rebalance()
        assert d.n_drop > 0
