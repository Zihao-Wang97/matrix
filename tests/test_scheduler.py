import torch
import pytest
from hawp_laq.runtime.scheduler import TokenBudgetScheduler, TokenState


class TestTokenState:
    def test_high_within_window(self):
        sched = TokenBudgetScheduler(total_budget=512, recent_window=64)
        for _ in range(100):
            sched.on_new_token()
        for pos in range(36, 100):
            assert sched.get_state(pos) == TokenState.HIGH

    def test_low_when_within_budget(self):
        sched = TokenBudgetScheduler(total_budget=512, recent_window=64)
        for _ in range(200):
            sched.on_new_token()
        assert sched.get_state(0) == TokenState.LOW

    def test_drop_when_over_budget(self):
        sched = TokenBudgetScheduler(total_budget=128, recent_window=32, high_ratio=0.25, low_ratio=0.50)
        for _ in range(256):
            sched.on_new_token()
        early_pos = 0
        assert sched.get_state(early_pos) == TokenState.DROP


class TestRebalance:
    def test_no_drop_when_under_budget(self):
        sched = TokenBudgetScheduler(total_budget=512)
        for _ in range(100):
            sched.on_new_token()
        assert sched.rebalance() == []

    def test_drop_indices_returned(self):
        sched = TokenBudgetScheduler(total_budget=64, recent_window=8, high_ratio=0.125, low_ratio=0.5)
        for _ in range(128):
            sched.on_new_token()
        drops = sched.rebalance()
        assert len(drops) > 0
        for idx in drops:
            assert sched.get_state(idx) == TokenState.DROP

    def test_recent_tokens_not_dropped(self):
        sched = TokenBudgetScheduler(total_budget=64, recent_window=16, high_ratio=0.25, low_ratio=0.5)
        for _ in range(200):
            sched.on_new_token()
        drops = sched.rebalance()
        recent_start = 200 - 16
        for idx in drops:
            assert idx < recent_start


class TestReset:
    def test_reset_clears_seq_len(self):
        sched = TokenBudgetScheduler(total_budget=512)
        for _ in range(100):
            sched.on_new_token()
        assert sched.seq_len == 100
        sched.reset()
        assert sched.seq_len == 0


class TestIndexError:
    def test_out_of_range_raises(self):
        sched = TokenBudgetScheduler(total_budget=512)
        sched.on_new_token()
        with pytest.raises(IndexError):
            sched.get_state(5)
