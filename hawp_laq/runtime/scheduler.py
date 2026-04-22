from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class TokenState(Enum):
    HIGH = "high"
    LOW = "low"
    DROP = "drop"


@dataclass
class SchedulerDecision:
    n_high: int
    n_low: int
    n_drop: int


class TokenBudgetScheduler:
    def __init__(
        self,
        total_budget: int,
        recent_window: int = 64,
        high_ratio: float = 0.25,
        low_ratio: float = 0.60,
        drop_strategy: str = "position",
    ):
        self.total_budget = total_budget
        self.recent_window = recent_window
        self.high_ratio = high_ratio
        self.low_ratio = low_ratio
        self.drop_strategy = drop_strategy
        self._current_seq_len: int = 0
        self._prev_n_drop: int = 0

    def on_new_token(self) -> None:
        self._current_seq_len += 1

    def on_tokens(self, n: int) -> None:
        self._current_seq_len += n

    def get_state(self, token_pos: int) -> TokenState:
        if token_pos >= self._current_seq_len:
            raise IndexError(f"token_pos {token_pos} >= seq_len {self._current_seq_len}")

        recent_start = max(0, self._current_seq_len - self.recent_window)
        if token_pos >= recent_start:
            return TokenState.HIGH

        low_budget = max(0, self.total_budget - self.recent_window)
        older_total = self._current_seq_len - self.recent_window
        if older_total <= 0:
            return TokenState.HIGH

        if older_total <= low_budget:
            return TokenState.LOW

        return TokenState.DROP

    def rebalance(self) -> SchedulerDecision:
        total = self._current_seq_len
        n_high = min(self.recent_window, total)
        remaining = total - n_high
        low_budget = max(0, self.total_budget - n_high)
        n_low = min(remaining, low_budget)
        n_drop = max(0, remaining - n_low)
        return SchedulerDecision(n_high=n_high, n_low=n_low, n_drop=n_drop)

    def compute_drop_count(self) -> int:
        decision = self.rebalance()
        new_drop = max(0, decision.n_drop - self._prev_n_drop)
        self._prev_n_drop = decision.n_drop
        return new_drop

    def reset(self) -> None:
        self._current_seq_len = 0
        self._prev_n_drop = 0

    @property
    def seq_len(self) -> int:
        return self._current_seq_len
