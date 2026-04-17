from __future__ import annotations

from enum import Enum


class TokenState(Enum):
    HIGH = "high"
    LOW = "low"
    DROP = "drop"


class TokenBudgetScheduler:
    def __init__(
        self,
        total_budget: int,
        recent_window: int = 64,
        high_ratio: float = 0.25,
        low_ratio: float = 0.60,
    ):
        self.total_budget = total_budget
        self.recent_window = recent_window
        self.high_ratio = high_ratio
        self.low_ratio = low_ratio
        self._current_seq_len: int = 0

    def on_new_token(self) -> None:
        self._current_seq_len += 1

    def get_state(self, token_pos: int) -> TokenState:
        if token_pos >= self._current_seq_len:
            raise IndexError(f"token_pos {token_pos} >= seq_len {self._current_seq_len}")

        recent_start = max(0, self._current_seq_len - self.recent_window)
        if token_pos >= recent_start:
            return TokenState.HIGH

        high_budget = max(self.recent_window, int(self.total_budget * self.high_ratio))
        low_budget = int(self.total_budget * self.low_ratio)
        older_total = self._current_seq_len - self.recent_window
        if older_total <= 0:
            return TokenState.HIGH

        if older_total <= low_budget:
            return TokenState.LOW

        return TokenState.DROP

    def rebalance(self) -> list[int]:
        if self._current_seq_len <= self.total_budget:
            return []

        drop_indices: list[int] = []
        for i in range(self._current_seq_len):
            if self.get_state(i) == TokenState.DROP:
                drop_indices.append(i)
        return drop_indices

    def reset(self) -> None:
        self._current_seq_len = 0

    @property
    def seq_len(self) -> int:
        return self._current_seq_len
