from __future__ import annotations

import torch

from hawp_laq.config import HAWPLAQConfig, build_k_quantizer, build_v_quantizer
from hawp_laq.runtime.latent_cache import LayerKVCache
from hawp_laq.runtime.scheduler import TokenBudgetScheduler, TokenState
from hawp_laq.utils.memory import format_nbytes


class CacheManager:
    """Multi-layer KV cache with recent/archive tiers and TurboQuant.

    * Recent tokens: stored as latent tensors in ``dtype`` precision.
    * Archive tokens: K via TurboQuantProd, V via TurboQuantMSE.

    Can be constructed either from a :class:`HAWPLAQConfig` or by
    passing quantizer instances directly.

    .. note::
        The current runtime cache only supports ``k_dim == v_dim``.
        Asymmetric ``r_k != r_v`` is valid in training and projector
        artifacts, but ``LayerKVCache`` has not yet been extended to
        store K and V latent tensors with different column counts.
        This is a known unimplemented limitation, not a user
        configuration error.

    Args:
        n_layers: Number of transformer layers.
        n_heads: Number of KV heads.
        head_dim: Latent dimension per head.
        k_dim: Latent K dimension per head (default: head_dim for full-rank).
        v_dim: Latent V dimension per head (default: head_dim for full-rank).
        scheduler: Token-budget scheduler (or a dummy one).
        cfg: Optional HAWPLAQConfig — used to build K/V quantizers.
        k_quantizer: Pre-built K quantizer (overrides cfg).
        v_quantizer: Pre-built V quantizer (overrides cfg).
        dtype: Storage dtype for recent tokens, forwarded to LayerKVCache.
            Must match the model weight dtype.  Required (no default).
    """

    def __init__(
        self,
        n_layers: int,
        n_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        k_dim: int | None = None,
        v_dim: int | None = None,
        scheduler: TokenBudgetScheduler | None = None,
        recent_window: int | None = None,
        cfg: HAWPLAQConfig | None = None,
        k_quantizer=None,
        v_quantizer=None,
        **_kwargs,
    ):
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.k_dim = k_dim if k_dim is not None else head_dim
        self.v_dim = v_dim if v_dim is not None else head_dim
        self.recent_window = recent_window if recent_window is not None else 0
        if self.k_dim != self.v_dim:
            raise NotImplementedError(
                f"CacheManager requires k_dim == v_dim, but got k_dim={self.k_dim}, "
                f"v_dim={self.v_dim}. LayerKVCache does not yet support asymmetric "
                f"latent dimensions (r_k != r_v). This is a known unimplemented "
                f"limitation — asymmetric r_k/r_v is valid in training and projector "
                f"artifacts, but the runtime cache path has not been extended to store "
                f"K and V latent tensors with different column counts."
            )
        self.scheduler = scheduler or TokenBudgetScheduler(total_budget=999999)

        if k_quantizer is None or v_quantizer is None:
            if cfg is None:
                cfg = HAWPLAQConfig()
            if k_quantizer is None:
                k_quantizer = build_k_quantizer(cfg, r_k=self.k_dim)
            if v_quantizer is None:
                v_quantizer = build_v_quantizer(cfg, r_v=self.v_dim)

        self._k_quantizer = k_quantizer
        self._v_quantizer = v_quantizer

        self._caches: list[LayerKVCache] = []
        for layer_idx in range(n_layers):
            from hawp_laq.runtime.turboquant import TurboQuantMSE, TurboQuantProd
            if isinstance(k_quantizer, TurboQuantProd):
                kq = TurboQuantProd(
                    dim=self.k_dim, bits=k_quantizer.bits,
                    use_rotation=k_quantizer.use_rotation,
                    group_size=k_quantizer.group_size,
                )
            else:
                kq = TurboQuantMSE(
                    dim=self.k_dim, bits=k_quantizer.bits,
                    use_rotation=k_quantizer.use_rotation,
                    group_size=k_quantizer.group_size,
                )
            vq = TurboQuantMSE(
                dim=self.v_dim, bits=v_quantizer.bits,
                use_rotation=v_quantizer.use_rotation,
                group_size=v_quantizer.group_size,
            )
            self._caches.append(LayerKVCache(n_heads, head_dim, kq, vq, dtype=dtype, recent_window=self.recent_window))

    # ------------------------------------------------------------------
    # Append
    # ------------------------------------------------------------------

    def append_token(self, k_per_layer: list[torch.Tensor], v_per_layer: list[torch.Tensor]) -> None:
        if len(k_per_layer) != self.n_layers or len(v_per_layer) != self.n_layers:
            raise ValueError(
                f"Expected {self.n_layers} layers, got {len(k_per_layer)} k and {len(v_per_layer)} v"
            )
        self.scheduler.on_new_token()
        for i, (k, v) in enumerate(zip(k_per_layer, v_per_layer)):
            self._caches[i].append_recent(k, v)

    # ------------------------------------------------------------------
    # Retrieve
    # ------------------------------------------------------------------

    def get_kv_for_attention(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get full KV for a layer: archive dequantized + recent.

        Returns:
            ``(k, v)`` tensors of shape ``[T, dim]`` in ``dtype``.
        """
        cache = self._caches[layer_idx]
        return cache.get_all_k(), cache.get_all_v()

    # ------------------------------------------------------------------
    # Demote
    # ------------------------------------------------------------------

    def demote_all(self) -> None:
        """Demote all recent tokens to the compressed archive tier."""
        for c in self._caches:
            c.demote_to_archive()

    # ------------------------------------------------------------------
    # Scheduler (stub — not used in this stage)
    # ------------------------------------------------------------------

    def apply_scheduler(self) -> int:
        drop_count = self.scheduler.compute_drop_count()
        if drop_count <= 0:
            return 0
        min_can_drop = drop_count
        for c in self._caches:
            min_can_drop = min(min_can_drop, c.n_archive)
        if min_can_drop <= 0:
            return 0
        actual_drops = []
        for c in self._caches:
            actual_drops.append(c.drop_oldest(min_can_drop))
        min_dropped = min(actual_drops) if actual_drops else 0
        if min_dropped > 0:
            self.scheduler.acknowledge_drop(min_dropped)
        return min_dropped

    # ------------------------------------------------------------------
    # Memory
    # ------------------------------------------------------------------

    def nbytes_recent(self) -> int:
        return sum(c.nbytes_recent() for c in self._caches)

    def nbytes_archive(self) -> int:
        return sum(c.nbytes_archive() for c in self._caches)

    def total_nbytes(self) -> int:
        return self.nbytes_recent() + self.nbytes_archive()

    def total_nbytes_formatted(self) -> str:
        return format_nbytes(self.total_nbytes())

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        recent = self._caches[0].n_recent if self._caches else 0
        archive = self._caches[0].n_archive if self._caches else 0
        return {
            "seq_len": self.scheduler.seq_len,
            "recent_tokens": recent,
            "archive_tokens": archive,
            "recent_nbytes": self.nbytes_recent(),
            "archive_nbytes": self.nbytes_archive(),
            "total_nbytes": self.total_nbytes(),
            "total_nbytes_formatted": self.total_nbytes_formatted(),
        }

    def __getitem__(self, layer_idx: int) -> LayerKVCache:
        return self._caches[layer_idx]

    def __len__(self) -> int:
        return self.n_layers


class ModelCacheCoordinator:
    """Coordinates TokenBudgetScheduler with HAWPAttention layers.

    Three-state token management:
        - HIGH: recent fp16 latent (within ``recent_window``)
        - LOW: TurboQuant archive latent (compressed)
        - DROP: removed from cache entirely

    Drop strategies:
        - ``position`` (default): drop oldest archive tokens first
        - ``norm``: drop archive tokens with smallest K latent norm

    Args:
        scheduler: Token budget scheduler.
        drop_strategy: ``"position"`` or ``"norm"``.
    """

    _DROP_STRATEGIES = ("position", "norm")

    def __init__(
        self,
        scheduler: TokenBudgetScheduler,
        drop_strategy: str = "position",
    ) -> None:
        if drop_strategy not in self._DROP_STRATEGIES:
            raise ValueError(
                f"drop_strategy must be one of {self._DROP_STRATEGIES}, "
                f"got '{drop_strategy}'"
            )
        self.scheduler = scheduler
        self.drop_strategy = drop_strategy
        self._layers: list = []

    @classmethod
    def from_model(
        cls,
        model: torch.nn.Module,
        scheduler: TokenBudgetScheduler,
        drop_strategy: str = "position",
    ) -> ModelCacheCoordinator:
        from hawp_laq.modeling.attention_hawp import HAWPAttention

        coord = cls(scheduler, drop_strategy)
        for mod in model.modules():
            if isinstance(mod, HAWPAttention) and mod.use_cache_manager:
                coord._layers.append(mod)
        return coord

    def on_prefill(self, prompt_len: int) -> None:
        self.scheduler.on_tokens(prompt_len)
        self._apply_drop()

    def on_new_token(self) -> None:
        self.scheduler.on_new_token()
        self._apply_drop()

    def _apply_drop(self) -> None:
        drop_count = self.scheduler.compute_drop_count()
        if drop_count <= 0:
            return
        min_can_drop = drop_count
        for layer in self._layers:
            min_can_drop = min(min_can_drop, layer.n_archive_tokens)
        if min_can_drop <= 0:
            return
        actual_drops = []
        for layer in self._layers:
            if self.drop_strategy == "norm":
                actual_drops.append(layer.drop_least_important_from_archive(min_can_drop))
            else:
                actual_drops.append(layer.drop_oldest_from_archive(min_can_drop))
        min_dropped = min(actual_drops) if actual_drops else 0
        if min_dropped > 0:
            self.scheduler.acknowledge_drop(min_dropped)

    def reset(self) -> None:
        self.scheduler.reset()

    def summary(self) -> dict:
        layers_info = []
        for layer in self._layers:
            s = layer.quant_cache_summary()
            layers_info.append(s)
        return {
            "seq_len": self.scheduler.seq_len,
            "drop_strategy": self.drop_strategy,
            "scheduler_decision": self.scheduler.rebalance(),
            "layers": layers_info,
        }
