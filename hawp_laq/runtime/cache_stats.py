"""Unified cache statistics across all generation modes.

Every mode returns a ``CacheStats`` with the same primary fields so that
scripts can compare results without knowing the implementation details.

Derived metrics (computed from primary fields):
  - kv_compression_ratio : baseline_kv_bytes / cache_runtime_bytes
  - bytes_per_token      : cache_runtime_bytes / cache_tokens_total
                           (all layers per sequence token)
  - recent_ratio         : recent_tokens / cache_tokens_total
  - archive_ratio        : archive_tokens / cache_tokens_total
  - memory_overhead_ratio: peak_gpu_bytes / cache_runtime_bytes
"""

from __future__ import annotations

from dataclasses import dataclass

from hawp_laq.utils.memory import format_nbytes


@dataclass
class CacheStats:
    cache_tokens_total: int = 0
    cache_runtime_bytes: int = 0
    cache_compressed_bytes: int = 0
    peak_gpu_bytes: int = 0
    impl: str = ""
    recent_tokens: int = 0
    archive_tokens: int = 0
    meta_bytes: int = 0
    baseline_kv_bytes: int = 0

    @property
    def kv_compression_ratio(self) -> float:
        if self.cache_runtime_bytes > 0 and self.baseline_kv_bytes > 0:
            return self.baseline_kv_bytes / self.cache_runtime_bytes
        return 0.0

    @property
    def bytes_per_token(self) -> float:
        if self.cache_tokens_total > 0:
            return self.cache_runtime_bytes / self.cache_tokens_total
        return 0.0

    @property
    def recent_ratio(self) -> float:
        if self.cache_tokens_total > 0:
            return self.recent_tokens / self.cache_tokens_total
        return 0.0

    @property
    def archive_ratio(self) -> float:
        if self.cache_tokens_total > 0:
            return self.archive_tokens / self.cache_tokens_total
        return 0.0

    @property
    def memory_overhead_ratio(self) -> float:
        if self.cache_runtime_bytes > 0:
            return self.peak_gpu_bytes / self.cache_runtime_bytes
        return 0.0

    def format_summary(self) -> str:
        lines = [
            f"[cache] impl={self.impl}",
            f"[cache] tokens_total={self.cache_tokens_total}",
            f"[cache] runtime_bytes={format_nbytes(self.cache_runtime_bytes)}",
            f"[cache] archive_quant_bytes={format_nbytes(self.cache_compressed_bytes)}",
            f"[peak_gpu] {format_nbytes(self.peak_gpu_bytes)}",
        ]
        if self.baseline_kv_bytes > 0 and self.kv_compression_ratio > 0:
            lines.append(f"[cache] kv_compression_ratio={self.kv_compression_ratio:.2f}x  (baseline_kv={format_nbytes(self.baseline_kv_bytes)})")
        if self.bytes_per_token > 0:
            lines.append(f"[cache] model_bytes_per_token={self.bytes_per_token:.1f} B (all layers per sequence token)")
        if self.memory_overhead_ratio > 0:
            lines.append(f"[cache] memory_overhead_ratio={self.memory_overhead_ratio:.2f}x  (peak_gpu/cache_runtime)")
        if self.recent_tokens > 0 or self.archive_tokens > 0:
            lines.append(
                f"[cache] recent={self.recent_tokens} ({self.recent_ratio:.1%})  "
                f"archive={self.archive_tokens} ({self.archive_ratio:.1%})  "
                f"meta={format_nbytes(self.meta_bytes)}"
            )
        return "\n".join(lines)


def _infer_kv_element_size(model) -> int:
    config = getattr(model, "config", None)
    torch_dtype = getattr(config, "torch_dtype", None)
    if torch_dtype is not None:
        import torch
        if isinstance(torch_dtype, torch.dtype):
            return torch_dtype.itemsize
    try:
        param = next(model.parameters())
        return param.element_size()
    except (StopIteration, AttributeError):
        return 2


def compute_baseline_kv_bytes(model, seq_len: int) -> int:
    config = getattr(model, "config", None)
    if config is None or seq_len <= 0:
        return 0
    n_layers = getattr(config, "num_hidden_layers", 12)
    n_heads = getattr(config, "num_attention_heads", 12)
    n_kv_heads = getattr(config, "num_key_value_heads", n_heads)
    hidden_size = getattr(config, "hidden_size", 768)
    head_dim = hidden_size // n_heads
    elem_size = _infer_kv_element_size(model)
    return n_layers * n_kv_heads * seq_len * head_dim * 2 * elem_size


def collect_cache_stats(model, kv_manager=None, peak_gpu_bytes: int = 0) -> CacheStats:
    from hawp_laq.modeling.attention_hawp import HAWPAttention
    from hawp_laq.runtime.pure_quant_hook import PureQuantKVManager

    has_modules = hasattr(model, "modules")
    has_hawp_cache = False
    if has_modules:
        has_hawp_cache = any(
            isinstance(m, HAWPAttention) and m.use_cache_manager
            for m in model.modules()
        )

    if has_hawp_cache:
        return _from_hawp_cache(model, peak_gpu_bytes)

    if kv_manager is not None and isinstance(kv_manager, PureQuantKVManager):
        return _from_pure_quant(kv_manager, peak_gpu_bytes)

    return CacheStats(peak_gpu_bytes=peak_gpu_bytes, impl="none")


def collect_cache_stats_from_tracker(tracker, peak_gpu_bytes: int = 0, impl: str = "past_kv_tracker") -> CacheStats:
    return CacheStats(
        cache_tokens_total=tracker.total_tokens,
        cache_runtime_bytes=tracker.total_bytes,
        cache_compressed_bytes=0,
        peak_gpu_bytes=peak_gpu_bytes,
        impl=impl,
    )


def aggregate_cache_stats(stats_list: list[CacheStats]) -> CacheStats:
    """Aggregate per-prompt cache stats into one run-level summary.

    Cache tensors are reset between prompts in the profiling path, so summing
    cache bytes would overstate the memory footprint.  We report average
    cache/token fields across prompts and keep peak GPU as the max capacity
    observed during any prompt.
    """
    if not stats_list:
        return CacheStats()
    if len(stats_list) == 1:
        return stats_list[0]

    n = len(stats_list)

    def avg_int(field_name: str) -> int:
        return int(round(sum(getattr(s, field_name) for s in stats_list) / n))

    impls = {s.impl for s in stats_list}
    base_impl = stats_list[0].impl if len(impls) == 1 else "mixed_cache"

    return CacheStats(
        cache_tokens_total=avg_int("cache_tokens_total"),
        cache_runtime_bytes=avg_int("cache_runtime_bytes"),
        cache_compressed_bytes=avg_int("cache_compressed_bytes"),
        peak_gpu_bytes=max(s.peak_gpu_bytes for s in stats_list),
        impl=f"{base_impl}_avg_over_{n}_prompts",
        recent_tokens=avg_int("recent_tokens"),
        archive_tokens=avg_int("archive_tokens"),
        meta_bytes=avg_int("meta_bytes"),
        baseline_kv_bytes=avg_int("baseline_kv_bytes"),
    )


def _from_hawp_cache(model, peak_gpu_bytes: int) -> CacheStats:
    from hawp_laq.modeling.attention_hawp import HAWPAttention

    total_recent = 0
    total_archive = 0
    total_runtime = 0
    total_compressed = 0
    total_meta = 0
    n_layers = 0

    for mod in model.modules():
        if not isinstance(mod, HAWPAttention) or not mod.use_cache_manager:
            continue
        s = mod.quant_cache_summary()
        total_recent += s["recent_tokens"]
        total_archive += s["archive_tokens"]
        total_runtime += s["total_runtime_bytes"]
        total_compressed += s["compressed_storage_bytes"]
        total_meta += s["archive_meta_bytes"]
        n_layers += 1

    return CacheStats(
        cache_tokens_total=(total_recent + total_archive) // n_layers if n_layers > 0 else 0,
        cache_runtime_bytes=total_runtime,
        cache_compressed_bytes=total_compressed,
        peak_gpu_bytes=peak_gpu_bytes,
        impl="hawp_quant_cache",
        recent_tokens=total_recent // n_layers if n_layers > 0 else 0,
        archive_tokens=total_archive // n_layers if n_layers > 0 else 0,
        meta_bytes=total_meta,
    )


def _from_pure_quant(kv_manager, peak_gpu_bytes: int) -> CacheStats:
    summaries = kv_manager.cache_summaries()
    n_layers = len(summaries) if summaries else 1
    total_recent = sum(s["recent_tokens"] for s in summaries)
    total_archive = sum(s["archive_tokens"] for s in summaries)
    total_runtime = sum(s["total_runtime_bytes"] for s in summaries)
    total_compressed = sum(s["compressed_storage_bytes"] for s in summaries)
    total_meta = sum(s.get("archive_meta_bytes", 0) for s in summaries)

    return CacheStats(
        cache_tokens_total=(total_recent + total_archive) // n_layers,
        cache_runtime_bytes=total_runtime,
        cache_compressed_bytes=total_compressed,
        peak_gpu_bytes=peak_gpu_bytes,
        impl="pure_quant_cache",
        recent_tokens=total_recent // n_layers,
        archive_tokens=total_archive // n_layers,
        meta_bytes=total_meta,
    )
