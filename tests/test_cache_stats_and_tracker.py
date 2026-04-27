from __future__ import annotations

import torch
import pytest

from hawp_laq.runtime.cache_stats import (
    CacheStats,
    aggregate_cache_stats,
    collect_cache_stats,
    collect_cache_stats_from_tracker,
    compute_baseline_kv_bytes,
)
from hawp_laq.runtime.past_kv_tracker import PastKVTracker
from hawp_laq.utils.memory import format_nbytes


class TestCacheStats:
    def test_default_values(self):
        s = CacheStats()
        assert s.cache_tokens_total == 0
        assert s.cache_runtime_bytes == 0
        assert s.cache_compressed_bytes == 0
        assert s.peak_gpu_bytes == 0
        assert s.impl == ""
        assert s.baseline_kv_bytes == 0

    def test_format_summary_basic(self):
        s = CacheStats(
            cache_tokens_total=512,
            cache_runtime_bytes=1024 * 1024,
            cache_compressed_bytes=512 * 1024,
            peak_gpu_bytes=2 * 1024 * 1024,
            impl="hawp_quant_cache",
        )
        text = s.format_summary()
        assert "hawp_quant_cache" in text
        assert "512" in text

    def test_format_summary_with_recent_archive(self):
        s = CacheStats(
            cache_tokens_total=512,
            cache_runtime_bytes=1024,
            peak_gpu_bytes=2048,
            impl="test",
            recent_tokens=64,
            archive_tokens=448,
        )
        text = s.format_summary()
        assert "recent=64" in text
        assert "archive=448" in text

    def test_format_summary_without_recent_archive(self):
        s = CacheStats(
            cache_tokens_total=100,
            cache_runtime_bytes=1024,
            impl="past_kv_baseline",
        )
        text = s.format_summary()
        assert "recent" not in text

    def test_kv_compression_ratio(self):
        s = CacheStats(
            cache_tokens_total=100,
            cache_runtime_bytes=1000,
            baseline_kv_bytes=4000,
        )
        assert s.kv_compression_ratio == 4.0

    def test_kv_compression_ratio_zero_runtime(self):
        s = CacheStats(baseline_kv_bytes=4000)
        assert s.kv_compression_ratio == 0.0

    def test_bytes_per_token(self):
        s = CacheStats(cache_tokens_total=100, cache_runtime_bytes=2000)
        assert s.bytes_per_token == 20.0

    def test_bytes_per_token_zero_tokens(self):
        s = CacheStats(cache_runtime_bytes=2000)
        assert s.bytes_per_token == 0.0

    def test_recent_ratio(self):
        s = CacheStats(cache_tokens_total=512, recent_tokens=64, archive_tokens=448)
        assert s.recent_ratio == pytest.approx(64 / 512)
        assert s.archive_ratio == pytest.approx(448 / 512)

    def test_recent_ratio_zero_tokens(self):
        s = CacheStats(recent_tokens=64, archive_tokens=448)
        assert s.recent_ratio == 0.0
        assert s.archive_ratio == 0.0

    def test_format_summary_with_compression_ratio(self):
        s = CacheStats(
            cache_tokens_total=100,
            cache_runtime_bytes=1000,
            baseline_kv_bytes=4000,
            peak_gpu_bytes=5000,
            impl="test",
        )
        text = s.format_summary()
        assert "kv_compression_ratio=4.00x" in text
        assert "model_bytes_per_token=10.0 B" in text
        assert "memory_overhead_ratio=5.00x" in text

    def test_memory_overhead_ratio(self):
        s = CacheStats(cache_runtime_bytes=1000, peak_gpu_bytes=5000)
        assert s.memory_overhead_ratio == 5.0

    def test_memory_overhead_ratio_zero_runtime(self):
        s = CacheStats(peak_gpu_bytes=5000)
        assert s.memory_overhead_ratio == 0.0

    def test_aggregate_cache_stats_averages_prompts_and_keeps_peak_gpu_max(self):
        stats = aggregate_cache_stats([
            CacheStats(
                cache_tokens_total=10,
                cache_runtime_bytes=100,
                cache_compressed_bytes=40,
                peak_gpu_bytes=1000,
                impl="hawp_quant_cache",
                recent_tokens=4,
                archive_tokens=6,
                meta_bytes=8,
                baseline_kv_bytes=200,
            ),
            CacheStats(
                cache_tokens_total=30,
                cache_runtime_bytes=300,
                cache_compressed_bytes=120,
                peak_gpu_bytes=900,
                impl="hawp_quant_cache",
                recent_tokens=8,
                archive_tokens=22,
                meta_bytes=16,
                baseline_kv_bytes=600,
            ),
        ])

        assert stats.cache_tokens_total == 20
        assert stats.cache_runtime_bytes == 200
        assert stats.cache_compressed_bytes == 80
        assert stats.peak_gpu_bytes == 1000
        assert stats.recent_tokens == 6
        assert stats.archive_tokens == 14
        assert stats.meta_bytes == 12
        assert stats.baseline_kv_bytes == 400
        assert stats.impl == "hawp_quant_cache_avg_over_2_prompts"


class TestComputeBaselineKvBytes:
    def test_with_model_config(self):
        from types import SimpleNamespace
        config = SimpleNamespace(
            num_hidden_layers=12,
            num_attention_heads=12,
            num_key_value_heads=12,
            hidden_size=768,
        )
        model = SimpleNamespace(config=config)
        result = compute_baseline_kv_bytes(model, 512)
        expected = 12 * 12 * 512 * 64 * 2 * 2
        assert result == expected

    def test_fp32_uses_4_bytes(self):
        import torch
        from types import SimpleNamespace
        config = SimpleNamespace(
            num_hidden_layers=12,
            num_attention_heads=12,
            num_key_value_heads=12,
            hidden_size=768,
            torch_dtype=torch.float32,
        )
        model = SimpleNamespace(config=config)
        result = compute_baseline_kv_bytes(model, 512)
        expected = 12 * 12 * 512 * 64 * 2 * 4
        assert result == expected

    def test_fp16_uses_2_bytes(self):
        import torch
        from types import SimpleNamespace
        config = SimpleNamespace(
            num_hidden_layers=6,
            num_attention_heads=8,
            num_key_value_heads=8,
            hidden_size=512,
            torch_dtype=torch.float16,
        )
        model = SimpleNamespace(config=config)
        result = compute_baseline_kv_bytes(model, 256)
        expected = 6 * 8 * 256 * 64 * 2 * 2
        assert result == expected

    def test_bf16_uses_2_bytes(self):
        import torch
        from types import SimpleNamespace
        config = SimpleNamespace(
            num_hidden_layers=6,
            num_attention_heads=8,
            num_key_value_heads=8,
            hidden_size=512,
            torch_dtype=torch.bfloat16,
        )
        model = SimpleNamespace(config=config)
        result = compute_baseline_kv_bytes(model, 256)
        expected = 6 * 8 * 256 * 64 * 2 * 2
        assert result == expected

    def test_zero_seq_len(self):
        from types import SimpleNamespace
        model = SimpleNamespace(config=SimpleNamespace())
        assert compute_baseline_kv_bytes(model, 0) == 0

    def test_no_config(self):
        from types import SimpleNamespace
        model = SimpleNamespace()
        assert compute_baseline_kv_bytes(model, 512) == 0


class TestPastKVTracker:
    def test_empty_tracker(self):
        t = PastKVTracker()
        assert t.total_tokens == 0
        assert t.total_bytes == 0

    def test_update_with_tuple_list(self):
        t = PastKVTracker()
        k = torch.randn(1, 2, 10, 64)
        v = torch.randn(1, 2, 10, 64)
        past = [(k, v)]
        t.update(past)
        assert t.total_tokens == 10
        expected = k.nelement() * k.element_size() + v.nelement() * v.element_size()
        assert t.total_bytes == expected

    def test_update_with_none(self):
        t = PastKVTracker()
        t.update(None)
        assert t.total_tokens == 0
        assert t.total_bytes == 0

    def test_update_replaces_previous(self):
        t = PastKVTracker()
        k1 = torch.randn(1, 2, 5, 64)
        v1 = torch.randn(1, 2, 5, 64)
        t.update([(k1, v1)])
        assert t.total_tokens == 5

        k2 = torch.randn(1, 2, 10, 64)
        v2 = torch.randn(1, 2, 10, 64)
        t.update([(k2, v2)])
        assert t.total_tokens == 10

    def test_reset(self):
        t = PastKVTracker()
        k = torch.randn(1, 2, 10, 64)
        v = torch.randn(1, 2, 10, 64)
        t.update([(k, v)])
        t.reset()
        assert t.total_tokens == 0
        assert t.total_bytes == 0

    def test_multi_layer_tuple_list(self):
        t = PastKVTracker()
        layers = []
        for _ in range(4):
            k = torch.randn(1, 2, 10, 64)
            v = torch.randn(1, 2, 10, 64)
            layers.append((k, v))
        t.update(layers)
        assert t.total_tokens == 10

    def test_fp16_bytes(self):
        t = PastKVTracker()
        k = torch.randn(1, 2, 10, 64, dtype=torch.float16)
        v = torch.randn(1, 2, 10, 64, dtype=torch.float16)
        t.update([(k, v)])
        expected = k.nelement() * 2 + v.nelement() * 2
        assert t.total_bytes == expected


class TestCollectCacheStatsFromTracker:
    def test_produces_cache_stats(self):
        t = PastKVTracker()
        k = torch.randn(1, 2, 10, 64)
        v = torch.randn(1, 2, 10, 64)
        t.update([(k, v)])

        stats = collect_cache_stats_from_tracker(t, peak_gpu_bytes=9999, impl="past_kv_baseline")
        assert isinstance(stats, CacheStats)
        assert stats.cache_tokens_total == 10
        assert stats.peak_gpu_bytes == 9999
        assert stats.impl == "past_kv_baseline"
        assert stats.cache_compressed_bytes == 0


class TestCollectCacheStatsNone:
    def test_returns_empty_stats_when_no_cache(self):
        from types import SimpleNamespace
        model = SimpleNamespace()
        stats = collect_cache_stats(model, kv_manager=None, peak_gpu_bytes=42)
        assert isinstance(stats, CacheStats)
        assert stats.peak_gpu_bytes == 42
        assert stats.impl == "none"

