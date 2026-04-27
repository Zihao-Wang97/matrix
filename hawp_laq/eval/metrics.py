from __future__ import annotations

from hawp_laq.modeling.attention_hawp import HAWPAttention
from hawp_laq.runtime.cache_stats import _infer_kv_element_size


def collect_kv_metrics(model) -> dict:
    total_recent = 0
    total_archive = 0
    total_recent_fp_bytes = 0
    total_archive_quant_bytes = 0
    total_archive_meta_bytes = 0
    total_runtime_bytes = 0
    total_compressed_storage_bytes = 0
    per_layer = []

    for mod in model.modules():
        if not isinstance(mod, HAWPAttention) or not mod.use_cache_manager:
            continue
        s = mod.quant_cache_summary()
        total_recent += s["recent_tokens"]
        total_archive += s["archive_tokens"]
        total_recent_fp_bytes += s["recent_fp_bytes"]
        total_archive_quant_bytes += s["archive_quant_bytes"]
        total_archive_meta_bytes += s["archive_meta_bytes"]
        total_runtime_bytes += s["total_runtime_bytes"]
        total_compressed_storage_bytes += s["compressed_storage_bytes"]
        per_layer.append(s)

    n_layers = len(per_layer)
    head_dim = 0
    n_kv_heads = 0
    for mod in model.modules():
        if isinstance(mod, HAWPAttention):
            head_dim = mod.head_dim
            n_kv_heads = mod.num_key_value_heads
            break

    tokens_per_layer = (total_recent + total_archive) // n_layers if n_layers > 0 else 0
    recent_per_layer = total_recent // n_layers if n_layers > 0 else 0
    archive_per_layer = total_archive // n_layers if n_layers > 0 else 0
    elem_size = _infer_kv_element_size(model)
    baseline_kv = tokens_per_layer * n_layers * n_kv_heads * head_dim * 2 * elem_size if tokens_per_layer > 0 else 0
    runtime_saving = 1.0 - total_runtime_bytes / baseline_kv if baseline_kv > 0 else 0.0
    compressed_saving = 1.0 - total_compressed_storage_bytes / baseline_kv if baseline_kv > 0 else 0.0

    return {
        "total_recent_tokens": recent_per_layer,
        "total_archive_tokens": archive_per_layer,
        "total_tokens": tokens_per_layer,
        "recent_fp_bytes": total_recent_fp_bytes,
        "archive_quant_bytes": total_archive_quant_bytes,
        "archive_meta_bytes": total_archive_meta_bytes,
        "total_runtime_bytes": total_runtime_bytes,
        "compressed_storage_bytes": total_compressed_storage_bytes,
        "baseline_kv_bytes": baseline_kv,
        "runtime_saving_ratio": runtime_saving,
        "compressed_saving_ratio": compressed_saving,
        "n_layers": n_layers,
        "per_layer": per_layer,
    }


def format_kv_metrics(metrics: dict) -> str:
    from hawp_laq.utils.memory import format_nbytes

    lines = []
    lines.append(f"  total_tokens: {metrics['total_tokens']}  "
                 f"(recent={metrics['total_recent_tokens']}, archive={metrics['total_archive_tokens']})")
    lines.append(f"  [runtime] total: {format_nbytes(metrics['total_runtime_bytes'])}  "
                 f"(recent_fp={format_nbytes(metrics['recent_fp_bytes'])}, "
                 f"archive_quant={format_nbytes(metrics['archive_quant_bytes'])}, "
                 f"archive_meta={format_nbytes(metrics['archive_meta_bytes'])})  "
                 f"saving={metrics['runtime_saving_ratio']:.1%}")
    lines.append(f"  [compressed storage] total: {format_nbytes(metrics['compressed_storage_bytes'])}  "
                 f"saving={metrics['compressed_saving_ratio']:.1%}")
    lines.append(f"  baseline_kv: {format_nbytes(metrics['baseline_kv_bytes'])}")
    lines.append(f"  per-layer KV bytes:")
    for s in metrics["per_layer"]:
        lines.append(f"    layer {s['layer']:>2d}: "
                     f"recent={s['recent_tokens']}  archive={s['archive_tokens']}  "
                     f"runtime={format_nbytes(s['total_runtime_bytes'])}  "
                     f"compressed={format_nbytes(s['compressed_storage_bytes'])}")
    return "\n".join(lines)
