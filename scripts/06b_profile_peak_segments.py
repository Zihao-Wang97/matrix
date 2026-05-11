#!/usr/bin/env python
"""Segmented peak-GPU profiler for generation.

This diagnostic script answers: where does ``torch.cuda.max_memory_allocated()``
rise during baseline / HAWP quant generation?

It intentionally keeps the generation loop close to
``hawp_laq.runtime.mode_runner.profile_generate_by_mode`` and adds memory
snapshots around:

  - model load and mode setup
  - tokenization / prompt transfer
  - prefill forward
  - decode forwards
  - selected HAWPAttention internals such as dequant, cat-heavy archive paths,
    quant-cache append, and repeat_kv

Example:
  python scripts/06b_profile_peak_segments.py configs/run_server.yaml --mode hawp_quant --seq-len 4096 --max-new-tokens 8
  python scripts/06b_profile_peak_segments.py configs/run_server.yaml --mode baseline --seq-len 4096 --max-new-tokens 8
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import torch

from hawp_laq.config import load_config
from hawp_laq.modeling.attention_hawp import HAWPAttention
from hawp_laq.runtime.cache_stats import (
    collect_cache_stats,
    collect_cache_stats_from_tracker,
    compute_baseline_kv_bytes,
)
from hawp_laq.runtime.forward_utils import prefill_forward_last_logits
from hawp_laq.runtime.generate import _resolve_device, load_baseline_model
from hawp_laq.runtime.mode_runner import make_reset_fn, setup_mode
from hawp_laq.runtime.past_kv_tracker import PastKVTracker
from hawp_laq.utils.io import save_json
from hawp_laq.utils.memory import format_nbytes


_MODES = (
    "baseline",
    "hawp_only",
    "quant_only",
    "pure_quant_only",
    "hawp_quant",
    "hawp_quant_all",
    "hawp_quant_sched",
)

_HAWP_PROBE_METHODS = (
    "_forward_low_rank",
    "_compute_archive_k_logits_approx",
    "_compute_archive_k_logits_dequant",
    "_dequant_archive_k",
    "_dequant_archive_v",
    "_compute_decode_attention_blockwise",
    "_compute_archive_k_logits_block_approx_grouped",
    "_stream_softmax_block",
    "_compute_recent_k_logits",
    "_quant_cache_append_latent",
    "_quant_cache_append_to_archive",
    "_quantize_to_chunk",
    "_append_or_merge_archive_chunk",
    "_repeat_kv",
)


class MemoryTracer:
    def __init__(self, *, synchronize: bool = True) -> None:
        self.records: list[dict[str, Any]] = []
        self.synchronize = synchronize
        self.t0 = time.perf_counter()
        self._last_allocated = 0
        self._last_peak = 0

    def reset_delta_baseline(self) -> None:
        self._last_allocated = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        self._last_peak = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0

    def record(self, label: str, **meta: Any) -> None:
        if torch.cuda.is_available() and self.synchronize:
            torch.cuda.synchronize()

        allocated = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        reserved = torch.cuda.memory_reserved() if torch.cuda.is_available() else 0
        peak = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        rec = {
            "i": len(self.records),
            "t_s": round(time.perf_counter() - self.t0, 6),
            "label": label,
            "allocated_bytes": int(allocated),
            "allocated": format_nbytes(int(allocated)),
            "reserved_bytes": int(reserved),
            "reserved": format_nbytes(int(reserved)),
            "peak_allocated_bytes": int(peak),
            "peak_allocated": format_nbytes(int(peak)),
            "delta_allocated_bytes": int(allocated - self._last_allocated),
            "delta_peak_bytes": int(peak - self._last_peak),
            **meta,
        }
        rec["delta_allocated"] = format_nbytes(abs(rec["delta_allocated_bytes"]))
        if rec["delta_allocated_bytes"] < 0:
            rec["delta_allocated"] = "-" + rec["delta_allocated"]
        rec["delta_peak"] = format_nbytes(abs(rec["delta_peak_bytes"]))
        if rec["delta_peak_bytes"] < 0:
            rec["delta_peak"] = "-" + rec["delta_peak"]
        self.records.append(rec)
        self._last_allocated = int(allocated)
        self._last_peak = int(peak)


def _build_prompt_for_profile(tokenizer, target_seq_len: int) -> tuple[str, int]:
    seed_text = "The " * target_seq_len
    enc = tokenizer(seed_text, return_tensors="pt")
    prompt_ids = enc["input_ids"][0][:target_seq_len]
    prompt = tokenizer.decode(
        prompt_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    actual_len = tokenizer(prompt, return_tensors="pt").input_ids.shape[1]
    return prompt, actual_len


def _shape_meta(value: Any) -> dict[str, Any]:
    if isinstance(value, torch.Tensor):
        return {
            "shape": tuple(value.shape),
            "dtype": str(value.dtype).replace("torch.", ""),
            "device": str(value.device),
        }
    return {}


def _collect_tensor_shapes(value: Any, *, limit: int = 4) -> list[tuple[int, ...]]:
    shapes: list[tuple[int, ...]] = []

    def visit(obj: Any) -> None:
        if len(shapes) >= limit:
            return
        if isinstance(obj, torch.Tensor):
            shapes.append(tuple(obj.shape))
        elif isinstance(obj, (tuple, list)):
            for item in obj:
                visit(item)
                if len(shapes) >= limit:
                    break
        elif hasattr(obj, "logits") and isinstance(obj.logits, torch.Tensor):
            visit(obj.logits)
        elif hasattr(obj, "last_hidden_state") and isinstance(obj.last_hidden_state, torch.Tensor):
            visit(obj.last_hidden_state)

    visit(value)
    return shapes


def _io_meta(value: Any, *, prefix: str) -> dict[str, Any]:
    if isinstance(value, torch.Tensor):
        meta = _shape_meta(value)
        return {f"{prefix}_{k}": v for k, v in meta.items()}
    shapes = _collect_tensor_shapes(value)
    if shapes:
        return {f"{prefix}_shapes": shapes}
    return {}


def _method_meta(method_name: str, args: tuple[Any, ...]) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    if args:
        meta.update(_shape_meta(args[0]))
    if method_name == "_forward_low_rank" and args and isinstance(args[0], torch.Tensor):
        meta["q_len"] = int(args[0].shape[2])
        meta["n_heads"] = int(args[0].shape[1])
        meta["head_dim"] = int(args[0].shape[-1])
    elif method_name in (
        "_compute_archive_k_logits_approx",
        "_compute_archive_k_logits_dequant",
        "_compute_recent_k_logits",
    ) and args and isinstance(args[0], torch.Tensor):
        meta["q_len"] = int(args[0].shape[2])
        meta["n_heads"] = int(args[0].shape[1])
        meta["latent_dim"] = int(args[0].shape[-1])
    elif method_name == "_repeat_kv" and args and isinstance(args[0], torch.Tensor):
        meta["input_kv_heads"] = int(args[0].shape[1])
        meta["seq_len"] = int(args[0].shape[2])
        meta["head_dim"] = int(args[0].shape[-1])
    return meta


def install_hawp_probes(model, tracer: MemoryTracer, *, include_repeat_kv: bool) -> int:
    """Install instance-local probes on HAWPAttention modules.

    The probes live only in this process and are intentionally installed by the
    diagnostic script rather than in production code.
    """
    n_wrapped = 0
    methods = _HAWP_PROBE_METHODS if include_repeat_kv else tuple(
        name for name in _HAWP_PROBE_METHODS if name != "_repeat_kv"
    )

    for mod in model.modules():
        if not isinstance(mod, HAWPAttention):
            continue
        layer_idx = getattr(mod, "layer_idx", None)

        def marker_callback(_module, name, meta, __layer=layer_idx):
            tracer.record(
                f"hawp.layer{__layer}.marker.{name}",
                layer=__layer,
                method="_forward_low_rank_marker",
                marker=name,
                **meta,
            )

        mod._memory_marker_callback = marker_callback
        n_wrapped += 1

        for method_name in methods:
            if not hasattr(mod, method_name):
                continue
            original = getattr(mod, method_name)

            def wrapped(*args, __orig=original, __name=method_name, __layer=layer_idx, **kwargs):
                meta = _method_meta(__name, args)
                tracer.record(f"hawp.layer{__layer}.{__name}.before", layer=__layer, method=__name, **meta)
                out = __orig(*args, **kwargs)
                out_meta = _shape_meta(out)
                if isinstance(out, tuple):
                    out_meta = {"output_shapes": [tuple(x.shape) for x in out if isinstance(x, torch.Tensor)]}
                tracer.record(f"hawp.layer{__layer}.{__name}.after", layer=__layer, method=__name, **out_meta)
                return out

            setattr(mod, method_name, wrapped)
            n_wrapped += 1

    return n_wrapped


def _get_nested_attr(obj: Any, path: str) -> Any:
    cur = obj
    for part in path.split("."):
        if cur is None or not hasattr(cur, part):
            return None
        cur = getattr(cur, part)
    return cur


def _find_transformer_layers(model) -> Any:
    for path in ("model.layers", "model.decoder.layers", "transformer.h", "gpt_neox.layers"):
        layers = _get_nested_attr(model, path)
        if layers is not None:
            return layers
    return None


def _find_final_norm(model) -> tuple[str, Any] | tuple[None, None]:
    for path in (
        "model.norm",
        "model.decoder.final_layer_norm",
        "model.final_layer_norm",
        "transformer.ln_f",
        "gpt_neox.final_layer_norm",
    ):
        mod = _get_nested_attr(model, path)
        if mod is not None:
            return path, mod
    return None, None


def install_model_block_probes(model, tracer: MemoryTracer) -> int:
    """Install coarse model-level probes around blocks, MLP, final norm, and lm_head."""
    n_hooks = 0

    def add_hooks(module, label: str, **meta: Any) -> None:
        nonlocal n_hooks

        def pre_hook(_module, args):
            tracer.record(f"{label}.before", **meta, **_io_meta(args, prefix="input"))

        def post_hook(_module, args, output):
            tracer.record(f"{label}.after", **meta, **_io_meta(output, prefix="output"))

        module.register_forward_pre_hook(pre_hook)
        module.register_forward_hook(post_hook)
        n_hooks += 2

    layers = _find_transformer_layers(model)
    if layers is not None:
        for layer_idx, layer in enumerate(layers):
            add_hooks(layer, f"model.layer{layer_idx}.block", layer=layer_idx, block_part="block")
            self_attn = getattr(layer, "self_attn", None) or getattr(layer, "attention", None) or getattr(layer, "attn", None)
            if self_attn is not None:
                add_hooks(self_attn, f"model.layer{layer_idx}.self_attn", layer=layer_idx, block_part="self_attn")
            mlp = getattr(layer, "mlp", None) or getattr(layer, "feed_forward", None) or getattr(layer, "ffn", None)
            if mlp is not None:
                add_hooks(mlp, f"model.layer{layer_idx}.mlp", layer=layer_idx, block_part="mlp")
                if hasattr(mlp, "_hawp_mlp_marker_callback"):
                    def mlp_marker_callback(_module, name, meta, __layer=layer_idx):
                        tracer.record(
                            f"model.layer{__layer}.mlp.marker.{name}",
                            layer=__layer,
                            block_part="mlp",
                            marker=name,
                            **meta,
                        )

                    mlp._hawp_mlp_marker_callback = mlp_marker_callback

    norm_path, final_norm = _find_final_norm(model)
    if final_norm is not None:
        add_hooks(final_norm, "model.final_norm", module_path=norm_path, block_part="final_norm")

    lm_head = getattr(model, "lm_head", None)
    if lm_head is not None:
        add_hooks(lm_head, "model.lm_head", block_part="lm_head")

    return n_hooks


@torch.inference_mode()
def run_segmented_profile(
    *,
    model,
    tokenizer,
    cfg,
    mode: str,
    prompt: str,
    max_new_tokens: int,
    coordinator=None,
    kv_manager=None,
    reset_fn=None,
    tracer: MemoryTracer,
    trace_decode_steps: int,
):
    if reset_fn is not None:
        reset_fn()
    elif coordinator is not None or kv_manager is not None:
        make_reset_fn(model, coordinator, kv_manager)()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    tracer.reset_delta_baseline()
    tracer.record("profile.start_after_reset", mode=mode)

    use_past_tracker = mode in ("baseline", "hawp_only")
    use_external_past = mode in ("baseline", "hawp_only")
    tracker = PastKVTracker() if use_past_tracker else None

    tracer.record("tokenize.before", mode=mode)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    bsz, prompt_len = input_ids.shape
    tracer.record("tokenize.after_to_device", mode=mode, prompt_len=int(prompt_len))

    total_seen_tokens = prompt_len + max_new_tokens
    prefill_mask = torch.ones(bsz, prompt_len, device=model.device, dtype=torch.long)
    prefill_pos = torch.arange(prompt_len, device=model.device, dtype=torch.long).unsqueeze(0)

    tracer.record("prefill.forward.before", mode=mode, prompt_len=int(prompt_len))
    outputs = prefill_forward_last_logits(
        model,
        input_ids=input_ids,
        attention_mask=prefill_mask,
        position_ids=prefill_pos,
        use_cache=True,
    )
    tracer.record("prefill.forward.after", mode=mode, prompt_len=int(prompt_len))

    if tracker is not None:
        tracker.update(outputs.past_key_values)
        tracer.record("prefill.past_tracker.after", mode=mode)

    if mode == "pure_quant_only" and kv_manager is not None:
        kv_manager.on_forward_done_from_output(outputs.past_key_values, prev_seq_len=0)
        tracer.record("prefill.pure_quant_cache_update.after", mode=mode)

    next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
    generated_ids = next_token
    tracer.record("prefill.argmax.after", mode=mode)

    if coordinator is not None:
        coordinator.on_prefill(prompt_len)
        tracer.record("prefill.coordinator.after", mode=mode)

    past_kv = outputs.past_key_values
    cur_pos = prompt_len

    for step in range(max(0, max_new_tokens - 1)):
        should_trace = step < trace_decode_steps or step == max_new_tokens - 2
        attention_mask = torch.ones(1, cur_pos + 1, device=model.device, dtype=torch.long)
        position_ids = torch.tensor([[cur_pos]], device=model.device, dtype=torch.long)
        fwd_kw: dict[str, Any] = {
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "use_cache": True,
        }

        if mode == "pure_quant_only" and kv_manager is not None:
            tracer.record("decode.get_past_kv.before", mode=mode, step=step, cur_pos=int(cur_pos))
            fwd_kw["past_key_values"] = kv_manager.get_past_kv()
            tracer.record("decode.get_past_kv.after", mode=mode, step=step, cur_pos=int(cur_pos))
        elif use_external_past and past_kv is not None:
            fwd_kw["past_key_values"] = past_kv

        if should_trace:
            tracer.record("decode.forward.before", mode=mode, step=step, cur_pos=int(cur_pos))
        outputs = model(input_ids=next_token, **fwd_kw)
        if should_trace:
            tracer.record("decode.forward.after", mode=mode, step=step, cur_pos=int(cur_pos))

        if tracker is not None:
            tracker.update(outputs.past_key_values)
            if should_trace:
                tracer.record("decode.past_tracker.after", mode=mode, step=step)

        if mode == "pure_quant_only" and kv_manager is not None:
            kv_manager.on_forward_done_from_output(outputs.past_key_values, prev_seq_len=cur_pos)
            if should_trace:
                tracer.record("decode.pure_quant_cache_update.after", mode=mode, step=step)

        past_kv = outputs.past_key_values
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        generated_ids = torch.cat([generated_ids, next_token], dim=1)
        cur_pos += 1
        if should_trace:
            tracer.record("decode.argmax_cat.after", mode=mode, step=step, cur_pos=int(cur_pos))

        if coordinator is not None:
            coordinator.on_new_token()
            if should_trace:
                tracer.record("decode.coordinator.after", mode=mode, step=step)

    full_ids = torch.cat([input_ids, generated_ids], dim=1)
    tracer.record("generation.full_ids_cat.after", mode=mode, total_len=int(full_ids.shape[1]))

    peak_gpu = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    if use_past_tracker:
        impl = "past_kv_baseline" if mode == "baseline" else "past_kv_hawp_only"
        stats = collect_cache_stats_from_tracker(tracker, peak_gpu, impl=impl)
    else:
        stats = collect_cache_stats(model, kv_manager, peak_gpu_bytes=peak_gpu)
    stats.baseline_kv_bytes = compute_baseline_kv_bytes(model, total_seen_tokens)
    tracer.record(
        "cache_stats.collected",
        mode=mode,
        cache_runtime_bytes=int(stats.cache_runtime_bytes),
        cache_runtime=format_nbytes(int(stats.cache_runtime_bytes)),
        baseline_kv_bytes=int(stats.baseline_kv_bytes),
        baseline_kv=format_nbytes(int(stats.baseline_kv_bytes)),
        kv_compression_ratio=round(stats.kv_compression_ratio, 4),
    )

    return stats, generated_ids[0].cpu()


def _print_summary(records: list[dict[str, Any]], *, top_n: int) -> None:
    print("\n[segments] chronological snapshots")
    print(f"{'#':>4} {'label':<52} {'alloc':>12} {'peak':>12} {'d_alloc':>12} {'d_peak':>12}")
    print("-" * 110)
    for rec in records:
        print(
            f"{rec['i']:>4} {rec['label'][:52]:<52} "
            f"{rec['allocated']:>12} {rec['peak_allocated']:>12} "
            f"{rec['delta_allocated']:>12} {rec['delta_peak']:>12}"
        )

    top_peak = sorted(records, key=lambda r: r["peak_allocated_bytes"], reverse=True)[:top_n]
    top_delta = sorted(records, key=lambda r: r["delta_peak_bytes"], reverse=True)[:top_n]

    print("\n[segments] top peak locations")
    for rec in top_peak:
        print(f"  #{rec['i']:>4} peak={rec['peak_allocated']:>12} alloc={rec['allocated']:>12} {rec['label']}")

    print("\n[segments] largest peak increases")
    for rec in top_delta:
        if rec["delta_peak_bytes"] <= 0:
            continue
        print(f"  #{rec['i']:>4} +peak={rec['delta_peak']:>12} peak={rec['peak_allocated']:>12} {rec['label']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Segmented Peak GPU diagnostic profiler")
    parser.add_argument("config", nargs="?", default=None)
    parser.add_argument("--mode", choices=_MODES, default="hawp_quant")
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--trace-decode-steps", type=int, default=4)
    parser.add_argument("--top-n", type=int, default=12)
    parser.add_argument("--no-hawp-internals", action="store_true")
    parser.add_argument("--no-model-block-probes", action="store_true")
    parser.add_argument("--trace-repeat-kv", action="store_true")
    parser.add_argument("--no-synchronize", action="store_true")
    args = parser.parse_args()

    config_path = args.config or Path(__file__).resolve().parent.parent / "configs" / "run_server.yaml"
    cfg = load_config(config_path)
    if args.max_new_tokens is not None:
        cfg.generation.max_new_tokens = args.max_new_tokens
    device = _resolve_device(cfg.train.device)
    tracer = MemoryTracer(synchronize=not args.no_synchronize)

    print("=" * 80)
    print(f"[segments] config={config_path}")
    print(f"[segments] mode={args.mode} seq_len={args.seq_len} max_new_tokens={cfg.generation.max_new_tokens}")
    print(f"[segments] model={cfg.model.model_id} dtype={cfg.model.torch_dtype} device={device}")
    print("=" * 80)

    tracer.record("process.start")
    model, tokenizer, _ = load_baseline_model(cfg)
    tracer.record("model.load_baseline.after")
    model, coordinator, kv_manager = setup_mode(model, cfg, device, args.mode)
    model.eval()
    tracer.record("mode.setup.after", mode=args.mode)

    n_model_probes = 0
    if not args.no_model_block_probes:
        n_model_probes = install_model_block_probes(model, tracer)
        tracer.record("model.block_probes.installed", n_model_probes=n_model_probes)

    n_probes = 0
    if not args.no_hawp_internals:
        n_probes = install_hawp_probes(model, tracer, include_repeat_kv=args.trace_repeat_kv)
        tracer.record("hawp.probes.installed", n_probes=n_probes)

    prompt, actual_seq_len = _build_prompt_for_profile(tokenizer, args.seq_len)
    tracer.record("prompt.built", requested_seq_len=args.seq_len, actual_seq_len=actual_seq_len)

    reset_fn = make_reset_fn(model, coordinator, kv_manager)
    stats, gen_ids = run_segmented_profile(
        model=model,
        tokenizer=tokenizer,
        cfg=cfg,
        mode=args.mode,
        prompt=prompt,
        max_new_tokens=cfg.generation.max_new_tokens,
        coordinator=coordinator,
        kv_manager=kv_manager,
        reset_fn=reset_fn,
        tracer=tracer,
        trace_decode_steps=max(0, args.trace_decode_steps),
    )

    result = {
        "config": str(config_path),
        "mode": args.mode,
        "requested_seq_len": args.seq_len,
        "actual_seq_len": actual_seq_len,
        "max_new_tokens": cfg.generation.max_new_tokens,
        "n_hawp_probes": n_probes,
        "n_model_block_probes": n_model_probes,
        "generated_token_count": int(gen_ids.numel()),
        "stats": {
            "cache_runtime_bytes": stats.cache_runtime_bytes,
            "cache_runtime": format_nbytes(stats.cache_runtime_bytes),
            "cache_compressed_bytes": stats.cache_compressed_bytes,
            "cache_compressed": format_nbytes(stats.cache_compressed_bytes),
            "baseline_kv_bytes": stats.baseline_kv_bytes,
            "baseline_kv": format_nbytes(stats.baseline_kv_bytes),
            "kv_compression_ratio": round(stats.kv_compression_ratio, 4),
            "peak_gpu_bytes": stats.peak_gpu_bytes,
            "peak_gpu": format_nbytes(stats.peak_gpu_bytes),
            "memory_overhead_ratio": round(stats.memory_overhead_ratio, 4),
            "recent_tokens": stats.recent_tokens,
            "archive_tokens": stats.archive_tokens,
            "impl": stats.impl,
        },
        "records": tracer.records,
    }

    _print_summary(tracer.records, top_n=args.top_n)
    print("\n[segments] cache summary")
    print(stats.format_summary())

    output = Path(args.output) if args.output else Path("artifacts/peak_segments") / f"{args.mode}_{actual_seq_len}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    save_json(result, output)
    print(f"\n[segments] saved to {output}")


if __name__ == "__main__":
    main()
