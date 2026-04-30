#!/usr/bin/env python
"""Compare quality, cache, and speed metrics across generation modes.

This script is intended for paper-style experiments where each mode is
evaluated with the same model, prompts, and decode settings.  It writes:

  - metrics_summary.csv
  - metrics_summary.json
  - needle_details.jsonl

LongBench is intentionally left as a later extension because it needs
task-specific prompts, local datasets, and metrics.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import time
from pathlib import Path
from typing import Any

import torch

from hawp_laq.config import load_config
from hawp_laq.eval.needle import needle_accuracy, run_needle_test
from hawp_laq.eval.perplexity import compute_stepwise_ppl
from hawp_laq.runtime.generate import _resolve_device, load_baseline_model
from hawp_laq.runtime.mode_runner import make_reset_fn, profile_generate_by_mode, setup_mode
from hawp_laq.utils.io import save_json
from hawp_laq.utils.memory import format_nbytes


_SUPPORTED_MODES = (
    "baseline",
    "hawp_only",
    "quant_only",
    "pure_quant_only",
    "hawp_quant",
    "hawp_quant_all",
    "hawp_quant_sched",
)


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


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if v != v or v in (float("inf"), float("-inf")):
        return None
    return v


def _round_or_none(value: Any, ndigits: int = 6) -> float | None:
    v = _safe_float(value)
    return round(v, ndigits) if v is not None else None


def _stats_to_dict(stats) -> dict[str, Any]:
    return {
        "impl": stats.impl,
        "cache_tokens_total": stats.cache_tokens_total,
        "cache_runtime_bytes": stats.cache_runtime_bytes,
        "cache_runtime_formatted": format_nbytes(stats.cache_runtime_bytes),
        "cache_compressed_bytes": stats.cache_compressed_bytes,
        "cache_compressed_formatted": format_nbytes(stats.cache_compressed_bytes),
        "baseline_kv_bytes": stats.baseline_kv_bytes,
        "baseline_kv_formatted": format_nbytes(stats.baseline_kv_bytes),
        "kv_compression_ratio": _round_or_none(stats.kv_compression_ratio, 4),
        "bytes_per_token": _round_or_none(stats.bytes_per_token, 4),
        "recent_tokens": stats.recent_tokens,
        "recent_ratio": _round_or_none(stats.recent_ratio, 6),
        "archive_tokens": stats.archive_tokens,
        "archive_ratio": _round_or_none(stats.archive_ratio, 6),
        "meta_bytes": stats.meta_bytes,
        "peak_gpu_bytes": stats.peak_gpu_bytes,
        "peak_gpu_formatted": format_nbytes(stats.peak_gpu_bytes),
        "memory_overhead_ratio": _round_or_none(stats.memory_overhead_ratio, 4),
    }


def _cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _run_ppl(model, tokenizer, cfg, mode: str, coordinator, kv_manager, reset_fn, device: str) -> dict[str, Any]:
    ppl_cfg = cfg.eval.ppl
    print(f"[{mode}] PPL stepwise: seq_len={ppl_cfg.seq_len} nsamples={ppl_cfg.nsamples}")
    result = compute_stepwise_ppl(
        model,
        tokenizer,
        coordinator=coordinator,
        kv_manager=kv_manager,
        reset_fn=reset_fn,
        seq_len=ppl_cfg.seq_len,
        nsamples=ppl_cfg.nsamples,
        device=device,
        use_past_kv=mode in ("baseline", "hawp_only"),
    )
    print(
        f"[{mode}] PPL={result.get('perplexity', float('nan')):.4f} "
        f"NLL={result.get('nll', float('nan')):.4f} "
        f"tokens={result.get('n_tokens', 0)}"
    )
    return result


def _run_needle(model, tokenizer, cfg, mode: str, coordinator, kv_manager, device: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    needle_cfg = cfg.eval.needle
    old_max_new_tokens = cfg.generation.max_new_tokens
    cfg.generation.max_new_tokens = needle_cfg.max_new_tokens
    reset_fn = make_reset_fn(model, coordinator, kv_manager)

    def generate_fn(prompt: str) -> str:
        _, _, gen_ids = profile_generate_by_mode(
            model,
            tokenizer,
            [prompt],
            cfg,
            mode,
            coordinator=coordinator,
            kv_manager=kv_manager,
            reset_fn=reset_fn,
        )
        return tokenizer.decode(gen_ids[0], skip_special_tokens=True)

    print(
        f"[{mode}] Needle: context_lens={needle_cfg.context_lens} "
        f"depths={needle_cfg.depths} max_new_tokens={needle_cfg.max_new_tokens}"
    )
    try:
        details = run_needle_test(
            model,
            tokenizer,
            context_lens=needle_cfg.context_lens,
            depths=needle_cfg.depths,
            device=device,
            max_new_tokens=needle_cfg.max_new_tokens,
            generate_fn=generate_fn,
            reset_fn=reset_fn,
        )
    finally:
        cfg.generation.max_new_tokens = old_max_new_tokens

    summary = needle_accuracy(details)
    overall = summary.get("overall", {})
    print(f"[{mode}] Needle recall={overall.get('accuracy', 0.0):.2%}")
    return summary, details


def _run_speed_profile(model, tokenizer, cfg, mode: str, coordinator, kv_manager) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    speed_cfg = cfg.eval.speed
    old_max_new_tokens = cfg.generation.max_new_tokens
    cfg.generation.max_new_tokens = speed_cfg.max_new_tokens
    reset_fn = make_reset_fn(model, coordinator, kv_manager)

    details: list[dict[str, Any]] = []
    try:
        for requested_seq_len in speed_cfg.seq_lens:
            prompt, actual_seq_len = _build_prompt_for_profile(tokenizer, requested_seq_len)
            print(
                f"[{mode}] Speed/KV: requested_seq_len={requested_seq_len} "
                f"actual_seq_len={actual_seq_len} max_new_tokens={speed_cfg.max_new_tokens}"
            )

            start = time.perf_counter()
            _, stats, _ = profile_generate_by_mode(
                model,
                tokenizer,
                [prompt],
                cfg,
                mode,
                coordinator=coordinator,
                kv_manager=kv_manager,
                reset_fn=reset_fn,
            )
            elapsed_s = time.perf_counter() - start
            tokens_per_s = speed_cfg.max_new_tokens / elapsed_s if elapsed_s > 0 else 0.0

            entry = {
                "mode": mode,
                "requested_seq_len": requested_seq_len,
                "seq_len": actual_seq_len,
                "max_new_tokens": speed_cfg.max_new_tokens,
                "elapsed_s": round(elapsed_s, 4),
                "tokens_per_s": round(tokens_per_s, 4),
                **_stats_to_dict(stats),
            }
            details.append(entry)
            print(
                f"[{mode}] seq={actual_seq_len} tok/s={tokens_per_s:.2f} "
                f"cache={entry['cache_runtime_formatted']} "
                f"archive={entry['cache_compressed_formatted']} "
                f"peak={entry['peak_gpu_formatted']}"
            )
    finally:
        cfg.generation.max_new_tokens = old_max_new_tokens

    if not details:
        return {}, details

    largest = max(details, key=lambda item: item["seq_len"])
    mean_tok_s = sum(d["tokens_per_s"] for d in details) / len(details)
    aggregate = {
        "speed_tokens_per_s_mean": round(mean_tok_s, 4),
        "speed_profile_seq_len": largest["seq_len"],
        "speed_profile_requested_seq_len": largest["requested_seq_len"],
        "speed_profile_elapsed_s": largest["elapsed_s"],
        "speed_profile_tokens_per_s": largest["tokens_per_s"],
        "cache_tokens_total": largest["cache_tokens_total"],
        "cache_runtime_bytes": largest["cache_runtime_bytes"],
        "cache_runtime_formatted": largest["cache_runtime_formatted"],
        "cache_compressed_bytes": largest["cache_compressed_bytes"],
        "cache_compressed_formatted": largest["cache_compressed_formatted"],
        "baseline_kv_bytes": largest["baseline_kv_bytes"],
        "baseline_kv_formatted": largest["baseline_kv_formatted"],
        "kv_compression_ratio": largest["kv_compression_ratio"],
        "bytes_per_token": largest["bytes_per_token"],
        "peak_gpu_bytes": max(d["peak_gpu_bytes"] for d in details),
        "peak_gpu_formatted": format_nbytes(max(d["peak_gpu_bytes"] for d in details)),
        "recent_tokens": largest["recent_tokens"],
        "recent_ratio": largest["recent_ratio"],
        "archive_tokens": largest["archive_tokens"],
        "archive_ratio": largest["archive_ratio"],
        "impl": largest["impl"],
    }
    return aggregate, details


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "mode",
        "status",
        "ppl",
        "delta_ppl",
        "nll",
        "needle_recall",
        "delta_needle_recall",
        "speed_tokens_per_s_mean",
        "speed_profile_seq_len",
        "speed_profile_tokens_per_s",
        "cache_runtime_bytes",
        "cache_compressed_bytes",
        "baseline_kv_bytes",
        "kv_compression_ratio",
        "bytes_per_token",
        "peak_gpu_bytes",
        "recent_ratio",
        "archive_ratio",
        "impl",
        "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _print_summary(rows: list[dict[str, Any]]) -> None:
    print("\n" + "=" * 112)
    print(
        f"{'mode':>16} {'status':>10} {'PPL':>10} {'dPPL':>10} "
        f"{'needle':>9} {'KVx':>8} {'cache':>12} {'archive':>12} "
        f"{'peak':>12} {'tok/s':>9}"
    )
    print("-" * 112)
    for row in rows:
        ppl = row.get("ppl")
        dppl = row.get("delta_ppl")
        needle = row.get("needle_recall")
        kvx = row.get("kv_compression_ratio")
        tok_s = row.get("speed_tokens_per_s_mean")
        print(
            f"{row.get('mode', ''):>16} {row.get('status', ''):>10} "
            f"{ppl if ppl is not None else 'NA':>10} "
            f"{dppl if dppl is not None else 'NA':>10} "
            f"{needle if needle is not None else 'NA':>9} "
            f"{kvx if kvx is not None else 'NA':>8} "
            f"{row.get('cache_runtime_formatted', 'NA'):>12} "
            f"{row.get('cache_compressed_formatted', 'NA'):>12} "
            f"{row.get('peak_gpu_formatted', 'NA'):>12} "
            f"{tok_s if tok_s is not None else 'NA':>9}"
        )


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description="Compare HAWP-LAQ evaluation metrics across modes")
    parser.add_argument("config", nargs="?", default=None)
    parser.add_argument("--modes", nargs="+", choices=_SUPPORTED_MODES, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--skip-ppl", action="store_true")
    parser.add_argument("--skip-needle", action="store_true")
    parser.add_argument("--skip-speed", action="store_true")
    args = parser.parse_args()

    config_path = args.config or Path(__file__).resolve().parent.parent / "configs" / "run_server.yaml"
    cfg = load_config(config_path)
    modes = args.modes if args.modes is not None else list(cfg.eval.modes)
    output_dir = Path(args.output_dir) if args.output_dir else Path(cfg.eval.output_dir)
    device = _resolve_device(cfg.train.device)

    unknown_modes = [m for m in modes if m not in _SUPPORTED_MODES]
    if unknown_modes:
        raise ValueError(f"Unsupported eval modes: {unknown_modes}")

    print("=" * 80)
    print(f"[eval] config={config_path}")
    print(f"[eval] model={cfg.model.model_id}")
    print(f"[eval] load_in_4bit={cfg.model.load_in_4bit} dtype={cfg.model.torch_dtype} device={device}")
    print(f"[eval] modes={modes}")
    print(f"[eval] output_dir={output_dir}")
    print("=" * 80)

    if cfg.eval.longbench.enabled:
        print("[longbench] enabled in config, but this script currently leaves LongBench as a later extension.")

    summary_rows: list[dict[str, Any]] = []
    all_speed_details: list[dict[str, Any]] = []
    all_needle_details: list[dict[str, Any]] = []

    for mode in modes:
        row: dict[str, Any] = {"mode": mode, "status": "ok", "error": ""}
        model = tokenizer = coordinator = kv_manager = None
        try:
            print("\n" + "=" * 80)
            print(f"[{mode}] loading and setting up")
            model, tokenizer, _ = load_baseline_model(cfg)
            model, coordinator, kv_manager = setup_mode(model, cfg, device, mode)
            model.eval()
            reset_fn = make_reset_fn(model, coordinator, kv_manager)

            if not args.skip_ppl:
                ppl = _run_ppl(model, tokenizer, cfg, mode, coordinator, kv_manager, reset_fn, device)
                row.update({
                    "ppl": _round_or_none(ppl.get("perplexity"), 6),
                    "nll": _round_or_none(ppl.get("nll"), 6),
                    "ppl_n_chunks": ppl.get("n_chunks"),
                    "ppl_n_tokens": ppl.get("n_tokens"),
                    "ppl_nll_prefill": _round_or_none(ppl.get("nll_prefill"), 6),
                    "ppl_nll_decode": _round_or_none(ppl.get("nll_decode"), 6),
                })

            if not args.skip_needle:
                needle_summary, needle_details = _run_needle(
                    model, tokenizer, cfg, mode, coordinator, kv_manager, device,
                )
                overall = needle_summary.get("overall", {})
                row["needle_recall"] = _round_or_none(overall.get("accuracy"), 6)
                row["needle_n"] = overall.get("n")
                row["needle_summary"] = needle_summary
                for detail in needle_details:
                    all_needle_details.append({"mode": mode, **detail})

            if not args.skip_speed:
                speed_agg, speed_details = _run_speed_profile(
                    model, tokenizer, cfg, mode, coordinator, kv_manager,
                )
                row.update(speed_agg)
                all_speed_details.extend(speed_details)

        except (RuntimeError, ValueError, FileNotFoundError, OSError) as exc:
            row["status"] = "skipped"
            row["error"] = repr(exc)
            print(f"[{mode}] SKIPPED: {exc}")
        finally:
            del model, tokenizer, coordinator, kv_manager
            _cleanup_cuda()

        summary_rows.append(row)

    baseline = next((row for row in summary_rows if row.get("mode") == "baseline" and row.get("status") == "ok"), None)
    baseline_ppl = _safe_float(baseline.get("ppl")) if baseline else None
    baseline_needle = _safe_float(baseline.get("needle_recall")) if baseline else None
    for row in summary_rows:
        ppl = _safe_float(row.get("ppl"))
        needle = _safe_float(row.get("needle_recall"))
        row["delta_ppl"] = round(ppl - baseline_ppl, 6) if ppl is not None and baseline_ppl is not None else None
        row["delta_needle_recall"] = (
            round(needle - baseline_needle, 6)
            if needle is not None and baseline_needle is not None
            else None
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "metrics_summary.csv"
    json_path = output_dir / "metrics_summary.json"
    needle_path = output_dir / "needle_details.jsonl"
    speed_path = output_dir / "speed_details.json"

    _write_csv(summary_rows, csv_path)
    save_json(
        {
            "config": str(config_path),
            "model_id": cfg.model.model_id,
            "load_in_4bit": cfg.model.load_in_4bit,
            "torch_dtype": cfg.model.torch_dtype,
            "modes": modes,
            "summary": summary_rows,
            "speed_details": all_speed_details,
        },
        json_path,
    )
    _write_jsonl(all_needle_details, needle_path)
    save_json(all_speed_details, speed_path)

    _print_summary(summary_rows)
    print(f"\n[eval] wrote {csv_path}")
    print(f"[eval] wrote {json_path}")
    print(f"[eval] wrote {needle_path}")
    print(f"[eval] wrote {speed_path}")


if __name__ == "__main__":
    main()
