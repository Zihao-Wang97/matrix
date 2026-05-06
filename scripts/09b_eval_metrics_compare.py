#!/usr/bin/env python
"""Compare quality, cache, speed, and LongBench-E metrics across modes.

This script is intended for paper-style experiments where each mode is
evaluated with the same model, prompts, and decode settings.  It writes a
top-level aggregate plus one subdirectory per mode, each containing:

  - metrics_summary.csv
  - metrics_summary.json
  - needle_details.jsonl
  - speed_details.json
  - longbench_predictions.jsonl
  - longbench_scores.json
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from hawp_laq.config import load_config
from hawp_laq.eval.distribution import compute_distribution_metrics, ideal_distribution_metrics
from hawp_laq.eval.longbench import (
    LONGBENCH_E_CATEGORIES,
    LONGBENCH_E_TASKS,
    iter_longbench_samples,
    length_bin,
    score_prediction,
    summarize_longbench,
)
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


def _get_distribution_cfg(cfg):
    return getattr(
        cfg.eval,
        "distribution",
        SimpleNamespace(
            enabled=True,
            seq_len=512,
            nsamples=8,
            top_k=[1, 5, 10],
            seed=0,
        ),
    )


def _get_longbench_cfg(cfg):
    return getattr(
        cfg.eval,
        "longbench",
        SimpleNamespace(
            enabled=False,
            data_dir=Path("data/longbench"),
            tasks=[],
            max_new_tokens=128,
        ),
    )


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


def _run_distribution(model, tokenizer, cfg, mode: str, coordinator, kv_manager, reset_fn, device: str) -> dict[str, Any]:
    dist_cfg = _get_distribution_cfg(cfg)
    if not dist_cfg.enabled:
        return {}
    if mode == "baseline":
        return ideal_distribution_metrics(
            dist_cfg.top_k,
            seq_len=dist_cfg.seq_len,
            nsamples=dist_cfg.nsamples,
        )

    print(
        f"[{mode}] Distribution: seq_len={dist_cfg.seq_len} "
        f"nsamples={dist_cfg.nsamples} top_k={dist_cfg.top_k} seed={dist_cfg.seed}"
    )
    baseline_model = None
    baseline_tokenizer = None
    try:
        baseline_model, baseline_tokenizer, _ = load_baseline_model(cfg)
        baseline_model.eval()
        baseline_reset_fn = make_reset_fn(baseline_model)
        result = compute_distribution_metrics(
            baseline_model,
            model,
            tokenizer,
            seq_len=dist_cfg.seq_len,
            nsamples=dist_cfg.nsamples,
            top_k=dist_cfg.top_k,
            seed=dist_cfg.seed,
            device=device,
            baseline_reset_fn=baseline_reset_fn,
            candidate_reset_fn=reset_fn,
            baseline_use_past_kv=True,
            candidate_use_past_kv=mode in ("baseline", "hawp_only"),
            candidate_coordinator=coordinator,
            candidate_kv_manager=kv_manager,
        )
        last_k = dist_cfg.top_k[-1] if dist_cfg.top_k else 1
        last_topk_key = f"top{last_k}_overlap"
        print(
            f"[{mode}] KL_mean={result.get('kl_mean')} "
            f"argmax={result.get('argmax_agreement')} "
            f"top{last_k}={result.get(last_topk_key)}"
        )
        return result
    finally:
        del baseline_model, baseline_tokenizer
        _cleanup_cuda()


def _longbench_field_name(category: str) -> str:
    return "longbench_" + category.lower().replace("-", "_")


def _mean_int(rows: list[dict[str, Any]], key: str) -> int | None:
    values = [
        _safe_float(row.get(key))
        for row in rows
        if _safe_float(row.get(key)) is not None
    ]
    if not values:
        return None
    return int(sum(values) / len(values))


def _mean_float(rows: list[dict[str, Any]], key: str, ndigits: int = 4) -> float | None:
    values = [
        _safe_float(row.get(key))
        for row in rows
        if _safe_float(row.get(key)) is not None
    ]
    if not values:
        return None
    return round(sum(values) / len(values), ndigits)


def _run_longbench(
    model,
    tokenizer,
    cfg,
    mode: str,
    coordinator,
    kv_manager,
    *,
    tasks: list[str],
    data_dir: Path,
    max_samples_per_task: int | None,
    max_input_tokens: int,
    max_new_tokens_cap: int | None,
    chat_template: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    reset_fn = make_reset_fn(model, coordinator, kv_manager)
    samples = iter_longbench_samples(
        data_dir,
        tasks=tasks,
        tokenizer=tokenizer,
        max_input_tokens=max_input_tokens,
        max_samples_per_task=max_samples_per_task,
        chat_template=chat_template,
    )
    if not samples:
        return {"longbench_n": 0}, [], {}

    print(
        f"[{mode}] LongBench-E: tasks={tasks} samples={len(samples)} "
        f"max_input_tokens={max_input_tokens} max_samples_per_task={max_samples_per_task}"
    )

    old_max_new_tokens = cfg.generation.max_new_tokens
    predictions: list[dict[str, Any]] = []
    try:
        for i, sample in enumerate(samples, start=1):
            max_new_tokens = sample.max_new_tokens
            if max_new_tokens_cap is not None and max_new_tokens_cap > 0:
                max_new_tokens = min(max_new_tokens, max_new_tokens_cap)
            cfg.generation.max_new_tokens = max_new_tokens

            prompt_len = tokenizer(sample.prompt, return_tensors="pt").input_ids.shape[1]
            print(
                f"[{mode}] LongBench-E {i}/{len(samples)} "
                f"task={sample.task} idx={sample.index} prompt_tokens={prompt_len} "
                f"max_new={max_new_tokens}"
            )

            _, stats, gen_ids = profile_generate_by_mode(
                model,
                tokenizer,
                [sample.prompt],
                cfg,
                mode,
                coordinator=coordinator,
                kv_manager=kv_manager,
                reset_fn=reset_fn,
            )
            prediction = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
            score = score_prediction(
                sample.task,
                prediction,
                sample.answers,
                sample.all_classes,
            )
            stat_dict = _stats_to_dict(stats)
            row = {
                "mode": mode,
                "task": sample.task,
                "index": sample.index,
                "length": sample.length,
                "length_bin": length_bin(sample.length),
                "prompt_tokens": prompt_len,
                "max_new_tokens": max_new_tokens,
                "prediction": prediction,
                "answers": sample.answers,
                "score": round(score, 6),
                "score_pct": round(100.0 * score, 4),
                **stat_dict,
            }
            predictions.append(row)
            print(
                f"[{mode}] LongBench-E task={sample.task} idx={sample.index} "
                f"score={100.0 * score:.2f} cache={row['cache_runtime_formatted']} "
                f"KVx={row['kv_compression_ratio']}"
            )
    finally:
        cfg.generation.max_new_tokens = old_max_new_tokens

    summary = summarize_longbench(predictions)
    category_scores = summary.get("category_scores", {})
    row = {
        "longbench_average": _round_or_none(summary.get("average"), 4),
        "longbench_task_average": _round_or_none(summary.get("task_average"), 4),
        "longbench_n": summary.get("n"),
        "longbench_cache_runtime_bytes_mean": _mean_int(predictions, "cache_runtime_bytes"),
        "longbench_cache_runtime_bytes_max": max(
            (int(p.get("cache_runtime_bytes", 0)) for p in predictions),
            default=None,
        ),
        "longbench_cache_compressed_bytes_mean": _mean_int(predictions, "cache_compressed_bytes"),
        "longbench_baseline_kv_bytes_mean": _mean_int(predictions, "baseline_kv_bytes"),
        "longbench_kv_compression_ratio_mean": _mean_float(predictions, "kv_compression_ratio"),
        "longbench_peak_gpu_bytes_max": max(
            (int(p.get("peak_gpu_bytes", 0)) for p in predictions),
            default=None,
        ),
    }
    if row["longbench_cache_runtime_bytes_mean"] is not None:
        row["longbench_cache_runtime_formatted_mean"] = format_nbytes(
            row["longbench_cache_runtime_bytes_mean"]
        )
    if row["longbench_peak_gpu_bytes_max"] is not None:
        row["longbench_peak_gpu_formatted_max"] = format_nbytes(
            row["longbench_peak_gpu_bytes_max"]
        )
    for category in LONGBENCH_E_CATEGORIES:
        score = category_scores.get(category, {}).get("score")
        row[f"{_longbench_field_name(category)}_score"] = _round_or_none(score, 4)

    print(
        f"[{mode}] LongBench-E Average={row['longbench_average']} "
        f"SingleQA={row.get('longbench_singleqa_score')} "
        f"MultiQA={row.get('longbench_multiqa_score')} "
        f"Summarization={row.get('longbench_summarization_score')} "
        f"Few-shot={row.get('longbench_few_shot_score')} "
        f"Synthetic={row.get('longbench_synthetic_score')} "
        f"Code={row.get('longbench_code_score')}"
    )
    return row, predictions, summary


def _sort_topk_fields(fields: set[str]) -> list[str]:
    def _key(name: str) -> int:
        try:
            return int(name[3:name.index("_overlap")])
        except (ValueError, IndexError):
            return 0
    return sorted(fields, key=_key)


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    topk_fields = _sort_topk_fields({
        key
        for row in rows
        for key in row
        if key.startswith("top") and key.endswith("_overlap")
    })
    fieldnames = [
        "mode",
        "status",
        "ppl",
        "delta_ppl",
        "nll",
        "kl_mean",
        "kl_p95",
        "kl_max",
        "argmax_agreement",
        *topk_fields,
        "distribution_n_chunks",
        "distribution_n_tokens",
        "distribution_error",
        "longbench_singleqa_score",
        "longbench_multiqa_score",
        "longbench_summarization_score",
        "longbench_few_shot_score",
        "longbench_synthetic_score",
        "longbench_code_score",
        "longbench_average",
        "longbench_task_average",
        "longbench_n",
        "longbench_cache_runtime_bytes_mean",
        "longbench_cache_runtime_bytes_max",
        "longbench_cache_compressed_bytes_mean",
        "longbench_baseline_kv_bytes_mean",
        "longbench_kv_compression_ratio_mean",
        "longbench_peak_gpu_bytes_max",
        "longbench_error",
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


def _write_eval_outputs(
    *,
    rows: list[dict[str, Any]],
    output_dir: Path,
    config_path: str | Path,
    cfg,
    modes: list[str],
    speed_details: list[dict[str, Any]],
    needle_details: list[dict[str, Any]],
    longbench_predictions: list[dict[str, Any]],
    longbench_scores: dict[str, Any],
    run_longbench: bool,
    longbench_tasks: list[str],
    longbench_data_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "metrics_summary.csv"
    json_path = output_dir / "metrics_summary.json"
    needle_path = output_dir / "needle_details.jsonl"
    speed_path = output_dir / "speed_details.json"
    longbench_pred_path = output_dir / "longbench_predictions.jsonl"
    longbench_score_path = output_dir / "longbench_scores.json"

    _write_csv(rows, csv_path)
    save_json(
        {
            "config": str(config_path),
            "model_id": cfg.model.model_id,
            "load_in_4bit": cfg.model.load_in_4bit,
            "torch_dtype": cfg.model.torch_dtype,
            "modes": modes,
            "summary": rows,
            "speed_details": speed_details,
            "longbench_enabled": run_longbench,
            "longbench_tasks": longbench_tasks,
            "longbench_data_dir": str(longbench_data_dir),
        },
        json_path,
    )
    _write_jsonl(needle_details, needle_path)
    save_json(speed_details, speed_path)
    _write_jsonl(longbench_predictions, longbench_pred_path)
    save_json(longbench_scores, longbench_score_path)

    return {
        "csv": csv_path,
        "json": json_path,
        "needle": needle_path,
        "speed": speed_path,
        "longbench_predictions": longbench_pred_path,
        "longbench_scores": longbench_score_path,
    }


def _write_outputs_by_mode(
    *,
    rows: list[dict[str, Any]],
    output_dir: Path,
    config_path: str | Path,
    cfg,
    speed_details: list[dict[str, Any]],
    needle_details: list[dict[str, Any]],
    longbench_predictions: list[dict[str, Any]],
    longbench_scores_by_mode: dict[str, Any],
    run_longbench: bool,
    longbench_tasks: list[str],
    longbench_data_dir: Path,
) -> list[dict[str, Any]]:
    written = []
    for row in rows:
        mode = row.get("mode")
        if not mode:
            continue
        mode_rows = [row]
        mode_speed = [item for item in speed_details if item.get("mode") == mode]
        mode_needle = [item for item in needle_details if item.get("mode") == mode]
        mode_longbench_predictions = [
            item for item in longbench_predictions if item.get("mode") == mode
        ]
        mode_longbench_scores = (
            {mode: longbench_scores_by_mode[mode]}
            if mode in longbench_scores_by_mode
            else {}
        )
        paths = _write_eval_outputs(
            rows=mode_rows,
            output_dir=output_dir / str(mode),
            config_path=config_path,
            cfg=cfg,
            modes=[str(mode)],
            speed_details=mode_speed,
            needle_details=mode_needle,
            longbench_predictions=mode_longbench_predictions,
            longbench_scores=mode_longbench_scores,
            run_longbench=run_longbench,
            longbench_tasks=longbench_tasks,
            longbench_data_dir=longbench_data_dir,
        )
        written.append({"mode": mode, **paths})
    return written


def _print_summary(rows: list[dict[str, Any]]) -> None:
    print("\n" + "=" * 132)
    print(
        f"{'mode':>16} {'status':>10} {'PPL':>10} {'dPPL':>10} "
        f"{'KL':>10} {'argmax':>9} {'LB avg':>9} "
        f"{'needle':>9} {'KVx':>8} {'cache':>12} {'archive':>12} "
        f"{'peak':>12} {'tok/s':>9}"
    )
    print("-" * 132)
    for row in rows:
        ppl = row.get("ppl")
        dppl = row.get("delta_ppl")
        kl = row.get("kl_mean")
        argmax = row.get("argmax_agreement")
        lb_avg = row.get("longbench_average")
        needle = row.get("needle_recall")
        kvx = row.get("kv_compression_ratio")
        tok_s = row.get("speed_tokens_per_s_mean")
        print(
            f"{row.get('mode', ''):>16} {row.get('status', ''):>10} "
            f"{ppl if ppl is not None else 'NA':>10} "
            f"{dppl if dppl is not None else 'NA':>10} "
            f"{kl if kl is not None else 'NA':>10} "
            f"{argmax if argmax is not None else 'NA':>9} "
            f"{lb_avg if lb_avg is not None else 'NA':>9} "
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
    parser.add_argument("--skip-distribution", action="store_true")
    parser.add_argument("--run-longbench", action="store_true")
    parser.add_argument("--skip-longbench", action="store_true")
    parser.add_argument("--only-longbench", action="store_true")
    parser.add_argument("--longbench-data-dir", default=None)
    parser.add_argument("--longbench-tasks", nargs="+", default=None)
    parser.add_argument("--longbench-max-samples-per-task", type=int, default=None)
    parser.add_argument("--longbench-max-input-tokens", type=int, default=8192)
    parser.add_argument("--longbench-max-new-tokens", type=int, default=None)
    parser.add_argument("--longbench-chat-template", choices=["auto", "always", "never"], default="auto")
    args = parser.parse_args()

    config_path = args.config or Path(__file__).resolve().parent.parent / "configs" / "run_server.yaml"
    cfg = load_config(config_path)
    modes = args.modes if args.modes is not None else list(cfg.eval.modes)
    output_dir = Path(args.output_dir) if args.output_dir else Path(cfg.eval.output_dir)
    device = _resolve_device(cfg.train.device)
    longbench_cfg = _get_longbench_cfg(cfg)
    run_longbench = (
        not args.skip_longbench
        and (args.run_longbench or args.only_longbench or bool(getattr(longbench_cfg, "enabled", False)))
    )
    longbench_data_dir = (
        Path(args.longbench_data_dir)
        if args.longbench_data_dir
        else Path(getattr(longbench_cfg, "data_dir", "data/longbench"))
    )
    longbench_tasks = (
        args.longbench_tasks
        if args.longbench_tasks is not None
        else list(getattr(longbench_cfg, "tasks", []) or LONGBENCH_E_TASKS)
    )

    if args.only_longbench:
        args.skip_ppl = True
        args.skip_needle = True
        args.skip_speed = True
        args.skip_distribution = True

    unknown_modes = [m for m in modes if m not in _SUPPORTED_MODES]
    if unknown_modes:
        raise ValueError(f"Unsupported eval modes: {unknown_modes}")

    print("=" * 80)
    print(f"[eval] config={config_path}")
    print(f"[eval] model={cfg.model.model_id}")
    print(f"[eval] load_in_4bit={cfg.model.load_in_4bit} dtype={cfg.model.torch_dtype} device={device}")
    print(f"[eval] modes={modes}")
    print(f"[eval] output_dir={output_dir}")
    print(f"[eval] longbench_enabled={run_longbench} data_dir={longbench_data_dir}")
    print("=" * 80)

    summary_rows: list[dict[str, Any]] = []
    all_speed_details: list[dict[str, Any]] = []
    all_needle_details: list[dict[str, Any]] = []
    all_longbench_predictions: list[dict[str, Any]] = []
    longbench_scores_by_mode: dict[str, Any] = {}

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

            dist_cfg = _get_distribution_cfg(cfg)
            if not args.skip_distribution and dist_cfg.enabled:
                try:
                    dist = _run_distribution(
                        model, tokenizer, cfg, mode, coordinator, kv_manager, reset_fn, device,
                    )
                    row.update({
                        key: _round_or_none(value, 6)
                        if isinstance(value, float)
                        else value
                        for key, value in dist.items()
                    })
                except (RuntimeError, ValueError, FileNotFoundError, OSError) as exc:
                    row["distribution_error"] = repr(exc)
                    print(f"[{mode}] distribution metrics skipped: {exc}")
                    _cleanup_cuda()

            if run_longbench:
                try:
                    lb_row, lb_predictions, lb_summary = _run_longbench(
                        model,
                        tokenizer,
                        cfg,
                        mode,
                        coordinator,
                        kv_manager,
                        tasks=longbench_tasks,
                        data_dir=longbench_data_dir,
                        max_samples_per_task=args.longbench_max_samples_per_task,
                        max_input_tokens=args.longbench_max_input_tokens,
                        max_new_tokens_cap=args.longbench_max_new_tokens,
                        chat_template=args.longbench_chat_template,
                    )
                    row.update(lb_row)
                    all_longbench_predictions.extend(lb_predictions)
                    longbench_scores_by_mode[mode] = lb_summary
                except (RuntimeError, ValueError, FileNotFoundError, OSError) as exc:
                    row["longbench_error"] = repr(exc)
                    print(f"[{mode}] LongBench-E skipped: {exc}")
                    _cleanup_cuda()

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

    top_paths = _write_eval_outputs(
        rows=summary_rows,
        output_dir=output_dir,
        config_path=config_path,
        cfg=cfg,
        modes=modes,
        speed_details=all_speed_details,
        needle_details=all_needle_details,
        longbench_predictions=all_longbench_predictions,
        longbench_scores=longbench_scores_by_mode,
        run_longbench=run_longbench,
        longbench_tasks=longbench_tasks,
        longbench_data_dir=longbench_data_dir,
    )
    mode_paths = _write_outputs_by_mode(
        rows=summary_rows,
        output_dir=output_dir,
        config_path=config_path,
        cfg=cfg,
        speed_details=all_speed_details,
        needle_details=all_needle_details,
        longbench_predictions=all_longbench_predictions,
        longbench_scores_by_mode=longbench_scores_by_mode,
        run_longbench=run_longbench,
        longbench_tasks=longbench_tasks,
        longbench_data_dir=longbench_data_dir,
    )

    _print_summary(summary_rows)
    print(f"\n[eval] wrote {top_paths['csv']}")
    print(f"[eval] wrote {top_paths['json']}")
    print(f"[eval] wrote {top_paths['needle']}")
    print(f"[eval] wrote {top_paths['speed']}")
    print(f"[eval] wrote {top_paths['longbench_predictions']}")
    print(f"[eval] wrote {top_paths['longbench_scores']}")
    for item in mode_paths:
        print(f"[eval:{item['mode']}] wrote {item['csv'].parent}")


if __name__ == "__main__":
    main()
