#!/usr/bin/env python
"""Evaluate batched scoring for all question files in one subject.

This is the subject-level companion to ``10d_eval_scoring_task_batch.py``.
It keeps one model loaded per mode, then evaluates each question/file_id with
batched prompts. Answers from different questions are never mixed in one prompt.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from hawp_laq.config import load_config
from hawp_laq.runtime.generate import _resolve_device, load_baseline_model
from hawp_laq.runtime.mode_runner import make_reset_fn, setup_mode


def _load_module(script_name: str, module_name: str):
    script_path = Path(__file__).resolve().parent / script_name
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


batch_eval = _load_module("10d_eval_scoring_task_batch.py", "scoring_task_batch_eval")
scoring = batch_eval.scoring

_SUPPORTED_MODES = ("baseline", "quant_only", "hawp_quant")


def _split_items(raw: list[str] | None) -> set[str]:
    items: set[str] = set()
    for value in raw or []:
        items.update(part.strip() for part in str(value).split(",") if part.strip())
    return items


def _subject_matches(row_subject: str, file_id: str, subject: str) -> bool:
    if not subject or subject.lower() in {"all", "*"}:
        return True
    return row_subject == subject or subject in row_subject or file_id.startswith(subject)


def discover_question_rows(
    question_file: Path,
    answer_dir: Path,
    *,
    subject: str,
    include_file_ids: set[str],
    exclude_file_ids: set[str],
    max_files: int | None,
) -> list[dict[str, str]]:
    question_rows = scoring._row_dicts(scoring.read_xlsx_first_sheet(question_file))
    selected: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in question_rows:
        row_subject = scoring._first_present(row, ["subject"], fallback_idx=0)
        file_id = scoring._first_present(row, ["file"], fallback_idx=1)
        if not file_id or file_id in seen:
            continue
        if include_file_ids and file_id not in include_file_ids:
            continue
        if exclude_file_ids and file_id in exclude_file_ids:
            continue
        if not _subject_matches(row_subject, file_id, subject):
            continue
        answer_file = answer_dir / f"{file_id}.xlsx"
        if not answer_file.exists():
            print(f"[subject-batch] skip missing answer file: {answer_file}")
            continue
        selected.append(row)
        seen.add(file_id)
        if max_files is not None and max_files > 0 and len(selected) >= max_files:
            break
    return selected


def load_subject_sample_groups(
    question_file: Path,
    answer_dir: Path,
    *,
    question_rows: list[dict[str, str]],
    samples_per_file: int | None,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], list[Any]]:
    groups: list[dict[str, Any]] = []
    all_samples: list[Any] = []
    for row in question_rows:
        file_id = scoring._first_present(row, ["file"], fallback_idx=1)
        subject = scoring._first_present(row, ["subject"], fallback_idx=0)
        question_type = scoring._first_present(row, ["question_type"], fallback_idx=2)
        _question_row, answer_file, samples = scoring.load_samples(
            question_file,
            answer_dir,
            file_id=file_id,
            answer_file=None,
            rows=None,
            start=0,
            limit=None,
            sample_size=samples_per_file,
            rng=rng,
        )
        groups.append(
            {
                "subject": subject,
                "file_id": file_id,
                "question_type": question_type,
                "answer_file": str(answer_file),
                "sample_n": len(samples),
                "excel_rows": [int(sample.row_index) for sample in samples],
                "samples": samples,
            }
        )
        all_samples.extend(samples)
        print(f"[subject-batch] {file_id} ({subject}): {len(samples)} samples")
    return groups, all_samples


def apply_total_sample_size(
    groups: list[dict[str, Any]],
    *,
    sample_size: int | None,
    rng: random.Random,
) -> list[dict[str, Any]]:
    if sample_size is None or sample_size <= 0:
        return groups
    all_pairs: list[tuple[int, Any]] = []
    for group_i, group in enumerate(groups):
        for sample in group["samples"]:
            all_pairs.append((group_i, sample))
    if sample_size >= len(all_pairs):
        return groups

    picked_pairs = rng.sample(all_pairs, sample_size)
    picked_by_group: dict[int, list[Any]] = defaultdict(list)
    for group_i, sample in picked_pairs:
        picked_by_group[group_i].append(sample)

    out: list[dict[str, Any]] = []
    for group_i, group in enumerate(groups):
        picked = sorted(picked_by_group.get(group_i, []), key=lambda item: int(item.row_index))
        if not picked:
            continue
        new_group = dict(group)
        new_group["samples"] = picked
        new_group["sample_n"] = len(picked)
        new_group["excel_rows"] = [int(sample.row_index) for sample in picked]
        out.append(new_group)
    return out


def _threshold_key(value: float) -> str:
    return str(value).replace(".", "_").replace("-", "neg_")


def _score_band_thresholds(full_score: float) -> tuple[str, list[float]]:
    if full_score < 10:
        return "lt10", [0.5, 1.0, 2.0, 3.0]
    if full_score < 20:
        return "10_20", [2.0, 4.0, 6.0]
    if full_score < 30:
        return "20_30", [3.0, 6.0, 9.0]
    return "ge30", [5.0, 10.0, 15.0]


def summarize_predictions_adaptive(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = scoring.summarize_predictions(rows)
    full_scores = [float(r.get("full_score") or 0.0) for r in rows if float(r.get("full_score") or 0.0) > 0.0]
    full_score = full_scores[0] if full_scores else 0.0
    score_band, thresholds = _score_band_thresholds(full_score)
    valid = [r for r in rows if r.get("pred_score") is not None]
    abs_errors = [abs(float(r["pred_score"]) - float(r["human_score"])) for r in valid]
    summary.update(
        {
            "full_score": full_score,
            "score_band": score_band,
            "adaptive_thresholds": ",".join(f"{x:g}" for x in thresholds),
        }
    )
    for threshold in thresholds:
        key = f"within_abs_{_threshold_key(threshold)}"
        summary[key] = sum(e <= threshold for e in abs_errors) / len(abs_errors) if abs_errors else None
    return summary


def per_file_summary(predictions: list[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        grouped[str(row.get("file", ""))].append(row)

    out: list[dict[str, Any]] = []
    for file_id in sorted(grouped):
        rows = grouped[file_id]
        out.append(
            {
                "mode": mode,
                "subject": rows[0].get("subject", ""),
                "file": file_id,
                "question_type": rows[0].get("question_type", ""),
                **summarize_predictions_adaptive(rows),
            }
        )
    return out


def build_per_file_comparison(summaries_by_mode: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    by_file: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for mode, rows in summaries_by_mode.items():
        for row in rows:
            by_file[str(row.get("file", ""))][mode] = row

    metric_keys = ("n", "valid_n", "parse_fail_n", "mae", "normalized_mae", "rmse", "bias", "pearson")
    out: list[dict[str, Any]] = []
    for file_id in sorted(by_file):
        mode_rows = by_file[file_id]
        first = next(iter(mode_rows.values()))
        row: dict[str, Any] = {
            "subject": first.get("subject", ""),
            "file": file_id,
            "question_type": first.get("question_type", ""),
            "full_score": first.get("full_score"),
            "score_band": first.get("score_band"),
            "adaptive_thresholds": first.get("adaptive_thresholds"),
        }
        for mode, summary in sorted(mode_rows.items()):
            for key in metric_keys:
                row[f"{mode}_{key}"] = summary.get(key)

        baseline = mode_rows.get("baseline")
        hawp = mode_rows.get("hawp_quant")
        if baseline is not None and hawp is not None:
            for key in ("mae", "normalized_mae", "rmse", "bias", "pearson"):
                b = baseline.get(key)
                h = hawp.get(key)
                row[f"hawp_minus_baseline_{key}"] = None if b is None or h is None else h - b
        out.append(row)
    return out


@torch.inference_mode()
def run_mode_subject(
    cfg,
    mode: str,
    groups: list[dict[str, Any]],
    *,
    device: str,
    batch_size: int,
    max_question_chars: int,
    max_ref_chars: int,
    max_answer_chars: int,
    profile_memory: bool,
    profile_memory_detail: bool,
    profile_memory_detail_samples: int,
    retry_missing: bool,
    retry_missing_attempts: int,
    retry_missing_batch_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any] | None]:
    print(f"\n[{mode}] loading model")
    recorder = scoring.MemoryRecorder(
        enabled=profile_memory,
        detail=profile_memory_detail,
        detail_samples=profile_memory_detail_samples,
    )
    if torch.cuda.is_available() and profile_memory:
        torch.cuda.reset_peak_memory_stats()
    recorder.record(f"{mode}.start")
    model, tokenizer, _ = load_baseline_model(cfg)
    recorder.record(f"{mode}.model_load.after")
    model, coordinator, kv_manager = setup_mode(model, cfg, device, mode)
    model.eval()
    reset_fn = make_reset_fn(model, coordinator, kv_manager)
    n_detail_hooks = scoring.install_memory_detail_probes(model, recorder) if profile_memory_detail else 0
    recorder.record(f"{mode}.setup.after", n_detail_hooks=n_detail_hooks)
    setup_allocated = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

    rows: list[dict[str, Any]] = []
    memory_batches: list[dict[str, Any]] = []
    retry_stats_total = {
        "retry_missing_initial_n": 0,
        "retry_attempted_n": 0,
        "retry_recovered_n": 0,
        "retry_failed_n": 0,
    }
    sample_i = 0
    global_batch_i = 0
    total_batches = sum(len(batch_eval._chunks(group["samples"], batch_size)) for group in groups)

    for group in groups:
        file_id = str(group["file_id"])
        samples = group["samples"]
        batches = batch_eval._chunks(samples, batch_size)
        for file_batch_i, batch in enumerate(batches, start=1):
            global_batch_i += 1
            reset_fn()
            prompt = batch_eval.format_batch_model_prompt(
                tokenizer,
                batch_eval.build_batch_prompt(
                    batch,
                    max_question_chars=max_question_chars,
                    max_ref_chars=max_ref_chars,
                    max_answer_chars=max_answer_chars,
                ),
            )
            prompt_len = tokenizer(prompt, return_tensors="pt").input_ids.shape[1]
            total_tokens_estimate = int(prompt_len + cfg.generation.max_new_tokens)
            baseline_kv_bytes = scoring.compute_baseline_kv_bytes(model, total_tokens_estimate)
            recorder.current_sample_i = global_batch_i
            if torch.cuda.is_available() and profile_memory:
                torch.cuda.reset_peak_memory_stats()
                recorder.reset_delta_baseline()
            recorder.record(
                f"{mode}.batch{global_batch_i}.generate.before",
                batch_i=global_batch_i,
                file=file_id,
                file_batch_i=file_batch_i,
                batch_size=len(batch),
                excel_rows=[int(sample.row_index) for sample in batch],
                prompt_tokens=int(prompt_len),
            )
            output = batch_eval.generate_by_mode_until_result(
                model,
                tokenizer,
                [prompt],
                cfg,
                mode,
                coordinator=coordinator,
                kv_manager=kv_manager,
            )[0]
            parsed = batch_eval.parse_batch_response(output, batch)
            initial_pred_by_row = {int(sample.row_index): parsed[int(sample.row_index)]["pred_score"] for sample in batch}
            parsed, retry_info_by_row, retry_stats = batch_eval.retry_missing_predictions(
                model,
                tokenizer,
                cfg,
                mode,
                parsed,
                batch,
                attempts=retry_missing_attempts if retry_missing else 0,
                retry_batch_size=retry_missing_batch_size,
                reset_fn=reset_fn,
                prompt_builder=lambda retry_batch: batch_eval.build_batch_prompt(
                    retry_batch,
                    max_question_chars=max_question_chars,
                    max_ref_chars=max_ref_chars,
                    max_answer_chars=max_answer_chars,
                ),
                prompt_formatter=batch_eval.format_batch_model_prompt,
                log_prefix=f"[{mode}] file={file_id} batch {file_batch_i}/{len(batches)}",
                coordinator=coordinator,
                kv_manager=kv_manager,
            )
            for key, value in retry_stats.items():
                retry_stats_total[key] += int(value)

            peak_gpu_bytes = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
            allocated_after = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            reserved_after = torch.cuda.memory_reserved() if torch.cuda.is_available() else 0
            cache_info = (
                scoring._cache_stats_dict(model, kv_manager, int(peak_gpu_bytes), int(baseline_kv_bytes))
                if profile_memory
                else {}
            )
            memory_batch = {
                "mode": mode,
                "batch_i": global_batch_i,
                "file_batch_i": file_batch_i,
                "file": file_id,
                "batch_size": len(batch),
                "excel_rows": [int(sample.row_index) for sample in batch],
                "prompt_tokens": int(prompt_len),
                "max_new_tokens": int(cfg.generation.max_new_tokens),
                "total_tokens_estimate": total_tokens_estimate,
                "setup_allocated_bytes": int(setup_allocated),
                "setup_allocated": scoring.format_nbytes(int(setup_allocated)),
                "peak_gpu_bytes": int(peak_gpu_bytes),
                "peak_gpu": scoring.format_nbytes(int(peak_gpu_bytes)),
                "peak_over_setup_bytes": int(peak_gpu_bytes - setup_allocated),
                "peak_over_setup": scoring.format_nbytes(max(0, int(peak_gpu_bytes - setup_allocated))),
                "allocated_after_bytes": int(allocated_after),
                "allocated_after": scoring.format_nbytes(int(allocated_after)),
                "reserved_after_bytes": int(reserved_after),
                "reserved_after": scoring.format_nbytes(int(reserved_after)),
                **cache_info,
            }
            memory_batches.append(memory_batch)
            recorder.record(f"{mode}.batch{global_batch_i}.generate.after", **memory_batch)
            recorder.current_sample_i = None

            for item_i, sample in enumerate(batch, start=1):
                sample_i += 1
                info = parsed[int(sample.row_index)]
                retry_info = retry_info_by_row[int(sample.row_index)]
                pred_score = info["pred_score"]
                row = {
                    "mode": mode,
                    "batch_i": global_batch_i,
                    "file_batch_i": file_batch_i,
                    "batch_size": len(batch),
                    "item_i": item_i,
                    "sample_i": sample_i,
                    "excel_row": sample.row_index,
                    "subject": sample.subject,
                    "file": sample.file,
                    "question_type": sample.question_type,
                    "full_score": sample.full_score,
                    "human_score": sample.human_score,
                    "initial_pred_score": initial_pred_by_row[int(sample.row_index)],
                    "pred_score": pred_score,
                    "abs_error": None if pred_score is None else abs(pred_score - sample.human_score),
                    "parsed_json": info["parsed_json"],
                    "reason": info["reason"],
                    "raw_output": output,
                    "retry_attempts": retry_info["retry_attempts"],
                    "retry_success": retry_info["retry_success"],
                    "retry_raw_output": retry_info["retry_raw_output"],
                    "student_answer": sample.student_answer,
                }
                if profile_memory:
                    row.update(memory_batch)
                rows.append(row)

            valid_n = sum(1 for item in batch if parsed[int(item.row_index)]["pred_score"] is not None)
            print(
                f"[{mode}] batch {global_batch_i}/{total_batches} file={file_id} "
                f"file_batch={file_batch_i}/{len(batches)} rows={[s.row_index for s in batch]} "
                f"parsed={valid_n}/{len(batch)}"
            )

    summary = scoring.summarize_predictions(rows)
    summary.update(
        {
            "answers_per_prompt": batch_size,
            "n_batches": total_batches,
            "n_files": len(groups),
            **retry_stats_total,
        }
    )

    memory_profile = None
    if profile_memory:
        peak_values = [int(item["peak_gpu_bytes"]) for item in memory_batches]
        over_setup_values = [int(item["peak_over_setup_bytes"]) for item in memory_batches]
        cache_values = [int(item.get("cache_runtime_bytes") or 0) for item in memory_batches]
        baseline_kv_values = [int(item.get("baseline_kv_bytes") or 0) for item in memory_batches]
        memory_summary = {
            "mode": mode,
            "n_batches": len(memory_batches),
            "answers_per_prompt": batch_size,
            "n_files": len(groups),
            "n_samples": len(rows),
            "setup_allocated_bytes": int(setup_allocated),
            "setup_allocated": scoring.format_nbytes(int(setup_allocated)),
            "peak_gpu_max_bytes": max(peak_values) if peak_values else 0,
            "peak_gpu_max": scoring.format_nbytes(max(peak_values) if peak_values else 0),
            "peak_over_setup_max_bytes": max(over_setup_values) if over_setup_values else 0,
            "peak_over_setup_max": scoring.format_nbytes(max(over_setup_values) if over_setup_values else 0),
            "cache_runtime_max_bytes": max(cache_values) if cache_values else 0,
            "cache_runtime_max": scoring.format_nbytes(max(cache_values) if cache_values else 0),
            "baseline_kv_max_bytes": max(baseline_kv_values) if baseline_kv_values else 0,
            "baseline_kv_max": scoring.format_nbytes(max(baseline_kv_values) if baseline_kv_values else 0),
            "n_detail_hooks": n_detail_hooks,
            "n_memory_records": len(recorder.records),
        }
        summary.update(memory_summary)
        memory_profile = {
            "summary": memory_summary,
            "batches": memory_batches,
            "samples": memory_batches,
            "records": recorder.records,
        }

    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows, summary, memory_profile


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate batched scoring for all files in one subject.")
    parser.add_argument("config")
    parser.add_argument("--subject", required=True, help="Subject to evaluate. Use all or * for all subjects.")
    parser.add_argument("--question-file", default="pingfen/questions.xlsx")
    parser.add_argument("--answer-dir", default="pingfen/answers")
    parser.add_argument("--modes", nargs="+", choices=_SUPPORTED_MODES, default=["baseline", "hawp_quant"])
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--include-file-ids", nargs="*", default=None, help="Optional file-id allowlist, comma or space separated.")
    parser.add_argument("--exclude-file-ids", nargs="*", default=None, help="Optional file-id blocklist, comma or space separated.")
    parser.add_argument("--max-files", type=int, default=None, help="Optional cap on number of question files.")
    parser.add_argument("--samples-per-file", type=int, default=100, help="Random samples per question file. Use 0 for all valid rows.")
    parser.add_argument("--sample-size", type=int, default=None, help="Optional total random sample cap after per-file sampling. Use 0 or omit for no total cap.")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--batch-size", "--answers-per-prompt", dest="batch_size", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-question-chars", type=int, default=10000)
    parser.add_argument("--max-ref-chars", type=int, default=6000)
    parser.add_argument("--max-answer-chars", type=int, default=4000)
    parser.add_argument("--profile-memory", action="store_true")
    parser.add_argument("--profile-memory-detail", action="store_true")
    parser.add_argument("--profile-memory-detail-samples", type=int, default=1)
    parser.add_argument("--no-retry-missing", dest="retry_missing", action="store_false", help="Disable retry for rows missing from the first batch response.")
    parser.set_defaults(retry_missing=True)
    parser.add_argument("--retry-missing-attempts", type=int, default=1)
    parser.add_argument("--retry-missing-batch-size", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.samples_per_file < 0:
        raise ValueError("--samples-per-file must be non-negative")
    if args.sample_size is not None and args.sample_size < 0:
        raise ValueError("--sample-size must be non-negative")
    if args.retry_missing_attempts < 0:
        raise ValueError("--retry-missing-attempts must be non-negative")
    if args.retry_missing_batch_size <= 0:
        raise ValueError("--retry-missing-batch-size must be positive")

    cfg = load_config(args.config)
    cfg.generation.max_new_tokens = int(args.max_new_tokens)
    device = _resolve_device(cfg.train.device)
    question_file = Path(args.question_file)
    answer_dir = Path(args.answer_dir)
    output_dir = Path(args.output_dir or f"artifacts/scoring_eval/{args.subject}_subject_batch")
    rng = random.Random(args.seed)
    samples_per_file = None if args.samples_per_file == 0 else args.samples_per_file

    question_rows = discover_question_rows(
        question_file,
        answer_dir,
        subject=args.subject,
        include_file_ids=_split_items(args.include_file_ids),
        exclude_file_ids=_split_items(args.exclude_file_ids),
        max_files=args.max_files,
    )
    if not question_rows:
        raise RuntimeError(f"No matching question files found for subject={args.subject!r}")

    groups, _all_samples = load_subject_sample_groups(
        question_file,
        answer_dir,
        question_rows=question_rows,
        samples_per_file=samples_per_file,
        rng=rng,
    )
    groups = apply_total_sample_size(groups, sample_size=args.sample_size, rng=random.Random(args.seed + 17))
    groups = [group for group in groups if group["samples"]]
    if not groups:
        raise RuntimeError("No valid scoring samples loaded.")
    total_samples = sum(len(group["samples"]) for group in groups)

    output_dir.mkdir(parents=True, exist_ok=True)
    selection_rows = [
        {key: value for key, value in group.items() if key != "samples"}
        for group in groups
    ]
    scoring._write_json(
        output_dir / "sample_selection.json",
        {
            "subject": args.subject,
            "question_file": str(question_file),
            "answer_dir": str(answer_dir),
            "samples_per_file": args.samples_per_file,
            "sample_size": args.sample_size,
            "seed": args.seed,
            "modes": args.modes,
            "batch_size": args.batch_size,
            "retry_missing": args.retry_missing,
            "retry_missing_attempts": args.retry_missing_attempts,
            "retry_missing_batch_size": args.retry_missing_batch_size,
            "n_files": len(groups),
            "n_samples": total_samples,
            "files": selection_rows,
        },
    )
    scoring._write_csv(output_dir / "sample_selection_files.csv", selection_rows)
    preview_prompt = batch_eval.build_batch_prompt(
        groups[0]["samples"][: args.batch_size],
        max_question_chars=args.max_question_chars,
        max_ref_chars=args.max_ref_chars,
        max_answer_chars=args.max_answer_chars,
    )
    (output_dir / "prompt_preview.txt").write_text(preview_prompt, encoding="utf-8")

    print("=" * 80)
    print(f"[subject-batch] config={args.config}")
    print(f"[subject-batch] subject={args.subject}")
    print(f"[subject-batch] files={len(groups)} samples={total_samples} samples_per_file={args.samples_per_file}")
    print(f"[subject-batch] batch_size={args.batch_size} max_new_tokens={args.max_new_tokens}")
    print(f"[subject-batch] modes={args.modes}")
    print(f"[subject-batch] output_dir={output_dir}")
    print("=" * 80)
    if args.dry_run:
        print("[subject-batch] dry run only")
        return

    summary_rows: list[dict[str, Any]] = []
    all_predictions: list[dict[str, Any]] = []
    all_file_summary_rows: list[dict[str, Any]] = []
    summaries_by_mode: dict[str, list[dict[str, Any]]] = {}

    for mode in args.modes:
        predictions, summary, memory_profile = run_mode_subject(
            cfg,
            mode,
            groups,
            device=device,
            batch_size=args.batch_size,
            max_question_chars=args.max_question_chars,
            max_ref_chars=args.max_ref_chars,
            max_answer_chars=args.max_answer_chars,
            profile_memory=args.profile_memory,
            profile_memory_detail=args.profile_memory_detail,
            profile_memory_detail_samples=args.profile_memory_detail_samples,
            retry_missing=args.retry_missing,
            retry_missing_attempts=args.retry_missing_attempts,
            retry_missing_batch_size=args.retry_missing_batch_size,
        )
        mode_dir = output_dir / mode
        scoring._write_jsonl(mode_dir / "predictions.jsonl", predictions)
        scoring._write_csv(mode_dir / "predictions.csv", predictions)
        scoring._write_json(mode_dir / "summary.json", summary)
        file_summary_rows = per_file_summary(predictions, mode)
        summaries_by_mode[mode] = file_summary_rows
        scoring._write_csv(mode_dir / "per_file_summary.csv", file_summary_rows)
        scoring._write_json(mode_dir / "per_file_summary.json", file_summary_rows)
        if memory_profile is not None:
            scoring._write_json(mode_dir / "memory_profile.json", memory_profile)
            scoring._write_csv(mode_dir / "memory_batches.csv", memory_profile["batches"])
            scoring._write_csv(mode_dir / "memory_samples.csv", memory_profile["samples"])
        summary_rows.append({"mode": mode, **summary})
        all_predictions.extend(predictions)
        all_file_summary_rows.extend(file_summary_rows)

    per_file_comparison_rows = build_per_file_comparison(summaries_by_mode)
    scoring._write_csv(output_dir / "summary.csv", summary_rows)
    scoring._write_json(output_dir / "summary.json", summary_rows)
    scoring._write_csv(output_dir / "per_file_summary.csv", all_file_summary_rows)
    scoring._write_json(output_dir / "per_file_summary.json", all_file_summary_rows)
    scoring._write_csv(output_dir / "per_file_comparison.csv", per_file_comparison_rows)
    scoring._write_json(output_dir / "per_file_comparison.json", per_file_comparison_rows)
    scoring._write_jsonl(output_dir / "predictions_all.jsonl", all_predictions)

    print("\n[subject-batch] overall summary")
    for row in summary_rows:
        mem_part = ""
        if args.profile_memory:
            mem_part = (
                f" peak={row.get('peak_gpu_max')} "
                f"peak-extra={row.get('peak_over_setup_max')} "
                f"cache={row.get('cache_runtime_max')}"
            )
        print(
            f"{row['mode']:>10} n={row['valid_n']}/{row['n']} "
            f"files={row['n_files']} batches={row['n_batches']} "
            f"MAE={row['mae']} normMAE={row['normalized_mae']} "
            f"within1={row['within_1']} within2={row['within_2']} "
            f"within3={row['within_3']} parse_fail={row['parse_fail_n']} pearson={row['pearson']}"
            f"{mem_part}"
        )
    print(f"[subject-batch] wrote {output_dir}")


if __name__ == "__main__":
    main()
