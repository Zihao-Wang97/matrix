#!/usr/bin/env python
"""Evaluate scoring by subject across all matching question files."""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from hawp_laq.config import load_config
from hawp_laq.runtime.generate import _resolve_device
from hawp_laq.utils.memory import format_nbytes


def _load_scoring_module():
    script_path = Path(__file__).resolve().parent / "10_eval_scoring_task.py"
    spec = importlib.util.spec_from_file_location("scoring_task_eval", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load scoring module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


scoring = _load_scoring_module()


def _split_items(raw: list[str] | None) -> set[str]:
    items: set[str] = set()
    for value in raw or []:
        items.update(part.strip() for part in str(value).split(",") if part.strip())
    return items


def _subject_matches(row_subject: str, file_id: str, subject: str) -> bool:
    if not subject or subject in {"all", "ALL", "*", "全部", "全科目"}:
        return True
    return row_subject == subject or subject in row_subject or file_id.startswith(subject)


def discover_question_files(
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
        row_subject = scoring._first_present(row, ["科目", "subject"], fallback_idx=0)
        file_id = scoring._first_present(row, ["文件", "file"], fallback_idx=1)
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
            print(f"[subject-eval] skip missing answer file: {answer_file}")
            continue
        selected.append(row)
        seen.add(file_id)
        if max_files is not None and max_files > 0 and len(selected) >= max_files:
            break
    return selected


def load_subject_samples(
    question_file: Path,
    answer_dir: Path,
    *,
    question_rows: list[dict[str, str]],
    samples_per_file: int | None,
    rng: random.Random,
) -> tuple[list[Any], list[dict[str, Any]]]:
    all_samples: list[Any] = []
    file_rows: list[dict[str, Any]] = []
    for row in question_rows:
        file_id = scoring._first_present(row, ["文件", "file"], fallback_idx=1)
        subject = scoring._first_present(row, ["科目", "subject"], fallback_idx=0)
        question_type = scoring._first_present(row, ["题目类型", "question_type"], fallback_idx=2)
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
        all_samples.extend(samples)
        file_rows.append(
            {
                "subject": subject,
                "file_id": file_id,
                "question_type": question_type,
                "answer_file": str(answer_file),
                "sample_n": len(samples),
                "excel_rows": [sample.row_index for sample in samples],
            }
        )
        print(f"[subject-eval] {file_id} ({subject}): {len(samples)} samples")
    return all_samples, file_rows


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
    out = []
    for file_id in sorted(grouped):
        rows = grouped[file_id]
        subject = rows[0].get("subject", "")
        question_type = rows[0].get("question_type", "")
        out.append(
            {
                "mode": mode,
                "subject": subject,
                "file": file_id,
                "question_type": question_type,
                **summarize_predictions_adaptive(rows),
            }
        )
    return out


def build_per_file_comparison(summaries_by_mode: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    by_file: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for mode, rows in summaries_by_mode.items():
        for row in rows:
            by_file[str(row.get("file", ""))][mode] = row

    metric_keys = [
        "n",
        "valid_n",
        "parse_fail_n",
        "mae",
        "normalized_mae",
        "rmse",
        "bias",
        "pearson",
    ]
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
            thresholds = str(summary.get("adaptive_thresholds") or "").split(",")
            for raw_threshold in thresholds:
                if not raw_threshold:
                    continue
                threshold = float(raw_threshold)
                metric = f"within_abs_{_threshold_key(threshold)}"
                row[f"{mode}_{metric}"] = summary.get(metric)

        baseline = mode_rows.get("baseline")
        hawp = mode_rows.get("hawp_quant")
        if baseline is not None and hawp is not None:
            for key in ("mae", "normalized_mae", "rmse", "bias", "pearson"):
                b = baseline.get(key)
                h = hawp.get(key)
                row[f"hawp_minus_baseline_{key}"] = None if b is None or h is None else h - b
            thresholds = str(first.get("adaptive_thresholds") or "").split(",")
            for raw_threshold in thresholds:
                if not raw_threshold:
                    continue
                threshold = float(raw_threshold)
                metric = f"within_abs_{_threshold_key(threshold)}"
                b = baseline.get(metric)
                h = hawp.get(metric)
                row[f"hawp_minus_baseline_{metric}"] = None if b is None or h is None else h - b
        out.append(row)
    return out


def run_mode_compat(
    cfg,
    mode: str,
    samples: list[Any],
    *,
    device: str,
    max_question_chars: int,
    max_ref_chars: int,
    max_answer_chars: int,
    profile_memory: bool,
    profile_memory_detail: bool,
    profile_memory_detail_samples: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any] | None]:
    signature = inspect.signature(scoring.run_mode)
    supports_memory_profile = "profile_memory" in signature.parameters
    if supports_memory_profile:
        return scoring.run_mode(
            cfg,
            mode,
            samples,
            device=device,
            max_question_chars=max_question_chars,
            max_ref_chars=max_ref_chars,
            max_answer_chars=max_answer_chars,
            profile_memory=profile_memory,
            profile_memory_detail=profile_memory_detail,
            profile_memory_detail_samples=profile_memory_detail_samples,
        )

    if profile_memory and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        setup_before = torch.cuda.memory_allocated()
    else:
        setup_before = 0

    predictions, summary = scoring.run_mode(
        cfg,
        mode,
        samples,
        device=device,
        max_question_chars=max_question_chars,
        max_ref_chars=max_ref_chars,
        max_answer_chars=max_answer_chars,
    )

    memory_profile = None
    if profile_memory:
        peak = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        allocated_after = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        reserved_after = torch.cuda.memory_reserved() if torch.cuda.is_available() else 0
        memory_summary = {
            "mode": mode,
            "n_samples": len(samples),
            "profile_granularity": "mode_fallback",
            "note": "Loaded 10_eval_scoring_task.py does not support per-sample profile_memory; recorded mode-level peak only.",
            "setup_allocated_bytes": int(setup_before),
            "setup_allocated": format_nbytes(int(setup_before)),
            "peak_gpu_max_bytes": int(peak),
            "peak_gpu_max": format_nbytes(int(peak)),
            "peak_over_setup_max_bytes": int(peak - setup_before),
            "peak_over_setup_max": format_nbytes(max(0, int(peak - setup_before))),
            "allocated_after_bytes": int(allocated_after),
            "allocated_after": format_nbytes(int(allocated_after)),
            "reserved_after_bytes": int(reserved_after),
            "reserved_after": format_nbytes(int(reserved_after)),
        }
        summary.update(memory_summary)
        memory_profile = {
            "summary": memory_summary,
            "samples": [],
            "records": [],
        }
        print(
            "[subject-eval] warning: 10_eval_scoring_task.py does not support detailed "
            "profile_memory; wrote mode-level peak only."
        )

    return predictions, summary, memory_profile


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate baseline vs hawp_quant for all files in one subject.")
    parser.add_argument("config")
    parser.add_argument("--subject", required=True, help="Subject to evaluate, e.g. 历史. Use all/全部 for all subjects.")
    parser.add_argument("--question-file", default="pingfen/题目信息.xlsx")
    parser.add_argument("--answer-dir", default="pingfen/评分数据")
    parser.add_argument("--samples-per-file", type=int, default=1000, help="Random samples per question file. Use 0 for all valid rows.")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--modes", nargs="+", choices=("baseline", "hawp_quant", "quant_only"), default=["baseline", "hawp_quant"])
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--include-file-ids", nargs="*", default=None, help="Optional file-id allowlist, comma or space separated.")
    parser.add_argument("--exclude-file-ids", nargs="*", default=None, help="Optional file-id blocklist, comma or space separated.")
    parser.add_argument("--max-files", type=int, default=None, help="Optional cap on number of question files.")
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--max-question-chars", type=int, default=6000)
    parser.add_argument("--max-ref-chars", type=int, default=4000)
    parser.add_argument("--max-answer-chars", type=int, default=4000)
    parser.add_argument("--profile-memory", action="store_true")
    parser.add_argument("--profile-memory-detail", action="store_true")
    parser.add_argument("--profile-memory-detail-samples", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg.generation.max_new_tokens = int(args.max_new_tokens)
    device = _resolve_device(cfg.train.device)
    question_file = Path(args.question_file)
    answer_dir = Path(args.answer_dir)
    output_dir = Path(args.output_dir or f"artifacts/scoring_eval/{args.subject}_subject_eval")
    rng = random.Random(args.seed)
    samples_per_file = None if args.samples_per_file <= 0 else args.samples_per_file

    question_rows = discover_question_files(
        question_file,
        answer_dir,
        subject=args.subject,
        include_file_ids=_split_items(args.include_file_ids),
        exclude_file_ids=_split_items(args.exclude_file_ids),
        max_files=args.max_files,
    )
    if not question_rows:
        raise RuntimeError(f"No matching question files found for subject={args.subject!r}")

    samples, file_rows = load_subject_samples(
        question_file,
        answer_dir,
        question_rows=question_rows,
        samples_per_file=samples_per_file,
        rng=rng,
    )
    if not samples:
        raise RuntimeError("No valid scoring samples loaded.")

    output_dir.mkdir(parents=True, exist_ok=True)
    scoring._write_json(
        output_dir / "sample_selection.json",
        {
            "subject": args.subject,
            "question_file": str(question_file),
            "answer_dir": str(answer_dir),
            "samples_per_file": args.samples_per_file,
            "seed": args.seed,
            "modes": args.modes,
            "n_files": len(file_rows),
            "n_samples": len(samples),
            "files": file_rows,
        },
    )
    scoring._write_csv(output_dir / "sample_selection_files.csv", file_rows)
    preview_prompt = scoring.build_prompt(
        samples[0],
        max_question_chars=args.max_question_chars,
        max_ref_chars=args.max_ref_chars,
        max_answer_chars=args.max_answer_chars,
    )
    (output_dir / "prompt_preview.txt").write_text(preview_prompt, encoding="utf-8")

    print("=" * 80)
    print(f"[subject-eval] config={args.config}")
    print(f"[subject-eval] subject={args.subject}")
    print(f"[subject-eval] files={len(file_rows)} samples={len(samples)} samples_per_file={args.samples_per_file}")
    print(f"[subject-eval] modes={args.modes}")
    print(f"[subject-eval] output_dir={output_dir}")
    print("=" * 80)
    if args.dry_run:
        print("[subject-eval] dry run only")
        return

    summary_rows: list[dict[str, Any]] = []
    all_predictions: list[dict[str, Any]] = []
    all_file_summary_rows: list[dict[str, Any]] = []
    summaries_by_mode: dict[str, list[dict[str, Any]]] = {}
    for mode in args.modes:
        predictions, summary, memory_profile = run_mode_compat(
            cfg,
            mode,
            samples,
            device=device,
            max_question_chars=args.max_question_chars,
            max_ref_chars=args.max_ref_chars,
            max_answer_chars=args.max_answer_chars,
            profile_memory=args.profile_memory,
            profile_memory_detail=args.profile_memory_detail,
            profile_memory_detail_samples=args.profile_memory_detail_samples,
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
            scoring._write_csv(mode_dir / "memory_samples.csv", memory_profile["samples"])
        summary_rows.append({"mode": mode, **summary})
        all_predictions.extend(predictions)
        all_file_summary_rows.extend(file_summary_rows)

    per_file_comparison_rows = build_per_file_comparison(summaries_by_mode)
    scoring._write_csv(output_dir / "summary.csv", summary_rows)
    scoring._write_json(output_dir / "summary.json", summary_rows)
    scoring._write_csv(output_dir / "per_file_summary_adaptive.csv", all_file_summary_rows)
    scoring._write_json(output_dir / "per_file_summary_adaptive.json", all_file_summary_rows)
    scoring._write_csv(output_dir / "per_file_comparison.csv", per_file_comparison_rows)
    scoring._write_json(output_dir / "per_file_comparison.json", per_file_comparison_rows)
    scoring._write_jsonl(output_dir / "predictions_all.jsonl", all_predictions)

    print("\n[subject-eval] overall summary")
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
            f"MAE={row['mae']} normMAE={row['normalized_mae']} "
            f"within1={row['within_1']} within2={row['within_2']} "
            f"within3={row['within_3']} parse_fail={row['parse_fail_n']} pearson={row['pearson']}"
            f"{mem_part}"
        )
    print("\n[subject-eval] per-file baseline vs hawp_quant")
    for row in per_file_comparison_rows:
        base_mae = row.get("baseline_mae")
        hawp_mae = row.get("hawp_quant_mae")
        delta_mae = row.get("hawp_minus_baseline_mae")
        base_norm = row.get("baseline_normalized_mae")
        hawp_norm = row.get("hawp_quant_normalized_mae")
        print(
            f"{row.get('file', '')}: full={row.get('full_score')} band={row.get('score_band')} "
            f"base_MAE={base_mae} hawp_MAE={hawp_mae} dMAE={delta_mae} "
            f"base_norm={base_norm} hawp_norm={hawp_norm}"
        )
    print(f"[subject-eval] wrote {output_dir}")


if __name__ == "__main__":
    main()
