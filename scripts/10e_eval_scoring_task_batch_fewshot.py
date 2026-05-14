#!/usr/bin/env python
"""Evaluate batched answer scoring with score-band few-shot examples.

This script is a few-shot companion to ``10d_eval_scoring_task_batch.py``.
For one question/file_id, it samples human-scored examples from different score
bands, inserts them into each prompt, and then scores N target answers per
prompt.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import random
import re
import sys
from pathlib import Path
from typing import Any

import torch

from hawp_laq.config import load_config
from hawp_laq.runtime.generate import _resolve_device, load_baseline_model
from hawp_laq.runtime.mode_runner import generate_by_mode, make_reset_fn, setup_mode


def _load_module(script_name: str, module_name: str):
    script_path = Path(__file__).resolve().parent / script_name
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


scoring = _load_module("10_eval_scoring_task.py", "scoring_task_eval")
batch_eval = _load_module("10d_eval_scoring_task_batch.py", "scoring_task_batch_eval")

_SUPPORTED_MODES = ("baseline", "quant_only", "hawp_quant")
_SCORE_BANDS = (
    ("b0_0_20", 0.0, 0.20),
    ("b1_20_40", 0.20, 0.40),
    ("b2_40_60", 0.40, 0.60),
    ("b3_60_80", 0.60, 0.80),
)


def _in_band(score: float, full_score: float, lo: float, hi: float) -> bool:
    if full_score <= 0:
        return False
    ratio = score / full_score
    if abs(lo) < 1e-12:
        return ratio >= lo and ratio <= hi
    return ratio > lo and ratio <= hi


def _informative_char_ratio(text: str) -> float:
    chars = [ch for ch in text if not ch.isspace()]
    if not chars:
        return 0.0
    informative = 0
    for ch in chars:
        if ch.isalnum() or "\u4e00" <= ch <= "\u9fff":
            informative += 1
    return informative / len(chars)


def _looks_like_usable_example_answer(
    text: str,
    *,
    min_chars: int,
    max_chars: int,
    min_informative_ratio: float,
) -> bool:
    text = str(text or "").strip()
    if len(text) < min_chars or len(text) > max_chars:
        return False
    if text.count("\ufffd") / max(1, len(text)) > 0.02:
        return False
    if _informative_char_ratio(text) < min_informative_ratio:
        return False
    compact = re.sub(r"\s+", "", text)
    if compact:
        most_common_ratio = max(compact.count(ch) for ch in set(compact)) / len(compact)
        if most_common_ratio > 0.45:
            return False
    return True


def select_score_band_examples(
    pool: list[Any],
    *,
    exclude_rows: set[int],
    rng: random.Random,
    per_band: int,
    allow_from_eval: bool,
    min_example_chars: int,
    max_example_candidate_chars: int,
    min_informative_ratio: float,
) -> tuple[list[Any], list[dict[str, Any]]]:
    candidates = pool if allow_from_eval else [s for s in pool if int(s.row_index) not in exclude_rows]
    selected: list[Any] = []
    report: list[dict[str, Any]] = []
    used_rows: set[int] = set()

    for band, lo, hi in _SCORE_BANDS:
        raw_band_items = [
            s for s in candidates
            if int(s.row_index) not in used_rows
            and _in_band(float(s.human_score), float(s.full_score), lo, hi)
        ]
        band_items = [
            s for s in raw_band_items
            if _looks_like_usable_example_answer(
                str(s.student_answer),
                min_chars=min_example_chars,
                max_chars=max_example_candidate_chars,
                min_informative_ratio=min_informative_ratio,
            )
        ]
        rng.shuffle(band_items)
        if band_items:
            lengths = sorted(len(str(s.student_answer).strip()) for s in band_items)
            median_len = lengths[len(lengths) // 2]
            band_items.sort(key=lambda s: (abs(len(str(s.student_answer).strip()) - median_len), int(s.row_index)))
        else:
            median_len = 0
        picked = band_items[: max(0, per_band)]
        selected.extend(picked)
        used_rows.update(int(s.row_index) for s in picked)
        report.append(
            {
                "band": band,
                "range": f"[{lo:g}, {hi:g}] * full_score" if abs(lo) < 1e-12 else f"({lo:g}, {hi:g}] * full_score",
                "raw_candidate_n": len(raw_band_items),
                "candidate_n": len(band_items),
                "filtered_out_n": len(raw_band_items) - len(band_items),
                "median_answer_chars": median_len,
                "picked_n": len(picked),
                "excel_rows": [int(s.row_index) for s in picked],
                "scores": [float(s.human_score) for s in picked],
                "score_ratios": [
                    None if float(s.full_score) <= 0 else float(s.human_score) / float(s.full_score)
                    for s in picked
                ],
                "answer_chars": [len(str(s.student_answer).strip()) for s in picked],
            }
        )

    return selected, report


def build_fewshot_batch_prompt(
    target_samples: list[Any],
    fewshot_examples: list[Any],
    *,
    max_question_chars: int,
    max_ref_chars: int,
    max_answer_chars: int,
    max_example_answer_chars: int,
) -> str:
    if not target_samples:
        raise ValueError("Cannot build a prompt with no target samples.")

    first = target_samples[0]
    role = f"{first.subject}{first.question_type} grading expert" if first.question_type else f"{first.subject} grading expert"

    example_blocks = []
    for i, sample in enumerate(fewshot_examples, start=1):
        detail = str(getattr(sample, "detail_score", "") or "").strip()
        detail_line = f"\nHuman detail: {detail}" if detail else ""
        example_blocks.append(
            f"Example {i}:\n"
            f"Student answer:\n{scoring._truncate(sample.student_answer, max_example_answer_chars)}\n"
            f"Human score: {sample.human_score:g} / {sample.full_score:g}"
            f"{detail_line}"
        )

    target_blocks = []
    for sample in target_samples:
        target_blocks.append(
            f"### excel_row: {sample.row_index}\n"
            f"{scoring._truncate(sample.student_answer, max_answer_chars)}"
        )

    examples_text = "\n\n".join(example_blocks) if example_blocks else "No examples are available."
    return f"""You are a strict {role}. Score each target answer independently for the same question.

Rules:
1. The score must be between 0 and {first.full_score:g}.
2. The question and reference answer are the primary grading standard.
3. If the reference answer contains point-by-point scoring, per-point values, full-score instructions, or discretionary scoring notes, follow them first.
4. Use the human-scored examples only to calibrate strictness for low and middle score ranges.
5. Do not directly imitate example scores; grade each target answer according to the question and reference answer.
6. Do not compare target students with each other.
7. Return only a valid JSON array. Do not output Markdown.
8. Put the JSON array between <RESULT_JSON> and </RESULT_JSON>.
9. The array must contain exactly {len(target_samples)} objects, one for each target excel_row.
10. Each object must have: "excel_row", "score", and "reason". Keep each reason concise.

JSON format:
<RESULT_JSON>
[
  {{"excel_row": "<excel_row>", "score": <number>, "reason": "<short reason>"}}
]
</RESULT_JSON>

Subject: {first.subject}
File: {first.file}
Question type: {first.question_type}
Full score: {first.full_score:g}

Question:
{scoring._truncate(first.question, max_question_chars)}

Reference answer:
{scoring._truncate(first.reference_answer, max_ref_chars)}

Scoring basis:
- The reference answer above is the main scoring standard.
- If it says how many points each item is worth, how many valid points are needed for full score, or how to award discretionary credit, apply that rule first.
- The examples below are only score-scale anchors from this same question and are not additional grading rules.
- Avoid giving a high score only because an answer resembles an example; score by matched valid points and historical correctness.

Human-scored examples for calibration:

{examples_text}

Target student answers:

{chr(10).join(target_blocks)}
"""


def format_model_prompt(tokenizer, prompt: str) -> str:
    messages = [
        {
            "role": "system",
            "content": "You are a grading assistant. Return only valid JSON that follows the requested schema.",
        },
        {"role": "user", "content": prompt},
    ]
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return messages[0]["content"] + "\n\n" + messages[1]["content"]


@torch.inference_mode()
def run_mode(
    cfg,
    mode: str,
    samples: list[Any],
    fewshot_examples: list[Any],
    *,
    device: str,
    batch_size: int,
    max_question_chars: int,
    max_ref_chars: int,
    max_answer_chars: int,
    max_example_answer_chars: int,
    profile_memory: bool,
    profile_memory_detail: bool,
    profile_memory_detail_samples: int,
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
    sample_i_by_row = {int(sample.row_index): i for i, sample in enumerate(samples, start=1)}
    batches = batch_eval._chunks(samples, batch_size)

    for batch_i, target_batch in enumerate(batches, start=1):
        reset_fn()
        prompt = format_model_prompt(
            tokenizer,
            build_fewshot_batch_prompt(
                target_batch,
                fewshot_examples,
                max_question_chars=max_question_chars,
                max_ref_chars=max_ref_chars,
                max_answer_chars=max_answer_chars,
                max_example_answer_chars=max_example_answer_chars,
            ),
        )
        prompt_len = tokenizer(prompt, return_tensors="pt").input_ids.shape[1]
        total_tokens_estimate = int(prompt_len + cfg.generation.max_new_tokens)
        baseline_kv_bytes = scoring.compute_baseline_kv_bytes(model, total_tokens_estimate)
        recorder.current_sample_i = batch_i
        if torch.cuda.is_available() and profile_memory:
            torch.cuda.reset_peak_memory_stats()
            recorder.reset_delta_baseline()
        recorder.record(
            f"{mode}.batch{batch_i}.generate.before",
            batch_i=batch_i,
            batch_size=len(target_batch),
            fewshot_n=len(fewshot_examples),
            excel_rows=[int(sample.row_index) for sample in target_batch],
            fewshot_rows=[int(sample.row_index) for sample in fewshot_examples],
            prompt_tokens=int(prompt_len),
        )
        output = generate_by_mode(model, tokenizer, [prompt], cfg, mode, coordinator=coordinator, kv_manager=kv_manager)[0]
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
            "batch_i": batch_i,
            "batch_size": len(target_batch),
            "fewshot_n": len(fewshot_examples),
            "excel_rows": [int(sample.row_index) for sample in target_batch],
            "fewshot_rows": [int(sample.row_index) for sample in fewshot_examples],
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
        recorder.record(f"{mode}.batch{batch_i}.generate.after", **memory_batch)
        recorder.current_sample_i = None

        parsed = batch_eval.parse_batch_response(output, target_batch)
        for item_i, sample in enumerate(target_batch, start=1):
            info = parsed[int(sample.row_index)]
            pred_score = info["pred_score"]
            row = {
                "mode": mode,
                "batch_i": batch_i,
                "batch_size": len(target_batch),
                "fewshot_n": len(fewshot_examples),
                "fewshot_rows": [int(item.row_index) for item in fewshot_examples],
                "item_i": item_i,
                "sample_i": sample_i_by_row[int(sample.row_index)],
                "excel_row": sample.row_index,
                "subject": sample.subject,
                "file": sample.file,
                "question_type": sample.question_type,
                "full_score": sample.full_score,
                "human_score": sample.human_score,
                "pred_score": pred_score,
                "abs_error": None if pred_score is None else abs(pred_score - sample.human_score),
                "parsed_json": info["parsed_json"],
                "reason": info["reason"],
                "raw_output": output,
                "student_answer": sample.student_answer,
            }
            if profile_memory:
                row.update(memory_batch)
            rows.append(row)

        valid_n = sum(1 for item in target_batch if parsed[int(item.row_index)]["pred_score"] is not None)
        print(
            f"[{mode}] batch {batch_i}/{len(batches)} "
            f"rows={[s.row_index for s in target_batch]} fewshot={len(fewshot_examples)} parsed={valid_n}/{len(target_batch)}"
        )

    summary = scoring.summarize_predictions(rows)
    summary.update(
        {
            "answers_per_prompt": batch_size,
            "n_batches": len(batches),
            "fewshot_n": len(fewshot_examples),
            "fewshot_rows": ",".join(str(int(s.row_index)) for s in fewshot_examples),
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
            "fewshot_n": len(fewshot_examples),
            "n_samples": len(samples),
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


def _fewshot_rows(examples: list[Any], report: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_row = {int(s.row_index): s for s in examples}
    out: list[dict[str, Any]] = []
    for band_report in report:
        band = band_report["band"]
        for row in band_report["excel_rows"]:
            sample = by_row[int(row)]
            out.append(
                {
                    "band": band,
                    "excel_row": int(sample.row_index),
                    "subject": sample.subject,
                    "file": sample.file,
                    "question_type": sample.question_type,
                    "full_score": sample.full_score,
                    "human_score": sample.human_score,
                    "detail_score": getattr(sample, "detail_score", ""),
                    "student_answer": sample.student_answer,
                }
            )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate batched scoring with score-band few-shot examples.")
    parser.add_argument("config")
    parser.add_argument("--question-file", default="pingfen/questions.xlsx")
    parser.add_argument("--answer-dir", default="pingfen/answers")
    parser.add_argument("--file-id", default=None)
    parser.add_argument("--answer-file", default=None)
    parser.add_argument("--modes", nargs="+", choices=_SUPPORTED_MODES, default=["baseline", "quant_only", "hawp_quant"])
    parser.add_argument("--output-dir", default="artifacts/scoring_eval/batch_fewshot")
    parser.add_argument("--rows", nargs="+", type=int, default=None)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--limit", type=int, default=20, help="Sequential sample count after --start. Use 0 for full data.")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--batch-size", "--answers-per-prompt", dest="batch_size", type=int, default=10)
    parser.add_argument("--fewshot-per-band", type=int, default=1, help="Random human-scored examples per score band.")
    parser.add_argument("--allow-fewshot-from-eval", action="store_true", help="Allow few-shot examples to overlap with target samples.")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-question-chars", type=int, default=10000)
    parser.add_argument("--max-ref-chars", type=int, default=6000)
    parser.add_argument("--max-answer-chars", type=int, default=4000)
    parser.add_argument("--max-example-answer-chars", type=int, default=1200)
    parser.add_argument("--min-example-chars", type=int, default=4, help="Minimum answer length for few-shot example candidates.")
    parser.add_argument("--max-example-candidate-chars", type=int, default=1200, help="Drop few-shot example candidates longer than this before sampling.")
    parser.add_argument("--min-example-informative-ratio", type=float, default=0.25, help="Minimum alnum/CJK ratio for few-shot example candidates.")
    parser.add_argument("--profile-memory", action="store_true")
    parser.add_argument("--profile-memory-detail", action="store_true")
    parser.add_argument("--profile-memory-detail-samples", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.fewshot_per_band < 0:
        raise ValueError("--fewshot-per-band must be non-negative")
    if args.min_example_chars < 0:
        raise ValueError("--min-example-chars must be non-negative")
    if args.max_example_candidate_chars <= 0:
        raise ValueError("--max-example-candidate-chars must be positive")
    if not (0.0 <= args.min_example_informative_ratio <= 1.0):
        raise ValueError("--min-example-informative-ratio must be between 0 and 1")

    cfg = load_config(args.config)
    cfg.generation.max_new_tokens = int(args.max_new_tokens)
    device = _resolve_device(cfg.train.device)
    output_dir = Path(args.output_dir)
    rng = random.Random(args.seed)
    answer_file_arg = Path(args.answer_file) if args.answer_file else None

    question_row, resolved_answer_file, samples = scoring.load_samples(
        Path(args.question_file),
        Path(args.answer_dir),
        file_id=args.file_id,
        answer_file=answer_file_arg,
        rows=args.rows,
        start=max(0, args.start),
        limit=args.limit if args.limit and args.limit > 0 else None,
        sample_size=args.sample_size,
        rng=rng,
    )
    if not samples:
        raise RuntimeError("No valid target samples loaded from answer file.")

    _pool_question_row, _pool_answer_file, pool = scoring.load_samples(
        Path(args.question_file),
        Path(args.answer_dir),
        file_id=samples[0].file,
        answer_file=answer_file_arg,
        rows=None,
        start=0,
        limit=None,
        sample_size=None,
        rng=random.Random(args.seed),
    )
    exclude_rows = {int(sample.row_index) for sample in samples}
    fewshot_examples, fewshot_report = select_score_band_examples(
        pool,
        exclude_rows=exclude_rows,
        rng=random.Random(args.seed + 1009),
        per_band=args.fewshot_per_band,
        allow_from_eval=args.allow_fewshot_from_eval,
        min_example_chars=args.min_example_chars,
        max_example_candidate_chars=args.max_example_candidate_chars,
        min_informative_ratio=args.min_example_informative_ratio,
    )
    if args.fewshot_per_band > 0 and not fewshot_examples:
        raise RuntimeError(
            "No few-shot examples were selected. Reduce target sample coverage, "
            "or pass --allow-fewshot-from-eval if leakage is acceptable for a smoke test."
        )

    print("=" * 80)
    print(f"[fewshot-batch-scoring] config={args.config}")
    print(f"[fewshot-batch-scoring] model={cfg.model.model_id}")
    print(f"[fewshot-batch-scoring] question_file={args.question_file}")
    print(f"[fewshot-batch-scoring] file_id={samples[0].file}")
    print(f"[fewshot-batch-scoring] answer_file={resolved_answer_file}")
    print(f"[fewshot-batch-scoring] modes={args.modes}")
    print(f"[fewshot-batch-scoring] samples={len(samples)} batch_size={args.batch_size} max_new_tokens={args.max_new_tokens}")
    print(f"[fewshot-batch-scoring] fewshot_n={len(fewshot_examples)} fewshot_rows={[int(s.row_index) for s in fewshot_examples]}")
    print(f"[fewshot-batch-scoring] output_dir={output_dir}")
    print("=" * 80)

    scoring._write_json(output_dir / "question_metadata.json", question_row)
    scoring._write_json(
        output_dir / "sample_selection.json",
        {
            "file_id": samples[0].file,
            "answer_file": str(resolved_answer_file),
            "excel_rows": [int(sample.row_index) for sample in samples],
            "rows_arg": args.rows,
            "start": args.start,
            "limit": args.limit,
            "sample_size": args.sample_size,
            "seed": args.seed,
            "batch_size": args.batch_size,
            "fewshot_per_band": args.fewshot_per_band,
            "allow_fewshot_from_eval": args.allow_fewshot_from_eval,
            "fewshot_score_bands": [
                {"band": band, "lo_ratio": lo, "hi_ratio": hi}
                for band, lo, hi in _SCORE_BANDS
            ],
            "min_example_chars": args.min_example_chars,
            "max_example_candidate_chars": args.max_example_candidate_chars,
            "min_example_informative_ratio": args.min_example_informative_ratio,
        },
    )
    scoring._write_json(output_dir / "fewshot_band_report.json", fewshot_report)
    scoring._write_csv(output_dir / "fewshot_examples.csv", _fewshot_rows(fewshot_examples, fewshot_report))
    scoring._write_json(output_dir / "fewshot_examples.json", _fewshot_rows(fewshot_examples, fewshot_report))

    preview_prompt = build_fewshot_batch_prompt(
        samples[: args.batch_size],
        fewshot_examples,
        max_question_chars=args.max_question_chars,
        max_ref_chars=args.max_ref_chars,
        max_answer_chars=args.max_answer_chars,
        max_example_answer_chars=args.max_example_answer_chars,
    )
    (output_dir / "prompt_preview.txt").parent.mkdir(parents=True, exist_ok=True)
    (output_dir / "prompt_preview.txt").write_text(preview_prompt, encoding="utf-8")
    if args.dry_run:
        print(f"[fewshot-batch-scoring] dry run wrote {output_dir / 'prompt_preview.txt'}")
        return

    summary_rows: list[dict[str, Any]] = []
    all_predictions: list[dict[str, Any]] = []
    for mode in args.modes:
        predictions, summary, memory_profile = run_mode(
            cfg,
            mode,
            samples,
            fewshot_examples,
            device=device,
            batch_size=args.batch_size,
            max_question_chars=args.max_question_chars,
            max_ref_chars=args.max_ref_chars,
            max_answer_chars=args.max_answer_chars,
            max_example_answer_chars=args.max_example_answer_chars,
            profile_memory=args.profile_memory,
            profile_memory_detail=args.profile_memory_detail,
            profile_memory_detail_samples=args.profile_memory_detail_samples,
        )
        mode_dir = output_dir / mode
        scoring._write_jsonl(mode_dir / "predictions.jsonl", predictions)
        scoring._write_csv(mode_dir / "predictions.csv", predictions)
        scoring._write_json(mode_dir / "summary.json", summary)
        if memory_profile is not None:
            scoring._write_json(mode_dir / "memory_profile.json", memory_profile)
            scoring._write_csv(mode_dir / "memory_batches.csv", memory_profile["batches"])
            scoring._write_csv(mode_dir / "memory_samples.csv", memory_profile["samples"])
        summary_rows.append({"mode": mode, **summary})
        all_predictions.extend(predictions)

    scoring._write_csv(output_dir / "summary.csv", summary_rows)
    scoring._write_json(output_dir / "summary.json", summary_rows)
    scoring._write_jsonl(output_dir / "predictions_all.jsonl", all_predictions)

    print("\n[fewshot-batch-scoring] summary")
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
            f"batches={row['n_batches']} answers_per_prompt={row['answers_per_prompt']} "
            f"fewshot={row['fewshot_n']} MAE={row['mae']} normMAE={row['normalized_mae']} "
            f"within1={row['within_1']} within2={row['within_2']} "
            f"within3={row['within_3']} parse_fail={row['parse_fail_n']} pearson={row['pearson']}"
            f"{mem_part}"
        )
    print(f"[fewshot-batch-scoring] wrote {output_dir}")


if __name__ == "__main__":
    main()
