#!/usr/bin/env python
"""Evaluate answer scoring with multiple student answers per prompt.

This is the batched-prompt companion to ``10_eval_scoring_task.py``. It keeps
the same input/output style, but asks the model to score N answers for the same
question in one JSON array.

Example:
  python scripts/10d_eval_scoring_task_batch.py configs/new_rank_scoring_all.yaml \
    --question-file pingfen/questions.xlsx \
    --answer-dir pingfen/answers \
    --file-id history_161 \
    --batch-size 10 \
    --modes baseline hawp_quant \
    --output-dir artifacts/scoring_eval/history_161_batch10
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import random
import re
import sys
from pathlib import Path
from typing import Any

import torch

from hawp_laq.config import load_config
from hawp_laq.runtime.forward_utils import prefill_forward_last_logits
from hawp_laq.runtime.generate import _has_real_past_key_values, _resolve_device, load_baseline_model
from hawp_laq.runtime.mode_runner import generate_by_mode, make_reset_fn, setup_mode


_STOP_TEXT = "</RESULT_JSON>"


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

_SUPPORTED_MODES = ("baseline", "quant_only", "hawp_quant")


def _chunks(items: list[Any], size: int) -> list[list[Any]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def _clean_one_line(text: str, max_chars: int = 300) -> str:
    text = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    text = re.sub(r"\s+", " ", text)
    return text[:max_chars]


def _to_int(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    match = re.search(r"-?\d+", str(value or ""))
    return int(match.group(0)) if match else None


def _score_from_obj(obj: dict[str, Any]) -> float:
    for key in ("score", "pred_score", "points", "grade"):
        if key in obj:
            return scoring._to_float(obj.get(key))
    return float("nan")


def _row_from_obj(obj: dict[str, Any]) -> int | None:
    for key in ("excel_row", "row", "row_index"):
        if key in obj:
            return _to_int(obj.get(key))
    return None


def _answer_id_from_obj(obj: dict[str, Any]) -> int | None:
    for key in ("answer_id", "id"):
        if key in obj:
            return _to_int(obj.get(key))
    return None


def _reason_from_obj(obj: dict[str, Any]) -> str:
    for key in ("reason", "rationale", "comment", "explanation"):
        if key in obj:
            return str(obj.get(key, "")).strip()
    return ""


def _extract_json_payload(raw: str) -> Any | None:
    text = (raw or "").strip()
    tag_match = re.search(r"<RESULT_JSON>\s*(.*?)\s*</RESULT_JSON>", text, flags=re.IGNORECASE | re.DOTALL)
    if tag_match:
        text = tag_match.group(1).strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)

    for candidate in (text,):
        try:
            return json.loads(candidate)
        except Exception:
            pass

    array_start = text.find("[")
    array_end = text.rfind("]")
    if array_start >= 0 and array_end > array_start:
        try:
            return json.loads(text[array_start : array_end + 1])
        except Exception:
            pass

    obj_start = text.find("{")
    obj_end = text.rfind("}")
    if obj_start >= 0 and obj_end > obj_start:
        try:
            return json.loads(text[obj_start : obj_end + 1])
        except Exception:
            pass

    return None


def parse_batch_response(
    text: str,
    samples: list[Any],
) -> dict[int, dict[str, Any]]:
    """Parse model output into a map keyed by Excel row number."""
    expected_rows = [int(sample.row_index) for sample in samples]
    full_score_by_row = {int(sample.row_index): float(sample.full_score) for sample in samples}
    raw = (text or "").strip()
    payload = _extract_json_payload(raw)
    parsed_json = payload is not None
    entries: list[Any] = []

    if isinstance(payload, list):
        entries = payload
    elif isinstance(payload, dict):
        for key in ("results", "scores", "items", "answers"):
            value = payload.get(key)
            if isinstance(value, list):
                entries = value
                break
        if not entries and any(key in payload for key in ("score", "pred_score", "points")):
            entries = [payload]

    parsed: dict[int, dict[str, Any]] = {}
    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict):
            continue
        answer_id = _answer_id_from_obj(entry)
        row = expected_rows[answer_id - 1] if answer_id is not None and 1 <= answer_id <= len(expected_rows) else None
        if row is None:
            row = _row_from_obj(entry)
        if row is None and idx < len(expected_rows):
            row = expected_rows[idx]
        if row not in full_score_by_row:
            continue
        score = _score_from_obj(entry)
        if math.isnan(score):
            continue
        full_score = full_score_by_row[row]
        parsed[row] = {
            "pred_score": min(max(float(score), 0.0), full_score),
            "reason": _reason_from_obj(entry),
            "parsed_json": parsed_json,
        }

    if not parsed:
        # Last-resort regex fallback for outputs that are almost JSON but broken.
        pattern = re.compile(
            r"(?:excel_row|row|row_index)\D*(?P<row>\d+).*?"
            r"(?:score|pred_score|points|grade)\D*(?P<score>-?\d+(?:\.\d+)?)",
            flags=re.IGNORECASE | re.DOTALL,
        )
        for match in pattern.finditer(raw):
            row = int(match.group("row"))
            if row not in full_score_by_row:
                continue
            score = float(match.group("score"))
            parsed[row] = {
                "pred_score": min(max(score, 0.0), full_score_by_row[row]),
                "reason": _clean_one_line(raw),
                "parsed_json": False,
            }
        if not parsed:
            id_pattern = re.compile(
                r"(?:answer_id|id)\D*(?P<answer_id>\d+).*?"
                r"(?:score|pred_score|points|grade)\D*(?P<score>-?\d+(?:\.\d+)?)",
                flags=re.IGNORECASE | re.DOTALL,
            )
            for match in id_pattern.finditer(raw):
                answer_id = int(match.group("answer_id"))
                if not (1 <= answer_id <= len(expected_rows)):
                    continue
                row = expected_rows[answer_id - 1]
                score = float(match.group("score"))
                parsed[row] = {
                    "pred_score": min(max(score, 0.0), full_score_by_row[row]),
                    "reason": _clean_one_line(raw),
                    "parsed_json": False,
                }

    for row in expected_rows:
        parsed.setdefault(
            row,
            {
                "pred_score": None,
                "reason": _clean_one_line(raw),
                "parsed_json": parsed_json,
            },
        )
    return parsed


def build_batch_prompt(
    samples: list[Any],
    *,
    max_question_chars: int,
    max_ref_chars: int,
    max_answer_chars: int,
) -> str:
    if not samples:
        raise ValueError("Cannot build a batch prompt with no samples.")

    first = samples[0]
    subject = first.subject or "unknown subject"
    role = f"{subject}{first.question_type} grading expert" if first.question_type else f"{subject} grading expert"
    answer_blocks = []
    for answer_id, sample in enumerate(samples, start=1):
        answer_blocks.append(
            f"### answer_id: {answer_id}\n"
            f"excel_row: {sample.row_index}\n"
            f"{scoring._truncate(sample.student_answer, max_answer_chars)}"
        )

    return f"""You are a strict {role}. Score each student answer independently for the same question.

Rules:
1. The score must be between 0 and {first.full_score:g}.
2. Use the reference answer and grading standard implied by the full score.
3. Do not compare students with each other; grade each answer independently.
4. Return only a valid JSON array. Do not output Markdown.
5. Put the JSON array between <RESULT_JSON> and </RESULT_JSON>.
6. The array must contain exactly {len(samples)} objects, one for each input answer_id from 1 to {len(samples)}.
7. Each object must have: "answer_id", "score", and "reason". Keep each reason concise.
8. Copy answer_id exactly from the target answer block. Do not skip, repeat, or renumber answer_id values.

JSON format:
<RESULT_JSON>
[
  {{"answer_id": 1, "score": <number>, "reason": "<short reason>"}}
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

Student answers:

{chr(10).join(answer_blocks)}
"""


def format_batch_model_prompt(tokenizer, prompt: str) -> str:
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


def _eos_token_ids(tokenizer) -> set[int]:
    ids: set[int] = set()
    for value in (
        getattr(tokenizer, "eos_token_id", None),
        getattr(tokenizer, "pad_token_id", None),
    ):
        if isinstance(value, int):
            ids.add(value)
        elif isinstance(value, (list, tuple)):
            ids.update(int(item) for item in value if isinstance(item, int))
    return ids


def _should_stop_generated(tokenizer, generated_ids: torch.Tensor, eos_ids: set[int]) -> bool:
    last_id = int(generated_ids[0, -1].item())
    if last_id in eos_ids:
        return True
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return _STOP_TEXT in generated_text


@torch.inference_mode()
def generate_by_mode_until_result(
    model,
    tokenizer,
    prompts: list[str],
    cfg,
    mode: str,
    coordinator=None,
    kv_manager=None,
) -> list[str]:
    """Generate scoring responses and stop once </RESULT_JSON> appears.

    This is intentionally local to the scoring scripts. It mirrors the runtime
    stepwise greedy loops, but returns only newly generated assistant text
    rather than prompt + output.
    """
    if mode not in ("baseline", "hawp_only", "hawp_quant", "hawp_quant_all", "hawp_quant_sched", "quant_only"):
        return generate_by_mode(model, tokenizer, prompts, cfg, mode, coordinator=coordinator, kv_manager=kv_manager)

    max_new_tokens = int(cfg.generation.max_new_tokens)
    eos_ids = _eos_token_ids(tokenizer)
    use_external_past = mode in ("baseline", "hawp_only")
    results: list[str] = []

    for prompt in prompts:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        bsz, prompt_len = input_ids.shape
        prefill_mask = torch.ones(bsz, prompt_len, device=model.device, dtype=torch.long)
        prefill_pos = torch.arange(prompt_len, device=model.device, dtype=torch.long).unsqueeze(0)

        outputs = prefill_forward_last_logits(
            model,
            input_ids=input_ids,
            attention_mask=prefill_mask,
            position_ids=prefill_pos,
            use_cache=True,
        )
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        generated_ids = next_token

        if coordinator is not None:
            coordinator.on_prefill(prompt_len)

        past_kv = outputs.past_key_values if use_external_past else None
        cur_pos = prompt_len

        if _should_stop_generated(tokenizer, generated_ids, eos_ids):
            results.append(tokenizer.decode(generated_ids[0], skip_special_tokens=True))
            continue

        for _ in range(max_new_tokens - 1):
            attention_mask = torch.ones(1, cur_pos + 1, device=model.device, dtype=torch.long)
            position_ids = torch.tensor([[cur_pos]], device=model.device, dtype=torch.long)
            fwd_kw: dict[str, Any] = {
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "use_cache": True,
            }
            if use_external_past and _has_real_past_key_values(past_kv):
                fwd_kw["past_key_values"] = past_kv

            outputs = model(input_ids=next_token, **fwd_kw)
            past_kv = outputs.past_key_values if use_external_past else None
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            cur_pos += 1

            if coordinator is not None:
                coordinator.on_new_token()

            if _should_stop_generated(tokenizer, generated_ids, eos_ids):
                break

        results.append(tokenizer.decode(generated_ids[0], skip_special_tokens=True))

    return results


@torch.inference_mode()
def retry_missing_predictions(
    model,
    tokenizer,
    cfg,
    mode: str,
    parsed: dict[int, dict[str, Any]],
    samples: list[Any],
    *,
    attempts: int,
    retry_batch_size: int,
    reset_fn,
    prompt_builder,
    prompt_formatter,
    log_prefix: str,
    coordinator=None,
    kv_manager=None,
) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]], dict[str, int]]:
    """Retry only rows that did not receive a parsed score."""
    retry_info: dict[int, dict[str, Any]] = {
        int(sample.row_index): {
            "retry_attempts": 0,
            "retry_success": False,
            "retry_raw_output": "",
        }
        for sample in samples
    }
    missing = [sample for sample in samples if parsed[int(sample.row_index)]["pred_score"] is None]
    stats = {
        "retry_missing_initial_n": len(missing),
        "retry_attempted_n": 0,
        "retry_recovered_n": 0,
        "retry_failed_n": len(missing),
    }
    if attempts <= 0 or retry_batch_size <= 0 or not missing:
        return parsed, retry_info, stats

    attempted_rows: set[int] = set()
    recovered_rows: set[int] = set()
    for attempt_i in range(1, attempts + 1):
        if not missing:
            break
        next_missing: list[Any] = []
        for retry_batch in _chunks(missing, retry_batch_size):
            reset_fn()
            prompt = prompt_formatter(tokenizer, prompt_builder(retry_batch))
            output = generate_by_mode_until_result(
                model,
                tokenizer,
                [prompt],
                cfg,
                mode,
                coordinator=coordinator,
                kv_manager=kv_manager,
            )[0]
            retry_parsed = parse_batch_response(output, retry_batch)
            for sample in retry_batch:
                row = int(sample.row_index)
                attempted_rows.add(row)
                info = retry_info[row]
                info["retry_attempts"] = attempt_i
                info["retry_raw_output"] = output
                retry_score = retry_parsed[row]["pred_score"]
                if retry_score is None:
                    next_missing.append(sample)
                    continue
                parsed[row] = retry_parsed[row]
                info["retry_success"] = True
                recovered_rows.add(row)

        print(
            f"{log_prefix} retry {attempt_i}/{attempts} "
            f"recovered={len(recovered_rows)}/{stats['retry_missing_initial_n']}"
        )
        missing = next_missing

    stats["retry_attempted_n"] = len(attempted_rows)
    stats["retry_recovered_n"] = len(recovered_rows)
    stats["retry_failed_n"] = len(missing)
    return parsed, retry_info, stats


@torch.inference_mode()
def run_mode(
    cfg,
    mode: str,
    samples: list[Any],
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
    memory_samples: list[dict[str, Any]] = []
    retry_stats_total = {
        "retry_missing_initial_n": 0,
        "retry_attempted_n": 0,
        "retry_recovered_n": 0,
        "retry_failed_n": 0,
    }
    sample_i_by_row = {int(sample.row_index): i for i, sample in enumerate(samples, start=1)}
    batches = _chunks(samples, batch_size)

    for batch_i, batch in enumerate(batches, start=1):
        reset_fn()
        prompt = format_batch_model_prompt(
            tokenizer,
            build_batch_prompt(
                batch,
                max_question_chars=max_question_chars,
                max_ref_chars=max_ref_chars,
                max_answer_chars=max_answer_chars,
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
            batch_size=len(batch),
            excel_rows=[int(sample.row_index) for sample in batch],
            prompt_tokens=int(prompt_len),
        )
        output = generate_by_mode_until_result(model, tokenizer, [prompt], cfg, mode, coordinator=coordinator, kv_manager=kv_manager)[0]
        parsed = parse_batch_response(output, batch)
        initial_pred_by_row = {int(sample.row_index): parsed[int(sample.row_index)]["pred_score"] for sample in batch}
        parsed, retry_info_by_row, retry_stats = retry_missing_predictions(
            model,
            tokenizer,
            cfg,
            mode,
            parsed,
            batch,
            attempts=retry_missing_attempts if retry_missing else 0,
            retry_batch_size=retry_missing_batch_size,
            reset_fn=reset_fn,
            coordinator=coordinator,
            kv_manager=kv_manager,
            prompt_builder=lambda retry_batch: build_batch_prompt(
                retry_batch,
                max_question_chars=max_question_chars,
                max_ref_chars=max_ref_chars,
                max_answer_chars=max_answer_chars,
            ),
            prompt_formatter=format_batch_model_prompt,
            log_prefix=f"[{mode}] batch {batch_i}/{len(batches)}",
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
        memory_sample = {
            "mode": mode,
            "batch_i": batch_i,
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
        memory_samples.append(memory_sample)
        recorder.record(f"{mode}.batch{batch_i}.generate.after", **memory_sample)
        recorder.current_sample_i = None

        for item_i, sample in enumerate(batch, start=1):
            info = parsed[int(sample.row_index)]
            retry_info = retry_info_by_row[int(sample.row_index)]
            pred_score = info["pred_score"]
            row = {
                "mode": mode,
                "batch_i": batch_i,
                "batch_size": len(batch),
                "item_i": item_i,
                "sample_i": sample_i_by_row[int(sample.row_index)],
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
                row.update(memory_sample)
            rows.append(row)

        valid_n = sum(1 for item in batch if parsed[int(item.row_index)]["pred_score"] is not None)
        print(f"[{mode}] batch {batch_i}/{len(batches)} rows={ [s.row_index for s in batch] } parsed={valid_n}/{len(batch)}")

    summary = scoring.summarize_predictions(rows)
    summary.update({"answers_per_prompt": batch_size, "n_batches": len(batches), **retry_stats_total})
    memory_profile = None
    if profile_memory:
        peak_values = [int(item["peak_gpu_bytes"]) for item in memory_samples]
        over_setup_values = [int(item["peak_over_setup_bytes"]) for item in memory_samples]
        cache_values = [int(item.get("cache_runtime_bytes") or 0) for item in memory_samples]
        baseline_kv_values = [int(item.get("baseline_kv_bytes") or 0) for item in memory_samples]
        memory_summary = {
            "mode": mode,
            "n_batches": len(memory_samples),
            "answers_per_prompt": batch_size,
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
            "batches": memory_samples,
            "samples": memory_samples,
            "records": recorder.records,
        }

    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows, summary, memory_profile


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate LLM scoring with multiple answers per prompt.")
    parser.add_argument("config")
    parser.add_argument("--question-file", default="pingfen/questions.xlsx")
    parser.add_argument("--answer-dir", default="pingfen/answers")
    parser.add_argument("--file-id", default=None)
    parser.add_argument("--answer-file", default=None)
    parser.add_argument("--modes", nargs="+", choices=_SUPPORTED_MODES, default=["baseline", "quant_only", "hawp_quant"])
    parser.add_argument("--output-dir", default="artifacts/scoring_eval/batch")
    parser.add_argument("--rows", nargs="+", type=int, default=None)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--limit", type=int, default=20, help="Sequential sample count after --start. Use 0 for full data.")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--batch-size", "--answers-per-prompt", dest="batch_size", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-question-chars", type=int, default=10000)
    parser.add_argument("--max-ref-chars", type=int, default=6000)
    parser.add_argument("--max-answer-chars", type=int, default=4000)
    parser.add_argument("--profile-memory", action="store_true", help="Record CUDA allocated/reserved/peak memory per mode and batch.")
    parser.add_argument("--profile-memory-detail", action="store_true", help="Also record model block and HAWP internal memory markers for the first N batches.")
    parser.add_argument("--profile-memory-detail-samples", type=int, default=1, help="Number of batches with detailed block/HAWP memory records.")
    parser.add_argument("--no-retry-missing", dest="retry_missing", action="store_false", help="Disable retry for rows missing from the first batch response.")
    parser.set_defaults(retry_missing=True)
    parser.add_argument("--retry-missing-attempts", type=int, default=1, help="Retry rounds for rows without a parsed score.")
    parser.add_argument("--retry-missing-batch-size", type=int, default=1, help="Answers per prompt when retrying missing rows.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.retry_missing_attempts < 0:
        raise ValueError("--retry-missing-attempts must be non-negative")
    if args.retry_missing_batch_size <= 0:
        raise ValueError("--retry-missing-batch-size must be positive")

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
        raise RuntimeError("No valid samples loaded from answer file.")

    print("=" * 80)
    print(f"[batch-scoring] config={args.config}")
    print(f"[batch-scoring] model={cfg.model.model_id}")
    print(f"[batch-scoring] question_file={args.question_file}")
    print(f"[batch-scoring] file_id={samples[0].file}")
    print(f"[batch-scoring] answer_file={resolved_answer_file}")
    print(f"[batch-scoring] modes={args.modes}")
    print(f"[batch-scoring] rows={args.rows}")
    print(f"[batch-scoring] samples={len(samples)} batch_size={args.batch_size} max_new_tokens={args.max_new_tokens}")
    print(f"[batch-scoring] output_dir={output_dir}")
    print("=" * 80)

    scoring._write_json(output_dir / "question_metadata.json", question_row)
    scoring._write_json(
        output_dir / "sample_selection.json",
        {
            "file_id": samples[0].file,
            "answer_file": str(resolved_answer_file),
            "excel_rows": [sample.row_index for sample in samples],
            "rows_arg": args.rows,
            "start": args.start,
            "limit": args.limit,
            "sample_size": args.sample_size,
            "seed": args.seed,
            "batch_size": args.batch_size,
            "retry_missing": args.retry_missing,
            "retry_missing_attempts": args.retry_missing_attempts,
            "retry_missing_batch_size": args.retry_missing_batch_size,
        },
    )
    preview_prompt = build_batch_prompt(
        samples[: args.batch_size],
        max_question_chars=args.max_question_chars,
        max_ref_chars=args.max_ref_chars,
        max_answer_chars=args.max_answer_chars,
    )
    (output_dir / "prompt_preview.txt").parent.mkdir(parents=True, exist_ok=True)
    (output_dir / "prompt_preview.txt").write_text(preview_prompt, encoding="utf-8")
    if args.dry_run:
        print(f"[batch-scoring] dry run wrote {output_dir / 'prompt_preview.txt'}")
        return

    summary_rows: list[dict[str, Any]] = []
    all_predictions: list[dict[str, Any]] = []
    for mode in args.modes:
        predictions, summary, memory_profile = run_mode(
            cfg,
            mode,
            samples,
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
        if memory_profile is not None:
            scoring._write_json(mode_dir / "memory_profile.json", memory_profile)
            scoring._write_csv(mode_dir / "memory_batches.csv", memory_profile["batches"])
            scoring._write_csv(mode_dir / "memory_samples.csv", memory_profile["samples"])
        summary_rows.append({"mode": mode, **summary})
        all_predictions.extend(predictions)

    scoring._write_csv(output_dir / "summary.csv", summary_rows)
    scoring._write_json(output_dir / "summary.json", summary_rows)
    scoring._write_jsonl(output_dir / "predictions_all.jsonl", all_predictions)

    print("\n[batch-scoring] summary")
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
            f"MAE={row['mae']} normMAE={row['normalized_mae']} "
            f"within1={row['within_1']} within2={row['within_2']} "
            f"within3={row['within_3']} parse_fail={row['parse_fail_n']} pearson={row['pearson']}"
            f"{mem_part}"
        )
    print(f"[batch-scoring] wrote {output_dir}")


if __name__ == "__main__":
    main()
