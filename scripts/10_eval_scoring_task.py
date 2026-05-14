"""Evaluate answer scoring with baseline / quant_only / hawp_quant modes.

Example:
  python scripts/10_eval_scoring_task.py configs/new_rank.yaml \
    --question-file pingfen/题目信息.xlsx \
    --answer-dir pingfen/评分数据 \
    --file-id 历史_161 \
    --rows 2 5 20 \
    --modes baseline quant_only hawp_quant \
    --limit 0 \
    --output-dir artifacts/scoring_eval/history_161_smoke
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import random
import re
import statistics
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import torch

from hawp_laq.config import load_config
from hawp_laq.modeling.attention_hawp import HAWPAttention
from hawp_laq.runtime.cache_stats import collect_cache_stats, compute_baseline_kv_bytes
from hawp_laq.runtime.generate import _resolve_device, load_baseline_model
from hawp_laq.runtime.mode_runner import generate_by_mode, make_reset_fn, setup_mode
from hawp_laq.utils.memory import format_nbytes


_SUPPORTED_MODES = ("baseline", "quant_only", "hawp_quant")
_NS = {
    "a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
}


@dataclass
class ScoringSample:
    row_index: int
    subject: str
    file: str
    question_type: str
    question: str
    reference_answer: str
    full_score: float
    human_score: float
    detail_score: str
    answer_chars: int | None
    student_answer: str


def _cell_col(ref: str) -> int:
    letters = "".join(ch for ch in ref if ch.isalpha())
    out = 0
    for ch in letters:
        out = out * 26 + (ord(ch.upper()) - ord("A") + 1)
    return max(out - 1, 0)


def _read_shared_strings(zf: zipfile.ZipFile) -> list[str]:
    try:
        data = zf.read("xl/sharedStrings.xml")
    except KeyError:
        return []
    root = ET.fromstring(data)
    values: list[str] = []
    for si in root.findall("a:si", _NS):
        parts = [t.text or "" for t in si.findall(".//a:t", _NS)]
        values.append("".join(parts))
    return values


def _cell_text(cell: ET.Element, shared_strings: list[str]) -> str:
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        texts = [t.text or "" for t in cell.findall(".//a:t", _NS)]
        return "".join(texts).strip()
    v = cell.find("a:v", _NS)
    if v is None or v.text is None:
        return ""
    raw = v.text
    if cell_type == "s":
        idx = int(raw)
        return shared_strings[idx].strip() if 0 <= idx < len(shared_strings) else ""
    return raw.strip()


def read_xlsx_first_sheet(path: Path) -> list[list[str]]:
    with zipfile.ZipFile(path) as zf:
        shared_strings = _read_shared_strings(zf)
        root = ET.fromstring(zf.read("xl/worksheets/sheet1.xml"))
    rows: list[list[str]] = []
    for row in root.findall(".//a:sheetData/a:row", _NS):
        values: list[str] = []
        for cell in row.findall("a:c", _NS):
            idx = _cell_col(cell.attrib.get("r", "A1"))
            while len(values) <= idx:
                values.append("")
            values[idx] = _cell_text(cell, shared_strings)
        rows.append(values)
    return rows


def _row_dicts(rows: list[list[str]]) -> list[dict[str, str]]:
    if not rows:
        return []
    header = [h.strip() for h in rows[0]]
    out: list[dict[str, str]] = []
    for row in rows[1:]:
        if not any(str(x).strip() for x in row):
            continue
        item = {header[i]: row[i].strip() if i < len(row) else "" for i in range(len(header))}
        out.append(item)
    return out


def _first_present(row: dict[str, str], names: list[str], fallback_idx: int | None = None) -> str:
    for name in names:
        if name in row:
            return row.get(name, "")
    if fallback_idx is not None:
        values = list(row.values())
        if fallback_idx < len(values):
            return values[fallback_idx]
    return ""


def _to_float(value: Any, default: float = float("nan")) -> float:
    text = str(value).strip()
    if not text:
        return default
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return default
    return float(match.group(0))


def resolve_file_id(
    question_rows: list[dict[str, str]],
    answer_dir: Path,
    *,
    file_id: str | None,
    answer_file: Path | None,
    rng: random.Random,
) -> tuple[str, Path, dict[str, str]]:
    if answer_file is not None:
        resolved_file_id = answer_file.stem
        resolved_answer_file = answer_file
    elif file_id:
        resolved_file_id = file_id
        resolved_answer_file = answer_dir / f"{file_id}.xlsx"
    else:
        candidates: list[tuple[str, Path, dict[str, str]]] = []
        for row in question_rows:
            candidate_id = _first_present(row, ["文件", "file"], fallback_idx=1)
            if not candidate_id:
                continue
            candidate_file = answer_dir / f"{candidate_id}.xlsx"
            if candidate_file.exists():
                candidates.append((candidate_id, candidate_file, row))
        if not candidates:
            raise ValueError(f"No question rows have matching answer files under {answer_dir}")
        return rng.choice(candidates)

    if not resolved_answer_file.exists():
        raise FileNotFoundError(f"Cannot find answer file: {resolved_answer_file}")

    question_row = None
    for row in question_rows:
        row_file_id = _first_present(row, ["文件", "file"], fallback_idx=1)
        if row_file_id == resolved_file_id:
            question_row = row
            break
    if question_row is None:
        available = [_first_present(row, ["文件", "file"], fallback_idx=1) for row in question_rows[:20]]
        raise ValueError(f"Cannot find question metadata for file_id='{resolved_file_id}'. First question files: {available}")

    return resolved_file_id, resolved_answer_file, question_row


def load_samples(
    question_file: Path,
    answer_dir: Path,
    *,
    file_id: str | None,
    answer_file: Path | None,
    rows: list[int] | None,
    start: int,
    limit: int | None,
    sample_size: int | None,
    rng: random.Random,
) -> tuple[dict[str, str], Path, list[ScoringSample]]:
    question_rows = _row_dicts(read_xlsx_first_sheet(question_file))
    resolved_file_id, resolved_answer_file, question_row = resolve_file_id(
        question_rows,
        answer_dir,
        file_id=file_id,
        answer_file=answer_file,
        rng=rng,
    )
    answer_rows = _row_dicts(read_xlsx_first_sheet(resolved_answer_file))

    subject = _first_present(question_row, ["科目", "subject"], fallback_idx=0)
    question_type = _first_present(question_row, ["题目类型", "question_type"], fallback_idx=2)
    question = _first_present(question_row, ["题目", "question"], fallback_idx=3)
    reference_answer = _first_present(question_row, ["参考答案", "reference_answer"], fallback_idx=4)
    full_score = _to_float(_first_present(question_row, ["满分值", "满分", "full_score"], fallback_idx=5), default=0.0)

    all_samples: list[ScoringSample] = []
    for offset, row in enumerate(answer_rows, start=2):
        human_score = _to_float(_first_present(row, ["最终分", "真实评分", "score"], fallback_idx=0))
        student_answer = _first_present(row, ["识别文本", "答案", "answer"], fallback_idx=3)
        if math.isnan(human_score) or not student_answer.strip():
            continue
        chars = _to_float(_first_present(row, ["字数", "chars"], fallback_idx=2), default=float("nan"))
        all_samples.append(
            ScoringSample(
                row_index=offset,
                subject=subject,
                file=resolved_file_id,
                question_type=question_type,
                question=question,
                reference_answer=reference_answer,
                full_score=full_score,
                human_score=human_score,
                detail_score=_first_present(row, ["明细分", "detail_score"], fallback_idx=1),
                answer_chars=None if math.isnan(chars) else int(chars),
                student_answer=student_answer,
            )
        )

    if rows:
        wanted = set(rows)
        selected = [sample for sample in all_samples if sample.row_index in wanted]
        missing = sorted(wanted - {sample.row_index for sample in selected})
        if missing:
            print(f"[scoring] warning: requested Excel rows not found or invalid: {missing}")
        return question_row, resolved_answer_file, selected

    selected = all_samples[max(0, start):]
    if sample_size is not None and sample_size > 0:
        if sample_size < len(selected):
            selected = sorted(rng.sample(selected, sample_size), key=lambda item: item.row_index)
    elif limit is not None:
        selected = selected[:limit]

    return question_row, resolved_answer_file, selected


def _truncate(text: str, max_chars: int) -> str:
    text = str(text or "").strip()
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[已截断]"


def build_prompt(sample: ScoringSample, *, max_question_chars: int, max_ref_chars: int, max_answer_chars: int) -> str:
    subject = sample.subject or "该科目"
    role = f"{subject}{sample.question_type}的专门评分员" if sample.question_type else f"{subject}科目的专门评分员"
    return f"""你是{role}。请根据题目、参考答案和满分，对考生答案打分。

要求：
1. 分数必须是 0 到 {sample.full_score:g} 之间的数字。
2. 只根据考生答案中明确表达的内容给分，不要脑补。
3. 如果答案有错别字或识别噪声，但意思明确，可以酌情给分。
4. 输出必须是 JSON，不要输出 Markdown，不要输出多余文字。

JSON 格式：
{{"score": 数字, "reason": "简短说明给分依据"}}

科目：{sample.subject}
文件：{sample.file}
题目类型：{sample.question_type}
满分：{sample.full_score:g}

题目：
{_truncate(sample.question, max_question_chars)}

参考答案：
{_truncate(sample.reference_answer, max_ref_chars)}

考生答案：
{_truncate(sample.student_answer, max_answer_chars)}
"""


def format_model_prompt(tokenizer, prompt: str) -> str:
    messages = [
        {"role": "system", "content": "你是一个严格、稳定、只输出 JSON 的考试评分助手。"},
        {"role": "user", "content": prompt},
    ]
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return messages[0]["content"] + "\n\n" + messages[1]["content"]


def parse_score_response(text: str, full_score: float) -> tuple[float | None, str, bool]:
    raw = (text or "").strip()
    parsed_json = False
    obj_text = ""
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        obj_text = raw[start:end + 1]
        try:
            obj = json.loads(obj_text)
            score = _to_float(obj.get("score"))
            reason = str(obj.get("reason", "")).strip()
            if not math.isnan(score):
                return min(max(score, 0.0), full_score), reason, True
        except Exception:
            pass

    patterns = [
        r'"score"\s*:\s*(-?\d+(?:\.\d+)?)',
        r"score\s*[:：]\s*(-?\d+(?:\.\d+)?)",
        r"得分\s*[:：]\s*(-?\d+(?:\.\d+)?)",
        r"评分\s*[:：]\s*(-?\d+(?:\.\d+)?)",
        r"分数\s*[:：]\s*(-?\d+(?:\.\d+)?)",
    ]
    for pattern in patterns:
        match = re.search(pattern, raw, flags=re.IGNORECASE)
        if match:
            score = min(max(float(match.group(1)), 0.0), full_score)
            return score, raw[:300], parsed_json
    return None, raw[:300], parsed_json


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    mx, my = statistics.mean(xs), statistics.mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mx) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - my) ** 2 for y in ys))
    if den_x == 0 or den_y == 0:
        return None
    return num / (den_x * den_y)


def summarize_predictions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [r for r in rows if r.get("pred_score") is not None]
    errors = [float(r["pred_score"]) - float(r["human_score"]) for r in valid]
    abs_errors = [abs(e) for e in errors]
    norm_abs_errors = [
        abs(float(r["pred_score"]) - float(r["human_score"])) / float(r["full_score"])
        for r in valid
        if float(r.get("full_score") or 0.0) > 0.0
    ]
    pred = [float(r["pred_score"]) for r in valid]
    human = [float(r["human_score"]) for r in valid]
    n = len(rows)
    nv = len(valid)
    return {
        "n": n,
        "valid_n": nv,
        "parse_fail_n": n - nv,
        "mae": statistics.mean(abs_errors) if abs_errors else None,
        "normalized_mae": statistics.mean(norm_abs_errors) if norm_abs_errors else None,
        "rmse": math.sqrt(statistics.mean([e * e for e in errors])) if errors else None,
        "bias": statistics.mean(errors) if errors else None,
        "within_0_5": sum(e <= 0.5 for e in abs_errors) / nv if nv else None,
        "within_1": sum(e <= 1.0 for e in abs_errors) / nv if nv else None,
        "within_2": sum(e <= 2.0 for e in abs_errors) / nv if nv else None,
        "within_3": sum(e <= 3.0 for e in abs_errors) / nv if nv else None,
        "exact_int": sum(round(p) == round(h) for p, h in zip(pred, human)) / nv if nv else None,
        "pearson": _pearson(pred, human),
    }


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


class MemoryRecorder:
    def __init__(self, *, enabled: bool, detail: bool, detail_samples: int, synchronize: bool = True) -> None:
        self.enabled = enabled
        self.detail = detail
        self.detail_samples = max(0, int(detail_samples))
        self.synchronize = synchronize
        self.records: list[dict[str, Any]] = []
        self.t0 = time.perf_counter()
        self.current_sample_i: int | None = None
        self._last_allocated = 0
        self._last_peak = 0

    def detail_active(self) -> bool:
        return (
            self.enabled
            and self.detail
            and self.current_sample_i is not None
            and self.current_sample_i <= self.detail_samples
        )

    def reset_delta_baseline(self) -> None:
        self._last_allocated = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        self._last_peak = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0

    def record(self, label: str, *, detail_only: bool = False, **meta: Any) -> None:
        if not self.enabled:
            return
        if detail_only and not self.detail_active():
            return
        if torch.cuda.is_available() and self.synchronize:
            torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        reserved = torch.cuda.memory_reserved() if torch.cuda.is_available() else 0
        peak = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        rec = {
            "i": len(self.records),
            "t_s": round(time.perf_counter() - self.t0, 6),
            "sample_i": self.current_sample_i,
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


def install_memory_detail_probes(model, recorder: MemoryRecorder) -> int:
    n_hooks = 0

    def add_hooks(module, label: str, **meta: Any) -> None:
        nonlocal n_hooks

        def pre_hook(_module, args):
            recorder.record(f"{label}.before", detail_only=True, **meta, **_io_meta(args, prefix="input"))

        def post_hook(_module, args, output):
            recorder.record(f"{label}.after", detail_only=True, **meta, **_io_meta(output, prefix="output"))

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

    norm_path, final_norm = _find_final_norm(model)
    if final_norm is not None:
        add_hooks(final_norm, "model.final_norm", module_path=norm_path, block_part="final_norm")

    lm_head = getattr(model, "lm_head", None)
    if lm_head is not None:
        add_hooks(lm_head, "model.lm_head", block_part="lm_head")

    for mod in model.modules():
        if not isinstance(mod, HAWPAttention):
            continue
        layer_idx = getattr(mod, "layer_idx", None)

        def marker_callback(_module, name, meta, __layer=layer_idx):
            recorder.record(
                f"hawp.layer{__layer}.marker.{name}",
                detail_only=True,
                layer=__layer,
                block_part="hawp_marker",
                marker=name,
                **meta,
            )

        mod._memory_marker_callback = marker_callback
        n_hooks += 1

    return n_hooks


def _cache_stats_dict(model, kv_manager, peak_gpu_bytes: int, baseline_kv_bytes: int) -> dict[str, Any]:
    stats = collect_cache_stats(model, kv_manager, peak_gpu_bytes=peak_gpu_bytes)
    stats.baseline_kv_bytes = int(baseline_kv_bytes)
    return {
        "cache_impl": stats.impl,
        "cache_tokens_total": stats.cache_tokens_total,
        "cache_runtime_bytes": stats.cache_runtime_bytes,
        "cache_runtime": format_nbytes(stats.cache_runtime_bytes),
        "cache_compressed_bytes": stats.cache_compressed_bytes,
        "cache_compressed": format_nbytes(stats.cache_compressed_bytes),
        "baseline_kv_bytes": stats.baseline_kv_bytes,
        "baseline_kv": format_nbytes(stats.baseline_kv_bytes),
        "kv_compression_ratio": stats.kv_compression_ratio,
        "recent_tokens": stats.recent_tokens,
        "archive_tokens": stats.archive_tokens,
        "meta_bytes": stats.meta_bytes,
    }


@torch.inference_mode()
def run_mode(
    cfg,
    mode: str,
    samples: list[ScoringSample],
    *,
    device: str,
    max_question_chars: int,
    max_ref_chars: int,
    max_answer_chars: int,
    profile_memory: bool,
    profile_memory_detail: bool,
    profile_memory_detail_samples: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any] | None]:
    print(f"\n[{mode}] loading model")
    recorder = MemoryRecorder(
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
    n_detail_hooks = install_memory_detail_probes(model, recorder) if profile_memory_detail else 0
    recorder.record(f"{mode}.setup.after", n_detail_hooks=n_detail_hooks)
    setup_allocated = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

    rows: list[dict[str, Any]] = []
    memory_samples: list[dict[str, Any]] = []
    for i, sample in enumerate(samples, start=1):
        reset_fn()
        prompt = format_model_prompt(
            tokenizer,
            build_prompt(
                sample,
                max_question_chars=max_question_chars,
                max_ref_chars=max_ref_chars,
                max_answer_chars=max_answer_chars,
            ),
        )
        prompt_len = tokenizer(prompt, return_tensors="pt").input_ids.shape[1]
        total_tokens_estimate = int(prompt_len + cfg.generation.max_new_tokens)
        baseline_kv_bytes = compute_baseline_kv_bytes(model, total_tokens_estimate)
        recorder.current_sample_i = i
        if torch.cuda.is_available() and profile_memory:
            torch.cuda.reset_peak_memory_stats()
            recorder.reset_delta_baseline()
        recorder.record(f"{mode}.sample{i}.generate.before", prompt_tokens=int(prompt_len))
        output = generate_by_mode(model, tokenizer, [prompt], cfg, mode, coordinator=coordinator, kv_manager=kv_manager)[0]
        peak_gpu_bytes = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        allocated_after = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        reserved_after = torch.cuda.memory_reserved() if torch.cuda.is_available() else 0
        cache_info = _cache_stats_dict(model, kv_manager, int(peak_gpu_bytes), int(baseline_kv_bytes)) if profile_memory else {}
        memory_sample = {
            "mode": mode,
            "sample_i": i,
            "excel_row": sample.row_index,
            "prompt_tokens": int(prompt_len),
            "max_new_tokens": int(cfg.generation.max_new_tokens),
            "total_tokens_estimate": total_tokens_estimate,
            "setup_allocated_bytes": int(setup_allocated),
            "setup_allocated": format_nbytes(int(setup_allocated)),
            "peak_gpu_bytes": int(peak_gpu_bytes),
            "peak_gpu": format_nbytes(int(peak_gpu_bytes)),
            "peak_over_setup_bytes": int(peak_gpu_bytes - setup_allocated),
            "peak_over_setup": format_nbytes(max(0, int(peak_gpu_bytes - setup_allocated))),
            "allocated_after_bytes": int(allocated_after),
            "allocated_after": format_nbytes(int(allocated_after)),
            "reserved_after_bytes": int(reserved_after),
            "reserved_after": format_nbytes(int(reserved_after)),
            **cache_info,
        }
        memory_samples.append(memory_sample)
        recorder.record(f"{mode}.sample{i}.generate.after", **memory_sample)
        recorder.current_sample_i = None
        pred_score, reason, parsed_json = parse_score_response(output, sample.full_score)
        row = {
            "mode": mode,
            "sample_i": i,
            "excel_row": sample.row_index,
            "subject": sample.subject,
            "file": sample.file,
            "question_type": sample.question_type,
            "full_score": sample.full_score,
            "human_score": sample.human_score,
            "pred_score": pred_score,
            "abs_error": None if pred_score is None else abs(pred_score - sample.human_score),
            "parsed_json": parsed_json,
            "reason": reason,
            "raw_output": output,
            "student_answer": sample.student_answer,
        }
        if profile_memory:
            row.update(memory_sample)
        rows.append(row)
        if i == 1 or i % 10 == 0 or i == len(samples):
            print(
                f"[{mode}] {i}/{len(samples)} "
                f"human={sample.human_score:g} pred={pred_score if pred_score is not None else 'PARSE_FAIL'}"
            )

    summary = summarize_predictions(rows)
    memory_profile = None
    if profile_memory:
        peak_values = [int(item["peak_gpu_bytes"]) for item in memory_samples]
        over_setup_values = [int(item["peak_over_setup_bytes"]) for item in memory_samples]
        cache_values = [int(item.get("cache_runtime_bytes") or 0) for item in memory_samples]
        baseline_kv_values = [int(item.get("baseline_kv_bytes") or 0) for item in memory_samples]
        memory_summary = {
            "mode": mode,
            "n_samples": len(memory_samples),
            "setup_allocated_bytes": int(setup_allocated),
            "setup_allocated": format_nbytes(int(setup_allocated)),
            "peak_gpu_max_bytes": max(peak_values) if peak_values else 0,
            "peak_gpu_max": format_nbytes(max(peak_values) if peak_values else 0),
            "peak_over_setup_max_bytes": max(over_setup_values) if over_setup_values else 0,
            "peak_over_setup_max": format_nbytes(max(over_setup_values) if over_setup_values else 0),
            "cache_runtime_max_bytes": max(cache_values) if cache_values else 0,
            "cache_runtime_max": format_nbytes(max(cache_values) if cache_values else 0),
            "baseline_kv_max_bytes": max(baseline_kv_values) if baseline_kv_values else 0,
            "baseline_kv_max": format_nbytes(max(baseline_kv_values) if baseline_kv_values else 0),
            "n_detail_hooks": n_detail_hooks,
            "n_memory_records": len(recorder.records),
        }
        summary.update(memory_summary)
        memory_profile = {
            "summary": memory_summary,
            "samples": memory_samples,
            "records": recorder.records,
        }
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows, summary, memory_profile


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate LLM scoring on local answer xlsx files.")
    parser.add_argument("config")
    parser.add_argument("--question-file", default="pingfen/题目信息.xlsx")
    parser.add_argument("--answer-dir", default="pingfen/评分数据")
    parser.add_argument("--file-id", default=None, help="Value to match in the question metadata '文件' column, e.g. 历史_161.")
    parser.add_argument("--answer-file", default=None, help="Optional direct answer xlsx path. Overrides --file-id/--answer-dir.")
    parser.add_argument("--modes", nargs="+", choices=_SUPPORTED_MODES, default=["baseline", "quant_only", "hawp_quant"])
    parser.add_argument("--output-dir", default="artifacts/scoring_eval/history_161")
    parser.add_argument("--rows", nargs="+", type=int, default=None, help="Specific Excel row numbers from the answer file, e.g. --rows 2 5 20.")
    parser.add_argument("--sample-size", type=int, default=None, help="Randomly sample N valid answer rows after --start.")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--limit", type=int, default=20, help="Sequential sample count after --start. Use 0 for full data.")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--max-question-chars", type=int, default=6000)
    parser.add_argument("--max-ref-chars", type=int, default=4000)
    parser.add_argument("--max-answer-chars", type=int, default=4000)
    parser.add_argument("--profile-memory", action="store_true", help="Record CUDA allocated/reserved/peak memory per mode and sample.")
    parser.add_argument("--profile-memory-detail", action="store_true", help="Also record model block and HAWP internal memory markers for the first N samples.")
    parser.add_argument("--profile-memory-detail-samples", type=int, default=1, help="Number of samples with detailed block/HAWP memory records.")
    parser.add_argument("--dry-run", action="store_true", help="Only load data and write prompt preview.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg.generation.max_new_tokens = int(args.max_new_tokens)
    device = _resolve_device(cfg.train.device)
    output_dir = Path(args.output_dir)
    rng = random.Random(args.seed)

    answer_file_arg = Path(args.answer_file) if args.answer_file else None
    question_row, resolved_answer_file, samples = load_samples(
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
    print(f"[scoring] config={args.config}")
    print(f"[scoring] model={cfg.model.model_id}")
    print(f"[scoring] question_file={args.question_file}")
    print(f"[scoring] file_id={samples[0].file}")
    print(f"[scoring] answer_file={resolved_answer_file}")
    print(f"[scoring] modes={args.modes}")
    print(f"[scoring] rows={args.rows}")
    print(f"[scoring] samples={len(samples)} start={args.start} limit={args.limit} sample_size={args.sample_size} seed={args.seed}")
    print(f"[scoring] output_dir={output_dir}")
    print("=" * 80)

    _write_json(output_dir / "question_metadata.json", question_row)
    _write_json(
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
        },
    )
    preview_prompt = build_prompt(
        samples[0],
        max_question_chars=args.max_question_chars,
        max_ref_chars=args.max_ref_chars,
        max_answer_chars=args.max_answer_chars,
    )
    (output_dir / "prompt_preview.txt").parent.mkdir(parents=True, exist_ok=True)
    (output_dir / "prompt_preview.txt").write_text(preview_prompt, encoding="utf-8")
    if args.dry_run:
        print(f"[scoring] dry run wrote {output_dir / 'prompt_preview.txt'}")
        return

    summary_rows: list[dict[str, Any]] = []
    all_predictions: list[dict[str, Any]] = []
    for mode in args.modes:
        predictions, summary, memory_profile = run_mode(
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
        _write_jsonl(mode_dir / "predictions.jsonl", predictions)
        _write_csv(mode_dir / "predictions.csv", predictions)
        _write_json(mode_dir / "summary.json", summary)
        if memory_profile is not None:
            _write_json(mode_dir / "memory_profile.json", memory_profile)
            _write_csv(mode_dir / "memory_samples.csv", memory_profile["samples"])
        summary_rows.append({"mode": mode, **summary})
        all_predictions.extend(predictions)

    _write_csv(output_dir / "summary.csv", summary_rows)
    _write_json(output_dir / "summary.json", summary_rows)
    _write_jsonl(output_dir / "predictions_all.jsonl", all_predictions)

    print("\n[scoring] summary")
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
    print(f"[scoring] wrote {output_dir}")


if __name__ == "__main__":
    main()
