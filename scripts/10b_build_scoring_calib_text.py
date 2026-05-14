#!/usr/bin/env python
"""Build subject-specific scoring calibration text from local xlsx files."""

from __future__ import annotations

import argparse
import math
import random
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET


_NS = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


@dataclass
class ScoringTextSample:
    subject: str
    file_id: str
    question_type: str
    question: str
    reference_answer: str
    full_score: float
    human_score: float
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
        out.append({header[i]: row[i].strip() if i < len(row) else "" for i in range(len(header))})
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


def _clip(text: str, max_chars: int) -> str:
    text = str(text or "").strip()
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[已截断]"


def _format_score(value: float) -> str:
    if abs(value - round(value)) < 1e-6:
        return str(int(round(value)))
    return f"{value:.2f}".rstrip("0").rstrip(".")


def build_prompt_text(sample: ScoringTextSample, *, max_question_chars: int, max_ref_chars: int, max_answer_chars: int) -> str:
    role = (
        f"{sample.subject}{sample.question_type}的专门评分员"
        if sample.question_type
        else f"{sample.subject}科目的专门评分员"
    )
    return f"""你是{role}。请根据题目、参考答案和满分，对考生答案打分。
评分要求：
1. 分数必须在 0 到 {sample.full_score:g} 分之间。
2. 严格依据参考答案和考生答案进行判断，不要因为表达方式不同就扣分。
3. 如果考生答案明显无关、空白、重复粘贴或无法判断，应给低分或 0 分。
4. 只输出一个 JSON，不要输出 Markdown，不要输出额外解释。
JSON 格式：{{"score": 数字, "reason": "简短评分理由"}}

科目：{sample.subject}
文件：{sample.file_id}
题目类型：{sample.question_type}
满分：{sample.full_score:g}

题目：
{_clip(sample.question, max_question_chars)}

参考答案：
{_clip(sample.reference_answer, max_ref_chars)}

考生答案：
{_clip(sample.student_answer, max_answer_chars)}

标准输出：
{{"score": {_format_score(sample.human_score)}}}
"""


def collect_samples(
    question_file: Path,
    answer_dir: Path,
    *,
    subject: str,
    max_rows_per_file: int | None,
) -> list[ScoringTextSample]:
    question_rows = _row_dicts(read_xlsx_first_sheet(question_file))
    samples: list[ScoringTextSample] = []
    for question_row in question_rows:
        row_subject = _first_present(question_row, ["科目", "subject"], fallback_idx=0)
        file_id = _first_present(question_row, ["文件", "file"], fallback_idx=1)
        if not file_id:
            continue
        if subject and subject not in row_subject and not file_id.startswith(subject):
            continue
        answer_file = answer_dir / f"{file_id}.xlsx"
        if not answer_file.exists():
            print(f"[build-scoring-calib] skip missing answer file: {answer_file}")
            continue

        question_type = _first_present(question_row, ["题目类型", "question_type"], fallback_idx=2)
        question = _first_present(question_row, ["题目", "question"], fallback_idx=3)
        reference_answer = _first_present(question_row, ["参考答案", "reference_answer"], fallback_idx=4)
        full_score = _to_float(_first_present(question_row, ["满分值", "满分", "full_score"], fallback_idx=5), default=0.0)

        answer_rows = _row_dicts(read_xlsx_first_sheet(answer_file))
        n_from_file = 0
        for answer_row in answer_rows:
            human_score = _to_float(_first_present(answer_row, ["最终分", "真实评分", "score"], fallback_idx=0))
            student_answer = _first_present(answer_row, ["识别文本", "答案", "answer"], fallback_idx=3)
            if math.isnan(human_score) or not student_answer.strip():
                continue
            samples.append(
                ScoringTextSample(
                    subject=row_subject or subject,
                    file_id=file_id,
                    question_type=question_type,
                    question=question,
                    reference_answer=reference_answer,
                    full_score=full_score,
                    human_score=human_score,
                    student_answer=student_answer,
                )
            )
            n_from_file += 1
            if max_rows_per_file is not None and n_from_file >= max_rows_per_file:
                break
        print(f"[build-scoring-calib] {file_id}: {n_from_file} samples")
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Build scoring calibration text for projector training.")
    parser.add_argument("--question-file", default="pingfen/题目信息.xlsx")
    parser.add_argument("--answer-dir", default="pingfen/评分数据")
    parser.add_argument("--subject", default="历史")
    parser.add_argument("--output", default="data/scoring_history_train.txt")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-rows-per-file", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-question-chars", type=int, default=6000)
    parser.add_argument("--max-ref-chars", type=int, default=4000)
    parser.add_argument("--max-answer-chars", type=int, default=4000)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    samples = collect_samples(
        Path(args.question_file),
        Path(args.answer_dir),
        subject=args.subject,
        max_rows_per_file=args.max_rows_per_file,
    )
    if args.max_samples is not None and args.max_samples > 0 and len(samples) > args.max_samples:
        samples = rng.sample(samples, args.max_samples)
    rng.shuffle(samples)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    sep = "\n\n" + "=" * 80 + "\n\n"
    text = sep.join(
        build_prompt_text(
            sample,
            max_question_chars=args.max_question_chars,
            max_ref_chars=args.max_ref_chars,
            max_answer_chars=args.max_answer_chars,
        )
        for sample in samples
    )
    output.write_text(text + "\n", encoding="utf-8")
    print(f"[build-scoring-calib] wrote {len(samples)} samples to {output}")


if __name__ == "__main__":
    main()
