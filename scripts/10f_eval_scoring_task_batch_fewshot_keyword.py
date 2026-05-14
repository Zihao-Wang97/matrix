#!/usr/bin/env python
"""Few-shot batched scoring with keyword-aware anchor selection.

This script reuses ``10e_eval_scoring_task_batch_fewshot.py`` and replaces only
the few-shot example selector. For each score-ratio band it first samples up to
20 usable candidates, then chooses examples whose keyword coverage is closest
to the band's expected coverage while keeping answer length typical.
"""

from __future__ import annotations

import importlib.util
import random
import re
import sys
from pathlib import Path
from typing import Any


def _load_10e_module():
    script_path = Path(__file__).resolve().parent / "10e_eval_scoring_task_batch_fewshot.py"
    spec = importlib.util.spec_from_file_location("scoring_task_batch_fewshot_eval", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


base = _load_10e_module()

_CANDIDATES_PER_BAND = 20
_STOPWORDS = {
    "参考答案",
    "特点",
    "材料",
    "结合",
    "所学",
    "概括",
    "根据",
    "每点",
    "满分",
    "任答",
    "酌情",
    "给分",
    "其他",
    "答案",
    "言之有理",
}


def _extract_keywords(text: str, *, max_keywords: int = 48) -> list[str]:
    text = str(text or "")
    text = re.sub(r"[，。；：、（）()\[\]【】《》“”\"'！？!?|#\s]+", "\n", text)
    raw_terms = re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z][A-Za-z0-9_-]{2,}|\d+(?:\.\d+)?%?", text)
    terms: list[str] = []
    seen: set[str] = set()
    for term in raw_terms:
        term = term.strip()
        if not term or term in _STOPWORDS:
            continue
        if len(term) > 16:
            # Long Chinese clauses are too brittle as exact-match anchors.
            chunks = re.findall(r"[\u4e00-\u9fff]{2,6}|[A-Za-z][A-Za-z0-9_-]{2,}|\d+(?:\.\d+)?%?", term)
        else:
            chunks = [term]
        for chunk in chunks:
            if chunk in _STOPWORDS or chunk in seen:
                continue
            seen.add(chunk)
            terms.append(chunk)
            if len(terms) >= max_keywords:
                return terms
    return terms


def _keyword_coverage(answer: str, keywords: list[str]) -> float:
    if not keywords:
        return 0.0
    answer = str(answer or "")
    hits = sum(1 for kw in keywords if kw and kw in answer)
    return hits / len(keywords)


def _select_by_keyword_coverage(
    candidates: list[Any],
    *,
    keywords: list[str],
    target_ratio: float,
    per_band: int,
) -> list[Any]:
    if not candidates or per_band <= 0:
        return []
    lengths = sorted(len(str(s.student_answer).strip()) for s in candidates)
    median_len = lengths[len(lengths) // 2]

    scored = []
    for sample in candidates:
        text = str(sample.student_answer or "")
        coverage = _keyword_coverage(text, keywords)
        length = len(text.strip())
        # Primary: coverage should match the score band. Secondary: length typical.
        scored.append(
            (
                abs(coverage - target_ratio),
                abs(length - median_len) / max(1, median_len),
                int(sample.row_index),
                coverage,
                length,
                sample,
            )
        )
    scored.sort(key=lambda item: item[:3])
    return [item[-1] for item in scored[:per_band]]


def select_score_band_examples_keyword(
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
    reference_answer = str(pool[0].reference_answer if pool else "")
    keywords = _extract_keywords(reference_answer)

    for band, lo, hi in base._SCORE_BANDS:
        target_ratio = (lo + hi) / 2.0
        raw_band_items = [
            s for s in candidates
            if int(s.row_index) not in used_rows
            and base._in_band(float(s.human_score), float(s.full_score), lo, hi)
        ]
        usable = [
            s for s in raw_band_items
            if base._looks_like_usable_example_answer(
                str(s.student_answer),
                min_chars=min_example_chars,
                max_chars=max_example_candidate_chars,
                min_informative_ratio=min_informative_ratio,
            )
        ]
        shuffled = list(usable)
        rng.shuffle(shuffled)
        sampled_candidates = shuffled[:_CANDIDATES_PER_BAND]
        picked = _select_by_keyword_coverage(
            sampled_candidates,
            keywords=keywords,
            target_ratio=target_ratio,
            per_band=max(0, per_band),
        )
        selected.extend(picked)
        used_rows.update(int(s.row_index) for s in picked)
        report.append(
            {
                "band": band,
                "range": f"[{lo:g}, {hi:g}] * full_score" if abs(lo) < 1e-12 else f"({lo:g}, {hi:g}] * full_score",
                "target_keyword_coverage": target_ratio,
                "reference_keyword_n": len(keywords),
                "reference_keywords": keywords,
                "raw_candidate_n": len(raw_band_items),
                "candidate_n": len(usable),
                "filtered_out_n": len(raw_band_items) - len(usable),
                "sampled_candidate_n": len(sampled_candidates),
                "candidate_sample_limit": _CANDIDATES_PER_BAND,
                "picked_n": len(picked),
                "excel_rows": [int(s.row_index) for s in picked],
                "scores": [float(s.human_score) for s in picked],
                "score_ratios": [
                    None if float(s.full_score) <= 0 else float(s.human_score) / float(s.full_score)
                    for s in picked
                ],
                "keyword_coverages": [_keyword_coverage(str(s.student_answer), keywords) for s in picked],
                "answer_chars": [len(str(s.student_answer).strip()) for s in picked],
            }
        )

    return selected, report


base.select_score_band_examples = select_score_band_examples_keyword


if __name__ == "__main__":
    base.main()
