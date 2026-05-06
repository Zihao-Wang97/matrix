from __future__ import annotations

import json
import re
import string
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable


LONGBENCH_E_TASKS = [
    "qasper",
    "multifieldqa_en",
    "hotpotqa",
    "2wikimqa",
    "gov_report",
    "multi_news",
    "trec",
    "triviaqa",
    "samsum",
    "passage_count",
    "passage_retrieval_en",
    "lcc",
    "repobench-p",
]

LONGBENCH_E_CATEGORIES = {
    "SingleQA": ["qasper", "multifieldqa_en"],
    "MultiQA": ["hotpotqa", "2wikimqa"],
    "Summarization": ["gov_report", "multi_news"],
    "Few-shot": ["trec", "triviaqa", "samsum"],
    "Synthetic": ["passage_count", "passage_retrieval_en"],
    "Code": ["lcc", "repobench-p"],
}

TASK_PROMPTS = {
    "qasper": (
        "You are given a scientific article and a question. Answer the question as concisely as you can, "
        "using a single phrase or sentence if possible. If the question cannot be answered based on the "
        "information in the article, write \"unanswerable\". If the question is a yes/no question, answer "
        "\"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\n"
        "Article: {context}\n\n"
        "Answer the question based on the above article as concisely as you can, using a single phrase or "
        "sentence if possible. If the question cannot be answered based on the information in the article, "
        "write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or "
        "\"unanswerable\". Do not provide any explanation.\n\n"
        "Question: {input}\n\nAnswer:"
    ),
    "multifieldqa_en": (
        "Read the following text and answer briefly.\n\n{context}\n\n"
        "Now, answer the following question based on the above text, only give me the answer and do not "
        "output any other words.\n\nQuestion: {input}\nAnswer:"
    ),
    "hotpotqa": (
        "Answer the question based on the given passages. Only give me the answer and do not output any "
        "other words.\n\nThe following are given passages.\n{context}\n\n"
        "Answer the question based on the given passages. Only give me the answer and do not output any "
        "other words.\n\nQuestion: {input}\nAnswer:"
    ),
    "2wikimqa": (
        "Answer the question based on the given passages. Only give me the answer and do not output any "
        "other words.\n\nThe following are given passages.\n{context}\n\n"
        "Answer the question based on the given passages. Only give me the answer and do not output any "
        "other words.\n\nQuestion: {input}\nAnswer:"
    ),
    "gov_report": (
        "You are given a report by a government agency. Write a one-page summary of the report.\n\n"
        "Report:\n{context}\n\nNow, write a one-page summary of the report.\n\nSummary:"
    ),
    "multi_news": (
        "You are given several news passages. Write a one-page summary of all news. \n\n"
        "News:\n{context}\n\nNow, write a one-page summary of all the news.\n\nSummary:"
    ),
    "trec": (
        "Please determine the type of the question below. Here are some examples of questions.\n\n"
        "{context}\n{input}"
    ),
    "triviaqa": (
        "Answer the question based on the given passage. Only give me the answer and do not output any "
        "other words. The following are some examples.\n\n{context}\n\n{input}"
    ),
    "samsum": (
        "Summarize the dialogue into a few short sentences. The following are some examples.\n\n"
        "{context}\n\n{input}"
    ),
    "passage_count": (
        "There are some paragraphs below sourced from Wikipedia. Some of them may be duplicates. Please "
        "carefully read these paragraphs and determine how many unique paragraphs there are after removing "
        "duplicates. In other words, how many non-repeating paragraphs are there in total?\n\n{context}\n\n"
        "Please enter the final count of unique paragraphs after removing duplicates. The output format "
        "should only contain the number, such as 1, 2, 3, and so on.\n\nThe final answer is: "
    ),
    "passage_retrieval_en": (
        "Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine which paragraph "
        "the abstract is from.\n\n{context}\n\nThe following is an abstract.\n\n{input}\n\n"
        "Please enter the number of the paragraph that the abstract is from. The answer format must be like "
        "\"Paragraph 1\", \"Paragraph 2\", etc.\n\nThe answer is: "
    ),
    "lcc": "Please complete the code given below.\n{context}Next line of code:\n",
    "repobench-p": "Please complete the code given below. \n{context}{input}Next line of code:\n",
}

TASK_MAX_NEW_TOKENS = {
    "qasper": 128,
    "multifieldqa_en": 64,
    "hotpotqa": 32,
    "2wikimqa": 32,
    "gov_report": 512,
    "multi_news": 512,
    "trec": 64,
    "triviaqa": 32,
    "samsum": 128,
    "passage_count": 32,
    "passage_retrieval_en": 32,
    "lcc": 64,
    "repobench-p": 64,
}

NO_CHAT_TEMPLATE_TASKS = {"trec", "triviaqa", "samsum", "lcc", "repobench-p"}


@dataclass
class LongBenchSample:
    task: str
    index: int
    prompt: str
    answers: list[str]
    all_classes: list[str]
    length: int | None
    max_new_tokens: int
    raw: dict[str, Any]


def resolve_longbench_tasks(tasks: list[str] | None = None) -> list[str]:
    if not tasks:
        return list(LONGBENCH_E_TASKS)

    resolved = []
    for task in tasks:
        base = task[:-2] if task.endswith("_e") else task
        if base not in LONGBENCH_E_TASKS:
            raise ValueError(f"Unsupported LongBench-E task: {task}")
        resolved.append(base)
    return resolved


def find_task_file(data_dir: str | Path, task: str, longbench_e: bool = True) -> Path:
    root = Path(data_dir)
    suffixes = [f"{task}_e.jsonl", f"{task}.jsonl"] if longbench_e else [f"{task}.jsonl"]
    candidates: list[Path] = []
    for name in suffixes:
        candidates.extend([
            root / name,
            root / "data" / name,
            root / "LongBench" / "data" / name,
        ])
        candidates.extend(root.rglob(name) if root.exists() else [])

    seen = set()
    unique_candidates = []
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(candidate)

    for candidate in unique_candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Cannot find LongBench-E file for task '{task}' under {root}. "
        f"Expected names like {suffixes[0]}."
    )


def load_jsonl(path: str | Path, max_samples: int | None = None) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if max_samples is not None and len(rows) >= max_samples:
                break
    return rows


def build_prompt(task: str, sample: dict[str, Any], tokenizer=None, max_input_tokens: int | None = None) -> str:
    prompt = TASK_PROMPTS[task].format(**sample)
    if tokenizer is None or max_input_tokens is None or max_input_tokens <= 0:
        return prompt

    tokenized = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
    if len(tokenized) <= max_input_tokens:
        return prompt

    half = max_input_tokens // 2
    left = tokenizer.decode(tokenized[:half], skip_special_tokens=True)
    right = tokenizer.decode(tokenized[-half:], skip_special_tokens=True)
    return left + right


def maybe_apply_chat_template(tokenizer, prompt: str, task: str, mode: str = "auto") -> str:
    if mode == "never" or task in NO_CHAT_TEMPLATE_TASKS:
        return prompt
    if mode not in {"auto", "always"}:
        raise ValueError(f"Unsupported chat template mode: {mode}")
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt


def iter_longbench_samples(
    data_dir: str | Path,
    tasks: list[str] | None = None,
    tokenizer=None,
    max_input_tokens: int | None = None,
    max_samples_per_task: int | None = None,
    chat_template: str = "auto",
) -> list[LongBenchSample]:
    samples: list[LongBenchSample] = []
    for task in resolve_longbench_tasks(tasks):
        path = find_task_file(data_dir, task, longbench_e=True)
        rows = load_jsonl(path, max_samples=max_samples_per_task)
        for index, row in enumerate(rows):
            prompt = build_prompt(task, row, tokenizer=tokenizer, max_input_tokens=max_input_tokens)
            prompt = maybe_apply_chat_template(tokenizer, prompt, task, mode=chat_template)
            answers = row.get("answers") or []
            if isinstance(answers, str):
                answers = [answers]
            all_classes = row.get("all_classes") or []
            if isinstance(all_classes, str):
                all_classes = [all_classes]
            samples.append(LongBenchSample(
                task=task,
                index=index,
                prompt=prompt,
                answers=[str(a) for a in answers],
                all_classes=[str(c) for c in all_classes],
                length=row.get("length"),
                max_new_tokens=TASK_MAX_NEW_TOKENS[task],
                raw=row,
            ))
    return samples


def normalize_answer(text: str) -> str:
    def remove_articles(value: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def remove_punc(value: str) -> str:
        return "".join(ch for ch in value if ch not in set(string.punctuation))

    return " ".join(remove_articles(remove_punc(text.lower())).split())


def _f1_score(prediction_tokens: list[str], ground_truth_tokens: list[str]) -> float:
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0 or not prediction_tokens or not ground_truth_tokens:
        return 0.0
    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    return (2 * precision * recall) / (precision + recall)


def qa_f1_score(prediction: str, ground_truth: str, **_: Any) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    return _f1_score(prediction_tokens, ground_truth_tokens)


def _lcs_len(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for token_a in a:
        cur = [0]
        for j, token_b in enumerate(b, start=1):
            if token_a == token_b:
                cur.append(prev[j - 1] + 1)
            else:
                cur.append(max(prev[j], cur[-1]))
        prev = cur
    return prev[-1]


def rouge_l_score(prediction: str, ground_truth: str, **_: Any) -> float:
    pred_tokens = prediction.split()
    gt_tokens = ground_truth.split()
    if not pred_tokens or not gt_tokens:
        return 0.0
    lcs = _lcs_len(pred_tokens, gt_tokens)
    if lcs == 0:
        return 0.0
    precision = lcs / len(pred_tokens)
    recall = lcs / len(gt_tokens)
    return (2 * precision * recall) / (precision + recall)


def classification_score(prediction: str, ground_truth: str, **kwargs: Any) -> float:
    all_classes = kwargs.get("all_classes") or []
    matches = [class_name for class_name in all_classes if class_name in prediction]
    matches = [
        match
        for match in matches
        if not (match in ground_truth and match != ground_truth)
    ]
    if ground_truth in matches and matches:
        return 1.0 / len(matches)
    return 0.0


def count_score(prediction: str, ground_truth: str, **_: Any) -> float:
    numbers = re.findall(r"\d+", prediction)
    if not numbers:
        return 0.0
    right = sum(1 for number in numbers if str(number) == str(ground_truth))
    return right / len(numbers)


def retrieval_score(prediction: str, ground_truth: str, **_: Any) -> float:
    matches = re.findall(r"Paragraph (\d+)", ground_truth)
    if not matches:
        return 0.0
    ground_truth_id = matches[0]
    numbers = re.findall(r"\d+", prediction)
    if not numbers:
        return 0.0
    right = sum(1 for number in numbers if str(number) == str(ground_truth_id))
    return right / len(numbers)


def code_sim_score(prediction: str, ground_truth: str, **_: Any) -> float:
    cleaned = ""
    for line in prediction.lstrip("\n").split("\n"):
        if "`" not in line and "#" not in line and "//" not in line:
            cleaned = line
            break
    return SequenceMatcher(None, cleaned, ground_truth).ratio()


TASK_METRICS: dict[str, Callable[..., float]] = {
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "gov_report": rouge_l_score,
    "multi_news": rouge_l_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_l_score,
    "passage_count": count_score,
    "passage_retrieval_en": retrieval_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}


def postprocess_prediction(task: str, prediction: str) -> str:
    if task in {"trec", "triviaqa", "samsum"}:
        return prediction.lstrip("\n").split("\n")[0]
    return prediction


def score_prediction(task: str, prediction: str, answers: list[str], all_classes: list[str]) -> float:
    prediction = postprocess_prediction(task, prediction)
    metric = TASK_METRICS[task]
    best = 0.0
    for answer in answers:
        best = max(best, metric(prediction, answer, all_classes=all_classes))
    return best


def length_bin(length: int | None) -> str:
    if length is None:
        return "unknown"
    if length < 4000:
        return "0-4k"
    if length < 8000:
        return "4-8k"
    return "8k+"


def summarize_longbench(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    by_task: dict[str, list[dict[str, Any]]] = {}
    for row in predictions:
        by_task.setdefault(row["task"], []).append(row)

    task_scores: dict[str, dict[str, Any]] = {}
    for task, rows in sorted(by_task.items()):
        scores = [float(row["score"]) for row in rows]
        bins: dict[str, list[float]] = {"0-4k": [], "4-8k": [], "8k+": [], "unknown": []}
        for row in rows:
            bins.setdefault(row.get("length_bin", "unknown"), []).append(float(row["score"]))
        task_scores[task] = {
            "score": round(100.0 * sum(scores) / len(scores), 4) if scores else None,
            "n": len(rows),
            "by_length": {
                key: round(100.0 * sum(values) / len(values), 4) if values else None
                for key, values in bins.items()
            },
        }

    category_scores: dict[str, dict[str, Any]] = {}
    for category, tasks in LONGBENCH_E_CATEGORIES.items():
        values = [
            task_scores[task]["score"]
            for task in tasks
            if task in task_scores and task_scores[task]["score"] is not None
        ]
        category_scores[category] = {
            "score": round(sum(values) / len(values), 4) if values else None,
            "tasks": tasks,
        }

    category_values = [
        item["score"]
        for item in category_scores.values()
        if item["score"] is not None
    ]
    task_values = [
        item["score"]
        for item in task_scores.values()
        if item["score"] is not None
    ]

    return {
        "task_scores": task_scores,
        "category_scores": category_scores,
        "average": round(sum(category_values) / len(category_values), 4) if category_values else None,
        "task_average": round(sum(task_values) / len(task_values), 4) if task_values else None,
        "n": len(predictions),
    }
