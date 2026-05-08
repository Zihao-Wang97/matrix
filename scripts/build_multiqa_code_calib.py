from __future__ import annotations

import argparse
import json
import random
import re
import shutil
from pathlib import Path
from typing import Any, Callable, Iterable


def _clean(text: Any, max_chars: int | None = None) -> str:
    if text is None:
        return ""
    if isinstance(text, (list, tuple)):
        text = " ".join(_clean(x) for x in text)
    else:
        text = str(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    text = text.strip()
    if max_chars is not None and len(text) > max_chars:
        text = text[:max_chars].rstrip()
    return text


def _answer_to_text(answer: Any) -> str:
    if isinstance(answer, dict):
        for key in ("value", "normalized_value", "text"):
            if answer.get(key):
                return _clean(answer[key])
        aliases = answer.get("aliases") or answer.get("normalized_aliases")
        if aliases:
            return _clean(aliases[0] if isinstance(aliases, list) else aliases)
    return _clean(answer)


def _context_to_text(context: Any, max_chars: int) -> str:
    parts: list[str] = []
    if isinstance(context, str):
        stripped = context.strip()
        if stripped.startswith("[") or stripped.startswith("{"):
            try:
                context = json.loads(stripped)
            except json.JSONDecodeError:
                pass
    if isinstance(context, dict):
        titles = context.get("title") or context.get("titles") or []
        sentences = context.get("sentences") or context.get("text") or context.get("paragraphs") or []
        if isinstance(titles, list) and isinstance(sentences, list):
            for title, sent in zip(titles, sentences):
                sent_text = _clean(sent)
                if sent_text:
                    parts.append(f"{_clean(title)}: {sent_text}")
        else:
            parts.append(_clean(context))
    elif isinstance(context, list):
        for item in context:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                parts.append(f"{_clean(item[0])}: {_clean(item[1])}")
            elif isinstance(item, dict):
                title = item.get("title") or item.get("name") or ""
                text = item.get("sentences") or item.get("text") or item.get("paragraph") or item
                piece = _clean(text)
                if piece:
                    parts.append(f"{_clean(title)}: {piece}" if title else piece)
            else:
                piece = _clean(item)
                if piece:
                    parts.append(piece)
    else:
        parts.append(_clean(context))
    return _clean("\n".join(parts), max_chars=max_chars)


def _format_multihop(kind: str, row: dict[str, Any], max_context_chars: int) -> str:
    question = _clean(row.get("question"))
    answer = _answer_to_text(row.get("answer"))
    context = _context_to_text(row.get("context") or row.get("contexts") or row.get("paragraphs"), max_context_chars)
    if not question or not context:
        return ""
    return (
        f"[{kind}]\n"
        f"Question:\n{question}\n\n"
        f"Context:\n{context}\n\n"
        f"Answer:\n{answer}\n"
    )


def _format_trivia(row: dict[str, Any]) -> str:
    question = _clean(row.get("question"))
    answer = _answer_to_text(row.get("answer"))
    if not question:
        return ""
    return f"[TriviaQA]\nQuestion:\n{question}\n\nAnswer:\n{answer}\n"


def _format_code(row: dict[str, Any], max_chars: int) -> str:
    content = _clean(row.get("content") or row.get("text") or row.get("code"), max_chars=max_chars)
    lang = _clean(row.get("lang") or row.get("language"))
    if len(content) < 80:
        return ""
    header = f"[Code: {lang}]" if lang else "[Code]"
    return f"{header}\n{content}\n"


def _iter_dataset(name: str, *args: Any, streaming: bool = True, trust_remote_code: bool = False, **kwargs: Any) -> Iterable[dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset(name, *args, split="train", streaming=streaming, trust_remote_code=trust_remote_code, **kwargs)
    return ds


def _iter_2wiki_parquet(split: str, streaming: bool = True) -> Iterable[dict[str, Any]]:
    from datasets import load_dataset

    if split not in {"train", "dev", "test"}:
        raise ValueError(f"Unsupported 2Wiki split: {split!r}")
    url = f"https://huggingface.co/datasets/xanhho/2WikiMultihopQA/resolve/main/{split}.parquet"
    return load_dataset("parquet", data_files={split: url}, split=split, streaming=streaming)


def _shuffle(ds: Iterable[dict[str, Any]], seed: int, buffer_size: int) -> Iterable[dict[str, Any]]:
    if hasattr(ds, "shuffle"):
        try:
            return ds.shuffle(seed=seed, buffer_size=buffer_size)
        except TypeError:
            return ds.shuffle(seed=seed)
    return ds


def _take(
    label: str,
    ds: Iterable[dict[str, Any]],
    count: int,
    formatter: Callable[[dict[str, Any]], str],
    min_chars: int,
) -> list[str]:
    samples: list[str] = []
    seen = 0
    for row in ds:
        seen += 1
        text = formatter(row)
        if len(text) >= min_chars:
            samples.append(text)
        if len(samples) >= count:
            break
        if seen % 1000 == 0:
            print(f"[{label}] kept {len(samples)}/{count} after scanning {seen}")
    print(f"[{label}] kept {len(samples)}/{count}")
    return samples


def _wiki_chunks(path: Path, count: int, max_chars: int, seed: int) -> list[str]:
    text = path.read_text(encoding="utf-8")
    chunks = [_clean(x, max_chars=max_chars) for x in re.split(r"\n\s*\n", text)]
    chunks = [x for x in chunks if len(x) >= 80]
    rng = random.Random(seed)
    rng.shuffle(chunks)
    return [f"[WikiText]\n{x}\n" for x in chunks[:count]]


def _split_counts(total: int, n: int) -> list[int]:
    base = total // n
    counts = [base] * n
    for i in range(total - base * n):
        counts[i] += 1
    return counts


def build(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_samples: list[str] = []

    print("[download] HotpotQA hotpotqa/hotpot_qa distractor train")
    hotpot = _shuffle(
        _iter_dataset("hotpotqa/hotpot_qa", "distractor", streaming=True),
        args.seed,
        args.shuffle_buffer,
    )
    all_samples.extend(
        _take(
            "hotpotqa",
            hotpot,
            args.n_hotpot,
            lambda row: _format_multihop("HotpotQA", row, args.max_context_chars),
            args.min_chars,
        )
    )

    print(f"[download] 2WikiMultihopQA parquet split={args.two_wiki_split}")
    wiki2 = _shuffle(
        _iter_2wiki_parquet(args.two_wiki_split, streaming=True),
        args.seed + 1,
        args.shuffle_buffer,
    )
    all_samples.extend(
        _take(
            "2wiki",
            wiki2,
            args.n_2wiki,
            lambda row: _format_multihop("2WikiMultihopQA", row, args.max_context_chars),
            args.min_chars,
        )
    )

    print("[download] TriviaQA mandarjoshi/trivia_qa unfiltered.nocontext train")
    trivia = _shuffle(
        _iter_dataset("mandarjoshi/trivia_qa", "unfiltered.nocontext", streaming=True),
        args.seed + 2,
        args.shuffle_buffer,
    )
    all_samples.extend(_take("triviaqa", trivia, args.n_trivia, _format_trivia, 20))

    code_languages = [x.strip() for x in args.code_languages.split(",") if x.strip()]
    code_counts = _split_counts(args.n_code, len(code_languages))
    for lang, count in zip(code_languages, code_counts):
        if count <= 0:
            continue
        print(f"[download] The Stack Smol XL bigcode/the-stack-smol-xl data/{lang} train")
        try:
            code_ds = _shuffle(
                _iter_dataset("bigcode/the-stack-smol-xl", streaming=True, data_dir=f"data/{lang}"),
                args.seed + 10 + len(all_samples),
                args.shuffle_buffer,
            )
            all_samples.extend(
                _take(
                    f"code:{lang}",
                    code_ds,
                    count,
                    lambda row: _format_code(row, args.max_code_chars),
                    args.min_chars,
                )
            )
        except Exception as exc:
            print(f"[warn] skipped code language {lang!r}: {exc}")

    wiki_path = Path(args.wikitext)
    if wiki_path.exists() and args.n_wiki > 0:
        print(f"[local] WikiText from {wiki_path}")
        all_samples.extend(_wiki_chunks(wiki_path, args.n_wiki, args.max_context_chars, args.seed + 3))
    elif args.n_wiki > 0:
        print(f"[warn] local wikitext file not found: {wiki_path}")

    rng.shuffle(all_samples)
    text = "\n\n".join(all_samples).strip() + "\n"
    out_path.write_text(text, encoding="utf-8", newline="\n")

    meta = {
        "output": str(out_path),
        "bytes": out_path.stat().st_size,
        "samples_total": len(all_samples),
        "requested": {
            "hotpotqa": args.n_hotpot,
            "2wiki": args.n_2wiki,
            "code": args.n_code,
            "triviaqa": args.n_trivia,
            "wikitext": args.n_wiki,
        },
        "code_languages": code_languages,
        "seed": args.seed,
    }
    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[save] {out_path} ({out_path.stat().st_size / 1024 / 1024:.2f} MB)")
    print(f"[save] {meta_path}")

    if args.replace_wikitext2:
        target = Path("data/wikitext2_train.txt")
        backup = Path("data/wikitext2_train.backup.txt")
        if target.exists() and not backup.exists():
            shutil.copyfile(target, backup)
            print(f"[backup] {target} -> {backup}")
        shutil.copyfile(out_path, target)
        print(f"[replace] {target} now points to the mixed calibration text")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a mixed MultiQA + code calibration text file.")
    parser.add_argument("--output", default="data/mixed/multiqa_code_calib_train.txt")
    parser.add_argument("--wikitext", default="data/wikitext2_train.txt")
    parser.add_argument("--replace-wikitext2", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle-buffer", type=int, default=10000)
    parser.add_argument("--n-hotpot", type=int, default=3500)
    parser.add_argument("--n-2wiki", type=int, default=3000)
    parser.add_argument("--two-wiki-split", choices=["train", "dev", "test"], default="train")
    parser.add_argument("--n-code", type=int, default=2000)
    parser.add_argument("--n-trivia", type=int, default=1000)
    parser.add_argument("--n-wiki", type=int, default=500)
    parser.add_argument("--code-languages", default="python,javascript,java,c++")
    parser.add_argument("--max-context-chars", type=int, default=6000)
    parser.add_argument("--max-code-chars", type=int, default=6000)
    parser.add_argument("--min-chars", type=int, default=80)
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
