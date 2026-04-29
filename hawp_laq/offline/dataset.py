from __future__ import annotations

from pathlib import Path
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


def get_calib_dataloader(
    tokenizer: AutoTokenizer,
    nsamples: int = 8,
    seq_len: int = 128,
    dataset_name: str = "wikitext2",
    seed: int = 42,
    data_root: str | Path = Path("./data"),
) -> DataLoader:
    tokens = _load_dataset_tokens(tokenizer, dataset_name, seed, data_root)
    tokens = tokens[: nsamples * seq_len + 1]
    samples = []
    for i in range(0, len(tokens) - seq_len, seq_len):
        input_ids = torch.tensor(tokens[i : i + seq_len], dtype=torch.long)
        samples.append(input_ids)
        if len(samples) >= nsamples:
            break
    if dataset_name == "wikitext2" and len(samples) < nsamples:
        raise RuntimeError(
            f"wikitext2 calibration data is too short: built {len(samples)} samples "
            f"but requested {nsamples} with seq_len={seq_len}. "
            f"Check that {Path(data_root) / 'wikitext2_train.txt'} exists and is the full file."
        )
    return DataLoader(samples, batch_size=1, shuffle=False)


def _load_dataset_tokens(
    tokenizer: AutoTokenizer,
    dataset_name: str,
    seed: int,
    data_root: str | Path = Path("./data"),
) -> list[int]:
    if dataset_name == "wikitext2":
        local_txt = Path(data_root) / "wikitext2_train.txt"
        if local_txt.exists():
            text = local_txt.read_text(encoding="utf-8")
            print(f"[data] loaded local wikitext2 train: {local_txt} ({local_txt.stat().st_size} bytes)")
        else:
            try:
                from datasets import load_dataset

                ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
                text = "\n\n".join(ds["text"])
                print("[data] loaded wikitext2 train via datasets")
            except Exception:
                raise RuntimeError(
                    f"Cannot load wikitext2 train data. Expected local file at "
                    f"{local_txt}, and online datasets loading failed."
                )
    else:
        text = _fallback_text()

    enc = tokenizer(text, return_tensors="pt")
    return enc["input_ids"][0].tolist()


def _fallback_text() -> str:
    return (
        "The quick brown fox jumps over the lazy dog. "
        * 200
    )
