#!/usr/bin/env python
"""Validate the prefill SDPA/FlashAttention memory baseline.

Runs (or summarizes existing outputs from) ``06b_profile_peak_segments.py`` for
baseline and hawp_quant across long sequence lengths, then reports:

  - prefill peak increase for each mode
  - final peak GPU for each mode
  - whether hawp_quant prefill <= baseline prefill
  - whether hawp_quant final peak < baseline final peak
  - whether the hawp_quant peak advantage grows with sequence length

Examples:
  python scripts/06c_validate_prefill_sdpa.py configs/new_rank.yaml
  python scripts/06c_validate_prefill_sdpa.py configs/new_rank.yaml --seq-lens 4096 8192 16384 --max-new-tokens 8
  python scripts/06c_validate_prefill_sdpa.py configs/new_rank.yaml --skip-run --output-dir artifacts/peak_segments/phase1_prefill_sdpa
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def _format_nbytes(nbytes: int) -> str:
    if nbytes == 0:
        return "0 B"
    units = ("B", "KB", "MB", "GB", "TB")
    value = float(abs(nbytes))
    unit = units[0]
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            break
        value /= 1024.0
    sign = "-" if nbytes < 0 else ""
    if unit == "B":
        return f"{sign}{int(value)} {unit}"
    return f"{sign}{value:.2f} {unit}"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _record_by_label(data: dict[str, Any], label: str) -> dict[str, Any]:
    for rec in data.get("records", []):
        if rec.get("label") == label:
            return rec
    raise KeyError(f"{data.get('mode')} missing record label {label!r}")


def _summarize_one(path: Path) -> dict[str, Any]:
    data = _load_json(path)
    start = _record_by_label(data, "profile.start_after_reset")
    prefill_before = _record_by_label(data, "prefill.forward.before")
    prefill_after = _record_by_label(data, "prefill.forward.after")
    setup_after = _record_by_label(data, "mode.setup.after")
    stats = data["stats"]

    prefill_delta = int(prefill_after["peak_allocated_bytes"]) - int(prefill_before["peak_allocated_bytes"])
    profile_delta = int(stats["peak_gpu_bytes"]) - int(start["peak_allocated_bytes"])

    return {
        "path": str(path),
        "mode": data["mode"],
        "requested_seq_len": data["requested_seq_len"],
        "actual_seq_len": data["actual_seq_len"],
        "max_new_tokens": data["max_new_tokens"],
        "setup_allocated_bytes": int(setup_after["allocated_bytes"]),
        "setup_allocated": setup_after["allocated"],
        "prefill_peak_delta_bytes": prefill_delta,
        "prefill_peak_delta": _format_nbytes(prefill_delta),
        "final_peak_bytes": int(stats["peak_gpu_bytes"]),
        "final_peak": stats["peak_gpu"],
        "profile_peak_delta_bytes": profile_delta,
        "profile_peak_delta": _format_nbytes(profile_delta),
        "cache_runtime_bytes": int(stats["cache_runtime_bytes"]),
        "cache_runtime": stats["cache_runtime"],
        "kv_compression_ratio": float(stats["kv_compression_ratio"]),
    }


def _run_profile(config: Path, mode: str, seq_len: int, max_new_tokens: int, output: Path) -> None:
    script = Path(__file__).resolve().parent / "06b_profile_peak_segments.py"
    cmd = [
        sys.executable,
        str(script),
        str(config),
        "--mode",
        mode,
        "--seq-len",
        str(seq_len),
        "--max-new-tokens",
        str(max_new_tokens),
        "--no-hawp-internals",
        "--output",
        str(output),
    ]
    print("[phase1] run:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _print_table(rows: list[dict[str, Any]]) -> None:
    print("\n[phase1] Prefill SDPA / peak validation")
    header = (
        f"{'seq':>7} {'prefill_base':>14} {'prefill_hawp':>14} "
        f"{'prefill_ok':>10} {'peak_base':>12} {'peak_hawp':>12} "
        f"{'final_ok':>9} {'advantage':>12} {'kvx':>7}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['actual_seq_len']:>7} "
            f"{row['baseline_prefill_peak_delta']:>14} "
            f"{row['hawp_prefill_peak_delta']:>14} "
            f"{str(row['prefill_ok']):>10} "
            f"{row['baseline_final_peak']:>12} "
            f"{row['hawp_final_peak']:>12} "
            f"{str(row['final_peak_ok']):>9} "
            f"{row['final_peak_advantage']:>12} "
            f"{row['hawp_kv_compression_ratio']:>7.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate HAWP prefill SDPA peak-memory behavior")
    parser.add_argument("config", nargs="?", default="configs/new_rank.yaml")
    parser.add_argument("--seq-lens", nargs="+", type=int, default=[4096, 8192, 16384])
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--output-dir", default="artifacts/peak_segments/phase1_prefill_sdpa")
    parser.add_argument("--summary", default=None)
    parser.add_argument("--skip-run", action="store_true")
    args = parser.parse_args()

    config = Path(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries_by_seq: list[dict[str, Any]] = []
    for seq_len in args.seq_lens:
        paths = {
            "baseline": output_dir / f"baseline_{seq_len}.json",
            "hawp_quant": output_dir / f"hawp_quant_{seq_len}.json",
        }

        if not args.skip_run:
            _run_profile(config, "baseline", seq_len, args.max_new_tokens, paths["baseline"])
            _run_profile(config, "hawp_quant", seq_len, args.max_new_tokens, paths["hawp_quant"])

        baseline = _summarize_one(paths["baseline"])
        hawp = _summarize_one(paths["hawp_quant"])
        advantage = baseline["final_peak_bytes"] - hawp["final_peak_bytes"]
        prefill_advantage = baseline["prefill_peak_delta_bytes"] - hawp["prefill_peak_delta_bytes"]
        summaries_by_seq.append({
            "requested_seq_len": seq_len,
            "actual_seq_len": hawp["actual_seq_len"],
            "baseline": baseline,
            "hawp_quant": hawp,
            "baseline_prefill_peak_delta": baseline["prefill_peak_delta"],
            "hawp_prefill_peak_delta": hawp["prefill_peak_delta"],
            "prefill_advantage_bytes": prefill_advantage,
            "prefill_advantage": _format_nbytes(prefill_advantage),
            "prefill_ok": hawp["prefill_peak_delta_bytes"] <= baseline["prefill_peak_delta_bytes"],
            "baseline_final_peak": baseline["final_peak"],
            "hawp_final_peak": hawp["final_peak"],
            "final_peak_advantage_bytes": advantage,
            "final_peak_advantage": _format_nbytes(advantage),
            "final_peak_ok": hawp["final_peak_bytes"] < baseline["final_peak_bytes"],
            "hawp_kv_compression_ratio": hawp["kv_compression_ratio"],
        })

    advantages = [row["final_peak_advantage_bytes"] for row in summaries_by_seq]
    monotonic_advantage = all(a <= b for a, b in zip(advantages, advantages[1:]))
    overall = {
        "config": str(config),
        "seq_lens": args.seq_lens,
        "max_new_tokens": args.max_new_tokens,
        "all_prefill_ok": all(row["prefill_ok"] for row in summaries_by_seq),
        "all_final_peak_ok": all(row["final_peak_ok"] for row in summaries_by_seq),
        "final_peak_advantage_monotonic": monotonic_advantage,
        "rows": summaries_by_seq,
    }

    _print_table(summaries_by_seq)
    print("\n[phase1] verdict")
    print(f"  all_prefill_ok={overall['all_prefill_ok']}")
    print(f"  all_final_peak_ok={overall['all_final_peak_ok']}")
    print(f"  final_peak_advantage_monotonic={overall['final_peak_advantage_monotonic']}")

    summary_path = Path(args.summary) if args.summary else output_dir / "summary.json"
    _save_json(overall, summary_path)
    print(f"\n[phase1] saved summary to {summary_path}")


if __name__ == "__main__":
    main()
