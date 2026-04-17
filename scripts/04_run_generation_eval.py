#!/usr/bin/env python
"""Generation eval: python scripts/04_run_generation_eval.py [config] [--mode MODE]"""

import argparse
from pathlib import Path

from hawp_laq.runtime.generate import run_baseline, run_hawp_only, run_hawp_quant, run_hawp_quant_sched


_MODE_MAP = {
    "baseline": run_baseline,
    "hawp_only": run_hawp_only,
    "hawp_quant": run_hawp_quant,
    "hawp_quant_sched": run_hawp_quant_sched,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="HAWP-LAQ generation eval")
    parser.add_argument(
        "config",
        nargs="?",
        default=None,
        help="Path to yaml config (default: configs/dev_local.yaml)",
    )
    parser.add_argument(
        "--mode",
        choices=list(_MODE_MAP.keys()),
        default="baseline",
        help="Generation mode (default: baseline)",
    )
    args = parser.parse_args()

    if args.config is None:
        script_dir = Path(__file__).resolve().parent
        args.config = script_dir.parent / "configs" / "dev_local.yaml"

    _MODE_MAP[args.mode](args.config)


if __name__ == "__main__":
    main()
