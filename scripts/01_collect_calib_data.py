#!/usr/bin/env python
"""Collect calibration data: python scripts/01_collect_calib_data.py [config]"""

import argparse
from pathlib import Path

from hawp_laq.config import load_config
from hawp_laq.offline.collector import run_calibration


def main() -> None:
    parser = argparse.ArgumentParser(description="HAWP-LAQ calibration data collection")
    parser.add_argument(
        "config",
        nargs="?",
        default=None,
        help="Path to yaml config (default: configs/dev_local.yaml)",
    )
    args = parser.parse_args()

    if args.config is None:
        args.config = Path(__file__).resolve().parent.parent / "configs" / "dev_local.yaml"

    cfg = load_config(args.config)
    run_calibration(cfg)


if __name__ == "__main__":
    main()
