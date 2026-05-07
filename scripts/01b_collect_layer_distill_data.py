#!/usr/bin/env python
"""Collect decoder-layer input/output tensors for full layer distillation."""

from __future__ import annotations

import argparse
from pathlib import Path

from hawp_laq.config import load_config
from hawp_laq.offline.layer_distill_data import run_layer_distill_collection


def main() -> None:
    ap = argparse.ArgumentParser(description="Collect hidden_in/hidden_out chunks for layer distillation")
    ap.add_argument("config", type=str)
    ap.add_argument("--output-dir", type=str, default=None, help="Override layer_distill.data_dir")
    ap.add_argument("--clean-output-dir", action="store_true", help="Remove output dir before collecting")
    args = ap.parse_args()

    cfg = load_config(args.config)
    output_dir = Path(args.output_dir) if args.output_dir else None
    run_layer_distill_collection(cfg, output_dir=output_dir, clean_output_dir=args.clean_output_dir)


if __name__ == "__main__":
    main()
