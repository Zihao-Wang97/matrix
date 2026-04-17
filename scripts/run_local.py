#!/usr/bin/env python
"""Convenience wrapper: python scripts/run_local.py"""
import sys
from pathlib import Path
from hawp_laq.offline.pipeline import run_offline_pipeline

if __name__ == "__main__":
    config = Path(__file__).resolve().parent.parent / "configs" / "dev_local.yaml"
    run_offline_pipeline(config)
