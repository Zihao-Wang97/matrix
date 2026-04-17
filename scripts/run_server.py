#!/usr/bin/env python
"""Convenience wrapper: python scripts/run_server.py"""
import sys
from pathlib import Path
from hawp_laq.runtime.server import start_server

if __name__ == "__main__":
    config = Path(__file__).resolve().parent.parent / "configs" / "run_server.yaml"
    start_server(config)
