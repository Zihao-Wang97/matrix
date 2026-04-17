"""Entry point: python -m hawp_laq.offline.pipeline <config>"""
import sys
from hawp_laq.offline.pipeline import run_offline_pipeline

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m hawp_laq.offline.pipeline <config_path>")
        sys.exit(1)
    run_offline_pipeline(sys.argv[1])
