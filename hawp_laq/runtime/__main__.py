"""Entry point: python -m hawp_laq.runtime.server <config>"""
import sys
from hawp_laq.runtime.server import start_server

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m hawp_laq.runtime.server <config_path>")
        sys.exit(1)
    start_server(sys.argv[1])
