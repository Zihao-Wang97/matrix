from pathlib import Path
from hawp_laq.config import load_config, HAWPLAQConfig


def evaluate(config_path: str | Path) -> dict:
    cfg: HAWPLAQConfig = load_config(config_path)
    print(f"[eval] mode={cfg.mode} model={cfg.model.name}")
    print("[eval] evaluation stub - not implemented yet")
    return {}
