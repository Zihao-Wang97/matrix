from pathlib import Path
from hawp_laq.config import load_config, HAWPLAQConfig


def run_offline_pipeline(config_path: str | Path) -> None:
    cfg: HAWPLAQConfig = load_config(config_path)
    print(f"[offline] mode={cfg.mode} model={cfg.model.name}")
    print("[offline] pipeline stub - not implemented yet")
