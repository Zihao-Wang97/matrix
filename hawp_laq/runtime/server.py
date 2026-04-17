from pathlib import Path
from hawp_laq.config import load_config, HAWPLAQConfig


def start_server(config_path: str | Path) -> None:
    cfg: HAWPLAQConfig = load_config(config_path)
    print(f"[runtime] mode={cfg.mode} model={cfg.model.name}")
    print(f"[runtime] serving at {cfg.serving.host}:{cfg.serving.port}")
    print("[runtime] server stub - not implemented yet")
