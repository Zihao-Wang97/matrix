from pathlib import Path
from hawp_laq.config import load_config, HAWPLAQConfig, DataConfig, ModelConfig


def test_load_dev_local():
    cfg = load_config(Path(__file__).resolve().parent.parent / "configs" / "dev_local.yaml")
    assert isinstance(cfg, HAWPLAQConfig)
    assert cfg.mode == "local"
    assert isinstance(cfg.data, DataConfig)
    assert isinstance(cfg.model, ModelConfig)
    assert cfg.train.device == "cpu"
    assert isinstance(cfg.data.root, Path)


def test_load_run_server():
    cfg = load_config(Path(__file__).resolve().parent.parent / "configs" / "run_server.yaml")
    assert cfg.mode == "server"
    assert cfg.train.device == "cuda"
    assert cfg.serving.port == 8080


def test_missing_config_raises():
    import pytest
    with pytest.raises(FileNotFoundError):
        load_config("/nonexistent/path.yaml")
