import torch
import pytest
from hawp_laq.config import load_config, HAWPLAQConfig, ModelConfig, GenerationConfig
from hawp_laq.runtime.generate import _resolve_device, _fmt_bytes, _DTYPE_MAP
from pathlib import Path


_CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


class TestConfigGenerationFields:
    def test_dev_local_generation(self):
        cfg = load_config(_CONFIG_DIR / "dev_local.yaml")
        assert isinstance(cfg.generation, GenerationConfig)
        assert cfg.generation.max_new_tokens == 32
        assert cfg.generation.do_sample is False
        assert len(cfg.generation.prompts) == 2

    def test_run_server_generation(self):
        cfg = load_config(_CONFIG_DIR / "run_server.yaml")
        assert cfg.generation.max_new_tokens == 256
        assert cfg.generation.do_sample is True

    def test_model_id_field(self):
        cfg = load_config(_CONFIG_DIR / "dev_local.yaml")
        assert cfg.model.model_id == "facebook/opt-125m"
        assert cfg.model.torch_dtype == "float32"
        assert cfg.model.load_in_4bit is False

    def test_server_model_id(self):
        cfg = load_config(_CONFIG_DIR / "run_server.yaml")
        assert cfg.model.torch_dtype == "float16"
        assert cfg.model.load_in_4bit is True


class TestResolveDevice:
    def test_cpu_stays_cpu(self):
        assert _resolve_device("cpu") == "cpu"

    def test_cuda_fallback(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        assert _resolve_device("cuda") == "cpu"

    def test_cuda_stays_cuda(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        assert _resolve_device("cuda") == "cuda"


class TestDtypeMap:
    def test_all_keys_valid(self):
        for key in ("float32", "float16", "bfloat16"):
            assert key in _DTYPE_MAP
            assert _DTYPE_MAP[key] in (torch.float32, torch.float16, torch.bfloat16)


class TestFmtBytes:
    def test_zero(self):
        assert "0" in _fmt_bytes(0)

    def test_kb(self):
        result = _fmt_bytes(2048)
        assert "KB" in result

    def test_mb(self):
        result = _fmt_bytes(10 * 1024 * 1024)
        assert "MB" in result
