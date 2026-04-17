import torch
import pytest
from pathlib import Path
from unittest.mock import MagicMock
from hawp_laq.config import load_config, CalibConfig
from hawp_laq.offline.hooks import _find_attention_modules, count_attention_layers
from hawp_laq.offline.collector import CalibrationCollector


_CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


class TestCalibConfig:
    def test_dev_local_calib(self):
        cfg = load_config(_CONFIG_DIR / "dev_local.yaml")
        assert isinstance(cfg.calib, CalibConfig)
        assert cfg.calib.nsamples == 2
        assert cfg.calib.seq_len == 64

    def test_run_server_calib(self):
        cfg = load_config(_CONFIG_DIR / "run_server.yaml")
        assert cfg.calib.nsamples == 128
        assert cfg.calib.seq_len == 2048


class TestFindAttention:
    def test_opt_has_attention_layers(self):
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
        layers = _find_attention_modules(model)
        assert len(layers) > 0
        assert count_attention_layers(model) == len(layers)

    def test_attention_indices_sequential(self):
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
        layers = _find_attention_modules(model)
        indices = [idx for idx, _ in layers]
        assert indices == list(range(len(layers)))


class TestCollectorOnQKV:
    def test_buffer_accumulates(self):
        model = MagicMock()
        tokenizer = MagicMock()
        cfg = load_config(_CONFIG_DIR / "dev_local.yaml")
        collector = CalibrationCollector(model, tokenizer, cfg)
        q = torch.randn(1, 8, 64)
        k = torch.randn(1, 8, 64)
        v = torch.randn(1, 8, 64)
        collector._on_qkv(0, q, k, v)
        collector._on_qkv(0, q, k, v)
        assert len(collector._buffers[0]["q"]) == 2
        assert collector.n_layers == 1

    def test_buffer_multi_layer(self):
        model = MagicMock()
        tokenizer = MagicMock()
        cfg = load_config(_CONFIG_DIR / "dev_local.yaml")
        collector = CalibrationCollector(model, tokenizer, cfg)
        for i in range(3):
            collector._on_qkv(i, torch.randn(1, 8), torch.randn(1, 8), torch.randn(1, 8))
        assert collector.n_layers == 3


class TestCollectorSave:
    def _make_model(self):
        model = MagicMock()
        model.config.num_attention_heads = 12
        return model

    def test_save_creates_layer_files(self, tmp_path):
        model = self._make_model()
        tokenizer = MagicMock()
        cfg = load_config(_CONFIG_DIR / "dev_local.yaml")
        collector = CalibrationCollector(model, tokenizer, cfg)
        for i in range(2):
            collector._on_qkv(i, torch.randn(1, 8, 64), torch.randn(1, 8, 64), torch.randn(1, 8, 64))
        out = collector.save(tmp_path / "calib")
        assert (out / "layer_0.pt").exists()
        assert (out / "layer_1.pt").exists()
        assert (out / "meta.pt").exists()
        d0 = torch.load(out / "layer_0.pt", map_location="cpu", weights_only=False)
        assert "q" in d0 and "k" in d0 and "v" in d0
        assert d0["q"].shape == (1, 8, 64)
        meta = torch.load(out / "meta.pt", map_location="cpu", weights_only=False)
        assert meta["n_heads"] == 12

    def test_save_clears_buffers(self, tmp_path):
        model = self._make_model()
        tokenizer = MagicMock()
        cfg = load_config(_CONFIG_DIR / "dev_local.yaml")
        collector = CalibrationCollector(model, tokenizer, cfg)
        collector._on_qkv(0, torch.randn(1, 8), torch.randn(1, 8), torch.randn(1, 8))
        collector.save(tmp_path / "calib")
        assert collector.n_layers == 0
