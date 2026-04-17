import torch
import pytest
from hawp_laq.utils.io import save_pt, load_pt, save_json, load_json
from hawp_laq.utils.memory import tensor_nbytes, format_nbytes
from hawp_laq.utils.seed import set_seed
from hawp_laq.utils.logging import build_logger


class TestIO:
    def test_pt_roundtrip(self, tmp_path):
        data = {"tensor": torch.randn(3, 4), "label": 42}
        p = tmp_path / "test.pt"
        save_pt(data, p)
        loaded = load_pt(p)
        assert loaded["label"] == 42
        assert torch.equal(data["tensor"], loaded["tensor"])

    def test_json_roundtrip(self, tmp_path):
        data = {"lr": 0.001, "epochs": 100, "tags": ["a", "b"]}
        p = tmp_path / "test.json"
        save_json(data, p)
        loaded = load_json(p)
        assert loaded == data


class TestMemory:
    def test_tensor_nbytes_float32(self):
        t = torch.randn(10, 20)
        assert tensor_nbytes(t) == 10 * 20 * 4

    def test_tensor_nbytes_float16(self):
        t = torch.randn(5, 6, dtype=torch.float16)
        assert tensor_nbytes(t) == 5 * 6 * 2

    def test_format_nbytes(self):
        assert format_nbytes(0) == "0 B"
        assert format_nbytes(1024) == "1.00 KB"
        assert format_nbytes(1048576) == "1.00 MB"


class TestSeed:
    def test_reproducibility(self):
        set_seed(42)
        a = torch.randn(100)
        set_seed(42)
        b = torch.randn(100)
        assert torch.equal(a, b)

    def test_different_seed(self):
        set_seed(1)
        a = torch.randn(100)
        set_seed(2)
        b = torch.randn(100)
        assert not torch.equal(a, b)


class TestBuildLogger:
    def test_logger_created(self, tmp_path):
        logger = build_logger("test_build", log_dir=tmp_path)
        assert logger.name == "test_build"
        assert len(logger.handlers) == 2

    def test_logger_no_file(self):
        logger = build_logger("test_nofile")
        assert len(logger.handlers) == 1

    def test_idempotent(self, tmp_path):
        a = build_logger("test_idem", log_dir=tmp_path)
        b = build_logger("test_idem", log_dir=tmp_path)
        assert a is b
