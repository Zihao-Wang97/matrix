from __future__ import annotations

import importlib.util
import inspect

import pytest


def _load_compare_module():
    spec = importlib.util.spec_from_file_location(
        "compare", "scripts/08_compare_modes.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestHawpOnlyUsesConvertAndLoadProjectors:
    def test_setup_delegates_to_mode_runner(self):
        mod = _load_compare_module()
        source = inspect.getsource(mod)
        assert "setup_mode" in source

    def test_no_manual_mode_setup_logic(self):
        mod = _load_compare_module()
        source = inspect.getsource(mod)
        assert "def _setup(" not in source
        assert "_setup_hawp_quant_on_model" not in source
        assert "_setup_pure_quant_only_on_model" not in source

    def test_imports_mode_runner(self):
        mod = _load_compare_module()
        assert hasattr(mod, "setup_mode")
        assert hasattr(mod, "make_reset_fn")
        assert hasattr(mod, "profile_generate_by_mode")

    def test_no_stale_convert_llama_to_hawp_import(self):
        mod = _load_compare_module()
        source = inspect.getsource(mod)
        lines_after_change = [
            l for l in source.splitlines()
            if "convert_llama_to_hawp" in l
        ]
        assert len(lines_after_change) == 0, (
            "convert_llama_to_hawp should no longer be imported in 08_compare_modes.py"
        )
