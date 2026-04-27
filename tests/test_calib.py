import torch
import pytest
from pathlib import Path
from unittest.mock import MagicMock
from hawp_laq.config import load_config, CalibConfig
from hawp_laq.offline.hooks import _find_attention_modules, count_attention_layers
from hawp_laq.offline.collector import (
    CalibrationCollector,
    _resolve_capture_mode,
    _expand_kv_for_trainer,
    _repeat_kv_4d,
)


_CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


class TestCalibConfig:
    def test_dev_local_calib(self):
        cfg = load_config(_CONFIG_DIR / "dev_local.yaml")
        assert isinstance(cfg.calib, CalibConfig)
        assert cfg.calib.nsamples == 2
        assert cfg.calib.seq_len == 64

    def test_capture_mode_field(self):
        cfg = load_config(_CONFIG_DIR / "dev_local.yaml")
        assert cfg.calib.capture_mode == "auto"


class TestResolveCaptureMode:
    def test_auto_opt_is_pre_rope(self):
        assert _resolve_capture_mode("auto", "opt") == "pre_rope"

    def test_auto_gpt_neox_is_pre_rope(self):
        assert _resolve_capture_mode("auto", "gpt_neox") == "pre_rope"

    def test_auto_llama_is_post_rope(self):
        assert _resolve_capture_mode("auto", "llama") == "post_rope"

    def test_auto_mistral_is_post_rope(self):
        assert _resolve_capture_mode("auto", "mistral") == "post_rope"

    def test_auto_qwen2_is_post_rope(self):
        assert _resolve_capture_mode("auto", "qwen2") == "post_rope"

    def test_auto_unknown_defaults_post_rope(self):
        assert _resolve_capture_mode("auto", "some_new_model") == "post_rope"

    def test_explicit_pre_rope(self):
        assert _resolve_capture_mode("pre_rope", "llama") == "pre_rope"

    def test_explicit_post_rope(self):
        assert _resolve_capture_mode("post_rope", "opt") == "post_rope"

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="Unknown capture_mode"):
            _resolve_capture_mode("invalid", "llama")


class TestRepeatKV4D:
    def test_no_repeat_is_noop(self):
        x = torch.randn(1, 8, 4, 64)
        out = _repeat_kv_4d(x, 1)
        assert torch.equal(out, x)

    def test_repeat_2x(self):
        x = torch.randn(1, 8, 4, 64)
        out = _repeat_kv_4d(x, 2)
        assert out.shape == (1, 16, 4, 64)
        for g in range(8):
            assert torch.equal(out[:, g * 2, :, :], x[:, g, :, :])
            assert torch.equal(out[:, g * 2 + 1, :, :], x[:, g, :, :])

    def test_repeat_4x(self):
        x = torch.randn(2, 4, 6, 32)
        out = _repeat_kv_4d(x, 4)
        assert out.shape == (2, 16, 6, 32)


class TestExpandKVForTrainer:
    def test_non_gqa_noop(self):
        k = torch.randn(1, 8, 512)
        out = _expand_kv_for_trainer(k, n_heads=8, n_kv_heads=8)
        assert torch.equal(out, k)

    def test_gqa_8kv_32q(self):
        n_q_heads = 32
        n_kv_heads = 8
        head_dim = 128
        k = torch.randn(1, 64, n_kv_heads * head_dim)
        out = _expand_kv_for_trainer(k, n_heads=n_q_heads, n_kv_heads=n_kv_heads)
        assert out.shape == (1, 64, n_q_heads * head_dim)

    def test_mqa_1kv_8q(self):
        n_q_heads = 8
        n_kv_heads = 1
        head_dim = 64
        k = torch.randn(2, 16, n_kv_heads * head_dim)
        out = _expand_kv_for_trainer(k, n_heads=n_q_heads, n_kv_heads=n_kv_heads)
        assert out.shape == (2, 16, n_q_heads * head_dim)


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

    def test_gqa_kv_expanded_to_match_q(self):
        model = MagicMock()
        model.config.num_attention_heads = 32
        model.config.num_key_value_heads = 8
        tokenizer = MagicMock()
        cfg = load_config(_CONFIG_DIR / "dev_local.yaml")
        collector = CalibrationCollector(model, tokenizer, cfg)

        n_q_heads, n_kv_heads, head_dim = 32, 8, 128
        q = torch.randn(1, 16, n_q_heads * head_dim)
        k = torch.randn(1, 16, n_kv_heads * head_dim)
        v = torch.randn(1, 16, n_kv_heads * head_dim)

        collector._on_qkv(0, q, k, v)
        saved_q = collector._buffers[0]["q"][0]
        saved_k = collector._buffers[0]["k"][0]
        saved_v = collector._buffers[0]["v"][0]

        assert saved_q.shape == (1, 16, n_q_heads * head_dim)
        assert saved_k.shape == (1, 16, n_q_heads * head_dim)
        assert saved_v.shape == (1, 16, n_q_heads * head_dim)


class TestCollectorPostRopeCallback:
    def _make_collector(self, n_q_heads=12, n_kv_heads=12):
        model = MagicMock()
        model.config.num_attention_heads = n_q_heads
        model.config.num_key_value_heads = n_kv_heads
        tokenizer = MagicMock()
        cfg = load_config(_CONFIG_DIR / "dev_local.yaml")
        return CalibrationCollector(model, tokenizer, cfg)

    def test_mha_callback_3d_shape(self):
        collector = self._make_collector(n_q_heads=12, n_kv_heads=12)
        head_dim = 64
        q_4d = torch.randn(1, 12, 8, head_dim)
        k_4d = torch.randn(1, 12, 8, head_dim)
        v_4d = torch.randn(1, 12, 8, head_dim)
        collector._on_post_rope_callback(0, q_4d, k_4d, v_4d)
        assert collector._buffers[0]["q"][0].shape == (1, 8, 12 * head_dim)
        assert collector._buffers[0]["k"][0].shape == (1, 8, 12 * head_dim)
        assert collector._buffers[0]["v"][0].shape == (1, 8, 12 * head_dim)

    def test_gqa_callback_3d_shape(self):
        collector = self._make_collector(n_q_heads=32, n_kv_heads=8)
        head_dim = 128
        q_4d = torch.randn(1, 32, 4, head_dim)
        k_4d = torch.randn(1, 8, 4, head_dim)
        v_4d = torch.randn(1, 8, 4, head_dim)
        collector._on_post_rope_callback(0, q_4d, k_4d, v_4d)
        assert collector._buffers[0]["q"][0].shape == (1, 4, 32 * head_dim)
        assert collector._buffers[0]["k"][0].shape == (1, 4, 32 * head_dim)
        assert collector._buffers[0]["v"][0].shape == (1, 4, 32 * head_dim)

    def test_mqa_callback_3d_shape(self):
        collector = self._make_collector(n_q_heads=16, n_kv_heads=1)
        head_dim = 64
        q_4d = torch.randn(1, 16, 8, head_dim)
        k_4d = torch.randn(1, 1, 8, head_dim)
        v_4d = torch.randn(1, 1, 8, head_dim)
        collector._on_post_rope_callback(0, q_4d, k_4d, v_4d)
        assert collector._buffers[0]["q"][0].shape == (1, 8, 16 * head_dim)
        assert collector._buffers[0]["k"][0].shape == (1, 8, 16 * head_dim)
        assert collector._buffers[0]["v"][0].shape == (1, 8, 16 * head_dim)


class TestCollectorSave:
    def _make_model(self, n_heads=12, n_kv_heads=None):
        model = MagicMock()
        model.config.num_attention_heads = n_heads
        model.config.num_key_value_heads = n_kv_heads or n_heads
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
        collector._on_qkv(0, torch.randn(1, 8, 64), torch.randn(1, 8, 64), torch.randn(1, 8, 64))
        collector.save(tmp_path / "calib")
        assert collector.n_layers == 0

    def test_meta_includes_new_fields(self, tmp_path):
        model = self._make_model(n_heads=32, n_kv_heads=8)
        tokenizer = MagicMock()
        cfg = load_config(_CONFIG_DIR / "dev_local.yaml")
        collector = CalibrationCollector(model, tokenizer, cfg)
        collector._capture_mode = "post_rope"
        collector._on_qkv(0, torch.randn(1, 4, 32 * 128), torch.randn(1, 4, 8 * 128), torch.randn(1, 4, 8 * 128))
        out = collector.save(tmp_path / "calib")
        meta = torch.load(out / "meta.pt", map_location="cpu", weights_only=False)
        assert meta["capture_mode"] == "post_rope"
        assert meta["rope_applied"] is True
        assert meta["n_kv_heads"] == 8


class TestCollectTeardown:
    def test_pre_rope_removes_hooks_on_error(self):
        model = MagicMock()
        model.config.num_attention_heads = 12
        model.config.num_key_value_heads = 12
        model.config.model_type = "opt"
        tokenizer = MagicMock()
        cfg = load_config(_CONFIG_DIR / "dev_local.yaml")
        collector = CalibrationCollector(model, tokenizer, cfg)

        class FailLoader:
            def __len__(self):
                return 1
            def __iter__(self):
                raise RuntimeError("boom")

        collector._handles = [MagicMock()]
        with pytest.raises(RuntimeError, match="boom"):
            collector.collect(FailLoader())
        assert collector._handles == []

    def test_post_rope_clears_callbacks_on_error(self):
        from hawp_laq.modeling.attention_hawp import HAWPAttention
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m", torch_dtype=torch.float32)
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
        cfg = load_config(_CONFIG_DIR / "dev_local.yaml")
        collector = CalibrationCollector(model, tokenizer, cfg)

        collector._capture_mode = "post_rope"
        collector._setup_post_rope_collection()

        hawp_modules = [m for m in model.modules() if isinstance(m, HAWPAttention)]
        assert len(hawp_modules) > 0, "Should have HAWPAttention modules after setup"
        assert all(m._calib_callback is not None for m in hawp_modules), "Callbacks should be set"

        class FailLoader:
            def __len__(self):
                return 1
            def __iter__(self):
                yield tokenizer("Hello", return_tensors="pt")["input_ids"]
                raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            collector._collect_post_rope(FailLoader())

        collector._teardown_post_rope_collection()
        assert all(m._calib_callback is None for m in hawp_modules), "All callbacks must be cleared after teardown"


class TestTrainGuardOldMeta:
    def test_old_opt_meta_not_rejected(self, tmp_path):
        import importlib.util
        from hawp_laq.utils.io import save_pt

        calib_dir = tmp_path / "calib"
        calib_dir.mkdir()
        meta = {
            "n_layers": 12,
            "n_heads": 12,
            "nsamples": 2,
            "seq_len": 64,
            "model_id": "facebook/opt-125m",
        }
        save_pt(meta, calib_dir / "meta.pt")

        q = torch.randn(2, 64, 768)
        k = torch.randn(2, 64, 768)
        v = torch.randn(2, 64, 768)
        save_pt({"q": q, "k": k, "v": v}, calib_dir / "layer_0.pt")

        cfg = load_config(_CONFIG_DIR / "dev_local.yaml")
        cfg.calib.output_dir = str(calib_dir)
        proj_dir = tmp_path / "projectors"
        proj_dir.mkdir()
        cfg.projector.output_dir = str(proj_dir)

        spec = importlib.util.spec_from_file_location(
            "train_proj", _CONFIG_DIR.parent / "scripts" / "02_train_projectors.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod._run_single_group(cfg, layer_idx=0)
        assert (Path(cfg.projector.output_dir) / "layer_0" / "projector.pt").exists()

    def test_llama_with_pre_rope_rejected(self, tmp_path):
        import importlib.util
        from hawp_laq.utils.io import save_pt

        calib_dir = tmp_path / "calib"
        calib_dir.mkdir()
        meta = {
            "capture_mode": "pre_rope",
            "model_type": "llama",
            "n_heads": 32,
            "n_layers": 1,
            "nsamples": 1,
            "seq_len": 8,
            "model_id": "meta-llama/test",
        }
        save_pt(meta, calib_dir / "meta.pt")

        q = torch.randn(1, 8, 4096)
        k = torch.randn(1, 8, 4096)
        v = torch.randn(1, 8, 4096)
        save_pt({"q": q, "k": k, "v": v}, calib_dir / "layer_0.pt")

        cfg = load_config(_CONFIG_DIR / "dev_local.yaml")
        cfg.calib.output_dir = str(calib_dir)
        proj_dir = tmp_path / "projectors"
        proj_dir.mkdir()
        cfg.projector.output_dir = str(proj_dir)

        spec = importlib.util.spec_from_file_location(
            "train_proj", _CONFIG_DIR.parent / "scripts" / "02_train_projectors.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        with pytest.raises(ValueError, match="post_rope"):
            mod._run_single_group(cfg, layer_idx=0)


class TestCollectorImplMeta:
    def test_pre_rope_meta_has_original_hooks(self, tmp_path):
        model = MagicMock()
        model.config.num_attention_heads = 12
        model.config.num_key_value_heads = 12
        model.config.model_type = "opt"
        tokenizer = MagicMock()
        cfg = load_config(_CONFIG_DIR / "dev_local.yaml")
        collector = CalibrationCollector(model, tokenizer, cfg)
        collector._on_qkv(0, torch.randn(1, 8, 64), torch.randn(1, 8, 64), torch.randn(1, 8, 64))
        collector._capture_mode = "pre_rope"
        collector._collector_impl = "original_model_hooks"
        out = collector.save(tmp_path / "calib")
        meta = torch.load(out / "meta.pt", map_location="cpu", weights_only=False)
        assert meta["collector_impl"] == "original_model_hooks"

    def test_post_rope_meta_has_hawp_eager(self, tmp_path):
        model = MagicMock()
        model.config.num_attention_heads = 12
        model.config.num_key_value_heads = 12
        tokenizer = MagicMock()
        cfg = load_config(_CONFIG_DIR / "dev_local.yaml")
        collector = CalibrationCollector(model, tokenizer, cfg)
        collector._on_qkv(0, torch.randn(1, 8, 64), torch.randn(1, 8, 64), torch.randn(1, 8, 64))
        collector._capture_mode = "post_rope"
        collector._collector_impl = "hawp_full_rank_eager"
        out = collector.save(tmp_path / "calib")
        meta = torch.load(out / "meta.pt", map_location="cpu", weights_only=False)
        assert meta["collector_impl"] == "hawp_full_rank_eager"


class TestGQADivisibility:
    def test_non_divisible_raises(self):
        with pytest.raises(ValueError, match="divisible"):
            _expand_kv_for_trainer(
                torch.randn(1, 8, 512),
                n_heads=7,
                n_kv_heads=3,
            )
