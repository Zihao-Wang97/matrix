from __future__ import annotations

import torch

from hawp_laq.config import load_config
from hawp_laq.modeling.attention_hawp import HAWPAttention
from hawp_laq.modeling.modeling_llama_hawp import convert_llama_to_hawp
from hawp_laq.runtime.projector_bank import save_projectors, load_projectors, load_ranks

_CONFIG_PATH = "configs/dev_local.yaml"

_CANONICAL_KEYS = ("p_k", "p_v", "gamma", "r_k", "r_v")


def _build_converted_model(r_k: int, r_v: int):
    from transformers import AutoModelForCausalLM

    cfg = load_config(_CONFIG_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.model_id, torch_dtype="auto",
    )
    return convert_llama_to_hawp(model, r_k=r_k, r_v=r_v)


def _collect_hawp_modules(model):
    return [
        m for _, m in model.named_modules() if isinstance(m, HAWPAttention)
    ]


def test_saved_file_has_canonical_fields(tmp_path):
    cfg = load_config(_CONFIG_PATH)
    r_k = cfg.projector.r_k or cfg.projector.rank
    r_v = cfg.projector.r_v or cfg.projector.rank
    model = _build_converted_model(r_k, r_v)
    save_projectors(model, tmp_path / "proj")

    for mod in _collect_hawp_modules(model):
        pt = tmp_path / "proj" / f"layer_{mod.layer_idx}" / "projector.pt"
        assert pt.exists(), f"Missing projector file for layer {mod.layer_idx}"
        data = torch.load(pt, map_location="cpu", weights_only=True)
        for key in _CANONICAL_KEYS:
            assert key in data, f"Missing key '{key}' in layer {mod.layer_idx}"


def test_r_k_r_v_consistent(tmp_path):
    cfg = load_config(_CONFIG_PATH)
    r_k = cfg.projector.r_k or cfg.projector.rank
    r_v = cfg.projector.r_v or cfg.projector.rank
    model = _build_converted_model(r_k, r_v)
    save_projectors(model, tmp_path / "proj")

    for mod in _collect_hawp_modules(model):
        pt = tmp_path / "proj" / f"layer_{mod.layer_idx}" / "projector.pt"
        data = torch.load(pt, map_location="cpu", weights_only=True)
        assert data["r_k"] == r_k, f"layer {mod.layer_idx}: r_k {data['r_k']} != {r_k}"
        assert data["r_v"] == r_v, f"layer {mod.layer_idx}: r_v {data['r_v']} != {r_v}"


def test_roundtrip_p_k_p_v_gamma(tmp_path):
    cfg = load_config(_CONFIG_PATH)
    r_k = cfg.projector.r_k or cfg.projector.rank
    r_v = cfg.projector.r_v or cfg.projector.rank

    model_a = _build_converted_model(r_k, r_v)
    for mod in _collect_hawp_modules(model_a):
        mod.p_k.data.normal_()
        mod.p_v.data.normal_()
        mod.gamma.data.fill_(2.5)

    save_projectors(model_a, tmp_path / "proj")

    model_b = _build_converted_model(r_k, r_v)
    load_projectors(model_b, tmp_path / "proj")

    for m_a, m_b in zip(_collect_hawp_modules(model_a), _collect_hawp_modules(model_b)):
        assert torch.allclose(m_a.p_k.data, m_b.p_k.data), f"p_k mismatch layer {m_a.layer_idx}"
        assert torch.allclose(m_a.p_v.data, m_b.p_v.data), f"p_v mismatch layer {m_a.layer_idx}"
        assert torch.allclose(m_a.gamma.data, m_b.gamma.data), f"gamma mismatch layer {m_a.layer_idx}"


def test_ranks_json_matches_model(tmp_path):
    cfg = load_config(_CONFIG_PATH)
    r_k = cfg.projector.r_k or cfg.projector.rank
    r_v = cfg.projector.r_v or cfg.projector.rank
    model = _build_converted_model(r_k, r_v)
    save_projectors(model, tmp_path / "proj")

    ranks = load_ranks(tmp_path / "proj")
    n_layers = len(_collect_hawp_modules(model))
    for i in range(n_layers):
        assert ranks[i] == (r_k, r_v), f"layer {i}: expected ({r_k},{r_v}), got {ranks[i]}"


def test_load_legacy_gamma_fields_if_present(tmp_path):
    cfg = load_config(_CONFIG_PATH)
    r_k = cfg.projector.r_k or cfg.projector.rank
    r_v = cfg.projector.r_v or cfg.projector.rank

    model = _build_converted_model(r_k, r_v)
    head_dim = _collect_hawp_modules(model)[0].head_dim

    for mod in _collect_hawp_modules(model):
        layer_dir = tmp_path / "proj" / f"layer_{mod.layer_idx}"
        layer_dir.mkdir(parents=True, exist_ok=True)
        legacy_data = {
            "p_k": torch.eye(head_dim),
            "p_v": torch.eye(head_dim),
            "gamma_k": torch.tensor([1.5]),
            "gamma_v": torch.tensor([2.7]),
            "r_k": r_k,
            "r_v": r_v,
        }
        torch.save(legacy_data, layer_dir / "projector.pt")

    load_projectors(model, tmp_path / "proj")

    for mod in _collect_hawp_modules(model):
        assert torch.allclose(mod.gamma.data.float(), torch.tensor([2.7]), atol=1e-2), \
            f"layer {mod.layer_idx}: gamma should be ~2.7 from gamma_v, got {mod.gamma.item()}"


def test_gamma_preferred_over_legacy_fields(tmp_path):
    cfg = load_config(_CONFIG_PATH)
    r_k = cfg.projector.r_k or cfg.projector.rank
    r_v = cfg.projector.r_v or cfg.projector.rank

    model = _build_converted_model(r_k, r_v)
    head_dim = _collect_hawp_modules(model)[0].head_dim

    for mod in _collect_hawp_modules(model):
        layer_dir = tmp_path / "proj" / f"layer_{mod.layer_idx}"
        layer_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "p_k": torch.eye(head_dim),
            "p_v": torch.eye(head_dim),
            "gamma": torch.tensor([3.3]),
            "gamma_k": torch.tensor([1.5]),
            "gamma_v": torch.tensor([2.7]),
            "r_k": r_k,
            "r_v": r_v,
        }
        torch.save(data, layer_dir / "projector.pt")

    load_projectors(model, tmp_path / "proj")

    for mod in _collect_hawp_modules(model):
        assert torch.allclose(mod.gamma.data.float(), torch.tensor([3.3]), atol=1e-2), \
            f"layer {mod.layer_idx}: gamma should be ~3.3 from gamma field, got {mod.gamma.item()}"
