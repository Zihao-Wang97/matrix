from __future__ import annotations

import warnings
from pathlib import Path

import pytest
import torch

from hawp_laq.offline.low_rank_attention_optimizer_torch import (
    OptimConfig,
    make_causal_mask,
    optimize_low_rank_attention_torch,
)
from hawp_laq.offline.projector_trainer import ProjectorTrainer
from hawp_laq.offline.rank_search import search_rank_per_layer
from hawp_laq.utils.io import save_pt


# ── helpers ──────────────────────────────────────────────────────────


def _random_qkv(batch=2, seq=8, d_h=16, seed=0):
    torch.manual_seed(seed)
    Q = torch.randn(batch, seq, d_h)
    K = torch.randn(batch, seq, d_h)
    V = torch.randn(batch, seq, d_h)
    mask = make_causal_mask(batch, seq)
    return Q, K, V, mask


def _make_calib_dir(tmp_path, n_layers=1, n_heads=4, d_model=64):
    calib_dir = tmp_path / "calib"
    calib_dir.mkdir()
    save_pt({"n_layers": n_layers, "n_heads": n_heads, "hidden_size": d_model}, calib_dir / "meta.pt")
    for i in range(n_layers):
        save_pt(
            {
                "q": torch.randn(2, 8, d_model),
                "k": torch.randn(2, 8, d_model),
                "v": torch.randn(2, 8, d_model),
            },
            calib_dir / f"layer_{i}.pt",
        )
    return calib_dir


# ── 1. Riemannian optimizer orthogonality ───────────────────────────


def test_riemannian_optimizer_orthogonality():
    Q, K, V, mask = _random_qkv(batch=2, seq=8, d_h=16, seed=42)
    cfg = OptimConfig(
        r_k=4,
        r_v=4,
        max_steps=30,
        warmup_steps=5,
        eval_every=10,
        verbose=False,
        seed=0,
    )
    result = optimize_low_rank_attention_torch(Q, K, V, mask, cfg)
    P_K = result["P_K"]
    P_V = result["P_V"]

    I_k = torch.eye(P_K.shape[1], dtype=P_K.dtype)
    I_v = torch.eye(P_V.shape[1], dtype=P_V.dtype)
    err_k = torch.linalg.norm(P_K.T @ P_K - I_k).item()
    err_v = torch.linalg.norm(P_V.T @ P_V - I_v).item()

    assert err_k < 1e-4, f"P_K.T @ P_K not identity: err={err_k:.2e}"
    assert err_v < 1e-4, f"P_V.T @ P_V not identity: err={err_v:.2e}"


# ── 2. Optimizer returns best-calib checkpoint ──────────────────────


def test_optimizer_returns_best_calib_checkpoint():
    Q, K, V, mask = _random_qkv(batch=2, seq=8, d_h=16, seed=7)
    cfg = OptimConfig(
        r_k=4,
        r_v=4,
        max_steps=60,
        warmup_steps=5,
        eval_every=10,
        patience=4,
        early_stopping=True,
        verbose=False,
        seed=0,
    )
    result = optimize_low_rank_attention_torch(Q, K, V, mask, cfg)

    assert "best_step" in result, "Missing best_step"
    assert "best_calib_total" in result, "Missing best_calib_total"
    assert "actual_steps" in result, "Missing actual_steps"
    assert "stopped_early" in result, "Missing stopped_early"

    assert isinstance(result["best_step"], int)
    assert isinstance(result["best_calib_total"], float)
    assert isinstance(result["actual_steps"], int)
    assert isinstance(result["stopped_early"], bool)

    assert result["best_step"] >= 1
    assert result["actual_steps"] >= 1
    assert result["best_calib_total"] < float("inf")


# ── 3. ProjectorTrainer output schema ───────────────────────────────


def test_projector_trainer_output_schema():
    d_model, rank_k, rank_v, n_heads = 64, 8, 6, 4
    trainer = ProjectorTrainer(d_model, rank_k, rank_v, n_heads, device="cpu")
    q = torch.randn(1, 4, d_model)
    k = torch.randn(1, 4, d_model)
    v = torch.randn(1, 4, d_model)

    result = trainer.train_one_group(
        q, k, v,
        n_steps=10,
        warmup_steps=2,
        eval_every=5,
        seed=0,
    )

    required_keys = ["p_k", "p_v", "gamma", "r_k", "r_v", "best_calib_total", "metrics"]
    for key in required_keys:
        assert key in result, f"Missing key '{key}' in train_one_group result"

    assert isinstance(result["p_k"], torch.Tensor)
    assert isinstance(result["p_v"], torch.Tensor)
    assert isinstance(result["gamma"], torch.Tensor)
    assert result["r_k"] == rank_k
    assert result["r_v"] == rank_v
    assert isinstance(result["best_calib_total"], float)
    assert isinstance(result["metrics"], dict)

    metrics_keys = ["calib_total", "calib_logits", "calib_attn", "calib_value"]
    for mk in metrics_keys:
        assert mk in result["metrics"], f"Missing metrics key '{mk}'"


# ── 4. Rank search uses best_calib metrics ──────────────────────────


def test_rank_search_uses_best_calib_metrics(tmp_path):
    import hawp_laq.offline.rank_search as rs_mod
    original = rs_mod._evaluate_rank

    def _mock_evaluate(q, k, v, r_k, r_v, d_model, n_heads, **kwargs):
        total_map = {(4, 4): 0.5, (8, 8): 0.05, (16, 16): 0.01}
        total = total_map.get((r_k, r_v), 0.1)
        return {
            "r_k": r_k,
            "r_v": r_v,
            "best_calib_total": total,
            "best_calib_logits": total * 0.6,
            "best_calib_attn": total * 0.2,
            "best_calib_value": total * 0.2,
            "best_step": kwargs.get("n_steps", 10),
            "actual_steps": kwargs.get("n_steps", 10),
            "stopped_early": False,
            "rank_cost": r_k + r_v,
            "p_k_shape": (16, r_k),
            "p_v_shape": (16, r_v),
            "final_loss": 999.0,
            "final_logits_loss": 999.0,
            "final_attn_loss": 999.0,
            "final_value_loss": 999.0,
        }

    rs_mod._evaluate_rank = _mock_evaluate
    try:
        calib_dir = _make_calib_dir(tmp_path, d_model=64, n_heads=4)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = search_rank_per_layer(
                calib_dir=calib_dir,
                rank_candidates=[4, 8, 16],
                n_steps=2,
                device="cpu",
                selection_mode="constraint",
                relative_tolerance=0.20,
            )

        assert result[0] == (16, 16), (
            f"Expected (16,16) — only baseline candidate passes constraint, "
            f"got {result[0]}"
        )
    finally:
        rs_mod._evaluate_rank = original


def test_rank_search_selects_smallest_rank_cost_among_passing(tmp_path):
    import hawp_laq.offline.rank_search as rs_mod
    original = rs_mod._evaluate_rank

    def _mock_evaluate(q, k, v, r_k, r_v, d_model, n_heads, **kwargs):
        return {
            "r_k": r_k,
            "r_v": r_v,
            "best_calib_total": 0.01,
            "best_calib_logits": 0.008,
            "best_calib_attn": 0.001,
            "best_calib_value": 0.001,
            "best_step": kwargs.get("n_steps", 10),
            "actual_steps": kwargs.get("n_steps", 10),
            "stopped_early": False,
            "rank_cost": r_k + r_v,
            "p_k_shape": (16, r_k),
            "p_v_shape": (16, r_v),
        }

    rs_mod._evaluate_rank = _mock_evaluate
    try:
        calib_dir = _make_calib_dir(tmp_path, d_model=64, n_heads=4)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = search_rank_per_layer(
                calib_dir=calib_dir,
                rank_candidates=[4, 8, 16],
                n_steps=2,
                device="cpu",
                selection_mode="constraint",
                relative_tolerance=0.20,
            )

        # All candidates have same best_calib_total → all pass → pick smallest rank_cost
        assert result[0] == (4, 4), f"Expected (4,4), got {result[0]}"
    finally:
        rs_mod._evaluate_rank = original


# ── 5. Projector save/load roundtrip ────────────────────────────────


def test_projector_save_load_roundtrip(tmp_path):
    d_model, rank_k, rank_v, n_heads = 64, 8, 6, 4
    trainer = ProjectorTrainer(d_model, rank_k, rank_v, n_heads, device="cpu")
    q = torch.randn(1, 4, d_model)
    k = torch.randn(1, 4, d_model)
    v = torch.randn(1, 4, d_model)

    result = trainer.train_one_group(
        q, k, v,
        n_steps=10,
        warmup_steps=2,
        eval_every=5,
        seed=0,
    )

    layer_idx = 0
    ProjectorTrainer.save_result(result, layer_idx, str(tmp_path / "proj"))

    pt_path = tmp_path / "proj" / f"layer_{layer_idx}" / "projector.pt"
    assert pt_path.exists(), "projector.pt not saved"

    loaded = torch.load(pt_path, map_location="cpu", weights_only=False)

    assert torch.allclose(result["p_k"], loaded["p_k"], atol=1e-6), "p_k mismatch after roundtrip"
    assert torch.allclose(result["p_v"], loaded["p_v"], atol=1e-6), "p_v mismatch after roundtrip"
    assert torch.allclose(
        result["gamma"].float() if isinstance(result["gamma"], torch.Tensor) else torch.tensor(result["gamma"]),
        loaded["gamma"].float(),
        atol=1e-6,
    ), "gamma mismatch after roundtrip"
