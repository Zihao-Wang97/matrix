#!/usr/bin/env python
"""Parallel projector training.

This script is a parallel companion to ``scripts/02_train_projectors.py``.
It keeps the same artifact layout:

  rank_search -> artifacts/ranks/layer_{i}_rank_search.json
                 artifacts/ranks/selected_ranks.json

  all         -> artifacts/projectors/layer_{i}/projector.pt
                 artifacts/projectors/ranks.json

Typical server usage:

  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    python -u scripts/02_train_projectors_parallel.py configs/server_standard.yaml \
    --mode rank_search --workers 8

  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    python -u scripts/02_train_projectors_parallel.py configs/server_standard.yaml \
    --mode all --workers 8
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import torch
from transformers import AutoConfig

from hawp_laq.config import load_config, resolve_projector_ranks
from hawp_laq.offline.projector_trainer import ProjectorTrainer
from hawp_laq.offline.rank_search import (
    _component_pass,
    _evaluate_rank,
    _print_header,
    _print_row,
    _signal_normalized_pass,
    build_rank_pairs,
    compute_signal_scales,
    get_layer_rank_floor,
    get_layer_tolerance_scale,
    infer_calib_dims,
)
from hawp_laq.runtime.projector_bank import load_ranks, rebuild_ranks_json
from hawp_laq.utils.io import load_pt, save_json


def _projector_train_kwargs(cfg) -> dict:
    return {
        "n_steps": cfg.projector.n_steps,
        "warmup_steps": cfg.projector.warmup_steps,
        "row_batch_size": cfg.projector.row_batch_size,
        "lr_pk": cfg.projector.lr_pk,
        "lr_pv": cfg.projector.lr_pv,
        "lr_xi": cfg.projector.lr_xi,
        "beta1": cfg.projector.beta1,
        "beta2": cfg.projector.beta2,
        "grad_clip": cfg.projector.grad_clip,
        "lambda_z": cfg.projector.lambda_z,
        "lambda_o": cfg.projector.lambda_o,
        "lambda_v": cfg.projector.lambda_v,
        "lambda_topk": cfg.projector.lambda_topk,
        "lambda_kl": cfg.projector.lambda_kl,
        "lambda_logit_topm": cfg.projector.lambda_logit_topm,
        "topk_k": cfg.projector.topk_k,
        "hard_neg_m": cfg.projector.hard_neg_m,
        "kl_top_m": cfg.projector.kl_top_m,
        "topk_margin": cfg.projector.topk_margin,
        "topk_loss_start_after_warmup": cfg.projector.topk_loss_start_after_warmup,
        "topk_metric_ks": cfg.projector.topk_metric_ks,
        "eval_every": cfg.projector.eval_every,
        "early_stopping": cfg.projector.early_stopping,
        "patience": cfg.projector.patience,
        "min_delta": cfg.projector.min_delta,
        "min_delta_mode": cfg.projector.min_delta_mode,
        "gamma_min": cfg.projector.gamma_min,
        "logit_scale_mode": cfg.hawp.logit_scale_mode,
        "eps_loss": cfg.projector.eps_loss,
        "adam_eps": cfg.projector.adam_eps,
        "optimizer": cfg.projector.optimizer,
    }


def _apply_overrides(cfg, overrides: dict) -> None:
    if overrides.get("ranks") is not None:
        cfg.rank_search.rank_candidates = overrides["ranks"]
        cfg.rank_search.r_k_candidates = None
        cfg.rank_search.r_v_candidates = None
        cfg.rank_search.rank_pair_candidates = None
    if overrides.get("relative_tolerance") is not None:
        cfg.rank_search.relative_tolerance = overrides["relative_tolerance"]
    if overrides.get("logits_abs_tolerance") is not None:
        cfg.rank_search.logits_abs_tolerance = overrides["logits_abs_tolerance"]
    if overrides.get("attn_abs_tolerance") is not None:
        cfg.rank_search.attn_abs_tolerance = overrides["attn_abs_tolerance"]
    if overrides.get("value_abs_tolerance") is not None:
        cfg.rank_search.value_abs_tolerance = overrides["value_abs_tolerance"]


def _set_worker_device(cfg, device: str) -> None:
    cfg.train.device = device
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.set_device(torch.device(device))


def _load_n_heads(cfg, meta: dict) -> int:
    n_heads = meta.get("n_heads")
    if n_heads is not None:
        return int(n_heads)
    model_cfg = AutoConfig.from_pretrained(
        cfg.model.model_id,
        local_files_only=Path(cfg.model.model_id).expanduser().is_dir(),
    )
    return int(model_cfg.num_attention_heads)


def _load_meta(cfg) -> dict:
    return load_pt(Path(cfg.calib.output_dir) / "meta.pt")


def _discover_layers(cfg, requested_layers: list[int] | None) -> list[int]:
    if requested_layers:
        return sorted(dict.fromkeys(int(x) for x in requested_layers))

    calib_dir = Path(cfg.calib.output_dir)
    meta = _load_meta(cfg)
    n_layers = int(meta.get("n_layers", 0) or 0)
    if n_layers > 0:
        return list(range(n_layers))

    layers = []
    for path in sorted(calib_dir.glob("layer_*.pt")):
        layers.append(int(path.stem.split("_", 1)[1]))
    if not layers:
        raise FileNotFoundError(f"No layer_*.pt files found in {calib_dir}")
    return layers


def _first_existing_layer_path(calib_dir: Path, layers: Iterable[int] | None = None) -> Path:
    if layers is not None:
        for layer_idx in layers:
            path = calib_dir / f"layer_{layer_idx}.pt"
            if path.exists():
                return path
    for path in sorted(calib_dir.glob("layer_*.pt")):
        return path
    raise FileNotFoundError(f"No layer_*.pt files found in {calib_dir}")


def _build_rank_pairs_from_config(cfg, head_dim: int) -> list[tuple[int, int]]:
    return build_rank_pairs(
        rank_candidates=getattr(cfg.rank_search, "rank_candidates", None),
        r_k_candidates=getattr(cfg.rank_search, "r_k_candidates", None),
        r_v_candidates=getattr(cfg.rank_search, "r_v_candidates", None),
        rank_pair_candidates=getattr(cfg.rank_search, "rank_pair_candidates", None),
        head_dim=head_dim,
    )


def _rank_search_layers(cfg, layers: list[int]) -> dict[int, tuple[int, int]]:
    calib_dir = Path(cfg.calib.output_dir)
    meta = _load_meta(cfg)
    n_heads = _load_n_heads(cfg, meta)

    sample = load_pt(_first_existing_layer_path(calib_dir, layers))
    d_model, head_dim = infer_calib_dims(sample["q"], n_heads, meta)
    rank_pairs = _build_rank_pairs_from_config(cfg, head_dim)
    if not rank_pairs:
        raise ValueError("rank_pairs is empty; nothing to search")

    output_dir = Path(cfg.rank_search.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    n_steps = getattr(cfg.rank_search, "n_steps", None)
    if n_steps is None:
        n_steps = cfg.projector.n_steps

    selection_mode = getattr(cfg.rank_search, "selection_mode", "constraint")
    valid_modes = {"constraint", "signal_normalized", "total", "logits", "attn_value_abs"}
    if selection_mode not in valid_modes:
        raise ValueError(f"Unknown selection_mode={selection_mode!r}. Supported: {sorted(valid_modes)}")

    selected_ranks: dict[int, tuple[int, int]] = {}

    for layer_idx in layers:
        layer_path = calib_dir / f"layer_{layer_idx}.pt"
        if not layer_path.exists():
            print(f"[rank_search] layer {layer_idx}: calib data not found, skipping")
            continue

        data = load_pt(layer_path)
        q = data["q"].float()
        k = data["k"].float()
        v = data["v"].float()
        d_model, head_dim = infer_calib_dims(q, n_heads, meta)

        layer_rank_pairs = list(rank_pairs)
        layer_scale = get_layer_tolerance_scale(layer_idx, cfg.rank_search.layer_tolerance_scale)
        min_r_k, min_r_v = get_layer_rank_floor(layer_idx, cfg.rank_search.layer_rank_floor)
        if min_r_k > 1 or min_r_v > 1:
            before = len(layer_rank_pairs)
            layer_rank_pairs = [
                (rk, rv)
                for rk, rv in layer_rank_pairs
                if rk >= min_r_k and rv >= min_r_v
            ]
            if not layer_rank_pairs:
                raise ValueError(
                    f"[rank_search] layer {layer_idx}: all candidates filtered out by "
                    f"rank floor (min_r_k={min_r_k}, min_r_v={min_r_v})"
                )
            if len(layer_rank_pairs) < before:
                print(
                    f"[rank_search] layer {layer_idx}: rank floor "
                    f"(min_r_k={min_r_k}, min_r_v={min_r_v}) "
                    f"filtered {before} -> {len(layer_rank_pairs)} candidates"
                )

        print(f"\n[rank_search] layer {layer_idx}: d_model={d_model} n_heads={n_heads} head_dim={head_dim}")

        for rk, rv in layer_rank_pairs:
            if not (1 <= rk <= head_dim):
                raise ValueError(f"[rank_search] layer {layer_idx}: r_k={rk} out of range [1, head_dim={head_dim}]")
            if not (1 <= rv <= head_dim):
                raise ValueError(f"[rank_search] layer {layer_idx}: r_v={rv} out of range [1, head_dim={head_dim}]")

        mode_label = f"selection={selection_mode}"
        if selection_mode == "signal_normalized":
            mode_label += f" layer_scale={layer_scale}"
            if min_r_k > 1 or min_r_v > 1:
                mode_label += f" floor=({min_r_k},{min_r_v})"

        print(
            f"[rank_search] candidates={layer_rank_pairs}  n_steps={n_steps}  "
            f"{mode_label}  rel_tol={cfg.rank_search.relative_tolerance}  "
            f"logits_abs_tol={cfg.rank_search.logits_abs_tolerance}  "
            f"attn_abs_tol={cfg.rank_search.attn_abs_tolerance}  "
            f"value_abs_tol={cfg.rank_search.value_abs_tolerance}"
        )

        signal_scales = None
        if selection_mode == "signal_normalized":
            signal_scales = compute_signal_scales(
                q, k, v,
                n_heads=n_heads,
                d_model=d_model,
                head_dim=head_dim,
                row_batch_size=getattr(cfg.projector, "row_batch_size", None),
            )
            print(
                f"[rank_search] signal logits={signal_scales['signal_logits']:.6f}"
                f" attn={signal_scales['signal_attn']:.6f}"
                f" value={signal_scales['signal_value']:.6f}"
            )

        results = []
        for rk, rv in layer_rank_pairs:
            r = _evaluate_rank(
                q,
                k,
                v,
                rk,
                rv,
                d_model,
                n_heads,
                n_steps=n_steps,
                lr=cfg.projector.lr,
                orthogonalize_every=cfg.projector.orthogonalize_every,
                w_logits=cfg.projector.w_logits,
                w_attn=cfg.projector.w_attn,
                w_value=cfg.projector.w_value,
                device=cfg.train.device,
                warmup_steps=getattr(cfg.projector, "warmup_steps", 30),
                row_batch_size=getattr(cfg.projector, "row_batch_size", None),
                lr_pk=getattr(cfg.projector, "lr_pk", 5e-3),
                lr_pv=getattr(cfg.projector, "lr_pv", 5e-3),
                lr_xi=getattr(cfg.projector, "lr_xi", 1e-2),
                beta1=getattr(cfg.projector, "beta1", 0.9),
                beta2=getattr(cfg.projector, "beta2", 0.99),
                grad_clip=getattr(cfg.projector, "grad_clip", 1.0),
                lambda_z=getattr(cfg.projector, "lambda_z", 1.0),
                lambda_o=getattr(cfg.projector, "lambda_o", 2.0),
                lambda_v=getattr(cfg.projector, "lambda_v", 0.05),
                lambda_topk=getattr(cfg.projector, "lambda_topk", 0.0),
                lambda_kl=getattr(cfg.projector, "lambda_kl", 0.0),
                lambda_logit_topm=getattr(cfg.projector, "lambda_logit_topm", 0.0),
                topk_k=getattr(cfg.projector, "topk_k", 8),
                hard_neg_m=getattr(cfg.projector, "hard_neg_m", 32),
                kl_top_m=getattr(cfg.projector, "kl_top_m", 64),
                topk_margin=getattr(cfg.projector, "topk_margin", 0.05),
                topk_loss_start_after_warmup=getattr(cfg.projector, "topk_loss_start_after_warmup", True),
                topk_metric_ks=getattr(cfg.projector, "topk_metric_ks", (5, 10)),
                eval_every=getattr(cfg.projector, "eval_every", 50),
                early_stopping=getattr(cfg.projector, "early_stopping", True),
                patience=getattr(cfg.projector, "patience", 5),
                min_delta=getattr(cfg.projector, "min_delta", 1e-4),
                min_delta_mode=getattr(cfg.projector, "min_delta_mode", "relative"),
                gamma_min=getattr(cfg.projector, "gamma_min", 1e-4),
                logit_scale_mode=getattr(cfg.hawp, "logit_scale_mode", "rk"),
                eps_loss=getattr(cfg.projector, "eps_loss", 1e-8),
                adam_eps=getattr(cfg.projector, "adam_eps", 1e-8),
                optimizer=getattr(cfg.projector, "optimizer", "riemannian_adam"),
            )
            results.append(r)
            print(
                f"  r_k={r['r_k']:>3d} r_v={r['r_v']:>3d}"
                f"  calib_total={r['best_calib_total']:.6f}"
                f"  calib_logits={r['best_calib_logits']:.6f}"
                f"  calib_attn={r['best_calib_attn']:.6f}"
                f"  calib_value={r['best_calib_value']:.6f}"
                f"  topk={r.get('best_calib_topk', 0.0):.6f}"
                f"  top10={r.get('best_calib_top_recalls', {}).get('top10_recall', 0.0):.4f}"
                f"  cost={r['rank_cost']}  step={r['best_step']} stopped={r['stopped_early']}"
            )

        if not results:
            continue

        if selection_mode == "signal_normalized":
            for r in results:
                _signal_normalized_pass(
                    r,
                    signal_scales,
                    cfg.rank_search.logits_signal_tolerance,
                    cfg.rank_search.attn_signal_tolerance,
                    cfg.rank_search.value_signal_tolerance,
                    layer_scale,
                )
        elif selection_mode == "constraint":
            baseline_result = min(results, key=lambda x: x["best_calib_total"])
            for r in results:
                r["logits_pass"] = _component_pass(
                    r["best_calib_logits"],
                    baseline_result["best_calib_logits"],
                    cfg.rank_search.relative_tolerance,
                    cfg.rank_search.logits_abs_tolerance,
                )
                r["attn_pass"] = _component_pass(
                    r["best_calib_attn"],
                    baseline_result["best_calib_attn"],
                    cfg.rank_search.relative_tolerance,
                    cfg.rank_search.attn_abs_tolerance,
                )
                r["value_pass"] = _component_pass(
                    r["best_calib_value"],
                    baseline_result["best_calib_value"],
                    cfg.rank_search.relative_tolerance,
                    cfg.rank_search.value_abs_tolerance,
                )
                r["all_pass"] = r["logits_pass"] and r["attn_pass"] and r["value_pass"]
        elif selection_mode == "total":
            best_total = min(r["best_calib_total"] for r in results)
            for r in results:
                r["total_pass"] = _component_pass(
                    r["best_calib_total"],
                    best_total,
                    cfg.rank_search.relative_tolerance,
                    best_total * cfg.rank_search.relative_tolerance + 1e-8,
                )
                r["all_pass"] = r["total_pass"]
        elif selection_mode == "logits":
            best_logits = min(r["best_calib_logits"] for r in results)
            for r in results:
                r["logits_pass"] = _component_pass(
                    r["best_calib_logits"],
                    best_logits,
                    cfg.rank_search.relative_tolerance,
                    cfg.rank_search.logits_abs_tolerance,
                )
                r["all_pass"] = r["logits_pass"]
        elif selection_mode == "attn_value_abs":
            for r in results:
                r["attn_pass"] = r["best_calib_attn"] <= cfg.rank_search.attn_abs_tolerance
                r["value_pass"] = r["best_calib_value"] <= cfg.rank_search.value_abs_tolerance
                r["all_pass"] = r["attn_pass"] and r["value_pass"]

        if selection_mode == "signal_normalized":
            print(
                f"\n  --- signal-normalized check "
                f"(logits_tol={cfg.rank_search.logits_signal_tolerance:.4f}*{layer_scale}="
                f"{cfg.rank_search.logits_signal_tolerance * layer_scale:.4f}, "
                f"attn_tol={cfg.rank_search.attn_signal_tolerance:.4f}*{layer_scale}="
                f"{cfg.rank_search.attn_signal_tolerance * layer_scale:.4f}, "
                f"value_tol={cfg.rank_search.value_signal_tolerance:.4f}*{layer_scale}="
                f"{cfg.rank_search.value_signal_tolerance * layer_scale:.4f}) ---"
            )
            _print_header("nlogits", "nattn", "nvalue")
            for r in results:
                _print_row(
                    r,
                    r.get("normalized_logits_error", 0),
                    r.get("normalized_attn_error", 0),
                    r.get("normalized_value_error", 0),
                )
        elif selection_mode == "attn_value_abs":
            print(
                f"\n  --- attn_value_abs check "
                f"(attn_tol={cfg.rank_search.attn_abs_tolerance:.1e}, "
                f"value_tol={cfg.rank_search.value_abs_tolerance:.1e}) ---"
            )
            _print_header("attn", "value", "cost")
            for r in results:
                _print_row(
                    r,
                    f"{'PASS' if r['attn_pass'] else 'FAIL':>10}",
                    f"{'PASS' if r['value_pass'] else 'FAIL':>10}",
                    r["rank_cost"],
                )
        else:
            print(f"\n  --- {selection_mode} check (rel_tol={cfg.rank_search.relative_tolerance}) ---")
            _print_header("logits", "attn", "value")
            for r in results:
                tag = "PASS" if r["all_pass"] else "REJECT"
                _print_row(r, f"{tag:>10}", "", f"{r['rank_cost']:>6d}")

        passing = [r for r in results if r["all_pass"]]
        if passing:
            passing.sort(key=lambda x: (x["rank_cost"], x["best_calib_total"]))
            best = passing[0]
        else:
            results.sort(key=lambda x: x["best_calib_total"])
            best = results[0]

        chosen_rk, chosen_rv = best["r_k"], best["r_v"]
        selected_ranks[layer_idx] = (chosen_rk, chosen_rv)
        print(
            f"[rank_search] layer {layer_idx}: selected r_k={chosen_rk} r_v={chosen_rv}"
            f"  cost={best['rank_cost']}  calib_total={best['best_calib_total']:.6f}"
            f"  best_step={best.get('best_step', '?')} stopped_early={best.get('stopped_early', '?')}"
        )

        norm_results = []
        for item in results:
            nr = dict(item)
            nr.setdefault("r_k", item.get("rank_k", item.get("r_k", 0)))
            nr.setdefault("r_v", item.get("rank_v", item.get("r_v", 0)))
            nr.setdefault("rank_cost", item.get("rank_cost", nr["r_k"] + nr["r_v"]))
            norm_results.append(nr)

        layer_json = {
            "layer_idx": layer_idx,
            "selected_r_k": chosen_rk,
            "selected_r_v": chosen_rv,
            "selection_method": selection_mode,
            "best_calib_total": best.get("best_calib_total", 0.0),
            "best_calib_logits": best.get("best_calib_logits", 0.0),
            "best_calib_attn": best.get("best_calib_attn", 0.0),
            "best_calib_value": best.get("best_calib_value", 0.0),
            "best_calib_topk": best.get("best_calib_topk", 0.0),
            "best_calib_kl_topm": best.get("best_calib_kl_topm", 0.0),
            "best_calib_logit_topm": best.get("best_calib_logit_topm", 0.0),
            "best_calib_top_recalls": best.get("best_calib_top_recalls", {}),
            "best_step": best.get("best_step", 0),
            "actual_steps": best.get("actual_steps", 0),
            "stopped_early": best.get("stopped_early", False),
            "relative_tolerance": cfg.rank_search.relative_tolerance,
            "logits_abs_tolerance": cfg.rank_search.logits_abs_tolerance,
            "attn_abs_tolerance": cfg.rank_search.attn_abs_tolerance,
            "value_abs_tolerance": cfg.rank_search.value_abs_tolerance,
            "candidates": layer_rank_pairs,
            "results": norm_results,
        }
        if selection_mode == "signal_normalized":
            layer_json["layer_tolerance_scale"] = layer_scale
            layer_json["rank_floor"] = {"min_r_k": min_r_k, "min_r_v": min_r_v}
            layer_json["signal_scales"] = signal_scales
            layer_json["logits_signal_tolerance"] = cfg.rank_search.logits_signal_tolerance
            layer_json["attn_signal_tolerance"] = cfg.rank_search.attn_signal_tolerance
            layer_json["value_signal_tolerance"] = cfg.rank_search.value_signal_tolerance
        save_json(layer_json, output_dir / f"layer_{layer_idx}_rank_search.json")

    return selected_ranks


def _projector_ranks_for_all(cfg, n_layers: int, n_heads: int, meta: dict) -> dict[int, tuple[int, int]]:
    output_dir = Path(cfg.projector.output_dir)
    selected_path = Path(cfg.rank_search.output_dir) / "selected_ranks.json"
    if selected_path.exists():
        ranks_per_layer = load_ranks(selected_path)
        print(f"[all] loaded ranks from {selected_path}")
        return ranks_per_layer

    ranks_per_layer = load_ranks(output_dir)
    if ranks_per_layer:
        print(f"[all] loaded ranks from {output_dir / 'ranks.json'}")
        return ranks_per_layer

    print("[all] WARNING: no ranks found; using projector.r_k / projector.r_v for all layers")
    calib_dir = Path(cfg.calib.output_dir)
    sample = load_pt(_first_existing_layer_path(calib_dir))
    sample_q = sample["q"].float()
    _, head_dim = infer_calib_dims(sample_q, n_heads, meta)
    r_k, r_v = resolve_projector_ranks(cfg.projector, head_dim=head_dim, mode="single_group")
    return {i: (r_k, r_v) for i in range(n_layers)}


def _train_projector_layers(cfg, layers: list[int]) -> list[int]:
    calib_dir = Path(cfg.calib.output_dir)
    output_dir = Path(cfg.projector.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    meta = _load_meta(cfg)
    n_layers = int(meta.get("n_layers", 0) or 0)
    if n_layers == 0:
        for p in sorted(calib_dir.glob("layer_*.pt")):
            n_layers = max(n_layers, int(p.stem.split("_", 1)[1]) + 1)
    n_heads = _load_n_heads(cfg, meta)
    ranks_per_layer = _projector_ranks_for_all(cfg, n_layers, n_heads, meta)
    train_kw = _projector_train_kwargs(cfg)

    trained_layers = []
    for layer_idx in layers:
        layer_path = calib_dir / f"layer_{layer_idx}.pt"
        if not layer_path.exists():
            print(f"[all] layer {layer_idx}: no calib data, skipping")
            continue

        r_k, r_v = ranks_per_layer.get(layer_idx, (None, None))
        if r_k is None or r_v is None:
            print(f"[all] layer {layer_idx}: no rank info, skipping")
            continue

        layer_data = load_pt(layer_path)
        q = layer_data["q"].float()
        k = layer_data["k"].float()
        v = layer_data["v"].float()
        d_model, _ = infer_calib_dims(q, n_heads, meta)

        print(f"\n[all] layer {layer_idx}: d_model={d_model} r_k={r_k} r_v={r_v} device={cfg.train.device}")

        trainer = ProjectorTrainer(
            d_model=d_model,
            rank_k=r_k,
            rank_v=r_v,
            n_heads=n_heads,
            lr=cfg.projector.lr,
            orthogonalize_every=cfg.projector.orthogonalize_every,
            w_logits=cfg.projector.w_logits,
            w_attn=cfg.projector.w_attn,
            w_value=cfg.projector.w_value,
            device=cfg.train.device,
        )
        result = trainer.train_one_group(q, k, v, **train_kw)
        ProjectorTrainer.save_result(result, layer_idx, str(output_dir))
        trained_layers.append(layer_idx)

    return trained_layers


def _rank_search_worker(config_path: str, layers: list[int], device: str, overrides: dict) -> dict[int, tuple[int, int]]:
    cfg = load_config(config_path)
    _apply_overrides(cfg, overrides)
    _set_worker_device(cfg, device)
    print(f"[worker:{device}] rank_search layers={layers}")
    return _rank_search_layers(cfg, layers)


def _all_worker(config_path: str, layers: list[int], device: str, overrides: dict) -> list[int]:
    cfg = load_config(config_path)
    _apply_overrides(cfg, overrides)
    _set_worker_device(cfg, device)
    print(f"[worker:{device}] all layers={layers}")
    return _train_projector_layers(cfg, layers)


def _normalize_devices(cfg_device: str, gpus: list[str] | None) -> list[str]:
    if gpus:
        devices = []
        for item in gpus:
            if item == "cpu" or item.startswith("cuda"):
                devices.append(item)
            else:
                devices.append(f"cuda:{item}")
        return devices

    if cfg_device.startswith("cuda") and torch.cuda.is_available():
        if cfg_device != "cuda":
            return [cfg_device]
        count = torch.cuda.device_count()
        if count > 0:
            return [f"cuda:{i}" for i in range(count)]

    return [cfg_device]


def _split_round_robin(items: list[int], n_shards: int) -> list[list[int]]:
    return [items[i::n_shards] for i in range(n_shards)]


def _run_parallel(mode: str, args, overrides: dict) -> None:
    cfg = load_config(args.config)
    layers = _discover_layers(cfg, args.layers)
    devices = _normalize_devices(cfg.train.device, args.gpus)
    workers = args.workers or min(len(devices), len(layers))
    workers = max(1, min(workers, len(layers)))
    shards = [s for s in _split_round_robin(layers, workers) if s]

    print("=" * 60)
    print(f"[parallel] mode={mode}")
    print(f"[parallel] layers={layers}")
    print(f"[parallel] workers={len(shards)} devices={devices}")
    print("=" * 60)

    if mode == "all" and args.clean_output_dir:
        output_dir = Path(cfg.projector.output_dir)
        if output_dir.exists():
            print(f"[all] --clean-output-dir: removing {output_dir}")
            shutil.rmtree(output_dir)

    context = mp.get_context("spawn")
    futures = []
    with ProcessPoolExecutor(max_workers=len(shards), mp_context=context) as executor:
        for idx, shard in enumerate(shards):
            device = devices[idx % len(devices)]
            if mode == "rank_search":
                futures.append(executor.submit(_rank_search_worker, args.config, shard, device, overrides))
            elif mode == "all":
                futures.append(executor.submit(_all_worker, args.config, shard, device, overrides))
            else:
                raise ValueError(f"Unsupported mode: {mode}")

        if mode == "rank_search":
            selected: dict[int, tuple[int, int]] = {}
            for future in as_completed(futures):
                selected.update(future.result())

            print(f"\n{'=' * 60}")
            print("[rank_search] summary")
            print(f"  {'layer':>6} {'r_k':>6} {'r_v':>6}")
            print(f"  {'-' * 20}")
            for idx in sorted(selected.keys()):
                rk, rv = selected[idx]
                print(f"  {idx:>6d} {rk:>6d} {rv:>6d}")

            out_dir = Path(cfg.rank_search.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            save_json(
                {"selected_ranks": {str(k): {"r_k": v[0], "r_v": v[1]} for k, v in sorted(selected.items())}},
                out_dir / "selected_ranks.json",
            )
            print(f"\n[rank_search] saved to {out_dir / 'selected_ranks.json'}")

        elif mode == "all":
            trained = []
            for future in as_completed(futures):
                trained.extend(future.result())
            trained = sorted(set(trained))
            ranks_path = rebuild_ranks_json(cfg.projector.output_dir)
            print(f"\n[all] trained layers={trained}")
            print(f"[all] rebuilt ranks.json at {ranks_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parallel HAWP-LAQ projector training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("config", nargs="?", default=None, help="Path to yaml config")
    parser.add_argument("--mode", choices=["rank_search", "all"], default="rank_search")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel worker processes")
    parser.add_argument("--gpus", nargs="+", default=None, help="GPU ids/devices, e.g. 0 1 2 or cuda:0 cuda:1")
    parser.add_argument("--layers", nargs="+", type=int, default=None, help="Optional layer subset")
    parser.add_argument("--ranks", nargs="+", type=int, default=None, help="Override rank candidates")
    parser.add_argument("--relative-tolerance", type=float, default=None)
    parser.add_argument("--logits-abs-tolerance", type=float, default=None)
    parser.add_argument("--attn-abs-tolerance", type=float, default=None)
    parser.add_argument("--value-abs-tolerance", type=float, default=None)
    parser.add_argument("--clean-output-dir", action="store_true", default=False)
    args = parser.parse_args()

    if args.config is None:
        args.config = str(Path(__file__).resolve().parent.parent / "configs" / "dev_local.yaml")
    else:
        args.config = str(Path(args.config).resolve())

    overrides = {
        "ranks": args.ranks,
        "relative_tolerance": args.relative_tolerance,
        "logits_abs_tolerance": args.logits_abs_tolerance,
        "attn_abs_tolerance": args.attn_abs_tolerance,
        "value_abs_tolerance": args.value_abs_tolerance,
    }
    _run_parallel(args.mode, args, overrides)


if __name__ == "__main__":
    main()
