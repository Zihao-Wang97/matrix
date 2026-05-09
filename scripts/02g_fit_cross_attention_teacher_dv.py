#!/usr/bin/env python
"""Fit D_V with a cross-attention teacher target.

This script intentionally leaves ``02f_fit_output_aware_dv.py`` untouched.
It reuses the same post-training setup but changes the closed-form target:

  Z = alpha_compressed @ QDQ(V P_V)
  U = alpha_full @ V

and solves:

  min_D || U - Z D ||^2 + lambda ||D||_F^2

The saved projector files still contain the same optional ``d_v`` tensor used
by runtime. Existing inference code needs no additional change.
"""

from __future__ import annotations

import argparse
import importlib.util
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch

from hawp_laq.config import load_config
from hawp_laq.offline.attention_distill_trainer import _projector_basis
from hawp_laq.offline.low_rank_attention_optimizer_torch import (
    low_rank_logit_scale_denominator,
    stable_softmax,
)
from hawp_laq.offline.projector_trainer import ProjectorTrainer
from hawp_laq.offline.rank_search import infer_calib_dims
from hawp_laq.runtime.projector_bank import load_ranks, normalize_projector_data, rebuild_ranks_json
from hawp_laq.utils.io import load_pt, save_json, save_pt


def _load_base_02f():
    path = Path(__file__).with_name("02f_fit_output_aware_dv.py")
    spec = importlib.util.spec_from_file_location("_hawp_dv_02f", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load helper script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_base = _load_base_02f()


def _get_arg(obj, key: str):
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key)


def _apply_overrides(cfg, args_or_overrides) -> None:
    _base._apply_overrides(cfg, args_or_overrides)
    teacher = _get_arg(args_or_overrides, "teacher")
    if teacher is not None:
        cfg.dv_output_aware.teacher = teacher
    full_teacher_weight = _get_arg(args_or_overrides, "full_teacher_weight")
    if full_teacher_weight is not None:
        cfg.dv_output_aware.full_teacher_weight = float(full_teacher_weight)


def _default_cross_output_dir(output_dir: Path) -> Path:
    output_dir = Path(output_dir)
    if output_dir.name.endswith("_cross"):
        return output_dir
    return output_dir.with_name(output_dir.name + "_cross")


def _full_teacher_weight(cfg) -> float:
    return float(getattr(cfg.dv_output_aware, "full_teacher_weight", 1.0))


def _validate_teacher(cfg) -> str:
    teacher = getattr(cfg.dv_output_aware, "teacher", "cross")
    valid = {"cross", "mixed_cross"}
    if teacher not in valid:
        raise ValueError(f"02g supports teacher in {sorted(valid)}, got {teacher!r}")
    if teacher == "mixed_cross" and _full_teacher_weight(cfg) < 0:
        raise ValueError("full_teacher_weight must be non-negative")
    return teacher


def _compressed_attention(
    q_b: torch.Tensor,
    k_lat: torch.Tensor,
    p_k: torch.Tensor,
    logit_scale: torch.Tensor,
    row_start: int,
    row_end: int,
    seq_len: int,
    device: torch.device,
) -> torch.Tensor:
    additive, valid = _base._make_causal_mask(row_start, row_end, seq_len, device)
    q_lat = q_b[:, row_start:row_end, :] @ p_k
    logits = (q_lat @ k_lat.transpose(-1, -2)) * logit_scale
    return stable_softmax(logits.float(), additive, valid)


def _full_attention(
    q_b: torch.Tensor,
    k_b: torch.Tensor,
    row_start: int,
    row_end: int,
    seq_len: int,
    head_dim: int,
    device: torch.device,
) -> torch.Tensor:
    additive, valid = _base._make_causal_mask(row_start, row_end, seq_len, device)
    logits = (q_b[:, row_start:row_end, :] @ k_b.transpose(-1, -2)) / (head_dim ** 0.5)
    return stable_softmax(logits.float(), additive, valid)


def _select_target(cfg, u_comp: torch.Tensor, u_full: torch.Tensor) -> torch.Tensor:
    teacher = _validate_teacher(cfg)
    if teacher == "cross":
        return u_full
    weight = _full_teacher_weight(cfg)
    return (u_comp + weight * u_full) / (1.0 + weight)


def _accumulate_layer_statistics(
    cfg,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    p_k: torch.Tensor,
    p_v: torch.Tensor,
    gamma: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    num_items, seq_len, head_dim = q.shape
    r_k = p_k.shape[1]
    r_v = p_v.shape[1]
    sample_batch_size = getattr(cfg.dv_output_aware, "sample_batch_size", 16)
    row_batch_size = max(1, int(getattr(cfg.dv_output_aware, "row_batch_size", 256)))
    denom = low_rank_logit_scale_denominator(cfg.hawp.logit_scale_mode, head_dim, r_k)
    logit_scale = gamma / float(denom)

    s_zz = torch.zeros(r_v, r_v, device=device, dtype=torch.float64)
    s_zu = torch.zeros(r_v, head_dim, device=device, dtype=torch.float64)
    total_rows = 0

    with torch.inference_mode():
        for batch_start, batch_end in _base._iter_batches(num_items, sample_batch_size):
            q_b = q[batch_start:batch_end].to(device=device, dtype=torch.float32, non_blocking=True)
            k_b = k[batch_start:batch_end].to(device=device, dtype=torch.float32, non_blocking=True)
            v_b = v[batch_start:batch_end].to(device=device, dtype=torch.float32, non_blocking=True)
            k_lat = k_b @ p_k
            v_lat = v_b @ p_v
            v_lat_q = _base._quantize_dequantize_v_latent(cfg, v_lat, r_v, device)

            for row_start in range(0, seq_len, row_batch_size):
                row_end = min(row_start + row_batch_size, seq_len)
                attn_c = _compressed_attention(q_b, k_lat, p_k, logit_scale, row_start, row_end, seq_len, device)
                attn_f = _full_attention(q_b, k_b, row_start, row_end, seq_len, head_dim, device)

                z = attn_c @ v_lat_q
                z = _base._apply_current_token_fp_correction(
                    cfg, z, attn_c, v_lat, v_lat_q, row_start, row_end
                )
                u_comp = attn_c @ v_b
                u_full = attn_f @ v_b
                target = _select_target(cfg, u_comp, u_full)

                zf = z.reshape(-1, r_v).float()
                uf = target.reshape(-1, head_dim).float()
                s_zz += (zf.transpose(0, 1) @ zf).double()
                s_zu += (zf.transpose(0, 1) @ uf).double()
                total_rows += zf.shape[0]

            del q_b, k_b, v_b, k_lat, v_lat, v_lat_q

    return s_zz, s_zu, total_rows


def _evaluate_layer_decoder(
    cfg,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    p_k: torch.Tensor,
    p_v: torch.Tensor,
    d_v: torch.Tensor,
    gamma: torch.Tensor,
    device: torch.device,
) -> dict:
    num_items, seq_len, head_dim = q.shape
    r_k = p_k.shape[1]
    r_v = p_v.shape[1]
    sample_batch_size = getattr(cfg.dv_output_aware, "sample_batch_size", 16)
    row_batch_size = max(1, int(getattr(cfg.dv_output_aware, "row_batch_size", 256)))
    eval_max_batches = getattr(cfg.dv_output_aware, "eval_max_batches", 8)
    denom = low_rank_logit_scale_denominator(cfg.hawp.logit_scale_mode, head_dim, r_k)
    logit_scale = gamma / float(denom)

    stats = {
        "full_teacher_sq": 0.0,
        "full_pv_err": 0.0,
        "full_dv_err": 0.0,
        "comp_teacher_sq": 0.0,
        "comp_pv_err": 0.0,
        "comp_dv_err": 0.0,
        "target_teacher_sq": 0.0,
        "target_pv_err": 0.0,
        "target_dv_err": 0.0,
    }
    count = 0
    batches = 0

    with torch.inference_mode():
        for batch_start, batch_end in _base._iter_batches(num_items, sample_batch_size):
            if eval_max_batches is not None and batches >= int(eval_max_batches):
                break
            q_b = q[batch_start:batch_end].to(device=device, dtype=torch.float32, non_blocking=True)
            k_b = k[batch_start:batch_end].to(device=device, dtype=torch.float32, non_blocking=True)
            v_b = v[batch_start:batch_end].to(device=device, dtype=torch.float32, non_blocking=True)
            k_lat = k_b @ p_k
            v_lat = v_b @ p_v
            v_lat_q = _base._quantize_dequantize_v_latent(cfg, v_lat, r_v, device)

            for row_start in range(0, seq_len, row_batch_size):
                row_end = min(row_start + row_batch_size, seq_len)
                attn_c = _compressed_attention(q_b, k_lat, p_k, logit_scale, row_start, row_end, seq_len, device)
                attn_f = _full_attention(q_b, k_b, row_start, row_end, seq_len, head_dim, device)

                z = attn_c @ v_lat_q
                z = _base._apply_current_token_fp_correction(
                    cfg, z, attn_c, v_lat, v_lat_q, row_start, row_end
                )
                baseline = z @ p_v.transpose(0, 1)
                decoded = z @ d_v
                teacher_comp = attn_c @ v_b
                teacher_full = attn_f @ v_b
                teacher_target = _select_target(cfg, teacher_comp, teacher_full)

                for prefix, teacher in (
                    ("full", teacher_full),
                    ("comp", teacher_comp),
                    ("target", teacher_target),
                ):
                    stats[f"{prefix}_teacher_sq"] += float((teacher.float() ** 2).sum().detach().cpu())
                    stats[f"{prefix}_pv_err"] += float(((baseline - teacher).float() ** 2).sum().detach().cpu())
                    stats[f"{prefix}_dv_err"] += float(((decoded - teacher).float() ** 2).sum().detach().cpu())
                count += teacher_full.numel()

            batches += 1
            del q_b, k_b, v_b, k_lat, v_lat, v_lat_q

    metrics = {"eval_batches": batches, "eval_elements": count}
    for prefix in ("full", "comp", "target"):
        denom_sq = max(stats[f"{prefix}_teacher_sq"], 1e-12)
        pv_norm = stats[f"{prefix}_pv_err"] / denom_sq
        dv_norm = stats[f"{prefix}_dv_err"] / denom_sq
        metrics[f"{prefix}_baseline_pv_mse"] = stats[f"{prefix}_pv_err"] / max(count, 1)
        metrics[f"{prefix}_dv_mse"] = stats[f"{prefix}_dv_err"] / max(count, 1)
        metrics[f"{prefix}_baseline_pv_normalized"] = pv_norm
        metrics[f"{prefix}_dv_normalized"] = dv_norm
        metrics[f"{prefix}_relative_improvement"] = 0.0 if pv_norm <= 0 else 1.0 - dv_norm / pv_norm
    return metrics


def _fit_dv_layer(cfg, layer_idx: int, device: torch.device, ranks_per_layer: dict[int, tuple[int, int]]) -> dict:
    teacher = _validate_teacher(cfg)
    calib_path = Path(cfg.calib.output_dir) / f"layer_{layer_idx}.pt"
    projector_path = Path(cfg.dv_output_aware.input_dir) / f"layer_{layer_idx}" / "projector.pt"
    if not calib_path.exists():
        return {"layer": layer_idx, "status": "skipped", "reason": f"missing {calib_path}"}
    if not projector_path.exists():
        return {"layer": layer_idx, "status": "skipped", "reason": f"missing {projector_path}"}

    meta = _base._load_meta(cfg)
    n_heads = _base._load_n_heads(cfg, meta)
    layer_data = load_pt(calib_path)
    q_raw = layer_data["q"].float()
    k_raw = layer_data["k"].float()
    v_raw = layer_data["v"].float()
    d_model, head_dim = infer_calib_dims(q_raw, n_heads, meta)
    q, _ = ProjectorTrainer._to_optim_input(q_raw, n_heads, d_model, head_dim)
    k, _ = ProjectorTrainer._to_optim_input(k_raw, n_heads, d_model, head_dim)
    v, _ = ProjectorTrainer._to_optim_input(v_raw, n_heads, d_model, head_dim)

    projector_data = normalize_projector_data(load_pt(projector_path), layer_idx)
    fallback_r_k, fallback_r_v = ranks_per_layer.get(layer_idx, (cfg.projector.r_k, cfg.projector.r_v))
    r_k = int(projector_data.get("r_k") or fallback_r_k or head_dim)
    r_v = int(projector_data.get("r_v") or fallback_r_v or head_dim)
    projector_data["r_k"] = r_k
    projector_data["r_v"] = r_v

    p_k = _projector_basis(projector_data, "p_k", head_dim, r_k).to(device=device, dtype=torch.float32)
    p_v = _projector_basis(projector_data, "p_v", head_dim, r_v).to(device=device, dtype=torch.float32)
    gamma = _base._effective_gamma(cfg, projector_data, device)

    print(
        f"[dv-cross] layer {layer_idx}: r_k={r_k} r_v={r_v} "
        f"items={q.shape[0]} seq={q.shape[1]} teacher={teacher} device={device}"
    )
    s_zz, s_zu, total_rows = _accumulate_layer_statistics(cfg, q, k, v, p_k, p_v, gamma, device)
    d_v = _base._solve_dv(s_zz, s_zu, float(cfg.dv_output_aware.lambda_ridge))
    metrics = _evaluate_layer_decoder(cfg, q, k, v, p_k, p_v, d_v, gamma, device)
    metrics.update(
        {
            "layer": layer_idx,
            "status": "ok",
            "r_k": r_k,
            "r_v": r_v,
            "head_dim": head_dim,
            "fit_rows": total_rows,
            "teacher": teacher,
            "full_teacher_weight": _full_teacher_weight(cfg),
            "quant_aware": _base._uses_quant_aware(cfg),
            "current_token_fp": bool(getattr(cfg.dv_output_aware, "current_token_fp", True)),
            "lambda_ridge": float(cfg.dv_output_aware.lambda_ridge),
            "logit_scale_mode": cfg.hawp.logit_scale_mode,
        }
    )

    output_dir = Path(cfg.dv_output_aware.output_dir) / f"layer_{layer_idx}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_data = dict(projector_data)
    output_data["d_v"] = d_v.detach().cpu()
    output_data["dv_output_aware"] = {
        "teacher": teacher,
        "full_teacher_weight": metrics["full_teacher_weight"],
        "quant_aware": metrics["quant_aware"],
        "current_token_fp": metrics["current_token_fp"],
        "lambda_ridge": metrics["lambda_ridge"],
        "full_dv_normalized": metrics["full_dv_normalized"],
        "comp_dv_normalized": metrics["comp_dv_normalized"],
        "target_dv_normalized": metrics["target_dv_normalized"],
    }
    save_pt(output_data, output_dir / "projector.pt")
    save_json(metrics, output_dir / "dv_cross_fit_metrics.json")
    print(
        f"[dv-cross] layer {layer_idx}: "
        f"full_pv={metrics['full_baseline_pv_normalized']:.6e} "
        f"full_dv={metrics['full_dv_normalized']:.6e} "
        f"full_imp={metrics['full_relative_improvement']:.2%} | "
        f"comp_pv={metrics['comp_baseline_pv_normalized']:.6e} "
        f"comp_dv={metrics['comp_dv_normalized']:.6e} "
        f"comp_imp={metrics['comp_relative_improvement']:.2%}"
    )
    return metrics


def _dv_worker(config_path: str, layers: list[int], device: str, overrides: dict) -> list[dict]:
    cfg = load_config(config_path)
    _apply_overrides(cfg, overrides)
    torch_device = _base._set_worker_device(cfg, device)
    ranks_per_layer = load_ranks(cfg.dv_output_aware.input_dir)
    print(f"[worker:{device}] dv-cross layers={layers}")
    return [_fit_dv_layer(cfg, layer_idx, torch_device, ranks_per_layer) for layer_idx in layers]


def _run(args) -> None:
    cfg = load_config(args.config)
    _apply_overrides(cfg, args)
    if args.output_dir is None:
        cfg.dv_output_aware.output_dir = _default_cross_output_dir(Path(cfg.dv_output_aware.output_dir))
    layers = _base._discover_layers(cfg, args.layers)
    devices = _base._normalize_devices(cfg.train.device, args.gpus)
    workers = args.workers or min(len(devices), len(layers))
    workers = max(1, min(workers, len(layers)))
    shards = [s for s in _base._split_round_robin(layers, workers) if s]
    output_dir = Path(cfg.dv_output_aware.output_dir)

    print("=" * 60)
    print("[dv-cross] cross-attention teacher closed-form D_V fit")
    print(f"[dv-cross] input_dir={Path(cfg.dv_output_aware.input_dir)}")
    print(f"[dv-cross] output_dir={output_dir}")
    print(f"[dv-cross] layers={layers}")
    print(f"[dv-cross] workers={len(shards)} devices={devices}")
    print(
        f"[dv-cross] teacher={cfg.dv_output_aware.teacher} "
        f"full_teacher_weight={_full_teacher_weight(cfg)} "
        f"quant_aware={cfg.dv_output_aware.quant_aware} "
        f"lambda_ridge={cfg.dv_output_aware.lambda_ridge}"
    )
    print("=" * 60)

    if args.clean_output_dir and output_dir.exists():
        print(f"[dv-cross] --clean-output-dir: removing {output_dir}")
        import shutil
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _base._prime_output_dir(Path(cfg.dv_output_aware.input_dir), output_dir)

    overrides = {
        "input_dir": str(cfg.dv_output_aware.input_dir),
        "output_dir": str(cfg.dv_output_aware.output_dir),
        "lambda_ridge": cfg.dv_output_aware.lambda_ridge,
        "sample_batch_size": cfg.dv_output_aware.sample_batch_size,
        "row_batch_size": cfg.dv_output_aware.row_batch_size,
        "quant_aware": cfg.dv_output_aware.quant_aware,
        "teacher": cfg.dv_output_aware.teacher,
        "full_teacher_weight": _full_teacher_weight(cfg),
    }

    results: list[dict] = []
    if len(shards) == 1:
        results.extend(_dv_worker(args.config, shards[0], devices[0], overrides))
    else:
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=len(shards), mp_context=context) as executor:
            futures = []
            for idx, shard in enumerate(shards):
                device = devices[idx % len(devices)]
                futures.append(executor.submit(_dv_worker, args.config, shard, device, overrides))
            for future in as_completed(futures):
                results.extend(future.result())

    ranks_path = rebuild_ranks_json(output_dir)
    ok = sorted(int(r["layer"]) for r in results if r.get("status") == "ok")
    skipped = [r for r in results if r.get("status") != "ok"]
    if ok:
        full_dv = [float(r["full_dv_normalized"]) for r in results if r.get("status") == "ok"]
        comp_dv = [float(r["comp_dv_normalized"]) for r in results if r.get("status") == "ok"]
        target_dv = [float(r["target_dv_normalized"]) for r in results if r.get("status") == "ok"]
        print(f"\n[dv-cross] fitted layers={ok}")
        print(
            f"[dv-cross] avg full_dv={sum(full_dv) / len(full_dv):.6e} "
            f"avg comp_dv={sum(comp_dv) / len(comp_dv):.6e} "
            f"avg target_dv={sum(target_dv) / len(target_dv):.6e}"
        )
    if skipped:
        for item in skipped:
            print(f"[dv-cross] skipped layer {item.get('layer')}: {item.get('reason')}")
    print(f"[dv-cross] rebuilt ranks.json at {ranks_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit D_V with cross-attention teacher target",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("config", nargs="?", default=None, help="Path to yaml config")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel worker processes")
    parser.add_argument("--gpus", nargs="+", default=None, help="GPU ids/devices, e.g. 0 1 2 or cuda:0 cuda:1")
    parser.add_argument("--layers", nargs="+", type=int, default=None, help="Optional layer subset")
    parser.add_argument("--input-dir", default=None, help="Override dv_output_aware.input_dir")
    parser.add_argument("--output-dir", default=None, help="Override output dir; default appends _cross")
    parser.add_argument("--lambda-ridge", type=float, default=None, help="Override ridge regularization")
    parser.add_argument("--sample-batch-size", type=int, default=None, help="Override effective head batch size")
    parser.add_argument("--row-batch-size", type=int, default=None, help="Override row block size")
    parser.add_argument("--quant-aware", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--teacher", choices=["cross", "mixed_cross"], default="cross")
    parser.add_argument("--full-teacher-weight", type=float, default=None, help="Weight used by mixed_cross")
    parser.add_argument("--clean-output-dir", action="store_true", default=False)
    args = parser.parse_args()

    if args.config is None:
        args.config = str(Path(__file__).resolve().parent.parent / "configs" / "dev_local.yaml")
    else:
        args.config = str(Path(args.config).resolve())
    _run(args)


if __name__ == "__main__":
    main()
