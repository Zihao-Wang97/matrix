#!/usr/bin/env python
"""Fit output-aware D_V decoders for trained HAWP projectors.

This is a post-training refinement stage:

  1. Keep P_K, P_V, and gamma fixed.
  2. Build compressed-teacher attention from Q P_K and K P_K.
  3. Optionally quantize/dequantize V P_V with the configured V quantizer.
  4. Solve a ridge-regression decoder D_V in closed form so that
     (alpha @ quant(V P_V)) D_V approximates alpha @ V.

The resulting projector.pt files contain an optional ``d_v`` tensor. Runtime
falls back to ``P_V.T`` when this tensor is absent.
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

from hawp_laq.config import build_v_quantizer, load_config
from hawp_laq.offline.attention_distill_trainer import _gamma_from_data, _projector_basis
from hawp_laq.offline.low_rank_attention_optimizer_torch import (
    low_rank_logit_scale_denominator,
    stable_softmax,
)
from hawp_laq.offline.projector_trainer import ProjectorTrainer
from hawp_laq.offline.rank_search import infer_calib_dims
from hawp_laq.runtime.projector_bank import load_ranks, normalize_projector_data, rebuild_ranks_json
from hawp_laq.utils.io import load_pt, save_json, save_pt


def _set_worker_device(cfg, device: str) -> torch.device:
    cfg.train.device = device
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.set_device(torch_device)
    return torch_device


def _apply_overrides(cfg, args_or_overrides) -> None:
    get = args_or_overrides.get if isinstance(args_or_overrides, dict) else lambda key: getattr(args_or_overrides, key)
    input_dir = get("input_dir")
    output_dir = get("output_dir")
    if input_dir is not None:
        cfg.dv_output_aware.input_dir = Path(input_dir)
    if output_dir is not None:
        cfg.dv_output_aware.output_dir = Path(output_dir)
    lambda_ridge = get("lambda_ridge")
    if lambda_ridge is not None:
        cfg.dv_output_aware.lambda_ridge = float(lambda_ridge)
    sample_batch_size = get("sample_batch_size")
    if sample_batch_size is not None:
        cfg.dv_output_aware.sample_batch_size = sample_batch_size
    row_batch_size = get("row_batch_size")
    if row_batch_size is not None:
        cfg.dv_output_aware.row_batch_size = int(row_batch_size)
    quant_aware = get("quant_aware")
    if quant_aware is not None:
        cfg.dv_output_aware.quant_aware = bool(quant_aware)


def _load_meta(cfg) -> dict:
    return load_pt(Path(cfg.calib.output_dir) / "meta.pt")


def _load_n_heads(cfg, meta: dict) -> int:
    n_heads = meta.get("n_heads")
    if n_heads is not None:
        return int(n_heads)
    model_cfg = AutoConfig.from_pretrained(
        cfg.model.model_id,
        local_files_only=Path(cfg.model.model_id).expanduser().is_dir(),
    )
    return int(model_cfg.num_attention_heads)


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


def _same_path(left: Path, right: Path) -> bool:
    try:
        return left.resolve() == right.resolve()
    except OSError:
        return left == right


def _prime_output_dir(input_dir: Path, output_dir: Path) -> None:
    if _same_path(input_dir, output_dir):
        return
    if not input_dir.exists():
        raise FileNotFoundError(f"dv_output_aware.input_dir not found: {input_dir}")
    has_projectors = any(output_dir.glob("layer_*/projector.pt"))
    if not has_projectors:
        print(f"[dv] priming output_dir from {input_dir}")
        shutil.copytree(input_dir, output_dir, dirs_exist_ok=True)


def _iter_batches(num_items: int, batch_size: int | None) -> Iterable[tuple[int, int]]:
    if batch_size is None or batch_size <= 0 or batch_size >= num_items:
        yield 0, num_items
        return
    for start in range(0, num_items, batch_size):
        yield start, min(start + batch_size, num_items)


def _make_causal_mask(row_start: int, row_end: int, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    rows = torch.arange(row_start, row_end, device=device).unsqueeze(1)
    cols = torch.arange(seq_len, device=device).unsqueeze(0)
    valid = cols <= rows
    additive = torch.where(
        valid,
        torch.zeros_like(valid, dtype=torch.float32),
        torch.full_like(valid, -1e4, dtype=torch.float32),
    )
    return additive.unsqueeze(0), valid.unsqueeze(0)


def _effective_gamma(cfg, projector_data: dict, device: torch.device) -> torch.Tensor:
    gamma_mode = getattr(cfg.hawp, "gamma_mode", "learned")
    if gamma_mode == "learned":
        return _gamma_from_data(
            projector_data,
            gamma_min=float(getattr(cfg.projector, "gamma_min", 1e-4)),
            device=device,
            dtype=torch.float32,
        )
    if gamma_mode == "fixed":
        gamma_value = getattr(cfg.hawp, "gamma_value", None)
        if gamma_value is None:
            return _gamma_from_data(
                projector_data,
                gamma_min=float(getattr(cfg.projector, "gamma_min", 1e-4)),
                device=device,
                dtype=torch.float32,
            )
        return torch.as_tensor(float(gamma_value), device=device, dtype=torch.float32)
    return torch.ones((), device=device, dtype=torch.float32)


def _quantize_dequantize_v_latent(
    cfg,
    v_lat: torch.Tensor,
    r_v: int,
    device: torch.device,
) -> torch.Tensor:
    if not bool(getattr(cfg.dv_output_aware, "quant_aware", True)) or not bool(getattr(cfg.quant, "enabled", False)):
        return v_lat

    quantizer = build_v_quantizer(cfg, r_v=r_v, device=str(device))
    flat = v_lat.reshape(-1, r_v).contiguous()
    out = torch.empty_like(flat)
    chunk_rows = max(1, int(getattr(cfg.dv_output_aware, "quant_chunk_rows", 65536)))
    for start in range(0, flat.shape[0], chunk_rows):
        end = min(start + chunk_rows, flat.shape[0])
        chunk = flat[start:end]
        qx = quantizer.quantize(chunk)
        out[start:end].copy_(quantizer.dequantize(qx).to(device=flat.device, dtype=flat.dtype))
    return out.reshape_as(v_lat)


def _uses_quant_aware(cfg) -> bool:
    return bool(getattr(cfg.dv_output_aware, "quant_aware", True)) and bool(getattr(cfg.quant, "enabled", False))


def _apply_current_token_fp_correction(
    cfg,
    z: torch.Tensor,
    attn: torch.Tensor,
    v_lat: torch.Tensor,
    v_lat_q: torch.Tensor,
    row_start: int,
    row_end: int,
) -> torch.Tensor:
    if not _uses_quant_aware(cfg) or not bool(getattr(cfg.dv_output_aware, "current_token_fp", True)):
        return z
    cols = torch.arange(row_start, row_end, device=attn.device)
    diag_weight = attn.gather(
        dim=-1,
        index=cols.view(1, -1, 1).expand(attn.shape[0], -1, 1),
    )
    delta = v_lat[:, row_start:row_end, :] - v_lat_q[:, row_start:row_end, :]
    return z + diag_weight.to(dtype=delta.dtype) * delta


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
        for batch_start, batch_end in _iter_batches(num_items, sample_batch_size):
            q_b = q[batch_start:batch_end].to(device=device, dtype=torch.float32, non_blocking=True)
            k_b = k[batch_start:batch_end].to(device=device, dtype=torch.float32, non_blocking=True)
            v_b = v[batch_start:batch_end].to(device=device, dtype=torch.float32, non_blocking=True)
            k_lat = k_b @ p_k
            v_lat = v_b @ p_v
            v_lat_q = _quantize_dequantize_v_latent(cfg, v_lat, r_v, device)

            for row_start in range(0, seq_len, row_batch_size):
                row_end = min(row_start + row_batch_size, seq_len)
                additive, valid = _make_causal_mask(row_start, row_end, seq_len, device)
                q_lat = q_b[:, row_start:row_end, :] @ p_k
                logits = (q_lat @ k_lat.transpose(-1, -2)) * logit_scale
                attn = stable_softmax(logits.float(), additive, valid)
                u = attn @ v_b
                z = attn @ v_lat_q
                z = _apply_current_token_fp_correction(cfg, z, attn, v_lat, v_lat_q, row_start, row_end)

                zf = z.reshape(-1, r_v).float()
                uf = u.reshape(-1, head_dim).float()
                s_zz += (zf.transpose(0, 1) @ zf).double()
                s_zu += (zf.transpose(0, 1) @ uf).double()
                total_rows += zf.shape[0]

            del q_b, k_b, v_b, k_lat, v_lat, v_lat_q

    return s_zz, s_zu, total_rows


def _solve_dv(s_zz: torch.Tensor, s_zu: torch.Tensor, lambda_ridge: float) -> torch.Tensor:
    r_v = s_zz.shape[0]
    ridge = max(0.0, float(lambda_ridge))
    eye = torch.eye(r_v, device=s_zz.device, dtype=torch.float64)
    gram = s_zz + ridge * eye
    try:
        d_v = torch.linalg.solve(gram, s_zu)
    except RuntimeError:
        d_v = torch.linalg.pinv(gram) @ s_zu
    return d_v.float()


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

    teacher_sq = 0.0
    pv_err = 0.0
    dv_err = 0.0
    count = 0
    batches = 0

    with torch.inference_mode():
        for batch_start, batch_end in _iter_batches(num_items, sample_batch_size):
            if eval_max_batches is not None and batches >= int(eval_max_batches):
                break
            q_b = q[batch_start:batch_end].to(device=device, dtype=torch.float32, non_blocking=True)
            k_b = k[batch_start:batch_end].to(device=device, dtype=torch.float32, non_blocking=True)
            v_b = v[batch_start:batch_end].to(device=device, dtype=torch.float32, non_blocking=True)
            k_lat = k_b @ p_k
            v_lat = v_b @ p_v
            v_lat_q = _quantize_dequantize_v_latent(cfg, v_lat, r_v, device)

            for row_start in range(0, seq_len, row_batch_size):
                row_end = min(row_start + row_batch_size, seq_len)
                additive, valid = _make_causal_mask(row_start, row_end, seq_len, device)
                q_lat = q_b[:, row_start:row_end, :] @ p_k
                logits = (q_lat @ k_lat.transpose(-1, -2)) * logit_scale
                attn = stable_softmax(logits.float(), additive, valid)
                teacher = attn @ v_b
                z = attn @ v_lat_q
                z = _apply_current_token_fp_correction(cfg, z, attn, v_lat, v_lat_q, row_start, row_end)
                baseline = z @ p_v.transpose(0, 1)
                decoded = z @ d_v

                teacher_sq += float((teacher.float() ** 2).sum().detach().cpu())
                pv_err += float(((baseline - teacher).float() ** 2).sum().detach().cpu())
                dv_err += float(((decoded - teacher).float() ** 2).sum().detach().cpu())
                count += teacher.numel()

            batches += 1
            del q_b, k_b, v_b, k_lat, v_lat, v_lat_q

    denom_sq = max(teacher_sq, 1e-12)
    pv_normalized = pv_err / denom_sq
    dv_normalized = dv_err / denom_sq
    improvement = 0.0 if pv_normalized <= 0 else 1.0 - dv_normalized / pv_normalized
    return {
        "eval_batches": batches,
        "eval_elements": count,
        "baseline_pv_mse": pv_err / max(count, 1),
        "dv_mse": dv_err / max(count, 1),
        "baseline_pv_normalized": pv_normalized,
        "dv_normalized": dv_normalized,
        "relative_improvement": improvement,
    }


def _fit_dv_layer(cfg, layer_idx: int, device: torch.device, ranks_per_layer: dict[int, tuple[int, int]]) -> dict:
    if getattr(cfg.dv_output_aware, "teacher", "compressed") != "compressed":
        raise ValueError("dv_output_aware.teacher currently supports only 'compressed'")

    calib_path = Path(cfg.calib.output_dir) / f"layer_{layer_idx}.pt"
    projector_path = Path(cfg.dv_output_aware.input_dir) / f"layer_{layer_idx}" / "projector.pt"
    if not calib_path.exists():
        return {"layer": layer_idx, "status": "skipped", "reason": f"missing {calib_path}"}
    if not projector_path.exists():
        return {"layer": layer_idx, "status": "skipped", "reason": f"missing {projector_path}"}

    meta = _load_meta(cfg)
    n_heads = _load_n_heads(cfg, meta)
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
    gamma = _effective_gamma(cfg, projector_data, device)

    print(
        f"[dv] layer {layer_idx}: r_k={r_k} r_v={r_v} "
        f"items={q.shape[0]} seq={q.shape[1]} device={device}"
    )
    s_zz, s_zu, total_rows = _accumulate_layer_statistics(cfg, q, k, v, p_k, p_v, gamma, device)
    d_v = _solve_dv(s_zz, s_zu, float(cfg.dv_output_aware.lambda_ridge))
    metrics = _evaluate_layer_decoder(cfg, q, k, v, p_k, p_v, d_v, gamma, device)
    metrics.update(
        {
            "layer": layer_idx,
            "status": "ok",
            "r_k": r_k,
            "r_v": r_v,
            "head_dim": head_dim,
            "fit_rows": total_rows,
            "teacher": "compressed",
            "quant_aware": _uses_quant_aware(cfg),
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
        "teacher": metrics["teacher"],
        "quant_aware": metrics["quant_aware"],
        "current_token_fp": metrics["current_token_fp"],
        "lambda_ridge": metrics["lambda_ridge"],
        "baseline_pv_normalized": metrics["baseline_pv_normalized"],
        "dv_normalized": metrics["dv_normalized"],
        "relative_improvement": metrics["relative_improvement"],
    }
    save_pt(output_data, output_dir / "projector.pt")
    save_json(metrics, output_dir / "dv_fit_metrics.json")
    print(
        f"[dv] layer {layer_idx}: pv_norm={metrics['baseline_pv_normalized']:.6e} "
        f"dv_norm={metrics['dv_normalized']:.6e} "
        f"improve={metrics['relative_improvement']:.2%}"
    )
    return metrics


def _dv_worker(config_path: str, layers: list[int], device: str, overrides: dict) -> list[dict]:
    cfg = load_config(config_path)
    _apply_overrides(cfg, overrides)
    torch_device = _set_worker_device(cfg, device)
    ranks_per_layer = load_ranks(cfg.dv_output_aware.input_dir)
    print(f"[worker:{device}] dv layers={layers}")
    return [_fit_dv_layer(cfg, layer_idx, torch_device, ranks_per_layer) for layer_idx in layers]


def _run(args) -> None:
    cfg = load_config(args.config)
    _apply_overrides(cfg, args)
    layers = _discover_layers(cfg, args.layers)
    devices = _normalize_devices(cfg.train.device, args.gpus)
    workers = args.workers or min(len(devices), len(layers))
    workers = max(1, min(workers, len(layers)))
    shards = [s for s in _split_round_robin(layers, workers) if s]
    output_dir = Path(cfg.dv_output_aware.output_dir)

    print("=" * 60)
    print("[dv] output-aware closed-form D_V fit")
    print(f"[dv] input_dir={Path(cfg.dv_output_aware.input_dir)}")
    print(f"[dv] output_dir={output_dir}")
    print(f"[dv] layers={layers}")
    print(f"[dv] workers={len(shards)} devices={devices}")
    print(
        f"[dv] teacher={cfg.dv_output_aware.teacher} "
        f"quant_aware={cfg.dv_output_aware.quant_aware} "
        f"lambda_ridge={cfg.dv_output_aware.lambda_ridge}"
    )
    print("=" * 60)

    if args.clean_output_dir and output_dir.exists():
        print(f"[dv] --clean-output-dir: removing {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _prime_output_dir(Path(cfg.dv_output_aware.input_dir), output_dir)

    overrides = {
        "input_dir": str(cfg.dv_output_aware.input_dir),
        "output_dir": str(cfg.dv_output_aware.output_dir),
        "lambda_ridge": cfg.dv_output_aware.lambda_ridge,
        "sample_batch_size": cfg.dv_output_aware.sample_batch_size,
        "row_batch_size": cfg.dv_output_aware.row_batch_size,
        "quant_aware": cfg.dv_output_aware.quant_aware,
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
        dv_norm = [float(r["dv_normalized"]) for r in results if r.get("status") == "ok"]
        pv_norm = [float(r["baseline_pv_normalized"]) for r in results if r.get("status") == "ok"]
        print(f"\n[dv] fitted layers={ok}")
        print(f"[dv] avg pv_norm={sum(pv_norm) / len(pv_norm):.6e} avg dv_norm={sum(dv_norm) / len(dv_norm):.6e}")
    if skipped:
        for item in skipped:
            print(f"[dv] skipped layer {item.get('layer')}: {item.get('reason')}")
    print(f"[dv] rebuilt ranks.json at {ranks_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit output-aware D_V decoders for trained HAWP projectors",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("config", nargs="?", default=None, help="Path to yaml config")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel worker processes")
    parser.add_argument("--gpus", nargs="+", default=None, help="GPU ids/devices, e.g. 0 1 2 or cuda:0 cuda:1")
    parser.add_argument("--layers", nargs="+", type=int, default=None, help="Optional layer subset")
    parser.add_argument("--input-dir", default=None, help="Override dv_output_aware.input_dir")
    parser.add_argument("--output-dir", default=None, help="Override dv_output_aware.output_dir")
    parser.add_argument("--lambda-ridge", type=float, default=None, help="Override ridge regularization")
    parser.add_argument("--sample-batch-size", type=int, default=None, help="Override effective head batch size")
    parser.add_argument("--row-batch-size", type=int, default=None, help="Override row block size")
    parser.add_argument("--quant-aware", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--clean-output-dir", action="store_true", default=False)
    args = parser.parse_args()

    if args.config is None:
        args.config = str(Path(__file__).resolve().parent.parent / "configs" / "dev_local.yaml")
    else:
        args.config = str(Path(args.config).resolve())
    _run(args)


if __name__ == "__main__":
    main()
