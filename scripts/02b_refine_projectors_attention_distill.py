#!/usr/bin/env python
"""Refine trained projectors with lightweight attention-output distillation.

This script keeps ranks fixed and starts from existing ``projector.pt`` files.
It optimizes only the attention output match:

    teacher: softmax(Q K^T / sqrt(d_h)) V
    student: low-rank HAWP attention output using p_k / p_v / gamma

By default refined projectors are written to
``artifacts/projectors_attn_distill`` with the same layer/ranks layout as
``02_train_projectors_parallel.py``.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch
from transformers import AutoConfig

from hawp_laq.config import load_config
from hawp_laq.offline.attention_distill_trainer import refine_attention_output_projector
from hawp_laq.offline.projector_trainer import _complete_to_orthonormal_basis
from hawp_laq.offline.rank_search import infer_calib_dims
from hawp_laq.runtime.projector_bank import normalize_projector_data, rebuild_ranks_json
from hawp_laq.utils.io import load_pt, save_pt


def _load_n_heads(cfg, meta: dict) -> int:
    n_heads = meta.get("n_heads")
    if n_heads is not None:
        return int(n_heads)
    model_cfg = AutoConfig.from_pretrained(
        cfg.model.model_id,
        local_files_only=Path(cfg.model.model_id).expanduser().is_dir(),
    )
    return int(model_cfg.num_attention_heads)


def _discover_layers(calib_dir: Path, meta: dict, requested_layers: list[int] | None) -> list[int]:
    if requested_layers:
        return sorted(dict.fromkeys(int(x) for x in requested_layers))
    n_layers = int(meta.get("n_layers", 0) or 0)
    if n_layers > 0:
        return list(range(n_layers))
    layers = []
    for p in sorted(calib_dir.glob("layer_*.pt")):
        try:
            layers.append(int(p.stem.split("_", 1)[1]))
        except ValueError:
            pass
    return sorted(layers)


def _chunk_layers(layers: list[int], workers: int) -> list[list[int]]:
    if workers <= 1:
        return [layers]
    chunks = [[] for _ in range(workers)]
    for i, layer_idx in enumerate(layers):
        chunks[i % workers].append(layer_idx)
    return [c for c in chunks if c]


def _resolve_worker_devices(cfg, workers: int, gpus: str | None) -> list[str]:
    if gpus:
        devices = []
        for item in gpus.split(","):
            item = item.strip()
            if not item:
                continue
            devices.append(item if item.startswith("cuda") or item == "cpu" else f"cuda:{item}")
        if devices:
            return devices

    cfg_device = str(cfg.train.device)
    if cfg_device.startswith("cuda") and torch.cuda.is_available():
        n = max(1, torch.cuda.device_count())
        return [f"cuda:{i}" for i in range(min(workers, n))]
    return [cfg_device]


def _distill_kwargs(cfg) -> dict:
    d = cfg.attention_distill
    return {
        "n_steps": d.n_steps,
        "sample_batch_size": d.sample_batch_size,
        "row_batch_size": d.row_batch_size,
        "eval_every": d.eval_every,
        "eval_batch_size": d.eval_batch_size,
        "eval_max_batches": d.eval_max_batches,
        "lr_pk": d.lr_pk,
        "lr_pv": d.lr_pv,
        "lr_xi": d.lr_xi,
        "optimizer": d.optimizer,
        "lr": d.lr,
        "orthogonalize_every": d.orthogonalize_every,
        "beta1": d.beta1,
        "beta2": d.beta2,
        "grad_clip": d.grad_clip,
        "gamma_min": d.gamma_min,
        "eps_loss": d.eps_loss,
        "adam_eps": d.adam_eps,
        "train_gamma": d.train_gamma,
        "logit_scale_mode": cfg.hawp.logit_scale_mode,
        "loss_mode": d.loss_mode,
        "early_stopping": d.early_stopping,
        "patience": d.patience,
        "min_delta": d.min_delta,
        "min_delta_mode": d.min_delta_mode,
        "seed": d.seed,
        "verbose": True,
    }


def _format_projector(p: torch.Tensor, *, head_dim: int, rank: int, original: torch.Tensor, save_format: str) -> torch.Tensor:
    save_format = save_format.lower()
    if save_format == "auto":
        save_format = "full" if original.ndim == 2 and original.shape[1] == head_dim else "low_rank"
    if save_format == "low_rank":
        return p[:, :rank].contiguous()
    if save_format == "full":
        return _complete_to_orthonormal_basis(p[:, :rank].contiguous(), head_dim)
    raise ValueError(f"attention_distill.save_format must be auto, low_rank, or full; got {save_format!r}")


def _save_refined_projector(
    original_data: dict,
    result,
    out_path: Path,
    *,
    head_dim: int,
    save_format: str,
    source_path: Path,
) -> None:
    r_k = int(original_data["r_k"])
    r_v = int(original_data["r_v"])
    out = dict(original_data)
    out["p_k"] = _format_projector(
        result.p_k, head_dim=head_dim, rank=r_k,
        original=original_data["p_k"], save_format=save_format,
    )
    out["p_v"] = _format_projector(
        result.p_v, head_dim=head_dim, rank=r_v,
        original=original_data["p_v"], save_format=save_format,
    )
    out["gamma"] = result.gamma
    out["r_k"] = r_k
    out["r_v"] = r_v
    out["logit_scale_mode"] = result.metrics.get("logit_scale_mode", "rk")
    out["attention_distill"] = {
        "source_projector": str(source_path),
        "logit_scale_mode": result.metrics.get("logit_scale_mode", "rk"),
        "best_step": result.best_step,
        "actual_steps": result.actual_steps,
        "stopped_early": result.stopped_early,
        "best_eval_loss": result.best_eval_loss,
        "best_eval_mse": result.best_eval_mse,
        "best_eval_normalized": result.best_eval_normalized,
        "metrics": result.metrics,
    }
    save_pt(out, out_path)


def _refine_one_layer(
    cfg,
    meta: dict,
    n_heads: int,
    layer_idx: int,
    *,
    input_dir: Path,
    output_dir: Path,
    device: str,
) -> bool:
    calib_dir = Path(cfg.calib.output_dir)
    layer_path = calib_dir / f"layer_{layer_idx}.pt"
    projector_path = input_dir / f"layer_{layer_idx}" / "projector.pt"
    if not layer_path.exists():
        print(f"[attention_distill] layer {layer_idx}: no calib data, skipping", flush=True)
        return False
    if not projector_path.exists():
        print(f"[attention_distill] layer {layer_idx}: no projector at {projector_path}, skipping", flush=True)
        return False

    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.set_device(torch.device(device))

    layer_data = load_pt(layer_path)
    q = layer_data["q"].float()
    k = layer_data["k"].float()
    v = layer_data["v"].float()
    d_model, head_dim = infer_calib_dims(q, n_heads, meta)

    projector_data = normalize_projector_data(load_pt(projector_path), layer_idx)
    if "r_k" not in projector_data or "r_v" not in projector_data:
        print(f"[attention_distill] layer {layer_idx}: projector missing r_k/r_v, skipping", flush=True)
        return False

    print(
        f"\n[attention_distill] layer {layer_idx}: "
        f"d_model={d_model} head_dim={head_dim} "
        f"r_k={projector_data['r_k']} r_v={projector_data['r_v']} device={device}",
        flush=True,
    )
    print(f"[attention_distill] layer {layer_idx}: q={tuple(q.shape)} k={tuple(k.shape)} v={tuple(v.shape)}", flush=True)

    result = refine_attention_output_projector(
        q, k, v, projector_data,
        d_model=d_model,
        n_heads=n_heads,
        device=device,
        **_distill_kwargs(cfg),
    )

    out_path = output_dir / f"layer_{layer_idx}" / "projector.pt"
    _save_refined_projector(
        projector_data, result, out_path,
        head_dim=head_dim,
        save_format=cfg.attention_distill.save_format,
        source_path=projector_path,
    )
    print(
        f"[save] {out_path}  r_k={projector_data['r_k']} r_v={projector_data['r_v']} "
        f"best_step={result.best_step} best_loss={result.best_eval_loss:.6e} "
        f"stopped_early={result.stopped_early}",
        flush=True,
    )
    return True


def _worker_refine_layers(
    config_path: str,
    layer_ids: list[int],
    input_dir_str: str,
    output_dir_str: str,
    device: str,
) -> list[int]:
    cfg = load_config(config_path)
    cfg.train.device = device
    meta = load_pt(Path(cfg.calib.output_dir) / "meta.pt")
    n_heads = _load_n_heads(cfg, meta)
    input_dir = Path(input_dir_str)
    output_dir = Path(output_dir_str)

    saved = []
    for layer_idx in layer_ids:
        ok = _refine_one_layer(
            cfg, meta, n_heads, layer_idx,
            input_dir=input_dir,
            output_dir=output_dir,
            device=device,
        )
        if ok:
            saved.append(layer_idx)
    return saved


def run(
    config_path: str | Path,
    *,
    layers: list[int] | None,
    in_place: bool,
    clean_output_dir: bool,
    workers: int,
    gpus: str | None,
) -> None:
    cfg = load_config(config_path)
    calib_dir = Path(cfg.calib.output_dir)
    meta = load_pt(calib_dir / "meta.pt")
    n_heads = _load_n_heads(cfg, meta)

    input_dir = Path(cfg.attention_distill.input_dir)
    if not input_dir.exists():
        input_dir = Path(cfg.projector.output_dir)
    output_dir = input_dir if in_place else Path(cfg.attention_distill.output_dir)

    if clean_output_dir and output_dir.exists() and output_dir != input_dir:
        print(f"[attention_distill] --clean-output-dir: removing {output_dir}")
        shutil.rmtree(output_dir)
    if output_dir != input_dir and input_dir.exists():
        shutil.copytree(input_dir, output_dir, dirs_exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    layer_ids = _discover_layers(calib_dir, meta, layers)
    workers = max(1, min(int(workers), len(layer_ids) if layer_ids else 1))
    devices = _resolve_worker_devices(cfg, workers, gpus)
    print("=" * 60)
    print(f"[attention_distill] calib_dir={calib_dir}")
    print(f"[attention_distill] input_dir={input_dir}")
    print(f"[attention_distill] output_dir={output_dir}")
    print(f"[attention_distill] layers={layer_ids}")
    print(f"[attention_distill] workers={workers} devices={devices}")
    print("=" * 60)

    saved_layers: list[int] = []
    if workers <= 1:
        device = devices[0] if devices else str(cfg.train.device)
        cfg.train.device = device
        for layer_idx in layer_ids:
            ok = _refine_one_layer(
                cfg, meta, n_heads, layer_idx,
                input_dir=input_dir,
                output_dir=output_dir,
                device=device,
            )
            if ok:
                saved_layers.append(layer_idx)
    else:
        chunks = _chunk_layers(layer_ids, workers)
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=len(chunks), mp_context=ctx) as ex:
            futures = []
            for i, chunk in enumerate(chunks):
                device = devices[i % len(devices)]
                futures.append(ex.submit(
                    _worker_refine_layers,
                    str(config_path),
                    chunk,
                    str(input_dir),
                    str(output_dir),
                    device,
                ))
            for fut in as_completed(futures):
                saved_layers.extend(fut.result())

    if saved_layers:
        ranks_path = rebuild_ranks_json(output_dir)
        print(f"\n[attention_distill] refined layers={sorted(saved_layers)}")
        print(f"[attention_distill] rebuilt ranks.json at {ranks_path}")
    else:
        print("\n[attention_distill] no projectors were refined")


def main() -> None:
    ap = argparse.ArgumentParser(description="Refine projectors with attention-output distillation")
    ap.add_argument("config", type=str)
    ap.add_argument("--layers", type=int, nargs="*", default=None, help="Optional layer ids to refine")
    ap.add_argument("--in-place", action="store_true", help="Write refined files back to attention_distill.input_dir")
    ap.add_argument("--clean-output-dir", action="store_true", help="Remove output dir first when it differs from input_dir")
    ap.add_argument("--workers", type=int, default=1, help="Number of layer-parallel worker processes")
    ap.add_argument("--gpus", type=str, default=None, help="Comma-separated GPU ids/devices, e.g. 0,1,2,3 or cuda:0,cuda:1")
    args = ap.parse_args()
    run(
        args.config,
        layers=args.layers,
        in_place=args.in_place,
        clean_output_dir=args.clean_output_dir,
        workers=args.workers,
        gpus=args.gpus,
    )


if __name__ == "__main__":
    main()
