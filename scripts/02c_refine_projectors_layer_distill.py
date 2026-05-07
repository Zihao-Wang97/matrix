#!/usr/bin/env python
"""Refine projectors with full decoder-layer output distillation.

This is the full "C" path:

    teacher: original decoder layer hidden_in -> hidden_out
    student: HAWP-converted decoder layer hidden_in -> hidden_out

It keeps ranks fixed, starts from existing projector.pt files, and optimizes
only HAWPAttention p_k / p_v / gamma inside the selected layer.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from hawp_laq.config import load_config
from hawp_laq.modeling.attention_hawp import HAWPAttention
from hawp_laq.modeling.modeling_llama_hawp import _find_layers_and_attn, convert_llama_to_hawp
from hawp_laq.offline.layer_distill_trainer import (
    discover_layer_chunk_paths,
    refine_layer_output_projector,
    save_refined_layer_projector,
)
from hawp_laq.runtime.projector_bank import load_projectors, load_ranks, normalize_projector_data, rebuild_ranks_json
from hawp_laq.utils.io import load_pt


_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def _normalize_gpus_arg(gpus: list[str] | None) -> str | None:
    if not gpus:
        return None
    return ",".join(gpus)


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


def _chunk_layers(layers: list[int], workers: int) -> list[list[int]]:
    if workers <= 1:
        return [layers]
    chunks = [[] for _ in range(workers)]
    for i, layer_idx in enumerate(layers):
        chunks[i % workers].append(layer_idx)
    return [c for c in chunks if c]


def _discover_layers(data_dir: Path, requested_layers: list[int] | None) -> list[int]:
    if requested_layers:
        return sorted(dict.fromkeys(int(x) for x in requested_layers))
    meta_path = data_dir / "meta.pt"
    if meta_path.exists():
        meta = load_pt(meta_path)
        n_layers = int(meta.get("n_layers", 0) or 0)
        if n_layers > 0:
            return list(range(n_layers))
    layers = []
    for d in sorted(data_dir.glob("layer_*")):
        if not d.is_dir():
            continue
        try:
            layers.append(int(d.name.split("_", 1)[1]))
        except ValueError:
            pass
    return sorted(layers)


def _layer_distill_kwargs(cfg) -> dict:
    d = cfg.layer_distill
    return {
        "n_steps": d.n_steps,
        "sample_batch_size": d.sample_batch_size,
        "eval_every": d.eval_every,
        "eval_max_batches": d.eval_max_batches,
        "optimizer": d.optimizer,
        "lr": d.lr,
        "lr_pk": d.lr_pk,
        "lr_pv": d.lr_pv,
        "lr_xi": d.lr_xi,
        "beta1": d.beta1,
        "beta2": d.beta2,
        "grad_clip": d.grad_clip,
        "train_gamma": d.train_gamma,
        "gamma_min": d.gamma_min,
        "gamma_max": d.gamma_max,
        "eps_loss": d.eps_loss,
        "adam_eps": d.adam_eps,
        "orthogonalize_every": d.orthogonalize_every,
        "alternate_pk_pv": d.alternate_pk_pv,
        "finite_guard": d.finite_guard,
        "bad_step_patience": d.bad_step_patience,
        "lr_backoff": d.lr_backoff,
        "loss_mode": d.loss_mode,
        "early_stopping": d.early_stopping,
        "patience": d.patience,
        "min_delta": d.min_delta,
        "min_delta_mode": d.min_delta_mode,
        "seed": d.seed,
        "verbose": True,
    }


def _head_dim_from_model(model: torch.nn.Module) -> int:
    config = model.config
    hidden_size = int(getattr(config, "hidden_size", 0) or getattr(config, "word_embed_proj_dim", 0) or 0)
    n_heads = int(getattr(config, "num_attention_heads", 0) or 0)
    if hidden_size <= 0 or n_heads <= 0:
        raise ValueError("Cannot infer head_dim from model config")
    return hidden_size // n_heads


def _ranks_from_projector_files(input_dir: Path) -> dict[int, tuple[int, int]]:
    ranks = load_ranks(input_dir)
    for d in sorted(input_dir.glob("layer_*")):
        if not d.is_dir():
            continue
        try:
            layer_idx = int(d.name.split("_", 1)[1])
        except ValueError:
            continue
        pt_path = d / "projector.pt"
        if not pt_path.exists():
            continue
        data = normalize_projector_data(load_pt(pt_path), layer_idx)
        if "r_k" in data and "r_v" in data:
            ranks[layer_idx] = (int(data["r_k"]), int(data["r_v"]))
    return ranks


def _load_student_model(config_path: str, input_dir: Path, device: str):
    cfg = load_config(config_path)
    if cfg.model.load_in_4bit:
        raise ValueError("layer_distill does not support model.load_in_4bit=true; use a floating-point model")
    cfg.train.device = device

    model_id = cfg.model.model_id
    dtype = _DTYPE_MAP.get(cfg.model.torch_dtype, torch.float32)
    is_local = Path(model_id).expanduser().is_dir()
    tok_kwargs = {"local_files_only": True} if is_local else {}
    model_kwargs = {
        "torch_dtype": dtype,
        "device_map": {"": device},
    }
    if is_local:
        model_kwargs["local_files_only"] = True

    print(f"[layer_distill] load model_id={model_id} dtype={cfg.model.torch_dtype} device={device}", flush=True)
    AutoTokenizer.from_pretrained(model_id, **tok_kwargs)
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)

    layers = _find_layers_and_attn(model)
    if not layers:
        raise RuntimeError("No compatible decoder layers found for HAWP conversion")

    head_dim = _head_dim_from_model(model)
    file_ranks = _ranks_from_projector_files(input_dir)
    ranks_per_layer = {
        idx: file_ranks.get(idx, (head_dim, head_dim))
        for idx in range(len(layers))
    }

    model = convert_llama_to_hawp(
        model,
        r_k=head_dim,
        r_v=head_dim,
        ranks_per_layer=ranks_per_layer,
        allow_default_full_rank=True,
        logit_scale_mode=cfg.hawp.logit_scale_mode,
        gamma_mode=cfg.hawp.gamma_mode,
        gamma_value=cfg.hawp.gamma_value,
        use_archive_k_ip_approx=cfg.hawp.use_archive_k_ip_approx,
    )
    if not hasattr(model, "hf_device_map"):
        model = model.to(device)
    load_projectors(model, input_dir, strict=True)
    model.eval()
    return model, cfg, ranks_per_layer


def _refine_one_layer(
    model: torch.nn.Module,
    cfg,
    layer_idx: int,
    *,
    data_dir: Path,
    input_dir: Path,
    output_dir: Path,
    device: str,
) -> bool:
    chunk_paths = discover_layer_chunk_paths(data_dir, layer_idx)
    projector_path = input_dir / f"layer_{layer_idx}" / "projector.pt"
    if not chunk_paths:
        print(f"[layer_distill] layer {layer_idx}: no chunk data, skipping", flush=True)
        return False
    if not projector_path.exists():
        print(f"[layer_distill] layer {layer_idx}: no projector at {projector_path}, skipping", flush=True)
        return False

    layers = _find_layers_and_attn(model)
    if layer_idx >= len(layers):
        print(f"[layer_distill] layer {layer_idx}: model has only {len(layers)} layers, skipping", flush=True)
        return False
    layer = layers[layer_idx][1]
    hawp_modules = [m for m in layer.modules() if isinstance(m, HAWPAttention)]
    if len(hawp_modules) != 1:
        print(f"[layer_distill] layer {layer_idx}: expected 1 HAWPAttention, found {len(hawp_modules)}, skipping", flush=True)
        return False
    module = hawp_modules[0]
    if module.r_k >= module.head_dim and module.r_v >= module.head_dim:
        print(
            f"[layer_distill] layer {layer_idx}: full-rank "
            f"(r_k={module.r_k}, r_v={module.r_v}, head_dim={module.head_dim}), skipping refine",
            flush=True,
        )
        return False

    original_projector = normalize_projector_data(load_pt(projector_path), layer_idx)
    print(
        f"\n[layer_distill] layer {layer_idx}: chunks={len(chunk_paths)} "
        f"r_k={module.r_k} r_v={module.r_v} device={device}",
        flush=True,
    )

    result = refine_layer_output_projector(
        layer,
        chunk_paths,
        device=device,
        **_layer_distill_kwargs(cfg),
    )

    out_path = output_dir / f"layer_{layer_idx}" / "projector.pt"
    save_refined_layer_projector(
        layer,
        original_projector,
        result,
        out_path,
        save_format=cfg.layer_distill.save_format,
        source_path=projector_path,
    )
    print(
        f"[save] {out_path}  r_k={module.r_k} r_v={module.r_v} "
        f"best_step={result.best_step} best_loss={result.best_eval_loss:.6e} "
        f"stopped_early={result.stopped_early}",
        flush=True,
    )
    return True


def _worker_refine_layers(
    config_path: str,
    layer_ids: list[int],
    data_dir_str: str,
    input_dir_str: str,
    output_dir_str: str,
    device: str,
) -> list[int]:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.set_device(torch.device(device))

    input_dir = Path(input_dir_str)
    model, cfg, _ranks = _load_student_model(config_path, input_dir, device)
    data_dir = Path(data_dir_str)
    output_dir = Path(output_dir_str)

    saved = []
    for layer_idx in layer_ids:
        ok = _refine_one_layer(
            model,
            cfg,
            layer_idx,
            data_dir=data_dir,
            input_dir=input_dir,
            output_dir=output_dir,
            device=device,
        )
        if ok:
            saved.append(layer_idx)
    return saved


def _resolve_input_dir(cfg) -> Path:
    candidates = [
        Path(cfg.layer_distill.input_dir),
        Path(cfg.attention_distill.output_dir),
        Path(cfg.projector.output_dir),
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "No projector input directory found. Checked: "
        + ", ".join(str(p) for p in candidates)
    )


def run(
    config_path: str | Path,
    *,
    layers: list[int] | None,
    input_dir: str | None,
    output_dir: str | None,
    in_place: bool,
    clean_output_dir: bool,
    workers: int,
    gpus: str | None,
) -> None:
    cfg = load_config(config_path)
    data_dir = Path(cfg.layer_distill.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(
            f"layer_distill data_dir not found: {data_dir}. "
            f"Run scripts/01b_collect_layer_distill_data.py first."
        )

    resolved_input = Path(input_dir) if input_dir else _resolve_input_dir(cfg)
    resolved_output = resolved_input if in_place else Path(output_dir or cfg.layer_distill.output_dir)

    if clean_output_dir and resolved_output.exists() and resolved_output != resolved_input:
        print(f"[layer_distill] --clean-output-dir: removing {resolved_output}")
        shutil.rmtree(resolved_output)
    if resolved_output != resolved_input:
        shutil.copytree(resolved_input, resolved_output, dirs_exist_ok=True)
    resolved_output.mkdir(parents=True, exist_ok=True)

    layer_ids = _discover_layers(data_dir, layers)
    workers = max(1, min(int(workers), len(layer_ids) if layer_ids else 1))
    devices = _resolve_worker_devices(cfg, workers, gpus)

    print("=" * 60)
    print(f"[layer_distill] data_dir={data_dir}")
    print(f"[layer_distill] input_dir={resolved_input}")
    print(f"[layer_distill] output_dir={resolved_output}")
    print(f"[layer_distill] layers={layer_ids}")
    print(f"[layer_distill] workers={workers} devices={devices}")
    print("=" * 60)

    saved_layers: list[int] = []
    if workers <= 1:
        device = devices[0] if devices else str(cfg.train.device)
        saved_layers = _worker_refine_layers(
            str(config_path),
            layer_ids,
            str(data_dir),
            str(resolved_input),
            str(resolved_output),
            device,
        )
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
                    str(data_dir),
                    str(resolved_input),
                    str(resolved_output),
                    device,
                ))
            for fut in as_completed(futures):
                saved_layers.extend(fut.result())

    if saved_layers:
        ranks_path = rebuild_ranks_json(resolved_output)
        print(f"\n[layer_distill] refined layers={sorted(saved_layers)}")
        print(f"[layer_distill] rebuilt ranks.json at {ranks_path}")
    else:
        print("\n[layer_distill] no projectors were refined")


def main() -> None:
    ap = argparse.ArgumentParser(description="Refine projectors with full decoder-layer output distillation")
    ap.add_argument("config", type=str)
    ap.add_argument("--layers", type=int, nargs="*", default=None, help="Optional layer ids to refine")
    ap.add_argument("--input-dir", type=str, default=None, help="Override layer_distill.input_dir")
    ap.add_argument("--output-dir", type=str, default=None, help="Override layer_distill.output_dir")
    ap.add_argument("--in-place", action="store_true", help="Write refined files back to the input dir")
    ap.add_argument("--clean-output-dir", action="store_true", help="Remove output dir first when it differs from input dir")
    ap.add_argument("--workers", type=int, default=1, help="Number of layer-parallel worker processes")
    ap.add_argument(
        "--gpus",
        nargs="*",
        default=None,
        help="Comma-separated GPU ids/devices, e.g. --gpus 0,1,2,3 or --gpus 0, 1, 2, 3",
    )
    args = ap.parse_args()
    run(
        args.config,
        layers=args.layers,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        in_place=args.in_place,
        clean_output_dir=args.clean_output_dir,
        workers=args.workers,
        gpus=_normalize_gpus_arg(args.gpus),
    )


if __name__ == "__main__":
    main()
