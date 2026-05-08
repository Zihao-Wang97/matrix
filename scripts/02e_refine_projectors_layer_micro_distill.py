#!/usr/bin/env python
"""Second-stage lightweight full-layer calibration.

This script is intended to run after attention-module distillation
(``02d_refine_projectors_attention_module_distill.py``).  It uses the same
``hidden_in -> hidden_out`` chunks as full layer distillation, but it trains
only a very small subset of parameters:

  - ``--train-scope gamma``: train gamma only (default)
  - ``--train-scope pv_gamma``: train P_V and gamma; keep P_K fixed

Ranks and projector file format are preserved.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import random
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from hawp_laq.config import load_config
from hawp_laq.modeling.attention_hawp import HAWPAttention
from hawp_laq.modeling.modeling_llama_hawp import _find_layers_and_attn, convert_llama_to_hawp
from hawp_laq.offline.layer_distill_trainer import (
    _all_finite_tensors,
    _backoff_lrs,
    _backoff_torch_optimizer,
    _call_decoder_layer,
    _format_projector,
    _hawp_modules,
    _is_finite_tensor,
    _load_chunk,
    _loss_stats,
    _orth_err,
    _param_dtype,
    _restore_hawp_state,
    _restore_riemannian_optimizer,
    _restore_torch_optimizer,
    _snapshot_hawp_state,
    _snapshot_riemannian_optimizer,
    _snapshot_torch_optimizer,
    discover_layer_chunk_paths,
)
from hawp_laq.offline.low_rank_attention_optimizer_torch import RiemannianAdam, clip_by_global_norm
from hawp_laq.runtime.projector_bank import load_projectors, load_ranks, normalize_projector_data, rebuild_ranks_json
from hawp_laq.utils.io import load_pt, save_pt


_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


@dataclass
class MicroLayerDistillResult:
    metrics: dict[str, Any]
    best_step: int
    actual_steps: int
    stopped_early: bool
    best_eval_loss: float
    best_eval_mse: float
    best_eval_normalized: float


def _normalize_gpus_arg(gpus: list[str] | None) -> str | None:
    if not gpus:
        return None
    return ",".join(str(x) for x in gpus)


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
        raise ValueError("micro layer distill does not support model.load_in_4bit=true")
    cfg.train.device = device

    model_id = cfg.model.model_id
    dtype = _DTYPE_MAP.get(cfg.model.torch_dtype, torch.float32)
    is_local = Path(model_id).expanduser().is_dir()
    tok_kwargs = {"local_files_only": True} if is_local else {}
    model_kwargs: dict[str, Any] = {"torch_dtype": dtype, "device_map": {"": device}}
    if is_local:
        model_kwargs["local_files_only"] = True

    print(f"[layer_micro] load model_id={model_id} dtype={cfg.model.torch_dtype} device={device}", flush=True)
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
    load_projectors(
        model,
        input_dir,
        strict=True,
        expected_logit_scale_mode=cfg.hawp.logit_scale_mode,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, cfg


def _module_is_low_rank(module: HAWPAttention) -> bool:
    return module.r_k < module.head_dim or module.r_v < module.head_dim


def _gamma_trainable_micro(module: HAWPAttention, train_gamma: bool) -> bool:
    if not train_gamma:
        return False
    if module.gamma_mode == "fixed" and module.gamma_value is not None:
        return False
    if module.gamma_mode not in ("learned", "fixed"):
        return False
    return _module_is_low_rank(module)


def _prepare_micro_params(
    layer: torch.nn.Module,
    *,
    train_scope: str,
    train_gamma: bool,
    gamma_min: float,
    gamma_max: float,
) -> tuple[list[HAWPAttention], list[torch.nn.Parameter]]:
    for p in layer.parameters():
        p.requires_grad_(False)

    modules = _hawp_modules(layer)
    params: list[torch.nn.Parameter] = []
    for module in modules:
        module.p_k.data = module.p_k.data.float()
        module.p_v.data = module.p_v.data.float()
        module.gamma.data = module.gamma.data.float().clamp_(min=gamma_min, max=gamma_max)

        module.p_k.requires_grad_(False)
        module.p_v.requires_grad_(train_scope == "pv_gamma" and module.r_v < module.head_dim)
        module.gamma.requires_grad_(_gamma_trainable_micro(module, train_gamma))

        if module.p_v.requires_grad:
            params.append(module.p_v)
        if module.gamma.requires_grad:
            params.append(module.gamma)
    return modules, params


@torch.no_grad()
def _eval_layer(
    layer: torch.nn.Module,
    chunk_paths: list[Path],
    *,
    device: torch.device,
    dtype: torch.dtype,
    sample_batch_size: Optional[int],
    eval_max_batches: Optional[int],
    loss_mode: str,
    eps_loss: float,
) -> dict[str, float]:
    paths = chunk_paths[:eval_max_batches] if eval_max_batches is not None and eval_max_batches > 0 else chunk_paths
    total_diff = torch.zeros((), device=device, dtype=torch.float32)
    total_teacher = torch.zeros((), device=device, dtype=torch.float32)
    total_count = 0
    for path in paths:
        hidden_in, target = _load_chunk(path, device, dtype, sample_batch_size)
        student = _call_decoder_layer(layer, hidden_in)
        total_diff = total_diff + (student.float() - target.float()).pow(2).sum()
        total_teacher = total_teacher + target.float().pow(2).sum()
        total_count += target.numel()
    mse = total_diff / max(total_count, 1)
    normalized = total_diff / (total_teacher + eps_loss)
    loss = mse if loss_mode == "absolute" else normalized
    return {
        "loss": float(loss.detach().cpu()),
        "mse": float(mse.detach().cpu()),
        "normalized": float(normalized.detach().cpu()),
    }


def _refine_micro_layer(
    layer: torch.nn.Module,
    chunk_paths: list[Path],
    *,
    device: str,
    train_scope: str,
    n_steps: int,
    sample_batch_size: Optional[int],
    eval_every: int,
    eval_max_batches: Optional[int],
    lr: float,
    lr_pv: float,
    lr_xi: float,
    beta1: float,
    beta2: float,
    grad_clip: float,
    train_gamma: bool,
    gamma_min: float,
    gamma_max: float,
    eps_loss: float,
    adam_eps: float,
    finite_guard: bool,
    bad_step_patience: int,
    lr_backoff: float,
    loss_mode: str,
    early_stopping: bool,
    patience: int,
    min_delta: float,
    min_delta_mode: str,
    seed: int,
) -> MicroLayerDistillResult:
    if not chunk_paths:
        raise FileNotFoundError("No layer distill chunk_*.pt files found")
    if train_scope not in ("gamma", "pv_gamma"):
        raise ValueError(f"train_scope must be 'gamma' or 'pv_gamma', got {train_scope!r}")
    if eval_every <= 0:
        raise ValueError("eval_every must be > 0")

    torch.manual_seed(seed)
    rng = random.Random(seed)
    dev = torch.device(device)
    layer.to(dev).eval()
    dtype = _param_dtype(layer)

    modules, trainable_params = _prepare_micro_params(
        layer,
        train_scope=train_scope,
        train_gamma=train_gamma,
        gamma_min=gamma_min,
        gamma_max=gamma_max,
    )
    if not modules:
        raise RuntimeError("Layer has no HAWPAttention modules")
    if not trainable_params:
        raise RuntimeError(f"Layer has no trainable parameters for train_scope={train_scope!r}")

    gamma_params = [m.gamma for m in modules if m.gamma.requires_grad]
    gamma_optimizer = (
        torch.optim.Adam(gamma_params, lr=lr_xi, betas=(beta1, beta2), eps=adam_eps)
        if gamma_params
        else None
    )
    pv_opts = {
        m.layer_idx: RiemannianAdam((m.head_dim, m.r_v), dev, torch.float32, lr_pv, beta1, beta2, adam_eps)
        for m in modules
        if train_scope == "pv_gamma" and m.r_v < m.head_dim
    }
    adam_optimizer = (
        torch.optim.Adam(trainable_params, lr=lr_xi, betas=(beta1, beta2), eps=adam_eps)
        if train_scope == "gamma" and gamma_params
        else None
    )

    best_eval_loss = float("inf")
    best_eval_mse = 0.0
    best_eval_normalized = 0.0
    best_state = _snapshot_hawp_state(layer)
    best_step = 0
    stale_checks = 0
    bad_steps = 0
    stopped_early = False
    history: list[dict[str, Any]] = []
    step = 0

    for step in range(1, n_steps + 1):
        path = chunk_paths[rng.randrange(len(chunk_paths))]
        hidden_in, target = _load_chunk(path, dev, dtype, sample_batch_size)

        state_before = _snapshot_hawp_state(layer)
        pv_state = {k: _snapshot_riemannian_optimizer(v) for k, v in pv_opts.items()}
        gamma_state = _snapshot_torch_optimizer(gamma_optimizer)
        adam_state = _snapshot_torch_optimizer(adam_optimizer)

        student = _call_decoder_layer(layer, hidden_in)
        if finite_guard and not _all_finite_tensors([student]):
            _restore_hawp_state(layer, state_before)
            bad_steps += 1
            for opt_obj in pv_opts.values():
                opt_obj.lr *= lr_backoff
            _backoff_lrs(None, None, gamma_optimizer, lr_backoff)
            _backoff_torch_optimizer(adam_optimizer, lr_backoff)
            print(f"[warn] step={step:04d} skipped: non-finite student output", flush=True)
            if bad_steps >= bad_step_patience:
                stopped_early = True
                break
            continue

        loss, _mse, _normalized = _loss_stats(student.float(), target.float(), loss_mode, eps_loss)
        if not loss.requires_grad:
            raise RuntimeError("Micro layer distill loss has no gradient path")
        if finite_guard and not _is_finite_tensor(loss):
            _restore_hawp_state(layer, state_before)
            bad_steps += 1
            for opt_obj in pv_opts.values():
                opt_obj.lr *= lr_backoff
            _backoff_lrs(None, None, gamma_optimizer, lr_backoff)
            _backoff_torch_optimizer(adam_optimizer, lr_backoff)
            print(f"[warn] step={step:04d} skipped: non-finite loss", flush=True)
            if bad_steps >= bad_step_patience:
                stopped_early = True
                break
            continue

        if train_scope == "gamma":
            assert adam_optimizer is not None
            adam_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grads = [p.grad for p in trainable_params]
            if finite_guard and not _all_finite_tensors(grads):
                _restore_hawp_state(layer, state_before)
                _restore_torch_optimizer(adam_optimizer, adam_state)
                bad_steps += 1
                _backoff_torch_optimizer(adam_optimizer, lr_backoff)
                print(f"[warn] step={step:04d} skipped: non-finite gamma gradient", flush=True)
                if bad_steps >= bad_step_patience:
                    stopped_early = True
                    break
                continue
            torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)
            adam_optimizer.step()
        else:
            grad_params: list[torch.nn.Parameter] = []
            grad_specs: list[tuple[str, HAWPAttention]] = []
            for module in modules:
                if module.r_v < module.head_dim:
                    grad_params.append(module.p_v)
                    grad_specs.append(("pv", module))
            for module in modules:
                if module.gamma.requires_grad:
                    grad_params.append(module.gamma)
                    grad_specs.append(("gamma", module))

            grads = list(torch.autograd.grad(loss, grad_params, allow_unused=True))
            sliced_grads: list[torch.Tensor | None] = []
            for (kind, module), grad in zip(grad_specs, grads):
                if grad is None:
                    sliced_grads.append(None)
                elif kind == "pv":
                    sliced_grads.append(grad[:, :module.r_v])
                else:
                    sliced_grads.append(grad)

            if finite_guard and not _all_finite_tensors(sliced_grads):
                _restore_hawp_state(layer, state_before)
                for k, v in pv_opts.items():
                    _restore_riemannian_optimizer(v, pv_state.get(k))
                _restore_torch_optimizer(gamma_optimizer, gamma_state)
                bad_steps += 1
                for opt_obj in pv_opts.values():
                    opt_obj.lr *= lr_backoff
                _backoff_lrs(None, None, gamma_optimizer, lr_backoff)
                print(f"[warn] step={step:04d} skipped: non-finite pv/gamma gradient", flush=True)
                if bad_steps >= bad_step_patience:
                    stopped_early = True
                    break
                continue

            clipped_grads = clip_by_global_norm(sliced_grads, grad_clip)
            if gamma_optimizer is not None:
                gamma_optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                for (kind, module), grad in zip(grad_specs, clipped_grads):
                    if grad is None:
                        continue
                    if kind == "pv":
                        pv_opts[module.layer_idx].step_(module.p_v[:, :module.r_v], grad.float())
                    else:
                        module.gamma.grad = grad.float()
            if gamma_optimizer is not None and any(kind == "gamma" for kind, _ in grad_specs):
                gamma_optimizer.step()

        with torch.no_grad():
            for module in modules:
                module.gamma.clamp_(min=gamma_min, max=gamma_max)

        if finite_guard:
            values = []
            for module in modules:
                values.extend([module.p_v[:, :module.r_v], module.gamma])
            if not _all_finite_tensors(values):
                _restore_hawp_state(layer, state_before)
                for k, v in pv_opts.items():
                    _restore_riemannian_optimizer(v, pv_state.get(k))
                _restore_torch_optimizer(gamma_optimizer, gamma_state)
                _restore_torch_optimizer(adam_optimizer, adam_state)
                bad_steps += 1
                for opt_obj in pv_opts.values():
                    opt_obj.lr *= lr_backoff
                _backoff_lrs(None, None, gamma_optimizer, lr_backoff)
                _backoff_torch_optimizer(adam_optimizer, lr_backoff)
                print(f"[warn] step={step:04d} skipped: non-finite parameters after update", flush=True)
                if bad_steps >= bad_step_patience:
                    stopped_early = True
                    break
                continue

        bad_steps = 0
        do_eval = step == 1 or step % eval_every == 0 or step == n_steps
        if do_eval:
            eval_stats = _eval_layer(
                layer,
                chunk_paths,
                device=dev,
                dtype=dtype,
                sample_batch_size=sample_batch_size,
                eval_max_batches=eval_max_batches,
                loss_mode=loss_mode,
                eps_loss=eps_loss,
            )
            gamma_val = float(modules[0].gamma.detach().float().cpu())
            orth_v = max((_orth_err(m.p_v, m.r_v) for m in modules), default=0.0)
            eval_loss = eval_stats["loss"]
            eval_mse = eval_stats["mse"]
            eval_norm = eval_stats["normalized"]
            history.append({
                "step": step,
                "eval_loss": eval_loss,
                "eval_mse": eval_mse,
                "eval_normalized": eval_norm,
                "gamma": gamma_val,
                "orthV": orth_v,
                "train_scope": train_scope,
                "bad_steps": bad_steps,
            })
            print(
                f"step={step:04d} micro_layer_loss={eval_loss:.6e} "
                f"mse={eval_mse:.6e} normalized={eval_norm:.6e} "
                f"gamma={gamma_val:.5f} orthV={orth_v:.2e} scope={train_scope}",
                flush=True,
            )

            if eval_loss < best_eval_loss:
                prev_best = best_eval_loss
                best_eval_loss = eval_loss
                best_eval_mse = eval_mse
                best_eval_normalized = eval_norm
                best_state = _snapshot_hawp_state(layer)
                best_step = step
                if prev_best == float("inf"):
                    significant = True
                elif min_delta_mode == "relative":
                    significant = (prev_best - eval_loss) / (abs(prev_best) + eps_loss) > min_delta
                else:
                    significant = (prev_best - eval_loss) > min_delta
                stale_checks = 0 if significant else stale_checks + 1
            else:
                stale_checks += 1

        if early_stopping and stale_checks >= patience:
            print(f"Early stopping at step {step} (no improvement for {stale_checks} eval checks).", flush=True)
            stopped_early = True
            break

    _restore_hawp_state(layer, best_state)
    layer.eval()
    return MicroLayerDistillResult(
        metrics={"layer_micro_distill": history},
        best_step=best_step,
        actual_steps=step,
        stopped_early=stopped_early,
        best_eval_loss=best_eval_loss,
        best_eval_mse=best_eval_mse,
        best_eval_normalized=best_eval_normalized,
    )


def _save_projector(
    layer: torch.nn.Module,
    original_projector: dict,
    result: MicroLayerDistillResult,
    out_path: Path,
    *,
    save_format: str,
    source_path: Path,
    train_scope: str,
) -> None:
    modules = _hawp_modules(layer)
    if len(modules) != 1:
        raise RuntimeError(f"Expected exactly one HAWPAttention in decoder layer, found {len(modules)}")
    module = modules[0]
    original_projector = normalize_projector_data(dict(original_projector), module.layer_idx)
    r_k = int(original_projector["r_k"])
    r_v = int(original_projector["r_v"])
    out = dict(original_projector)
    out["p_k"] = original_projector["p_k"]
    if train_scope == "gamma":
        out["p_v"] = original_projector["p_v"]
    else:
        out["p_v"] = _format_projector(
            module.p_v,
            head_dim=module.head_dim,
            rank=r_v,
            original=original_projector["p_v"],
            save_format=save_format,
        )
    out["gamma"] = module.gamma.detach().cpu()
    out["r_k"] = r_k
    out["r_v"] = r_v
    out["logit_scale_mode"] = module.logit_scale_mode
    out["layer_micro_distill"] = {
        "source_projector": str(source_path),
        "logit_scale_mode": module.logit_scale_mode,
        "train_scope": train_scope,
        "best_step": result.best_step,
        "actual_steps": result.actual_steps,
        "stopped_early": result.stopped_early,
        "best_eval_loss": result.best_eval_loss,
        "best_eval_mse": result.best_eval_mse,
        "best_eval_normalized": result.best_eval_normalized,
        "metrics": result.metrics,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_pt(out, out_path)


def _layer_kwargs(cfg, *, train_scope: str) -> dict[str, Any]:
    d = cfg.layer_distill
    return {
        "train_scope": train_scope,
        "n_steps": int(d.n_steps),
        "sample_batch_size": d.sample_batch_size,
        "eval_every": int(d.eval_every),
        "eval_max_batches": d.eval_max_batches,
        "lr": float(d.lr),
        "lr_pv": float(d.lr_pv),
        "lr_xi": float(d.lr_xi),
        "beta1": float(d.beta1),
        "beta2": float(d.beta2),
        "grad_clip": float(d.grad_clip),
        "train_gamma": bool(d.train_gamma),
        "gamma_min": float(d.gamma_min),
        "gamma_max": float(d.gamma_max),
        "eps_loss": float(d.eps_loss),
        "adam_eps": float(d.adam_eps),
        "finite_guard": bool(d.finite_guard),
        "bad_step_patience": int(d.bad_step_patience),
        "lr_backoff": float(d.lr_backoff),
        "loss_mode": str(d.loss_mode),
        "early_stopping": bool(d.early_stopping),
        "patience": int(d.patience),
        "min_delta": float(d.min_delta),
        "min_delta_mode": str(d.min_delta_mode),
        "seed": int(d.seed),
        "save_format": str(d.save_format),
    }


def _refine_one_layer(
    model: torch.nn.Module,
    cfg,
    layer_idx: int,
    *,
    data_dir: Path,
    input_dir: Path,
    output_dir: Path,
    device: str,
    train_scope: str,
) -> bool:
    chunk_paths = discover_layer_chunk_paths(data_dir, layer_idx)
    projector_path = input_dir / f"layer_{layer_idx}" / "projector.pt"
    if not chunk_paths:
        print(f"[layer_micro] layer {layer_idx}: no chunk data, skipping", flush=True)
        return False
    if not projector_path.exists():
        print(f"[layer_micro] layer {layer_idx}: no projector at {projector_path}, skipping", flush=True)
        return False

    layers = _find_layers_and_attn(model)
    if layer_idx >= len(layers):
        print(f"[layer_micro] layer {layer_idx}: model has only {len(layers)} layers, skipping", flush=True)
        return False
    layer = layers[layer_idx][1]
    modules = [m for m in layer.modules() if isinstance(m, HAWPAttention)]
    if len(modules) != 1:
        print(f"[layer_micro] layer {layer_idx}: expected 1 HAWPAttention, found {len(modules)}, skipping", flush=True)
        return False
    module = modules[0]
    if module.r_k >= module.head_dim and module.r_v >= module.head_dim:
        print(
            f"[layer_micro] layer {layer_idx}: full-rank "
            f"(r_k={module.r_k}, r_v={module.r_v}, head_dim={module.head_dim}), skipping refine",
            flush=True,
        )
        return False

    original_projector = normalize_projector_data(load_pt(projector_path), layer_idx)
    params = _layer_kwargs(cfg, train_scope=train_scope)
    print(
        f"\n[layer_micro] layer {layer_idx}: chunks={len(chunk_paths)} "
        f"r_k={module.r_k} r_v={module.r_v} scope={train_scope} device={device}",
        flush=True,
    )
    result = _refine_micro_layer(
        layer,
        chunk_paths,
        device=device,
        train_scope=params["train_scope"],
        n_steps=params["n_steps"],
        sample_batch_size=params["sample_batch_size"],
        eval_every=params["eval_every"],
        eval_max_batches=params["eval_max_batches"],
        lr=params["lr"],
        lr_pv=params["lr_pv"],
        lr_xi=params["lr_xi"],
        beta1=params["beta1"],
        beta2=params["beta2"],
        grad_clip=params["grad_clip"],
        train_gamma=params["train_gamma"],
        gamma_min=params["gamma_min"],
        gamma_max=params["gamma_max"],
        eps_loss=params["eps_loss"],
        adam_eps=params["adam_eps"],
        finite_guard=params["finite_guard"],
        bad_step_patience=params["bad_step_patience"],
        lr_backoff=params["lr_backoff"],
        loss_mode=params["loss_mode"],
        early_stopping=params["early_stopping"],
        patience=params["patience"],
        min_delta=params["min_delta"],
        min_delta_mode=params["min_delta_mode"],
        seed=params["seed"] + layer_idx,
    )
    out_path = output_dir / f"layer_{layer_idx}" / "projector.pt"
    _save_projector(
        layer,
        original_projector,
        result,
        out_path,
        save_format=params["save_format"],
        source_path=projector_path,
        train_scope=train_scope,
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
    train_scope: str,
) -> list[int]:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.set_device(torch.device(device))
    input_dir = Path(input_dir_str)
    model, cfg = _load_student_model(config_path, input_dir, device)
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
            train_scope=train_scope,
        )
        if ok:
            saved.append(layer_idx)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return saved


def _default_attn_module_input(cfg) -> Path | None:
    base = Path(cfg.attention_distill.input_dir)
    candidate = base.with_name(base.name + "_attn_module_distill")
    return candidate if candidate.exists() else None


def _resolve_input_dir(cfg) -> Path:
    candidates = [
        _default_attn_module_input(cfg),
        Path(cfg.layer_distill.input_dir),
        Path(cfg.attention_distill.output_dir),
        Path(cfg.projector.output_dir),
    ]
    for path in candidates:
        if path is not None and path.exists():
            return path
    raise FileNotFoundError(
        "No projector input directory found. Checked: "
        + ", ".join(str(p) for p in candidates if p is not None)
    )


def _default_output_dir(input_dir: Path, train_scope: str) -> Path:
    suffix = "gamma" if train_scope == "gamma" else "pv_gamma"
    return input_dir.with_name(input_dir.name + f"_layer_micro_{suffix}")


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
    train_scope: str,
) -> None:
    cfg = load_config(config_path)
    data_dir = Path(cfg.layer_distill.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(
            f"layer_distill data_dir not found: {data_dir}. "
            "Run scripts/01b_collect_layer_distill_data.py first."
        )
    resolved_input = Path(input_dir) if input_dir else _resolve_input_dir(cfg)
    resolved_output = resolved_input if in_place else Path(output_dir) if output_dir else _default_output_dir(resolved_input, train_scope)

    if clean_output_dir and resolved_output.exists() and resolved_output != resolved_input:
        print(f"[layer_micro] --clean-output-dir: removing {resolved_output}")
        shutil.rmtree(resolved_output)
    if resolved_output != resolved_input:
        shutil.copytree(resolved_input, resolved_output, dirs_exist_ok=True)
    resolved_output.mkdir(parents=True, exist_ok=True)

    layer_ids = _discover_layers(data_dir, layers)
    workers = max(1, min(int(workers), len(layer_ids) if layer_ids else 1))
    devices = _resolve_worker_devices(cfg, workers, gpus)

    print("=" * 70)
    print(f"[layer_micro] data_dir={data_dir}")
    print(f"[layer_micro] input_dir={resolved_input}")
    print(f"[layer_micro] output_dir={resolved_output}")
    print(f"[layer_micro] train_scope={train_scope}")
    print(f"[layer_micro] layers={layer_ids}")
    print(f"[layer_micro] workers={workers} devices={devices}")
    print("=" * 70)

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
            train_scope,
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
                    train_scope,
                ))
            for fut in as_completed(futures):
                saved_layers.extend(fut.result())

    if saved_layers:
        ranks_path = rebuild_ranks_json(resolved_output)
        print(f"\n[layer_micro] refined layers={sorted(saved_layers)}")
        print(f"[layer_micro] rebuilt ranks.json at {ranks_path}")
    else:
        print("\n[layer_micro] no projectors were refined")


def main() -> None:
    ap = argparse.ArgumentParser(description="Second-stage lightweight full-layer projector calibration")
    ap.add_argument("config")
    ap.add_argument("--layers", type=int, nargs="*", default=None)
    ap.add_argument("--input-dir", default=None, help="First-stage projector dir; defaults to *_attn_module_distill when present")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--in-place", action="store_true")
    ap.add_argument("--clean-output-dir", action="store_true")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--gpus", nargs="*", default=None, help="GPU ids/devices: --gpus 0 1 2 or --gpus 0,1,2")
    ap.add_argument("--train-scope", choices=["gamma", "pv_gamma"], default="gamma")
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
        train_scope=args.train_scope,
    )


if __name__ == "__main__":
    main()
