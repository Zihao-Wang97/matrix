#!/usr/bin/env python
"""Refine projectors with full attention-module output distillation.

This is the heavier first-stage distillation variant.  It uses decoder-layer
``hidden_in`` chunks collected by ``01b_collect_layer_distill_data.py`` and
optimizes only the post-o_proj self-attention module output:

    teacher: original HF attention module output
    student: HAWPAttention output with the same q/k/v/o projections

Ranks stay fixed.  Only P_K / P_V / gamma are trainable.
"""

from __future__ import annotations

import argparse
import copy
import math
import multiprocessing as mp
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch

from hawp_laq.config import load_config
from hawp_laq.modeling.attention_hawp import HAWPAttention
from hawp_laq.modeling.modeling_llama_hawp import (
    _find_layers_and_attn,
    _install_opt_use_cache_bridge,
)
from hawp_laq.offline.layer_distill_trainer import (
    _active_update_blocks,
    _all_finite_tensors,
    _backoff_lrs,
    _backoff_torch_optimizer,
    _call_decoder_layer,
    _format_projector,
    _gamma_trainable,
    _hawp_modules,
    _is_finite_tensor,
    _loss_stats,
    _orth_err,
    _orthogonalize_module,
    _param_dtype,
    _prepare_trainable_projectors_fp32,
    _restore_hawp_state,
    _restore_riemannian_optimizer,
    _restore_torch_optimizer,
    _snapshot_hawp_state,
    _snapshot_riemannian_optimizer,
    _snapshot_torch_optimizer,
)
from hawp_laq.offline.low_rank_attention_optimizer_torch import (
    OptimConfig,
    RiemannianAdam,
    clip_by_global_norm,
    topk_preservation_losses,
    topk_recall_metrics,
)
from hawp_laq.runtime.generate import load_baseline_model
from hawp_laq.runtime.projector_bank import normalize_projector_data, rebuild_ranks_json
from hawp_laq.utils.io import load_pt, save_pt


@dataclass
class AttentionModuleDistillResult:
    metrics: dict[str, Any]
    best_step: int
    actual_steps: int
    stopped_early: bool
    best_eval_loss: float
    best_eval_mse: float
    best_eval_normalized: float
    best_eval_topk: float = 0.0
    best_eval_kl_topm: float = 0.0
    best_eval_logit_topm: float = 0.0
    best_eval_top_recalls: dict[str, float] | None = None


def _chunk_layers(layers: list[int], workers: int) -> list[list[int]]:
    if workers <= 1:
        return [layers]
    chunks = [[] for _ in range(workers)]
    for i, layer_idx in enumerate(layers):
        chunks[i % workers].append(layer_idx)
    return [c for c in chunks if c]


def _parse_devices(cfg, workers: int, gpus: list[str] | None) -> list[str]:
    items: list[str] = []
    for raw in gpus or []:
        items.extend(part.strip() for part in str(raw).split(",") if part.strip())
    if items:
        return [x if x == "cpu" or x.startswith("cuda") else f"cuda:{x}" for x in items]
    cfg_device = str(cfg.train.device)
    if cfg_device.startswith("cuda") and torch.cuda.is_available():
        n = max(1, torch.cuda.device_count())
        return [f"cuda:{i}" for i in range(min(workers, n))]
    return [cfg_device]


def _discover_layers(data_dir: Path, requested_layers: list[int] | None) -> list[int]:
    if requested_layers:
        return sorted(dict.fromkeys(int(x) for x in requested_layers))
    meta_path = data_dir / "meta.pt"
    if meta_path.exists():
        meta = load_pt(meta_path)
        n_layers = int(meta.get("n_layers", 0) or 0)
        if n_layers > 0:
            return list(range(n_layers))
    layers: list[int] = []
    for d in sorted(data_dir.glob("layer_*")):
        if not d.is_dir():
            continue
        try:
            layers.append(int(d.name.split("_", 1)[1]))
        except ValueError:
            pass
    return sorted(layers)


def _discover_chunk_paths(data_dir: Path, layer_idx: int) -> list[Path]:
    return sorted((data_dir / f"layer_{layer_idx}").glob("chunk_*.pt"))


def _load_hidden_in(
    path: Path,
    device: torch.device,
    dtype: torch.dtype,
    sample_batch_size: Optional[int],
) -> torch.Tensor:
    data = load_pt(path)
    hidden = data["hidden_in"]
    if sample_batch_size is not None and 0 < sample_batch_size < hidden.shape[0]:
        idx = torch.randperm(hidden.shape[0])[:sample_batch_size]
        hidden = hidden[idx]
    return hidden.to(device=device, dtype=dtype)


def _attn_attr(layer: torch.nn.Module) -> tuple[str, torch.nn.Module]:
    for attr in ("self_attn", "attention"):
        if hasattr(layer, attr):
            return attr, getattr(layer, attr)
    raise RuntimeError(f"No attention module found on {type(layer).__name__}")


def _output_tensor(output) -> torch.Tensor:
    if isinstance(output, tuple):
        return output[0]
    return output


class _ForwardCapture:
    def __init__(self) -> None:
        self.value: torch.Tensor | None = None

    def hook(self, _module, _inputs, output) -> None:
        self.value = _output_tensor(output)

    def pop(self) -> torch.Tensor:
        if self.value is None:
            raise RuntimeError("Attention forward hook did not capture an output")
        value = self.value
        self.value = None
        return value


class _RankSignalCapture:
    def __init__(self, modules: list[HAWPAttention]) -> None:
        self.modules = {m.layer_idx: m for m in modules}
        self.records: list[tuple[HAWPAttention, torch.Tensor, torch.Tensor]] = []
        self._old_callbacks: list[tuple[HAWPAttention, Any]] = []

    def __enter__(self):
        self.records.clear()
        self._old_callbacks = [(m, getattr(m, "_calib_callback", None)) for m in self.modules.values()]
        for module in self.modules.values():
            module._calib_callback = self._hook
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        for module, old_callback in self._old_callbacks:
            module._calib_callback = old_callback

    def _hook(self, layer_idx: int, query: torch.Tensor, key: torch.Tensor, _value: torch.Tensor) -> None:
        module = self.modules.get(layer_idx)
        if module is not None:
            self.records.append((module, query, key))


def _causal_valid_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    return torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool)).unsqueeze(0)


def _module_rank_stats(
    records: list[tuple[HAWPAttention, torch.Tensor, torch.Tensor]],
    *,
    eps_loss: float,
    lambda_topk: float,
    lambda_kl: float,
    lambda_logit_topm: float,
    topk_k: int,
    hard_neg_m: int,
    kl_top_m: int,
    topk_margin: float,
) -> dict[str, torch.Tensor]:
    if not records:
        zero = torch.zeros((), dtype=torch.float32)
        return {"topk": zero, "kl_topm": zero, "logit_topm": zero}

    losses: dict[str, torch.Tensor] | None = None
    for module, query, key in records:
        key_full = module._repeat_kv(key) if key.shape[1] != query.shape[1] else key
        bsz, n_heads, seq_len, head_dim = query.shape
        q = query.float()
        k = key_full.float()

        Z = q @ k.transpose(-1, -2)
        if not module.is_opt:
            Z = Z / math.sqrt(head_dim)

        pk = module.p_k[:, :module.r_k].to(device=q.device, dtype=q.dtype)
        q_lat = q @ pk
        k_lat = k @ pk
        Z_hat = (q_lat @ k_lat.transpose(-1, -2)) * module._compute_low_rank_logit_scale(q_lat)

        valid_mask = _causal_valid_mask(seq_len, q.device)
        cfg = OptimConfig(
            r_k=module.r_k,
            r_v=module.r_v,
            lambda_topk=lambda_topk,
            lambda_kl=lambda_kl,
            lambda_logit_topm=lambda_logit_topm,
            topk_k=topk_k,
            hard_neg_m=hard_neg_m,
            kl_top_m=kl_top_m,
            topk_margin=topk_margin,
            eps_loss=eps_loss,
        )
        module_losses = topk_preservation_losses(
            Z.reshape(bsz * n_heads, seq_len, seq_len),
            Z_hat.reshape(bsz * n_heads, seq_len, seq_len),
            valid_mask,
            cfg,
        )
        if losses is None:
            losses = {k: v for k, v in module_losses.items()}
        else:
            for key_name, value in module_losses.items():
                losses[key_name] = losses[key_name] + value

    assert losses is not None
    denom = float(len(records))
    return {key: value / denom for key, value in losses.items()}


@torch.no_grad()
def _module_rank_recalls(
    records: list[tuple[HAWPAttention, torch.Tensor, torch.Tensor]],
    metric_ks: tuple[int, ...],
) -> dict[str, float]:
    if not records:
        return {f"top{int(k)}_recall": 0.0 for k in metric_ks}
    sums: dict[str, float] = {}
    for module, query, key in records:
        key_full = module._repeat_kv(key) if key.shape[1] != query.shape[1] else key
        bsz, n_heads, seq_len, head_dim = query.shape
        q = query.float()
        k = key_full.float()
        Z = q @ k.transpose(-1, -2)
        if not module.is_opt:
            Z = Z / math.sqrt(head_dim)
        pk = module.p_k[:, :module.r_k].to(device=q.device, dtype=q.dtype)
        q_lat = q @ pk
        k_lat = k @ pk
        Z_hat = (q_lat @ k_lat.transpose(-1, -2)) * module._compute_low_rank_logit_scale(q_lat)
        recalls = topk_recall_metrics(
            Z.reshape(bsz * n_heads, seq_len, seq_len),
            Z_hat.reshape(bsz * n_heads, seq_len, seq_len),
            _causal_valid_mask(seq_len, q.device),
            metric_ks,
        )
        for key_name, value in recalls.items():
            sums[key_name] = sums.get(key_name, 0.0) + value
    return {key: value / len(records) for key, value in sums.items()}


def _make_student_layer(
    model: torch.nn.Module,
    teacher_layer: torch.nn.Module,
    layer_idx: int,
    projector_data: dict,
    cfg,
) -> torch.nn.Module:
    student_layer = copy.deepcopy(teacher_layer)
    attn_name, orig_attn = _attn_attr(student_layer)
    hawp_attn = HAWPAttention.from_attention(
        orig_attn,
        model=model,
        layer_idx=layer_idx,
        r_k=int(projector_data["r_k"]),
        r_v=int(projector_data["r_v"]),
        logit_scale_mode=cfg.hawp.logit_scale_mode,
        gamma_mode=cfg.hawp.gamma_mode,
        gamma_value=cfg.hawp.gamma_value,
        use_archive_k_ip_approx=cfg.hawp.use_archive_k_ip_approx,
    )
    if type(student_layer).__name__ != "OPTDecoderLayer":
        hawp_attn._hawp_parent_expects_2_outputs = True
    else:
        _install_opt_use_cache_bridge(student_layer, hawp_attn)
    setattr(student_layer, attn_name, hawp_attn)
    artifact_scale = projector_data.get("logit_scale_mode")
    if artifact_scale is not None and artifact_scale != cfg.hawp.logit_scale_mode:
        raise ValueError(
            f"Layer {layer_idx}: projector logit_scale_mode={artifact_scale!r} "
            f"does not match configured hawp.logit_scale_mode={cfg.hawp.logit_scale_mode!r}."
        )
    hawp_attn.load_projector_data(projector_data, strict=True)
    return student_layer


def _distill_cfg(cfg) -> dict[str, Any]:
    attn = cfg.attention_distill
    layer = cfg.layer_distill
    return {
        "n_steps": int(attn.n_steps),
        "sample_batch_size": attn.sample_batch_size,
        "eval_every": int(attn.eval_every),
        "eval_max_batches": attn.eval_max_batches,
        "optimizer": str(attn.optimizer).lower(),
        "lr": float(attn.lr),
        "lr_pk": float(attn.lr_pk),
        "lr_pv": float(attn.lr_pv),
        "lr_xi": float(attn.lr_xi),
        "beta1": float(attn.beta1),
        "beta2": float(attn.beta2),
        "grad_clip": float(attn.grad_clip),
        "train_pk": bool(getattr(attn, "train_pk", True)),
        "train_gamma": bool(attn.train_gamma),
        "gamma_min": float(attn.gamma_min),
        "gamma_max": float(getattr(layer, "gamma_max", 2.0)),
        "eps_loss": float(attn.eps_loss),
        "adam_eps": float(attn.adam_eps),
        "orthogonalize_every": int(attn.orthogonalize_every),
        "alternate_pk_pv": bool(getattr(layer, "alternate_pk_pv", True)),
        "finite_guard": bool(getattr(layer, "finite_guard", True)),
        "bad_step_patience": int(getattr(layer, "bad_step_patience", 20)),
        "lr_backoff": float(getattr(layer, "lr_backoff", 0.5)),
        "loss_mode": str(attn.loss_mode),
        "lambda_topk": float(getattr(attn, "lambda_topk", 0.0)),
        "lambda_kl": float(getattr(attn, "lambda_kl", 0.0)),
        "lambda_logit_topm": float(getattr(attn, "lambda_logit_topm", 0.0)),
        "topk_k": int(getattr(attn, "topk_k", 8)),
        "hard_neg_m": int(getattr(attn, "hard_neg_m", 32)),
        "kl_top_m": int(getattr(attn, "kl_top_m", 64)),
        "topk_margin": float(getattr(attn, "topk_margin", 0.05)),
        "topk_metric_ks": tuple(getattr(attn, "topk_metric_ks", (5, 10))),
        "early_stopping": bool(attn.early_stopping),
        "patience": int(attn.patience),
        "min_delta": float(attn.min_delta),
        "min_delta_mode": str(attn.min_delta_mode),
        "seed": int(attn.seed),
        "save_format": str(attn.save_format),
    }


def _attention_outputs(
    teacher_layer: torch.nn.Module,
    student_layer: torch.nn.Module,
    hidden_in: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    teacher_cap = _ForwardCapture()
    student_cap = _ForwardCapture()
    _, teacher_attn = _attn_attr(teacher_layer)
    _, student_attn = _attn_attr(student_layer)
    h_teacher = teacher_attn.register_forward_hook(teacher_cap.hook)
    h_student = student_attn.register_forward_hook(student_cap.hook)
    try:
        with torch.no_grad():
            _call_decoder_layer(teacher_layer, hidden_in)
            teacher_out = teacher_cap.pop().detach()
        _call_decoder_layer(student_layer, hidden_in)
        student_out = student_cap.pop()
    finally:
        h_teacher.remove()
        h_student.remove()
    return student_out, teacher_out


@torch.no_grad()
def _eval_attention_module(
    teacher_layer: torch.nn.Module,
    student_layer: torch.nn.Module,
    chunk_paths: list[Path],
    *,
    device: torch.device,
    dtype: torch.dtype,
    sample_batch_size: Optional[int],
    eval_max_batches: Optional[int],
    loss_mode: str,
    eps_loss: float,
    lambda_topk: float = 0.0,
    lambda_kl: float = 0.0,
    lambda_logit_topm: float = 0.0,
    topk_k: int = 8,
    hard_neg_m: int = 32,
    kl_top_m: int = 64,
    topk_margin: float = 0.05,
    topk_metric_ks: tuple[int, ...] = (5, 10),
) -> dict[str, float]:
    paths = chunk_paths[:eval_max_batches] if eval_max_batches is not None and eval_max_batches > 0 else chunk_paths
    total_diff = torch.zeros((), device=device, dtype=torch.float32)
    total_teacher = torch.zeros((), device=device, dtype=torch.float32)
    total_topk = torch.zeros((), device=device, dtype=torch.float32)
    total_kl_topm = torch.zeros((), device=device, dtype=torch.float32)
    total_logit_topm = torch.zeros((), device=device, dtype=torch.float32)
    total_count = 0
    total_rank_batches = 0
    recall_sums: dict[str, float] = {}
    modules = _hawp_modules(student_layer)
    for path in paths:
        hidden_in = _load_hidden_in(path, device, dtype, sample_batch_size)
        with _RankSignalCapture(modules) as rank_cap:
            student, teacher = _attention_outputs(teacher_layer, student_layer, hidden_in)
        total_diff = total_diff + (student.float() - teacher.float()).pow(2).sum()
        total_teacher = total_teacher + teacher.float().pow(2).sum()
        total_count += teacher.numel()
        if lambda_topk != 0.0 or lambda_kl != 0.0 or lambda_logit_topm != 0.0:
            rank_stats = _module_rank_stats(
                rank_cap.records,
                eps_loss=eps_loss,
                lambda_topk=lambda_topk,
                lambda_kl=lambda_kl,
                lambda_logit_topm=lambda_logit_topm,
                topk_k=topk_k,
                hard_neg_m=hard_neg_m,
                kl_top_m=kl_top_m,
                topk_margin=topk_margin,
            )
            total_topk = total_topk + rank_stats["topk"].to(total_topk.device)
            total_kl_topm = total_kl_topm + rank_stats["kl_topm"].to(total_kl_topm.device)
            total_logit_topm = total_logit_topm + rank_stats["logit_topm"].to(total_logit_topm.device)
            total_rank_batches += 1
        for key, value in _module_rank_recalls(rank_cap.records, topk_metric_ks).items():
            recall_sums[key] = recall_sums.get(key, 0.0) + value
    mse = total_diff / max(total_count, 1)
    normalized = total_diff / (total_teacher + eps_loss)
    loss = mse if loss_mode == "absolute" else normalized
    if total_rank_batches > 0:
        L_topk = total_topk / total_rank_batches
        L_kl_topm = total_kl_topm / total_rank_batches
        L_logit_topm = total_logit_topm / total_rank_batches
        loss = (
            loss
            + lambda_topk * L_topk
            + lambda_kl * L_kl_topm
            + lambda_logit_topm * L_logit_topm
        )
    else:
        L_topk = total_topk
        L_kl_topm = total_kl_topm
        L_logit_topm = total_logit_topm
    recall_den = max(len(paths), 1)
    return {
        "loss": float(loss.detach().cpu()),
        "mse": float(mse.detach().cpu()),
        "normalized": float(normalized.detach().cpu()),
        "topk": float(L_topk.detach().cpu()),
        "kl_topm": float(L_kl_topm.detach().cpu()),
        "logit_topm": float(L_logit_topm.detach().cpu()),
        **{key: value / recall_den for key, value in recall_sums.items()},
    }


def _refine_attention_module(
    teacher_layer: torch.nn.Module,
    student_layer: torch.nn.Module,
    chunk_paths: list[Path],
    *,
    device: str,
    n_steps: int,
    sample_batch_size: Optional[int],
    eval_every: int,
    eval_max_batches: Optional[int],
    optimizer: str,
    lr: float,
    lr_pk: float,
    lr_pv: float,
    lr_xi: float,
    beta1: float,
    beta2: float,
    grad_clip: float,
    train_pk: bool,
    train_gamma: bool,
    gamma_min: float,
    gamma_max: float,
    eps_loss: float,
    adam_eps: float,
    orthogonalize_every: int,
    alternate_pk_pv: bool,
    finite_guard: bool,
    bad_step_patience: int,
    lr_backoff: float,
    loss_mode: str,
    lambda_topk: float,
    lambda_kl: float,
    lambda_logit_topm: float,
    topk_k: int,
    hard_neg_m: int,
    kl_top_m: int,
    topk_margin: float,
    topk_metric_ks: tuple[int, ...],
    early_stopping: bool,
    patience: int,
    min_delta: float,
    min_delta_mode: str,
    seed: int,
) -> AttentionModuleDistillResult:
    if not chunk_paths:
        raise FileNotFoundError("No layer distill chunk_*.pt files found")
    if eval_every <= 0:
        raise ValueError("eval_every must be > 0")
    optimizer = optimizer.lower()
    if optimizer not in ("adam", "riemannian_adam"):
        raise ValueError(f"optimizer must be 'adam' or 'riemannian_adam', got {optimizer!r}")

    torch.manual_seed(seed)
    rng = __import__("random").Random(seed)
    dev = torch.device(device)
    teacher_layer.to(dev).eval()
    student_layer.to(dev).eval()
    for p in teacher_layer.parameters():
        p.requires_grad_(False)

    dtype = _param_dtype(student_layer)
    modules = _prepare_trainable_projectors_fp32(
        student_layer,
        train_pk=train_pk,
        train_gamma=train_gamma,
        gamma_min=gamma_min,
        gamma_max=gamma_max,
    )
    if not modules:
        raise RuntimeError("Student layer has no HAWPAttention module")
    trainable_params = [
        p for module in modules for p in (module.p_k, module.p_v, module.gamma)
        if p.requires_grad
    ]
    if not trainable_params:
        raise RuntimeError("Student layer has no trainable low-rank projector parameters")

    adam_optimizer = (
        torch.optim.Adam(trainable_params, lr=lr, betas=(beta1, beta2), eps=adam_eps)
        if optimizer == "adam"
        else None
    )
    pk_opts = {
        m.layer_idx: RiemannianAdam((m.head_dim, m.r_k), dev, torch.float32, lr_pk, beta1, beta2, adam_eps)
        for m in modules if train_pk and m.r_k < m.head_dim
    }
    pv_opts = {
        m.layer_idx: RiemannianAdam((m.head_dim, m.r_v), dev, torch.float32, lr_pv, beta1, beta2, adam_eps)
        for m in modules if m.r_v < m.head_dim
    }
    gamma_params = [m.gamma for m in modules if _gamma_trainable(m, train_gamma)]
    gamma_optimizer = (
        torch.optim.Adam(gamma_params, lr=lr_xi, betas=(beta1, beta2), eps=adam_eps)
        if optimizer == "riemannian_adam" and gamma_params
        else None
    )

    best_eval_loss = float("inf")
    best_eval_mse = 0.0
    best_eval_normalized = 0.0
    best_eval_topk = 0.0
    best_eval_kl_topm = 0.0
    best_eval_logit_topm = 0.0
    best_eval_top_recalls: dict[str, float] = {}
    best_state = _snapshot_hawp_state(student_layer)
    best_step = 0
    stale_checks = 0
    bad_steps = 0
    stopped_early = False
    history: list[dict[str, Any]] = []
    step = 0

    for step in range(1, n_steps + 1):
        path = chunk_paths[rng.randrange(len(chunk_paths))]
        hidden_in = _load_hidden_in(path, dev, dtype, sample_batch_size)

        state_before = _snapshot_hawp_state(student_layer)
        pk_state = {k: _snapshot_riemannian_optimizer(v) for k, v in pk_opts.items()}
        pv_state = {k: _snapshot_riemannian_optimizer(v) for k, v in pv_opts.items()}
        gamma_state = _snapshot_torch_optimizer(gamma_optimizer)
        adam_state = _snapshot_torch_optimizer(adam_optimizer)

        with _RankSignalCapture(modules) as rank_cap:
            student, teacher = _attention_outputs(teacher_layer, student_layer, hidden_in)
        if finite_guard and not _all_finite_tensors([student, teacher]):
            _restore_hawp_state(student_layer, state_before)
            bad_steps += 1
            for opt_obj in pk_opts.values():
                opt_obj.lr *= lr_backoff
            for opt_obj in pv_opts.values():
                opt_obj.lr *= lr_backoff
            _backoff_torch_optimizer(adam_optimizer, lr_backoff)
            _backoff_lrs(None, None, gamma_optimizer, lr_backoff)
            print(f"[warn] step={step:04d} skipped: non-finite attention output", flush=True)
            if bad_steps >= bad_step_patience:
                stopped_early = True
                break
            continue

        loss, _mse, _normalized = _loss_stats(student.float(), teacher.float(), loss_mode, eps_loss)
        if lambda_topk != 0.0 or lambda_kl != 0.0 or lambda_logit_topm != 0.0:
            rank_stats = _module_rank_stats(
                rank_cap.records,
                eps_loss=eps_loss,
                lambda_topk=lambda_topk,
                lambda_kl=lambda_kl,
                lambda_logit_topm=lambda_logit_topm,
                topk_k=topk_k,
                hard_neg_m=hard_neg_m,
                kl_top_m=kl_top_m,
                topk_margin=topk_margin,
            )
            if rank_cap.records:
                loss = (
                    loss
                    + lambda_topk * rank_stats["topk"].to(loss.device)
                    + lambda_kl * rank_stats["kl_topm"].to(loss.device)
                    + lambda_logit_topm * rank_stats["logit_topm"].to(loss.device)
                )
        if not loss.requires_grad:
            raise RuntimeError("Attention module distill loss has no gradient path")
        if finite_guard and not _is_finite_tensor(loss):
            _restore_hawp_state(student_layer, state_before)
            bad_steps += 1
            for opt_obj in pk_opts.values():
                opt_obj.lr *= lr_backoff
            for opt_obj in pv_opts.values():
                opt_obj.lr *= lr_backoff
            _backoff_torch_optimizer(adam_optimizer, lr_backoff)
            _backoff_lrs(None, None, gamma_optimizer, lr_backoff)
            print(f"[warn] step={step:04d} skipped: non-finite loss", flush=True)
            if bad_steps >= bad_step_patience:
                stopped_early = True
                break
            continue

        if optimizer == "adam":
            adam_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grads = [p.grad for p in trainable_params]
            if finite_guard and not _all_finite_tensors(grads):
                _restore_hawp_state(student_layer, state_before)
                _restore_torch_optimizer(adam_optimizer, adam_state)
                bad_steps += 1
                _backoff_torch_optimizer(adam_optimizer, lr_backoff)
                print(f"[warn] step={step:04d} skipped: non-finite Adam gradient", flush=True)
                if bad_steps >= bad_step_patience:
                    stopped_early = True
                    break
                continue
            torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)
            adam_optimizer.step()
        else:
            update_pk, update_pv = _active_update_blocks(modules, step, alternate_pk_pv)
            update_pk = update_pk and train_pk
            if not train_pk and not update_pv:
                update_pv = True
            grad_params: list[torch.nn.Parameter] = []
            grad_specs: list[tuple[str, HAWPAttention]] = []
            for module in modules:
                if update_pk and module.r_k < module.head_dim:
                    grad_params.append(module.p_k)
                    grad_specs.append(("pk", module))
                if update_pv and module.r_v < module.head_dim:
                    grad_params.append(module.p_v)
                    grad_specs.append(("pv", module))
            if gamma_params and (update_pk or not train_pk):
                for module in modules:
                    if _gamma_trainable(module, train_gamma):
                        grad_params.append(module.gamma)
                        grad_specs.append(("gamma", module))

            grads = list(torch.autograd.grad(loss, grad_params, allow_unused=True))
            sliced_grads: list[torch.Tensor | None] = []
            for (kind, module), grad in zip(grad_specs, grads):
                if grad is None:
                    sliced_grads.append(None)
                elif kind == "pk":
                    sliced_grads.append(grad[:, :module.r_k])
                elif kind == "pv":
                    sliced_grads.append(grad[:, :module.r_v])
                else:
                    sliced_grads.append(grad)

            if finite_guard and not _all_finite_tensors(sliced_grads):
                _restore_hawp_state(student_layer, state_before)
                for k, v in pk_opts.items():
                    _restore_riemannian_optimizer(v, pk_state.get(k))
                for k, v in pv_opts.items():
                    _restore_riemannian_optimizer(v, pv_state.get(k))
                _restore_torch_optimizer(gamma_optimizer, gamma_state)
                bad_steps += 1
                for opt_obj in pk_opts.values():
                    opt_obj.lr *= lr_backoff
                for opt_obj in pv_opts.values():
                    opt_obj.lr *= lr_backoff
                _backoff_lrs(None, None, gamma_optimizer, lr_backoff)
                print(f"[warn] step={step:04d} skipped: non-finite Riemannian gradient", flush=True)
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
                    if kind == "pk":
                        pk_opts[module.layer_idx].step_(module.p_k[:, :module.r_k], grad.float())
                    elif kind == "pv":
                        pv_opts[module.layer_idx].step_(module.p_v[:, :module.r_v], grad.float())
                    else:
                        module.gamma.grad = grad.float()
            if gamma_optimizer is not None and any(kind == "gamma" for kind, _ in grad_specs):
                gamma_optimizer.step()

        with torch.no_grad():
            for module in modules:
                module.gamma.clamp_(min=gamma_min, max=gamma_max)
                if optimizer == "adam" and step % orthogonalize_every == 0:
                    _orthogonalize_module(module)

        if finite_guard:
            values = []
            for module in modules:
                values.extend([module.p_k[:, :module.r_k], module.p_v[:, :module.r_v], module.gamma])
            if not _all_finite_tensors(values):
                _restore_hawp_state(student_layer, state_before)
                if optimizer == "adam":
                    _restore_torch_optimizer(adam_optimizer, adam_state)
                else:
                    for k, v in pk_opts.items():
                        _restore_riemannian_optimizer(v, pk_state.get(k))
                    for k, v in pv_opts.items():
                        _restore_riemannian_optimizer(v, pv_state.get(k))
                    _restore_torch_optimizer(gamma_optimizer, gamma_state)
                bad_steps += 1
                for opt_obj in pk_opts.values():
                    opt_obj.lr *= lr_backoff
                for opt_obj in pv_opts.values():
                    opt_obj.lr *= lr_backoff
                _backoff_lrs(None, None, gamma_optimizer, lr_backoff)
                print(f"[warn] step={step:04d} skipped: non-finite parameters after update", flush=True)
                if bad_steps >= bad_step_patience:
                    stopped_early = True
                    break
                continue

        bad_steps = 0
        do_eval = step == 1 or step % eval_every == 0 or step == n_steps
        if do_eval:
            eval_stats = _eval_attention_module(
                teacher_layer,
                student_layer,
                chunk_paths,
                device=dev,
                dtype=dtype,
                sample_batch_size=sample_batch_size,
                eval_max_batches=eval_max_batches,
                loss_mode=loss_mode,
                eps_loss=eps_loss,
                lambda_topk=lambda_topk,
                lambda_kl=lambda_kl,
                lambda_logit_topm=lambda_logit_topm,
                topk_k=topk_k,
                hard_neg_m=hard_neg_m,
                kl_top_m=kl_top_m,
                topk_margin=topk_margin,
                topk_metric_ks=topk_metric_ks,
            )
            gamma_val = float(modules[0].gamma.detach().float().cpu())
            orth_k = max((_orth_err(m.p_k, m.r_k) for m in modules), default=0.0)
            orth_v = max((_orth_err(m.p_v, m.r_v) for m in modules), default=0.0)
            eval_loss = eval_stats["loss"]
            eval_mse = eval_stats["mse"]
            eval_norm = eval_stats["normalized"]
            eval_topk = eval_stats["topk"]
            eval_kl_topm = eval_stats["kl_topm"]
            eval_logit_topm = eval_stats["logit_topm"]
            history.append({
                "step": step,
                "eval_loss": eval_loss,
                "eval_mse": eval_mse,
                "eval_normalized": eval_norm,
                "eval_topk": eval_topk,
                "eval_kl_topm": eval_kl_topm,
                "eval_logit_topm": eval_logit_topm,
                "train_pk": train_pk,
                **{k: v for k, v in eval_stats.items() if k.startswith("top") and k.endswith("_recall")},
                "gamma": gamma_val,
                "orthK": orth_k,
                "orthV": orth_v,
                "optimizer": optimizer,
                "bad_steps": bad_steps,
            })
            print(
                f"step={step:04d} attn_module_loss={eval_loss:.6e} "
                f"mse={eval_mse:.6e} normalized={eval_norm:.6e} "
                f"topk={eval_topk:.6e} kl_topm={eval_kl_topm:.6e} "
                f"gamma={gamma_val:.5f} orthK={orth_k:.2e} orthV={orth_v:.2e}",
                flush=True,
            )

            if eval_loss < best_eval_loss:
                prev_best = best_eval_loss
                best_eval_loss = eval_loss
                best_eval_mse = eval_mse
                best_eval_normalized = eval_norm
                best_eval_topk = eval_topk
                best_eval_kl_topm = eval_kl_topm
                best_eval_logit_topm = eval_logit_topm
                best_eval_top_recalls = {
                    k: float(v)
                    for k, v in eval_stats.items()
                    if k.startswith("top") and k.endswith("_recall")
                }
                best_state = _snapshot_hawp_state(student_layer)
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

    _restore_hawp_state(student_layer, best_state)
    student_layer.eval()
    return AttentionModuleDistillResult(
        metrics={
            "attention_module_distill": history,
            "train_pk": train_pk,
            "best_eval_topk": best_eval_topk,
            "best_eval_kl_topm": best_eval_kl_topm,
            "best_eval_logit_topm": best_eval_logit_topm,
            "best_eval_top_recalls": best_eval_top_recalls,
        },
        best_step=best_step,
        actual_steps=step,
        stopped_early=stopped_early,
        best_eval_loss=best_eval_loss,
        best_eval_mse=best_eval_mse,
        best_eval_normalized=best_eval_normalized,
        best_eval_topk=best_eval_topk,
        best_eval_kl_topm=best_eval_kl_topm,
        best_eval_logit_topm=best_eval_logit_topm,
        best_eval_top_recalls=best_eval_top_recalls,
    )


def _save_projector(
    student_layer: torch.nn.Module,
    original_projector: dict,
    result: AttentionModuleDistillResult,
    out_path: Path,
    *,
    save_format: str,
    source_path: Path,
) -> None:
    modules = _hawp_modules(student_layer)
    if len(modules) != 1:
        raise RuntimeError(f"Expected exactly one HAWPAttention in decoder layer, found {len(modules)}")
    module = modules[0]
    r_k = int(original_projector["r_k"])
    r_v = int(original_projector["r_v"])
    out = dict(original_projector)
    out["p_k"] = _format_projector(
        module.p_k,
        head_dim=module.head_dim,
        rank=r_k,
        original=original_projector["p_k"],
        save_format=save_format,
    )
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
    out["attention_module_distill"] = {
        "source_projector": str(source_path),
        "logit_scale_mode": module.logit_scale_mode,
        "best_step": result.best_step,
        "actual_steps": result.actual_steps,
        "stopped_early": result.stopped_early,
        "best_eval_loss": result.best_eval_loss,
        "best_eval_mse": result.best_eval_mse,
        "best_eval_normalized": result.best_eval_normalized,
        "best_eval_topk": result.best_eval_topk,
        "best_eval_kl_topm": result.best_eval_kl_topm,
        "best_eval_logit_topm": result.best_eval_logit_topm,
        "best_eval_top_recalls": result.best_eval_top_recalls or {},
        "metrics": result.metrics,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_pt(out, out_path)


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
    layers = _find_layers_and_attn(model)
    if layer_idx >= len(layers):
        print(f"[attn_module_distill] layer {layer_idx}: model has only {len(layers)} layers, skipping", flush=True)
        return False
    chunk_paths = _discover_chunk_paths(data_dir, layer_idx)
    if not chunk_paths:
        print(f"[attn_module_distill] layer {layer_idx}: no hidden_in chunks, skipping", flush=True)
        return False
    projector_path = input_dir / f"layer_{layer_idx}" / "projector.pt"
    if not projector_path.exists():
        print(f"[attn_module_distill] layer {layer_idx}: no projector at {projector_path}, skipping", flush=True)
        return False

    projector_data = normalize_projector_data(load_pt(projector_path), layer_idx)
    if "r_k" not in projector_data or "r_v" not in projector_data:
        print(f"[attn_module_distill] layer {layer_idx}: projector missing r_k/r_v, skipping", flush=True)
        return False

    teacher_layer = layers[layer_idx][1]
    student_layer = _make_student_layer(model, teacher_layer, layer_idx, projector_data, cfg)
    modules = _hawp_modules(student_layer)
    if not modules:
        print(f"[attn_module_distill] layer {layer_idx}: no HAWPAttention after conversion, skipping", flush=True)
        return False
    module = modules[0]
    if module.r_k >= module.head_dim and module.r_v >= module.head_dim:
        print(
            f"[attn_module_distill] layer {layer_idx}: full-rank "
            f"(r_k={module.r_k}, r_v={module.r_v}, head_dim={module.head_dim}), skipping refine",
            flush=True,
        )
        return False

    params = _distill_cfg(cfg)
    print(
        f"\n[attn_module_distill] layer {layer_idx}: chunks={len(chunk_paths)} "
        f"r_k={module.r_k} r_v={module.r_v} head_dim={module.head_dim} "
        f"train_pk={params['train_pk']} device={device}",
        flush=True,
    )
    result = _refine_attention_module(
        teacher_layer,
        student_layer,
        chunk_paths,
        device=device,
        n_steps=params["n_steps"],
        sample_batch_size=params["sample_batch_size"],
        eval_every=params["eval_every"],
        eval_max_batches=params["eval_max_batches"],
        optimizer=params["optimizer"],
        lr=params["lr"],
        lr_pk=params["lr_pk"],
        lr_pv=params["lr_pv"],
        lr_xi=params["lr_xi"],
        beta1=params["beta1"],
        beta2=params["beta2"],
        grad_clip=params["grad_clip"],
        train_pk=params["train_pk"],
        train_gamma=params["train_gamma"],
        gamma_min=params["gamma_min"],
        gamma_max=params["gamma_max"],
        eps_loss=params["eps_loss"],
        adam_eps=params["adam_eps"],
        orthogonalize_every=params["orthogonalize_every"],
        alternate_pk_pv=params["alternate_pk_pv"],
        finite_guard=params["finite_guard"],
        bad_step_patience=params["bad_step_patience"],
        lr_backoff=params["lr_backoff"],
        loss_mode=params["loss_mode"],
        lambda_topk=params["lambda_topk"],
        lambda_kl=params["lambda_kl"],
        lambda_logit_topm=params["lambda_logit_topm"],
        topk_k=params["topk_k"],
        hard_neg_m=params["hard_neg_m"],
        kl_top_m=params["kl_top_m"],
        topk_margin=params["topk_margin"],
        topk_metric_ks=params["topk_metric_ks"],
        early_stopping=params["early_stopping"],
        patience=params["patience"],
        min_delta=params["min_delta"],
        min_delta_mode=params["min_delta_mode"],
        seed=params["seed"] + layer_idx,
    )
    out_path = output_dir / f"layer_{layer_idx}" / "projector.pt"
    _save_projector(
        student_layer,
        projector_data,
        result,
        out_path,
        save_format=params["save_format"],
        source_path=projector_path,
    )
    print(
        f"[save] {out_path}  r_k={module.r_k} r_v={module.r_v} "
        f"best_step={result.best_step} best_loss={result.best_eval_loss:.6e} "
        f"stopped_early={result.stopped_early}",
        flush=True,
    )
    del student_layer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
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
    cfg = load_config(config_path)
    cfg.train.device = device
    model, _tokenizer, _device = load_baseline_model(cfg)
    for p in model.parameters():
        p.requires_grad_(False)
    saved: list[int] = []
    for layer_idx in layer_ids:
        ok = _refine_one_layer(
            model,
            cfg,
            layer_idx,
            data_dir=Path(data_dir_str),
            input_dir=Path(input_dir_str),
            output_dir=Path(output_dir_str),
            device=device,
        )
        if ok:
            saved.append(layer_idx)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return saved


def _default_output_dir(input_dir: Path) -> Path:
    return input_dir.with_name(input_dir.name + "_attn_module_distill")


def run(
    config_path: str | Path,
    *,
    layers: list[int] | None,
    data_dir: str | None,
    input_dir: str | None,
    output_dir: str | None,
    in_place: bool,
    clean_output_dir: bool,
    workers: int,
    gpus: list[str] | None,
) -> None:
    cfg = load_config(config_path)
    data_path = Path(data_dir) if data_dir else Path(cfg.layer_distill.data_dir)
    if input_dir:
        input_path = Path(input_dir)
    else:
        input_path = Path(cfg.attention_distill.input_dir)
        if not input_path.exists():
            input_path = Path(cfg.projector.output_dir)
    output_path = input_path if in_place else (Path(output_dir) if output_dir else _default_output_dir(input_path))

    if not data_path.exists():
        raise FileNotFoundError(
            f"Layer hidden_in data not found: {data_path}. "
            "Run scripts/01b_collect_layer_distill_data.py first."
        )
    if not input_path.exists():
        raise FileNotFoundError(f"Projector input dir not found: {input_path}")

    if clean_output_dir and output_path.exists() and output_path != input_path:
        print(f"[attn_module_distill] --clean-output-dir: removing {output_path}")
        shutil.rmtree(output_path)
    if output_path != input_path:
        shutil.copytree(input_path, output_path, dirs_exist_ok=True)
    output_path.mkdir(parents=True, exist_ok=True)

    layer_ids = _discover_layers(data_path, layers)
    workers = max(1, min(int(workers), len(layer_ids) if layer_ids else 1))
    devices = _parse_devices(cfg, workers, gpus)
    print("=" * 70)
    print(f"[attn_module_distill] data_dir={data_path}")
    print(f"[attn_module_distill] input_dir={input_path}")
    print(f"[attn_module_distill] output_dir={output_path}")
    print(f"[attn_module_distill] layers={layer_ids}")
    print(f"[attn_module_distill] workers={workers} devices={devices}")
    print("=" * 70)

    saved_layers: list[int] = []
    if workers <= 1:
        device = devices[0] if devices else str(cfg.train.device)
        saved_layers = _worker_refine_layers(
            str(config_path),
            layer_ids,
            str(data_path),
            str(input_path),
            str(output_path),
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
                    str(data_path),
                    str(input_path),
                    str(output_path),
                    device,
                ))
            for fut in as_completed(futures):
                saved_layers.extend(fut.result())

    if saved_layers:
        ranks_path = rebuild_ranks_json(output_path)
        print(f"\n[attn_module_distill] refined layers={sorted(saved_layers)}")
        print(f"[attn_module_distill] rebuilt ranks.json at {ranks_path}")
    else:
        print("\n[attn_module_distill] no projectors were refined")


def main() -> None:
    ap = argparse.ArgumentParser(description="Refine projectors with full attention-module output distillation")
    ap.add_argument("config")
    ap.add_argument("--layers", type=int, nargs="*", default=None)
    ap.add_argument("--data-dir", default=None, help="Layer hidden_in data dir from 01b; defaults to layer_distill.data_dir")
    ap.add_argument("--input-dir", default=None, help="Initial projector dir; defaults to attention_distill.input_dir/projector.output_dir")
    ap.add_argument("--output-dir", default=None, help="Output projector dir; defaults to <input_dir>_attn_module_distill")
    ap.add_argument("--in-place", action="store_true")
    ap.add_argument("--clean-output-dir", action="store_true")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--gpus", nargs="*", default=None, help="GPU ids/devices: --gpus 0 1 2 or --gpus 0,1,2")
    args = ap.parse_args()
    run(
        args.config,
        layers=args.layers,
        data_dir=args.data_dir,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        in_place=args.in_place,
        clean_output_dir=args.clean_output_dir,
        workers=args.workers,
        gpus=args.gpus,
    )


if __name__ == "__main__":
    main()
