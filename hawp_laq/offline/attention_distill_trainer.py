from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from hawp_laq.offline.low_rank_attention_optimizer_torch import (
    RiemannianAdam,
    clip_by_global_norm,
    inv_softplus,
    qr_retraction,
    sample_rows,
    stable_softmax,
)
from hawp_laq.offline.projector_trainer import ProjectorTrainer


Tensor = torch.Tensor


@dataclass
class AttentionDistillResult:
    p_k: Tensor
    p_v: Tensor
    gamma: Tensor
    metrics: dict
    best_step: int
    actual_steps: int
    stopped_early: bool
    best_eval_loss: float
    best_eval_mse: float
    best_eval_normalized: float


def _projector_basis(data: dict, key: str, head_dim: int, rank: int) -> Tensor:
    p = data.get(key)
    if p is None:
        raise ValueError(f"projector data missing {key!r}")
    if p.ndim != 2 or p.shape[0] != head_dim:
        raise ValueError(
            f"{key} shape must be ({head_dim}, {head_dim}) or ({head_dim}, {rank}), "
            f"got {tuple(p.shape)}"
        )
    if p.shape[1] == head_dim:
        return p[:, :rank].clone()
    if p.shape[1] == rank:
        return p.clone()
    raise ValueError(
        f"{key} shape must be ({head_dim}, {head_dim}) or ({head_dim}, {rank}), "
        f"got {tuple(p.shape)}"
    )


def _gamma_from_data(data: dict, gamma_min: float, device: torch.device, dtype: torch.dtype) -> Tensor:
    raw = data.get("gamma")
    if raw is None:
        raw = data.get("gamma_v", data.get("gamma_k", torch.ones(1)))
    gamma = torch.as_tensor(raw, device=device, dtype=dtype).reshape(())
    return gamma.clamp_min(gamma_min + 1e-8)


def _orth_err(P: Tensor) -> float:
    eye = torch.eye(P.shape[1], device=P.device, dtype=P.dtype)
    return float(torch.linalg.norm(P.transpose(0, 1) @ P - eye).detach().cpu())


def _sample_batch_indices(num_items: int, sample_batch_size: Optional[int], device: torch.device) -> Optional[Tensor]:
    if sample_batch_size is None or sample_batch_size >= num_items:
        return None
    return torch.randperm(num_items, device=device)[:sample_batch_size]


def _make_causal_masks(seq_len: int, row_idx: Optional[Tensor], device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
    valid = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
    if row_idx is not None:
        valid = valid[row_idx, :]
    additive = torch.where(
        valid,
        torch.zeros_like(valid, dtype=dtype),
        torch.full_like(valid, -1e4, dtype=dtype),
    )
    return additive.unsqueeze(0), valid.unsqueeze(0)


def _attention_output_stats(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    P_K: Tensor,
    P_V: Tensor,
    gamma: Tensor,
    *,
    row_idx: Optional[Tensor],
) -> tuple[Tensor, Tensor, int]:
    seq_len = Q.shape[1]
    d_h = Q.shape[-1]
    additive_mask, valid_mask = _make_causal_masks(seq_len, row_idx, Q.device, Q.dtype)

    Qs = Q if row_idx is None else Q[:, row_idx, :]

    Z = (Qs @ K.transpose(-1, -2)) / math.sqrt(d_h)
    A = stable_softmax(Z, additive_mask, valid_mask)
    O = A @ V

    Ql = Qs @ P_K
    Kl = K @ P_K
    Vl = V @ P_V

    Z_hat = (gamma / math.sqrt(P_K.shape[1])) * (Ql @ Kl.transpose(-1, -2))
    A_hat = stable_softmax(Z_hat, additive_mask, valid_mask)
    O_hat = (A_hat @ Vl) @ P_V.transpose(0, 1)

    diff_sq = (O_hat - O).pow(2).sum()
    teacher_sq = O.pow(2).sum()
    return diff_sq, teacher_sq, O.numel()


def _loss_from_stats(diff_sq: Tensor, teacher_sq: Tensor, count: int, loss_mode: str, eps_loss: float) -> Tensor:
    if loss_mode == "absolute":
        return diff_sq / max(count, 1)
    if loss_mode == "normalized":
        return diff_sq / (teacher_sq + eps_loss)
    raise ValueError(f"attention distill loss_mode must be 'absolute' or 'normalized', got {loss_mode!r}")


def _eval_projector(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    P_K: Tensor,
    P_V: Tensor,
    gamma: Tensor,
    *,
    loss_mode: str,
    eps_loss: float,
    eval_batch_size: int,
    eval_max_batches: Optional[int],
) -> dict:
    num_items = Q.shape[0]
    if eval_max_batches is not None and eval_max_batches > 0:
        limit = min(num_items, eval_batch_size * eval_max_batches)
        eval_indices = torch.arange(limit, device=Q.device)
    else:
        eval_indices = torch.arange(num_items, device=Q.device)

    total_diff = torch.zeros((), device=Q.device, dtype=Q.dtype)
    total_teacher = torch.zeros((), device=Q.device, dtype=Q.dtype)
    total_count = 0
    for start in range(0, eval_indices.numel(), eval_batch_size):
        idx = eval_indices[start:start + eval_batch_size]
        diff_sq, teacher_sq, count = _attention_output_stats(
            Q[idx], K[idx], V[idx], P_K, P_V, gamma, row_idx=None,
        )
        total_diff = total_diff + diff_sq
        total_teacher = total_teacher + teacher_sq
        total_count += count

    mse = total_diff / max(total_count, 1)
    normalized = total_diff / (total_teacher + eps_loss)
    loss = _loss_from_stats(total_diff, total_teacher, total_count, loss_mode, eps_loss)
    return {
        "loss": float(loss.detach().cpu()),
        "mse": float(mse.detach().cpu()),
        "normalized": float(normalized.detach().cpu()),
    }


def refine_attention_output_projector(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    projector_data: dict,
    *,
    d_model: int,
    n_heads: int,
    device: str = "cpu",
    n_steps: int = 300,
    sample_batch_size: Optional[int] = 128,
    row_batch_size: Optional[int] = 128,
    eval_every: int = 25,
    eval_batch_size: int = 32,
    eval_max_batches: Optional[int] = 16,
    lr_pk: float = 1e-3,
    lr_pv: float = 1e-3,
    lr_xi: float = 1e-3,
    optimizer: str = "riemannian_adam",
    lr: float = 1e-3,
    orthogonalize_every: int = 1,
    beta1: float = 0.9,
    beta2: float = 0.99,
    grad_clip: float = 1.0,
    gamma_min: float = 1e-4,
    eps_loss: float = 1e-8,
    adam_eps: float = 1e-8,
    train_gamma: bool = True,
    loss_mode: str = "absolute",
    early_stopping: bool = True,
    patience: int = 5,
    min_delta: float = 1e-5,
    min_delta_mode: str = "relative",
    seed: int = 0,
    verbose: bool = True,
) -> AttentionDistillResult:
    if min_delta_mode not in ("relative", "absolute"):
        raise ValueError(f"min_delta_mode must be 'relative' or 'absolute', got {min_delta_mode!r}")
    if eval_every <= 0:
        raise ValueError("eval_every must be > 0")
    if eval_batch_size <= 0:
        raise ValueError("eval_batch_size must be > 0")
    optimizer = optimizer.lower()
    if optimizer not in ("riemannian_adam", "adam"):
        raise ValueError(
            f"attention distill optimizer must be 'riemannian_adam' or 'adam', got {optimizer!r}"
        )
    if orthogonalize_every <= 0:
        raise ValueError("orthogonalize_every must be > 0")

    torch.manual_seed(seed)
    dev = torch.device(device)
    dtype = torch.float32
    head_dim = d_model // n_heads
    r_k = int(projector_data["r_k"])
    r_v = int(projector_data["r_v"])

    q = q.to(dev, dtype=dtype)
    k = k.to(dev, dtype=dtype)
    v = v.to(dev, dtype=dtype)
    Q, d_h = ProjectorTrainer._to_optim_input(q, n_heads, d_model, head_dim)
    K, _ = ProjectorTrainer._to_optim_input(k, n_heads, d_model, head_dim)
    V, _ = ProjectorTrainer._to_optim_input(v, n_heads, d_model, head_dim)
    if Q.shape != K.shape or Q.shape != V.shape:
        raise ValueError(f"Q/K/V shapes must match after reshape, got {Q.shape}, {K.shape}, {V.shape}")
    if r_k > d_h or r_v > d_h:
        raise ValueError(f"ranks exceed head_dim={d_h}: r_k={r_k}, r_v={r_v}")

    p_k_init = _projector_basis(projector_data, "p_k", d_h, r_k).to(dev, dtype=dtype)
    p_v_init = _projector_basis(projector_data, "p_v", d_h, r_v).to(dev, dtype=dtype)
    P_K = nn.Parameter(qr_retraction(p_k_init))
    P_V = nn.Parameter(qr_retraction(p_v_init))

    gamma0 = _gamma_from_data(projector_data, gamma_min, dev, dtype)
    xi = nn.Parameter(inv_softplus(gamma0 - gamma_min))
    xi.requires_grad_(train_gamma)

    pk_optimizer = RiemannianAdam(P_K.shape, dev, dtype, lr_pk, beta1, beta2, adam_eps)
    pv_optimizer = RiemannianAdam(P_V.shape, dev, dtype, lr_pv, beta1, beta2, adam_eps)
    xi_optimizer = (
        torch.optim.Adam([xi], lr=lr_xi, betas=(beta1, beta2), eps=adam_eps)
        if train_gamma
        else None
    )
    adam_params = [P_K, P_V] + ([xi] if train_gamma else [])
    adam_optimizer = (
        torch.optim.Adam(adam_params, lr=lr, betas=(beta1, beta2), eps=adam_eps)
        if optimizer == "adam"
        else None
    )

    best_eval_loss = float("inf")
    best_eval_mse = 0.0
    best_eval_normalized = 0.0
    best_P_K = P_K.detach().clone()
    best_P_V = P_V.detach().clone()
    best_xi = xi.detach().clone()
    best_step = 0
    stale_checks = 0
    stopped_early = False
    history: list[dict] = []

    for step in range(1, n_steps + 1):
        batch_idx = _sample_batch_indices(Q.shape[0], sample_batch_size, dev)
        row_idx = sample_rows(Q.shape[1], row_batch_size, dev)
        Qb = Q if batch_idx is None else Q[batch_idx]
        Kb = K if batch_idx is None else K[batch_idx]
        Vb = V if batch_idx is None else V[batch_idx]

        gamma = F.softplus(xi) + gamma_min
        diff_sq, teacher_sq, count = _attention_output_stats(
            Qb, Kb, Vb, P_K, P_V, gamma, row_idx=row_idx,
        )
        loss = _loss_from_stats(diff_sq, teacher_sq, count, loss_mode, eps_loss)

        if optimizer == "riemannian_adam":
            params = [P_K, P_V] + ([xi] if train_gamma else [])
            grads = torch.autograd.grad(loss, params)
            grads = clip_by_global_norm(list(grads), grad_clip)
            g_pk, g_pv = grads[0], grads[1]
            g_xi = grads[2] if train_gamma else None

            if train_gamma:
                xi_optimizer.zero_grad(set_to_none=True)
                xi.grad = g_xi
                xi_optimizer.step()

            with torch.no_grad():
                pk_optimizer.step_(P_K, g_pk)
                pv_optimizer.step_(P_V, g_pv)
        else:
            adam_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adam_params, grad_clip)
            adam_optimizer.step()

            if step % orthogonalize_every == 0:
                with torch.no_grad():
                    P_K.copy_(qr_retraction(P_K))
                    P_V.copy_(qr_retraction(P_V))

        do_eval = step == 1 or step % eval_every == 0 or step == n_steps
        if do_eval:
            with torch.no_grad():
                eval_stats = _eval_projector(
                    Q, K, V, P_K, P_V, F.softplus(xi) + gamma_min,
                    loss_mode=loss_mode,
                    eps_loss=eps_loss,
                    eval_batch_size=eval_batch_size,
                    eval_max_batches=eval_max_batches,
                )
                eval_loss = eval_stats["loss"]
                eval_mse = eval_stats["mse"]
                eval_norm = eval_stats["normalized"]
                gamma_val = float((F.softplus(xi) + gamma_min).detach().cpu())
                orth_k = _orth_err(P_K)
                orth_v = _orth_err(P_V)

            history.append({
                "step": step,
                "eval_loss": eval_loss,
                "eval_mse": eval_mse,
                "eval_normalized": eval_norm,
                "gamma": gamma_val,
                "orthK": orth_k,
                "orthV": orth_v,
                "optimizer": optimizer,
            })

            if verbose:
                print(
                    f"step={step:04d} distill_loss={eval_loss:.6e} "
                    f"mse={eval_mse:.6e} normalized={eval_norm:.6e} "
                    f"gamma={gamma_val:.5f} orthK={orth_k:.2e} orthV={orth_v:.2e}"
                )

            if eval_loss < best_eval_loss:
                prev_best = best_eval_loss
                best_eval_loss = eval_loss
                best_eval_mse = eval_mse
                best_eval_normalized = eval_norm
                best_P_K = P_K.detach().clone()
                best_P_V = P_V.detach().clone()
                best_xi = xi.detach().clone()
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
            if verbose:
                print(f"Early stopping at step {step} (no improvement for {stale_checks} eval checks).")
            stopped_early = True
            break

    best_gamma = F.softplus(best_xi) + gamma_min
    return AttentionDistillResult(
        p_k=best_P_K.detach().cpu(),
        p_v=best_P_V.detach().cpu(),
        gamma=best_gamma.detach().cpu().reshape(1),
        metrics={"attention_distill": history},
        best_step=best_step,
        actual_steps=step,
        stopped_early=stopped_early,
        best_eval_loss=best_eval_loss,
        best_eval_mse=best_eval_mse,
        best_eval_normalized=best_eval_normalized,
    )
