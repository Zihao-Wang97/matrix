"""PyTorch implementation of low-rank attention calibration with alternating Riemannian-Adam.

This module solves

    min_{P_K, P_V, gamma}
        lambda_z * ||Z - Z_hat||_M^2 / (||Z||_M^2 + eps)
      + lambda_o * ||O - O_hat||_F^2 / (||O||_F^2 + eps)
      + lambda_v * ||V - (V P_V) P_V^T||_F^2 / (||V||_F^2 + eps)

subject to

    P_K^T P_K = I,   P_V^T P_V = I.

Compared with the NumPy prototype, this version uses PyTorch autograd and is
ready to be integrated into a neural-network calibration pipeline. The solver
uses:

1. spectral initialization,
2. alternating block updates,
3. Riemannian-Adam on Stiefel manifolds for P_K and P_V,
4. QR retraction,
5. row sampling for acceleration,
6. a two-stage warmup/full training schedule.

Typical usage:

    cfg = OptimConfig(r_k=16, r_v=16, device="cuda")
    result = optimize_low_rank_attention_torch(Q, K, V, mask, cfg)

where Q, K, V have shape [batch, seq_len, d_h], and mask has shape
[batch, seq_len, seq_len].
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


Tensor = torch.Tensor


@dataclass
class OptimConfig:
    r_k: int
    r_v: int
    lambda_z: float = 1.0
    lambda_o: float = 2.0
    lambda_v: float = 0.05
    eps_loss: float = 1e-8
    adam_eps: float = 1e-8
    beta1: float = 0.9
    beta2: float = 0.99
    lr_pk: float = 5e-3
    lr_pv: float = 5e-3
    lr_xi: float = 1e-2
    max_steps: int = 300
    warmup_steps: int = 50
    row_batch_size: Optional[int] = 256
    gamma_min: float = 1e-4
    grad_clip: float = 1.0
    # --- full-calib evaluation & early stopping ---
    eval_every: int = 50
    early_stopping: bool = True
    patience: int = 5
    min_delta: float = 1e-4
    min_delta_mode: str = "relative"  # "relative" or "absolute"
    # --- misc ---
    seed: int = 0
    verbose: bool = True
    log_every: int = 10
    device: str = "cpu"
    dtype: torch.dtype = torch.float32


def sym(X: Tensor) -> Tensor:
    return 0.5 * (X + X.transpose(-1, -2))


def tangent_projection(P: Tensor, G: Tensor) -> Tensor:
    return G - P @ sym(P.transpose(-1, -2) @ G)


def qr_retraction(Y: Tensor) -> Tensor:
    Q, R = torch.linalg.qr(Y, mode="reduced")
    sign = torch.sign(torch.diagonal(R, dim1=-2, dim2=-1))
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    return Q @ torch.diag(sign)


def inv_softplus(y: Tensor) -> Tensor:
    y = torch.clamp(y, min=1e-12)
    return torch.where(y > 20, y, torch.log(torch.expm1(y)))


def build_additive_and_valid_mask(mask: Tensor) -> Tuple[Tensor, Tensor]:
    """Return additive mask M and valid mask W.

    Supported formats:
    1) bool mask: True means valid and False means invalid.
    2) float mask: additive attention mask where invalid entries are large negative values.
    """
    if mask.dtype == torch.bool:
        valid = mask
        additive = torch.where(
            valid,
            torch.zeros_like(mask, dtype=torch.float32),
            torch.full_like(mask, -1e4, dtype=torch.float32),
        )
        return additive, valid

    additive = mask
    valid = additive > -1e3
    return additive, valid


def stable_softmax(logits: Tensor, additive_mask: Tensor, valid_mask: Optional[Tensor] = None) -> Tensor:
    x = logits + additive_mask
    x = x - x.max(dim=-1, keepdim=True).values
    probs = F.softmax(x, dim=-1)
    if valid_mask is not None:
        probs = probs * valid_mask.to(dtype=probs.dtype)
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return probs


def covariance_top_eig(X: Tensor, rank: int) -> Tensor:
    Xf = X.reshape(-1, X.shape[-1])
    C = Xf.transpose(0, 1) @ Xf
    _, eigvecs = torch.linalg.eigh(C)
    return eigvecs[:, -rank:]


def init_pk(Q: Tensor, K: Tensor, r_k: int) -> Tensor:
    d = Q.shape[-1]
    C = torch.zeros(d, d, device=Q.device, dtype=Q.dtype)
    for b in range(Q.shape[0]):
        QQt = Q[b] @ Q[b].transpose(0, 1)
        C = C + K[b].transpose(0, 1) @ QQt @ K[b]
    _, eigvecs = torch.linalg.eigh(C)
    return eigvecs[:, -r_k:]


def init_pv(V: Tensor, r_v: int) -> Tensor:
    return covariance_top_eig(V, r_v)


def init_gamma(
    Q: Tensor,
    K: Tensor,
    valid_mask: Tensor,
    P_K: Tensor,
    r_k: int,
    delta: float = 1e-8,
) -> float:
    d_h = Q.shape[-1]
    Z = (Q @ K.transpose(-1, -2)) / math.sqrt(d_h)
    B = Q @ P_K @ P_K.transpose(0, 1) @ K.transpose(-1, -2)
    W = valid_mask.to(dtype=Z.dtype)
    num = (W * Z * B).sum()
    den = (W * B * B).sum() + delta
    alpha0 = num / den
    gamma0 = math.sqrt(r_k) * alpha0.item()
    return max(gamma0, 1e-4)


def sample_rows(num_rows: int, row_batch_size: Optional[int], device: torch.device) -> Optional[Tensor]:
    if row_batch_size is None or row_batch_size >= num_rows:
        return None
    return torch.randperm(num_rows, device=device)[:row_batch_size]


def stage_name(step: int, warmup_steps: int) -> str:
    return "warmup" if step <= warmup_steps else "full"


def clip_by_global_norm(grads: List[Optional[Tensor]], max_norm: float) -> List[Optional[Tensor]]:
    total_sq = None
    for g in grads:
        if g is None:
            continue
        val = g.pow(2).sum()
        total_sq = val if total_sq is None else total_sq + val

    if total_sq is None:
        return grads

    total_norm = total_sq.sqrt()
    if total_norm <= max_norm:
        return grads

    scale = max_norm / (total_norm + 1e-12)
    return [None if g is None else g * scale for g in grads]


class RiemannianAdam:
    def __init__(self, shape: Tuple[int, ...], device: torch.device, dtype: torch.dtype, lr: float, beta1: float, beta2: float, eps: float):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.step_num = 0
        self.m = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros(shape, device=device, dtype=dtype)

    @torch.no_grad()
    def step_(self, P: Tensor, G: Tensor) -> None:
        G_tan = tangent_projection(P, G)
        self.step_num += 1

        self.m.mul_(self.beta1).add_(G_tan, alpha=1.0 - self.beta1)
        self.v.mul_(self.beta2).addcmul_(G_tan, G_tan, value=1.0 - self.beta2)

        m_hat = self.m / (1.0 - self.beta1 ** self.step_num)
        v_hat = self.v / (1.0 - self.beta2 ** self.step_num)
        direction = m_hat / (v_hat.sqrt() + self.eps)
        direction = tangent_projection(P, direction)

        Y = P - self.lr * direction
        P.copy_(qr_retraction(Y))
        self.m.copy_(tangent_projection(P, self.m))


def compute_objective(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    additive_mask: Tensor,
    valid_mask: Tensor,
    P_K: Tensor,
    P_V: Tensor,
    xi: Tensor,
    cfg: OptimConfig,
    row_idx: Optional[Tensor] = None,
    stage: str = "full",
) -> Tuple[Tensor, Dict[str, float]]:
    d_h = Q.shape[-1]

    if row_idx is None:
        Qs = Q
        Ms = additive_mask
        Ws = valid_mask
    else:
        Qs = Q[:, row_idx, :]
        Ms = additive_mask[:, row_idx, :]
        Ws = valid_mask[:, row_idx, :]

    gamma = F.softplus(xi) + cfg.gamma_min

    Z = (Qs @ K.transpose(-1, -2)) / math.sqrt(d_h)
    A = stable_softmax(Z, Ms, Ws)
    O = A @ V

    Ql = Qs @ P_K
    Kl = K @ P_K
    Vl = V @ P_V

    Z_hat = (gamma / math.sqrt(cfg.r_k)) * (Ql @ Kl.transpose(-1, -2))
    A_hat = stable_softmax(Z_hat, Ms, Ws)
    O_hat = (A_hat @ Vl) @ P_V.transpose(0, 1)

    V_rec = (V @ P_V) @ P_V.transpose(0, 1)
    Ws_f = Ws.to(dtype=Q.dtype)

    L_z = (Ws_f * (Z - Z_hat).pow(2)).sum() / ((Ws_f * Z.pow(2)).sum() + cfg.eps_loss)
    L_o = (O - O_hat).pow(2).sum() / (O.pow(2).sum() + cfg.eps_loss)
    L_v = (V - V_rec).pow(2).sum() / (V.pow(2).sum() + cfg.eps_loss)

    if stage == "warmup":
        loss = L_z + 0.1 * L_o + 0.05 * L_v
    else:
        loss = cfg.lambda_z * L_z + cfg.lambda_o * L_o + cfg.lambda_v * L_v

    metrics = {
        "loss": float(loss.detach().cpu()),
        "L_z": float(L_z.detach().cpu()),
        "L_o": float(L_o.detach().cpu()),
        "L_v": float(L_v.detach().cpu()),
        "gamma": float(gamma.detach().cpu()),
    }
    return loss, metrics


@torch.no_grad()
def evaluate_full(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    additive_mask: Tensor,
    valid_mask: Tensor,
    P_K: Tensor,
    P_V: Tensor,
    xi: Tensor,
    cfg: OptimConfig,
) -> Dict[str, float]:
    loss, raw = compute_objective(
        Q=Q, K=K, V=V,
        additive_mask=additive_mask, valid_mask=valid_mask,
        P_K=P_K, P_V=P_V, xi=xi, cfg=cfg,
        row_idx=None, stage="full",
    )
    del loss
    pk_err = torch.linalg.norm(P_K.transpose(0, 1) @ P_K - torch.eye(P_K.shape[1], device=P_K.device, dtype=P_K.dtype))
    pv_err = torch.linalg.norm(P_V.transpose(0, 1) @ P_V - torch.eye(P_V.shape[1], device=P_V.device, dtype=P_V.dtype))
    return {
        "total": float(raw["loss"]),   # loss from compute_objective → total
        "logits": float(raw["L_z"]),
        "attn": float(raw["L_o"]),
        "value": float(raw["L_v"]),
        "gamma": float(raw["gamma"]),
        "pk_orth_err": float(pk_err.cpu()),
        "pv_orth_err": float(pv_err.cpu()),
    }


def optimize_low_rank_attention_torch(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    mask: Tensor,
    cfg: OptimConfig,
) -> Dict[str, object]:
    torch.manual_seed(cfg.seed)

    device = torch.device(cfg.device)
    dtype = cfg.dtype

    Q = Q.to(device=device, dtype=dtype)
    K = K.to(device=device, dtype=dtype)
    V = V.to(device=device, dtype=dtype)
    mask = mask.to(device=device)

    if Q.ndim != 3 or K.ndim != 3 or V.ndim != 3:
        raise ValueError("Q, K, V must have shape [batch, seq_len, d_h].")
    if Q.shape != K.shape or Q.shape != V.shape:
        raise ValueError("Q, K, V must have identical shapes.")
    if mask.shape != (Q.shape[0], Q.shape[1], Q.shape[1]):
        raise ValueError("mask must have shape [batch, seq_len, seq_len].")
    if cfg.r_k > Q.shape[-1] or cfg.r_v > Q.shape[-1]:
        raise ValueError("r_k and r_v must not exceed d_h.")
    if cfg.eval_every <= 0:
        raise ValueError("eval_every must be > 0")
    if cfg.min_delta_mode not in ("relative", "absolute"):
        raise ValueError(
            f"min_delta_mode must be 'relative' or 'absolute', "
            f"got {cfg.min_delta_mode!r}"
        )

    additive_mask, valid_mask = build_additive_and_valid_mask(mask)
    additive_mask = additive_mask.to(device=device, dtype=dtype)
    valid_mask = valid_mask.to(device=device)

    # ---- init ----
    P_K = nn.Parameter(init_pk(Q, K, cfg.r_k).to(device=device, dtype=dtype))
    P_V = nn.Parameter(init_pv(V, cfg.r_v).to(device=device, dtype=dtype))

    gamma0 = init_gamma(Q, K, valid_mask, P_K.detach(), cfg.r_k)
    xi0 = inv_softplus(torch.tensor(gamma0 - cfg.gamma_min + 1e-8, device=device, dtype=dtype))
    xi = nn.Parameter(xi0)

    xi_optimizer = torch.optim.Adam([xi], lr=cfg.lr_xi, betas=(cfg.beta1, cfg.beta2), eps=cfg.adam_eps)
    pk_optimizer = RiemannianAdam(P_K.shape, device, dtype, cfg.lr_pk, cfg.beta1, cfg.beta2, cfg.adam_eps)
    pv_optimizer = RiemannianAdam(P_V.shape, device, dtype, cfg.lr_pv, cfg.beta1, cfg.beta2, cfg.adam_eps)

    # ---- best checkpoint state ----
    best_calib_total = float("inf")
    best_P_K = P_K.detach().clone()
    best_P_V = P_V.detach().clone()
    best_xi = xi.detach().clone()
    best_step = 0
    best_calib_logits = 0.0
    best_calib_attn = 0.0
    best_calib_value = 0.0
    stale_steps = 0

    history: List[Dict[str, object]] = []
    stopped_early = False
    actual_steps = 0

    for step in range(1, cfg.max_steps + 1):
        actual_steps = step
        stage = stage_name(step, cfg.warmup_steps)
        row_idx = sample_rows(Q.shape[1], cfg.row_batch_size, device)

        # ----- Block 1: update (P_K, xi) while fixing P_V -----
        loss_k, raw_k = compute_objective(
            Q=Q, K=K, V=V,
            additive_mask=additive_mask, valid_mask=valid_mask,
            P_K=P_K, P_V=P_V.detach(), xi=xi,
            cfg=cfg, row_idx=row_idx, stage=stage,
        )
        g_pk, g_xi = torch.autograd.grad(loss_k, [P_K, xi])
        g_pk, g_xi = clip_by_global_norm([g_pk, g_xi], cfg.grad_clip)

        xi_optimizer.zero_grad(set_to_none=True)
        xi.grad = g_xi
        xi_optimizer.step()

        with torch.no_grad():
            pk_optimizer.step_(P_K, g_pk)

        # ----- Block 2: update P_V while fixing (P_K, xi) -----
        loss_v, raw_v = compute_objective(
            Q=Q, K=K, V=V,
            additive_mask=additive_mask, valid_mask=valid_mask,
            P_K=P_K.detach(), P_V=P_V, xi=xi.detach(),
            cfg=cfg, row_idx=row_idx, stage=stage,
        )
        (g_pv,) = torch.autograd.grad(loss_v, [P_V])
        (g_pv,) = clip_by_global_norm([g_pv], cfg.grad_clip)

        with torch.no_grad():
            pv_optimizer.step_(P_V, g_pv)

        # Record sampled train metrics every step
        history.append({
            "kind": "sampled_train",
            "step": step,
            "stage": stage,
            "loss": float(loss_v.detach().cpu()),
            "L_z": float(raw_v["L_z"]),
            "L_o": float(raw_v["L_o"]),
            "L_v": float(raw_v["L_v"]),
            "gamma": float(raw_v["gamma"]),
        })

        # ---- full-calibration evaluation (only on eval_every or first/last) ----
        do_full_eval = (
            step == 1
            or step % cfg.eval_every == 0
            or step == cfg.max_steps
        )
        if do_full_eval:
            calib = evaluate_full(Q, K, V, additive_mask, valid_mask, P_K, P_V, xi, cfg)
            calib["kind"] = "full_calib"
            calib["step"] = step
            calib["stage"] = stage
            history.append(calib)

            current_total = calib["total"]

            if cfg.verbose:
                print(
                    f"step={step:04d} stage={stage:<6} "
                    f"total={calib['total']:.6e} "
                    f"logits={calib['logits']:.6e} "
                    f"attn={calib['attn']:.6e} "
                    f"value={calib['value']:.6e} "
                    f"gamma={calib['gamma']:.5f} "
                    f"orthK={calib['pk_orth_err']:.2e} "
                    f"orthV={calib['pv_orth_err']:.2e}"
                )

            # ---- update best checkpoint ----
            # Always save best when current_total improves, so best_* reflects
            # the true minimum.  stale_steps resets only on "significant" improvement.
            if current_total < best_calib_total:
                prev_best = best_calib_total
                best_calib_total = current_total
                best_calib_logits = calib["logits"]
                best_calib_attn = calib["attn"]
                best_calib_value = calib["value"]
                best_P_K = P_K.detach().clone()
                best_P_V = P_V.detach().clone()
                best_xi = xi.detach().clone()
                best_step = step

                if cfg.min_delta_mode == "relative":
                    significant = (prev_best - current_total) / (abs(prev_best) + cfg.eps_loss) > cfg.min_delta
                else:  # absolute
                    significant = (prev_best - current_total) > cfg.min_delta
                if significant:
                    stale_steps = 0
                else:
                    stale_steps += 1
            else:
                stale_steps += 1

        # ---- early stopping ----
        if cfg.early_stopping and stale_steps >= cfg.patience:
            if cfg.verbose:
                print(f"Early stopping at step {step} (no improvement for {stale_steps} eval checks).")
            stopped_early = True
            break

    # Final cleanup: if never did full eval (shouldn't happen since step==1 does it),
    # do one final eval to populate best_* fields
    best_gamma = float((F.softplus(best_xi) + cfg.gamma_min).detach().cpu())
    return {
        "P_K": best_P_K,
        "P_V": best_P_V,
        "gamma": best_gamma,
        "history": history,
        "best_step": best_step,
        "actual_steps": actual_steps,
        "stopped_early": stopped_early,
        "best_calib_total": best_calib_total,
        "best_calib_logits": best_calib_logits,
        "best_calib_attn": best_calib_attn,
        "best_calib_value": best_calib_value,
    }


def make_causal_mask(batch_size: int, seq_len: int, device: Optional[torch.device] = None) -> Tensor:
    valid = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=0)
    return valid.unsqueeze(0).expand(batch_size, -1, -1)


def demo() -> None:
    torch.manual_seed(7)

    batch_size, seq_len, d_h = 2, 20, 12
    Q = torch.randn(batch_size, seq_len, d_h)
    K = torch.randn(batch_size, seq_len, d_h)
    V = torch.randn(batch_size, seq_len, d_h)
    mask = make_causal_mask(batch_size, seq_len)

    cfg = OptimConfig(
        r_k=4,
        r_v=4,
        max_steps=60,
        warmup_steps=10,
        row_batch_size=10,
        lr_pk=5e-3,
        lr_pv=5e-3,
        lr_xi=1e-2,
        eval_every=10,
        early_stopping=True,
        patience=3,
        min_delta=1e-4,
        min_delta_mode="relative",
        verbose=True,
        log_every=10,
        seed=42,
        device="cpu",
        dtype=torch.float32,
    )

    result = optimize_low_rank_attention_torch(Q, K, V, mask, cfg)
    P_K = result["P_K"]
    P_V = result["P_V"]
    gamma = result["gamma"]

    print("\nOptimization finished.")
    print("P_K shape:", tuple(P_K.shape))
    print("P_V shape:", tuple(P_V.shape))
    print("gamma:", gamma)
    print("best_step:", result["best_step"])
    print("actual_steps:", result["actual_steps"])
    print("stopped_early:", result["stopped_early"])
    print(f"best_calib_total={result['best_calib_total']:.6e}")
    print(f"best_calib_logits={result['best_calib_logits']:.6e}")
    print(f"best_calib_attn={result['best_calib_attn']:.6e}")
    print(f"best_calib_value={result['best_calib_value']:.6e}")
    fc_count = sum(1 for h in result["history"] if h.get("kind") == "full_calib")
    st_count = sum(1 for h in result["history"] if h.get("kind") == "sampled_train")
    print(f"history entries: {st_count} sampled_train + {fc_count} full_calib")
    print("||P_K^T P_K - I||_F =", torch.linalg.norm(P_K.transpose(0, 1) @ P_K - torch.eye(P_K.shape[1], dtype=P_K.dtype, device=P_K.device)).item())
    print("||P_V^T P_V - I||_F =", torch.linalg.norm(P_V.transpose(0, 1) @ P_V - torch.eye(P_V.shape[1], dtype=P_V.dtype, device=P_V.device)).item())


if __name__ == "__main__":
    demo()
