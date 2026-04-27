from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_logits_fp(q: torch.Tensor, k: torch.Tensor, scale: float | None = None) -> torch.Tensor:
    if scale is None:
        scale = q.shape[-1] ** -0.5
    return (q @ k.transpose(-2, -1)) * scale


def compute_logits_hat(q: torch.Tensor, k: torch.Tensor, p_k: torch.Tensor, scale: float | None = None) -> torch.Tensor:
    if scale is None:
        scale = q.shape[-1] ** -0.5
    k_recon = k @ p_k @ p_k.T
    return (q @ k_recon.transpose(-2, -1)) * scale


def logits_mse_loss(logits_fp: torch.Tensor, logits_hat: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
    if valid_mask is not None:
        diff = (logits_hat - logits_fp) ** 2
        valid_mask = valid_mask.expand_as(diff)
        return diff[valid_mask].mean()
    return F.mse_loss(logits_hat, logits_fp)


def attention_output_mse_loss(attn_out: torch.Tensor, attn_out_hat: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(attn_out_hat, attn_out)


def value_reconstruction_loss(v: torch.Tensor, p_v: torch.Tensor, gamma: torch.Tensor | None = None) -> torch.Tensor:
    v_recon = v @ p_v @ p_v.T
    if gamma is not None:
        v_recon = gamma * v_recon
    return F.mse_loss(v_recon, v)


def total_projector_loss(
    logits_fp: torch.Tensor,
    logits_hat: torch.Tensor,
    attn_out: torch.Tensor,
    attn_out_hat: torch.Tensor,
    v: torch.Tensor,
    p_v: torch.Tensor,
    gamma: torch.Tensor | None = None,
    w_logits: float = 1.0,
    w_attn: float = 1.0,
    w_value: float = 0.5,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    l_logits = logits_mse_loss(logits_fp, logits_hat, valid_mask=valid_mask)
    l_attn = attention_output_mse_loss(attn_out, attn_out_hat)
    l_val = value_reconstruction_loss(v, p_v, gamma)
    return w_logits * l_logits + w_attn * l_attn + w_value * l_val
