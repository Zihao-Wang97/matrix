from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from hawp_laq.offline.losses import (
    attention_output_mse_loss,
    logits_mse_loss,
)
from hawp_laq.utils.io import save_pt
from hawp_laq.utils.math_utils import orthogonalize


def _complete_to_orthonormal_basis(basis: torch.Tensor, dim: int) -> torch.Tensor:
    """Complete a partial column-orthogonal basis into a full orthonormal matrix.

    Takes a ``[dim, r]`` matrix whose first ``r`` columns are the learned
    basis and returns a ``[dim, dim]`` orthonormal matrix where the first
    ``r`` columns are exactly ``basis`` and the remaining ``dim - r`` columns
    span the orthogonal complement.

    Current implementation uses a two-step QR decomposition: first QR on the
    padded matrix to find the full column space, then replace the first ``r``
    columns with the exact learned basis and QR again.  This is an
    engineering-quality approximation; a more rigorous approach would use
    null-space computation (e.g. SVD-based), but the current method is
    sufficient for training stability and orthonormality guarantees.
    """
    full = torch.zeros(dim, dim, dtype=basis.dtype)
    r = basis.shape[1]
    full[:, :r] = basis
    if r < dim:
        q, _ = torch.linalg.qr(full)
        full = q
        full[:, :r] = basis
        q_final, _ = torch.linalg.qr(full)
        full = q_final
    return full


class ProjectorModule(nn.Module):
    def __init__(self, d_model: int, rank_k: int, rank_v: int, n_heads: int,
                 init_p_k: torch.Tensor | None = None,
                 init_p_v: torch.Tensor | None = None):
        super().__init__()
        self.d_model = d_model
        self.rank_k = rank_k
        self.rank_v = rank_v
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        if not (1 <= rank_k <= self.head_dim):
            raise ValueError(
                f"rank_k={rank_k} must satisfy 1 <= rank_k <= head_dim={self.head_dim} "
                f"(d_model={d_model}, n_heads={n_heads})"
            )
        if not (1 <= rank_v <= self.head_dim):
            raise ValueError(
                f"rank_v={rank_v} must satisfy 1 <= rank_v <= head_dim={self.head_dim} "
                f"(d_model={d_model}, n_heads={n_heads})"
            )
        if init_p_k is not None:
            self.p_k_basis = nn.Parameter(init_p_k.clone())
        else:
            self.p_k_basis = nn.Parameter(
                torch.randn(self.head_dim, rank_k) * (rank_k ** -0.5)
            )
        if init_p_v is not None:
            self.p_v_basis = nn.Parameter(init_p_v.clone())
        else:
            self.p_v_basis = nn.Parameter(
                torch.randn(self.head_dim, rank_v) * (rank_v ** -0.5)
            )
        self.gamma = nn.Parameter(torch.ones(1))

    def orthogonalize_projectors(self) -> None:
        with torch.no_grad():
            self.p_k_basis.copy_(orthogonalize(self.p_k_basis.data))
            self.p_v_basis.copy_(orthogonalize(self.p_v_basis.data))

    def _reshape_mh(self, x: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.shape
        return x.view(b, s, self.n_heads, self.head_dim).transpose(1, 2)

    @staticmethod
    def _make_causal_mask(q_len: int, kv_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        mask = torch.full((q_len, kv_len), float("-inf"), device=device, dtype=dtype)
        mask = torch.triu(mask, diagonal=kv_len - q_len + 1)
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        q_mh = self._reshape_mh(q)
        k_mh = self._reshape_mh(k)
        v_mh = self._reshape_mh(v)

        q_lat = q_mh @ self.p_k_basis
        k_lat = k_mh @ self.p_k_basis
        v_lat = v_mh @ self.p_v_basis

        lr_scale = self.gamma / math.sqrt(self.rank_k)
        logits_fp = (q_mh @ k_mh.transpose(-2, -1)) * (self.head_dim ** -0.5)
        logits_hat = (q_lat @ k_lat.transpose(-2, -1)) * lr_scale

        q_len = q_mh.shape[2]
        kv_len = k_mh.shape[2]
        assert q_len == kv_len, f"ProjectorModule requires q_len == kv_len, got {q_len} vs {kv_len}"
        causal_mask = self._make_causal_mask(q_len, kv_len, q_mh.device, q_mh.dtype)

        logits_fp_masked = logits_fp + causal_mask
        logits_hat_masked = logits_hat + causal_mask

        attn_probs = torch.softmax(logits_fp_masked, dim=-1)
        attn_probs_hat = torch.softmax(logits_hat_masked, dim=-1)

        attn_out = attn_probs @ v_mh
        attn_out_hat_lat = attn_probs_hat @ v_lat
        attn_out_hat = attn_out_hat_lat @ self.p_v_basis.T

        k_recon_mh = k_lat @ self.p_k_basis.T
        v_recon_mh = v_lat @ self.p_v_basis.T
        k_recon = k_recon_mh.transpose(1, 2).contiguous().view(*k.shape)
        v_recon = v_recon_mh.transpose(1, 2).contiguous().view(*v.shape)

        valid = causal_mask == 0
        return logits_fp, logits_hat, valid, attn_out, attn_out_hat, k_recon, v_recon


class ProjectorTrainer:
    def __init__(
        self,
        d_model: int,
        rank_k: int,
        rank_v: int,
        n_heads: int,
        lr: float = 1e-3,
        orthogonalize_every: int = 10,
        w_logits: float = 1.0,
        w_attn: float = 1.0,
        w_value: float = 0.5,
        device: str = "cpu",
    ):
        self.d_model = d_model
        self.rank_k = rank_k
        self.rank_v = rank_v
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.lr = lr
        self.orthogonalize_every = orthogonalize_every
        self.w_logits = w_logits
        self.w_attn = w_attn
        self.w_value = w_value
        self.device = device

        if not (1 <= rank_k <= self.head_dim):
            raise ValueError(
                f"rank_k={rank_k} must satisfy 1 <= rank_k <= head_dim={self.head_dim} "
                f"(d_model={d_model}, n_heads={n_heads})"
            )
        if not (1 <= rank_v <= self.head_dim):
            raise ValueError(
                f"rank_v={rank_v} must satisfy 1 <= rank_v <= head_dim={self.head_dim} "
                f"(d_model={d_model}, n_heads={n_heads})"
            )

    @staticmethod
    def _svd_init_basis(x: torch.Tensor, n_heads: int, rank: int) -> torch.Tensor:
        b, s, d_model = x.shape
        head_dim = d_model // n_heads
        x_mh = x.view(b, s, n_heads, head_dim).transpose(1, 2)
        x_flat = x_mh.reshape(-1, head_dim)
        _, _, Vh = torch.linalg.svd(x_flat, full_matrices=False)
        return Vh[:rank].T

    def train_one_group(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        n_steps: int = 200,
    ) -> dict:
        q = q.to(self.device).float()
        k = k.to(self.device).float()
        v = v.to(self.device).float()

        module = ProjectorModule(
            self.d_model, self.rank_k, self.rank_v, self.n_heads,
            init_p_k=self._svd_init_basis(k, self.n_heads, self.rank_k),
            init_p_v=self._svd_init_basis(v, self.n_heads, self.rank_v),
        ).to(self.device)
        optimizer = torch.optim.Adam(module.parameters(), lr=self.lr)

        metrics: dict[str, list] = {
            "total": [],
            "logits": [],
            "attn": [],
            "value": [],
        }

        for step in range(1, n_steps + 1):
            optimizer.zero_grad()
            logits_fp, logits_hat, causal_valid, attn_out, attn_out_hat, k_recon, v_recon = module(q, k, v)

            l_logits = logits_mse_loss(logits_fp, logits_hat, valid_mask=causal_valid)
            l_attn = attention_output_mse_loss(attn_out, attn_out_hat)
            l_val = F.mse_loss(v_recon, v)
            loss = (
                self.w_logits * l_logits
                + self.w_attn * l_attn
                + self.w_value * l_val
            )

            loss.backward()
            optimizer.step()

            if step % self.orthogonalize_every == 0:
                module.orthogonalize_projectors()

            metrics["total"].append(loss.item())
            metrics["logits"].append(l_logits.item())
            metrics["attn"].append(l_attn.item())
            metrics["value"].append(l_val.item())

            if step % max(1, n_steps // 5) == 0 or step == 1:
                print(f"  step {step:>4d}/{n_steps}  total={loss.item():.6f}  logits={l_logits.item():.6f}  attn={l_attn.item():.6f}  val={l_val.item():.6f}")

        module.orthogonalize_projectors()

        p_k_full = _complete_to_orthonormal_basis(
            module.p_k_basis.data.cpu().detach(), self.head_dim
        )
        p_v_full = _complete_to_orthonormal_basis(
            module.p_v_basis.data.cpu().detach(), self.head_dim
        )

        return {
            "p_k": p_k_full,
            "p_v": p_v_full,
            "gamma": module.gamma.data.cpu().detach(),
            "r_k": self.rank_k,
            "r_v": self.rank_v,
            "metrics": metrics,
            "causal_mask": True,
        }

    @staticmethod
    def save_result(result: dict, layer_idx: int, output_dir: str | Path) -> Path:
        out = Path(output_dir)
        layer_dir = out / f"layer_{layer_idx}"
        layer_dir.mkdir(parents=True, exist_ok=True)
        save_pt(result, layer_dir / "projector.pt")
        print(
            f"[save] {layer_dir / 'projector.pt'}  "
            f"p_k={tuple(result['p_k'].shape)} p_v={tuple(result['p_v'].shape)} "
            f"r_k={result['r_k']} r_v={result['r_v']} gamma={result['gamma'].item():.4f}"
        )
        return layer_dir
