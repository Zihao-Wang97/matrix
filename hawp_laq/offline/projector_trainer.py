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
from hawp_laq.offline.low_rank_attention_optimizer_torch import (
    OptimConfig,
    optimize_low_rank_attention_torch,
)
from hawp_laq.utils.io import save_pt
from hawp_laq.utils.math_utils import orthogonalize


def _complete_to_orthonormal_basis(basis: torch.Tensor, dim: int) -> torch.Tensor:
    """Complete a partial column-orthogonal basis into a full orthonormal matrix.

    Takes a ``[dim, r]`` matrix whose first ``r`` columns are the learned
    basis and returns a ``[dim, dim]`` orthonormal matrix where the first
    ``r`` columns are exactly ``basis`` (orthonormalized) and the remaining
    columns span the orthogonal complement.

    Uses projection-based complement generation followed by QR to ensure
    the complement is orthonormal and orthogonal to the basis, without
    perturbing the first ``r`` columns.

    If the basis is already sufficiently orthonormal, the orthonormalization
    step is skipped to preserve the optimizer's output exactly.
    """
    r = basis.shape[1]
    if r >= dim:
        return basis
    # Only orthonormalize if needed (preserves optimizer output when already good)
    I_r = torch.eye(r, dtype=basis.dtype, device=basis.device)
    dev = (basis.T @ basis - I_r).norm().item()
    if dev > 1e-5:
        basis = orthogonalize(basis)
    # Generate random vectors, project out the basis, then QR the complement
    rand = torch.randn(dim, dim - r, dtype=basis.dtype, device=basis.device)
    rand = rand - basis @ (basis.T @ rand)
    q_comp, _ = torch.linalg.qr(rand, mode="reduced")
    return torch.cat([basis, q_comp], dim=1)


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

    @staticmethod
    def _to_optim_input(x: torch.Tensor, n_heads: int, d_model: int, head_dim: int) -> tuple[torch.Tensor, int]:
        """Reshape x into [B_eff, T, d_h], using explicit dims to disambiguate.

        Supports:
          - [B, T, d_model]   → [B*n_heads, T, head_dim]
          - [B, H, T, d_h]    → [B*H, T, d_h]
          - [B*H, T, head_dim] → no-op

        Args:
            x: Input tensor.
            n_heads: Number of attention heads.
            d_model: Full model dimension (D if [B,T,D] with D == d_model).
            head_dim: Per-head dimension (= d_model // n_heads).

        Returns (tensor, d_h).
        """
        if x.ndim == 4:
            B, H, T, dh = x.shape
            return x.reshape(B * H, T, dh), dh

        if x.ndim == 3:
            B, T, D = x.shape

            if D == d_model:
                if d_model % n_heads != 0:
                    raise ValueError(
                        f"d_model={d_model} not divisible by n_heads={n_heads}"
                    )
                x = x.view(B, T, n_heads, head_dim)
                x = x.permute(0, 2, 1, 3).contiguous()
                x = x.view(B * n_heads, T, head_dim)
                return x, head_dim

            if D == head_dim:
                # Already [B*H, T, head_dim]
                return x, head_dim

            raise ValueError(
                f"Cannot infer 3D input shape: expected last dim d_model={d_model} "
                f"or head_dim={head_dim}, got D={D}, shape={tuple(x.shape)}"
            )

        raise ValueError(
            f"Unsupported input shape {tuple(x.shape)} (ndim={x.ndim}) for optimizer."
        )

    @staticmethod
    def _make_bool_causal_mask(batch_size: int, seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool)).unsqueeze(0).expand(batch_size, -1, -1)

    def train_one_group(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        n_steps: int = 200,
        *,
        warmup_steps: int = 30,
        row_batch_size: int | None = None,
        lr_pk: float = 5e-3,
        lr_pv: float = 5e-3,
        lr_xi: float = 1e-2,
        beta1: float = 0.9,
        beta2: float = 0.99,
        grad_clip: float = 1.0,
        lambda_z: float = 1.0,
        lambda_o: float = 2.0,
        lambda_v: float = 0.05,
        eval_every: int = 50,
        early_stopping: bool = True,
        patience: int = 5,
        min_delta: float = 1e-4,
        min_delta_mode: str = "relative",
        gamma_min: float = 1e-4,
        eps_loss: float = 1e-8,
        adam_eps: float = 1e-8,
        seed: int = 0,
        optimizer: str = "riemannian_adam",
        device_override: str | None = None,
    ) -> dict:
        if optimizer != "riemannian_adam":
            return self._train_one_group_legacy(q, k, v, n_steps)

        device = torch.device(device_override) if device_override else torch.device(self.device)

        q = q.to(device).float()
        k = k.to(device).float()
        v = v.to(device).float()

        q_opt, d_h = self._to_optim_input(q, self.n_heads, self.d_model, self.head_dim)
        k_opt, _ = self._to_optim_input(k, self.n_heads, self.d_model, self.head_dim)
        v_opt, _ = self._to_optim_input(v, self.n_heads, self.d_model, self.head_dim)

        if q_opt.shape != k_opt.shape or q_opt.shape != v_opt.shape:
            raise ValueError(
                f"After reshape, Q/K/V shapes must match: "
                f"Q={tuple(q_opt.shape)} K={tuple(k_opt.shape)} V={tuple(v_opt.shape)}"
            )

        T = q_opt.shape[1]
        B_eff = q_opt.shape[0]
        mask = self._make_bool_causal_mask(1, T, device)

        if self.rank_k > d_h:
            raise ValueError(
                f"rank_k={self.rank_k} exceeds data head_dim={d_h}. "
                f"Reduce rank_k or check calibration data shape."
            )
        if self.rank_v > d_h:
            raise ValueError(
                f"rank_v={self.rank_v} exceeds data head_dim={d_h}. "
                f"Reduce rank_v or check calibration data shape."
            )
        rk, rv = self.rank_k, self.rank_v

        cfg = OptimConfig(
            r_k=rk, r_v=rv,
            lambda_z=lambda_z, lambda_o=lambda_o, lambda_v=lambda_v,
            eps_loss=eps_loss,
            adam_eps=adam_eps,
            beta1=beta1, beta2=beta2,
            lr_pk=lr_pk, lr_pv=lr_pv, lr_xi=lr_xi,
            max_steps=n_steps,
            warmup_steps=warmup_steps,
            row_batch_size=row_batch_size,
            gamma_min=gamma_min,
            grad_clip=grad_clip,
            eval_every=eval_every,
            early_stopping=early_stopping,
            patience=patience,
            min_delta=min_delta,
            min_delta_mode=min_delta_mode,
            seed=seed,
            verbose=True,
            log_every=max(1, n_steps // 10),
            device=str(device),
            dtype=torch.float32,
        )

        result = optimize_low_rank_attention_torch(q_opt, k_opt, v_opt, mask, cfg)

        P_K = result["P_K"].cpu()
        P_V = result["P_V"].cpu()
        gamma = result["gamma"]

        # Complete to full orthonormal basis for HAWPAttention compatibility
        p_k_full = _complete_to_orthonormal_basis(P_K, d_h)
        p_v_full = _complete_to_orthonormal_basis(P_V, d_h)

        # Extract full_calib metrics from history
        calib_entries = [h for h in result["history"] if h.get("kind") == "full_calib"]
        calib_total = [float(e["total"]) for e in calib_entries]
        calib_logits = [float(e["logits"]) for e in calib_entries]
        calib_attn = [float(e["attn"]) for e in calib_entries]
        calib_value = [float(e["value"]) for e in calib_entries]

        metrics: dict[str, list] = {
            "calib_total": calib_total,
            "calib_logits": calib_logits,
            "calib_attn": calib_attn,
            "calib_value": calib_value,
            # backward-compat aliases for old rank_search code
            "total": calib_total,
            "logits": calib_logits,
            "attn": calib_attn,
            "value": calib_value,
        }

        return {
            "p_k": p_k_full,
            "p_v": p_v_full,
            "gamma": torch.tensor(gamma),
            "r_k": rk,
            "r_v": rv,
            "metrics": metrics,
            "causal_mask": True,
            "best_step": result["best_step"],
            "best_calib_total": result["best_calib_total"],
            "best_calib_logits": result["best_calib_logits"],
            "best_calib_attn": result["best_calib_attn"],
            "best_calib_value": result["best_calib_value"],
            "actual_steps": result["actual_steps"],
            "stopped_early": result["stopped_early"],
        }

    def _train_one_group_legacy(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        n_steps: int = 200,
    ) -> dict:
        """Old Adam + orthogonalize_every path (optimizer != "riemannian_adam")."""
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
            "best_step": n_steps,
            "best_calib_total": metrics["total"][-1] if metrics["total"] else 0.0,
            "actual_steps": n_steps,
            "stopped_early": False,
        }

    @staticmethod
    def save_result(result: dict, layer_idx: int, output_dir: str | Path) -> Path:
        out = Path(output_dir)
        layer_dir = out / f"layer_{layer_idx}"
        layer_dir.mkdir(parents=True, exist_ok=True)
        to_save = {
            "p_k": result["p_k"],
            "p_v": result["p_v"],
            "gamma": result["gamma"],
            "r_k": result["r_k"],
            "r_v": result["r_v"],
            "best_step": result.get("best_step", 0),
            "best_calib_total": result.get("best_calib_total", 0.0),
            "actual_steps": result.get("actual_steps", 0),
            "stopped_early": result.get("stopped_early", False),
            "metrics": result.get("metrics", {}),
            "causal_mask": result.get("causal_mask", True),
        }
        save_pt(to_save, layer_dir / "projector.pt")
        gamma_val = result["gamma"].item() if isinstance(result["gamma"], torch.Tensor) else result["gamma"]
        print(
            f"[save] {layer_dir / 'projector.pt'}  "
            f"p_k={tuple(result['p_k'].shape)} p_v={tuple(result['p_v'].shape)} "
            f"r_k={result['r_k']} r_v={result['r_v']} gamma={gamma_val:.4f}"
            f"  best_step={result.get('best_step', '?')} stopped_early={result.get('stopped_early', False)}"
        )
        return layer_dir
