from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from hawp_laq.offline.losses import (
    attention_output_mse_loss,
    logits_mse_loss,
    total_projector_loss,
    value_reconstruction_loss,
)
from hawp_laq.utils.io import save_pt
from hawp_laq.utils.math_utils import orthogonalize


class ProjectorModule(nn.Module):
    def __init__(self, d_model: int, rank: int, n_heads: int):
        super().__init__()
        self.d_model = d_model
        self.rank = rank
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.p_k = nn.Parameter(torch.randn(d_model, rank) * (rank ** -0.5))
        self.p_v = nn.Parameter(torch.randn(d_model, rank) * (rank ** -0.5))
        self.gamma = nn.Parameter(torch.ones(1))

    def orthogonalize_projectors(self) -> None:
        with torch.no_grad():
            self.p_k.copy_(orthogonalize(self.p_k.data))
            self.p_v.copy_(orthogonalize(self.p_v.data))

    def _reshape_mh(self, x: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.shape
        return x.view(b, s, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        k_recon = (k @ self.p_k) @ self.p_k.T
        v_recon = self.gamma * (v @ self.p_v) @ self.p_v.T

        q_mh = self._reshape_mh(q)
        k_mh = self._reshape_mh(k)
        k_recon_mh = self._reshape_mh(k_recon)
        v_mh = self._reshape_mh(v)
        v_recon_mh = self._reshape_mh(v_recon)

        scale = self.head_dim ** -0.5
        logits_fp = (q_mh @ k_mh.transpose(-2, -1)) * scale
        logits_hat = (q_mh @ k_recon_mh.transpose(-2, -1)) * scale

        attn_probs = torch.softmax(logits_fp, dim=-1)
        attn_probs_hat = torch.softmax(logits_hat, dim=-1)

        attn_out = attn_probs @ v_mh
        attn_out_hat = attn_probs_hat @ v_recon_mh

        return logits_fp, logits_hat, attn_out, attn_out_hat, k_recon, v_recon


class ProjectorTrainer:
    def __init__(
        self,
        d_model: int,
        rank: int,
        n_heads: int,
        lr: float = 1e-3,
        orthogonalize_every: int = 10,
        w_logits: float = 1.0,
        w_attn: float = 1.0,
        w_value: float = 0.5,
        device: str = "cpu",
    ):
        self.d_model = d_model
        self.rank = rank
        self.n_heads = n_heads
        self.lr = lr
        self.orthogonalize_every = orthogonalize_every
        self.w_logits = w_logits
        self.w_attn = w_attn
        self.w_value = w_value
        self.device = device

    def train_one_group(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        n_steps: int = 200,
    ) -> dict:
        module = ProjectorModule(self.d_model, self.rank, self.n_heads).to(self.device)
        optimizer = torch.optim.Adam(module.parameters(), lr=self.lr)

        q = q.to(self.device).float()
        k = k.to(self.device).float()
        v = v.to(self.device).float()

        metrics: dict[str, list] = {
            "total": [],
            "logits": [],
            "attn": [],
            "value": [],
        }

        for step in range(1, n_steps + 1):
            optimizer.zero_grad()
            logits_fp, logits_hat, attn_out, attn_out_hat, k_recon, v_recon = module(q, k, v)

            l_logits = logits_mse_loss(logits_fp, logits_hat)
            l_attn = attention_output_mse_loss(attn_out, attn_out_hat)
            l_val = value_reconstruction_loss(v, module.p_v, module.gamma)
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

        return {
            "p_k": module.p_k.data.cpu().detach(),
            "p_v": module.p_v.data.cpu().detach(),
            "gamma": module.gamma.data.cpu().detach(),
            "metrics": metrics,
        }

    @staticmethod
    def save_result(result: dict, layer_idx: int, output_dir: str | Path) -> Path:
        out = Path(output_dir)
        layer_dir = out / f"layer_{layer_idx}"
        layer_dir.mkdir(parents=True, exist_ok=True)
        save_pt(result, layer_dir / "projector.pt")
        print(f"[save] {layer_dir / 'projector.pt'}  p_k={tuple(result['p_k'].shape)} p_v={tuple(result['p_v'].shape)} gamma={result['gamma'].item():.4f}")
        return layer_dir
