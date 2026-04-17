from __future__ import annotations

from pathlib import Path

import torch

from hawp_laq.modeling.attention_hawp import HAWPAttention
from hawp_laq.runtime.quantizer import KQuantizer, VQuantizer
from hawp_laq.runtime.scheduler import TokenBudgetScheduler
from hawp_laq.utils.io import save_json, save_pt
from hawp_laq.utils.memory import tensor_nbytes, format_nbytes


class CompressorPackage:
    def __init__(
        self,
        projector_dir: str | Path,
        n_layers: int,
        n_heads: int,
        head_dim: int,
        k_group_size: int = 128,
        v_group_size: int = 128,
        use_rotation: bool = False,
        outlier_threshold: float | None = None,
        total_budget: int = 4096,
        recent_window: int = 64,
        high_ratio: float = 0.25,
        low_ratio: float = 0.60,
    ):
        self.projector_dir = Path(projector_dir)
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.k_group_size = k_group_size
        self.v_group_size = v_group_size
        self.use_rotation = use_rotation
        self.outlier_threshold = outlier_threshold
        self.total_budget = total_budget
        self.recent_window = recent_window
        self.high_ratio = high_ratio
        self.low_ratio = low_ratio

        self._projectors: dict[int, dict] = {}
        self._load_projectors()

    def _load_projectors(self) -> None:
        for layer_idx in range(self.n_layers):
            pt_path = self.projector_dir / f"layer_{layer_idx}" / "projector.pt"
            if pt_path.exists():
                data = torch.load(pt_path, map_location="cpu", weights_only=True)
                self._projectors[layer_idx] = data

    @property
    def ranks(self) -> dict[int, tuple[int, int]]:
        result = {}
        for idx, data in self._projectors.items():
            r_k = data.get("r_k", self.head_dim)
            r_v = data.get("r_v", self.head_dim)
            result[idx] = (r_k, r_v)
        return result

    def kv_bytes_per_layer(self, seq_len: int) -> dict[int, dict]:
        per_layer: dict[int, dict] = {}
        for layer_idx in range(self.n_layers):
            proj = self._projectors.get(layer_idx)
            if proj is not None:
                r_k = proj.get("r_k", self.head_dim)
                r_v = proj.get("r_v", self.head_dim)
            else:
                r_k = self.head_dim
                r_v = self.head_dim

            n_kv_heads = self.n_heads

            baseline_bytes = 2 * seq_len * n_kv_heads * self.head_dim * 2
            latent_bytes = 2 * seq_len * n_kv_heads * r_k * 2
            if r_v != r_k:
                latent_bytes = seq_len * n_kv_heads * r_k * 2 + seq_len * n_kv_heads * r_v * 2

            k_quant = KQuantizer(group_size=self.k_group_size, use_rotation=self.use_rotation)
            v_quant = VQuantizer(group_size=self.v_group_size, outlier_threshold=self.outlier_threshold)

            k_latent_per_token = n_kv_heads * r_k
            v_latent_per_token = n_kv_heads * r_v

            k_scale_bytes = seq_len * n_kv_heads * ((k_latent_per_token + self.k_group_size - 1) // self.k_group_size) * 4
            k_q_bytes = seq_len * k_latent_per_token
            v_scale_bytes = seq_len * n_kv_heads * ((v_latent_per_token + self.v_group_size - 1) // self.v_group_size) * 4
            v_zp_bytes = v_scale_bytes
            v_q_bytes = seq_len * v_latent_per_token

            quant_bytes = k_q_bytes + k_scale_bytes + v_q_bytes + v_scale_bytes + v_zp_bytes

            per_layer[layer_idx] = {
                "baseline_bytes": baseline_bytes,
                "latent_bytes": latent_bytes,
                "quant_bytes": quant_bytes,
                "r_k": r_k,
                "r_v": r_v,
                "baseline_formatted": format_nbytes(baseline_bytes),
                "latent_formatted": format_nbytes(latent_bytes),
                "quant_formatted": format_nbytes(quant_bytes),
            }
        return per_layer

    def total_kv_bytes(self, seq_len: int) -> dict:
        per_layer = self.kv_bytes_per_layer(seq_len)
        baseline_total = sum(v["baseline_bytes"] for v in per_layer.values())
        latent_total = sum(v["latent_bytes"] for v in per_layer.values())
        quant_total = sum(v["quant_bytes"] for v in per_layer.values())
        return {
            "seq_len": seq_len,
            "n_layers": self.n_layers,
            "baseline_total_bytes": baseline_total,
            "latent_total_bytes": latent_total,
            "quant_total_bytes": quant_total,
            "baseline_formatted": format_nbytes(baseline_total),
            "latent_formatted": format_nbytes(latent_total),
            "quant_formatted": format_nbytes(quant_total),
            "latent_saving_ratio": 1.0 - latent_total / baseline_total if baseline_total > 0 else 0.0,
            "quant_saving_ratio": 1.0 - quant_total / baseline_total if baseline_total > 0 else 0.0,
            "per_layer": per_layer,
        }

    def apply_to_model(self, model: torch.nn.Module) -> None:
        for name, module in model.named_modules():
            if isinstance(module, HAWPAttention):
                proj = self._projectors.get(module.layer_idx)
                if proj is None:
                    continue
                p_k = proj["p_k"].to(module.p_k.device, module.p_k.dtype)
                p_v = proj["p_v"].to(module.p_v.device, module.p_v.dtype)
                if p_k.shape == module.p_k.shape:
                    module.p_k.data.copy_(p_k)
                if p_v.shape == module.p_v.shape:
                    module.p_v.data.copy_(p_v)
                if "gamma_k" in proj:
                    module.gamma.data.copy_(
                        proj["gamma_k"].to(module.gamma.device, module.gamma.dtype),
                    )
                elif "gamma" in proj:
                    module.gamma.data.copy_(
                        proj["gamma"].to(module.gamma.device, module.gamma.dtype),
                    )

    def save(self, output_dir: str | Path) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        meta = {
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "head_dim": self.head_dim,
            "k_group_size": self.k_group_size,
            "v_group_size": self.v_group_size,
            "use_rotation": self.use_rotation,
            "total_budget": self.total_budget,
            "recent_window": self.recent_window,
            "high_ratio": self.high_ratio,
            "low_ratio": self.low_ratio,
            "ranks": {str(k): {"r_k": v[0], "r_v": v[1]} for k, v in self.ranks.items()},
        }
        save_json(meta, output_dir / "compressor_meta.json")

        torch.save(
            {idx: data for idx, data in self._projectors.items()},
            output_dir / "projectors.pt",
        )

        for seq_len in (512, 1024, 2048, 4096, 8192):
            summary = self.total_kv_bytes(seq_len)
            per_layer_only = summary.pop("per_layer")
            save_json(summary, output_dir / f"kv_profile_seqlen{seq_len}.json")
            save_json(per_layer_only, output_dir / f"kv_profile_seqlen{seq_len}_per_layer.json")

        return output_dir

    @classmethod
    def from_directory(cls, pkg_dir: str | Path) -> CompressorPackage:
        pkg_dir = Path(pkg_dir)
        meta = load_json(str(pkg_dir / "compressor_meta.json")) if (pkg_dir / "compressor_meta.json").exists() else {}
        return cls(
            projector_dir=pkg_dir,
            n_layers=meta.get("n_layers", 0),
            n_heads=meta.get("n_heads", 0),
            head_dim=meta.get("head_dim", 0),
            k_group_size=meta.get("k_group_size", 128),
            v_group_size=meta.get("v_group_size", 128),
            use_rotation=meta.get("use_rotation", False),
            total_budget=meta.get("total_budget", 4096),
            recent_window=meta.get("recent_window", 64),
            high_ratio=meta.get("high_ratio", 0.25),
            low_ratio=meta.get("low_ratio", 0.60),
        )


def load_json(path: str | Path) -> dict:
    import json
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)
