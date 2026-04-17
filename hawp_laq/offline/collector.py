from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from hawp_laq.config import HAWPLAQConfig
from hawp_laq.offline.hooks import register_qkv_hooks, remove_hooks, count_attention_layers
from hawp_laq.offline.dataset import get_calib_dataloader
from hawp_laq.utils.io import save_pt


class CalibrationCollector:
    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, cfg: HAWPLAQConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = cfg
        self._buffers: dict[int, dict[str, list[torch.Tensor]]] = {}
        self._handles: list = []

    def _on_qkv(self, layer_idx: int, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
        if layer_idx not in self._buffers:
            self._buffers[layer_idx] = {"q": [], "k": [], "v": []}
        self._buffers[layer_idx]["q"].append(q.cpu())
        self._buffers[layer_idx]["k"].append(k.cpu())
        self._buffers[layer_idx]["v"].append(v.cpu())

    def install_hooks(self) -> None:
        self._handles = register_qkv_hooks(self.model, self._on_qkv)
        n_layers = count_attention_layers(self.model)
        n_handles = len(self._handles)
        print(f"[hooks] {n_layers} attention layers, {n_handles} hooks installed")

    def remove_hooks(self) -> None:
        remove_hooks(self._handles)
        self._handles = []

    @torch.inference_mode()
    def collect(self, dataloader: DataLoader) -> None:
        self._buffers = {}
        self.install_hooks()
        device = self.model.device
        total = len(dataloader)
        for i, input_ids in enumerate(dataloader):
            input_ids = input_ids.to(device)
            self.model(input_ids)
            if (i + 1) % max(1, total // 5) == 0 or i == total - 1:
                print(f"[collect] {i + 1}/{total} samples done")
        self.remove_hooks()

    def save(self, output_dir: str | Path | None = None) -> Path:
        out = Path(output_dir) if output_dir else self.cfg.calib.output_dir
        out.mkdir(parents=True, exist_ok=True)
        for idx, data in sorted(self._buffers.items()):
            q_stack = torch.cat(data["q"], dim=0)
            k_stack = torch.cat(data["k"], dim=0)
            v_stack = torch.cat(data["v"], dim=0)
            save_pt({"q": q_stack, "k": k_stack, "v": v_stack}, out / f"layer_{idx}.pt")
            print(f"[save] layer_{idx}.pt  q={tuple(q_stack.shape)} k={tuple(k_stack.shape)} v={tuple(v_stack.shape)}")
        n_heads = getattr(self.model.config, "num_attention_heads", None)
        meta = {
            "n_layers": len(self._buffers),
            "n_heads": n_heads,
            "nsamples": self.cfg.calib.nsamples,
            "seq_len": self.cfg.calib.seq_len,
            "model_id": self.cfg.model.model_id,
        }
        save_pt(meta, out / "meta.pt")
        print(f"[save] meta.pt  {meta}")
        self._buffers = {}
        return out

    @property
    def n_layers(self) -> int:
        return len(self._buffers)


def run_calibration(cfg: HAWPLAQConfig) -> Path:
    from hawp_laq.runtime.generate import load_baseline_model, print_device_info

    print_device_info(cfg.train.device)
    model, tokenizer, _ = load_baseline_model(cfg)

    dl = get_calib_dataloader(
        tokenizer,
        nsamples=cfg.calib.nsamples,
        seq_len=cfg.calib.seq_len,
        dataset_name=cfg.calib.dataset,
    )

    collector = CalibrationCollector(model, tokenizer, cfg)
    collector.collect(dl)
    out = collector.save()
    print(f"[done] calibration artifacts saved to {out}")
    return out
