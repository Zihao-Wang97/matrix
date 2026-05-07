from __future__ import annotations

import inspect
import shutil
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from hawp_laq.config import HAWPLAQConfig
from hawp_laq.modeling.modeling_llama_hawp import _find_layers_and_attn
from hawp_laq.offline.dataset import get_calib_dataloader
from hawp_laq.utils.io import save_pt


_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def _storage_dtype(name: str) -> torch.dtype:
    if name not in _DTYPE_MAP:
        raise ValueError(f"layer_distill.storage_dtype must be one of {sorted(_DTYPE_MAP)}, got {name!r}")
    return _DTYPE_MAP[name]


def _model_accepts_kwarg(model: torch.nn.Module, name: str) -> bool:
    try:
        sig = inspect.signature(model.forward)
    except (TypeError, ValueError):
        return False
    return name in sig.parameters


class LayerDistillDataCollector:
    """Collect teacher decoder-layer inputs and outputs for full layer distillation."""

    def __init__(self, model: torch.nn.Module, cfg: HAWPLAQConfig):
        self.model = model
        self.cfg = cfg
        self.storage_dtype = _storage_dtype(cfg.layer_distill.storage_dtype)
        self.layers = _find_layers_and_attn(model)
        self.handles: list[Any] = []
        self.current: dict[int, dict[str, torch.Tensor]] = {}

    def install_hooks(self) -> None:
        if not self.layers:
            model_type = getattr(self.model.config, "model_type", "")
            raise RuntimeError(
                f"[layer_distill:data] No compatible decoder layers found (model_type={model_type!r})."
            )

        for layer_idx, (_name, layer, _attn) in enumerate(self.layers):
            self.handles.append(layer.register_forward_pre_hook(
                self._make_pre_hook(layer_idx),
                with_kwargs=True,
            ))
            self.handles.append(layer.register_forward_hook(
                self._make_post_hook(layer_idx),
                with_kwargs=True,
            ))
        print(f"[layer_distill:data] {len(self.layers)} decoder layers, {len(self.handles)} hooks installed")

    def remove_hooks(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def _to_store(self, x: torch.Tensor) -> torch.Tensor:
        return x.detach().to(device="cpu", dtype=self.storage_dtype).contiguous()

    def _make_pre_hook(self, layer_idx: int):
        def hook(_module, args, kwargs):
            if args:
                hidden = args[0]
            else:
                hidden = kwargs.get("hidden_states")
            if hidden is None:
                raise RuntimeError(f"[layer_distill:data] layer {layer_idx}: cannot find hidden_states input")
            self.current.setdefault(layer_idx, {})["hidden_in"] = self._to_store(hidden)

        return hook

    def _make_post_hook(self, layer_idx: int):
        def hook(_module, _args, _kwargs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            if hidden is None:
                raise RuntimeError(f"[layer_distill:data] layer {layer_idx}: cannot find hidden_states output")
            self.current.setdefault(layer_idx, {})["hidden_out"] = self._to_store(hidden)

        return hook

    def _forward_model(self, input_ids: torch.Tensor) -> None:
        bsz, seq_len = input_ids.shape
        device = input_ids.device
        kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": torch.ones(bsz, seq_len, device=device, dtype=torch.long),
            "use_cache": False,
        }
        if _model_accepts_kwarg(self.model, "position_ids"):
            kwargs["position_ids"] = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0).expand(bsz, -1)
        self.model(**kwargs)

    @torch.inference_mode()
    def collect(self, dataloader: DataLoader, output_dir: str | Path | None = None) -> Path:
        out = Path(output_dir) if output_dir else self.cfg.layer_distill.data_dir
        out.mkdir(parents=True, exist_ok=True)
        for layer_idx in range(len(self.layers)):
            (out / f"layer_{layer_idx}").mkdir(parents=True, exist_ok=True)

        self.model.eval()
        device = next(self.model.parameters()).device
        total = len(dataloader)
        chunks_written = 0
        samples_written = 0

        self.install_hooks()
        try:
            for chunk_idx, input_ids in enumerate(dataloader):
                self.current = {}
                input_ids = input_ids.to(device)
                self._forward_model(input_ids)

                missing = [
                    i for i in range(len(self.layers))
                    if "hidden_in" not in self.current.get(i, {}) or "hidden_out" not in self.current.get(i, {})
                ]
                if missing:
                    raise RuntimeError(f"[layer_distill:data] missing hidden tensors for layers: {missing}")

                for layer_idx in range(len(self.layers)):
                    layer_dir = out / f"layer_{layer_idx}"
                    save_pt(
                        {
                            "hidden_in": self.current[layer_idx]["hidden_in"],
                            "hidden_out": self.current[layer_idx]["hidden_out"],
                        },
                        layer_dir / f"chunk_{chunk_idx:05d}.pt",
                    )

                chunks_written += 1
                samples_written += int(input_ids.shape[0])
                if (chunk_idx + 1) % max(1, total // 5) == 0 or chunk_idx == total - 1:
                    print(f"[layer_distill:data] {chunk_idx + 1}/{total} batches saved")
        finally:
            self.remove_hooks()
            self.current = {}

        meta = self._build_meta(chunks_written, samples_written)
        save_pt(meta, out / "meta.pt")
        print(f"[save] layer_distill meta.pt  {meta}")
        return out

    def _build_meta(self, chunks_written: int, samples_written: int) -> dict[str, Any]:
        config = self.model.config
        model_type = getattr(config, "model_type", "") or ""
        hidden_size = int(getattr(config, "hidden_size", 0) or getattr(config, "word_embed_proj_dim", 0) or 0)
        n_heads = int(getattr(config, "num_attention_heads", 0) or 0)
        n_kv_heads = int(getattr(config, "num_key_value_heads", n_heads) or n_heads)
        return {
            "n_layers": len(self.layers),
            "nsamples": samples_written,
            "seq_len": int(self.cfg.layer_distill.seq_len or self.cfg.calib.seq_len),
            "chunks": chunks_written,
            "batch_size": self.cfg.layer_distill.batch_size,
            "model_id": self.cfg.model.model_id,
            "model_type": str(model_type),
            "hidden_size": hidden_size,
            "n_heads": n_heads,
            "n_kv_heads": n_kv_heads,
            "storage_dtype": self.cfg.layer_distill.storage_dtype,
            "collector_impl": "decoder_layer_input_output_hooks",
        }


def run_layer_distill_collection(
    cfg: HAWPLAQConfig,
    *,
    output_dir: str | Path | None = None,
    clean_output_dir: bool = False,
) -> Path:
    from hawp_laq.runtime.generate import load_baseline_model, print_device_info

    out = Path(output_dir) if output_dir else cfg.layer_distill.data_dir
    if clean_output_dir and out.exists():
        print(f"[layer_distill:data] --clean-output-dir: removing {out}")
        shutil.rmtree(out)

    print_device_info(cfg.train.device)
    model, tokenizer, _device = load_baseline_model(cfg)

    nsamples = int(cfg.layer_distill.nsamples or cfg.calib.nsamples)
    seq_len = int(cfg.layer_distill.seq_len or cfg.calib.seq_len)
    base_dl = get_calib_dataloader(
        tokenizer,
        nsamples=nsamples,
        seq_len=seq_len,
        dataset_name=cfg.calib.dataset,
        data_root=cfg.data.root,
    )
    dataloader = DataLoader(
        list(base_dl.dataset),
        batch_size=max(1, int(cfg.layer_distill.batch_size)),
        shuffle=False,
    )

    collector = LayerDistillDataCollector(model, cfg)
    out = collector.collect(dataloader, out)
    print(f"[done] layer distill data saved to {out}")
    return out
