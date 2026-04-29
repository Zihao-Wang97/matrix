from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from hawp_laq.config import HAWPLAQConfig
from hawp_laq.offline.hooks import register_qkv_hooks, remove_hooks, count_attention_layers
from hawp_laq.offline.dataset import get_calib_dataloader
from hawp_laq.utils.io import save_pt

_NON_ROPE_MODEL_TYPES = {"opt", "gpt_neox"}


def _resolve_capture_mode(capture_mode: str, model_type: str) -> str:
    if capture_mode == "post_rope" and model_type.lower() in _NON_ROPE_MODEL_TYPES:
        raise ValueError(
            f"capture_mode='post_rope' is incompatible with non-RoPE model type '{model_type}'. "
            f"Non-RoPE models (OPT, GPT-NeoX) pre-scale Q by 1/sqrt(d_h) inside the attention "
            f"forward, which causes double-scaling when captured via post_rope. "
            f"Use capture_mode='pre_rope' (or 'auto') instead."
        )
    if capture_mode in ("pre_rope", "post_rope"):
        return capture_mode
    if capture_mode != "auto":
        raise ValueError(f"Unknown capture_mode='{capture_mode}'. Supported: auto, pre_rope, post_rope")
    if model_type.lower() in _NON_ROPE_MODEL_TYPES:
        return "pre_rope"
    return "post_rope"


_ROPE_MODEL_TYPES = {"llama", "mistral", "qwen2", "phi3", "gemma", "gemma2"}


def _warn_pre_rope_with_rope_model(model_type: str) -> None:
    if model_type.lower() in _ROPE_MODEL_TYPES:
        warnings.warn(
            f"[calib:pre_rope] Model type '{model_type}' uses RoPE. "
            f"pre_rope captures K/V BEFORE RoPE is applied, which means "
            f"the projector will need to compensate for the missing RoPE. "
            f"If the model uses FlashAttention2 (common on GPU), hooks may "
            f"not find the attention modules. Consider using "
            f"capture_mode='post_rope' (or 'auto') for more reliable results.",
            UserWarning,
            stacklevel=3,
        )


def _repeat_kv_4d(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """[bsz, n_kv_heads, seq, head_dim] -> [bsz, n_kv_heads * n_rep, seq, head_dim]"""
    if n_rep == 1:
        return x
    bsz, n_kv_heads, seq_len, head_dim = x.shape
    x = x[:, :, None, :, :].expand(bsz, n_kv_heads, n_rep, seq_len, head_dim)
    return x.reshape(bsz, n_kv_heads * n_rep, seq_len, head_dim)


def _expand_kv_for_trainer(
    k: torch.Tensor,
    n_heads: int,
    n_kv_heads: int,
) -> torch.Tensor:
    """Expand k/v from [bsz, seq, n_kv_heads*head_dim] to [bsz, seq, n_heads*head_dim].

    For non-GQA models (n_kv_heads == n_heads) this is a no-op.
    For GQA/MQA models, repeats kv heads to match n_heads, matching
    the _repeat_kv operation in attention forward.
    """
    n_rep = n_heads // n_kv_heads
    if n_heads % n_kv_heads != 0:
        raise ValueError(
            f"n_heads ({n_heads}) must be divisible by n_kv_heads ({n_kv_heads}) "
            f"for GQA kv expansion"
        )
    if n_rep == 1:
        return k
    bsz, seq_len, _ = k.shape
    head_dim = k.shape[-1] // n_kv_heads
    k_4d = k.view(bsz, seq_len, n_kv_heads, head_dim).transpose(1, 2)
    k_4d = _repeat_kv_4d(k_4d, n_rep)
    return k_4d.transpose(1, 2).reshape(bsz, seq_len, n_heads * head_dim)


class CalibrationCollector:
    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, cfg: HAWPLAQConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = cfg
        self._buffers: dict[int, dict[str, list[torch.Tensor]]] = {}
        self._handles: list = []
        self._capture_mode: str | None = None
        self._collector_impl: str | None = None
        self._n_q_heads: int | None = None
        self._n_kv_heads: int | None = None

    def _init_head_counts(self) -> None:
        if self._n_q_heads is not None:
            return
        config = self.model.config
        n_q = getattr(config, "num_attention_heads", None)
        n_kv = getattr(config, "num_key_value_heads", None)
        if not isinstance(n_q, int):
            n_q = 12
        if not isinstance(n_kv, int):
            n_kv = n_q
        self._n_q_heads = n_q
        self._n_kv_heads = n_kv

    def _on_qkv(self, layer_idx: int, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
        self._init_head_counts()
        k = _expand_kv_for_trainer(k, self._n_q_heads, self._n_kv_heads)
        v = _expand_kv_for_trainer(v, self._n_q_heads, self._n_kv_heads)
        if layer_idx not in self._buffers:
            self._buffers[layer_idx] = {"q": [], "k": [], "v": []}
        self._buffers[layer_idx]["q"].append(q.cpu())
        self._buffers[layer_idx]["k"].append(k.cpu())
        self._buffers[layer_idx]["v"].append(v.cpu())

    def _on_post_rope_callback(
        self,
        layer_idx: int,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> None:
        self._init_head_counts()
        bsz, n_q_heads, seq_len, head_dim = query_states.shape
        n_kv_heads = key_states.shape[1]
        n_rep = n_q_heads // n_kv_heads
        if n_q_heads % n_kv_heads != 0:
            raise ValueError(
                f"n_q_heads ({n_q_heads}) must be divisible by n_kv_heads ({n_kv_heads}) "
                f"for GQA kv expansion"
            )

        q_3d = query_states.transpose(1, 2).reshape(bsz, seq_len, n_q_heads * head_dim)
        k_expanded = _repeat_kv_4d(key_states, n_rep)
        v_expanded = _repeat_kv_4d(value_states, n_rep)
        k_3d = k_expanded.transpose(1, 2).reshape(bsz, seq_len, n_q_heads * head_dim)
        v_3d = v_expanded.transpose(1, 2).reshape(bsz, seq_len, n_q_heads * head_dim)

        if layer_idx not in self._buffers:
            self._buffers[layer_idx] = {"q": [], "k": [], "v": []}
        self._buffers[layer_idx]["q"].append(q_3d.cpu())
        self._buffers[layer_idx]["k"].append(k_3d.cpu())
        self._buffers[layer_idx]["v"].append(v_3d.cpu())

    def _setup_post_rope_collection(self) -> None:
        from hawp_laq.modeling.attention_hawp import HAWPAttention
        from hawp_laq.modeling.modeling_llama_hawp import convert_llama_to_hawp

        self._init_head_counts()
        head_dim = None
        for _, mod in self.model.named_modules():
            for attr in ("self_attn", "attention"):
                if hasattr(mod, attr):
                    attn = getattr(mod, attr)
                    if hasattr(attn, "q_proj") and hasattr(attn, "k_proj"):
                        head_dim = attn.q_proj.out_features // self._n_q_heads
                        break
            if head_dim is not None:
                break

        if head_dim is None:
            raise ValueError("Cannot determine head_dim for post_rope calibration")

        self.model = convert_llama_to_hawp(
            self.model,
            r_k=head_dim,
            r_v=head_dim,
            allow_default_full_rank=True,
            logit_scale_mode="dh",
            gamma_mode="off",
            gamma_value=None,
            use_archive_k_ip_approx=True,
        )

        callback: Callable = self._on_post_rope_callback
        for _, mod in self.model.named_modules():
            if isinstance(mod, HAWPAttention):
                mod._calib_callback = callback

        n_attn = sum(1 for _, m in self.model.named_modules() if isinstance(m, HAWPAttention))
        warnings.warn(
            "[calib:post_rope] Collection uses HAWPAttention (eager matmul+softmax), "
            "which may differ numerically from the original SDPA/Flash path. "
            "The RoPE alignment fix is more significant than this difference.",
            UserWarning,
            stacklevel=2,
        )
        print(f"[calib:post_rope] converted model to full-rank HAWPAttention, {n_attn} layers")

    def _teardown_post_rope_collection(self) -> None:
        from hawp_laq.modeling.attention_hawp import HAWPAttention

        for _, mod in self.model.named_modules():
            if isinstance(mod, HAWPAttention):
                mod._calib_callback = None

    def install_hooks(self) -> None:
        self._handles = register_qkv_hooks(self.model, self._on_qkv)
        n_layers = count_attention_layers(self.model)
        n_handles = len(self._handles)
        print(f"[calib:pre_rope] {n_layers} attention layers, {n_handles} hooks installed")

        if n_layers == 0:
            model_type = getattr(self.model.config, "model_type", "")
            raise RuntimeError(
                f"[calib:pre_rope] No attention modules found in model "
                f"(model_type={model_type!r}). The hook-based pre_rope path "
                f"requires attention modules whose class names are in the "
                f"recognized set (see hooks._find_attention_modules). "
                f"If your model uses FlashAttention2, make sure the class name "
                f"is registered. Alternatively, use capture_mode='post_rope' "
                f"which bypasses hooks entirely."
            )

    def remove_hooks(self) -> None:
        remove_hooks(self._handles)
        self._handles = []

    @torch.inference_mode()
    def collect(self, dataloader: DataLoader) -> None:
        self._buffers = {}

        model_type = getattr(self.model.config, "model_type", "").lower()
        self._capture_mode = _resolve_capture_mode(self.cfg.calib.capture_mode, model_type)
        print(f"[calib] capture_mode: cfg={self.cfg.calib.capture_mode} -> resolved={self._capture_mode} (model_type={model_type})")

        if self._capture_mode == "post_rope":
            self._collector_impl = "hawp_full_rank_eager"
            self._setup_post_rope_collection()
            try:
                self._collect_post_rope(dataloader)
            finally:
                self._teardown_post_rope_collection()
        else:
            self._collector_impl = "original_model_hooks"
            _warn_pre_rope_with_rope_model(model_type)
            self.install_hooks()
            try:
                self._collect_pre_rope(dataloader)
            finally:
                self.remove_hooks()

    @torch.inference_mode()
    def _collect_pre_rope(self, dataloader: DataLoader) -> None:
        device = self.model.device
        total = len(dataloader)
        for i, input_ids in enumerate(dataloader):
            input_ids = input_ids.to(device)
            self.model(input_ids)
            if (i + 1) % max(1, total // 5) == 0 or i == total - 1:
                print(f"[calib:pre_rope] {i + 1}/{total} samples done")

    @torch.inference_mode()
    def _collect_post_rope(self, dataloader: DataLoader) -> None:
        device = self.model.device
        total = len(dataloader)
        for i, input_ids in enumerate(dataloader):
            input_ids = input_ids.to(device)
            bsz, seq_len = input_ids.shape
            attention_mask = torch.ones(bsz, seq_len, device=device, dtype=torch.long)
            position_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0).expand(bsz, -1)
            self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
            )
            if (i + 1) % max(1, total // 5) == 0 or i == total - 1:
                print(f"[calib:post_rope] {i + 1}/{total} samples done")

    def save(self, output_dir: str | Path | None = None) -> Path:
        out = Path(output_dir) if output_dir else self.cfg.calib.output_dir
        out.mkdir(parents=True, exist_ok=True)
        for idx, data in sorted(self._buffers.items()):
            q_stack = torch.cat(data["q"], dim=0)
            k_stack = torch.cat(data["k"], dim=0)
            v_stack = torch.cat(data["v"], dim=0)
            save_pt({"q": q_stack, "k": k_stack, "v": v_stack}, out / f"layer_{idx}.pt")
            print(f"[save] layer_{idx}.pt  q={tuple(q_stack.shape)} k={tuple(k_stack.shape)} v={tuple(v_stack.shape)}")

        model_type = getattr(self.model.config, "model_type", "") or ""
        if not isinstance(model_type, str):
            model_type = str(model_type)
        self._init_head_counts()
        n_heads = self._n_q_heads
        n_kv_heads = self._n_kv_heads
        meta: dict[str, Any] = {
            "n_layers": len(self._buffers),
            "n_heads": n_heads,
            "n_kv_heads": n_kv_heads,
            "nsamples": self.cfg.calib.nsamples,
            "seq_len": self.cfg.calib.seq_len,
            "model_id": self.cfg.model.model_id,
            "capture_mode": self._capture_mode or "unknown",
            "collector_impl": self._collector_impl or "unknown",
            "rope_applied": (self._capture_mode == "post_rope"),
            "model_type": model_type,
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
        data_root=cfg.data.root,
    )

    collector = CalibrationCollector(model, tokenizer, cfg)
    collector.collect(dl)
    out = collector.save()
    print(f"[done] calibration artifacts saved to {out}")
    return out
