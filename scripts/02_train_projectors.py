#!/usr/bin/env python
"""Train projectors: python scripts/02_train_projectors.py [config]"""

import argparse
from pathlib import Path

import torch
from transformers import AutoConfig

from hawp_laq.config import load_config
from hawp_laq.offline.projector_trainer import ProjectorTrainer
from hawp_laq.utils.io import load_pt


def main() -> None:
    parser = argparse.ArgumentParser(description="HAWP-LAQ projector training (single group)")
    parser.add_argument(
        "config",
        nargs="?",
        default=None,
        help="Path to yaml config (default: configs/dev_local.yaml)",
    )
    parser.add_argument("--layer", type=int, default=None, help="Override target layer index")
    args = parser.parse_args()

    if args.config is None:
        args.config = Path(__file__).resolve().parent.parent / "configs" / "dev_local.yaml"

    cfg = load_config(args.config)
    layer_idx = args.layer if args.layer is not None else cfg.projector.target_layer

    calib_dir = cfg.calib.output_dir
    meta = load_pt(Path(calib_dir) / "meta.pt")
    n_heads = meta.get("n_heads")
    if n_heads is None:
        model_cfg = AutoConfig.from_pretrained(cfg.model.model_id)
        n_heads = model_cfg.num_attention_heads

    layer_data = load_pt(Path(calib_dir) / f"layer_{layer_idx}.pt")
    q = layer_data["q"].float()
    k = layer_data["k"].float()
    v = layer_data["v"].float()
    d_model = q.shape[-1]

    print("=" * 60)
    print(f"[train] layer={layer_idx}  d_model={d_model}  n_heads={n_heads}  rank={cfg.projector.rank}")
    print(f"[train] q={tuple(q.shape)} k={tuple(k.shape)} v={tuple(v.shape)}")
    print(f"[train] n_steps={cfg.projector.n_steps}  lr={cfg.projector.lr}")
    print("=" * 60)

    trainer = ProjectorTrainer(
        d_model=d_model,
        rank=cfg.projector.rank,
        n_heads=n_heads,
        lr=cfg.projector.lr,
        orthogonalize_every=cfg.projector.orthogonalize_every,
        w_logits=cfg.projector.w_logits,
        w_attn=cfg.projector.w_attn,
        w_value=cfg.projector.w_value,
        device=cfg.train.device,
    )

    result = trainer.train_one_group(q, k, v, n_steps=cfg.projector.n_steps)
    first_loss = result["metrics"]["total"][0]
    last_loss = result["metrics"]["total"][-1]
    print(f"\n[result] loss: {first_loss:.6f} -> {last_loss:.6f}  (delta={last_loss - first_loss:.6f})")

    ProjectorTrainer.save_result(result, layer_idx, cfg.projector.output_dir)


if __name__ == "__main__":
    main()
