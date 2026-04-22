#!/usr/bin/env python
"""KV memory profiling: per-layer and total KV bytes for each mode.

Usage:
  python scripts/06_profile_kv_memory.py configs/dev_local.yaml --mode hawp_quant
  python scripts/06_profile_kv_memory.py configs/run_server.yaml --mode hawp_quant_sched
"""

import argparse
from pathlib import Path

import torch
from transformers import AutoConfig

from hawp_laq.config import load_config
from hawp_laq.runtime.generate import (
    load_baseline_model,
    _resolve_device,
    _setup_hawp_quant_on_model,
    _setup_quant_only_on_model,
    generate_hawp_quant,
)
from hawp_laq.modeling.attention_hawp import HAWPAttention
from hawp_laq.modeling.modeling_llama_hawp import convert_llama_to_hawp
from hawp_laq.eval.metrics import collect_kv_metrics, format_kv_metrics
from hawp_laq.utils.memory import format_nbytes
from hawp_laq.utils.io import save_json


def _setup_mode(model, cfg, device, mode: str):
    if mode == "baseline":
        return model, None
    if mode == "hawp_only":
        r_k, r_v = cfg.projector.r_k, cfg.projector.r_v
        model = convert_llama_to_hawp(model, r_k=r_k, r_v=r_v)
        model = model.to(device).eval()
        return model, None
    if mode == "quant_only":
        model, _ = _setup_quant_only_on_model(model, cfg, device)
        return model, "quant"
    if mode in ("hawp_quant", "hawp_quant_all", "hawp_quant_sched"):
        if mode == "hawp_quant_all":
            from hawp_laq.runtime.generate import _setup_hawp_quant_all_on_model
            model = _setup_hawp_quant_all_on_model(model, cfg, device)
        else:
            model = _setup_hawp_quant_on_model(model, cfg, device)
        coordinator = None
        if mode == "hawp_quant_sched":
            from hawp_laq.runtime.scheduler import TokenBudgetScheduler
            from hawp_laq.runtime.cache_manager import ModelCacheCoordinator
            sched = TokenBudgetScheduler(
                total_budget=cfg.sched.total_budget,
                recent_window=cfg.sched.recent_window,
                high_ratio=cfg.sched.high_ratio,
                low_ratio=cfg.sched.low_ratio,
                drop_strategy=getattr(cfg.sched, "drop_strategy", "position"),
            )
            coordinator = ModelCacheCoordinator.from_model(
                model, sched, drop_strategy=getattr(cfg.sched, "drop_strategy", "position"),
            )
        return model, coordinator
    raise ValueError(f"Unknown mode: {mode}")


def _compute_baseline_kv(model_cfg, seq_len: int) -> dict:
    n_layers = getattr(model_cfg, "num_hidden_layers", 12)
    n_heads = getattr(model_cfg, "num_attention_heads", 12)
    n_kv_heads = getattr(model_cfg, "num_key_value_heads", n_heads)
    head_dim = getattr(model_cfg, "hidden_size", 768) // n_heads

    per_layer = seq_len * n_kv_heads * head_dim * 2 * 2
    baseline_total = n_layers * per_layer

    return {
        "n_layers": n_layers,
        "n_kv_heads": n_kv_heads,
        "head_dim": head_dim,
        "baseline_per_layer_bytes": per_layer,
        "baseline_total_bytes": baseline_total,
        "baseline_per_layer_formatted": format_nbytes(per_layer),
        "baseline_total_formatted": format_nbytes(baseline_total),
    }


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description="HAWP-LAQ KV memory profiling")
    parser.add_argument("config", nargs="?", default=None)
    parser.add_argument("--mode",
                        choices=["baseline", "hawp_only", "quant_only", "hawp_quant", "hawp_quant_all", "hawp_quant_sched"],
                        default="baseline")
    parser.add_argument("--seq-lens", nargs="+", type=int,
                        default=[512, 1024, 2048, 4096])
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if args.config is None:
        args.config = Path(__file__).resolve().parent.parent / "configs" / "run_server.yaml"

    cfg = load_config(args.config)
    device = _resolve_device(cfg.train.device)

    model_cfg = AutoConfig.from_pretrained(cfg.model.model_id)
    baseline_info = _compute_baseline_kv(model_cfg, 0)

    print("=" * 60)
    print(f"[profile] mode={args.mode}  model={cfg.model.model_id}")
    print(f"[profile] n_layers={baseline_info['n_layers']}  "
          f"n_kv_heads={baseline_info['n_kv_heads']}  head_dim={baseline_info['head_dim']}")
    print("=" * 60)

    all_results = []

    if args.mode in ("quant_only", "hawp_quant", "hawp_quant_all", "hawp_quant_sched"):
        model, tokenizer, _ = load_baseline_model(cfg)
        model, coordinator = _setup_mode(model, cfg, device, args.mode)
        model.eval()

        for seq_len in args.seq_lens:
            for mod in model.modules():
                if isinstance(mod, HAWPAttention) and mod.use_cache_manager:
                    mod.reset_quant_cache()
            if coordinator is not None:
                coordinator.reset()

            seed_text = "The " * seq_len
            enc = tokenizer(seed_text, return_tensors="pt")
            input_ids = enc["input_ids"][:, :seq_len].to(device)

            outputs = model(input_ids=input_ids, use_cache=False)
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)

            for _ in range(min(31, seq_len)):
                attention_mask = torch.ones(1, input_ids.shape[1] + 1, device=device, dtype=torch.long)
                position_ids = torch.tensor([[input_ids.shape[1]]], device=device, dtype=torch.long)
                outputs = model(input_ids=next_token, attention_mask=attention_mask,
                                position_ids=position_ids, use_cache=False)
                next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
                if coordinator is not None:
                    coordinator.on_new_token()

            metrics = collect_kv_metrics(model)
            bl = _compute_baseline_kv(model_cfg, seq_len + 32)

            entry = {
                "seq_len": seq_len,
                "mode": args.mode,
                "baseline_total_bytes": bl["baseline_total_bytes"],
                "baseline_formatted": bl["baseline_total_formatted"],
                "total_runtime_bytes": metrics["total_runtime_bytes"],
                "compressed_storage_bytes": metrics["compressed_storage_bytes"],
                "runtime_formatted": format_nbytes(metrics["total_runtime_bytes"]),
                "compressed_formatted": format_nbytes(metrics["compressed_storage_bytes"]),
                "runtime_saving_ratio": metrics["runtime_saving_ratio"],
                "compressed_saving_ratio": metrics["compressed_saving_ratio"],
                "recent_tokens": metrics["total_recent_tokens"],
                "archive_tokens": metrics["total_archive_tokens"],
                "per_layer": metrics["per_layer"],
            }
            all_results.append(entry)

            print(f"\n--- seq_len={seq_len} ---")
            print(f"  baseline: {entry['baseline_formatted']}")
            print(f"  [runtime]  {entry['runtime_formatted']}  saving={entry['runtime_saving_ratio']:.1%}")
            print(f"  [compressed storage]  {entry['compressed_formatted']}  saving={entry['compressed_saving_ratio']:.1%}")
            print(f"  recent={entry['recent_tokens']}  archive={entry['archive_tokens']}")
    else:
        for seq_len in args.seq_lens:
            bl = _compute_baseline_kv(model_cfg, seq_len)
            entry = {
                "seq_len": seq_len,
                "mode": args.mode,
                "baseline_total_bytes": bl["baseline_total_bytes"],
                "baseline_formatted": bl["baseline_total_formatted"],
                "kv_total_bytes": bl["baseline_total_bytes"],
                "kv_formatted": bl["baseline_total_formatted"],
                "saving_ratio": 0.0,
                "recent_tokens": seq_len,
                "archive_tokens": 0,
                "per_layer": [],
            }
            all_results.append(entry)
            print(f"  seq_len={seq_len}: baseline={entry['baseline_formatted']}  (no compression)")

    print(f"\n{'='*60}")
    print(f"{'seq_len':>8} {'mode':>18} {'baseline':>12} {'runtime':>12} {'compressed':>12} {'rt_save':>8} {'cmp_save':>8}")
    print("-" * 82)
    for r in all_results:
        rt_fmt = r.get('runtime_formatted', format_nbytes(r.get('kv_total_bytes', 0)))
        cmp_fmt = r.get('compressed_formatted', rt_fmt)
        rt_save = r.get('runtime_saving_ratio', r.get('saving_ratio', 0.0))
        cmp_save = r.get('compressed_saving_ratio', 0.0)
        print(f"{r['seq_len']:>8d} {r['mode']:>18} {r['baseline_formatted']:>12} "
              f"{rt_fmt:>12} {cmp_fmt:>12} {rt_save:>7.1%} {cmp_save:>7.1%}")

    if args.output:
        out_path = Path(args.output)
    else:
        out_path = Path("artifacts/kv_memory_profile.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(all_results, out_path)
    print(f"\n[profile] saved to {out_path}")


if __name__ == "__main__":
    main()
