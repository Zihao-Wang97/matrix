#!/usr/bin/env python
"""Debug latent KV quantization: python scripts/07_debug_latent_quant.py [options]

Reads latent K/V tensors from disk (or generates synthetic ones),
runs TurboQuantMSE quantize/dequantize, and reports MSE + memory savings.
"""

import argparse
import json
from pathlib import Path

import torch

from hawp_laq.runtime.latent_quant_bridge import (
    create_kv_quantizers,
    quantize_kv_latents,
    dequantize_kv_latents,
    latent_kv_bytes,
    baseline_kv_bytes,
    saving_ratio,
)


def _load_latents(path: str | Path) -> dict[str, torch.Tensor]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Latent file not found: {p}")
    data = torch.load(p, map_location="cpu", weights_only=True)
    if "k_lat" not in data or "v_lat" not in data:
        raise ValueError("Expected keys 'k_lat' and 'v_lat' in latent file")
    return data


def _generate_synthetic_latents(
    seq_len: int, r_k: int, r_v: int, seed: int = 42
) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    k_lat = torch.randn(seq_len, r_k)
    k_lat[:, :4] *= 5.0
    v_lat = torch.randn(seq_len, r_v)
    return {"k_lat": k_lat, "v_lat": v_lat}


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug latent KV quantization")
    parser.add_argument("--latents", type=str, default=None, help="Path to .pt file with k_lat/v_lat")
    parser.add_argument("--seq-len", type=int, default=128, help="Seq len for synthetic latents")
    parser.add_argument("--r-k", type=int, default=64, help="Key latent dim")
    parser.add_argument("--r-v", type=int, default=64, help="Value latent dim")
    parser.add_argument("--k-bits", type=int, default=4, help="Key quantization bits")
    parser.add_argument("--v-bits", type=int, default=4, help="Value quantization bits")
    parser.add_argument("--use-rotation", action="store_true", default=True, help="Use rotation")
    parser.add_argument("--no-rotation", action="store_true", help="Disable rotation")
    parser.add_argument("--group-size", type=int, default=128, help="Quantization group size")
    parser.add_argument("--save-output", type=str, default=None, help="Save quantized result to .pt")
    args = parser.parse_args()

    use_rotation = not args.no_rotation

    if args.latents:
        data = _load_latents(args.latents)
        k_lat = data["k_lat"].float()
        v_lat = data["v_lat"].float()
        r_k = k_lat.shape[-1]
        r_v = v_lat.shape[-1]
        print(f"[load] k_lat={tuple(k_lat.shape)}  v_lat={tuple(v_lat.shape)}")
    else:
        r_k = args.r_k
        r_v = args.r_v
        data = _generate_synthetic_latents(args.seq_len, r_k, r_v)
        k_lat = data["k_lat"]
        v_lat = data["v_lat"]
        print(f"[synth] k_lat={tuple(k_lat.shape)}  v_lat={tuple(v_lat.shape)}")

    print("=" * 60)
    print(f"[config] r_k={r_k}  r_v={r_v}  k_bits={args.k_bits}  v_bits={args.v_bits}")
    print(f"[config] use_rotation={use_rotation}  group_size={args.group_size}")
    print("=" * 60)

    kq, vq = create_kv_quantizers(
        r_k=r_k, r_v=r_v,
        k_bits=args.k_bits, v_bits=args.v_bits,
        use_rotation=use_rotation,
        group_size=args.group_size,
    )

    qkv = quantize_kv_latents(k_lat, v_lat, kq, vq)
    k_hat, v_hat = dequantize_kv_latents(qkv, kq, vq)

    k_mse = (k_lat - k_hat).pow(2).mean().item()
    v_mse = (v_lat - v_hat).pow(2).mean().item()
    k_max_err = (k_lat - k_hat).abs().max().item()
    v_max_err = (v_lat - v_hat).abs().max().item()

    quant_info = latent_kv_bytes(qkv, kq, vq)
    base_info = baseline_kv_bytes(k_lat.shape[0], r_k, r_v)
    ratio = saving_ratio(qkv, kq, vq)

    print(f"\n--- Reconstruction Quality ---")
    print(f"  K  MSE={k_mse:.6f}  max_err={k_max_err:.4f}")
    print(f"  V  MSE={v_mse:.6f}  max_err={v_max_err:.4f}")

    print(f"\n--- Memory ---")
    print(f"  Baseline (fp16): {base_info['k_bytes']:>8d} + {base_info['v_bytes']:>8d} = {base_info['total_bytes']:>8d} bytes")
    print(f"  Quantized:       {quant_info['k_bytes']:>8d} + {quant_info['v_bytes']:>8d} = {quant_info['total_bytes']:>8d} bytes")
    print(f"  Saving ratio:    {ratio:.1%}")

    if args.save_output:
        out_path = Path(args.save_output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "k_q": qkv.k_q,
            "v_q": qkv.v_q,
            "k_hat": k_hat,
            "v_hat": v_hat,
            "k_mse": k_mse,
            "v_mse": v_mse,
            "quant_bytes": quant_info,
            "baseline_bytes": base_info,
            "saving_ratio": ratio,
        }, out_path)
        print(f"\n[save] {out_path}")


if __name__ == "__main__":
    main()
