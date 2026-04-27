from __future__ import annotations

from dataclasses import dataclass

import torch

from hawp_laq.runtime.turboquant import TurboQuantMSE, TurboQuantizedTensor


@dataclass
class QuantizedLatentKV:
    """Container for a pair of quantized latent K/V tensors.

    Attributes:
        k_q: Quantized latent keys.
        v_q: Quantized latent values.
    """

    k_q: TurboQuantizedTensor
    v_q: TurboQuantizedTensor


def create_kv_quantizers(
    r_k: int,
    r_v: int,
    k_bits: int = 4,
    v_bits: int = 4,
    use_rotation: bool = True,
    group_size: int = 128,
    device: torch.device | str | None = None,
) -> tuple[TurboQuantMSE, TurboQuantMSE]:
    """Create a matched pair of TurboQuantMSE instances for K and V latents.

    Args:
        r_k: Latent dimension for keys.
        r_v: Latent dimension for values.
        k_bits: Bits per element for key quantization.
        v_bits: Bits per element for value quantization.
        use_rotation: Whether to enable random rotation.
        group_size: Quantization group size.
        device: Device for rotation matrices.

    Returns:
        ``(k_quantizer, v_quantizer)`` tuple.
    """

    k_quantizer = TurboQuantMSE(
        dim=r_k, bits=k_bits, use_rotation=use_rotation,
        group_size=group_size, device=device,
    )
    v_quantizer = TurboQuantMSE(
        dim=r_v, bits=v_bits, use_rotation=use_rotation,
        group_size=group_size, device=device,
    )
    return k_quantizer, v_quantizer


def quantize_kv_latents(
    k_lat: torch.Tensor,
    v_lat: torch.Tensor,
    k_quantizer: TurboQuantMSE,
    v_quantizer: TurboQuantMSE,
) -> QuantizedLatentKV:
    """Quantize latent K and V tensors.

    Args:
        k_lat: Latent keys of shape ``[T, r_k]`` or ``[B, T, r_k]``.
        v_lat: Latent values of shape ``[T, r_v]`` or ``[B, T, r_v]``.
        k_quantizer: TurboQuantMSE configured for ``dim=r_k``.
        v_quantizer: TurboQuantMSE configured for ``dim=r_v``.

    Returns:
        A :class:`QuantizedLatentKV` holding both quantized tensors.

    Raises:
        ValueError: If ``k_lat`` or ``v_lat`` have unsupported shapes or
            mismatched last dimensions.
    """
    if k_lat.dim() not in (2, 3):
        raise ValueError(
            f"k_lat must be 2-D [T, r_k] or 3-D [B, T, r_k], "
            f"got {k_lat.dim()}-D shape {tuple(k_lat.shape)}"
        )
    if v_lat.dim() not in (2, 3):
        raise ValueError(
            f"v_lat must be 2-D [T, r_v] or 3-D [B, T, r_v], "
            f"got {v_lat.dim()}-D shape {tuple(v_lat.shape)}"
        )
    if k_lat.shape[-1] != k_quantizer.dim:
        raise ValueError(
            f"k_lat last dim ({k_lat.shape[-1]}) != k_quantizer.dim ({k_quantizer.dim})"
        )
    if v_lat.shape[-1] != v_quantizer.dim:
        raise ValueError(
            f"v_lat last dim ({v_lat.shape[-1]}) != v_quantizer.dim ({v_quantizer.dim})"
        )

    k_q = k_quantizer.quantize(k_lat)
    v_q = v_quantizer.quantize(v_lat)
    return QuantizedLatentKV(k_q=k_q, v_q=v_q)


def dequantize_kv_latents(
    qkv: QuantizedLatentKV,
    k_quantizer: TurboQuantMSE,
    v_quantizer: TurboQuantMSE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dequantize latent K and V back to float.

    Args:
        qkv: Previously quantized latent KV pair.
        k_quantizer: The same TurboQuantMSE used for key quantization.
        v_quantizer: The same TurboQuantMSE used for value quantization.

    Returns:
        ``(k_hat, v_hat)`` float tensors with shapes matching the originals.
    """
    k_hat = k_quantizer.dequantize(qkv.k_q)
    v_hat = v_quantizer.dequantize(qkv.v_q)
    return k_hat, v_hat


def latent_kv_bytes(
    qkv: QuantizedLatentKV,
    k_quantizer: TurboQuantMSE,
    v_quantizer: TurboQuantMSE,
) -> dict[str, int]:
    """Estimate storage size for quantized latent KV.

    Args:
        qkv: Quantized latent KV pair.
        k_quantizer: Key quantizer (used for ``estimate_num_bytes``).
        v_quantizer: Value quantizer (used for ``estimate_num_bytes``).

    Returns:
        Dictionary with ``k_bytes``, ``v_bytes``, and ``total_bytes``.
    """
    k_bytes = k_quantizer.estimate_num_bytes(qkv.k_q)
    v_bytes = v_quantizer.estimate_num_bytes(qkv.v_q)
    return {
        "k_bytes": k_bytes,
        "v_bytes": v_bytes,
        "total_bytes": k_bytes + v_bytes,
    }


def baseline_kv_bytes(
    seq_len: int,
    r_k: int,
    r_v: int,
    dtype: torch.dtype = torch.float16,
) -> dict[str, int]:
    """Compute the baseline (unquantized) KV storage size in bytes.

    Args:
        seq_len: Number of tokens.
        r_k: Key latent dimension.
        r_v: Value latent dimension.
        dtype: Assumed storage dtype (default float16).

    Returns:
        Dictionary with ``k_bytes``, ``v_bytes``, and ``total_bytes``.
    """
    elem_size = torch.tensor([], dtype=dtype).element_size()
    k_bytes = seq_len * r_k * elem_size
    v_bytes = seq_len * r_v * elem_size
    return {
        "k_bytes": k_bytes,
        "v_bytes": v_bytes,
        "total_bytes": k_bytes + v_bytes,
    }


def saving_ratio(
    qkv: QuantizedLatentKV,
    k_quantizer: TurboQuantMSE,
    v_quantizer: TurboQuantMSE,
) -> float:
    """Compute KV compression ratio: 1 - quantized / baseline (float16).

    Args:
        qkv: Quantized latent KV pair.
        k_quantizer: Key quantizer.
        v_quantizer: Value quantizer.

    Returns:
        Saving ratio in [0, 1).  Higher is better.
    """
    quant = latent_kv_bytes(qkv, k_quantizer, v_quantizer)
    seq_len = qkv.k_q.shape_orig[0] if qkv.k_q.shape_orig[0] > 0 else qkv.k_q.q.shape[0]
    r_k = qkv.k_q.shape_orig[-1]
    r_v = qkv.v_q.shape_orig[-1]
    base = baseline_kv_bytes(seq_len, r_k, r_v)
    if base["total_bytes"] == 0:
        return 0.0
    return 1.0 - quant["total_bytes"] / base["total_bytes"]
