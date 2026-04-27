from __future__ import annotations

from dataclasses import dataclass

import torch

from hawp_laq.utils.math_utils import orthogonalize
from hawp_laq.utils.packbits import pack_uint4, unpack_uint4, pack_uint2, unpack_uint2, pack_bool, unpack_bool


@dataclass
class TurboQuantState:
    """Snapshot of a TurboQuantMSE instance for serialization."""

    rotation: torch.Tensor | None
    bits: int
    group_size: int
    dim: int
    use_rotation: bool


@dataclass
class TurboQuantizedTensor:
    """Container for a tensor quantized by TurboQuantMSE.

    Attributes:
        q: Stored quantized integers, dtype uint8. For bits < 8 this
            tensor is packed rather than [N, D]: 4-bit uses
            [N, ceil(D / 2)], 2-bit uses [N, ceil(D / 4)], and 3-bit
            currently uses the 4-bit nibble packing format. 8-bit keeps
            shape [N, D].
        scale: Per-group scale, shape [N, n_groups], dtype float32.
        zero_point: Per-group zero-point, shape [N, n_groups], dtype uint8.
        rotation: Orthogonal rotation matrix [D, D] or None.
        shape_orig: Original input shape before reshape.
        bits: Bits per element.
        group_size: Quantization group size.
    """

    q: torch.Tensor
    scale: torch.Tensor
    zero_point: torch.Tensor
    rotation: torch.Tensor | None
    shape_orig: tuple[int, ...]
    bits: int
    group_size: int


class TurboQuantMSE:
    """MSE-optimized TurboQuant for low-dimensional latent KV.

    Pipeline:
        1. (Optional) Apply random orthogonal rotation to spread energy.
        2. Group-based affine scalar quantization with min-max scale + zero-point.

    The rotation step redistributes energy uniformly across dimensions,
    making per-group scalar quantization significantly more effective.

    Args:
        dim: Last dimension of tensors to be quantized.
        bits: Bits per element. Supported: 2, 3, 4, 8.
            Note: 3-bit values are currently stored with the 4-bit nibble
            packing format, so storage is 4-bit packed while the value range
            remains 3-bit.
        use_rotation: Whether to apply random orthogonal rotation before
            quantization.  Default True.
        group_size: Number of consecutive elements per quantization group.
            Must be positive.  Defaults to 128.
        device: Device on which to allocate the rotation matrix.

    Raises:
        ValueError: If dim <= 0, bits is unsupported, or group_size <= 0.
    """

    _SUPPORTED_BITS = (2, 3, 4, 8)

    def __init__(
        self,
        dim: int,
        bits: int = 4,
        use_rotation: bool = True,
        group_size: int | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if bits not in self._SUPPORTED_BITS:
            raise ValueError(f"bits must be one of {self._SUPPORTED_BITS}, got {bits}")

        self.dim = dim
        self.bits = bits
        self.use_rotation = use_rotation
        self.group_size = group_size if group_size is not None else 128
        if self.group_size <= 0:
            raise ValueError(f"group_size must be positive, got {self.group_size}")
        self.device = torch.device(device) if device is not None else None

        self._rotation: torch.Tensor | None = None
        if self.use_rotation:
            self.build_rotation()

    # ------------------------------------------------------------------
    # Rotation
    # ------------------------------------------------------------------

    def build_rotation(self) -> torch.Tensor:
        """Generate a new random orthogonal rotation matrix of shape [dim, dim].

        Overwrites any previously stored rotation.  Can be called multiple
        times to re-initialize.

        Returns:
            The newly created rotation matrix.
        """
        dev = self.device or torch.device("cpu")
        R = torch.randn(self.dim, self.dim, device=dev)
        R = orthogonalize(R)
        self._rotation = R
        return R

    # ------------------------------------------------------------------
    # State export / import
    # ------------------------------------------------------------------

    def get_state(self) -> TurboQuantState:
        """Return a serialisable snapshot of the quantiser configuration."""
        return TurboQuantState(
            rotation=self._rotation.detach().clone() if self._rotation is not None else None,
            bits=self.bits,
            group_size=self.group_size,
            dim=self.dim,
            use_rotation=self.use_rotation,
        )

    @classmethod
    def from_state(cls, state: TurboQuantState) -> TurboQuantMSE:
        """Reconstruct a TurboQuantMSE from a previously saved state."""
        dev = state.rotation.device if state.rotation is not None else None
        tq = cls(
            dim=state.dim,
            bits=state.bits,
            use_rotation=state.use_rotation,
            group_size=state.group_size,
            device=dev,
        )
        if state.rotation is not None:
            tq._rotation = state.rotation
        return tq

    # ------------------------------------------------------------------
    # Quantize
    # ------------------------------------------------------------------

    def quantize(self, x: torch.Tensor) -> TurboQuantizedTensor:
        """Quantize a float tensor.

        Args:
            x: Input tensor of shape ``[T, D]`` or ``[B, T, D]`` where
               ``D == self.dim``.

        Returns:
            A :class:`TurboQuantizedTensor` holding all data needed for
            dequantization.

        Raises:
            ValueError: If the input is not 2-D/3-D or the last dimension
                does not equal ``self.dim``.
        """
        if x.dim() not in (2, 3):
            raise ValueError(
                f"Expected 2-D [T, D] or 3-D [B, T, D] input, "
                f"got {x.dim()}-D with shape {tuple(x.shape)}"
            )
        if x.shape[-1] != self.dim:
            raise ValueError(
                f"Expected last dim = {self.dim}, got {x.shape[-1]}"
            )

        shape_orig = x.shape
        x_2d = x.reshape(-1, self.dim).float()  # [N, D]

        # --- Step 1: optional rotation ---
        rotation: torch.Tensor | None = None
        if self.use_rotation and self._rotation is not None:
            rotation = self._rotation.to(x_2d.device, torch.float32)
            x_2d = x_2d @ rotation.T

        # --- Step 2: group-based affine quantization ---
        N, D = x_2d.shape
        gs = self.group_size
        max_val = 2 ** self.bits - 1

        # Pad last dim to a multiple of group_size
        pad_len = (gs - D % gs) % gs
        if pad_len > 0:
            x_pad = torch.nn.functional.pad(x_2d, (0, pad_len))
        else:
            x_pad = x_2d

        D_padded = x_pad.shape[-1]
        n_groups = D_padded // gs

        x_grouped = x_pad.reshape(N, n_groups, gs)

        # Extend range to include 0 so that zero_point is always valid
        x_min = x_grouped.amin(dim=-1, keepdim=True)
        x_max = x_grouped.amax(dim=-1, keepdim=True)
        x_min = torch.minimum(x_min, torch.zeros_like(x_min))
        x_max = torch.maximum(x_max, torch.zeros_like(x_max))

        scale = (x_max - x_min).clamp(min=1e-8) / max_val
        zero_point = torch.round(-x_min / scale).clamp(0, max_val)

        q_grouped = torch.round(x_grouped / scale + zero_point).clamp(0, max_val).to(torch.uint8)

        # Remove padding and squeeze group dim
        q_flat = q_grouped.reshape(N, D_padded)[:, :D]
        scale_out = scale.squeeze(-1)  # [N, n_groups]
        zero_point_out = zero_point.squeeze(-1).to(torch.uint8)  # [N, n_groups]

        if self.bits == 4:
            q_out = pack_uint4(q_flat)
        elif self.bits == 2:
            q_out = pack_uint2(q_flat)
        elif self.bits == 3:
            # 3-bit values are currently stored in 4-bit nibbles. This keeps
            # the implementation simple and makes estimate_num_bytes reflect
            # the actual stored tensor payload rather than ideal 3-bit packing.
            q_out = pack_uint4(q_flat)
        else:
            q_out = q_flat

        return TurboQuantizedTensor(
            q=q_out,
            scale=scale_out,
            zero_point=zero_point_out,
            rotation=rotation.detach().clone() if rotation is not None else None,
            shape_orig=shape_orig,
            bits=self.bits,
            group_size=gs,
        )

    # ------------------------------------------------------------------
    # Dequantize
    # ------------------------------------------------------------------

    def dequantize(self, qx: TurboQuantizedTensor) -> torch.Tensor:
        """Dequantize a :class:`TurboQuantizedTensor` back to float.

        Args:
            qx: A quantized tensor previously returned by :meth:`quantize`.

        Returns:
            Float tensor with the same shape as the original input.
        """
        D = qx.shape_orig[-1]
        N = qx.q.shape[0]

        if qx.bits == 4:
            q_unpacked = unpack_uint4(qx.q, D)
        elif qx.bits == 2:
            q_unpacked = unpack_uint2(qx.q, D)
        elif qx.bits == 3:
            # See quantize(): 3-bit values are stored in 4-bit nibbles.
            q_unpacked = unpack_uint4(qx.q, D)
        else:
            q_unpacked = qx.q

        gs = qx.group_size

        # Pad to match the group structure used during quantization
        pad_len = (gs - D % gs) % gs
        if pad_len > 0:
            q_pad = torch.nn.functional.pad(q_unpacked, (0, pad_len))
        else:
            q_pad = q_unpacked

        D_padded = q_pad.shape[-1]
        n_groups = D_padded // gs

        q_grouped = q_pad.reshape(N, n_groups, gs).float()
        scale = qx.scale.unsqueeze(-1)  # [N, n_groups, 1]
        zero_point = qx.zero_point.unsqueeze(-1).float()  # [N, n_groups, 1]

        x_grouped = (q_grouped - zero_point) * scale
        x_2d = x_grouped.reshape(N, D_padded)[:, :D]

        # Inverse rotation
        if qx.rotation is not None:
            rot = qx.rotation.to(x_2d.device, x_2d.dtype)
            x_2d = x_2d @ rot

        return x_2d.reshape(qx.shape_orig)

    # ------------------------------------------------------------------
    # Memory estimation
    # ------------------------------------------------------------------

    def estimate_num_bytes(self, qx: TurboQuantizedTensor) -> int:
        """Estimate the stored tensor payload size in bytes.

        The quantized integer payload is counted from the actual stored
        ``qx.q`` tensor (``nelement() * element_size()``), so 4-bit and
        2-bit use their packed representation, 8-bit uses one byte per
        element, and 3-bit reflects the current 4-bit nibble storage.
        Scale tensors are counted as float32, zero-points as uint8, and
        rotation as one shared float32 matrix when present.

        Args:
            qx: A :class:`TurboQuantizedTensor`.

        Returns:
            Estimated number of bytes.
        """
        N = qx.q.shape[0]
        D = qx.shape_orig[-1]
        gs = qx.group_size

        # q: bit-packed
        q_bytes = qx.q.nelement() * qx.q.element_size()

        # Per-group overhead
        pad_len = (gs - D % gs) % gs
        n_groups = (D + pad_len) // gs
        scale_bytes = N * n_groups * 4  # float32
        zp_bytes = N * n_groups * 1  # uint8

        # Rotation matrix (shared, counted once)
        rot_bytes = 0
        if qx.rotation is not None:
            rot_bytes = qx.rotation.nelement() * 4  # float32

        return q_bytes + scale_bytes + zp_bytes + rot_bytes


# ======================================================================
# TurboQuantProd — inner-product-optimized quantizer for Keys
# ======================================================================


@dataclass
class TurboQuantProdResult:
    """Container for a tensor quantized by TurboQuantProd.

    Two-stage residual quantization:
        1. MSE stage: standard group-based affine quantization (TurboQuantMSE).
        2. Residual stage: 1-bit sign quantization of the residual
           ``r = x - dequant(mse_stage)``, plus per-row residual norms.

    Attributes:
        mse: The MSE-stage quantized tensor.
        residual_sign: Sign of residual, shape [N, D], dtype bool.
        residual_norm: Per-row L2 norm of residual, shape [N], dtype float32.
        shape_orig: Original input shape.
        dim: Latent dimension.
    """

    mse: TurboQuantizedTensor
    residual_sign: torch.Tensor
    residual_norm: torch.Tensor
    shape_orig: tuple[int, ...]
    dim: int


class TurboQuantProd:
    """Inner-product-optimized TurboQuant for Keys.

    Two-stage pipeline:
        1. **MSE stage** — Reuses :class:`TurboQuantMSE` to produce a coarse
           reconstruction ``x_hat``.
        2. **Residual stage** — Computes ``r = x - x_hat`` and stores the
           per-element sign and per-row L2 norm.  This is a simplified,
           structure-preserving approximation of the QJL (Quantized Johnson-
           Lindenstrauss) sketch used in TurboQuant.

    The 1-bit residual preserves inner-product fidelity much better than
    pure MSE quantization because the dot product of the residual signs
    acts as a random-projection estimator of the true residual correlation.

    .. note::
        This is a **structural approximation** of full QJL.  A future
        version should replace the random sign-based residual with a
        proper JL sketch (e.g. sparse random projection + 1-bit hashing).

    Args:
        dim: Last dimension of tensors to be quantized.
        bits: Bits per element for the MSE stage.
        use_rotation: Whether to apply random orthogonal rotation.
        group_size: Quantization group size for the MSE stage.
        device: Device for the rotation matrix.

    Raises:
        ValueError: If dim <= 0 or other parameters are invalid.
    """

    def __init__(
        self,
        dim: int,
        bits: int = 4,
        use_rotation: bool = True,
        group_size: int | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        self.dim = dim
        self.bits = bits
        self.use_rotation = use_rotation
        self.group_size = group_size if group_size is not None else 128

        self._mse_quantizer = TurboQuantMSE(
            dim=dim,
            bits=bits,
            use_rotation=use_rotation,
            group_size=self.group_size,
            device=device,
        )

    def build_rotation(self) -> torch.Tensor:
        """Rebuild the random orthogonal rotation (delegated to MSE stage)."""
        return self._mse_quantizer.build_rotation()

    @property
    def _rotation(self) -> torch.Tensor | None:
        return self._mse_quantizer._rotation

    def quantize(self, x: torch.Tensor) -> TurboQuantProdResult:
        """Two-stage quantization: MSE + 1-bit residual.

        Args:
            x: Input tensor of shape ``[T, D]`` or ``[B, T, D]`` where
               ``D == self.dim``.

        Returns:
            A :class:`TurboQuantProdResult`.

        Raises:
            ValueError: If input shape is invalid.
        """
        if x.dim() not in (2, 3):
            raise ValueError(
                f"Expected 2-D or 3-D input, got {x.dim()}-D shape {tuple(x.shape)}"
            )
        if x.shape[-1] != self.dim:
            raise ValueError(
                f"Expected last dim = {self.dim}, got {x.shape[-1]}"
            )

        shape_orig = x.shape
        x_flat = x.reshape(-1, self.dim).float()

        # Stage 1: MSE quantization
        mse_result = self._mse_quantizer.quantize(x_flat.reshape(shape_orig))
        x_hat = self._mse_quantizer.dequantize(mse_result)
        x_hat_flat = x_hat.reshape(-1, self.dim)

        # Stage 2: 1-bit residual
        residual = x_flat - x_hat_flat
        residual_sign = residual >= 0  # bool [N, D]
        residual_sign_packed = pack_bool(residual_sign)
        residual_norm = residual.norm(dim=-1)  # [N]

        return TurboQuantProdResult(
            mse=mse_result,
            residual_sign=residual_sign_packed,
            residual_norm=residual_norm,
            shape_orig=shape_orig,
            dim=self.dim,
        )

    def dequantize(self, qx: TurboQuantProdResult) -> torch.Tensor:
        """Full dequantization: MSE + scaled sign residual.

        The residual is reconstructed as:
            r_hat = residual_norm / sqrt(D) * (2 * sign - 1)

        This preserves the expected L2 norm of each row's residual.

        Args:
            qx: A :class:`TurboQuantProdResult` from :meth:`quantize`.

        Returns:
            Float tensor with the original input shape.
        """
        x_hat = self._mse_quantizer.dequantize(qx.mse)
        x_hat_flat = x_hat.reshape(-1, qx.dim)

        sign_unpacked = unpack_bool(qx.residual_sign, qx.dim)
        sign_float = sign_unpacked.float() * 2.0 - 1.0  # {-1, +1}
        scale = (qx.residual_norm / (qx.dim ** 0.5)).unsqueeze(-1)  # [N, 1]
        r_hat = scale * sign_float

        x_flat = x_hat_flat + r_hat
        return x_flat.reshape(qx.shape_orig)

    def approx_inner_product(
        self,
        q: torch.Tensor,
        qx: TurboQuantProdResult,
    ) -> torch.Tensor:
        """Approximate inner product(s) without full dequantization.

        Computes ``q @ x^T`` where ``x`` is the dual-stage quantized
        tensor.  The MSE part uses the quantized representation directly,
        and the residual part uses the 1-bit signs for a fast estimate.

        Args:
            q: Query tensor of shape ``[D]``, ``[1, D]``, ``[Q, D]``,
               or ``[B, Q, D]``.
            qx: Quantized key tensor from :meth:`quantize`.

        Returns:
            Approximate inner product(s).  Shape depends on ``q``:
                - ``[D]`` or ``[1, D]`` → ``[N]``
                - ``[Q, D]`` → ``[Q, N]``
                - ``[B, Q, D]`` → ``[B, Q, N]``
        """
        # Part 1: MSE contribution — dequantize MSE part and matmul
        x_hat = self._mse_quantizer.dequantize(qx.mse)  # shape_orig
        x_hat_2d = x_hat.reshape(-1, qx.dim)  # [N, D]

        q_2d = q.reshape(-1, qx.dim).float()  # [Q, D]
        mse_ip = q_2d @ x_hat_2d.T  # [Q, N]

        # Part 2: Residual contribution — 1-bit sign dot product
        sign_unpacked = unpack_bool(qx.residual_sign, qx.dim)
        sign_float = sign_unpacked.float() * 2.0 - 1.0  # [N, D]
        scale = qx.residual_norm / (qx.dim ** 0.5)  # [N]
        sign_ip = q_2d @ sign_float.T * scale.unsqueeze(0)  # [Q, N]

        result = mse_ip + sign_ip

        # Reshape to match query structure
        if q.dim() <= 1 or (q.dim() == 2 and q.shape[0] == 1):
            return result.squeeze(0)
        return result

    def estimate_num_bytes(self, qx: TurboQuantProdResult) -> int:
        """Estimate storage size for a TurboQuantProdResult.

        Includes MSE stage bytes plus:
            - residual_sign: 1 bit per element (ideal packing)
            - residual_norm: 4 bytes per row (float32)

        Args:
            qx: A :class:`TurboQuantProdResult`.

        Returns:
            Estimated number of bytes.
        """
        mse_bytes = self._mse_quantizer.estimate_num_bytes(qx.mse)
        sign_bytes = qx.residual_sign.nelement() * qx.residual_sign.element_size()
        N = qx.residual_norm.shape[0]
        norm_bytes = N * 4  # float32
        return mse_bytes + sign_bytes + norm_bytes
