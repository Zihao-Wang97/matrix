from __future__ import annotations

from dataclasses import dataclass

import torch


_INT4_MAX = 7
_INT4_MIN = -8
_INT8_MAX = 127
_INT8_MIN = -128


@dataclass
class KQuantizeResult:
    q: torch.Tensor
    scale: torch.Tensor
    rotation: torch.Tensor | None


class KQuantizer:
    def __init__(self, group_size: int = 128, use_rotation: bool = False):
        self.group_size = group_size
        self.use_rotation = use_rotation
        self._rotation: torch.Tensor | None = None

    def _make_rotation(self, dim: int, device: torch.device) -> torch.Tensor:
        from hawp_laq.utils.math_utils import orthogonalize
        R = torch.randn(dim, dim, device=device)
        return orthogonalize(R)

    def quantize(self, x: torch.Tensor) -> KQuantizeResult:
        orig_shape = x.shape
        d = x.shape[-1]

        rotation = None
        if self.use_rotation:
            rotation = self._make_rotation(d, x.device)
            x = x @ rotation.T

        x_grouped, pad_len = _pad_to_group(x, self.group_size)
        scale = x_grouped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        q_float = x_grouped / scale * _INT4_MAX
        q_int = q_float.round().clamp(_INT4_MIN, _INT4_MAX).to(torch.int8)

        if pad_len > 0:
            q_int = q_int[..., :self.group_size]

        scale = scale.squeeze(-1)
        if pad_len > 0:
            scale = scale[:orig_shape[-1]]

        return KQuantizeResult(q=q_int, scale=scale, rotation=rotation)

    @staticmethod
    def dequantize(result: KQuantizeResult) -> torch.Tensor:
        scale = result.scale
        if scale.dim() == 1:
            x = result.q.float() * (scale / _INT4_MAX).unsqueeze(-1)
        else:
            x = result.q.float() * (scale / _INT4_MAX)
        if result.rotation is not None:
            x = x @ result.rotation
        return x


@dataclass
class VQuantizeResult:
    q: torch.Tensor
    scale: torch.Tensor
    zero_point: torch.Tensor
    residual: torch.Tensor | None


class VQuantizer:
    def __init__(self, group_size: int = 128, outlier_threshold: float | None = None):
        self.group_size = group_size
        self.outlier_threshold = outlier_threshold

    def quantize(self, x: torch.Tensor) -> VQuantizeResult:
        orig_shape = x.shape
        d = x.shape[-1]

        residual = None
        if self.outlier_threshold is not None:
            mask = x.abs() > self.outlier_threshold
            residual = torch.zeros_like(x)
            residual[mask] = x[mask]
            x = x.clone()
            x[mask] = 0.0

        x_grouped, pad_len = _pad_to_group(x, self.group_size)
        x_min = x_grouped.amin(dim=-1, keepdim=True)
        x_max = x_grouped.amax(dim=-1, keepdim=True)
        scale = (x_max - x_min).clamp(min=1e-8) / (_INT8_MAX - _INT8_MIN)
        zero_point = (_INT8_MIN - x_min / scale).round().clamp(_INT8_MIN, _INT8_MAX).to(torch.int8)

        q_float = x_grouped / scale + zero_point.float()
        q_int = q_float.round().clamp(_INT8_MIN, _INT8_MAX).to(torch.int8)

        if pad_len > 0:
            q_int = q_int[..., :self.group_size]

        scale = scale.squeeze(-1)
        zero_point = zero_point.squeeze(-1)
        if pad_len > 0:
            scale = scale[:orig_shape[-1]]
            zero_point = zero_point[:orig_shape[-1]]

        return VQuantizeResult(q=q_int, scale=scale, zero_point=zero_point, residual=residual)

    @staticmethod
    def dequantize(result: VQuantizeResult) -> torch.Tensor:
        if result.scale.dim() == 1:
            x = (result.q.float() - result.zero_point.float().unsqueeze(-1)) * result.scale.unsqueeze(-1)
        else:
            x = (result.q.float() - result.zero_point.float()) * result.scale
        if result.residual is not None:
            x = x + result.residual
        return x


def _pad_to_group(x: torch.Tensor, group_size: int) -> tuple[torch.Tensor, int]:
    d = x.shape[-1]
    pad_len = (group_size - d % group_size) % group_size
    if pad_len == 0:
        return x, 0
    x_pad = torch.nn.functional.pad(x, (0, pad_len))
    return x_pad, pad_len
