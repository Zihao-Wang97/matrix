import torch


_INT4_MAX = 7
_INT4_MIN = -8
_MASK_LO = 0x0F
_MASK_HI = 0xF0


def _sign_extend_4bit(x: torch.Tensor) -> torch.Tensor:
    x = x.to(torch.int16)
    x = torch.where(x >= 8, x - 16, x)
    return x.to(torch.int8)


def pack_int4(tensor: torch.Tensor) -> torch.Tensor:
    t = tensor.clamp(_INT4_MIN, _INT4_MAX).to(torch.int8)
    flat = t.flatten()
    if flat.numel() % 2 != 0:
        flat = torch.cat([flat, flat.new_zeros(1)])
    lo = (flat[0::2].to(torch.uint8) & _MASK_LO)
    hi = ((flat[1::2].to(torch.uint8) & _MASK_LO) << 4)
    packed = lo | hi
    return packed.view(torch.uint8)


def unpack_int4(packed: torch.Tensor, original_numel: int) -> torch.Tensor:
    flat = packed.flatten().to(torch.uint8)
    lo = flat & _MASK_LO
    hi = (flat >> 4) & _MASK_LO
    lo = _sign_extend_4bit(lo)
    hi = _sign_extend_4bit(hi)
    unpacked = torch.stack([lo, hi], dim=-1).flatten()
    return unpacked[:original_numel]
