import torch


_INT4_MAX = 7
_INT4_MIN = -8
_MASK_LO = 0x0F
_MASK_HI = 0xF0


def _sign_extend_4bit(x: torch.Tensor) -> torch.Tensor:
    x = x.to(torch.int16)
    x = torch.where(x >= 8, x - 16, x)
    return x.to(torch.int8)


def pack_uint4(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    N, D = tensor.shape
    flat = tensor.clamp(0, 15).to(torch.uint8)
    if D % 2 != 0:
        flat = torch.nn.functional.pad(flat, (0, 1))
    lo = flat[:, 0::2]
    hi = flat[:, 1::2] << 4
    packed = (lo | hi).to(torch.uint8)
    return packed


def unpack_uint4(packed: torch.Tensor, original_D: int) -> torch.Tensor:
    if packed.dim() == 1:
        packed = packed.unsqueeze(0)
    lo = (packed & 0x0F).to(torch.uint8)
    hi = ((packed >> 4) & 0x0F).to(torch.uint8)
    unpacked = torch.stack([lo, hi], dim=-1).reshape(packed.shape[0], -1)
    return unpacked[:, :original_D].contiguous()


def pack_uint2(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    N, D = tensor.shape
    flat = tensor.clamp(0, 3).to(torch.uint8)
    pad_len = (4 - D % 4) % 4
    if pad_len > 0:
        flat = torch.nn.functional.pad(flat, (0, pad_len))
    b0 = flat[:, 0::4]
    b1 = flat[:, 1::4]
    b2 = flat[:, 2::4]
    b3 = flat[:, 3::4]
    packed = (b0 | (b1 << 2) | (b2 << 4) | (b3 << 6)).to(torch.uint8)
    return packed


def unpack_uint2(packed: torch.Tensor, original_D: int) -> torch.Tensor:
    if packed.dim() == 1:
        packed = packed.unsqueeze(0)
    b0 = (packed & 0x03).to(torch.uint8)
    b1 = ((packed >> 2) & 0x03).to(torch.uint8)
    b2 = ((packed >> 4) & 0x03).to(torch.uint8)
    b3 = ((packed >> 6) & 0x03).to(torch.uint8)
    unpacked = torch.stack([b0, b1, b2, b3], dim=-1).reshape(packed.shape[0], -1)
    return unpacked[:, :original_D].contiguous()


def pack_bool(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    N, D = tensor.shape
    flat = tensor.to(torch.uint8)
    pad_len = (8 - D % 8) % 8
    if pad_len > 0:
        flat = torch.nn.functional.pad(flat, (0, pad_len))
    result = torch.zeros(N, flat.shape[1] // 8, dtype=torch.uint8, device=tensor.device)
    for bit in range(8):
        result |= (flat[:, bit::8] << bit).to(torch.uint8)
    return result


def unpack_bool(packed: torch.Tensor, original_D: int) -> torch.Tensor:
    if packed.dim() == 1:
        packed = packed.unsqueeze(0)
    chunks = []
    for bit in range(8):
        chunks.append(((packed >> bit) & 1).to(torch.bool))
    unpacked = torch.stack(chunks, dim=-1).reshape(packed.shape[0], -1)
    return unpacked[:, :original_D].contiguous()


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
