import torch

_SUFFIXES = ["B", "KB", "MB", "GB"]


def tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.nelement() * tensor.element_size()


def format_nbytes(nbytes: int) -> str:
    if nbytes == 0:
        return "0 B"
    idx = 0
    val = float(nbytes)
    while val >= 1024.0 and idx < len(_SUFFIXES) - 1:
        val /= 1024.0
        idx += 1
    if idx == 0:
        return f"{int(val)} {_SUFFIXES[idx]}"
    return f"{val:.2f} {_SUFFIXES[idx]}"
