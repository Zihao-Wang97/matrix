import torch


def orthogonalize(weight: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    u, _, vh = torch.linalg.svd(weight.float(), full_matrices=False)
    return (u @ vh).to(weight.dtype)


def topk_recall(
    scores: torch.Tensor,
    targets: torch.Tensor,
    k: int,
) -> float:
    if scores.dim() == 1:
        scores = scores.unsqueeze(0)
        targets = targets.unsqueeze(0)
    _, topk_idx = scores.topk(k, dim=-1)
    hits = torch.zeros_like(targets, dtype=torch.bool)
    hits.scatter_(-1, topk_idx, True)
    return (hits * targets).sum().item() / targets.sum().clamp(min=1).item()


def pairwise_hinge_ranking_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    margin: float = 1.0,
) -> torch.Tensor:
    d_pos = (anchor - positive).pow(2).sum(dim=-1)
    d_neg = (anchor - negative).pow(2).sum(dim=-1)
    return torch.clamp(margin + d_pos - d_neg, min=0.0).mean()
