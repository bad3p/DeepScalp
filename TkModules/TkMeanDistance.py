
import torch

def mean_distance(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:

    """
    Mean distance where the middle index (D/2) is treated as zero.

    Args:
        p (torch.Tensor): shape (N, D)
        q (torch.Tensor): shape (N, D)

    Returns:
        torch.Tensor: shape (N,), absolute difference of centered means
    """

    assert p.shape == q.shape, "Distributions must have the same shape"
    assert p.dim() == 2, "Expected shape (N, D)"

    N, D = p.shape

    # Create centered support: e.g.
    # D=5 → [-2, -1, 0, 1, 2]
    # D=6 → [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]
    center = (D - 1) / 2
    support = torch.arange(D, device=p.device, dtype=p.dtype) - center  # (D,)

    # Compute means
    mean_p = (p * support).sum(dim=1)
    mean_q = (q * support).sum(dim=1)

    return torch.abs(mean_p - mean_q)