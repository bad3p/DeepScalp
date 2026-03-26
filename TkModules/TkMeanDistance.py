
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


def tail_mean_distance(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Compare distributions using distances between tail means:
    (mean below mean) and (mean above mean), summed.

    Args:
        p, q: shape (N, D)
        eps: numerical stability

    Returns:
        Tensor of shape (N,)
    """
    assert p.shape == q.shape, "Shapes must match"
    assert p.dim() == 2, "Expected (N, D)"

    N, D = p.shape

    # Centered support
    center = (D - 1) / 2
    support = torch.arange(D, device=p.device, dtype=p.dtype) - center  # (D,)
    support = support.unsqueeze(0)  # (1, D)

    # ---- Means ----
    mean_p = (p * support).sum(dim=1, keepdim=True)  # (N, 1)
    mean_q = (q * support).sum(dim=1, keepdim=True)  # (N, 1)

    # ---- Masks ----
    left_mask_p  = (support < mean_p).to(p.dtype)
    right_mask_p = (support > mean_p).to(p.dtype)

    left_mask_q  = (support < mean_q).to(q.dtype)
    right_mask_q = (support > mean_q).to(q.dtype)

    # ---- Tail probabilities ----
    p_left_mass  = (p * left_mask_p).sum(dim=1, keepdim=True) + eps
    p_right_mass = (p * right_mask_p).sum(dim=1, keepdim=True) + eps

    q_left_mass  = (q * left_mask_q).sum(dim=1, keepdim=True) + eps
    q_right_mass = (q * right_mask_q).sum(dim=1, keepdim=True) + eps

    # ---- Tail means ----
    p_left_mean = (p * support * left_mask_p).sum(dim=1) / p_left_mass.squeeze(1)
    p_right_mean = (p * support * right_mask_p).sum(dim=1) / p_right_mass.squeeze(1)

    q_left_mean = (q * support * left_mask_q).sum(dim=1) / q_left_mass.squeeze(1)
    q_right_mean = (q * support * right_mask_q).sum(dim=1) / q_right_mass.squeeze(1)

    # ---- Distance ----
    left_diff = torch.abs(p_left_mean - q_left_mean)
    right_diff = torch.abs(p_right_mean - q_right_mean)

    return left_diff + right_diff  # (N,)

def smooth_tail_mean_distance(p: torch.Tensor, q: torch.Tensor, temperature: float = 10.0, eps: float = 1e-8, shared_split: bool = True) -> torch.Tensor:
    """
    Differentiable version of tail mean distance using soft masks.

    Args:
        p, q: shape (N, D)
        temperature: controls sharpness of tail split (higher = sharper)
        eps: numerical stability
        shared_split: if True, use midpoint of means for both distributions

    Returns:
        Tensor of shape (N,)
    """
    assert p.shape == q.shape
    assert p.dim() == 2

    N, D = p.shape

    # ---- Centered support ----
    center = (D - 1) / 2
    support = torch.arange(D, device=p.device, dtype=p.dtype) - center
    support = support.unsqueeze(0)  # (1, D)

    # ---- Means ----
    mean_p = (p * support).sum(dim=1, keepdim=True)
    mean_q = (q * support).sum(dim=1, keepdim=True)

    if shared_split:
        split = 0.5 * (mean_p + mean_q)
    else:
        split = mean_p
    split_q = split if shared_split else mean_q

    # ---- Smooth masks (sigmoid) ----
    # right tail ≈ sigmoid(temp * (x - split))
    right_mask_p = torch.sigmoid(temperature * (support - split))
    left_mask_p  = 1.0 - right_mask_p

    right_mask_q = torch.sigmoid(temperature * (support - split_q))
    left_mask_q  = 1.0 - right_mask_q

    # ---- Tail masses ----
    p_left_mass  = (p * left_mask_p).sum(dim=1) + eps
    p_right_mass = (p * right_mask_p).sum(dim=1) + eps

    q_left_mass  = (q * left_mask_q).sum(dim=1) + eps
    q_right_mass = (q * right_mask_q).sum(dim=1) + eps

    # ---- Tail means ----
    p_left_mean = (p * support * left_mask_p).sum(dim=1) / p_left_mass
    p_right_mean = (p * support * right_mask_p).sum(dim=1) / p_right_mass

    q_left_mean = (q * support * left_mask_q).sum(dim=1) / q_left_mass
    q_right_mean = (q * support * right_mask_q).sum(dim=1) / q_right_mass

    # ---- Final distance ----
    left_diff = torch.abs(p_left_mean - q_left_mean)
    right_diff = torch.abs(p_right_mean - q_right_mean)

    return left_diff + right_diff