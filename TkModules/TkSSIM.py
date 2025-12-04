
import torch

# --------------------------------------------------------------------------------------------------------------
# Fast SSIM for 1D signals using box window via cumulative sum
# x, y: (B, C, L)
# Returns scalar SSIM value (higher = better).
# --------------------------------------------------------------------------------------------------------------

def fast_ssim_1d(x, y, win_size=11, C1=0.01**2, C2=0.03**2):

    B, C, L = x.shape
    pad = win_size // 2

    # Reflect padding for border stability
    x_pad = torch.nn.functional.pad(x, (pad, pad), mode='reflect')
    y_pad = torch.nn.functional.pad(y, (pad, pad), mode='reflect')

    def box_filter(t):
        # t: (B, C, L+2*pad)
        cs = torch.cumsum(t, dim=2)
        left = cs[:, :, :-win_size]
        right = cs[:, :, win_size:]
        return (right - left) / win_size  # local mean

    # Means
    mu_x = box_filter(x_pad)
    mu_y = box_filter(y_pad)

    # Variances & covariance
    sigma_x2 = box_filter(x_pad * x_pad) - mu_x * mu_x
    sigma_y2 = box_filter(y_pad * y_pad) - mu_y * mu_y
    sigma_xy = box_filter(x_pad * y_pad) - mu_x * mu_y

    # SSIM components
    num = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    den = (mu_x * mu_x + mu_y * mu_y + C1) * (sigma_x2 + sigma_y2 + C2)

    ssim_map = num / den
    return ssim_map.mean()  # global SSIM

# --------------------------------------------------------------------------------------------------------------
# Differentiable SSIM loss for VAE: lower = better.
# Loss = (1 - SSIM) / 2 (normalized to roughly 0..1)
# --------------------------------------------------------------------------------------------------------------

def fast_ssim_loss_1d(x, y, win_size=11, C1=0.01**2, C2=0.03**2):
    ssim_val = fast_ssim_1d(x, y, win_size, C1, C2)
    loss = (1 - ssim_val) * 0.5
    return loss

# --------------------------------------------------------------------------------------------------------------
# Hybrid SSIM + L1 loss for 1D VAE.
# Alpha: weight on SSIM (0.7�0.9 recommended)
# Win_size: SSIM window size (11 is good for most signals)
# --------------------------------------------------------------------------------------------------------------

def hybrid_ssim_l1_loss_1d(x, y, alpha=0.85, win_size=11):
    # Fast SSIM loss defined earlier
    ssim_loss = fast_ssim_loss_1d(x, y, win_size=win_size)

    # Standard L1 loss
    l1_loss = torch.nn.functional.l1_loss(x, y)

    # hybrid loss
    return alpha * ssim_loss + (1 - alpha) * l1_loss

# --------------------------------------------------------------------------------------------------------------
# Multi-scale SSIM for 1D signals.
# x, y: (B, C, L)
# levels: number of scales (4 recommended for VAE)
# --------------------------------------------------------------------------------------------------------------

def fast_ms_ssim_1d(x, y, win_size=11, levels=4):
    weights = [0.44, 0.27, 0.17, 0.12]  # standard MS-SSIM weights (sum=1)

    ssims = []
    cur_x, cur_y = x, y

    for i in range(levels):
        ssim_i = fast_ssim_1d(cur_x, cur_y, win_size=win_size)
        ssims.append(ssim_i)

        # downsample for next scale
        if i < levels - 1:
            cur_x = torch.nn.functional.avg_pool1d(cur_x, kernel_size=2, stride=2, padding=0)
            cur_y = torch.nn.functional.avg_pool1d(cur_y, kernel_size=2, stride=2, padding=0)

    # weighted geometric combination
    ms_ssim = torch.prod(torch.stack([ssims[i] ** weights[i] for i in range(levels)]))
    return ms_ssim

# --------------------------------------------------------------------------------------------------------------
# Differentiable MS-SSIM loss for VAE: lower = better.
# Loss = (1 - MS-SSIM) / 2 (normalized to roughly 0..1)
# --------------------------------------------------------------------------------------------------------------

def fast_ms_ssim_loss_1d(x, y, win_size=11, levels=4):
    ms = fast_ms_ssim_1d(x, y, win_size=win_size, levels=levels)
    return (1 - ms) * 0.5

# --------------------------------------------------------------------------------------------------------------
# Hybrid MS-SSIM + L1 loss for 1D VAE.
# Alpha: weight for MS-SSIM (0.8-0.95 recommended)
# --------------------------------------------------------------------------------------------------------------

def hybrid_ms_ssim_l1_loss_1d(x, y, alpha=0.9, win_size=11, levels=4):
    ms_ssim_loss = fast_ms_ssim_loss_1d(x, y, win_size=win_size, levels=levels)
    l1_loss = torch.nn.functional.l1_loss(x, y)

    return alpha * ms_ssim_loss + (1 - alpha) * l1_loss