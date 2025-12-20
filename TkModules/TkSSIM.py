
import torch
import torch.nn.functional as F


def gaussian_kernel_1d(win_size=11, sigma=1.5, channels=1, device='cpu', dtype=torch.float32):
    """Returns a 1D Gaussian convolution kernel for depthwise conv."""
    half = win_size // 2
    x = torch.arange(-half, half + 1, device=device, dtype=dtype)
    g = torch.exp(-0.5 * (x / sigma)**2)
    g = g / g.sum()

    # Depthwise convolution kernel shape: (channels, 1, win_size)
    g = g.view(1, 1, win_size).repeat(channels, 1, 1)
    return g

def ssim_1d_gaussian(x, y, win_size=11, sigma=1.5, C1=0.01**2, C2=0.03**2):
    """
    Gaussian-window SSIM for 1D signals.
    x, y: (B, C, L)
    """
    B, C, L = x.shape
    device, dtype = x.device, x.dtype

    kernel = gaussian_kernel_1d(win_size, sigma, C, device=device, dtype=dtype)

    # Reflect padding like original SSIM
    pad = win_size // 2

    # Means
    mu_x = F.conv1d(x, kernel, padding=pad, groups=C)
    mu_y = F.conv1d(y, kernel, padding=pad, groups=C)

    # Variances & covariance
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y

    sigma_x2 = F.conv1d(x * x, kernel, padding=pad, groups=C) - mu_x2
    sigma_y2 = F.conv1d(y * y, kernel, padding=pad, groups=C) - mu_y2
    sigma_xy = F.conv1d(x * y, kernel, padding=pad, groups=C) - mu_x * mu_y

    # SSIM formula
    num = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    den = (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)

    ssim_map = num / (den + 1e-12)
    return ssim_map.mean()

# --------------------------------------------------------------------------------------------------------------
# Differentiable SSIM loss for VAE: lower = better.
# Loss = (1 - SSIM) / 2 (normalized to roughly 0..1)
# --------------------------------------------------------------------------------------------------------------

def ssim_1d_gaussian_loss(x, y, win_size=11, C1=0.01**2, C2=0.03**2):
    ssim_val = ssim_1d_gaussian(x, y, win_size, C1, C2)
    loss = (1 - ssim_val) * 0.5
    return loss

# --------------------------------------------------------------------------------------------------------------
# Hybrid SSIM + L1 loss for 1D VAE.
# Alpha: weight on SSIM (0.7�0.9 recommended)
# Win_size: SSIM window size (11 is good for most signals)
# --------------------------------------------------------------------------------------------------------------

def hybrid_ssim_1d_gaussian_l1_loss(x, y, alpha=0.85, win_size=11):
    # Fast SSIM loss defined earlier
    ssim_loss = ssim_1d_gaussian_loss(x, y, win_size=win_size)

    # Standard L1 loss
    l1_loss = torch.nn.functional.l1_loss(x, y)

    # hybrid loss
    return alpha * ssim_loss + (1 - alpha) * l1_loss

# --------------------------------------------------------------------------------------------------------------
# Multi-scale SSIM for 1D signals.
# x, y: (B, C, L)
# levels: number of scales (4 recommended for VAE)
# --------------------------------------------------------------------------------------------------------------

def ms_ssim_1d_gaussian(x, y, win_size=11, levels=5):
    # weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]  # standard MS-SSIM weights (sum=1)
    weights = [0.4375, 0.375, 0.25, 0.125, 0.0625]  # LOB data weights

    ssims = []
    cur_x, cur_y = x, y

    for i in range(levels):
        ssim_i = ssim_1d_gaussian(cur_x, cur_y, win_size=win_size)
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

def ms_ssim_1d_gaussian_loss(x, y, win_size=11, levels=5):
    ms = ms_ssim_1d_gaussian(x, y, win_size=win_size, levels=levels)
    return (1 - ms) * 0.5

# --------------------------------------------------------------------------------------------------------------
# Hybrid MS-SSIM + L1 loss for 1D VAE.
# Alpha: weight for MS-SSIM (0.8-0.95 recommended)
# --------------------------------------------------------------------------------------------------------------

def hybrid_ms_ssim_1d_gaussian_l1_loss(x, y, alpha=0.9, win_size=11, levels=5):
    ms_ssim_loss = ms_ssim_1d_gaussian_loss(x, y, win_size=win_size, levels=levels)
    l1_loss = torch.nn.functional.l1_loss(x, y)

    return alpha * ms_ssim_loss + (1 - alpha) * l1_loss

# --------------------------------------------------------------------------------------------------------------
# Hybrid limit orderbook VAE loss: MS-SSIM + L1 loss + Huber + Total variation
# Alpha: weight for MS-SSIM (0.8-0.95 recommended)
# Beta: weight for L1
# Gamma: weight for Huber
# Delta: weight for TV
# --------------------------------------------------------------------------------------------------------------

def hybrid_lob_loss(x, y, alpha=0.9, beta=0.1, gamma=0.05, delta=0.001, win_size=11, levels=5):
    ms_ssim_loss = ms_ssim_1d_gaussian_loss(x, y, win_size=win_size, levels=levels)
    l1_loss = torch.nn.functional.l1_loss(x, y)
    huber_loss = torch.nn.functional.smooth_l1_loss(x, y)
    tv_loss = torch.mean(torch.abs(x[:, :, 1:] - x[:, :, :-1]))

    return alpha * ms_ssim_loss + beta * l1_loss + gamma * huber_loss + delta * tv_loss

# --------------------------------------------------------------------------------------------------------------
# Hybrid limit orderbook VAE loss: 2x MS-SSIM + L1 loss + Huber + Total variation
# Alphas: weight for MS-SSIM (0.8-0.95 recommended)
# Beta: weight for L1
# Gamma: weight for Huber
# Delta: weight for TV
# --------------------------------------------------------------------------------------------------------------

def hybrid_lob_multi_loss(x, y, alpha_1=0.5, alpha_2=0.4, beta=0.1, gamma=0.05, delta=0.001, win_size_1=11, levels_1=4, win_size_2=5, levels_2=3):
    ms_ssim_loss_1 = ms_ssim_1d_gaussian_loss(x, y, win_size=win_size_1, levels=levels_1)
    ms_ssim_loss_2 = ms_ssim_1d_gaussian_loss(x, y, win_size=win_size_2, levels=levels_2)
    l1_loss = torch.nn.functional.l1_loss(x, y)
    huber_loss = torch.nn.functional.smooth_l1_loss(x, y)
    tv_loss = torch.mean(torch.abs(x[:, :, 1:] - x[:, :, :-1]))

    return alpha_1 * ms_ssim_loss_1 + alpha_2 * ms_ssim_loss_2 + beta * l1_loss + gamma * huber_loss + delta * tv_loss
