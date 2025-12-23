
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
    return ssim_map

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

# --------------------------------------------------------------------------------------------------------------
# Per-sample MS-SSIM
# --------------------------------------------------------------------------------------------------------------

def gaussian_window_1d(window_size, sigma, channels, device):
    coords = torch.arange(window_size, device=device).float() - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window = g.view(1, 1, -1)
    window = window.expand(channels, 1, window_size)
    return window


def ssim_1d_per_channel(x, y, window, C1, C2):
    # x, y: [B, C, L]
    mu_x = torch.nn.functional.conv1d(x, window, padding=window.size(-1) // 2, groups=x.size(1))
    mu_y = torch.nn.functional.conv1d(y, window, padding=window.size(-1) // 2, groups=x.size(1))

    mu_x2 = mu_x.pow(2)
    mu_y2 = mu_y.pow(2)
    mu_xy = mu_x * mu_y

    sigma_x2 = torch.nn.functional.conv1d(x * x, window, padding=window.size(-1) // 2, groups=x.size(1)) - mu_x2
    sigma_y2 = torch.nn.functional.conv1d(y * y, window, padding=window.size(-1) // 2, groups=x.size(1)) - mu_y2
    sigma_xy = torch.nn.functional.conv1d(x * y, window, padding=window.size(-1) // 2, groups=x.size(1)) - mu_xy

    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2))

    cs_map = (2 * sigma_xy + C2) / (sigma_x2 + sigma_y2 + C2)

    return ssim_map, cs_map


class MS_SSIM_1D_Loss(torch.nn.Module):
    def __init__(
        self,
        alpha=0.5, # MS-SSIM contribution
        beta=0.3, # L1 contribution
        gamma=0.2, # gradient loss contribution
        window_size=11,
        sigma=1.5,
        data_range=1.0,
        weights=(0.0448, 0.2856, 0.3001, 0.2363, 0.1333),        
        K1=0.01,
        K2=0.03,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.window_size = window_size
        self.sigma = sigma
        self.data_range = data_range
        self.weights = weights
        self.K1 = K1
        self.K2 = K2

    def forward(self, x, y):
        """
        x, y: [B, C, L]
        returns: [B]  (per-sample MS-SSIM loss)
        """
        B, C, L = x.shape
        device = x.device

        window = gaussian_window_1d(
            self.window_size, self.sigma, C, device
        )

        C1 = (self.K1 * self.data_range) ** 2
        C2 = (self.K2 * self.data_range) ** 2

        msssim = torch.ones(B, device=device)

        for i, weight in enumerate(self.weights):
            ssim_map, cs_map = ssim_1d_per_channel(x, y, window, C1, C2)

            # Average over channel + length → per-sample
            ssim_val = ssim_map.mean(dim=[1, 2])
            cs_val = cs_map.mean(dim=[1, 2])

            if i == len(self.weights) - 1:
                msssim = msssim * (ssim_val ** weight)
            else:
                msssim = msssim * (cs_val ** weight)
                x = F.avg_pool1d(x, kernel_size=2, stride=2)
                y = F.avg_pool1d(y, kernel_size=2, stride=2)

        grad_x = x[:, :, 1:] - x[:, :, :-1]
        grad_y = y[:, :, 1:] - y[:, :, :-1]
        grad_loss = torch.abs(grad_x - grad_y).mean(dim=[1,2])

        return ( 1.0 - msssim ) * self.alpha + torch.nn.functional.l1_loss(x, y) * self.beta + grad_loss * self.gamma
