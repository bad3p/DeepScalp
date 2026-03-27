import torch
import torch.nn as nn
import math

# --------------------------------------------------------------------------------------------------
# Optimized S4D (Diagonal State Space Model)
# - Replaces O(L) sequential loop with O(L log L) parallel FFT Convolution
# - Vectorizes all input/output projections
# --------------------------------------------------------------------------------------------------

class TkStateSpaceModule(torch.nn.Module):

    def __init__(self, d_input, d_state, d_output):
        super().__init__()
        self.d_input = d_input
        self.d_state = d_state
        self.d_output = d_output

        # 1. Diagonal complex spectrum
        self.log_lambda_real = nn.Parameter(torch.randn(d_state))
        self.lambda_imag = nn.Parameter(torch.randn(d_state))

        # 2. Per-state timestep 
        dt = 0.01 + torch.randn(d_state) * 0.01 
        dt = dt.clamp(0.005, 0.1) 
        self.log_dt = nn.Parameter(torch.log(dt))

        # 3. Input / Output projections
        self.B = nn.Parameter(torch.randn(d_state, d_input) / d_state**0.5)
        self.C_real = nn.Parameter(torch.randn(d_output, d_state) / d_state**0.5)
        self.C_imag = nn.Parameter(torch.randn(d_output, d_state) / d_state**0.5)

    def _get_lambda(self):
        real = -torch.exp(self.log_lambda_real)
        imag = self.lambda_imag
        return torch.complex(real, imag)

    def _get_dt(self):
        return torch.exp(self.log_dt).clamp(0.005, 0.1) 

    def _discretize(self, Lambda, B, dt):
        dt = dt.to(Lambda.device)
        denom = (1 - 0.5 * dt * Lambda)
        A_bar = (1 + 0.5 * dt * Lambda) / denom
        B_bar = (dt / denom).unsqueeze(-1) * B
        return A_bar, B_bar

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        Lambda = self._get_lambda()
        dt = self._get_dt()
        A_bar, B_bar = self._discretize(Lambda, self.B, dt)

        # --- 1. Vectorized Input Projection ---
        # Project entire sequence in one matrix multiplication
        u_real = torch.matmul(x, B_bar.real.T)
        u_imag = torch.matmul(x, B_bar.imag.T)
        u = torch.complex(u_real, u_imag) # Shape: (batch_size, seq_len, d_state)

        # --- 2. Compute the Convolution Kernel ---
        # Generate powers of A_bar: [A^0, A^1, ..., A^(L-1)]
        A_poles = A_bar.unsqueeze(1).expand(-1, seq_len)
        A_powers = torch.cumprod(A_poles, dim=1)
        
        # Shift to the right by 1 to start at A^0 = 1
        A_powers = torch.cat([torch.ones_like(A_powers[:, :1]), A_powers[:, :-1]], dim=1)

        # --- 3. Parallel Scan via FFT Convolution ---
        u = u.transpose(1, 2) # Shape: (batch, d_state, seq_len)

        # Calculate optimal FFT padding size (next power of 2 for speed)
        n_fft = 2 ** math.ceil(math.log2(seq_len * 2 - 1))
        
        # Transform to frequency domain
        u_f = torch.fft.fft(u, n=n_fft, dim=-1)
        A_f = torch.fft.fft(A_powers, n=n_fft, dim=-1)
        
        # Multiply in frequency domain (A_f broadcasts over batch dimension)
        h_f = u_f * A_f.unsqueeze(0)
        
        # Inverse FFT and truncate back to original sequence length
        h = torch.fft.ifft(h_f, n=n_fft, dim=-1)[..., :seq_len] # Shape: (batch, d_state, seq_len)
        
        # --- 4. Vectorized Output Projection ---
        h = h.transpose(1, 2) # Shape: (batch, seq_len, d_state)
        C_c = torch.complex(self.C_real, self.C_imag)
        
        # Taking the real part of (h @ C_c.T) mathematically equates to: 
        # (h.real @ C_real.T) - (h.imag @ C_imag.T)
        outputs = (h @ C_c.T).real # Shape: (batch, seq_len, d_output)

        return outputs