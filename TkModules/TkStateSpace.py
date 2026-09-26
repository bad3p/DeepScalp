import torch
import torch.nn as nn
import math

#----------------------------------------------------------------------------------------------------
# S4D for TIDES Architecture (Explicit Time-Gating)
#----------------------------------------------------------------------------------------------------

class TkStateSpaceModule(torch.nn.Module):
    def __init__(self, d_input, d_state, d_output):
        super().__init__()
        self.d_input = d_input
        self.d_state = d_state
        self.d_output = d_output

        # 1. Diagonal complex spectrum
        self.log_lambda_real = nn.Parameter(torch.randn(d_state))
        self.lambda_imag = nn.Parameter(torch.randn(d_state))

        # 2. Per-state timestep[cite: 1]
        dt = 0.01 + torch.randn(d_state) * 0.01 
        dt = dt.clamp(0.005, 0.1) 
        self.log_dt = nn.Parameter(torch.log(dt))

        # 3. Input / Output projections[cite: 1]
        self.B = nn.Parameter(torch.randn(d_state, d_input) / d_state**0.5)
        self.C_real = nn.Parameter(torch.randn(d_output, d_state) / d_state**0.5)
        self.C_imag = nn.Parameter(torch.randn(d_output, d_state) / d_state**0.5)

    def _get_lambda(self):
        real = -torch.exp(self.log_lambda_real)
        imag = self.lambda_imag
        return torch.complex(real, imag)

    def _get_dt(self, observed_log_dt):
        """
        Combines the irregular observed log_delta_times with the learnable per-channel log_dt.
        """
        if observed_log_dt.dim() == 2:
            # Broadcast to match d_state dimension
            observed_log_dt = observed_log_dt.unsqueeze(-1)
        
        # Modify observed irregular log-delta-times using learnable log_dt
        log_dt_combined = observed_log_dt + self.log_dt
        return torch.exp(log_dt_combined).clamp(0.005, 100.0) 

    def _discretize(self, Lambda, B, dt):
        dt = dt.to(Lambda.device)
        
        # dt is now shape (batch_size, seq_len, d_state)
        # denom and A_bar will also be (batch_size, seq_len, d_state)
        denom = (1 - 0.5 * dt * Lambda)
        A_bar = (1 + 0.5 * dt * Lambda) / denom
        
        # B is (d_state, d_input), unsqueeze dt/denom to align dimensions
        # B_bar becomes (batch_size, seq_len, d_state, d_input)
        B_bar = (dt / denom).unsqueeze(-1) * B
        return A_bar, B_bar

    def forward(self, x, observed_log_dt):
        batch_size, seq_len, _ = x.shape
        
        Lambda = self._get_lambda()
        dt = self._get_dt(observed_log_dt) 
        A_bar, B_bar = self._discretize(Lambda, self.B, dt)

        # --- 1. Vectorized Input Projection ---
        # Project the sequence using the time-varying B_bar via einsum
        u_real = torch.einsum('bsi,bsdi->bsd', x, B_bar.real)
        u_imag = torch.einsum('bsi,bsdi->bsd', x, B_bar.imag)
        u = torch.complex(u_real, u_imag) # Shape: (batch_size, seq_len, d_state)

        # --- 2. Sequential Scan (Time-Gated Recurrence) ---
        # Because A_bar varies per time-step (due to irregular dt), 
        # the standard O(L log L) FFT convolution cannot be used. 
        h_list = []
        h_t = torch.zeros(batch_size, self.d_state, dtype=torch.complex64, device=x.device)
        
        for t in range(seq_len):
            h_t = A_bar[:, t, :] * h_t + u[:, t, :]
            h_list.append(h_t)
            
        h = torch.stack(h_list, dim=1) # Shape: (batch_size, seq_len, d_state)
        
        # --- 3. Vectorized Output Projection ---
        C_c = torch.complex(self.C_real, self.C_imag)
        
        outputs = (h @ C_c.T).real # Shape: (batch_size, seq_len, d_output)

        return outputs