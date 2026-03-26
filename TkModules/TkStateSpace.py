import torch
import torch.nn as nn

# --------------------------------------------------------------------------------------------------
# S4D (Diagonal State Space Model) with PER-STATE dt
# - Each state learns its own timescale
# - Stable (negative real spectrum + bilinear discretization)
# - Optimized: D matrix removed in favor of explicit outer residual connections
# --------------------------------------------------------------------------------------------------

class TkStateSpaceModule(torch.nn.Module):

    def __init__(self, d_input, d_state, d_output):
        super().__init__()
        self.d_input = d_input
        self.d_state = d_state
        self.d_output = d_output

        # ------------------------------------------------------------------
        # 1. Diagonal complex spectrum
        # λ = -exp(real) + i * imag  (stable)
        # ------------------------------------------------------------------
        self.log_lambda_real = nn.Parameter(torch.randn(d_state))
        self.lambda_imag = nn.Parameter(torch.randn(d_state))

        # ------------------------------------------------------------------
        # 2. Per-state timestep (KEY UPGRADE)
        # each state learns its own dt
        # ------------------------------------------------------------------
        dt = 0.01 + torch.randn(d_state) * 0.01 # todo: configure
        dt = dt.clamp(0.005, 0.1) # todo: configure
        log_dt = torch.log(dt)
        self.log_dt = nn.Parameter( log_dt )

        # ------------------------------------------------------------------
        # 3. Input / Output projections
        # D matrix removed; feed-through is handled by external residual
        # ------------------------------------------------------------------
        self.B = nn.Parameter(torch.randn(d_state, d_input) / d_state**0.5)
        self.C_real = nn.Parameter(torch.randn(d_output, d_state) / d_state**0.5)
        self.C_imag = nn.Parameter(torch.randn(d_output, d_state) / d_state**0.5)

    def _get_lambda(self):
        """Stable complex eigenvalues"""
        real = -torch.exp(self.log_lambda_real)
        imag = self.lambda_imag
        return torch.complex(real, imag)

    def _get_dt(self):
        """Per-state timestep (clamped for stability)"""
        return torch.exp(self.log_dt).clamp(0.005, 0.1) # TODO: configure

    def _discretize(self, Lambda, B, dt):
        """
        Elementwise bilinear transform (per-state)
        """
        # reshape for broadcasting
        dt = dt.to(Lambda.device)

        denom = (1 - 0.5 * dt * Lambda)
        A_bar = (1 + 0.5 * dt * Lambda) / denom

        # B_bar per state
        B_bar = (dt / denom).unsqueeze(-1) * B

        return A_bar, B_bar

    def forward(self, x):
        """
        x: (batch, seq, d_input)
        """
        batch_size, seq_len, _ = x.shape
        device = x.device

        Lambda = self._get_lambda()
        dt = self._get_dt()

        A_bar, B_bar = self._discretize(Lambda, self.B, dt)

        # complex hidden state
        h = torch.zeros(batch_size, self.d_state, dtype=torch.cfloat, device=device)
        outputs = []

        for t in range(seq_len):
            x_t = x[:, t, :]

            # project input → state space
            u_real = torch.einsum('bd,sd->bs', x_t, B_bar.real)
            u_imag = torch.einsum('bd,sd->bs', x_t, B_bar.imag)
            u = torch.complex(u_real, u_imag)

            # diagonal recurrence
            h = h * A_bar + u

            # project back to real output (D matrix feed-through removed)
            y_t = (h.real @ self.C_real.T) - (h.imag @ self.C_imag.T)
            outputs.append(y_t)

        return torch.stack(outputs, dim=1)