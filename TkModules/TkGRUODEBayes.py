import torch
import torch.nn as nn
from torchdiffeq import odeint

class TkGRUODEContinuous(nn.Module):
    """
    Defines the continuous-time dynamics of the hidden state.
    Reparameterized to integrate over tau in [0, 1] scaled by dt.
    """
    def __init__(self, hidden_size):
        super(TkGRUODEContinuous, self).__init__()
        
        self.lin_r = nn.Linear(hidden_size, hidden_size)
        self.lin_z = nn.Linear(hidden_size, hidden_size)
        self.lin_h = nn.Linear(hidden_size, hidden_size)
        
        # This will be dynamically injected by the wrapper before each integration step
        self.current_log_dt = None 

    def forward(self, tau, h):
        """
        tau: dummy integration time in [0, 1]
        h: hidden state at tau (Batch, Hidden_Size)
        """
        r = torch.sigmoid(self.lin_r(h))
        z = torch.sigmoid(self.lin_z(h))
        g = torch.tanh(self.lin_h(r * h))
        
        # Standard GRU-ODE flow: dh/dt
        dh_dt = (1 - z) * (g - h)
        
        # Reparameterization: dh/d(tau) = dh/dt * dt
        # Exp safely converts log(dt) back to the actual time delta
        dt = torch.exp(self.current_log_dt)
        
        return dh_dt * dt


class TkGRUObservationJump(nn.Module):
    """
    Applies the discrete jump to the hidden state when an observation occurs.
    """
    def __init__(self, input_size, hidden_size):
        super(TkGRUObservationJump, self).__init__()
        # Concatenate features and mask to handle missingness
        self.gru_cell = nn.GRUCell(input_size * 2, hidden_size)
        self.gru_layer_norm = nn.LayerNorm( hidden_size )

    def forward(self, h_minus, x, mask):
        x_combined = torch.cat([x, mask], dim=1)
        h_plus = self.gru_cell(x_combined, h_minus)

        #assert not torch.isnan(h_plus).any(), "h_plus has NaNs!"
        #h_plus = self.gru_layer_norm( h_plus )
        
        has_obs = (mask.sum(dim=1) > 0).float().unsqueeze(1)
        h_updated = has_obs * h_plus + (1 - has_obs) * h_minus
        
        return h_updated


class TkGRUODEBayesModule(nn.Module):
    """
    Refactored GRU-ODE wrapper for log(dt) time inputs.
    """
    def __init__(self, input_size, hidden_size, solver_method='rk4'):
        super(TkGRUODEBayesModule, self).__init__()
        self.hidden_size = hidden_size
        self.solver_method = solver_method
        
        self.ode_func = TkGRUODEContinuous(hidden_size)
        self.jump_func = TkGRUObservationJump(input_size, hidden_size)

    def forward(self, log_dts, values, masks):
        """
        log_dts: Log of time elapsed since previous observation (T, Batch, 1). 
                 Allows unique time gaps per batch item.
        values: Data tensor (T, Batch, Features). Missing values = 0.
        masks: Boolean/Float mask tensor (T, Batch, Features). 1=observed, 0=missing.
        """
        device = values.device
        seq_len, batch_size, _ = values.shape
        
        # Initialize hidden state
        h = torch.zeros(batch_size, self.hidden_size, device=device)
        
        outputs = []
        # We integrate over a fixed interval [0, 1] for tau. 
        # The true time scaling is handled inside ode_func via self.current_log_dt.
        integration_times = torch.tensor([0.0, 1.0], device=device)
        
        for i in range(seq_len):
            # 1. Inject the current batch's log(dt) into the continuous function
            self.ode_func.current_log_dt = log_dts[i]
            
            # 2. Continuous Evolution (ODE Solve)
            # If log(dt) is very negative (e.g., dt ~ 0), exp(log(dt)) ~ 0, 
            # and the state naturally won't evolve, handling synchronous points perfectly.
            h = odeint(self.ode_func, h, integration_times, method=self.solver_method)[-1]

            # Bound the integrated state to prevent inf from reaching the GRUCell
            h = torch.clamp(h, min=-10000.0, max=10000.0) # TODO: configure
            
            # 3. Discrete Jump (Observation Update)
            h = self.jump_func(h, values[i], masks[i])
            outputs.append(h)
            
        return torch.stack(outputs, dim=0)

