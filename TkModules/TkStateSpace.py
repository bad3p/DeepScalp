
import torch

# --------------------------------------------------------------------------------------------------------------
# State space module with HiPPO matrix
# --------------------------------------------------------------------------------------------------------------

class TkStateSpaceModule(torch.nn.Module):

    @staticmethod
    def make_hippo(d_state):
        """
        Generates the HiPPO-LegS matrix for optimal long-term memory.
        """
        q = torch.arange(d_state, dtype=torch.float32)
        # Create a grid of n and k indices
        n, k = torch.meshgrid(q, q, indexing="ij")
    
        # Apply the HiPPO-LegS formula
        # Condition 1: n > k
        A = torch.where(n > k, -torch.sqrt(2 * n + 1) * torch.sqrt(2 * k + 1), torch.zeros_like(n))
        # Condition 2: n == k
        A = torch.where(n == k, -(n + 1), A)
    
        return A

    def __init__(self, d_input, d_state, d_output):
        super().__init__()
        self.d_input = d_input
        self.d_state = d_state
        self.d_output = d_output
        
        # 1. State Space Matrices
        # Initialize A with the HiPPO matrix
        # self.A = torch.nn.Parameter(TkStateSpaceModule.make_hippo(d_state))
        self.A = torch.nn.Parameter(torch.randn(d_state, d_state) / d_state)
        
        # B is typically initialized as a uniform vector or ones when using HiPPO
        self.B = torch.nn.Parameter(torch.ones(d_state, d_input))
        
        # C and D remain randomly initialized
        self.C = torch.nn.Parameter(torch.randn(d_output, d_state) / d_state)
        self.D = torch.nn.Parameter(torch.randn(d_output, d_input) / d_input)
        
        # 2. Time Step
        self.log_dt = torch.nn.Parameter(torch.log(torch.ones(1) * 0.1))

    def forward(self, x):
        """
        Expects x of shape: (batch_size, seq_len, d_input)
        Returns y of shape: (batch_size, seq_len, d_output)
        """
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        # 3. Discretization Step (Euler Method)
        dt = torch.exp(self.log_dt)
        I = torch.eye(self.d_state, device=device)
        
        A_bar = I + dt * self.A
        B_bar = dt * self.B
        
        # 4. Sequential Scan
        h = torch.zeros(batch_size, self.d_state, device=device)
        outputs = []
        
        for t in range(seq_len):
            x_t = x[:, t, :]
            
            h = torch.matmul(h, A_bar.t()) + torch.matmul(x_t, B_bar.t())
            y_t = torch.matmul(h, self.C.t()) + torch.matmul(x_t, self.D.t())
            outputs.append(y_t)
            
        return torch.stack(outputs, dim=1)
