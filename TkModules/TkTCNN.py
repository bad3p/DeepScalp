
import torch
import math

#------------------------------------------------------------------------------------------------------------------------
# Stochastic Depth : Drops entire (residual) branch per-sample
#------------------------------------------------------------------------------------------------------------------------

class StochasticDepth(torch.nn.Module):
    def __init__(self, prob: float = 0.0):
        super().__init__()
        self._prob = prob

    def forward(self, x):
        if not self.training or self._prob == 0.0:
            return x

        keep_prob = 1.0 - self._prob

        # Shape: (batch, 1, 1, ...)
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)

        random_tensor = keep_prob + torch.rand( shape, dtype=x.dtype, device=x.device )
        binary_mask = random_tensor.floor()

        # Scale to preserve expected value
        return x / keep_prob * binary_mask

#------------------------------------------------------------------------------------------------------------------------
# Gated feature mixer
# Input: Tensor of shape (batch, time, channels)
# Output: Tensor of same shape
#
# Initial gate bias == 2 -> sigmoid(2) ≈ 0.88 -> gates mostly open
#------------------------------------------------------------------------------------------------------------------------

class GatedFeatureMixer(torch.nn.Module):    
    def __init__( self, channels: int, history_size :int, hidden_dim: int = 128, dropout: float = 0.1, gate_bias_init: float = 2.0):
        
        super().__init__()

        self._norm = torch.nn.LayerNorm([channels,history_size])

        self._mlp = torch.nn.Sequential(
            torch.nn.Linear(channels, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, channels)
        )

        # Initialize last layer bias to keep gates open initially
        torch.nn.init.constant_(self._mlp[-1].bias, gate_bias_init)

    def forward(self, x):
        # x: (B, T, C)        
        # Normalize across feature dimension
        x_norm = self._norm(x)

        # Temporal summary (robust to noise)
        # Shape: (B, C)
        x_summary = x_norm.mean(dim=2)

        # Compute gates
        # Shape: (B, C)
        x_gates = torch.sigmoid(self._mlp(x_summary))

        # gates debug
        # ( if STD -> 0 and MANY GATES < 0.2 ) -> gating is killing signal before conv squeez
        #print(
        #    x_gates.mean().item(),
        #    x_gates.std().item(),
        #    (x_gates < 0.2).float().mean().item()
        #)

        # Apply gates
        # Broadcast over time dimension
        x_gated = x * x_gates.unsqueeze(2)

        return x_gated

#------------------------------------------------------------------------------------------------------------------------
# Temporal attention pooling
#------------------------------------------------------------------------------------------------------------------------

class TemporalAttentionPooling(torch.nn.Module):
    def __init__(self, channels : int):
        super().__init__()
        self.attn = torch.nn.Linear(channels, 1)

    def forward(self, x):
        # x: (B, C, T) ->  (B, T, C)
        y = x.transpose(1,2)

        # Compute attention weights
        # Shape: (B, T)
        weights = torch.softmax(self.attn(y).squeeze(-1), dim=1)

        # Weighted sum
        # Shape: (B, C)
        pooled = torch.sum(y * weights.unsqueeze(-1), dim=1)

        return pooled

#------------------------------------------------------------------------------------------------------------------------
# Causal temporal convolution module
#------------------------------------------------------------------------------------------------------------------------

class CausalTemporalConv1d(torch.nn.Module):
    def __init__(self, input_size:int, output_size:int, kernel_size:int, dilation:int, leakage:float, dropout:float, path_dropout:float, use_skip_path:bool, residual_path_scale:float):
        super().__init__()        
        padding = dilation * (kernel_size - 1)
        self._pad = torch.nn.ConstantPad1d((padding, 0), 0.0)
        self._conv1d = torch.nn.Conv1d( input_size, output_size, kernel_size=kernel_size, stride=1, padding=0, dilation=dilation)
        self._batchNorm1d = torch.nn.BatchNorm1d( output_size )
        self._lrelu = torch.nn.LeakyReLU( leakage )
        self._dropout1d = torch.nn.Dropout1d( dropout) if dropout > 0 else None
        self._path_dropout = StochasticDepth( path_dropout ) if path_dropout > 0 else None
        self._skip_path = ( torch.nn.Conv1d( input_size, output_size, 1) if input_size != output_size else torch.nn.Identity() ) if use_skip_path else None
        self._residual_path_scale = residual_path_scale
        self.initWeights()

    def forward(self, x):
        y = self._pad( x )
        y = self._conv1d( y )
        y = self._batchNorm1d( y )
        y = self._lrelu( y )
        if self._dropout1d != None:
            y = self._dropout1d( y )   
        if self._path_dropout != None:
            y = self._path_dropout( y )
        if self._skip_path != None:
            y = y * self._residual_path_scale + self._skip_path(x)
            y = self._lrelu( y )
        return y
    
    def initWeights(self) -> None:
        for m in self.modules():
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.ConvTranspose1d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)                


# --------------------------------------------------------------------------------------------------------------
# Temporal CNN module
# --------------------------------------------------------------------------------------------------------------

class TkTCNN(torch.nn.Module):

    def __init__(self, _input_size : int, _history_size : int, _specification : list):
        super(TkTCNN, self).__init__()
          
        self._input_size = _input_size
        self._history_size = _history_size
        self._specification = _specification
        
        self._layers = []

        for i in range( len(self._specification) ):
            input_size = self._specification[i][0]
            output_size = self._specification[i][1]
            kernel_size = self._specification[i][2]
            dilation = self._specification[i][3]
            leakage = self._specification[i][4]            
            dropout = self._specification[i][5]
            path_dropout = self._specification[i][6]
            residual_path_scale = 1.0 / math.sqrt( i + 1 )
            self._layers.append( CausalTemporalConv1d( input_size, output_size, kernel_size, dilation, leakage, dropout, path_dropout, use_skip_path=True, residual_path_scale=residual_path_scale) )

        #self._layers.append( torch.nn.AdaptiveAvgPool1d(1) )        
        #self._layers.append( torch.nn.Conv1d( self._specification[-1][1], self._specification[-1][1], kernel_size=self._history_size) )
        self._layers.append( GatedFeatureMixer( self._specification[-1][1], self._history_size, self._specification[-1][1] * 2 ) )
        self._layers.append( TemporalAttentionPooling( self._specification[-1][1] ) )
        self._layers = torch.nn.ModuleList( self._layers )
        #self._layerNorm = torch.nn.LayerNorm( [self._specification[-1][1]] )

        self.initWeights()

    def initWeights(self) -> None:

        for name, p in self.named_parameters():
            if 'lstm' in name:
                if 'weight_ih' in name:
                    torch.nn.init.xavier_uniform_(p.data)
                    #torch.nn.init.normal_(p.data, 0.0, 0.2)
                elif 'weight_hh' in name:
                    torch.nn.init.orthogonal_(p.data)
                    #torch.nn.init.normal_(p.data, 0.0, 0.2)
                elif 'bias_ih' in name:
                    p.data.fill_(0)
                    # Set forget-gate bias to 1
                    n = p.size(0)
                    p.data[(n // 4):(n // 2)].fill_(1)
                elif 'bias_hh' in name:
                    p.data.fill_(0)
            elif 'fc' in name:
                if 'weight' in name:
                    torch.nn.init.xavier_uniform_(p.data)
                elif 'bias' in name:
                    p.data.fill_(0)

        for m in self.modules():
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.ConvTranspose1d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d) or isinstance(m, torch.nn.LayerNorm):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, 0, 0.01)
                torch.nn.init.constant_(m.bias, 0)
            

    def forward(self, input):

        batch_size = input.shape[0]

        y = input.transpose(1,2)

        for layer_id in range(0, len(self._layers)):
            y = self._layers[layer_id]( y )

        y = y.reshape( (batch_size, self._specification[-1][1]) )
        #y = self._layerNorm( y )
        return y
