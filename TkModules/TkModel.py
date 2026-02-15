
import torch

#------------------------------------------------------------------------------------------------------------------------
# Residual layer for MLP
#------------------------------------------------------------------------------------------------------------------------

class Residual(torch.nn.Module):
    def __init__(self, idx:int, in_dim:int, out_dim:int, scale:float, dropout:float, nonlinearity:torch.nn.Module):
        super().__init__()
        self._index = idx
        self._in_dim = in_dim
        self._out_dim = out_dim
        self._scale = scale
        self._dropout = torch.nn.Dropout(dropout) if dropout > 0 else None
        self._projection = torch.nn.Linear(in_dim, out_dim)
        self._nonlinearity = nonlinearity
        self.initWeights()

    def index(self):
        return self._index

    def forward(self, x_prev, x):
        if self._nonlinearity == None:
            if self._dropout == None:
                return x + self._projection(x_prev) * self._scale
            else:
                return x + self._dropout(self._projection(x_prev)) * self._scale
        else:
            if self._dropout == None:
                return x + self._nonlinearity(self._projection(x_prev)) * self._scale
            else:
                return x + self._dropout(self._nonlinearity(self._projection(x_prev))) * self._scale
    
    def initWeights(self) -> None:
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, 0, 0.001)
                torch.nn.init.constant_(m.bias, 0)

#------------------------------------------------------------------------------------------------------------------------
# Clamp as layer
#------------------------------------------------------------------------------------------------------------------------

class Clamp(torch.nn.Module):
    def __init__(self, min_val, max_val):
        super(Clamp, self).__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, x):
        return torch.clamp(x, min=self.min_val, max=self.max_val)

#------------------------------------------------------------------------------------------------------------------------
# Exp as layer
#------------------------------------------------------------------------------------------------------------------------

class Exp(torch.nn.Module):
    def __init__(self):
        super(Exp, self).__init__()

    def forward(self, x):
        return torch.exp(x)
    
#------------------------------------------------------------------------------------------------------------------------
# Channel normalization
#------------------------------------------------------------------------------------------------------------------------

class ChannelNorm(torch.nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.ln = torch.nn.LayerNorm([num_channels])

    def forward(self, x):
        # x: (B, C, L)
        x = x.transpose(1, 2)   # (B, L, C)
        x = self.ln(x)
        return x.transpose(1, 2)

#------------------------------------------------------------------------------------------------------------------------
# Simple sequence of pytorch layers, constructed using given specification (list of layer descriptors).
# Each layer descriptor is a dictionary containing the single key-value pair.
# The key is string describing the type of layer, the value is a list describing its parameters.
#   {"Conv":[in_features,out_features,kernel,stride]}, 
#   {"Deconv":[in_features,out_features,kernel,stride]}, 
#   {"LReLU":[leakege]}, 
#   {"Drop":[probability]}, 
#   {"Flatten":[]},
#   {"Linear":[in_neurons,out_neurons]},
#   {"Unflatten":[size_x,size_y]},
#   ...
#------------------------------------------------------------------------------------------------------------------------

class TkModel(torch.nn.Module):

    def __init__(self, _layer_descriptors : list):
        
        super(TkModel, self).__init__()

        def create_conv_layer(params : list):
            layer_in_channels = params[0]
            layer_out_channels = params[1]
            layer_kernel_size = params[2]
            layer_stride = params[3]
            layer_padding = 0
            if len(params) > 4:
                layer_padding = params[4]
            return torch.nn.Conv1d( in_channels=layer_in_channels, out_channels=layer_out_channels, kernel_size=layer_kernel_size, stride=layer_stride, padding=layer_padding)

        def create_deconv_layer(params : list):
            layer_in_channels = params[0]
            layer_out_channels = params[1]
            layer_kernel_size = params[2]
            layer_stride = params[3]
            layer_padding = 0
            if len(params) > 4:
                layer_padding = params[4]
            return torch.nn.ConvTranspose1d( in_channels=layer_in_channels, out_channels=layer_out_channels, kernel_size=layer_kernel_size, stride=layer_stride, padding=layer_padding)
        
        def create_maxpool_layer( params:list ):
            layer_kernel_size = params[0]
            layer_stride = params[1]
            return torch.nn.MaxPool1d( kernel_size=layer_kernel_size, stride=layer_stride)
        
        def create_avgpool_layer( params:list ):
            layer_kernel_size = params[0]
            layer_stride = params[1]
            return torch.nn.AvgPool1d( kernel_size=layer_kernel_size, stride=layer_stride)
        
        def create_upsample_layer( params:list ):
            layer_scale_factor = params[0]
            return torch.nn.Upsample( scale_factor=layer_scale_factor )

        def create_lrelu_layer(params : list):
            layer_leakage = params[0]
            return torch.nn.LeakyReLU(layer_leakage)
        
        def create_tanh_layer(params : list):
            return torch.nn.Tanh()
        
        def create_elu_layer(params : list):
            alpha = params[0]
            return torch.nn.ELU(alpha)
        
        def create_gelu_layer(params : list):
            return torch.nn.GELU()

        def create_prelu_layer(params : list):
            channels = params[0]
            alpha = params[1]
            return torch.nn.PReLU(num_parameters=channels, init=alpha)

        def create_sigmoid_layer(params : list):
            return torch.nn.Sigmoid()

        def create_softmax_layer(params : list):
            dimension = params[0]
            return torch.nn.Softmax(dim=dimension)
        
        def create_softplus_layer(params : list):
            return torch.nn.Softplus()

        def create_dropout_layer(params : list):
            layer_dropout_prob = params[0]
            return torch.nn.Dropout( p=layer_dropout_prob )

        def create_flatten_layer(params : list):
            return torch.nn.Flatten()

        def create_unflatten_layer(params : list):
            layer_size_x = params[0]
            layer_size_y = params[1]
            return torch.nn.Unflatten(1, (layer_size_x,layer_size_y))

        def create_linear_layer(params : list):
            layer_in_channels = params[0]
            layer_out_channels = params[1]
            return torch.nn.Linear( layer_in_channels, layer_out_channels )
        
        def create_lnorm_layer(params:list):
            layer_channels = params[0]
            return torch.nn.LayerNorm( [layer_channels] )
        
        def create_cnorm_layer(params:list):
            layer_channels = params[0]
            return ChannelNorm( layer_channels )
        
        def create_clamp_layer(params:list):
            min = params[0]
            max = params[1]
            return Clamp( min, max )
        
        def create_exp_layer(params:list):
            return Exp()
        
        def create_residual_layer(params:list):
            index = params[0]
            layer_in_channels = params[1]
            layer_out_channels = params[2]
            scale = 1.0
            if len(params) > 3:
                residual_scale = params[3]
            dropout = 0.0
            if len(params) > 4:
                dropout = params[4]
            nonlinearity = None
            if len(params) > 5:
                nonlinearity = create_layer( {params[5]: [0.01]})
            return Residual(index, layer_in_channels, layer_out_channels, scale, dropout, nonlinearity)

        def create_layer(layer_descriptor : dict):
            if len(layer_descriptor.keys()) == 0:
                raise RuntimeError('Empty layer descriptor!')
            layer_type = list(layer_descriptor.keys())[0]
            if layer_type == 'Conv':
                return create_conv_layer( layer_descriptor[layer_type] )
            if layer_type == 'Deconv':
                return create_deconv_layer( layer_descriptor[layer_type] )
            if layer_type == 'MaxPool':
                return create_maxpool_layer( layer_descriptor[layer_type] )
            if layer_type == 'AvgPool':
                return create_avgpool_layer( layer_descriptor[layer_type] )
            if layer_type == 'Upsample':
                return create_upsample_layer( layer_descriptor[layer_type] )
            if layer_type == 'LReLU':
                return create_lrelu_layer( layer_descriptor[layer_type] )
            if layer_type == 'TanH':
                return create_tanh_layer( layer_descriptor[layer_type] )
            if layer_type == 'ELU':
                return create_elu_layer( layer_descriptor[layer_type] )
            if layer_type == 'GELU':
                return create_gelu_layer( layer_descriptor[layer_type] )
            if layer_type == 'PReLU':
                return create_prelu_layer( layer_descriptor[layer_type] )
            if layer_type == 'Sigmoid':
                return create_sigmoid_layer( layer_descriptor[layer_type] )
            if layer_type == 'Softplus':
                return create_softplus_layer( layer_descriptor[layer_type] )
            if layer_type == 'Softmax':
                return create_softmax_layer( layer_descriptor[layer_type] )
            if layer_type == 'Drop':
                return create_dropout_layer( layer_descriptor[layer_type] )
            if layer_type == 'Flatten':
                return create_flatten_layer( layer_descriptor[layer_type] )
            if layer_type == 'Unflatten':
                return create_unflatten_layer( layer_descriptor[layer_type] )
            if layer_type == 'Linear':
                return create_linear_layer( layer_descriptor[layer_type] )
            if layer_type == 'Residual':
                return create_residual_layer( layer_descriptor[layer_type] )
            if layer_type == 'LNorm':
                return create_lnorm_layer( layer_descriptor[layer_type])
            if layer_type == 'CNorm':
                return create_cnorm_layer( layer_descriptor[layer_type])
            if layer_type == 'Clamp':
                return create_clamp_layer( layer_descriptor[layer_type])
            if layer_type == 'Exp':
                return create_exp_layer( layer_descriptor[layer_type])
            else:
                raise RuntimeError('Unknown layer type: ' + layer_type + "!")

        self._layers = []

        for layer_descriptor in _layer_descriptors:
            self._layers.append( create_layer( layer_descriptor ) )

        self._layers = torch.nn.ModuleList( self._layers )
        self._layer_outputs = []

        self.initWeights()

    def initWeights(self, conv_init_mode='kaiming_normal', conv_init_gain:float=1.0) -> None:
        for m in self.modules():
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.ConvTranspose1d):
                if conv_init_mode == 'kaiming_normal':
                    torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                else:
                    torch.nn.init.xavier_normal_(m.weight,gain=conv_init_gain)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, 0, 0.001)
                torch.nn.init.constant_(m.bias, 0)

    def forward(self, input):

        self._layer_outputs = [input]
        self._layer_outputs.append( self._layers[0]( input ) )

        for layer_id in range(1, len(self._layers)):
            if isinstance(self._layers[layer_id], Residual):
                residual: Residual = self._layers[layer_id]
                index = residual.index()
                self._layer_outputs.append( self._layers[layer_id]( self._layer_outputs[index], self._layer_outputs[-1] ) )
            else:
                self._layer_outputs.append( self._layers[layer_id]( self._layer_outputs[-1] ) )
        
        return self._layer_outputs[-1]
    
    def layer_outputs(self):
        return self._layer_outputs
