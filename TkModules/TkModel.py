
import torch

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
#------------------------------------------------------------------------------------------------------------------------

class TkModel(torch.nn.Module):

    def __init__(self, _layer_descriptors : list):
        
        super(TkModel, self).__init__()

        def create_conv_layer(params : list):
            layer_in_channels = params[0]
            layer_out_channels = params[1]
            layer_kernel_size = params[2]
            layer_stride = params[3]
            return torch.nn.Conv1d( in_channels=layer_in_channels, out_channels=layer_out_channels, kernel_size=layer_kernel_size, stride=layer_stride, padding=0)

        def create_deconv_layer(params : list):
            layer_in_channels = params[0]
            layer_out_channels = params[1]
            layer_kernel_size = params[2]
            layer_stride = params[3]
            return torch.nn.ConvTranspose1d( in_channels=layer_in_channels, out_channels=layer_out_channels, kernel_size=layer_kernel_size, stride=layer_stride, padding=0)

        def create_lrelu_layer(params : list):
            layer_leakage = params[0]
            return torch.nn.LeakyReLU(layer_leakage)
        
        def create_tanh_layer(params : list):
            return torch.nn.Tanh()

        def create_prelu_layer(params : list):
            channels = params[0]
            alpha = params[1]
            return torch.nn.PReLU(num_parameters=channels, init=alpha)

        def create_sigmoid_layer(params : list):
            return torch.nn.Sigmoid()

        def create_softmax_layer(params : list):
            dimension = params[0]
            return torch.nn.Softmax(dim=dimension)

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

        def create_layer(layer_descriptor : dict):
            if len(layer_descriptor.keys()) == 0:
                raise RuntimeError('Empty layer descriptor!')
            layer_type = list(layer_descriptor.keys())[0]
            if layer_type == 'Conv':
                return create_conv_layer( layer_descriptor[layer_type] )
            if layer_type == 'Deconv':
                return create_deconv_layer( layer_descriptor[layer_type] )
            if layer_type == 'LReLU':
                return create_lrelu_layer( layer_descriptor[layer_type] )
            if layer_type == 'TanH':
                return create_tanh_layer( layer_descriptor[layer_type] )
            if layer_type == 'PReLU':
                return create_prelu_layer( layer_descriptor[layer_type] )
            if layer_type == 'Sigmoid':
                return create_sigmoid_layer( layer_descriptor[layer_type] )
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
            else:
                raise RuntimeError('Unknown layer type: ' + layer_type + "!")

        self._layers = []

        for layer_descriptor in _layer_descriptors:
            self._layers.append( create_layer( layer_descriptor ) )

        self._layers = torch.nn.ModuleList( self._layers )

        self.initWeights()

    def initWeights(self) -> None:
        for m in self.modules():
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.ConvTranspose1d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, 0, 0.001)
                torch.nn.init.constant_(m.bias, 0)

    def forward(self, input):

        y = self._layers[0]( input )

        for layer_id in range(1, len(self._layers)):
            y = self._layers[layer_id]( y )
        
        return y
