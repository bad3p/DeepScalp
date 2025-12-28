import configparser
import torch
import json
from TkModules.TkModel import TkModel

#------------------------------------------------------------------------------------------------------------------------

class TkLastTradesAutoencoder(torch.nn.Module):

    def __init__(self, _cfg : configparser.ConfigParser):

        super(TkLastTradesAutoencoder, self).__init__()

        self._cfg = _cfg
        self._code = None
        self._encoder = TkModel( json.loads(_cfg['Autoencoders']['LastTradesEncoder']) )
        self._decoder = TkModel( json.loads(_cfg['Autoencoders']['LastTradesDecoder']) )
        self._hidden_layer_size = int( _cfg['Autoencoders']['LastTradesAutoencoderHiddenLayerSize'])
        self._code_layer_size = int( _cfg['Autoencoders']['LastTradesAutoencoderCodeLayerSize'])        
        self._code_scale = float( _cfg['Autoencoders']['LastTradesAutoencoderCodeScale'] )
        self._smoothness_loss_noise_std = 0.01 # TODO: configure

        self._mean_layer = torch.nn.Linear(self._hidden_layer_size, self._code_layer_size)
        self._logvar_layer = torch.nn.Linear(self._hidden_layer_size, self._code_layer_size)
        self._reparametrization_layer = torch.nn.Linear(self._code_layer_size, self._hidden_layer_size)

    def code_layer_size(self):
        return self._code_layer_size

    def code(self):
        return self._code
    
    def get_layer_by_parameter(model, target_param):
        for name, param in model.named_parameters():
            # Check if the parameter object matches the target
            if param is target_param:
                # The 'name' is in the format 'layer_name.sub_layer_name.weight'
                # We need to extract the actual module/layer
            
                # Split the name by '.' to get the layer path
                parts = name.split('.')
                # The last part is typically the parameter name itself (e.g., 'weight', 'bias')
                layer_name = '.'.join(parts[:-1])
            
                # Access the module using getattr recursively
                current_module = model
                for part in parts[:-1]:
                    current_module = getattr(current_module, part)
                return current_module
            
        return None

    def get_trainable_parameters(self, conv_weight_decay:float, dense_weight_decay:float):

        encoder_conv_params = []
        encoder_dense_params = []
        encoder_no_decay_params = []
        for name, param in self._encoder.named_parameters():
            if not param.requires_grad:
                continue
            if not any(nd in name for nd in ["bias", "norm"]):
                layer = self.get_layer_by_parameter( param )
                if isinstance( layer, torch.nn.Linear):
                    encoder_dense_params.append(param)
                else:
                    encoder_conv_params.append(param)
            else:
                encoder_no_decay_params.append(param)
        
        decoder_conv_params = []
        decoder_dense_params = []
        decoder_no_decay_params = []
        for name, param in self._decoder.named_parameters():
            if not param.requires_grad:
                continue
            if not any(nd in name for nd in ["bias", "norm"]):
                layer = self.get_layer_by_parameter( param )
                if isinstance( layer, torch.nn.Linear):
                    decoder_dense_params.append(param)
                else:
                    decoder_conv_params.append(param)
            else:
                decoder_no_decay_params.append(param)

        return [
            {"params": encoder_conv_params, "weight_decay": conv_weight_decay},
            {"params": encoder_dense_params, "weight_decay": dense_weight_decay},
            {"params": encoder_no_decay_params, "weight_decay": 0.0},
            {"params": decoder_conv_params, "weight_decay": conv_weight_decay},
            {"params": decoder_dense_params, "weight_decay": dense_weight_decay},
            {"params": decoder_no_decay_params, "weight_decay": 0.0}
        ]

    def encode(self, input):
        y = self._encoder( input )
        mean = self._mean_layer(y)
        return mean * self._code_scale
    
    def decode(self, input):
        batch_size = input.shape[0]
        mean = input * ( 1.0 / self._code_scale )        
        z = self._reparametrization_layer( mean )
        z = self._decoder( z )
        return z

    def forward(self, input):
        y = self._encoder( input )
        self._mean, self._logvar = self._mean_layer(y), self._logvar_layer(y)
        self._logvar = self._logvar.clamp( -6.0, 2.0 )
        self._code = torch.cat( (self._mean, self._logvar), dim=1 )

        z = self._mean + torch.randn_like( torch.exp(0.5 * self._logvar) )
        z = self._reparametrization_layer(z)
        z = self._decoder( z )
        z = z.clamp( 0, 1 )

        # smoothness loss
        eps = self._smoothness_loss_noise_std * torch.randn_like( input )
        y_noise = self._encoder( input + eps )
        y_noise_mean = self._mean_layer(y_noise)
        
        return z, self._mean, self._logvar, torch.mean(( self._mean - y_noise_mean).pow(2))
    
    def freeze_parameters(self):
        for p in self.parameters():
            p.requires_grad = False