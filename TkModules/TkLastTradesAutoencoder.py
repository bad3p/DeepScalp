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

        self._mean_layer = torch.nn.Linear(self._hidden_layer_size, self._code_layer_size)
        self._logvar_layer = torch.nn.Linear(self._hidden_layer_size, self._code_layer_size)
        self._reparametrization_layer = torch.nn.Linear(self._code_layer_size, self._hidden_layer_size)

    def code(self):
        return self._code

    def encode(self, input):
        y = self._encoder( input )
        mean = self._mean_layer(y)
        return mean * self._code_scale

    def forward(self, input):
        y = self._encoder( input )
        self._mean, self._logvar = self._mean_layer(y), self._logvar_layer(y)
        self._code = torch.cat( (self._mean, self._logvar), dim=1 )

        epsilon = torch.randn_like(self._logvar).to(y.device)  
        z = self._mean + self._logvar * epsilon
        z = self._reparametrization_layer(z)
        z = self._decoder( z )
        
        return z, self._mean, self._logvar