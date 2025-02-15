import configparser
import torch
import json
from TkModules.TkModel import TkModel

#------------------------------------------------------------------------------------------------------------------------

class TkOrderbookAutoencoder(torch.nn.Module):

    def __init__(self, _cfg : configparser.ConfigParser):

        super(TkOrderbookAutoencoder, self).__init__()

        self._cfg = _cfg
        self._code = None
        self._encoder = TkModel( json.loads(_cfg['Autoencoders']['OrderbookEncoder']) )
        self._decoder = TkModel( json.loads(_cfg['Autoencoders']['OrderbookDecoder']) )
        self._hidden_layer_size = int( _cfg['Autoencoders']['OrderbookAutoencoderHiddenLayerSize'])
        self._code_layer_size = int( _cfg['Autoencoders']['OrderbookAutoencoderCodeLayerSize'])
        self._code_scale = float( _cfg['Autoencoders']['OrderbookAutoencoderCodeScale'] )
        

        self._mean_layer = torch.nn.Linear(self._hidden_layer_size, int(self._code_layer_size/2))
        self._logvar_layer = torch.nn.Linear(self._hidden_layer_size, int(self._code_layer_size/2))
        self._reparametrization_layer = torch.nn.Linear(int(self._code_layer_size/2), self._hidden_layer_size)

    def code(self):
        return self._code

    def encode(self, input):
        y = self._encoder( input )
        mean, logvar = self._mean_layer(y), self._logvar_layer(y)
        return torch.cat( (mean, logvar), dim=1 ) * self._code_scale

    def forward(self, input):
        y = self._encoder( input )
        self._mean, self._logvar = self._mean_layer(y), self._logvar_layer(y)
        self._code = torch.cat( (self._mean, self._logvar), dim=1 )

        epsilon = torch.randn_like(self._logvar).to(y.device)  
        z = self._mean + self._logvar * epsilon
        z = self._reparametrization_layer(z)
        z = self._decoder( z )
        
        return z, self._mean, self._logvar
