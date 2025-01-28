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
        self._sigmoid = torch.nn.Sigmoid()

    def code(self):
        return self._code

    def encode(self, input):
        return self._encoder( input )

    def forward(self, input):
        self._code = self._encoder( input )
        return self._sigmoid( self._decoder( self._code ) )