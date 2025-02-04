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
        self._sparsity = int( _cfg['Autoencoders']['LastTradesAutoencoderSparsity'])
        self._encoder_output = torch.nn.Sigmoid()
        self._soft_max = torch.nn.Softmax(dim=2)

    def code(self):
        return self._code

    def encode(self, input):
        return self._encoder_output( self._encoder( input ) )

    def forward(self, input):
        self._code = self._encoder_output( self._encoder( input ) )

        _, indices = torch.topk(self._code, self._sparsity)
        mask = torch.zeros(self._code.size()).cuda()
        mask.scatter_(1, indices, 1)
        self._code = torch.mul(self._code , mask)

        return self._soft_max( self._decoder( self._code ) )