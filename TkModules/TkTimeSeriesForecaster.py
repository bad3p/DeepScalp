
import configparser
import torch
import json
from TkModules.TkModel import TkModel
from TkModules.TkStackedLSTM import TkStackedLSTM

# --------------------------------------------------------------------------------------------------------------
# Time series forecasting model
# --------------------------------------------------------------------------------------------------------------

class TkTimeSeriesForecaster(torch.nn.Module):

    def __init__(self, _cfg : configparser.ConfigParser):

        super(TkTimeSeriesForecaster, self).__init__()

        self._cfg = _cfg
        self._prior_steps_count = int(_cfg['TimeSeries']['PriorStepsCount']) 
        self._input_width = int(_cfg['TimeSeries']['InputWidth']) 
        self._target_code_width = int(_cfg['Autoencoders']['LastTradesAutoencoderCodeLayerSize']) 
        self._input_slices = json.loads(_cfg['TimeSeries']['InputSlices'])
        self._lstm_specification = json.loads(_cfg['TimeSeries']['LSTM'])
        self._mlp = TkModel( json.loads(_cfg['TimeSeries']['MLP']) )
        self._soft_max = torch.nn.Softmax(dim=1)

        if len(self._input_slices) != len(self._lstm_specification):
            raise RuntimeError('InputSlices and LSTM config mismatched!')

        self._lstm = []
        self._mlp_input_size = 0
        for i in range(len(self._input_slices)):
            ch0 = self._input_slices[i][0]
            ch1 = self._input_slices[i][1]
            slice_size = ch1 - ch0
            self._lstm.append( TkStackedLSTM( slice_size, self._prior_steps_count, self._lstm_specification[i]) )
            self._mlp_input_size = self._mlp_input_size + self._lstm_specification[i][-1]

        self._lstm = torch.nn.ModuleList( self._lstm )
        
        self._mlp_input_tensors = None

    def mlp_input(self):
        return self._mlp_input_tensors

    def input_slice(self, idx:int):
        return self._input_slice_tensors[idx]

    def forward(self, input):

        batch_size = input.shape[0]

        input = torch.reshape( input, ( batch_size, self._prior_steps_count, self._input_width) )

        self._input_slice_tensors = []
        self._mlp_input_tensors = []

        for i in range(len(self._input_slices)):
            ch0 = self._input_slices[i][0]
            ch1 = self._input_slices[i][1]
            slice_size = ch1 - ch0
            input_slice_tensor = input[:, :, ch0:ch1]
            self._input_slice_tensors.append( input_slice_tensor )
            self._mlp_input_tensors.append( self._lstm[i](input_slice_tensor) )
        
        self._mlp_input_tensors = tuple(self._mlp_input_tensors)
        self._mlp_input_tensors = torch.cat( self._mlp_input_tensors, dim=1) 

        y = torch.reshape( self._mlp_input_tensors, (batch_size, self._mlp_input_size) )
        y = self._mlp.forward( y )
        y = torch.reshape( y, ( batch_size, self._target_code_width) )

        return y