
import configparser
import torch
import json
from TkModules.TkModel import TkModel
from TkModules.TkStackedLSTM import TkStackedLSTM
from TkModules.TkSelfAttention import TkSelfAttention
from TkModules.TkTCNN import TkTCNN

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
        self._tcnn_specification = json.loads(_cfg['TimeSeries']['TCNN'])
        self._lstm_specification = json.loads(_cfg['TimeSeries']['LSTM'])
        self._mlp = TkModel( json.loads(_cfg['TimeSeries']['MLP']) )

        if len(self._input_slices) != len(self._lstm_specification):
            raise RuntimeError('InputSlices and LSTM config mismatched!')
        
        self._tcnn = TkTCNN( self._input_width, self._prior_steps_count, self._tcnn_specification )

        self._lstm = []
        self._mlp_input_size = self._tcnn_specification[-1][1]
        for i in range(len(self._input_slices)):
            ch0 = self._input_slices[i][0]
            ch1 = self._input_slices[i][1]
            slice_size = ch1 - ch0
            self._lstm.append( TkStackedLSTM( slice_size, self._prior_steps_count, self._lstm_specification[i]) )
            self._mlp_input_size = self._mlp_input_size + self._lstm_specification[i][-1]

        self._lstm = torch.nn.ModuleList( self._lstm )
        
        self._lstm_output_tensors = None

    def get_trainable_parameters(self, lstm_weight_decay:float, tcnn_weight_decay:float, mlp_weight_decay:float):

        lstm_decay_params = self._lstm.parameters()
        lstm_no_decay_params = []

        tcnn_decay_params = []
        tcnn_no_decay_params = []
        for name, param in self._tcnn.named_parameters():
            if not param.requires_grad:
                continue
            if not any(nd in name for nd in ["bias", "norm"]):
                tcnn_decay_params.append(param)
            else:
                tcnn_no_decay_params.append(param)
        
        mlp_decay_params = []
        mlp_no_decay_params = []
        for name, param in self._mlp.named_parameters():
            if not param.requires_grad:
                continue
            if not any(nd in name for nd in ["bias", "norm"]):
                mlp_decay_params.append(param)
            else:
                mlp_no_decay_params.append(param)

        return [
            {"params": lstm_decay_params, "weight_decay": lstm_weight_decay},
            {"params": lstm_no_decay_params, "weight_decay": 0.0},
            {"params": tcnn_decay_params, "weight_decay": tcnn_weight_decay},
            {"params": tcnn_no_decay_params, "weight_decay": 0.0},
            {"params": mlp_decay_params, "weight_decay": mlp_weight_decay},
            {"params": mlp_no_decay_params, "weight_decay": 0.0},
        ]

    def lstm_output(self):
        return self._lstm_output_tensors

    def input_slice(self, idx:int):
        return self._input_slice_tensors[idx]

    def forward(self, input):

        batch_size = input.shape[0]

        input = torch.reshape( input, ( batch_size, self._prior_steps_count, self._input_width) )

        self._input_slice_tensors = []
        self._lstm_output_tensors = []

        self._lstm_output_tensors.append( self._tcnn( input ) )

        for i in range(len(self._input_slices)):
            ch0 = self._input_slices[i][0]
            ch1 = self._input_slices[i][1]
            slice_size = ch1 - ch0
            input_slice_tensor = input[:, :, ch0:ch1]            
            self._input_slice_tensors.append( input_slice_tensor )
            self._lstm_output_tensors.append( self._lstm[i]( input_slice_tensor ) )
        
        self._lstm_output_tensors = tuple(self._lstm_output_tensors)
        self._lstm_output_tensors = torch.cat( self._lstm_output_tensors, dim=1) 

        self._lstm_output_tensors = torch.reshape( self._lstm_output_tensors, (batch_size, self._mlp_input_size) )
        y = self._mlp.forward( self._lstm_output_tensors )
        y = torch.reshape( y, ( batch_size, self._target_code_width) )

        return y