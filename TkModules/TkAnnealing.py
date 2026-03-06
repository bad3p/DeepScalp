
import torch
import json

#------------------------------------------------------------------------------------------------------------------------
# Annealing controller
# {"Constant":[const_val]}
# {"Annealing":[start_epoch, end_epoch, start_val, end_val]} 
# {"ComplexAnnealing":[[epoch0,val0], [epoch1,val1], ... [epochN,valN]]} 
#------------------------------------------------------------------------------------------------------------------------

class TkAnnealing():

    def __init__(self, json_payload_str:str):

        params = json.loads(json_payload_str)

        if len(params.keys()) == 0:
            raise RuntimeError('Empty params!')
        
        self._annealing_type = list(params.keys())[0]

        self._start_epoch = 0.0
        self._end_epoch = 1.0
        self._start_value = 0.0
        self._end_value = 0.0
        self._epoch_values = []

        if self._annealing_type == 'Constant':
            args = params[self._annealing_type]
            self._start_value = args[0]
            self._end_value = args[0]
        elif self._annealing_type == 'Annealing':
            args = params[self._annealing_type]
            self._start_epoch = args[0]
            self._end_epoch = args[1]
            self._start_value = args[2]
            self._end_value = args[3]
        elif self._annealing_type == 'ComplexAnnealing':
            self._epoch_values = params[self._annealing_type]
        else:
            raise RuntimeError('Unknown annealing type: ' + annealing_type + "!")
            
    def get_value(self, smooth_epoch:float):

        if self._annealing_type == 'ComplexAnnealing':
            if smooth_epoch < self._epoch_values[0][0]:
                return self._epoch_values[0][1]
            elif smooth_epoch > self._epoch_values[-1][0]:
                return self._epoch_values[-1][1]
            else:
                for i in range(len(self._epoch_values)-1):
                    if smooth_epoch >= self._epoch_values[i][0] and smooth_epoch <= self._epoch_values[i+1][0]:
                        t = ( smooth_epoch - self._epoch_values[i][0] ) / (self._epoch_values[i+1][0] - self._epoch_values[i][0])
                        t = max( 0.0, min(1.0, t) )
                        return self._epoch_values[i][1] * ( 1.0 - t ) + self._epoch_values[i+1][1] * t
                return self._epoch_values[-1][1]

        else:

            if smooth_epoch < self._start_epoch:
                return self._start_value
            elif smooth_epoch > self._end_epoch:
                return self._end_value
            else:
                t = ( smooth_epoch - self._start_epoch ) / (self._end_epoch - self._start_epoch)
                t = max( 0.0, min(1.0, t) )
                return self._start_value * ( 1.0 - t ) + self._end_value * t
        
