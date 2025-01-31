
import os
import math
from TkModules.TkIO import TkIO

#------------------------------------------------------------------------------------------------------------------------

class TkAutoencoderTrainingHistory():

    def __init__(self, _history_path:str, _history_size:int):
        self._history_path = _history_path
        self._history_size = _history_size

        if os.path.isfile(self._history_path):
            file_content = TkIO.read_at_path(self._history_path)
            self._training_sample_id = file_content[0]
            self._test_sample_id = file_content[1]
            self._loss_history = file_content[2]
            self._accuracy_history = file_content[3]
            self._epoch_loss_history = file_content[4]
            self._epoch_accuracy_history = file_content[5]
        else:
            self._training_sample_id = 0
            self._test_sample_id = 0
            self._loss_history = []
            self._accuracy_history = []
            self._epoch_loss_history = [(0.0,0,0)]
            self._epoch_accuracy_history = [(0.0,0,0)]

    def training_sample_id(self):
        return self._training_sample_id

    def test_sample_id(self):
        return self._test_sample_id

    def loss_history(self):
        return self._loss_history

    def epoch_loss_history(self):
        return [self._epoch_loss_history[i][0] for i in range(0, len(self._epoch_loss_history))]

    def accuracy_history(self):
        return self._accuracy_history

    def epoch_accuracy_history(self):
        return [self._epoch_accuracy_history[i][0] for i in range(0, len(self._epoch_accuracy_history))]

    def save(self):
        TkIO.write_at_path(self._history_path, self._training_sample_id)
        TkIO.append_at_path(self._history_path, self._test_sample_id)
        TkIO.append_at_path(self._history_path, self._loss_history)
        TkIO.append_at_path(self._history_path,self._accuracy_history)
        TkIO.append_at_path(self._history_path,self._epoch_loss_history)
        TkIO.append_at_path(self._history_path,self._epoch_accuracy_history)

    def log(self, training_sample_id:int, test_sample_id:int, loss:float, accuracy:float):

        def accumulate_epoch_data(epoch_data:list, value:float, is_end_of_epoch:bool):
            if math.isnan(value) or math.isinf(value):
                return
            if is_end_of_epoch:
                epoch_data.append( (value, 1) )
            else:
                prev_avg = epoch_data[-1][0]
                prev_avg_norm = epoch_data[-1][1]
                next_avg_norm = prev_avg_norm + 1
                next_avg = (prev_avg * prev_avg_norm + value) / next_avg_norm
                epoch_data[-1] = (next_avg, next_avg_norm)
        
        is_end_of_training_epoch = training_sample_id < self._training_sample_id
        is_end_of_test_epoch = test_sample_id < self._test_sample_id
        self._training_sample_id = training_sample_id
        self._test_sample_id = test_sample_id

        self._loss_history.append( loss )
        if len(self._loss_history) > self._history_size:
            del self._loss_history[0]

        self._accuracy_history.append( accuracy )
        if len(self._accuracy_history) > self._history_size:
            del self._accuracy_history[0]

        accumulate_epoch_data( self._epoch_loss_history, loss, is_end_of_training_epoch )
        accumulate_epoch_data( self._epoch_accuracy_history, accuracy, is_end_of_test_epoch )

#------------------------------------------------------------------------------------------------------------------------

class TkTimeSeriesTrainingHistory():

    def __init__(self, _history_path:str, _history_size:int):
        self._history_path = _history_path
        self._history_size = _history_size

        if os.path.isfile(self._history_path):
            file_content = TkIO.read_at_path(self._history_path)
            self._priority_sample_id = file_content[0]
            self._regular_sample_id = file_content[1]
            self._test_sample_id = file_content[2]
            self._loss_history = file_content[3]
            self._accuracy_history = file_content[4]
            self._epoch_loss_history = file_content[5]
            self._epoch_accuracy_history = file_content[6]
        else:
            self._priority_sample_id = 0
            self._regular_sample_id = 0
            self._test_sample_id = 0
            self._loss_history = []
            self._accuracy_history = []
            self._epoch_loss_history = [(0.0,0,0)]
            self._epoch_accuracy_history = [(0.0,0,0)]

    def priority_sample_id(self):
        return self._priority_sample_id

    def regular_sample_id(self):
        return self._regular_sample_id

    def test_sample_id(self):
        return self._test_sample_id

    def loss_history(self):
        return self._loss_history

    def epoch_loss_history(self):
        return [self._epoch_loss_history[i][0] for i in range(0, len(self._epoch_loss_history))]

    def accuracy_history(self):
        return self._accuracy_history

    def epoch_accuracy_history(self):
        return [self._epoch_accuracy_history[i][0] for i in range(0, len(self._epoch_accuracy_history))]

    def save(self):
        TkIO.write_at_path(self._history_path, self._priority_sample_id)
        TkIO.append_at_path(self._history_path, self._regular_sample_id)
        TkIO.append_at_path(self._history_path, self._test_sample_id)
        TkIO.append_at_path(self._history_path, self._loss_history)
        TkIO.append_at_path(self._history_path,self._accuracy_history)
        TkIO.append_at_path(self._history_path,self._epoch_loss_history)
        TkIO.append_at_path(self._history_path,self._epoch_accuracy_history)

    def log(self, priority_sample_id:int, regular_sample_id:int, test_sample_id:int, loss:float, accuracy:float):

        def accumulate_epoch_data(epoch_data:list, value:float, is_end_of_epoch:bool):
            if math.isnan(value) or math.isinf(value):
                return
            if is_end_of_epoch:
                epoch_data.append( (value, 1) )
            else:
                prev_avg = epoch_data[-1][0]
                prev_avg_norm = epoch_data[-1][1]
                next_avg_norm = prev_avg_norm + 1
                next_avg = (prev_avg * prev_avg_norm + value) / next_avg_norm
                epoch_data[-1] = (next_avg, next_avg_norm)
        
        is_end_of_training_epoch = regular_sample_id < self._regular_sample_id
        is_end_of_test_epoch = test_sample_id < self._test_sample_id
        self._priority_sample_id = priority_sample_id
        self._regular_sample_id = regular_sample_id
        self._test_sample_id = test_sample_id

        self._loss_history.append( loss )
        if len(self._loss_history) > self._history_size:
            del self._loss_history[0]

        self._accuracy_history.append( accuracy )
        if len(self._accuracy_history) > self._history_size:
            del self._accuracy_history[0]

        accumulate_epoch_data( self._epoch_loss_history, loss, is_end_of_training_epoch )
        accumulate_epoch_data( self._epoch_accuracy_history, accuracy, is_end_of_test_epoch )