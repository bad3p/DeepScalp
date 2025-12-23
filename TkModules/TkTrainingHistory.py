
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
            self._recon_loss_history = file_content[2]
            self._kld_loss_history = file_content[3]
            self._recon_accuracy_history = file_content[4]
            self._kld_accuracy_history = file_content[5]
            self._epoch_recon_loss_history = file_content[6]
            self._epoch_kld_loss_history = file_content[7]
            self._epoch_recon_accuracy_history = file_content[8]
            self._epoch_kld_accuracy_history = file_content[9]
        else:
            self._training_sample_id = 0
            self._test_sample_id = 0
            self._recon_loss_history = []
            self._kld_loss_history = []
            self._recon_accuracy_history = []
            self._kld_accuracy_history = []
            self._epoch_recon_loss_history = [(0.0,0,0)]
            self._epoch_kld_loss_history = [(0.0,0,0)]
            self._epoch_recon_accuracy_history = [(0.0,0,0)]
            self._epoch_kld_accuracy_history = [(0.0,0,0)]            

        print( _history_path, self._epoch_recon_accuracy_history[-1], self._epoch_kld_accuracy_history[-1])

    def get_smooth_epoch(self,epoch_size:int):
        return len(self._epoch_recon_loss_history) - 1 + self._training_sample_id / epoch_size

    def training_sample_id(self):
        return self._training_sample_id

    def test_sample_id(self):
        return self._test_sample_id

    def recon_loss_history(self):
        return self._recon_loss_history

    def epoch_recon_loss_history(self):
        return [self._epoch_recon_loss_history[i][0] for i in range(0, len(self._epoch_recon_loss_history))]

    def recon_accuracy_history(self):
        return self._recon_accuracy_history

    def epoch_recon_accuracy_history(self):
        return [self._epoch_recon_accuracy_history[i][0] for i in range(0, len(self._epoch_recon_accuracy_history))]
    
    def kld_loss_history(self):
        return self._kld_loss_history

    def epoch_kld_loss_history(self):
        return [self._epoch_kld_loss_history[i][0] for i in range(0, len(self._epoch_kld_loss_history))]

    def kld_accuracy_history(self):
        return self._kld_accuracy_history

    def epoch_kld_accuracy_history(self):
        return [self._epoch_kld_accuracy_history[i][0] for i in range(0, len(self._epoch_kld_accuracy_history))]

    def save(self):
        TkIO.write_at_path(self._history_path, self._training_sample_id)
        TkIO.append_at_path(self._history_path, self._test_sample_id)
        TkIO.append_at_path(self._history_path, self._recon_loss_history)
        TkIO.append_at_path(self._history_path, self._kld_loss_history)
        TkIO.append_at_path(self._history_path, self._recon_accuracy_history)
        TkIO.append_at_path(self._history_path, self._kld_accuracy_history)
        TkIO.append_at_path(self._history_path, self._epoch_recon_loss_history)
        TkIO.append_at_path(self._history_path, self._epoch_kld_loss_history)
        TkIO.append_at_path(self._history_path, self._epoch_recon_accuracy_history)
        TkIO.append_at_path(self._history_path, self._epoch_kld_accuracy_history)

    def log(self, training_sample_id:int, test_sample_id:int, recon_loss:float, recon_accuracy:float, kld_loss:float, kld_accuracy:float):

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

        self._recon_loss_history.append( recon_loss )
        if len(self._recon_loss_history) > self._history_size:
            del self._recon_loss_history[0]

        self._recon_accuracy_history.append( recon_accuracy )
        if len(self._recon_accuracy_history) > self._history_size:
            del self._recon_accuracy_history[0]

        accumulate_epoch_data( self._epoch_recon_loss_history, recon_loss, is_end_of_training_epoch )
        accumulate_epoch_data( self._epoch_recon_accuracy_history, recon_accuracy, is_end_of_test_epoch )

        # KL

        self._kld_loss_history.append( kld_loss )
        if len(self._kld_loss_history) > self._history_size:
            del self._kld_loss_history[0]

        self._kld_accuracy_history.append( kld_accuracy )
        if len(self._kld_accuracy_history) > self._history_size:
            del self._kld_accuracy_history[0]

        accumulate_epoch_data( self._epoch_kld_loss_history, kld_loss, is_end_of_training_epoch )
        accumulate_epoch_data( self._epoch_kld_accuracy_history, kld_accuracy, is_end_of_test_epoch )

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