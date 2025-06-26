
import os
import gc
import time
import numpy as np
import configparser
import json
import copy
import random
import math
import torch
from os import listdir
from os.path import isfile, join
from datetime import date, datetime, timezone
from dateutil import parser
import dearpygui.dearpygui as dpg
import itertools
import threading
from joblib import Parallel, delayed
from tinkoff.invest.constants import INVEST_GRPC_API
from tinkoff.invest import Client
from tinkoff.invest import InstrumentType
from tinkoff.invest import InstrumentIdType
from tinkoff.invest import SecurityTradingStatus
from tinkoff.invest import GetOrderBookResponse, GetLastTradesResponse
from tinkoff.invest import HistoricCandle
from tinkoff.invest.exceptions import RequestError
from TkModules.TkQuotation import quotation_to_float
from TkModules.TkIO import TkIO
from TkModules.TkInstrument import TkInstrument
from TkModules.TkStatistics import TkStatistics
from TkModules.TkUI import TkUI
from TkModules.TkTrainingHistory import TkTimeSeriesTrainingHistory
from TkModules.TkModel import TkModel
from TkModules.TkStackedLSTM import TkStackedLSTM
from TkModules.TkLastTradesAutoencoder import TkLastTradesAutoencoder
from TkModules.TkTimeSeriesForecaster import TkTimeSeriesForecaster

#------------------------------------------------------------------------------------------------------------------------

class TkTimeSeriesDataLoader():

    def __init__(self, _cfg : configparser.ConfigParser, _priority_sample_id = 0, _regular_sample_id = 0, _test_sample_id = 0):
        
        self._data_path = _cfg['Paths']['DataPath']
        self._time_series_index_filename = _cfg['Paths']['TimeSeriesIndexFileName']
        self._time_series_training_data_filename = _cfg['Paths']['TimeSeriesTrainingDataFileName']
        self._time_series_test_data_filename = _cfg['Paths']['TimeSeriesTestDataFileName']
        
        self._training_batch_size = int(_cfg['TimeSeries']['TrainingBatchSize'])
        self._priority_batch_size = int(_cfg['TimeSeries']['PriorityBatchSize'])
        self._test_batch_size = int(_cfg['TimeSeries']['TestBatchSize'])

        self._regular_sample_id = _regular_sample_id
        self._priority_sample_id = _priority_sample_id
        self._test_sample_id = _test_sample_id

        index_content = TkIO.read_at_path( join(self._data_path, self._time_series_index_filename) )

        self._priority_table = index_content[0]
        self._regular_table = index_content[1]
        self._training_index = index_content[2]
        self._test_index = index_content[3]

        priority_sample_count = len(self._priority_table) 
        regular_sample_count = len(self._regular_table)
        test_sample_count = len(self._test_index)
        print( '\nPriority samples:', priority_sample_count, 'Regular samples:', regular_sample_count, ', Test samples:', test_sample_count)
        regular_epoch_size = int(regular_sample_count/(self._training_batch_size-self._priority_batch_size))
        print( 'Regular epoch size:', regular_epoch_size )
        test_epoch_size = int(test_sample_count/regular_epoch_size)
        print( 'Balanced test batch size:', test_epoch_size )

        self._training_data_stream = open( join(self._data_path, self._time_series_training_data_filename), 'rb+')
        self._test_data_stream = open( join(self._data_path, self._time_series_test_data_filename), 'rb+')

        self._input_samples = None
        self._target_code_samples = None
        self._target_true_samples = None
        self._loading_thread = None

    def close(self):
        self._training_data_stream.close()
        self._test_data_stream.close()

    def regular_sample_id(self):
        return self._regular_sample_id

    def priority_sample_id(self):
        return self._priority_sample_id

    def test_sample_id(self):
        return self._test_sample_id

    def get_training_indices(self):
        result = []
        for batchId in range(self._training_batch_size):
            isPrioritySample = batchId < self._priority_batch_size
            if isPrioritySample:
                result.append( self._priority_table[self._priority_sample_id] )
                self._priority_sample_id = self._priority_sample_id + 1
                if self._priority_sample_id >= len(self._priority_table):
                    self._priority_sample_id = 0
            else:
                result.append( self._regular_table[self._regular_sample_id] )
                self._regular_sample_id = self._regular_sample_id + 1
                if self._regular_sample_id >= len(self._regular_table):
                    self._regular_sample_id = 0
        return result

    def get_training_sample(self, idx : int):
        self._training_data_stream.seek( self._training_index[idx], 0 )
        input_sample = TkIO.read_from_file( self._training_data_stream )
        target_code_sample = TkIO.read_from_file( self._training_data_stream )
        target_true_sample = TkIO.read_from_file( self._training_data_stream )
        return input_sample, target_code_sample, target_true_sample
        
    def get_test_sample(self, idx : int):
        self._test_data_stream.seek( self._test_index[idx], 0 )
        input_sample = TkIO.read_from_file( self._test_data_stream )
        target_code_sample = TkIO.read_from_file( self._test_data_stream )
        target_true_sample = TkIO.read_from_file( self._test_data_stream )
        return input_sample, target_code_sample, target_true_sample

    def start_load_training_data(self):
        if self._loading_thread != None:
            raise RuntimeError('Loading thread is active!')

        def load_training_data_thread():
            self._input_samples = [None] * self._training_batch_size
            self._target_code_samples = [None] * self._training_batch_size
            self._target_true_samples = [None] * self._training_batch_size
            indices = self.get_training_indices()
            for batch_id in range(self._training_batch_size):
                input_sample, target_code_sample, target_true_sample = self.get_training_sample( indices[batch_id] )
                self._input_samples[batch_id] = input_sample
                self._target_code_samples[batch_id] = target_code_sample
                self._target_true_samples[batch_id] = target_true_sample

        self._loading_thread = threading.Thread( target=load_training_data_thread )
        self._loading_thread.start()

    def start_load_test_data(self):
        if self._loading_thread != None:
            raise RuntimeError('Loading thread is active!')

        def load_test_data_thread():
            self._input_samples = [None] * self._test_batch_size
            self._target_code_samples = [None] * self._test_batch_size
            self._target_true_samples = [None] * self._test_batch_size
            for batch_id in range(self._test_batch_size):
                input_sample, target_code_sample, target_true_sample = self.get_test_sample( self._test_sample_id )
                self._test_sample_id = self._test_sample_id + 1
                if self._test_sample_id >= len(self._test_index):
                    self._test_sample_id = 0
                self._input_samples[batch_id] = input_sample
                self._target_code_samples[batch_id] = target_code_sample
                self._target_true_samples[batch_id] = target_true_sample

        self._loading_thread = threading.Thread( target=load_test_data_thread )
        self._loading_thread.start()

    def complete_loading(self):
        if self._loading_thread == None:
            raise RuntimeError('Loading thread is not active!')
        self._loading_thread.join()
        self._loading_thread = None
        return self._input_samples, self._target_code_samples, self._target_true_samples
         
#------------------------------------------------------------------------------------------------------------------------

TOKEN = os.environ["TK_TOKEN"]

cuda = torch.device("cuda")

config = configparser.ConfigParser()
config.read( 'TkConfig.ini' )

data_path = config['Paths']['DataPath']
ts_model_path =  join( config['Paths']['ModelsPath'], config['Paths']['TimeSeriesModelFileName'] )
ts_optimizer_path =  join( config['Paths']['ModelsPath'], config['Paths']['TimeSeriesOptimizerFileName'] )
ts_history_path = join( config['Paths']['ModelsPath'], config['Paths']['TimeSeriesTrainingHistoryFileName'] )
lta_model_path = join( config['Paths']['ModelsPath'], config['Paths']['LastTradesAutoencoderModelFileName'] )

prior_steps_count = int(config['TimeSeries']['PriorStepsCount'])
input_width = int(config['TimeSeries']['InputWidth'])
target_code_width = int(config['Autoencoders']['LastTradesAutoencoderCodeLayerSize'])
target_true_width = int(config['Autoencoders']['LastTradesWidth'])
input_slices = json.loads(config['TimeSeries']['InputSlices'])
display_slice = int(config['TimeSeries']['DisplaySlice'])
training_batch_size = int(config['TimeSeries']['TrainingBatchSize'])
test_batch_size = int(config['TimeSeries']['TestBatchSize'])
learning_rate = float(config['TimeSeries']['LearningRate'])
weight_decay = float(config['TimeSeries']['WeightDecay'])
history_size = int( config['TimeSeries']['HistorySize'] )
cooldown = float( config['TimeSeries']['Cooldown'] )

lta_model = TkLastTradesAutoencoder(config)
lta_model.to(cuda)
lta_model.load_state_dict(torch.load(lta_model_path))        
lta_model.eval()

ts_model = TkTimeSeriesForecaster(config)
ts_model.to(cuda)
if os.path.isfile(ts_model_path):
    ts_model.load_state_dict(torch.load(ts_model_path))
ts_optimizer = torch.optim.RAdam( ts_model.parameters(), lr=learning_rate, weight_decay=weight_decay )
if os.path.isfile(ts_optimizer_path):
    ts_optimizer.load_state_dict(torch.load(ts_optimizer_path))
ts_loss = torch.nn.MSELoss()
ts_accuracy = torch.nn.KLDivLoss(reduction = "batchmean", log_target=False)
ts_training_history = TkTimeSeriesTrainingHistory(ts_history_path, history_size)

data_loader = TkTimeSeriesDataLoader(
    config,
    ts_training_history.priority_sample_id(),
    ts_training_history.regular_sample_id(),
    ts_training_history.test_sample_id()
)

with Client(TOKEN, target=INVEST_GRPC_API) as client:

    dpg.create_context()
    dpg.create_viewport(title='Time series training', width=2392, height=936)
    dpg.setup_dearpygui()

    with dpg.window(tag="primary_window", label="Training"):
        for ui_tag in ["training", "test"]:
            with dpg.group(horizontal=True):
                dpg.add_text( default_value = ( "Training" if ui_tag == "training" else "Test" ) )
            with dpg.group(horizontal=True):
                with dpg.plot(label="Input", width=512, height=256):
                    dpg.add_plot_legend()
                    dpg.add_plot_axis(dpg.mvXAxis, tag="x_axis_input_"+ui_tag )
                    dpg.add_plot_axis(dpg.mvYAxis, tag="y_axis_input_"+ui_tag )
                    dpg.add_bar_series( [j for j in range(0, 32)], [random.random() for j in range(0, 32)], label="Input", parent="x_axis_input_"+ui_tag, tag=ui_tag+"_input_series" )
                with dpg.plot(label="Slice", width=512, height=256):
                    dpg.add_plot_legend()
                    dpg.add_plot_axis(dpg.mvXAxis, tag="x_axis_slice_"+ui_tag )
                    dpg.add_plot_axis(dpg.mvYAxis, tag="y_axis_slice_"+ui_tag )
                    dpg.add_bar_series( [j for j in range(0, 32)], [random.random() for j in range(0, 32)], label="Slice", parent="x_axis_slice_"+ui_tag, tag=ui_tag+"_slice_series" )
                with dpg.plot(label="LSTM", width=512, height=256):
                    dpg.add_plot_legend()
                    dpg.add_plot_axis(dpg.mvXAxis, tag="x_axis_lstm_"+ui_tag )
                    dpg.add_plot_axis(dpg.mvYAxis, tag="y_axis_lstm_"+ui_tag )
                    dpg.add_bar_series( [j for j in range(0, 32)], [random.random() for j in range(0, 32)], label="LSTM", parent="x_axis_lstm_"+ui_tag, tag=ui_tag+"_lstm_series" )
                with dpg.plot(label="Code", width=384, height=256):
                    dpg.add_plot_legend()
                    dpg.add_plot_axis(dpg.mvXAxis, tag="x_axis_code_"+ui_tag )
                    dpg.add_plot_axis(dpg.mvYAxis, tag="y_axis_code_"+ui_tag )
                    dpg.add_line_series( [j for j in range(0, 32)], [random.random() for j in range(0, 32)], label="Output", parent="x_axis_code_"+ui_tag, tag=ui_tag+"_code_output_series" )
                    dpg.add_line_series( [j for j in range(0, 32)], [random.random() for j in range(0, 32)], label="Target", parent="x_axis_code_"+ui_tag, tag=ui_tag+"_code_target_series" )
                with dpg.plot(label="True", width=384, height=256):
                    dpg.add_plot_legend()
                    dpg.add_plot_axis(dpg.mvXAxis, tag="x_axis_true_"+ui_tag )
                    dpg.add_plot_axis(dpg.mvYAxis, tag="y_axis_true_"+ui_tag )
                    dpg.add_line_series( [j for j in range(0, 32)], [random.random() for j in range(0, 32)], label="Output", parent="x_axis_true_"+ui_tag, tag=ui_tag+"_true_output_series" )
                    dpg.add_line_series( [j for j in range(0, 32)], [random.random() for j in range(0, 32)], label="Target", parent="x_axis_true_"+ui_tag, tag=ui_tag+"_true_target_series" )
        with dpg.group(horizontal=True):
            with dpg.plot(label="Training", width=512, height=256):
                dpg.add_plot_legend()
                dpg.add_plot_axis(dpg.mvXAxis, tag="x_axis_training" )
                dpg.add_plot_axis(dpg.mvYAxis, tag="y_axis_training" )
                dpg.add_line_series( [j for j in range(0, 32)], [random.random() for j in range(0, 32)], label="Loss", parent="x_axis_training", tag="loss_series" )
            with dpg.plot(label="Training per epoch", width=512, height=256):
                dpg.add_plot_legend()
                dpg.add_plot_axis(dpg.mvXAxis, tag="x_axis_training_epoch" )
                dpg.add_plot_axis(dpg.mvYAxis, tag="y_axis_training_epoch" )
                dpg.add_line_series( [j for j in range(0, 32)], [random.random() for j in range(0, 32)], label="Loss", parent="x_axis_training_epoch", tag="loss_series_epoch" )
            with dpg.plot(label="Accuracy", width=512, height=256):
                dpg.add_plot_legend()
                dpg.add_plot_axis(dpg.mvXAxis, tag="x_axis_accuracy" )
                dpg.add_plot_axis(dpg.mvYAxis, tag="y_axis_accuracy" )
                dpg.add_line_series( [j for j in range(0, 32)], [random.random() for j in range(0, 32)], label="KL-Div", parent="x_axis_accuracy", tag="accuracy_series" )
            with dpg.plot(label="Accuracy per epoch", width=512, height=256):
                dpg.add_plot_legend()
                dpg.add_plot_axis(dpg.mvXAxis, tag="x_axis_accuracy_epoch" )
                dpg.add_plot_axis(dpg.mvYAxis, tag="y_axis_accuracy_epoch" )
                dpg.add_line_series( [j for j in range(0, 32)], [random.random() for j in range(0, 32)], label="KL-Div", parent="x_axis_accuracy_epoch", tag="accuracy_series_epoch" )

    dpg.show_viewport()
    dpg.set_primary_window("primary_window", True)

    data_loader.start_load_training_data()

    show_priority_sample = False

    while dpg.is_dearpygui_running():

        show_priority_sample = not show_priority_sample

        input_samples, target_code_samples, target_true_samples = data_loader.complete_loading()

        input = torch.Tensor( list( itertools.chain.from_iterable(input_samples) ) )
        input = torch.reshape( input, ( training_batch_size, prior_steps_count * input_width) )
        input = input.to(cuda)

        target_code = torch.Tensor( list( itertools.chain.from_iterable(target_code_samples) ) )
        target_code = torch.reshape( target_code, ( training_batch_size, target_code_width ) )
        target_code = target_code.to(cuda)

        target_true = torch.Tensor( list( itertools.chain.from_iterable(target_true_samples) ) )
        target_true = torch.reshape( target_true, ( training_batch_size, target_true_width ) )
        target_true = target_true.to(cuda)

        data_loader.start_load_test_data()

        y = ts_model.forward( input )

        z = lta_model.decode( y )

        display_batch_id = 0 if show_priority_sample else training_batch_size-1
        TkUI.set_series_from_tensor("x_axis_input_training", "y_axis_input_training", "training_input_series", input, display_batch_id)
        TkUI.set_series_from_tensor("x_axis_code_training", "y_axis_code_training", "training_code_output_series", y, display_batch_id)
        TkUI.set_series_from_tensor("x_axis_code_training", "y_axis_code_training", "training_code_target_series", target_code, display_batch_id)
        TkUI.set_series_from_tensor("x_axis_true_training","y_axis_true_training","training_true_output_series", z, display_batch_id)
        TkUI.set_series_from_tensor("x_axis_true_training","y_axis_true_training","training_true_target_series", target_true, display_batch_id)        

        y_loss = ts_loss( y, target_code )
        y_loss = y_loss.mean()
        ts_optimizer.zero_grad()
        y_loss.backward()
        ts_optimizer.step()
        y_loss_val = y_loss.item()

        mlp_input = ts_model.mlp_input()
        TkUI.set_series_from_tensor("x_axis_lstm_training", "y_axis_lstm_training", "training_lstm_series", mlp_input, display_batch_id)

        input_slice_size = ( input_slices[display_slice][1] - input_slices[display_slice][0] ) * prior_steps_count
        input_slice = ts_model.input_slice(display_slice)
        input_slice = torch.reshape( input_slice, ( training_batch_size, input_slice_size ) )
        TkUI.set_series_from_tensor("x_axis_slice_training", "y_axis_slice_training", "training_slice_series", input_slice, display_batch_id)

        dpg.render_dearpygui_frame()

        input_samples, target_code_samples, target_true_samples = data_loader.complete_loading()

        input = torch.Tensor( list( itertools.chain.from_iterable(input_samples) ) )
        input = torch.reshape( input, ( test_batch_size, prior_steps_count * input_width) )
        input = input.to(cuda)

        target_code = torch.Tensor( list( itertools.chain.from_iterable(target_code_samples) ) )
        target_code = torch.reshape( target_code, ( test_batch_size, target_code_width ) )
        target_code = target_code.to(cuda)

        target_true = torch.Tensor( list( itertools.chain.from_iterable(target_true_samples) ) )
        target_true = torch.reshape( target_true, ( test_batch_size, 1, target_true_width ) )
        target_true = target_true.to(cuda)

        data_loader.start_load_training_data() 

        ts_model.train(False)
        y = ts_model.forward( input )
        ts_model.train(True)

        z = lta_model.decode( y )

        display_batch_id = 0 if show_priority_sample else test_batch_size-1
        TkUI.set_series_from_tensor("x_axis_input_test", "y_axis_input_test","test_input_series", input, 0)
        TkUI.set_series_from_tensor("x_axis_code_test", "y_axis_code_test", "test_code_output_series", y, 0)
        TkUI.set_series_from_tensor("x_axis_code_test", "y_axis_code_test", "test_code_target_series", target_code, 0)
        TkUI.set_series_from_tensor("x_axis_true_test","y_axis_true_test","test_true_output_series", z, display_batch_id)
        TkUI.set_series_from_tensor("x_axis_true_test","y_axis_true_test","test_true_target_series", target_true, display_batch_id)        

        input_slice_size = ( input_slices[display_slice][1] - input_slices[display_slice][0] ) * prior_steps_count
        input_slice = ts_model.input_slice(display_slice)
        input_slice = torch.reshape( input_slice, ( test_batch_size, input_slice_size ) )
        TkUI.set_series_from_tensor("x_axis_slice_test", "y_axis_slice_test", "test_slice_series", input_slice, 0)
        
        z_accuracy = ts_accuracy( z, target_true ).detach()
        z_accuracy = z_accuracy.mean()
        z_accuracy_val = z_accuracy.item()

        mlp_input = ts_model.mlp_input()
        TkUI.set_series_from_tensor("x_axis_lstm_test", "y_axis_lstm_test", "test_lstm_series", mlp_input, display_batch_id)

        dpg.render_dearpygui_frame()

        ts_training_history.log(data_loader.priority_sample_id(), data_loader.regular_sample_id(), data_loader.test_sample_id(), y_loss_val, z_accuracy_val)

        TkUI.set_series("x_axis_training", "y_axis_training", "loss_series", ts_training_history.loss_history())
        TkUI.set_series("x_axis_training_epoch", "y_axis_training_epoch", "loss_series_epoch", ts_training_history.epoch_loss_history())
        TkUI.set_series("x_axis_accuracy", "y_axis_accuracy", "accuracy_series", ts_training_history.accuracy_history())
        TkUI.set_series("x_axis_accuracy_epoch", "y_axis_accuracy_epoch", "accuracy_series_epoch", ts_training_history.epoch_accuracy_history())

        cooldownRemaining = cooldown
        cooldownStep = 1.0 / 30.0
        while dpg.is_dearpygui_running() and cooldownRemaining > 0.0:            
            time.sleep(cooldownStep)
            cooldownRemaining -= cooldownStep
            dpg.render_dearpygui_frame()
        

    dpg.destroy_context()

    data_loader.complete_loading()
    data_loader.close()

    ts_training_history.save()
    torch.save( ts_model.state_dict(), ts_model_path )
    torch.save( ts_optimizer.state_dict(), ts_optimizer_path )    

