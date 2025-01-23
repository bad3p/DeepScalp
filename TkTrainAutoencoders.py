
import os
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

#------------------------------------------------------------------------------------------------------------------------

class TkModel(torch.nn.Module):

    def __init__(self, _layer_descriptors : list):
        
        super(TkModel, self).__init__()

        def create_conv_layer(params : list):
            layer_in_channels = params[0]
            layer_out_channels = params[1]
            layer_kernel_size = params[2]
            layer_stride = params[3]
            return torch.nn.Conv1d( in_channels=layer_in_channels, out_channels=layer_out_channels, kernel_size=layer_kernel_size, stride=layer_stride, padding=0)

        def create_deconv_layer(params : list):
            layer_in_channels = params[0]
            layer_out_channels = params[1]
            layer_kernel_size = params[2]
            layer_stride = params[3]
            return torch.nn.ConvTranspose1d( in_channels=layer_in_channels, out_channels=layer_out_channels, kernel_size=layer_kernel_size, stride=layer_stride, padding=0)

        def create_lrelu_layer(params : list):
            layer_leakage = params[0]
            return torch.nn.LeakyReLU(layer_leakage)

        def create_dropout_layer(params : list):
            layer_dropout_prob = params[0]
            return torch.nn.Dropout( p=layer_dropout_prob )

        def create_flatten_layer(params : list):
            return torch.nn.Flatten()

        def create_unflatten_layer(params : list):
            layer_size_x = params[0]
            layer_size_y = params[1]
            return torch.nn.Unflatten(1, (layer_size_x,layer_size_y))

        def create_linear_layer(params : list):
            layer_in_channels = params[0]
            layer_out_channels = params[1]
            return torch.nn.Linear( layer_in_channels, layer_out_channels )

        def create_layer(layer_descriptor : dict):
            if len(layer_descriptor.keys()) == 0:
                raise RuntimeError('Empty layer descriptor!')
            layer_type = list(layer_descriptor.keys())[0]
            if layer_type == 'Conv':
                return create_conv_layer( layer_descriptor[layer_type] )
            if layer_type == 'Deconv':
                return create_deconv_layer( layer_descriptor[layer_type] )
            if layer_type == 'LReLU':
                return create_lrelu_layer( layer_descriptor[layer_type] )
            if layer_type == 'Drop':
                return create_dropout_layer( layer_descriptor[layer_type] )
            if layer_type == 'Flatten':
                return create_flatten_layer( layer_descriptor[layer_type] )
            if layer_type == 'Unflatten':
                return create_unflatten_layer( layer_descriptor[layer_type] )
            if layer_type == 'Linear':
                return create_linear_layer( layer_descriptor[layer_type] )
            else:
                raise RuntimeError('Unknown layer type: ' + layer_type + "!")

        self._layers = []

        for layer_descriptor in _layer_descriptors:
            self._layers.append( create_layer( layer_descriptor ) )

        self._layers = torch.nn.ModuleList( self._layers )

        self.initWeights()

    def initWeights(self) -> None:
        for m in self.modules():
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.ConvTranspose1d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, 0, 0.001)
                torch.nn.init.constant_(m.bias, 0)

    def forward(self, input):

        y = self._layers[0]( input )

        for layer_id in range(1, len(self._layers)):
            y = self._layers[layer_id]( y )
        
        return y

#------------------------------------------------------------------------------------------------------------------------

class TkOrderbookAutoencoder(torch.nn.Module):

    def __init__(self, _cfg : configparser.ConfigParser):

        super(TkOrderbookAutoencoder, self).__init__()

        self._cfg = _cfg
        self._code = None
        self._encoder = TkModel( json.loads(_cfg['Autoencoders']['OrderbookEncoder']) )
        self._decoder = TkModel( json.loads(_cfg['Autoencoders']['OrderbookDecoder']) )

    def code(self):
        return self._code

    def encode(self, input):
        return self._encoder( input )

    def forward(self, input):
        self._code = self._encoder( input )
        return self._decoder( self._code )

#------------------------------------------------------------------------------------------------------------------------

class TkAutoencoderDataLoader():

    def __init__(self, _cfg : configparser.ConfigParser, _orderbook_training_sample_id = 0, _orderbook_test_sample_id = 0, _last_trades_training_sample_id = 0, _last_trades_test_sample_id = 0):
        
        self._data_path = _cfg['Paths']['DataPath']
        self._orderbook_index_filename = _cfg['Paths']['OrderBookIndexFileName']
        self._orderbook_training_data_filename = _cfg['Paths']['OrderBookTrainingDataFileName']
        self._orderbook_test_data_filename = _cfg['Paths']['OrderBookTestDataFileName']

        self._last_trades_index_filename = _cfg['Paths']['LastTradesIndexFileName']
        self._last_trades_training_data_filename = _cfg['Paths']['LastTradesTrainingDataFileName']
        self._last_trades_test_data_filename = _cfg['Paths']['LastTradesTestDataFileName']
        
        self._sample_scale = json.loads(_cfg['Autoencoders']['SampleScale'])
        self._training_batch_size = int(_cfg['Autoencoders']['TrainingBatchSize'])
        self._test_batch_size = int(_cfg['Autoencoders']['TestBatchSize'])

        orderbook_index_content = TkIO.read_at_path( join(self._data_path, self._orderbook_index_filename) )
        self._orderbook_training_index = orderbook_index_content[0]
        self._orderbook_test_index = orderbook_index_content[1]
        self._orderbook_training_sample_id = _orderbook_training_sample_id
        self._orderbook_test_sample_id = _orderbook_test_sample_id
        self._orderbook_training_data_stream = open( join(self._data_path, self._orderbook_training_data_filename), 'rb+')
        self._orderbook_test_data_stream = open( join(self._data_path, self._orderbook_test_data_filename), 'rb+')

        last_trades_index_content = TkIO.read_at_path( join(self._data_path, self._last_trades_index_filename) )
        self._last_trades_training_index = last_trades_index_content[0]
        self._last_trades_test_index = last_trades_index_content[1]
        self._last_trades_training_sample_id = _last_trades_training_sample_id
        self._last_trades_test_sample_id = _last_trades_test_sample_id
        self._last_trades_training_data_stream = open( join(self._data_path, self._last_trades_training_data_filename), 'rb+')
        self._last_trades_test_data_stream = open( join(self._data_path, self._last_trades_test_data_filename), 'rb+')

        self._orderbook_samples = None
        self._last_trades_samples = None
        self._loading_thread = None

    def close(self):
        self._orderbook_training_data_stream.close()
        self._orderbook_test_data_stream.close()
        self._last_trades_training_data_stream.close()
        self._last_trades_test_data_stream.close()

    def orderbook_training_sample_id(self):
        return self._orderbook_training_sample_id

    def orderbook_test_sample_id(self):
        return self._orderbook_test_sample_id

    def last_trades_training_sample_id(self):
        return self._last_trades_training_sample_id
    
    def last_trades_test_sample_id(self):
        return self._last_trades_test_sample_id

    @staticmethod
    def load_sample(id : int, index : list, stream):
        stream.seek( index[id], 0 )
        sample = TkIO.read_from_file( stream )
        id = id + 1
        if id >= len(index): 
            id = 0
        return id, sample

    def get_sample_scale(self):
        weight = random.uniform( 0.0, 1.0 )
        cumulative_weight = 0.0
        i = 0
        while i < len(self._sample_scale)-1 and weight > self._sample_scale[i][0] + cumulative_weight:
            cumulative_weight = cumulative_weight + self._sample_scale[i][0]
            i = i + 1
        return random.uniform( self._sample_scale[i][1], self._sample_scale[i][2] )

    def get_orderbook_training_sample(self):
        self._orderbook_training_sample_id, orderbook_sample = TkAutoencoderDataLoader.load_sample(
            self._orderbook_training_sample_id,
            self._orderbook_training_index,
            self._orderbook_training_data_stream
        )
        orderbook_sample *= self.get_sample_scale()
        return orderbook_sample

    def get_orderbook_test_sample(self):
        self._orderbook_test_sample_id, orderbook_sample = TkAutoencoderDataLoader.load_sample(
            self._orderbook_test_sample_id,
            self._orderbook_test_index,
            self._orderbook_test_data_stream
        )
        orderbook_sample *= self.get_sample_scale()
        return orderbook_sample

    def get_last_trades_training_sample(self):
        self._last_trades_training_sample_id, last_trades_sample = TkAutoencoderDataLoader.load_sample(
            self._last_trades_training_sample_id,
            self._last_trades_training_index,
            self._last_trades_training_data_stream
        )
        last_trades_sample *= self.get_sample_scale()
        return last_trades_sample

    def get_last_trades_test_sample(self):
        self._last_trades_test_sample_id, last_trades_sample = TkAutoencoderDataLoader.load_sample(
            self._last_trades_test_sample_id,
            self._last_trades_test_index,
            self._last_trades_test_data_stream
        )
        last_trades_sample *= self.get_sample_scale()
        return last_trades_sample

    def start_load_training_data(self):
        if self._loading_thread != None:
            raise RuntimeError('Loading thread is active!')

        def load_training_data_thread():
            self._orderbook_samples = [None] * self._training_batch_size
            self._last_trades_samples = [None] * self._training_batch_size
            for batch_id in range(self._training_batch_size):
                self._orderbook_samples[batch_id] = self.get_orderbook_training_sample()
                self._last_trades_samples[batch_id] = self.get_last_trades_training_sample()

        self._loading_thread = threading.Thread( target=load_training_data_thread )
        self._loading_thread.start()

    def start_load_test_data(self):
        if self._loading_thread != None:
            raise RuntimeError('Loading thread is active!')

        def load_test_data_thread():
            self._orderbook_samples = [None] * self._test_batch_size
            self._last_trades_samples = [None] * self._test_batch_size
            for batch_id in range(self._test_batch_size):
                self._orderbook_samples[batch_id] = self.get_orderbook_test_sample()
                self._last_trades_samples[batch_id] = self.get_last_trades_test_sample()

        self._loading_thread = threading.Thread( target=load_test_data_thread )
        self._loading_thread.start()

    def complete_loading(self):
        if self._loading_thread == None:
            raise RuntimeError('Loading thread is not active!')
        self._loading_thread.join()
        self._loading_thread = None
        return self._orderbook_samples, self._last_trades_samples

#------------------------------------------------------------------------------------------------------------------------

class TkTrainingHistory():

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

TOKEN = os.environ["TK_TOKEN"]

cuda = torch.device("cuda")

config = configparser.ConfigParser()
config.read( 'TkConfig.ini' )

data_path = config['Paths']['DataPath']
orderbook_model_path =  join( config['Paths']['ModelsPath'], config['Paths']['OrderbookAutoencoderModelFileName'] )
orderbook_optimizer_path =  join( config['Paths']['ModelsPath'], config['Paths']['OrderbookAutoencoderOptimizerFileName'] )
orderbook_history_path = join( config['Paths']['ModelsPath'], config['Paths']['OrderbookAutoencoderTrainingHistoryFileName'] )

training_batch_size = int(config['Autoencoders']['TrainingBatchSize'])
test_batch_size = int(config['Autoencoders']['TestBatchSize'])
orderbook_width = int(config['Autoencoders']['OrderBookWidth'])
last_trades_width = int(config['Autoencoders']['LastTradesWidth'])
orderbook_code_layer_size = int(config['Autoencoders']['OrderbookAutoencoderCodeLayerSize'])
learning_rate = float(config['Autoencoders']['LearningRate'])
weight_decay = float(config['Autoencoders']['WeightDecay'])
history_size = int( config['Autoencoders']['HistorySize'] )

orderbook_autoencoder = TkOrderbookAutoencoder(config)
orderbook_autoencoder.to(cuda)
if os.path.isfile(orderbook_model_path):
    orderbook_autoencoder.load_state_dict(torch.load(orderbook_model_path))
orderbook_optimizer = torch.optim.RAdam( orderbook_autoencoder.parameters(), lr=learning_rate, weight_decay=weight_decay )
if os.path.isfile(orderbook_optimizer_path):
    orderbook_optimizer.load_state_dict(torch.load(orderbook_optimizer_path))
orderbook_loss = torch.nn.MSELoss()
orderbook_accuracy = torch.nn.MSELoss()
orderbook_training_history = TkTrainingHistory(orderbook_history_path, history_size)

data_loader = TkAutoencoderDataLoader(
    config,
    orderbook_training_history.training_sample_id(),
    orderbook_training_history.test_sample_id(),
    0,
    0
)

with Client(TOKEN, target=INVEST_GRPC_API) as client:

    dpg.create_context()
    dpg.create_viewport(title='Autoencoder training', width=1572, height=768)
    dpg.setup_dearpygui()

    with dpg.window(tag="primary_window", label="Preprocess data"):
        with dpg.group(horizontal=True):
            with dpg.plot(label="Orderbook input", width=512, height=256):
                dpg.add_plot_legend()
                dpg.add_plot_axis(dpg.mvXAxis, tag="x_axis_orderbook" )
                dpg.add_plot_axis(dpg.mvYAxis, tag="y_axis_orderbook" )
                dpg.add_bar_series( [j for j in range(0, 32)], [random.random() for j in range(0, 32)], label="Orderbook input", parent="x_axis_orderbook", tag="orderbook_series" )
            with dpg.plot(label="Code layer", width=256, height=256):
                dpg.add_plot_legend()
                dpg.add_plot_axis(dpg.mvXAxis, tag="x_axis_orderbook_code" )
                dpg.add_plot_axis(dpg.mvYAxis, tag="y_axis_orderbook_code" )
                dpg.add_bar_series( [j for j in range(0, 32)], [random.random() for j in range(0, 32)], label="Code layer", parent="x_axis_orderbook_code", tag="orderbook_code_series" )
            with dpg.plot(label="Orderbook output", width=512, height=256):
                dpg.add_plot_legend()
                dpg.add_plot_axis(dpg.mvXAxis, tag="x_axis_orderbook_output" )
                dpg.add_plot_axis(dpg.mvYAxis, tag="y_axis_orderbook_output" )
                dpg.add_bar_series( [j for j in range(0, 32)], [random.random() for j in range(0, 32)], label="Orderbook output", parent="x_axis_orderbook_output", tag="orderbook_output_series" )
        with dpg.group(horizontal=True):
            with dpg.plot(label="Orderbook training", width=512, height=256):
                dpg.add_plot_legend()
                dpg.add_plot_axis(dpg.mvXAxis, tag="x_axis_orderbook_training" )
                dpg.add_plot_axis(dpg.mvYAxis, tag="y_axis_orderbook_training" )
                dpg.add_line_series( [j for j in range(0, 32)], [random.random() for j in range(0, 32)], label="Loss", parent="x_axis_orderbook_training", tag="orderbook_loss_series" )
                dpg.add_line_series( [j for j in range(0, 32)], [random.random() for j in range(0, 32)], label="Accuracy", parent="x_axis_orderbook_training", tag="orderbook_accuracy_series" )
            with dpg.plot(label="Orderbook training per epoch", width=512, height=256):
                dpg.add_plot_legend()
                dpg.add_plot_axis(dpg.mvXAxis, tag="x_axis_orderbook_training_epoch" )
                dpg.add_plot_axis(dpg.mvYAxis, tag="y_axis_orderbook_training_epoch" )
                dpg.add_line_series( [j for j in range(0, 32)], [random.random() for j in range(0, 32)], label="Loss", parent="x_axis_orderbook_training_epoch", tag="orderbook_loss_series_epoch" )
                dpg.add_line_series( [j for j in range(0, 32)], [random.random() for j in range(0, 32)], label="Accuracy", parent="x_axis_orderbook_training_epoch", tag="orderbook_accuracy_series_epoch" )

    dpg.show_viewport()
    dpg.set_primary_window("primary_window", True)

    data_loader.start_load_training_data()

    while dpg.is_dearpygui_running():

        orderbook_samples, last_trades_samples = data_loader.complete_loading()

        orderbook_input = torch.Tensor( list( itertools.chain.from_iterable(orderbook_samples) ) )
        orderbook_input = torch.reshape( orderbook_input, ( training_batch_size, 1, orderbook_width) )
        orderbook_input = orderbook_input.to(cuda)

        data_loader.start_load_test_data()

        y = orderbook_autoencoder.forward( orderbook_input )

        TkUI.set_series_from_tensor("x_axis_orderbook", "y_axis_orderbook","orderbook_series",orderbook_input,0)
        TkUI.set_series_from_tensor("x_axis_orderbook_code", "y_axis_orderbook_code","orderbook_code_series",orderbook_autoencoder.code(),0)
        TkUI.set_series_from_tensor("x_axis_orderbook_output", "y_axis_orderbook_output","orderbook_output_series",y,0)

        loss = orderbook_loss( y, orderbook_input )
        loss = loss.mean()
        orderbook_optimizer.zero_grad()
        loss.backward()
        orderbook_optimizer.step()
        loss_val = loss.item()

        dpg.render_dearpygui_frame()

        orderbook_samples, last_trades_samples = data_loader.complete_loading()

        orderbook_input = torch.Tensor( list( itertools.chain.from_iterable(orderbook_samples) ) )
        orderbook_input = torch.reshape( orderbook_input, ( test_batch_size, 1, orderbook_width) )
        orderbook_input = orderbook_input.to(cuda)

        data_loader.start_load_training_data()

        orderbook_autoencoder.train(False)
        y = orderbook_autoencoder.forward( orderbook_input )
        orderbook_autoencoder.train(True)

        TkUI.set_series_from_tensor("x_axis_orderbook", "y_axis_orderbook","orderbook_series",orderbook_input,0)
        TkUI.set_series_from_tensor("x_axis_orderbook_code", "y_axis_orderbook_code","orderbook_code_series",orderbook_autoencoder.code(),0)
        TkUI.set_series_from_tensor("x_axis_orderbook_output", "y_axis_orderbook_output","orderbook_output_series",y,0)

        accuracy = orderbook_accuracy( y, orderbook_input )
        accuracy = accuracy.mean()
        accuracy_val = accuracy.item()

        dpg.render_dearpygui_frame()

        orderbook_training_history.log(data_loader.orderbook_training_sample_id(), data_loader.orderbook_test_sample_id(), loss_val, accuracy_val)

        TkUI.set_series("x_axis_orderbook_training", "y_axis_orderbook_training", "orderbook_loss_series", orderbook_training_history.loss_history())
        TkUI.set_series("x_axis_orderbook_training_epoch", "y_axis_orderbook_training_epoch", "orderbook_loss_series_epoch", orderbook_training_history.epoch_loss_history())

        TkUI.set_series("x_axis_orderbook_training", "y_axis_orderbook_training", "orderbook_accuracy_series", orderbook_training_history.accuracy_history())
        TkUI.set_series("x_axis_orderbook_training_epoch", "y_axis_orderbook_training_epoch", "orderbook_accuracy_series_epoch", orderbook_training_history.epoch_accuracy_history())
        

    dpg.destroy_context()

    data_loader.complete_loading()
    data_loader.close()

    orderbook_training_history.save()
    torch.save( orderbook_autoencoder.state_dict(), orderbook_model_path )
    torch.save( orderbook_optimizer.state_dict(), orderbook_optimizer_path )    