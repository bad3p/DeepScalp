
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

    def __init__(self, _cfg : configparser.ConfigParser):
        
        self._data_path = config['Paths']['DataPath']
        self._orderbook_index_filename = config['Paths']['OrderBookIndexFileName']
        self._orderbook_training_data_filename = config['Paths']['OrderBookTrainingDataFileName']
        self._orderbook_test_data_filename = config['Paths']['OrderBookTestDataFileName']

        self._last_trades_index_filename = config['Paths']['LastTradesIndexFileName']
        self._last_trades_training_data_filename = config['Paths']['LastTradesTrainingDataFileName']
        self._last_trades_test_data_filename = config['Paths']['LastTradesTestDataFileName']

        self._training_batch_size = int(config['Autoencoders']['TrainingBatchSize'])
        self._test_batch_size = int(config['Autoencoders']['TestBatchSize'])

        orderbook_index_content = TkIO.read_at_path( join(self._data_path, self._orderbook_index_filename) )
        self._orderbook_training_index = orderbook_index_content[0]
        self._orderbook_test_index = orderbook_index_content[1]
        self._orderbook_training_sample_id = 0
        self._orderbook_test_sample_id = 0
        self._orderbook_training_data_stream = open( join(self._data_path, self._orderbook_training_data_filename), 'rb+')
        self._orderbook_test_data_stream = open( join(self._data_path, self._orderbook_test_data_filename), 'rb+')

        last_trades_index_content = TkIO.read_at_path( join(self._data_path, self._last_trades_index_filename) )
        self._last_trades_training_index = last_trades_index_content[0]
        self._last_trades_test_index = last_trades_index_content[1]
        self._last_trades_training_sample_id = 0
        self._last_trades_test_sample_id = 0
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

    def load_sample(self, id : int, index : list, stream):
        stream.seek( index[id], 0 )
        sample = TkIO.read_from_file( stream )
        id = id + 1
        if id >= len(index) : 
            id = 0
        return id, sample

    def get_orderbook_training_sample(self):
        self._orderbook_training_data_stream.seek( self._orderbook_training_index[self._orderbook_training_sample_id], 0 )
        orderbook_sample = TkIO.read_from_file( self._orderbook_training_data_stream )
        orderbook_sample *= random.uniform( 0.1, 10.0 )
        self._orderbook_training_sample_id = self._orderbook_training_sample_id + 1
        if self._orderbook_training_sample_id >= len(self._orderbook_training_index) : 
            self._orderbook_training_sample_id = 0
        return orderbook_sample

    def get_orderbook_test_sample(self):
        self._orderbook_test_data_stream.seek( self._orderbook_test_index[self._orderbook_test_sample_id], 0 )
        orderbook_sample = TkIO.read_from_file( self._orderbook_test_data_stream )
        orderbook_sample *= random.uniform( 0.1, 10.0 )
        self._orderbook_test_sample_id = self._orderbook_test_sample_id + 1
        if self._orderbook_test_sample_id >= len(self._orderbook_test_index) : 
            self._orderbook_test_sample_id = 0
        return orderbook_sample

    def get_last_trades_training_sample(self):
        self._last_trades_training_data_stream.seek( self._last_trades_training_index[self._last_trades_training_sample_id], 0 )
        last_trades_sample = TkIO.read_from_file( self._last_trades_training_data_stream )
        self._last_trades_training_sample_id = self._last_trades_training_sample_id + 1
        if self._last_trades_training_sample_id >= len(self._last_trades_training_index) : 
            self._last_trades_training_sample_id = 0
        return last_trades_sample

    def get_last_trades_test_sample(self):
        self._last_trades_test_data_stream.seek( self._last_trades_test_index[self._last_trades_test_sample_id], 0 )
        last_trades_sample = TkIO.read_from_file( self._last_trades_test_data_stream )
        self._last_trades_test_sample_id = self._last_trades_test_sample_id + 1
        if self._last_trades_test_sample_id >= len(self._last_trades_test_index) : 
            self._last_trades_test_sample_id = 0
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
       
#------------------------------------------------------------------------------------------------------------------------

TOKEN = os.environ["TK_TOKEN"]

cuda = torch.device("cuda")

config = configparser.ConfigParser()
config.read( 'TkConfig.ini' )

data_path = config['Paths']['DataPath']
training_batch_size = int(config['Autoencoders']['TrainingBatchSize'])
test_batch_size = int(config['Autoencoders']['TestBatchSize'])
orderbook_width = int(config['Autoencoders']['OrderBookWidth'])
last_trades_width = int(config['Autoencoders']['LastTradesWidth'])
orderbook_code_layer_size = int(config['Autoencoders']['OrderbookAutoencoderCodeLayerSize'])
learning_rate = float(config['Autoencoders']['LearningRate'])

orderbook_autoencoder = TkOrderbookAutoencoder(config)
orderbook_autoencoder.to(cuda)
orderbook_optimizer = torch.optim.Adam( orderbook_autoencoder.parameters(), lr=learning_rate )
orderbook_loss = torch.nn.MSELoss()

data_loader = TkAutoencoderDataLoader(config)

with Client(TOKEN, target=INVEST_GRPC_API) as client:

    dpg.create_context()
    dpg.create_viewport()
    dpg.setup_dearpygui()

    with dpg.window(tag="primary_window", label="Preprocess data"):
        with dpg.group(horizontal=True):
            with dpg.plot(label="Orderbook input", width=512, height=256):
                dpg.add_plot_legend()
                dpg.add_plot_axis(dpg.mvXAxis, tag="x_axis_orderbook" )
                dpg.add_plot_axis(dpg.mvYAxis, tag="y_axis_orderbook" )
                dpg.add_line_series( [j for j in range(0, 32)], [random.random() for j in range(0, 32)], label="Orderbook input", parent="x_axis_orderbook", tag="orderbook_series" )
            with dpg.plot(label="Code layer", width=256, height=256):
                dpg.add_plot_legend()
                dpg.add_plot_axis(dpg.mvXAxis, tag="x_axis_orderbook_code" )
                dpg.add_plot_axis(dpg.mvYAxis, tag="y_axis_orderbook_code" )
                dpg.add_line_series( [j for j in range(0, 32)], [random.random() for j in range(0, 32)], label="Code layer", parent="x_axis_orderbook_code", tag="orderbook_code_series" )
            with dpg.plot(label="Orderbook output", width=512, height=256):
                dpg.add_plot_legend()
                dpg.add_plot_axis(dpg.mvXAxis, tag="x_axis_orderbook_output" )
                dpg.add_plot_axis(dpg.mvYAxis, tag="y_axis_orderbook_output" )
                dpg.add_line_series( [j for j in range(0, 32)], [random.random() for j in range(0, 32)], label="Orderbook output", parent="x_axis_orderbook_output", tag="orderbook_output_series" )

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
        orderbook_optimizer.zero_grad()
        loss.backward()
        orderbook_optimizer.param_groups[0]['lr'] = learning_rate
        orderbook_optimizer.step()

        dpg.render_dearpygui_frame()

        data_loader.complete_loading()

        data_loader.start_load_training_data()

        dpg.render_dearpygui_frame()

    dpg.destroy_context()

    data_loader.complete_loading()
    data_loader.close()