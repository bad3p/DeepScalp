
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
from TkModules.TkOrderbookAutoencoder import TkOrderbookAutoencoder
from TkModules.TkLastTradesAutoencoder import TkLastTradesAutoencoder
from TkModules.TkTrainingHistory import TkAutoencoderTrainingHistory
from TkModules.TkSSIM import hybrid_lob_multi_loss, hybrid_lob_loss, hybrid_ms_ssim_1d_gaussian_l1_loss

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
        
        self._max_training_batch_size = int(_cfg['Autoencoders']['MaxTrainingBatchSize'])        

        orderbook_index_content = TkIO.read_at_path( join(self._data_path, self._orderbook_index_filename) )
        self._orderbook_training_index = orderbook_index_content[0]
        self._orderbook_test_index = orderbook_index_content[1]
        self._orderbook_training_sample_id = _orderbook_training_sample_id
        self._orderbook_test_sample_id = _orderbook_test_sample_id
        self._orderbook_training_data_stream = open( join(self._data_path, self._orderbook_training_data_filename), 'rb+')
        self._orderbook_test_data_stream = open( join(self._data_path, self._orderbook_test_data_filename), 'rb+')

        random.shuffle(self._orderbook_training_index)
        random.shuffle(self._orderbook_test_index)

        last_trades_index_content = TkIO.read_at_path( join(self._data_path, self._last_trades_index_filename) )
        self._last_trades_training_index = last_trades_index_content[0]
        self._last_trades_test_index = last_trades_index_content[1]
        self._last_trades_training_sample_id = _last_trades_training_sample_id
        self._last_trades_test_sample_id = _last_trades_test_sample_id
        self._last_trades_training_data_stream = open( join(self._data_path, self._last_trades_training_data_filename), 'rb+')
        self._last_trades_test_data_stream = open( join(self._data_path, self._last_trades_test_data_filename), 'rb+')

        random.shuffle(self._last_trades_training_index)
        random.shuffle(self._last_trades_test_index)

        self._epoch_size = 1
        self._orderbook_training_batch_size = 1
        self._orderbook_test_batch_size = 1
        self._last_trades_training_batch_size = 1
        self._last_trades_test_batch_size = 1

        # compute balanced batch sizes

        if  len(self._orderbook_training_index) > len(self._last_trades_training_index) :
            self._orderbook_training_batch_size = self._max_training_batch_size
            self._epoch_size = int(round(len(self._orderbook_training_index) / self._max_training_batch_size))
            self._orderbook_test_batch_size = max(1,int(round(len(self._orderbook_test_index) / self._epoch_size)))
            self._last_trades_training_batch_size = max(1,int(round(len(self._last_trades_training_index) / self._epoch_size)))
            self._last_trades_test_batch_size = max(1,int(round(len(self._last_trades_test_index) / self._epoch_size)))
        else:
            self._last_trades_training_batch_size = self._max_training_batch_size
            self._epoch_size = int(round(len(self._last_trades_training_index) / self._max_training_batch_size))
            self._orderbook_training_batch_size = max(1,int(round(len(self._orderbook_training_index) / self._epoch_size)))
            self._orderbook_test_batch_size = max(1,int(round(len(self._orderbook_test_index) / self._epoch_size)))
            self._last_trades_test_batch_size = max(1,int(round(len(self._last_trades_test_index) / self._epoch_size)))

        print("Epoch_size:", self._epoch_size )
        print("Orderbook training batch size:", self._orderbook_training_batch_size)
        print("Orderbook test batch size:", self._orderbook_test_batch_size)
        print("Last trades training batch size:", self._last_trades_training_batch_size)
        print("Last trades test batch size:", self._last_trades_test_batch_size)

        self._orderbook_samples = None
        self._last_trades_samples = None
        self._loading_thread = None

    def close(self):
        self._orderbook_training_data_stream.close()
        self._orderbook_test_data_stream.close()
        self._last_trades_training_data_stream.close()
        self._last_trades_test_data_stream.close()

    def epoch_size(self):
        return self._epoch_size

    def orderbook_training_batch_size(self):
        return self._orderbook_training_batch_size
    
    def orderbook_test_batch_size(self):
        return self._orderbook_test_batch_size
    
    def last_trades_training_batch_size(self):
        return self._last_trades_training_batch_size
    
    def last_trades_test_batch_size(self):
        return self._last_trades_test_batch_size

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

    def get_orderbook_training_sample(self):
        self._orderbook_training_sample_id, orderbook_sample = TkAutoencoderDataLoader.load_sample(
            self._orderbook_training_sample_id,
            self._orderbook_training_index,
            self._orderbook_training_data_stream
        )
        return orderbook_sample

    def get_orderbook_test_sample(self):
        self._orderbook_test_sample_id, orderbook_sample = TkAutoencoderDataLoader.load_sample(
            self._orderbook_test_sample_id,
            self._orderbook_test_index,
            self._orderbook_test_data_stream
        )
        return orderbook_sample

    def get_last_trades_training_sample(self):
        self._last_trades_training_sample_id, last_trades_sample = TkAutoencoderDataLoader.load_sample(
            self._last_trades_training_sample_id,
            self._last_trades_training_index,
            self._last_trades_training_data_stream
        )
        return last_trades_sample

    def get_last_trades_test_sample(self):
        self._last_trades_test_sample_id, last_trades_sample = TkAutoencoderDataLoader.load_sample(
            self._last_trades_test_sample_id,
            self._last_trades_test_index,
            self._last_trades_test_data_stream
        )
        return last_trades_sample

    def start_load_training_data(self):
        if self._loading_thread != None:
            raise RuntimeError('Loading thread is active!')

        def load_training_data_thread():
            self._orderbook_samples = [None] * self._orderbook_training_batch_size
            for batch_id in range(self._orderbook_training_batch_size):
                self._orderbook_samples[batch_id] = self.get_orderbook_training_sample()

            self._last_trades_samples = [None] * self._last_trades_training_batch_size
            for batch_id in range(self._last_trades_training_batch_size):
                self._last_trades_samples[batch_id] = self.get_last_trades_training_sample()

        self._loading_thread = threading.Thread( target=load_training_data_thread )
        self._loading_thread.start()

    def start_load_test_data(self):
        if self._loading_thread != None:
            raise RuntimeError('Loading thread is active!')

        def load_test_data_thread():
            self._orderbook_samples = [None] * self._orderbook_test_batch_size
            for batch_id in range(self._orderbook_test_batch_size):
                self._orderbook_samples[batch_id] = self.get_orderbook_test_sample()

            self._last_trades_samples = [None] * self._last_trades_test_batch_size
            for batch_id in range(self._last_trades_test_batch_size):
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

TOKEN = os.environ["TK_TOKEN"]

cuda = torch.device("cuda")

config = configparser.ConfigParser()
config.read( 'TkConfig.ini' )

data_path = config['Paths']['DataPath']
orderbook_model_path =  join( config['Paths']['ModelsPath'], config['Paths']['OrderbookAutoencoderModelFileName'] )
orderbook_optimizer_path =  join( config['Paths']['ModelsPath'], config['Paths']['OrderbookAutoencoderOptimizerFileName'] )
orderbook_history_path = join( config['Paths']['ModelsPath'], config['Paths']['OrderbookAutoencoderTrainingHistoryFileName'] )
last_trades_model_path =  join( config['Paths']['ModelsPath'], config['Paths']['LastTradesAutoencoderModelFileName'] )
last_trades_optimizer_path =  join( config['Paths']['ModelsPath'], config['Paths']['LastTradesAutoencoderOptimizerFileName'] )
last_trades_history_path = join( config['Paths']['ModelsPath'], config['Paths']['LastTradesAutoencoderTrainingHistoryFileName'] )

cooldown = float(config['Autoencoders']['Cooldown'])
orderbook_width = int(config['Autoencoders']['OrderBookWidth'])
last_trades_width = int(config['Autoencoders']['LastTradesWidth'])
orderbook_code_layer_size = int(config['Autoencoders']['OrderbookAutoencoderCodeLayerSize'])
last_trades_code_layer_size = int(config['Autoencoders']['LastTradesAutoencoderCodeLayerSize'])
orderbook_autoencoder_free_bits = float(config['Autoencoders']['OrderbookAutoencoderFreeBits'])
last_trades_autoencoder_free_bits = float(config['Autoencoders']['LastTradesAutoencoderFreeBits'])
orderbook_autoencoder_beta = float(config['Autoencoders']['OrderbookAutoencoderBeta'])
last_trades_autoencoder_beta = float(config['Autoencoders']['LastTradesAutoencoderBeta'])
orderbook_autoencoder_learning_rate = float(config['Autoencoders']['OrderbookAutoencoderLearningRate'])
orderbook_autoencoder_weight_decay = float(config['Autoencoders']['OrderbookAutoencoderWeightDecay'])
last_trades_autoencoder_learning_rate = float(config['Autoencoders']['LastTradesAutoencoderLearningRate'])
last_trades_autoencoder_weight_decay = float(config['Autoencoders']['LastTradesAutoencoderWeightDecay'])
history_size = int( config['Autoencoders']['HistorySize'] )

orderbook_autoencoder = TkOrderbookAutoencoder(config)
orderbook_autoencoder.to(cuda)
if os.path.isfile(orderbook_model_path):
    orderbook_autoencoder.load_state_dict(torch.load(orderbook_model_path))
orderbook_optimizer = torch.optim.Adam( orderbook_autoencoder.parameters(), lr=orderbook_autoencoder_learning_rate, weight_decay=orderbook_autoencoder_weight_decay )
if os.path.isfile(orderbook_optimizer_path):
    orderbook_optimizer.load_state_dict(torch.load(orderbook_optimizer_path))

orderbook_loss = lambda x,y: hybrid_lob_multi_loss(x, y, alpha_1=0.55, alpha_2=0.35, beta=0.1, gamma=0.05, delta=0.001, win_size_1=11, levels_1=4, win_size_2=5, levels_2=3) # torch.nn.BCELoss() #
orderbook_training_history = TkAutoencoderTrainingHistory(orderbook_history_path, history_size)

last_trades_autoencoder = TkLastTradesAutoencoder(config)
last_trades_autoencoder.to(cuda)
if os.path.isfile(last_trades_model_path):   
    last_trades_autoencoder.load_state_dict(torch.load(last_trades_model_path))
last_trades_optimizer = torch.optim.Adam( last_trades_autoencoder.parameters(), lr=last_trades_autoencoder_learning_rate, weight_decay=last_trades_autoencoder_weight_decay )
if os.path.isfile(last_trades_optimizer_path):
    last_trades_optimizer.load_state_dict(torch.load(last_trades_optimizer_path))

last_trades_loss = lambda x,y: hybrid_ms_ssim_1d_gaussian_l1_loss(x,y,win_size=11,levels=4,alpha=0.95) # torch.nn.BCELoss() #
last_trades_training_history = TkAutoencoderTrainingHistory(last_trades_history_path, history_size)

data_loader = TkAutoencoderDataLoader(
    config,
    orderbook_training_history.training_sample_id(),
    orderbook_training_history.test_sample_id(),
    last_trades_training_history.training_sample_id(),
    last_trades_training_history.test_sample_id()
)

with Client(TOKEN, target=INVEST_GRPC_API) as client:

    dpg.create_context()
    dpg.create_viewport(title='Autoencoder training', width=2108, height=1102)
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
            with dpg.plot(label="Last trades input", width=512, height=256):
                dpg.add_plot_legend()
                dpg.add_plot_axis(dpg.mvXAxis, tag="x_axis_last_trades" )
                dpg.add_plot_axis(dpg.mvYAxis, tag="y_axis_last_trades" )
                dpg.add_bar_series( [j for j in range(0, 32)], [random.random() for j in range(0, 32)], label="Last trades input", parent="x_axis_last_trades", tag="last_trades_series" )
            with dpg.plot(label="Code layer", width=256, height=256):
                dpg.add_plot_legend()
                dpg.add_plot_axis(dpg.mvXAxis, tag="x_axis_last_trades_code" )
                dpg.add_plot_axis(dpg.mvYAxis, tag="y_axis_last_trades_code" )
                dpg.add_bar_series( [j for j in range(0, 32)], [random.random() for j in range(0, 32)], label="Code layer", parent="x_axis_last_trades_code", tag="last_trades_code_series" )
            with dpg.plot(label="Last trades output", width=512, height=256):
                dpg.add_plot_legend()
                dpg.add_plot_axis(dpg.mvXAxis, tag="x_axis_last_trades_output" )
                dpg.add_plot_axis(dpg.mvYAxis, tag="y_axis_last_trades_output" )
                dpg.add_bar_series( [j for j in range(0, 32)], [random.random() for j in range(0, 32)], label="Last trades output", parent="x_axis_last_trades_output", tag="last_trades_output_series" )
        with dpg.group(horizontal=True):
            with dpg.plot(label="Orderbook reconstruction", width=512, height=256):
                dpg.add_plot_legend()
                dpg.add_plot_axis(dpg.mvXAxis, tag="x_axis_orderbook_recon" )
                dpg.add_plot_axis(dpg.mvYAxis, tag="y_axis_orderbook_recon" )
                dpg.add_line_series( [j for j in range(0, 32)], [random.random() for j in range(0, 32)], label="Loss", parent="x_axis_orderbook_recon", tag="orderbook_recon_loss_series" )
                dpg.add_line_series( [j for j in range(0, 32)], [random.random() for j in range(0, 32)], label="Accuracy", parent="x_axis_orderbook_recon", tag="orderbook_recon_accuracy_series" )
            with dpg.plot(label="Orderbook reconstruction per epoch", width=512, height=256):
                dpg.add_plot_legend()
                dpg.add_plot_axis(dpg.mvXAxis, tag="x_axis_orderbook_recon_epoch" )
                dpg.add_plot_axis(dpg.mvYAxis, tag="y_axis_orderbook_recon_epoch" )
                dpg.add_line_series( [j for j in range(0, 32)], [random.random() for j in range(0, 32)], label="Loss", parent="x_axis_orderbook_recon_epoch", tag="orderbook_recon_loss_series_epoch" )
                dpg.add_line_series( [j for j in range(0, 32)], [random.random() for j in range(0, 32)], label="Accuracy", parent="x_axis_orderbook_recon_epoch", tag="orderbook_recon_accuracy_series_epoch" )
            with dpg.plot(label="Orderbook KLD", width=512, height=256):
                dpg.add_plot_legend()
                dpg.add_plot_axis(dpg.mvXAxis, tag="x_axis_orderbook_kld" )
                dpg.add_plot_axis(dpg.mvYAxis, tag="y_axis_orderbook_kld" )
                dpg.add_line_series( [j for j in range(0, 32)], [random.random() for j in range(0, 32)], label="Loss", parent="x_axis_orderbook_kld", tag="orderbook_kld_loss_series" )
                dpg.add_line_series( [j for j in range(0, 32)], [random.random() for j in range(0, 32)], label="Accuracy", parent="x_axis_orderbook_kld", tag="orderbook_kld_accuracy_series" )
            with dpg.plot(label="Orderbook KLD per epoch", width=512, height=256):
                dpg.add_plot_legend()
                dpg.add_plot_axis(dpg.mvXAxis, tag="x_axis_orderbook_kld_epoch" )
                dpg.add_plot_axis(dpg.mvYAxis, tag="y_axis_orderbook_kld_epoch" )
                dpg.add_line_series( [j for j in range(0, 32)], [random.random() for j in range(0, 32)], label="Loss", parent="x_axis_orderbook_kld_epoch", tag="orderbook_kld_loss_series_epoch" )
                dpg.add_line_series( [j for j in range(0, 32)], [random.random() for j in range(0, 32)], label="Accuracy", parent="x_axis_orderbook_kld_epoch", tag="orderbook_kld_accuracy_series_epoch" )
        with dpg.group(horizontal=True):
            with dpg.plot(label="Last trades reconstruction", width=512, height=256):
                dpg.add_plot_legend()
                dpg.add_plot_axis(dpg.mvXAxis, tag="x_axis_last_trades_recon" )
                dpg.add_plot_axis(dpg.mvYAxis, tag="y_axis_last_trades_recon" )
                dpg.add_line_series( [j for j in range(0, 32)], [random.random() for j in range(0, 32)], label="Loss", parent="x_axis_last_trades_recon", tag="last_trades_recon_loss_series" )
                dpg.add_line_series( [j for j in range(0, 32)], [random.random() for j in range(0, 32)], label="Accuracy", parent="x_axis_last_trades_recon", tag="last_trades_recon_accuracy_series" )
            with dpg.plot(label="Last trades reconstruction per epoch", width=512, height=256):
                dpg.add_plot_legend()
                dpg.add_plot_axis(dpg.mvXAxis, tag="x_axis_last_trades_recon_epoch" )
                dpg.add_plot_axis(dpg.mvYAxis, tag="y_axis_last_trades_recon_epoch" )
                dpg.add_line_series( [j for j in range(0, 32)], [random.random() for j in range(0, 32)], label="Loss", parent="x_axis_last_trades_recon_epoch", tag="last_trades_recon_loss_series_epoch" )
                dpg.add_line_series( [j for j in range(0, 32)], [random.random() for j in range(0, 32)], label="Accuracy", parent="x_axis_last_trades_recon_epoch", tag="last_trades_recon_accuracy_series_epoch" )
            with dpg.plot(label="Last trades KLD", width=512, height=256):
                dpg.add_plot_legend()
                dpg.add_plot_axis(dpg.mvXAxis, tag="x_axis_last_trades_kld" )
                dpg.add_plot_axis(dpg.mvYAxis, tag="y_axis_last_trades_kld" )
                dpg.add_line_series( [j for j in range(0, 32)], [random.random() for j in range(0, 32)], label="Loss", parent="x_axis_last_trades_kld", tag="last_trades_kld_loss_series" )
                dpg.add_line_series( [j for j in range(0, 32)], [random.random() for j in range(0, 32)], label="Accuracy", parent="x_axis_last_trades_kld", tag="last_trades_kld_accuracy_series" )
            with dpg.plot(label="Last trades KLD per epoch", width=512, height=256):
                dpg.add_plot_legend()
                dpg.add_plot_axis(dpg.mvXAxis, tag="x_axis_last_trades_kld_epoch" )
                dpg.add_plot_axis(dpg.mvYAxis, tag="y_axis_last_trades_kld_epoch" )
                dpg.add_line_series( [j for j in range(0, 32)], [random.random() for j in range(0, 32)], label="Loss", parent="x_axis_last_trades_kld_epoch", tag="last_trades_kld_loss_series_epoch" )
                dpg.add_line_series( [j for j in range(0, 32)], [random.random() for j in range(0, 32)], label="Accuracy", parent="x_axis_last_trades_kld_epoch", tag="last_trades_kld_accuracy_series_epoch" )

    dpg.show_viewport()
    dpg.set_primary_window("primary_window", True)

    data_loader.start_load_training_data()

    while dpg.is_dearpygui_running():

        #def get_epoch_schedule_weight(smooth_epoch : float):
        #    return 1.0 - math.exp( -0.0625 * smooth_epoch )

        #orderbook_smooth_epoch = orderbook_training_history.get_smooth_epoch( data_loader.epoch_size() * data_loader.orderbook_training_batch_size() )
        #orderbook_autoencoder_beta = float(config['Autoencoders']['OrderbookAutoencoderBeta']) * get_epoch_schedule_weight( orderbook_smooth_epoch )
        #orderbook_autoencoder_weight_decay = float(config['Autoencoders']['OrderbookAutoencoderWeightDecay']) * get_epoch_schedule_weight( orderbook_smooth_epoch )
        orderbook_optimizer.param_groups[0]['weight_decay'] = orderbook_autoencoder_weight_decay
        orderbook_optimizer.param_groups[0]['lr'] = orderbook_autoencoder_learning_rate

        #last_trades_smooth_epoch = orderbook_training_history.get_smooth_epoch( data_loader.epoch_size() * data_loader.last_trades_training_batch_size() )
        #last_trades_autoencoder_beta = float(config['Autoencoders']['LastTradesAutoencoderBeta']) * get_epoch_schedule_weight( last_trades_smooth_epoch )
        #last_trades_autoencoder_weight_decay = float(config['Autoencoders']['LastTradesAutoencoderWeightDecay']) * get_epoch_schedule_weight( last_trades_smooth_epoch )
        last_trades_optimizer.param_groups[0]['weight_decay'] = last_trades_autoencoder_weight_decay
        last_trades_optimizer.param_groups[0]['lr'] = last_trades_autoencoder_learning_rate

        orderbook_samples, last_trades_samples = data_loader.complete_loading()

        orderbook_input = torch.Tensor( list( itertools.chain.from_iterable(orderbook_samples) ) )
        orderbook_input = torch.reshape( orderbook_input, ( data_loader.orderbook_training_batch_size(), 1, orderbook_width) )
        orderbook_input = orderbook_input.to(cuda)

        last_trades_input = torch.Tensor( list( itertools.chain.from_iterable(last_trades_samples) ) )
        last_trades_input = torch.reshape( last_trades_input, ( data_loader.last_trades_training_batch_size(), 1, last_trades_width) )
        last_trades_input = last_trades_input.to(cuda)

        data_loader.start_load_test_data()

        y, y_mean, y_logvar = orderbook_autoencoder.forward( orderbook_input )
        z, z_mean, z_logvar = last_trades_autoencoder.forward( last_trades_input )

        #TkUI.set_series_from_tensor("x_axis_orderbook", "y_axis_orderbook","orderbook_series",orderbook_input,0)
        #TkUI.set_series_from_tensor("x_axis_orderbook_code", "y_axis_orderbook_code","orderbook_code_series",orderbook_autoencoder.code(),0)
        #TkUI.set_series_from_tensor("x_axis_orderbook_output", "y_axis_orderbook_output","orderbook_output_series",y,0)

        #TkUI.set_series_from_tensor("x_axis_last_trades", "y_axis_last_trades","last_trades_series",last_trades_input,0)
        #TkUI.set_series_from_tensor("x_axis_last_trades_code", "y_axis_last_trades_code","last_trades_code_series",last_trades_autoencoder.code(),0)
        #TkUI.set_series_from_tensor("x_axis_last_trades_output", "y_axis_last_trades_output","last_trades_output_series",z,0)

        #y_KLD_loss = -0.5 * torch.mean( torch.sum( 1 + y_logvar - y_mean.pow(2) - y_logvar.exp(), dim=1), dim=0 )
        y_KLD_loss = -0.5 * torch.sum( 1 + y_logvar - y_mean.pow(2) - y_logvar.exp(), dim=1)

        y_KLD_loss = torch.max( y_KLD_loss, torch.tensor( orderbook_autoencoder_free_bits, device=y_KLD_loss.device))

        y_recon_loss = orderbook_loss( y, orderbook_input )
        y_loss = y_recon_loss + y_KLD_loss * orderbook_autoencoder_beta
        y_loss = y_loss.mean()
        orderbook_optimizer.zero_grad()
        y_loss.backward()
        orderbook_optimizer.step()        
        y_KLD_loss_val = y_KLD_loss.mean().item()
        y_recon_loss_val = y_recon_loss.mean().item()

        #z_KLD_loss = -0.5 * torch.mean( torch.sum( 1 + z_logvar - z_mean.pow(2) - z_logvar.exp(), dim=1), dim=0 )
        z_KLD_loss = -0.5 * torch.sum( 1 + z_logvar - z_mean.pow(2) - z_logvar.exp(), dim=1)

        z_KLD_loss = torch.max( z_KLD_loss, torch.tensor( last_trades_autoencoder_free_bits, device=z_KLD_loss.device))

        z_recon_loss = last_trades_loss( z, last_trades_input )        
        z_loss = z_recon_loss + z_KLD_loss * last_trades_autoencoder_beta
        z_loss = z_loss.mean()
        last_trades_optimizer.zero_grad()
        z_loss.backward()
        last_trades_optimizer.step()
        z_KLD_loss_val = z_KLD_loss.mean().item()
        z_recon_loss_val = z_recon_loss.mean().item()

        dpg.render_dearpygui_frame()

        orderbook_samples, last_trades_samples = data_loader.complete_loading()

        orderbook_input = torch.Tensor( list( itertools.chain.from_iterable(orderbook_samples) ) )
        orderbook_input = torch.reshape( orderbook_input, ( data_loader.orderbook_test_batch_size(), 1, orderbook_width) )
        orderbook_input = orderbook_input.to(cuda)

        last_trades_input = torch.Tensor( list( itertools.chain.from_iterable(last_trades_samples) ) )
        last_trades_input = torch.reshape( last_trades_input, ( data_loader.last_trades_test_batch_size(), 1, last_trades_width) )
        last_trades_input = last_trades_input.to(cuda)

        data_loader.start_load_training_data()

        orderbook_autoencoder.train(False)
        y, y_mean, y_logvar = orderbook_autoencoder.forward( orderbook_input )
        orderbook_autoencoder.train(True)

        last_trades_autoencoder.train(False)
        z, z_mean, z_logvar = last_trades_autoencoder.forward( last_trades_input )
        last_trades_autoencoder.train(True)

        TkUI.set_series_from_tensor("x_axis_orderbook", "y_axis_orderbook","orderbook_series",orderbook_input,0)
        TkUI.set_series_from_tensor("x_axis_orderbook_code", "y_axis_orderbook_code","orderbook_code_series",orderbook_autoencoder.code(),0)
        TkUI.set_series_from_tensor("x_axis_orderbook_output", "y_axis_orderbook_output","orderbook_output_series",y,0)

        TkUI.set_series_from_tensor("x_axis_last_trades", "y_axis_last_trades","last_trades_series",last_trades_input,0)
        TkUI.set_series_from_tensor("x_axis_last_trades_code", "y_axis_last_trades_code","last_trades_code_series",last_trades_autoencoder.code(),0)
        TkUI.set_series_from_tensor("x_axis_last_trades_output", "y_axis_last_trades_output","last_trades_output_series",z,0)

        #y_KLD_accuracy = -0.5 * torch.mean( torch.sum( 1 + y_logvar - y_mean.pow(2) - y_logvar.exp(), dim=1), dim=0 )
        y_KLD_accuracy = -0.5 * torch.sum( 1 + y_logvar - y_mean.pow(2) - y_logvar.exp(), dim=1)

        y_recon_accuracy = orderbook_loss( y, orderbook_input )        
        y_KLD_accuracy_val = y_KLD_accuracy.mean().item()
        y_recon_accuracy_val = y_recon_accuracy.mean().item()

        #z_KLD_accuracy = -0.5 * torch.mean( torch.sum( 1 + z_logvar - z_mean.pow(2) - z_logvar.exp(), dim=1), dim=0 )
        z_KLD_accuracy = -0.5 * torch.sum( 1 + z_logvar - z_mean.pow(2) - z_logvar.exp(), dim=1)

        z_recon_accuracy = last_trades_loss( z, last_trades_input )
        z_KLD_accuracy_val = z_KLD_accuracy.mean().item()
        z_recon_accuracy_val = z_recon_accuracy.mean().item()

        dpg.render_dearpygui_frame()        

        orderbook_training_history.log( data_loader.orderbook_training_sample_id(), data_loader.orderbook_test_sample_id(), y_recon_loss_val, y_recon_accuracy_val, y_KLD_loss_val, y_KLD_accuracy_val)
        last_trades_training_history.log(data_loader.last_trades_training_sample_id(), data_loader.last_trades_test_sample_id(), z_recon_loss_val, z_recon_accuracy_val, z_KLD_loss_val, z_KLD_accuracy_val )

        TkUI.set_series("x_axis_orderbook_recon", "y_axis_orderbook_recon", "orderbook_recon_loss_series", orderbook_training_history.recon_loss_history())
        TkUI.set_series("x_axis_orderbook_recon_epoch", "y_axis_orderbook_recon_epoch", "orderbook_recon_loss_series_epoch", orderbook_training_history.epoch_recon_loss_history())
        TkUI.set_series("x_axis_orderbook_recon", "y_axis_orderbook_recon", "orderbook_recon_accuracy_series", orderbook_training_history.recon_accuracy_history())
        TkUI.set_series("x_axis_orderbook_recon_epoch", "y_axis_orderbook_recon_epoch", "orderbook_recon_accuracy_series_epoch", orderbook_training_history.epoch_recon_accuracy_history())

        TkUI.set_series("x_axis_last_trades_recon", "y_axis_last_trades_recon", "last_trades_recon_loss_series", last_trades_training_history.recon_loss_history())
        TkUI.set_series("x_axis_last_trades_recon_epoch", "y_axis_last_trades_recon_epoch", "last_trades_recon_loss_series_epoch", last_trades_training_history.epoch_recon_loss_history())
        TkUI.set_series("x_axis_last_trades_recon", "y_axis_last_trades_recon", "last_trades_recon_accuracy_series", last_trades_training_history.recon_accuracy_history())
        TkUI.set_series("x_axis_last_trades_recon_epoch", "y_axis_last_trades_recon_epoch", "last_trades_recon_accuracy_series_epoch", last_trades_training_history.epoch_recon_accuracy_history())

        #KL

        TkUI.set_series("x_axis_orderbook_kld", "y_axis_orderbook_kld", "orderbook_kld_loss_series", orderbook_training_history.kld_loss_history())
        TkUI.set_series("x_axis_orderbook_kld_epoch", "y_axis_orderbook_kld_epoch", "orderbook_kld_loss_series_epoch", orderbook_training_history.epoch_kld_loss_history())
        TkUI.set_series("x_axis_orderbook_kld", "y_axis_orderbook_kld", "orderbook_kld_accuracy_series", orderbook_training_history.kld_accuracy_history())
        TkUI.set_series("x_axis_orderbook_kld_epoch", "y_axis_orderbook_kld_epoch", "orderbook_kld_accuracy_series_epoch", orderbook_training_history.epoch_kld_accuracy_history())

        TkUI.set_series("x_axis_last_trades_kld", "y_axis_last_trades_kld", "last_trades_kld_loss_series", last_trades_training_history.kld_loss_history())
        TkUI.set_series("x_axis_last_trades_kld_epoch", "y_axis_last_trades_kld_epoch", "last_trades_kld_loss_series_epoch", last_trades_training_history.epoch_kld_loss_history())
        TkUI.set_series("x_axis_last_trades_kld", "y_axis_last_trades_kld", "last_trades_kld_accuracy_series", last_trades_training_history.kld_accuracy_history())
        TkUI.set_series("x_axis_last_trades_kld_epoch", "y_axis_last_trades_kld_epoch", "last_trades_kld_accuracy_series_epoch", last_trades_training_history.epoch_kld_accuracy_history())

        dpg.render_dearpygui_frame()

        cooldownRemaining = cooldown
        cooldownStep = 1.0 / 30.0
        while dpg.is_dearpygui_running() and cooldownRemaining > 0.0:
            time.sleep(cooldownStep)
            cooldownRemaining -= cooldownStep
            dpg.render_dearpygui_frame()
        

    dpg.destroy_context()

    data_loader.complete_loading()
    data_loader.close()

    orderbook_training_history.save()
    torch.save( orderbook_autoencoder.state_dict(), orderbook_model_path )
    torch.save( orderbook_optimizer.state_dict(), orderbook_optimizer_path )    

    last_trades_training_history.save()
    torch.save( last_trades_autoencoder.state_dict(), last_trades_model_path )
    torch.save( last_trades_optimizer.state_dict(), last_trades_optimizer_path )    