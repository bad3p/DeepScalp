
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
from TkModules.TkSSIM import MS_SSIM_1D_Loss
from TkModules.TkAnnealing import TkAnnealing

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
history_size = int( config['Autoencoders']['HistorySize'] )

orderbook_width = int(config['Autoencoders']['OrderbookWidth'])
orderbook_depth = int(config['Autoencoders']['OrderbookDepth'])
orderbook_reconstruction_channel_0 = int(config['Autoencoders']['OrderbookReconstructionChannel0'])
orderbook_reconstruction_channel_1 = int(config['Autoencoders']['OrderbookReconstructionChannel1'])
last_trades_width = int(config['Autoencoders']['LastTradesWidth'])
last_trades_depth = int(config['Autoencoders']['LastTradesDepth'])
last_trades_reconstruction_channel = int(config['Autoencoders']['LastTradesReconstructionChannel'])
orderbook_code_layer_size = int(config['Autoencoders']['OrderbookAutoencoderCodeLayerSize'])
last_trades_code_layer_size = int(config['Autoencoders']['LastTradesAutoencoderCodeLayerSize'])

orderbook_autoencoder_conv_weight_decay = float(config['Autoencoders']['OrderbookAutoencoderConvWeightDecay'])
orderbook_autoencoder_dense_weight_decay = float(config['Autoencoders']['OrderbookAutoencoderDenseWeightDecay'])
orderbook_autoencoder_log_volume_loss_weight = TkAnnealing(config['Autoencoders']['OrderbookAutoencoderLogVolumeLossWeight'])
orderbook_autoencoder_kl_loss_weight = TkAnnealing(config['Autoencoders']['OrderbookAutoencoderKLLossWeight'])
orderbook_autoencoder_vq_loss_weight = TkAnnealing(config['Autoencoders']['OrderbookAutoencoderVQLossWeight'])
orderbook_autoencoder_learning_rate = TkAnnealing(config['Autoencoders']['OrderbookAutoencoderLearningRate'])

last_trades_autoencoder_conv_weight_decay = float(config['Autoencoders']['LastTradesAutoencoderConvWeightDecay'])
last_trades_autoencoder_dense_weight_decay = float(config['Autoencoders']['LastTradesAutoencoderDenseWeightDecay'])
last_trades_autoencoder_smoothness_loss_weight = TkAnnealing(config['Autoencoders']['LastTradesAutoencoderSmoothnessLossWeight'])
last_trades_autoencoder_log_volume_loss_weight = TkAnnealing(config['Autoencoders']['LastTradesAutoencoderLogVolumeLossWeight'])
last_trades_autoencoder_vq_loss_weight = TkAnnealing(config['Autoencoders']['LastTradesAutoencoderVQLossWeight'])
last_trades_autoencoder_learning_rate = TkAnnealing(config['Autoencoders']['LastTradesAutoencoderLearningRate'])

orderbook_autoencoder = TkOrderbookAutoencoder(config)
orderbook_autoencoder.to(cuda)
if os.path.isfile(orderbook_model_path):
    orderbook_autoencoder.load_state_dict(torch.load(orderbook_model_path))
orderbook_optimizer = torch.optim.AdamW( 
    orderbook_autoencoder.get_trainable_parameters(orderbook_autoencoder_conv_weight_decay, orderbook_autoencoder_dense_weight_decay),
    lr=orderbook_autoencoder_learning_rate.get_value(0.0)
)
if os.path.isfile(orderbook_optimizer_path):
    orderbook_optimizer.load_state_dict(torch.load(orderbook_optimizer_path))

orderbook_loss = torch.nn.BCELoss(reduction="none") #  MS_SSIM_1D_Loss(window_size=9) # lambda x,y: hybrid_lob_multi_loss(x, y, alpha_1=0.55, alpha_2=0.35, beta=0.1, gamma=0.05, delta=0.001, win_size_1=9, levels_1=3, win_size_2=5, levels_2=2) #  
orderbook_training_history = TkAutoencoderTrainingHistory(orderbook_history_path, history_size)

last_trades_autoencoder = TkLastTradesAutoencoder(config)
last_trades_autoencoder.to(cuda)
if os.path.isfile(last_trades_model_path):   
    last_trades_autoencoder.load_state_dict(torch.load(last_trades_model_path))
last_trades_optimizer = torch.optim.AdamW( 
    last_trades_autoencoder.get_trainable_parameters(last_trades_autoencoder_conv_weight_decay, last_trades_autoencoder_dense_weight_decay),
    lr=last_trades_autoencoder_learning_rate.get_value(0.0)
)
if os.path.isfile(last_trades_optimizer_path):
    last_trades_optimizer.load_state_dict(torch.load(last_trades_optimizer_path))

last_trades_loss = MS_SSIM_1D_Loss(window_size=3) # lambda x,y: hybrid_ssim_1d_gaussian_l1_loss(x,y,win_size=9,alpha=0.9) # torch.nn.BCELoss(reduction="none") #
last_trades_training_history = TkAutoencoderTrainingHistory(last_trades_history_path, history_size)

data_loader = TkAutoencoderDataLoader(
    config,
    orderbook_training_history.training_sample_id(),
    orderbook_training_history.test_sample_id(),
    last_trades_training_history.training_sample_id(),
    last_trades_training_history.test_sample_id() 
)

def save_orderbook_autoencoder():
    orderbook_training_history.save()
    torch.save( orderbook_autoencoder.state_dict(), orderbook_model_path )
    torch.save( orderbook_optimizer.state_dict(), orderbook_optimizer_path )    

def save_last_trades_autoencoder():
    last_trades_training_history.save()
    torch.save( last_trades_autoencoder.state_dict(), last_trades_model_path )
    torch.save( last_trades_optimizer.state_dict(), last_trades_optimizer_path )    

#orderbook_training_history.set_end_of_epoch_callback( save_orderbook_autoencoder )
#last_trades_training_history.set_end_of_epoch_callback( save_last_trades_autoencoder )

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
                dpg.add_bar_series( [j for j in range(0, 32)], [random.random() for j in range(0, 32)], label="Orderbook input", parent="x_axis_orderbook", tag="orderbook_series_0" )
                dpg.add_bar_series( [j for j in range(0, 32)], [random.random() for j in range(0, 32)], label="Orderbook input", parent="x_axis_orderbook", tag="orderbook_series_1" )
            with dpg.plot(label="Code layer", width=256, height=256):
                dpg.add_plot_legend()
                dpg.add_plot_axis(dpg.mvXAxis, tag="x_axis_orderbook_code" )
                dpg.add_plot_axis(dpg.mvYAxis, tag="y_axis_orderbook_code" )
                dpg.add_bar_series( [j for j in range(0, 32)], [random.random() for j in range(0, 32)], label="Code layer", parent="x_axis_orderbook_code", tag="orderbook_code_series" )
            with dpg.plot(label="Orderbook output", width=512, height=256):
                dpg.add_plot_legend()
                dpg.add_plot_axis(dpg.mvXAxis, tag="x_axis_orderbook_output" )
                dpg.add_plot_axis(dpg.mvYAxis, tag="y_axis_orderbook_output" )
                dpg.add_bar_series( [j for j in range(0, 32)], [random.random() for j in range(0, 32)], label="Orderbook output", parent="x_axis_orderbook_output", tag="orderbook_output_series_0" )
                dpg.add_bar_series( [j for j in range(0, 32)], [random.random() for j in range(0, 32)], label="Orderbook output", parent="x_axis_orderbook_output", tag="orderbook_output_series_1" )
            with dpg.plot(label="Model codebook", width=512, height=256):
                dpg.add_plot_legend()
                dpg.add_plot_axis(dpg.mvXAxis, tag="x_axis_orderbook_codebook" )
                dpg.add_plot_axis(dpg.mvYAxis, tag="y_axis_orderbook_codebook" )
                dpg.add_line_series( [j for j in range(0, 32)], [random.random() for j in range(0, 32)], label="Active codes", parent="x_axis_orderbook_codebook", tag="orderbook_active_codes_series" )
                dpg.add_line_series( [j for j in range(0, 32)], [random.random() for j in range(0, 32)], label="Dead codes", parent="x_axis_orderbook_codebook", tag="orderbook_dead_codes_series" )
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
            with dpg.plot(label="Model codebook", width=512, height=256):
                dpg.add_plot_legend()
                dpg.add_plot_axis(dpg.mvXAxis, tag="x_axis_last_trades_codebook" )
                dpg.add_plot_axis(dpg.mvYAxis, tag="y_axis_last_trades_codebook" )
                dpg.add_line_series( [j for j in range(0, 32)], [random.random() for j in range(0, 32)], label="Active codes", parent="x_axis_last_trades_codebook", tag="last_trades_active_codes_series" )
                dpg.add_line_series( [j for j in range(0, 32)], [random.random() for j in range(0, 32)], label="Dead codes", parent="x_axis_last_trades_codebook", tag="last_trades_dead_codes_series" )
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

    # torch.autograd.set_detect_anomaly(True)

    data_loader.start_load_training_data()

    cuda_memory_allocated = []

    def cuda_memory_leak_per_loop():
        result = 0
        norm = 0
        for i in range( 1, len(cuda_memory_allocated)):
            result = result + ( cuda_memory_allocated[i] - cuda_memory_allocated[i-1] )
            norm = norm + 1
        return  result / norm if norm > 0 else result        

    while dpg.is_dearpygui_running():

        orderbook_smooth_epoch = orderbook_training_history.get_smooth_epoch( data_loader.epoch_size() * data_loader.orderbook_training_batch_size() )
        for param_group in orderbook_optimizer.param_groups:
            param_group['lr'] = orderbook_autoencoder_learning_rate.get_value(orderbook_smooth_epoch)

        last_trades_smooth_epoch = last_trades_training_history.get_smooth_epoch( data_loader.epoch_size() * data_loader.last_trades_training_batch_size() )
        for param_group in last_trades_optimizer.param_groups:
            param_group['lr'] = last_trades_autoencoder_learning_rate.get_value(last_trades_smooth_epoch)

        orderbook_samples, last_trades_samples = data_loader.complete_loading()

        orderbook_input = torch.Tensor( list( itertools.chain.from_iterable(orderbook_samples) ) )
        orderbook_input = torch.reshape( orderbook_input, ( data_loader.orderbook_training_batch_size(), orderbook_depth, orderbook_width) )
        orderbook_input = orderbook_input.to(cuda)

        orderbook_recon_target_0 = orderbook_input[:, (orderbook_reconstruction_channel_0):(orderbook_reconstruction_channel_0+1), :]
        orderbook_recon_target_1 = orderbook_input[:, (orderbook_reconstruction_channel_1):(orderbook_reconstruction_channel_1+1), :]

        last_trades_input = torch.Tensor( list( itertools.chain.from_iterable(last_trades_samples) ) )
        last_trades_input = torch.reshape( last_trades_input, ( data_loader.last_trades_training_batch_size(), last_trades_depth, last_trades_width) )
        last_trades_input = last_trades_input.to(cuda)

        last_trades_recon_target = last_trades_input[:, (last_trades_reconstruction_channel):(last_trades_reconstruction_channel+1), :]

        data_loader.start_load_test_data()

        y, y_log_volume_loss, y_vq_loss, y_kl_loss = orderbook_autoencoder.forward( orderbook_input )
        y_active_codes, y_dead_codes, _ = orderbook_autoencoder.get_code_usage()
        z, z_log_volume_loss, z_vq_loss, z_smoothness_loss = last_trades_autoencoder.forward( last_trades_input )
        z_active_codes, z_dead_codes, _ = last_trades_autoencoder.get_code_usage()

        y_0 = y[:, 0:1, :]
        y_1 = y[:, 1:2, :]

        #TkUI.set_series_from_tensor("x_axis_orderbook", "y_axis_orderbook","orderbook_series_0",orderbook_recon_target_0,0)
        #TkUI.set_series_from_tensor("x_axis_orderbook", "y_axis_orderbook","orderbook_series_1",orderbook_recon_target_1,0)
        #TkUI.set_series_from_tensor("x_axis_orderbook_code", "y_axis_orderbook_code","orderbook_code_series",orderbook_autoencoder.code(),0)
        #TkUI.set_series_from_tensor("x_axis_orderbook_output", "y_axis_orderbook_output","orderbook_output_series_0",y_0,0)
        #TkUI.set_series_from_tensor("x_axis_orderbook_output", "y_axis_orderbook_output","orderbook_output_series_1",y_1,0)

        #TkUI.set_series_from_tensor("x_axis_last_trades", "y_axis_last_trades","last_trades_series",last_trades_recon_target,0)
        #TkUI.set_series_from_tensor("x_axis_last_trades_code", "y_axis_last_trades_code","last_trades_code_series",last_trades_autoencoder.code(),0)
        #TkUI.set_series_from_tensor("x_axis_last_trades_output", "y_axis_last_trades_output","last_trades_output_series",z,0)

        y_recon_loss = orderbook_loss( y_0, orderbook_recon_target_0 ) + orderbook_loss( y_1, orderbook_recon_target_1 ) 
        y_loss = ( y_recon_loss + 
                   y_vq_loss * orderbook_autoencoder_vq_loss_weight.get_value(orderbook_smooth_epoch) + 
                   y_log_volume_loss * orderbook_autoencoder_log_volume_loss_weight.get_value(orderbook_smooth_epoch) +
                   y_kl_loss * orderbook_autoencoder_kl_loss_weight.get_value(orderbook_smooth_epoch) )
        y_loss = y_loss.mean()
        orderbook_optimizer.zero_grad()
        y_loss.backward()
        torch.nn.utils.clip_grad_norm_( orderbook_autoencoder.parameters(), max_norm=1.0 ) # TODO: configure
        orderbook_optimizer.step()        
        y_KLD_loss_val = y_vq_loss.mean().item()
        y_recon_loss_val = y_recon_loss.mean().item()

        z_recon_loss = last_trades_loss( z, last_trades_recon_target )
        z_loss = ( z_recon_loss + 
                   z_vq_loss * last_trades_autoencoder_vq_loss_weight.get_value(last_trades_smooth_epoch) + 
                   z_log_volume_loss * last_trades_autoencoder_log_volume_loss_weight.get_value(last_trades_smooth_epoch) + 
                   z_smoothness_loss * last_trades_autoencoder_smoothness_loss_weight.get_value(last_trades_smooth_epoch) )
        z_loss = z_loss.mean()
        last_trades_optimizer.zero_grad()
        z_loss.backward()
        torch.nn.utils.clip_grad_norm_( last_trades_autoencoder.parameters(), max_norm=1.0 ) # TODO: configure
        last_trades_optimizer.step()        
        z_KLD_loss_val = z_vq_loss.mean().item()
        z_recon_loss_val = z_recon_loss.mean().item()

        dpg.render_dearpygui_frame()

        with torch.no_grad():

            orderbook_samples, last_trades_samples = data_loader.complete_loading()

            orderbook_input = torch.Tensor( list( itertools.chain.from_iterable(orderbook_samples) ) )
            orderbook_input = torch.reshape( orderbook_input, ( data_loader.orderbook_test_batch_size(), orderbook_depth, orderbook_width) )
            orderbook_input = orderbook_input.to(cuda)

            orderbook_recon_target_0 = orderbook_input[:, (orderbook_reconstruction_channel_0):(orderbook_reconstruction_channel_0+1), :]
            orderbook_recon_target_1 = orderbook_input[:, (orderbook_reconstruction_channel_1):(orderbook_reconstruction_channel_1+1), :]

            last_trades_input = torch.Tensor( list( itertools.chain.from_iterable(last_trades_samples) ) )
            last_trades_input = torch.reshape( last_trades_input, ( data_loader.last_trades_test_batch_size(), last_trades_depth, last_trades_width) )
            last_trades_input = last_trades_input.to(cuda)

            last_trades_recon_target = last_trades_input[:, (last_trades_reconstruction_channel):(last_trades_reconstruction_channel+1), :]

            data_loader.start_load_training_data()

            orderbook_autoencoder.train(False)
            y, y_log_volume_loss, y_vq_loss, y_kl_loss = orderbook_autoencoder.forward( orderbook_input )
            orderbook_autoencoder.train(True)

            y_0 = y[:, 0:1, :]
            y_1 = y[:, 1:2, :]

            last_trades_autoencoder.train(False)
            z, z_log_volume_loss, z_vq_loss, z_smoothness_loss = last_trades_autoencoder.forward( last_trades_input )
            last_trades_autoencoder.train(True)

            TkUI.set_series_from_tensor("x_axis_orderbook", "y_axis_orderbook","orderbook_series_0", orderbook_recon_target_0.detach().cpu(), 0)
            TkUI.set_series_from_tensor("x_axis_orderbook", "y_axis_orderbook","orderbook_series_1",orderbook_recon_target_1.detach().cpu(), 0)
            TkUI.set_series_from_tensor("x_axis_orderbook_code", "y_axis_orderbook_code","orderbook_code_series",orderbook_autoencoder.code().detach().cpu(), 0)
            TkUI.set_series_from_tensor("x_axis_orderbook_output", "y_axis_orderbook_output","orderbook_output_series_0",y_0.detach().cpu(), 0)
            TkUI.set_series_from_tensor("x_axis_orderbook_output", "y_axis_orderbook_output","orderbook_output_series_1",y_1.detach().cpu(), 0)

            TkUI.set_series_from_tensor("x_axis_last_trades", "y_axis_last_trades","last_trades_series",last_trades_recon_target.detach().cpu(), 0)
            TkUI.set_series_from_tensor("x_axis_last_trades_code", "y_axis_last_trades_code","last_trades_code_series",last_trades_autoencoder.code().detach().cpu(), 0)
            TkUI.set_series_from_tensor("x_axis_last_trades_output", "y_axis_last_trades_output","last_trades_output_series",z.detach().cpu(), 0)

            y_recon_accuracy = orderbook_loss( y_0, orderbook_recon_target_0 ) + orderbook_loss( y_1, orderbook_recon_target_1 )
            y_KLD_accuracy_val = y_vq_loss.mean().item()
            y_recon_accuracy_val = y_recon_accuracy.mean().item()

            z_recon_accuracy = last_trades_loss( z, last_trades_recon_target )
            z_KLD_accuracy_val = z_vq_loss.mean().item()
            z_recon_accuracy_val = z_recon_accuracy.mean().item()

            dpg.render_dearpygui_frame()        

            orderbook_training_history.log( data_loader.orderbook_training_sample_id(), data_loader.orderbook_test_sample_id(), y_recon_loss_val, y_recon_accuracy_val, y_KLD_loss_val, y_KLD_accuracy_val, y_active_codes, y_dead_codes)
            last_trades_training_history.log(data_loader.last_trades_training_sample_id(), data_loader.last_trades_test_sample_id(), z_recon_loss_val, z_recon_accuracy_val, z_KLD_loss_val, z_KLD_accuracy_val, z_active_codes, z_dead_codes )

            TkUI.set_series("x_axis_orderbook_codebook", "y_axis_orderbook_codebook", "orderbook_active_codes_series", orderbook_training_history.active_codes())
            TkUI.set_series("x_axis_orderbook_codebook", "y_axis_orderbook_codebook", "orderbook_dead_codes_series", orderbook_training_history.dead_codes())
            TkUI.set_series("x_axis_orderbook_recon", "y_axis_orderbook_recon", "orderbook_recon_loss_series", orderbook_training_history.recon_loss_history())
            TkUI.set_series("x_axis_orderbook_recon_epoch", "y_axis_orderbook_recon_epoch", "orderbook_recon_loss_series_epoch", orderbook_training_history.epoch_recon_loss_history())
            TkUI.set_series("x_axis_orderbook_recon", "y_axis_orderbook_recon", "orderbook_recon_accuracy_series", orderbook_training_history.recon_accuracy_history())
            TkUI.set_series("x_axis_orderbook_recon_epoch", "y_axis_orderbook_recon_epoch", "orderbook_recon_accuracy_series_epoch", orderbook_training_history.epoch_recon_accuracy_history())

            TkUI.set_series("x_axis_last_trades_codebook", "y_axis_last_trades_codebook", "last_trades_active_codes_series", last_trades_training_history.active_codes())
            TkUI.set_series("x_axis_last_trades_codebook", "y_axis_last_trades_codebook", "last_trades_dead_codes_series", last_trades_training_history.dead_codes())
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

        torch.cuda.empty_cache()
        
        cuda_memory_allocated.append( torch.cuda.memory_allocated() / 1024**2 )
        if len( cuda_memory_allocated ) > 32 :
            del cuda_memory_allocated[0]

        print( torch.cuda.memory_reserved() / 1024**2, "/", cuda_memory_allocated[-1], ":", cuda_memory_leak_per_loop(), "MB")

        if cuda_memory_allocated[-1] > 9998:
            print( "Forced termination.")
            break
        

    dpg.destroy_context()

    data_loader.complete_loading()
    data_loader.close()

    save_orderbook_autoencoder()
    save_last_trades_autoencoder()