
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
import uuid
from os import listdir
from os.path import isfile, join
from datetime import date, datetime, timezone
from dateutil import parser
from timeit import default_timer
import dearpygui.dearpygui as dpg
import itertools
import threading
import multiprocessing
import multiprocessing.connection as mpc
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
from TkModules.TkOrderbookAutoencoder import TkOrderbookAutoencoder
from TkModules.TkTimeSeriesForecaster import TkTimeSeriesForecaster

#------------------------------------------------------------------------------------------------------------------------
# UI
#------------------------------------------------------------------------------------------------------------------------

class TkMainPanel():

    def __init__(self, windowTag:str, numSamples:int):
        self._tag = uuid.uuid1()
        self._windowTag = windowTag
        self._numSamples = numSamples

        self._labelGroupTag = str(uuid.uuid1())
        self._labelGroup = dpg.add_group(horizontal=True, tag=self._labelGroupTag, parent=self._windowTag)
        
        self._tickerLabelTag = str(uuid.uuid1())
        dpg.add_text( default_value="Ticker: ", parent=self._labelGroupTag )
        dpg.add_text( tag=self._tickerLabelTag, default_value="XYZW", color=[255, 254, 255], parent=self._labelGroupTag)

        self._groupTag = str(uuid.uuid1())
        self._group = dpg.add_group(horizontal=True, tag=self._groupTag, parent=self._windowTag)

        self._orderbookPlotTag = str(uuid.uuid1())
        self._orderbookPlot = dpg.add_plot( label='Orderbook', width=512, height=256, tag=self._orderbookPlotTag, parent=self._group)
        dpg.add_plot_legend(parent=self._orderbookPlot)
        self._orderbookXAxisTag = str(uuid.uuid1())
        dpg.add_plot_axis(dpg.mvXAxis, tag=self._orderbookXAxisTag, parent=self._orderbookPlot)
        self._orderbookYAxisTag = str(uuid.uuid1())
        dpg.add_plot_axis(dpg.mvYAxis, tag=self._orderbookYAxisTag, parent=self._orderbookPlot)
        self._orderbookSeriesTags = []
        for i in range(self._numSamples):                    
            orderbookSeriesTag = str(uuid.uuid1())
            self._orderbookSeriesTags.append(orderbookSeriesTag)
            dpg.add_line_series( [j for j in range(0, 32)], [random.random() for j in range(0, 32)], label='Orderbook ' + str(i), parent=self._orderbookXAxisTag, tag=orderbookSeriesTag )

    def setTicker(self, ticker:str):
        dpg.set_value( self._tickerLabelTag, ticker )

    def setOrderbook(self, index:int, orderbook:list, labels:list):
        if index >=0 and index < len(self._orderbookSeriesTags):
            dpg.set_value( self._orderbookSeriesTags[index], [labels,orderbook])
            dpg.fit_axis_data( self._orderbookXAxisTag )
            dpg.fit_axis_data( self._orderbookYAxisTag )

#------------------------------------------------------------------------------------------------------------------------

TOKEN = os.environ["TK_TOKEN"]

cuda = torch.device("cuda")

config = configparser.ConfigParser()
config.read( 'TkConfig.ini' )

data_path = config['Paths']['DataPath']
orderbook_model_path = join( config['Paths']['ModelsPath'], config['Paths']['OrderbookAutoencoderModelFileName'] )
last_trades_model_path = join( config['Paths']['ModelsPath'], config['Paths']['LastTradesAutoencoderModelFileName'] )
ts_model_path =  join( config['Paths']['ModelsPath'], config['Paths']['TimeSeriesModelFileName'] )

orderbook_width = int(config['Autoencoders']['OrderBookWidth'])
last_trades_width = int(config['Autoencoders']['LastTradesWidth'])
min_price_increment_factor = int(config['Autoencoders']['MinPriceIncrementFactor'])

prior_steps_count = int(config['TimeSeries']['PriorStepsCount'])
input_width = int(config['TimeSeries']['InputWidth'])
target_width = int(config['TimeSeries']['TargetWidth'])
input_slices = json.loads(config['TimeSeries']['InputSlices'])
display_slice = int(config['TimeSeries']['DisplaySlice'])
training_batch_size = int(config['TimeSeries']['TrainingBatchSize'])
test_batch_size = int(config['TimeSeries']['TestBatchSize'])
learning_rate = float(config['TimeSeries']['LearningRate'])
weight_decay = float(config['TimeSeries']['WeightDecay'])
history_size = int( config['TimeSeries']['HistorySize'] )

ipcAddress = config['IPC']['Address']
ipcPort = int(config['IPC']['Port'])
ipcAuthKey = bytes( config['IPC']['AuthKey'], 'ascii' )

#------------------------------------------------------------------------------------------------------------------------
# IPC Thread
# Queues the name of freshly updated data file got from TkGatherData.py process
#------------------------------------------------------------------------------------------------------------------------

ipcMessageQueue = []

def ipc_thread_func():
    print('IPC thread started')
    while True:    
        with mpc.Listener( (ipcAddress,ipcPort), authkey=ipcAuthKey ) as listener:
            with listener.accept() as conn:
                message = conn.recv()
                ipcMessageQueue.append( message )

ipc_thread = threading.Thread( target=ipc_thread_func )
ipc_thread.daemon = True
ipc_thread.start()

#------------------------------------------------------------------------------------------------------------------------
# Load samples from the given path
# The samples are arranged into list of tuples [(orderbook, last_trades), ...]
#------------------------------------------------------------------------------------------------------------------------

def load_samples(path:str, num_samples:int):
    raw_indices = TkIO.index_at_path(path)
    raw_sample_count = int( len(raw_indices) / 2 )

    if raw_sample_count >= num_samples:

        data_stream = open( path, 'rb+')

        try:        
            result = []
            for i in range(num_samples):
                iid = len(raw_indices) - num_samples * 2 + i * 2
                data_stream.seek( raw_indices[iid], 0 )
                orderbook_sample = TkIO.read_from_file( data_stream )
                data_stream.seek( raw_indices[iid+1], 0 )
                last_trades_sample = TkIO.read_from_file( data_stream )
                result.append( (orderbook_sample, last_trades_sample) )
            data_stream.close()
            return result
        except:
            data_stream.close()
            print('Error loading data from stream!')
            return None
    else:
        return None

#------------------------------------------------------------------------------------------------------------------------
# Convert orderbook & last trades samples to TkTimeSeriesForecaster input format
#------------------------------------------------------------------------------------------------------------------------

def preprocess_samples(instrument:TkInstrument, samples:list, orderbook_width:int, last_trades_width:int, min_price_increment_factor:int, orderbook_autoencoder:TkOrderbookAutoencoder, last_trades_autoencoder:TkLastTradesAutoencoder, main_panel:TkMainPanel):

    global cuda

    num_samples = len(samples)
    price = [0.0] * num_samples
    orderbook_volume = [0] * num_samples
    last_trades_volume = [0] * num_samples
    orderbook = [None] * num_samples
    last_trades = [None] * num_samples

    min_price_increment = quotation_to_float( instrument.min_price_increment() )

    for i in range( num_samples ):
        orderbook_sample = samples[i][0]
        distribution, descriptor, volume, pivot_price = TkStatistics.orderbook_distribution( orderbook_sample, orderbook_width, min_price_increment * min_price_increment_factor )
        if volume > 0:
            distribution *= 1.0 / volume
        price[i] = quotation_to_float( orderbook_sample.last_price )
        orderbook_volume[i] = volume
        orderbook[i] = distribution
        if main_panel != None:
            labels = [ 0.5 * (item[0] + item[1]) for item in descriptor]
            main_panel.setOrderbook( i, distribution.tolist(), labels )

        last_trades_sample = samples[i][1]
        distribution, descriptor, volume = TkStatistics.trades_distribution( last_trades_sample, pivot_price, last_trades_width, min_price_increment * min_price_increment_factor )
        if volume > 0:
            distribution *= 1.0 / volume
        last_trades_volume[i] = volume
        last_trades[i] = distribution

    orderbook_input = torch.Tensor( np.concatenate( orderbook ) )
    orderbook_input = torch.reshape( orderbook_input, ( num_samples, 1, orderbook_width ) )
    orderbook_input = orderbook_input.cuda()
    orderbook_code = orderbook_autoencoder.encode(orderbook_input)
    orderbook_code = torch.reshape( orderbook_code, (num_samples, orderbook_autoencoder.code_layer_size() ) )
    orderbook_code = orderbook_code.tolist()

    last_trades_input = torch.Tensor( np.concatenate( last_trades ) )
    last_trades_input = torch.reshape( last_trades_input, ( num_samples, 1, last_trades_width ) )
    last_trades_input = last_trades_input.cuda()
    last_trades_code = last_trades_autoencoder.encode(last_trades_input)
    last_trades_code = torch.reshape( last_trades_code, (num_samples, last_trades_autoencoder.code_layer_size() ) )
    last_trades_code = last_trades_code.tolist()

    base_price = price[-1]
    base_orderbook_volume = sum( orderbook_volume[0:num_samples] ) / num_samples
    base_last_trades_volume = sum( last_trades_volume[0:num_samples] ) / num_samples

    result = [None] * num_samples

    for i in range( num_samples ):
        result[i] = orderbook_code[i].copy()
        result[i].extend( last_trades_code[i].copy() )

        sample_price = price[i]
        sample_price = ( sample_price / base_price - 1.0 ) * 100
        result[i].append( sample_price )

        sample_orderbook_volume = orderbook_volume[i]
        sample_orderbook_volume = ( sample_orderbook_volume / base_orderbook_volume - 1.0 ) * 100 if ( base_orderbook_volume > 0 ) else 0.0
        result[i].append( sample_orderbook_volume )

        sample_last_trades_volume = last_trades_volume[i]
        sample_last_trades_volume = ( sample_last_trades_volume / base_last_trades_volume - 1.0 ) * 100 if ( base_last_trades_volume > 0 ) else 0.0
        result[i].append( sample_last_trades_volume )

    result = list( itertools.chain.from_iterable(result) )

    return result

#------------------------------------------------------------------------------------------------------------------------
# Runs TkTimeSeriesForecaster model
#------------------------------------------------------------------------------------------------------------------------

def forecast(input:list, prior_steps_count:int, input_width:int, model:TkTimeSeriesForecaster):

    global cuda

    input = torch.Tensor( input ) 
    input = torch.reshape( input, ( 1, prior_steps_count * input_width) )
    input = input.to(cuda)

    output = model.forward( input )
    return output 

#------------------------------------------------------------------------------------------------------------------------
# Main loop
#------------------------------------------------------------------------------------------------------------------------

print('Loading orderbook autoencoder...')
orderbook_autoencoder = TkOrderbookAutoencoder(config)
orderbook_autoencoder.to(cuda)
orderbook_autoencoder.load_state_dict(torch.load(orderbook_model_path))
orderbook_autoencoder.eval()

print('Loading last trades autoencoder...')
last_trades_autoencoder = TkLastTradesAutoencoder(config)
last_trades_autoencoder.to(cuda)
last_trades_autoencoder.load_state_dict(torch.load(last_trades_model_path))
last_trades_autoencoder.eval()

print('Loading timew series forecaster...')
time_series_forecaster = TkTimeSeriesForecaster(config)
time_series_forecaster.to(cuda)
time_series_forecaster.load_state_dict(torch.load(ts_model_path))
time_series_forecaster.eval()

with Client(TOKEN, target=INVEST_GRPC_API) as client:

    dpg.create_context()
    dpg.create_viewport(title='Forecasting service', width=1972, height=936)
    dpg.setup_dearpygui()

    main_window = dpg.add_window(tag="primary_window", label="Forecasting service")
    main_panel = TkMainPanel("primary_window", prior_steps_count)

    dpg.show_viewport()
    dpg.set_primary_window("primary_window", True)

    while dpg.is_dearpygui_running():
        
        if len(ipcMessageQueue) > 0:
            filename = ipcMessageQueue[0]
            del ipcMessageQueue[0]
            ticker = filename[ 0: filename.find("_") ]
            instrument = TkInstrument(client, config,  InstrumentType.INSTRUMENT_TYPE_SHARE, ticker, "TQBR")
            
            samples = load_samples( join( data_path, filename), prior_steps_count )
            if samples != None and len(samples) >= prior_steps_count:

                t0 = default_timer()
                input = preprocess_samples( instrument, samples, orderbook_width, last_trades_width, min_price_increment_factor, orderbook_autoencoder, last_trades_autoencoder, main_panel)
                preprocess_samples_time = default_timer() - t0

                t0 = default_timer()
                output = forecast(input, prior_steps_count, input_width, time_series_forecaster)
                forecast_time = default_timer() - t0

                main_panel.setTicker( filename + " / " + ticker + " / " + instrument.figi() + ", prep: " + str(preprocess_samples_time) + " forecast: " + str(forecast_time) )

            else:

                main_panel.setTicker( filename + " / " + ticker + " / " + instrument.figi() )

        dpg.render_dearpygui_frame()
        

    dpg.destroy_context()

