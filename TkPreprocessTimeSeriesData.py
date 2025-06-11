
import os
import time
import numpy as np
import configparser
import json
import copy
import random
import math
from os import listdir
from os.path import isfile, join
from datetime import date, datetime, timezone, timedelta
from dateutil import parser
import dearpygui.dearpygui as dpg
import itertools
import threading
import torch
from collections import defaultdict
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

#------------------------------------------------------------------------------------------------------------------------

class TkTimeSeriesDataPreprocessor():

    def __init__(self, _cfg : configparser.ConfigParser):

        self._cuda = torch.device("cuda")

        orderbook_model_path = join( _cfg['Paths']['ModelsPath'], _cfg['Paths']['OrderbookAutoencoderModelFileName'] )
        
        if not os.path.isfile(orderbook_model_path):
            raise RuntimeError('Orderbook autoencoder model is missing!')

        self._orderbook_autoencoder = TkOrderbookAutoencoder(_cfg)
        self._orderbook_autoencoder.to(self._cuda)
        self._orderbook_autoencoder.load_state_dict(torch.load(orderbook_model_path))
        self._orderbook_autoencoder.eval()

        last_trades_model_path = join( _cfg['Paths']['ModelsPath'], _cfg['Paths']['LastTradesAutoencoderModelFileName'] )

        if not os.path.isfile(last_trades_model_path):
            raise RuntimeError('Last trades autoencoder model is missing!')

        self._last_trades_autoencoder = TkLastTradesAutoencoder(_cfg)
        self._last_trades_autoencoder.to(self._cuda)
        self._last_trades_autoencoder.load_state_dict(torch.load(last_trades_model_path))        
        self._last_trades_autoencoder.eval()

        self._orderbook_width = int(_cfg['Autoencoders']['OrderBookWidth'])
        self._last_trades_width = int(_cfg['Autoencoders']['LastTradesWidth'])
        self._min_price_increment_factor = int(_cfg['Autoencoders']['MinPriceIncrementFactor'])        
        self._test_data_ratio = float(_cfg['Autoencoders']['TestDataRatio'])
        self._orderbook_autoencoder_code_layer_size = int(_cfg['Autoencoders']['OrderbookAutoencoderCodeLayerSize'])
        self._last_trades_autoencoder_code_layer_size = int(_cfg['Autoencoders']['LastTradesAutoencoderCodeLayerSize'])

        self._data_path = _cfg['Paths']['DataPath']
        self._time_series_index_filename = _cfg['Paths']['TimeSeriesIndexFileName']
        self._time_series_training_data_filename = _cfg['Paths']['TimeSeriesTrainingDataFileName']
        self._time_series_test_data_filename = _cfg['Paths']['TimeSeriesTestDataFileName']

        self._prior_steps_count = int(_cfg['TimeSeries']['PriorStepsCount'])
        self._future_steps_count = int(_cfg['TimeSeries']['FutureStepsCount'])
        self._priority_tail_epsilon = float(_cfg['TimeSeries']['PriorityTailEpsilon'])
        self._priority_tail_threshold = float(_cfg['TimeSeries']['PriorityTailThreshold'])

        self._priority_table = []
        self._regular_table = []
        self._training_index = []
        self._test_index = []
        self._training_data_offset = 0
        self._test_data_offset = 0
        self._training_data_stream = open( join(self._data_path, self._time_series_training_data_filename), 'wb+')
        self._test_data_stream = open( join(self._data_path, self._time_series_test_data_filename), 'wb+')

    def num_training_samples(self):
        return len(self._training_index)

    def num_priority_samples(self):
        return len(self._priority_table)

    def num_regular_samples(self):
        return len(self._regular_table)

    def num_test_samples(self):
        return len(self._test_index)

    def flush(self):
        TkIO.write_at_path( join(self._data_path, self._time_series_index_filename), self._priority_table )
        TkIO.append_at_path( join(self._data_path, self._time_series_index_filename), self._regular_table )
        TkIO.append_at_path( join(self._data_path, self._time_series_index_filename), self._training_index )
        TkIO.append_at_path( join(self._data_path, self._time_series_index_filename), self._test_index )
        self._training_data_stream.close()
        self._test_data_stream.close()

    def add_samples(self, share : TkInstrument, raw_samples : list, is_test_data_source : bool, render_callback):

        min_price_increment = quotation_to_float( share.min_price_increment() )
        
        raw_sample_count = int( len(raw_samples) / 2 ) # [ orderbook, last_trades, .... ]

        if raw_sample_count < self._prior_steps_count + self._future_steps_count:
            print( 'Not enough sample count: ', raw_sample_count )
            return

        # encode orderbook & last trades

        price = [0.0] * raw_sample_count
        orderbook_volume = [0] * raw_sample_count
        last_trades_volume = [0] * raw_sample_count
        orderbook = [None] * raw_sample_count
        last_trades = [None] * raw_sample_count

        for i in range( raw_sample_count ):

            orderbook_sample = raw_samples[i*2]
            distribution, descriptor, volume, pivot_price = TkStatistics.orderbook_distribution( orderbook_sample, self._orderbook_width, min_price_increment * self._min_price_increment_factor )
            if volume > 0:
                distribution *= 1.0 / volume
            price[i] = quotation_to_float( orderbook_sample.last_price )
            orderbook_volume[i] = volume
            orderbook[i] = distribution

            last_trades_sample = raw_samples[i*2+1]
            distribution, descriptor, volume = TkStatistics.trades_distribution( last_trades_sample, pivot_price, self._last_trades_width, min_price_increment * self._min_price_increment_factor )
            if volume > 0:
                distribution *= 1.0 / volume
            last_trades_volume[i] = volume
            last_trades[i] = distribution

        orderbook_input = torch.Tensor( np.concatenate( orderbook ) )
        orderbook_input = torch.reshape( orderbook_input, ( raw_sample_count, 1, self._orderbook_width ) )
        orderbook_input = orderbook_input.cuda()
        orderbook_code = self._orderbook_autoencoder.encode(orderbook_input)
        orderbook_code = torch.reshape( orderbook_code, (raw_sample_count, self._orderbook_autoencoder_code_layer_size ) )
        orderbook_code = orderbook_code.tolist()

        last_trades_input = torch.Tensor( np.concatenate( last_trades ) )
        last_trades_input = torch.reshape( last_trades_input, ( raw_sample_count, 1, self._last_trades_width ) )
        last_trades_input = last_trades_input.cuda()
        last_trades_code = self._last_trades_autoencoder.encode(last_trades_input)
        last_trades_code = torch.reshape( last_trades_code, (raw_sample_count, self._last_trades_autoencoder_code_layer_size ) )
        last_trades_code = last_trades_code.tolist()

        start_range = self._prior_steps_count - 1
        end_range = raw_sample_count - self._future_steps_count - 1
        range_len = end_range - start_range

        # encode future trades distribution

        future_trades = [None] * (end_range)
        future_trades_descriptor = [None] * (end_range)
        future_trades_volume = [None] * (end_range)

        for i in range( start_range, end_range ):
            ts_base_price = price[i]
            last_trades_sample = raw_samples[(i+1)*2+1]

            ts_target_distribution, ts_target_descriptor, ts_target_volume = TkStatistics.trades_distribution( last_trades_sample, ts_base_price, self._last_trades_width, min_price_increment * self._min_price_increment_factor )

            for j in range( 1, self._future_steps_count ):
                last_trades_sample = raw_samples[(i+j+1)*2+1]
                ts_target_volume = TkStatistics.accumulate_trades_distribution( ts_target_distribution, ts_target_descriptor, ts_target_volume, last_trades_sample, ts_base_price)

            if ts_target_volume > 0:
                ts_target_distribution *= 1.0 / ts_target_volume

            future_trades_descriptor[i] = ts_target_descriptor
            future_trades_volume[i] = ts_target_volume
            future_trades[i] = ts_target_distribution

        future_trades_input = [future_trades[i] for i in range(start_range, end_range)]
        future_trades_input = torch.Tensor( np.concatenate( future_trades_input ) )
        future_trades_input = torch.reshape( future_trades_input, ( (end_range-start_range), 1, self._last_trades_width ) )
        future_trades_input = future_trades_input.cuda()
        future_trades_code = self._last_trades_autoencoder.encode(future_trades_input)
        future_trades_code = torch.reshape( future_trades_code, ( (end_range-start_range), self._last_trades_autoencoder_code_layer_size ) )
        future_trades_code = future_trades_code.tolist()

        for i in range( start_range ):
            future_trades_code.insert( 0, None )

        # combine data samples

        callback_indices = [start_range + int(i / 10.0 * range_len) for i in range(1,10)]

        num_invalid_samples = 0

        for i in range( start_range, end_range ):

            ts_input = [None] * self._prior_steps_count
            ts_base_price = price[i]

            i0 = i-self._prior_steps_count+1
            i1 = i+1
            ts_base_orderbook_volume = sum( orderbook_volume[i0:i1] ) / self._prior_steps_count
            ts_base_last_trades_volume = sum( last_trades_volume[i0:i1] ) / self._prior_steps_count

            for j in range( self._prior_steps_count ):
                k = i-self._prior_steps_count+j+1

                ts_input[j] = orderbook_code[k].copy()
                ts_input[j].extend( last_trades_code[k].copy() )

                ts_sample_price = price[k]
                ts_sample_price = ( ts_sample_price / ts_base_price - 1.0 ) * 100
                ts_input[j].append( ts_sample_price )

                ts_sample_orderbook_volume = orderbook_volume[k]
                ts_sample_orderbook_volume = ( ts_sample_orderbook_volume / ts_base_orderbook_volume - 1.0 ) * 100 if ( ts_base_orderbook_volume > 0 ) else 0.0
                ts_input[j].append( ts_sample_orderbook_volume )

                ts_sample_last_trades_volume = last_trades_volume[k]
                ts_sample_last_trades_volume = ( ts_sample_last_trades_volume / ts_base_last_trades_volume - 1.0 ) * 100 if ( ts_base_last_trades_volume > 0 ) else 0.0
                ts_input[j].append( ts_sample_last_trades_volume )

            ts_input = list( itertools.chain.from_iterable(ts_input) )

            # ignore sample if volume of future trades is zero

            if future_trades_volume[i] > 0:

                ts_target_code = future_trades_code[i].copy()
                ts_target = future_trades[i].tolist()

                verbalize = True
                if is_test_data_source:
                    self._test_index.append( self._test_data_offset )
                    TkIO.write_to_file( self._test_data_stream, ts_input )
                    TkIO.write_to_file( self._test_data_stream, ts_target_code )
                    TkIO.write_to_file( self._test_data_stream, ts_target )
                    self._test_data_offset = self._test_data_stream.tell()
                else:
                    ts_target_left_tail, ts_target_right_tail = TkStatistics.get_distribution_tails( ts_target_distribution, ts_target_descriptor, self._priority_tail_epsilon )
                    ts_threshold_price = self._priority_tail_threshold * 0.5 * (ts_target_descriptor[-1][0] + ts_target_descriptor[-1][1])
                    is_priority_sample = ( ts_target_left_tail <= -ts_threshold_price ) or ( ts_target_right_tail >= ts_threshold_price )
                    if is_priority_sample:
                        self._priority_table.append( len(self._training_index) )
                    else:
                        self._regular_table.append( len(self._training_index) )
                    self._training_index.append( self._training_data_offset )
                    TkIO.write_to_file( self._training_data_stream, ts_input )
                    TkIO.write_to_file( self._training_data_stream, ts_target_code )
                    TkIO.write_to_file( self._training_data_stream, ts_target )
                    self._training_data_offset = self._training_data_stream.tell()
                    verbalize = is_priority_sample

                if len(callback_indices) > 0 and i >= callback_indices[0] and verbalize:
                    del callback_indices[0]
                    if render_callback != None:
                        render_callback( ts_input, ts_target )
            else:
                num_invalid_samples = num_invalid_samples + 1

#------------------------------------------------------------------------------------------------------------------------

def date_from_filename(filename:str): # TICKER_Month_Day_Year_Anchor.obs
    date_str = filename[ filename.find("_") + 1: ]
    date_str = date_str[ 0 : date_str.find(".") ]
    date_anchor = date_str[ date_str.rfind("_") + 1: ]
    date_str = date_str[ 0 : date_str.rfind("_") ]
    result = datetime.strptime( date_str,'%B_%d_%Y' )
    if date_anchor == 'Evening':
        result = result + timedelta(hours=19)
    else:
        result = result + timedelta(hours=10)
    return result

def ticker_from_filename(filename:str):
    return filename[ 0: filename.find("_") ]

def group_by_ticker(filenames:list):
    result = defaultdict(list)
    for filename in filenames:
        ticker = ticker_from_filename(filename)
        date = date_from_filename(filename)
        result[ticker].append( (date, filename) )

    for key in result:
        files = result[key]
        files = sorted( files, key=lambda x: x[0] )
        result[key] = files

    return result

#------------------------------------------------------------------------------------------------------------------------    

TOKEN = os.environ["TK_TOKEN"]

config = configparser.ConfigParser()
config.read( 'TkConfig.ini' )

data_path = config['Paths']['DataPath']
data_extension = config['Paths']['OrderbookFileExtension']
test_data_ratio = float(config['TimeSeries']['TestDataRatio'])

data_files = [filename for filename in listdir(data_path) if (data_extension in filename) and isfile(join(data_path, filename))]
print( 'Data files found:', len(data_files) )

files_by_ticker = group_by_ticker(data_files)
print( 'Tickers found:', len(files_by_ticker) )

preprocessor = TkTimeSeriesDataPreprocessor( config )

with Client(TOKEN, target=INVEST_GRPC_API) as client:

    dpg.create_context()
    dpg.create_viewport(title='Data preprocessor', width=1572, height=768)
    dpg.setup_dearpygui()

    with dpg.window(tag="primary_window", label="Preprocess data"):
        with dpg.group(horizontal=True):
            dpg.add_text( default_value="Files processed: " )
            dpg.add_text( tag="files_processed", default_value="0/0", color=[255, 254, 255])
        with dpg.group(horizontal=True):
            dpg.add_text( default_value="Samples processed (total/regular/priority): " )
            dpg.add_text( tag="samples_processed", default_value="0/0/0", color=[255, 254, 255])
        with dpg.group(horizontal=True):
            dpg.add_text( default_value="Filename: " )
            dpg.add_text( tag="filename", default_value="", color=[255, 254, 255])
        with dpg.group(horizontal=True):
            with dpg.plot(label="Input", width=1024, height=256):
                dpg.add_plot_legend()
                dpg.add_plot_axis(dpg.mvXAxis, tag="x_axis_input" )
                dpg.add_plot_axis(dpg.mvYAxis, tag="y_axis_input" )
                dpg.add_line_series( [j for j in range(0, 32)], [random.random() for j in range(0, 32)], label="Input", parent="x_axis_input", tag="input_series" )
            with dpg.plot(label="Target", width=512, height=256):
                dpg.add_plot_legend()
                dpg.add_plot_axis(dpg.mvXAxis, tag="x_axis_target" )
                dpg.add_plot_axis(dpg.mvYAxis, tag="y_axis_target" )
                dpg.add_line_series( [j for j in range(0, 32)], [random.random() for j in range(0, 32)], label="Target", parent="x_axis_target", tag="target_series" )

    dpg.show_viewport()
    dpg.set_primary_window("primary_window", True)

    def render_samples( ts_input : list, ts_target : list ):
        TkUI.set_series("x_axis_input","y_axis_input","input_series", ts_input)
        TkUI.set_series("x_axis_target","y_axis_target","target_series", ts_target)
        dpg.render_dearpygui_frame()

    total_samples = 0
    files_processed = 0
    start_time = time.time()
    for ticker in files_by_ticker:

        share = TkInstrument(client, config,  InstrumentType.INSTRUMENT_TYPE_SHARE, ticker, "TQBR")

        num_data_sources = len(files_by_ticker[ticker])
        num_test_data_sources = max(1, int( num_data_sources * test_data_ratio ))
        num_training_data_sources = num_data_sources - num_test_data_sources

        for i in range(num_data_sources):

            date_and_filename = files_by_ticker[ticker][i]
            date = date_and_filename[0]
            filename = date_and_filename[1]
            is_test_data_source = i+1 >= num_training_data_sources
        
            dpg.set_value("filename", filename)

            dpg.render_dearpygui_frame()
            if not dpg.is_dearpygui_running():
                break
        
            raw_samples = TkIO.read_at_path( join( data_path, filename) )

            preprocessor.add_samples(share, raw_samples, is_test_data_source, render_samples)

            total_samples = total_samples + int( len( raw_samples ) / 2 )
            files_processed = files_processed + 1
            
            num_training_samples = preprocessor.num_training_samples()
            num_regular_samples = preprocessor.num_regular_samples()
            num_priority_samples = preprocessor.num_priority_samples()
            num_test_samples = preprocessor.num_test_samples()
            dpg.set_value("files_processed", str(files_processed)+"/"+str(len(data_files)))
            dpg.set_value("samples_processed", str(num_training_samples)+"/"+str(num_regular_samples)+"/"+str(num_priority_samples)+"/"+str(num_test_samples))

            dpg.render_dearpygui_frame()
            if not dpg.is_dearpygui_running():
                break

    end_time = time.time()
    print('Elapsed time:',end_time-start_time)
    
    dpg.set_value("filename", '...all is done!')
    dpg.render_dearpygui_frame()

    preprocessor.flush()

    while dpg.is_dearpygui_running():
        dpg.render_dearpygui_frame()

    dpg.destroy_context()