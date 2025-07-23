
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
from LSHash import LSHash

#------------------------------------------------------------------------------------------------------------------------

class TkAutoencoderDataPreprocessor():

    def __init__(self, _cfg : configparser.ConfigParser):
        self._orderbook_width = int(_cfg['Autoencoders']['OrderBookWidth'])
        self._last_trades_width = int(_cfg['Autoencoders']['LastTradesWidth'])
        self._min_price_increment_factor = int(_cfg['Autoencoders']['MinPriceIncrementFactor'])
        self._lshash_size = int(_cfg['Autoencoders']['LSHashSize'])
        self._orderbook_sample_similarity = float(_cfg['Autoencoders']['OrderBookSampleSimilarity'])
        self._last_trades_sample_similarity = float(_cfg['Autoencoders']['LastTradesSampleSimilarity'])
        self._synthetic_orderbook_sample_similarity = float(_cfg['Autoencoders']['SyntheticOrderBookSampleSimilarity'])
        self._synthetic_last_trades_sample_similarity = float(_cfg['Autoencoders']['SyntheticLastTradesSampleSimilarity'])
        self._future_steps_count = int(_cfg['TimeSeries']['FutureStepsCount'])

        self._data_path = _cfg['Paths']['DataPath']
        self._orderbook_index_filename = _cfg['Paths']['OrderBookIndexFileName']
        self._orderbook_training_data_filename = _cfg['Paths']['OrderBookTrainingDataFileName']
        self._orderbook_test_data_filename = _cfg['Paths']['OrderBookTestDataFileName']

        self._last_trades_index_filename = _cfg['Paths']['LastTradesIndexFileName']
        self._last_trades_training_data_filename = _cfg['Paths']['LastTradesTrainingDataFileName']
        self._last_trades_test_data_filename = _cfg['Paths']['LastTradesTestDataFileName']

        self._orderbook_training_index = []
        self._orderbook_test_index = []
        self._orderbook_training_data_offset = 0
        self._orderbook_test_data_offset = 0
        self._orderbook_training_data_stream = open( join(self._data_path, self._orderbook_training_data_filename), 'wb+')
        self._orderbook_test_data_stream = open( join(self._data_path, self._orderbook_test_data_filename), 'wb+')

        self._last_trades_training_index = []
        self._last_trades_test_index = []
        self._last_trades_training_data_offset = 0
        self._last_trades_test_data_offset = 0
        self._last_trades_training_data_stream = open( join(self._data_path, self._last_trades_training_data_filename), 'wb+')
        self._last_trades_test_data_stream = open( join(self._data_path, self._last_trades_test_data_filename), 'wb+')

        self._normalized_orderbook_samples = []
        self._normalized_last_trades_samples = []

        self._orderbook_lsh = LSHash(self._lshash_size, self._orderbook_width)
        self._last_trades_lsh = LSHash(self._lshash_size, self._last_trades_width)

    def clear_lsh(self):
        self._orderbook_lsh = LSHash(self._lshash_size, self._orderbook_width)
        self._last_trades_lsh = LSHash(self._lshash_size, self._last_trades_width)

    def num_orderbook_samples(self):
        return len(self._orderbook_training_index) + len(self._orderbook_test_index)

    def num_last_trades_samples(self):
        return len(self._last_trades_training_index) + len(self._last_trades_test_index)

    def write_samples(self, samples, index, offset, stream):
        for i in range(len(samples)):            
            index.append(offset)
            TkIO.write_to_file(stream, samples[i])
            offset = stream.tell()            
        return offset

    def flush(self):
        TkIO.write_at_path( join(self._data_path, self._orderbook_index_filename), self._orderbook_training_index )
        TkIO.append_at_path( join(self._data_path, self._orderbook_index_filename), self._orderbook_test_index )
        TkIO.write_at_path( join(self._data_path, self._last_trades_index_filename), self._last_trades_training_index )
        TkIO.append_at_path( join(self._data_path, self._last_trades_index_filename), self._last_trades_test_index )
        self._orderbook_training_data_stream.close()
        self._orderbook_test_data_stream.close()
        self._last_trades_training_data_stream.close()
        self._last_trades_test_data_stream.close()

    def add_samples(self, share : TkInstrument, raw_samples : list, training_category : bool, render_callback):

        min_price_increment = quotation_to_float( share.min_price_increment() )
        raw_sample_count = int( len(raw_samples) / 2 ) # orderbook, last_trades

        self._normalized_orderbook_samples = []
        self._normalized_last_trades_samples = []

        callback_indices = [int(i / 100.0 * raw_sample_count) for i in range(1,100)]

        for i in range(raw_sample_count):
            orderbook_sample = raw_samples[i*2]
            distribution, descriptor, volume, pivot_price = TkStatistics.orderbook_distribution( orderbook_sample, self._orderbook_width, min_price_increment * self._min_price_increment_factor ) 
            if volume > 0:
                distribution *= 1.0 / volume
            lsh_query = self._orderbook_lsh.query( distribution, num_results=1 )
            if len(lsh_query) == 0 or lsh_query[0][1] > self._orderbook_sample_similarity:
                self._normalized_orderbook_samples.append(distribution)
                self._orderbook_lsh.index(distribution)

            # last trades sample
            last_trades_sample = raw_samples[i*2+1]
            distribution, descriptor, volume = TkStatistics.trades_distribution( last_trades_sample, pivot_price, self._last_trades_width, min_price_increment * self._min_price_increment_factor )
            if volume > 0:
                distribution *= 1.0 / volume
            lsh_query = self._last_trades_lsh.query( distribution, num_results=1 )
            if len(lsh_query) == 0 or lsh_query[0][1] > self._last_trades_sample_similarity:
                self._normalized_last_trades_samples.append(distribution)
                self._last_trades_lsh.index(distribution)

            # accumulated last trades sample (time series output)
            if i + self._future_steps_count < raw_sample_count / 2:
                for j in range( 1, self._future_steps_count ):
                    last_trades_sample = raw_samples[(i+j+1)*2+1]
                    volume = TkStatistics.accumulate_trades_distribution( distribution, descriptor, volume, last_trades_sample, pivot_price)
                if volume > 0:
                    distribution *= 1.0 / volume
                lsh_query = self._last_trades_lsh.query( distribution, num_results=1 )
                if len(lsh_query) == 0 or lsh_query[0][1] > self._last_trades_sample_similarity:
                    self._normalized_last_trades_samples.append(distribution)
                    self._last_trades_lsh.index(distribution)

            if len(callback_indices) > 0 and i >= callback_indices[0]:
                del callback_indices[0]
                if render_callback != None:
                    render_callback( self._normalized_orderbook_samples, self._normalized_last_trades_samples )

        if training_category:                    
            self._orderbook_training_data_offset = self.write_samples( 
                self._normalized_orderbook_samples,
                self._orderbook_training_index, 
                self._orderbook_training_data_offset, 
                self._orderbook_training_data_stream
            )
            self._last_trades_training_data_offset = self.write_samples(
                self._normalized_last_trades_samples, 
                self._last_trades_training_index, 
                self._last_trades_training_data_offset, 
                self._last_trades_training_data_stream
            )
        else:
            self._orderbook_test_data_offset = self.write_samples( 
                self._normalized_orderbook_samples,
                self._orderbook_test_index, 
                self._orderbook_test_data_offset, 
                self._orderbook_test_data_stream
            )
            self._last_trades_test_data_offset = self.write_samples(
                self._normalized_last_trades_samples, 
                self._last_trades_test_index, 
                self._last_trades_test_data_offset, 
                self._last_trades_test_data_stream
            )

    def add_preprocessed_samples(self, orderbook_samples : list, last_trades_samples : list, training_category : bool, render_callback):

        self._normalized_orderbook_samples = []
        self._normalized_last_trades_samples = []

        callback_indices = [int(i / 100.0 * len(orderbook_samples)) for i in range(1,100)]

        for i in range(len(orderbook_samples)):
            distribution = orderbook_samples[i]            
            lsh_query = self._orderbook_lsh.query( distribution, num_results=1 )
            if len(lsh_query) == 0 or lsh_query[0][1] > self._orderbook_sample_similarity:
                self._normalized_orderbook_samples.append(distribution)
                self._orderbook_lsh.index(distribution)
            if len(callback_indices) > 0 and i >= callback_indices[0]:
                del callback_indices[0]
                if render_callback != None:
                    render_callback( self._normalized_orderbook_samples, self._normalized_last_trades_samples )

        callback_indices = [int(i / 100.0 * len(last_trades_samples)) for i in range(1,100)]

        for i in range(len(last_trades_samples)):
            distribution = last_trades_samples[i]            
            lsh_query = self._last_trades_lsh.query( distribution, num_results=1 )
            if len(lsh_query) == 0 or lsh_query[0][1] > self._last_trades_sample_similarity:
                self._normalized_last_trades_samples.append(distribution)
                self._last_trades_lsh.index(distribution)
            if len(callback_indices) > 0 and i >= callback_indices[0]:
                del callback_indices[0]
                if render_callback != None:
                    render_callback( self._normalized_orderbook_samples, self._normalized_last_trades_samples )

        if training_category:                    
            self._orderbook_training_data_offset = self.write_samples( 
                self._normalized_orderbook_samples,
                self._orderbook_training_index, 
                self._orderbook_training_data_offset, 
                self._orderbook_training_data_stream
            )
            self._last_trades_training_data_offset = self.write_samples(
                self._normalized_last_trades_samples, 
                self._last_trades_training_index, 
                self._last_trades_training_data_offset, 
                self._last_trades_training_data_stream
            )
        else:
            self._orderbook_test_data_offset = self.write_samples( 
                self._normalized_orderbook_samples,
                self._orderbook_test_index, 
                self._orderbook_test_data_offset, 
                self._orderbook_test_data_stream
            )
            self._last_trades_test_data_offset = self.write_samples(
                self._normalized_last_trades_samples, 
                self._last_trades_test_index, 
                self._last_trades_test_data_offset, 
                self._last_trades_test_data_stream
            )

    def generate_synthetic_samples(self, training_category : bool, render_callback):

        synthetic_sample_ratio = float(config['Autoencoders']['SyntheticSampleRatio'])

        if synthetic_sample_ratio <= 0.0:
            return

        num_orderbook_training_samples = len(self._orderbook_training_index)
        num_orderbook_test_samples = len(self._orderbook_test_index)
        num_last_trades_training_samples = len(self._last_trades_training_index)
        num_last_trades_test_samples = len(self._last_trades_test_index)

        num_samples = 0

        if training_category:
            num_samples = int( synthetic_sample_ratio * max( num_orderbook_training_samples, num_last_trades_training_samples ) )
        else:
            num_samples = int( synthetic_sample_ratio * max( num_orderbook_test_samples, num_last_trades_test_samples ) )
            if num_samples == 0:
                num_samples = int( synthetic_sample_ratio * max( num_orderbook_training_samples, num_last_trades_training_samples ) )

        synthetic_orderbook_scheme = json.loads(config['Autoencoders']['SyntheticOrderbookScheme'])
        if not type(synthetic_orderbook_scheme) is list:
            raise RuntimeError('SyntheticOrderbookScheme is expected to be a list of lists!')

        synthetic_last_trades_scheme = json.loads(config['Autoencoders']['SyntheticLastTradesScheme'])
        if not type(synthetic_last_trades_scheme) is list:
            raise RuntimeError('SyntheticLastTradesScheme is expected to be a list of lists!')

        synthetic_sample_bias = float(config['Autoencoders']['SyntheticSampleBias'])
        synthetic_orderbook_sample_central_bias = float(config['Autoencoders']['SyntheticOrderbookSampleCentralBias'])
        synthetic_last_trades_sample_max_variance = float(config['Autoencoders']['SyntheticLastTradesSampleMaxVariance'])        

        self._normalized_orderbook_samples = []
        self._normalized_last_trades_samples = []

        discarded_orderbook_samples = 0
        discarded_last_trades_samples = 0

        callback_indices = [int(i / 100.0 * num_samples) for i in range(1,100)]

        for i in range(num_samples):
            orderbook_scheme = synthetic_orderbook_scheme[random.randint(0, len(synthetic_orderbook_scheme)-1)]
            distribution = np.zeros( self._orderbook_width, dtype=float)
            distribution[int(self._orderbook_width/2)-2] = random.uniform(0.0, synthetic_orderbook_sample_central_bias) # center synthetic orderbook sample to the maximal ask price
            TkStatistics.generate_distribution( distribution, orderbook_scheme, synthetic_sample_bias )
            TkStatistics.to_cumulative_distribution(distribution)
            lsh_query = self._orderbook_lsh.query( distribution, num_results=1 )
            if len(lsh_query) == 0 or lsh_query[0][1] > self._synthetic_orderbook_sample_similarity:
                self._normalized_orderbook_samples.append(distribution)
                self._orderbook_lsh.index(distribution)
            else:
                discarded_orderbook_samples = discarded_orderbook_samples + 1

            last_trades_scheme = synthetic_last_trades_scheme[random.randint(0, len(synthetic_last_trades_scheme)-1)]
            distribution = np.zeros( self._last_trades_width, dtype=float)
            TkStatistics.generate_clustered_distribution( distribution, last_trades_scheme, synthetic_sample_bias, synthetic_last_trades_sample_max_variance )
            lsh_query = self._last_trades_lsh.query( distribution, num_results=1 )
            if len(lsh_query) == 0 or lsh_query[0][1] > self._synthetic_last_trades_sample_similarity:
                self._normalized_last_trades_samples.append(distribution)
                self._last_trades_lsh.index(distribution)
            else:
                discarded_last_trades_samples = discarded_last_trades_samples + 1

            if len(callback_indices) > 0 and i >= callback_indices[0]:
                del callback_indices[0]
                if render_callback != None:
                    render_callback( self._normalized_orderbook_samples, self._normalized_last_trades_samples )

        print('Generated samples: ', num_samples-discarded_orderbook_samples, ',', num_samples-discarded_last_trades_samples )

        if render_callback != None:
            render_callback( self._normalized_orderbook_samples, self._normalized_last_trades_samples )

        if training_category:
            self._orderbook_training_data_offset = self.write_samples( 
                self._normalized_orderbook_samples,
                self._orderbook_training_index, 
                self._orderbook_training_data_offset, 
                self._orderbook_training_data_stream
            )
            self._last_trades_training_data_offset = self.write_samples(
                self._normalized_last_trades_samples, 
                self._last_trades_training_index, 
                self._last_trades_training_data_offset, 
                self._last_trades_training_data_stream
            )
        else:
            self._orderbook_test_data_offset = self.write_samples( 
                self._normalized_orderbook_samples,
                self._orderbook_test_index, 
                self._orderbook_test_data_offset, 
                self._orderbook_test_data_stream
            )
            self._last_trades_test_data_offset = self.write_samples(
                self._normalized_last_trades_samples, 
                self._last_trades_test_index, 
                self._last_trades_test_data_offset, 
                self._last_trades_test_data_stream
            )

#------------------------------------------------------------------------------------------------------------------------
# Main loop
#------------------------------------------------------------------------------------------------------------------------

TOKEN = os.environ["TK_TOKEN"]

config = configparser.ConfigParser()
config.read( 'TkConfig.ini' )

data_path = config['Paths']['DataPath']
data_extension = config['Paths']['OrderbookFileExtension']
test_data_ratio = float(config['Autoencoders']['TestDataRatio'])

data_files = [filename for filename in listdir(data_path) if (data_extension in filename) and isfile(join(data_path, filename))]
print( 'Data files found:', len(data_files) )

files_by_ticker = TkInstrument.group_by_ticker(data_files)
print( 'Tickers found:', len(files_by_ticker) )

preprocessor = TkAutoencoderDataPreprocessor( config )

with Client(TOKEN, target=INVEST_GRPC_API) as client:

    dpg.create_context()
    dpg.create_viewport(title='Data preprocessor', width=1572, height=768)
    dpg.setup_dearpygui()

    with dpg.window(tag="primary_window", label="Preprocess data"):
        with dpg.group(horizontal=True):
            dpg.add_text( default_value="Files processed: " )
            dpg.add_text( tag="files_processed", default_value="0/0", color=[255, 254, 255])
        with dpg.group(horizontal=True):
            dpg.add_text( default_value="Orderbook samples: " )
            dpg.add_text( tag="orderbook_samples", default_value="0/0", color=[255, 254, 255])
        with dpg.group(horizontal=True):
            dpg.add_text( default_value="Last trades samples: " )
            dpg.add_text( tag="last_trades_samples", default_value="0/0", color=[255, 254, 255])
        with dpg.group(horizontal=True):
            dpg.add_text( default_value="Filename: " )
            dpg.add_text( tag="filename", default_value="", color=[255, 254, 255])
        with dpg.group(horizontal=True):
            with dpg.plot(label="Orderbook", width=512, height=256):
                dpg.add_plot_legend()
                dpg.add_plot_axis(dpg.mvXAxis, tag="x_axis_orderbook" )
                dpg.add_plot_axis(dpg.mvYAxis, tag="y_axis_orderbook" )
                dpg.add_line_series( [j for j in range(0, 32)], [random.random() for j in range(0, 32)], label="Orderbook", parent="x_axis_orderbook", tag="orderbook_series" )
            with dpg.plot(label="Last trades", width=512, height=256):
                dpg.add_plot_legend()
                dpg.add_plot_axis(dpg.mvXAxis, tag="x_axis_last_trades" )
                dpg.add_plot_axis(dpg.mvYAxis, tag="y_axis_last_trades" )
                dpg.add_line_series( [j for j in range(0, 32)], [random.random() for j in range(0, 32)], label="Last trades", parent="x_axis_last_trades", tag="last_trades_series" )

    dpg.show_viewport()
    dpg.set_primary_window("primary_window", True)

    def render_samples( orderbook_samples, last_trades_samples ):
        if orderbook_samples:
            TkUI.set_series("x_axis_orderbook","y_axis_orderbook","orderbook_series", orderbook_samples[-1].tolist())
        if last_trades_samples:
            TkUI.set_series("x_axis_last_trades","y_axis_last_trades","last_trades_series", last_trades_samples[-1].tolist())
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

            preprocessor.add_samples(share, raw_samples, not is_test_data_source, render_samples)            

            total_samples = total_samples + int( len( raw_samples ) / 2 )
            files_processed = files_processed + 1

            dpg.set_value("files_processed", str(files_processed)+"/"+str(len(data_files)))
            dpg.set_value("orderbook_samples", str(preprocessor.num_orderbook_samples())+"/"+str(total_samples) )
            dpg.set_value("last_trades_samples", str(preprocessor.num_last_trades_samples())+"/"+str(total_samples) )

            dpg.render_dearpygui_frame()
            if not dpg.is_dearpygui_running():
                break

    preprocessor.clear_lsh()

    end_time = time.time()
    print('Elapsed time:',end_time-start_time)

    if dpg.is_dearpygui_running():

        dpg.set_value("filename", 'Generating synthetic samples...')
        dpg.render_dearpygui_frame()
    
        preprocessor.generate_synthetic_samples( True, render_samples )
        preprocessor.clear_lsh()
        preprocessor.generate_synthetic_samples( False, render_samples )
        dpg.render_dearpygui_frame()

        dpg.set_value("orderbook_samples", str(preprocessor.num_orderbook_samples())+"/"+str(total_samples) )
        dpg.set_value("last_trades_samples", str(preprocessor.num_last_trades_samples())+"/"+str(total_samples) )
        dpg.set_value("filename", '...all is done!')
        dpg.render_dearpygui_frame()

    preprocessor.flush()

    while dpg.is_dearpygui_running():
        dpg.render_dearpygui_frame()

    dpg.destroy_context()