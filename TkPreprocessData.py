
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
        self._orderbook_width = int(config['Autoencoders']['OrderBookWidth'])
        self._last_trades_width = int(config['Autoencoders']['LastTradesWidth'])
        self._min_price_increment_factor = int(config['Autoencoders']['MinPriceIncrementFactor'])
        self._lshash_size = int(config['Autoencoders']['LSHashSize'])
        self._orderbook_sample_similarity = float(config['Autoencoders']['OrderBookSampleSimilarity'])
        self._last_trades_sample_similarity = float(config['Autoencoders']['LastTradesSampleSimilarity'])
        self._test_data_ratio = float(config['Autoencoders']['TestDataRatio'])

        self._data_path = config['Paths']['DataPath']
        self._orderbook_index_filename = config['Paths']['OrderBookIndexFileName']
        self._orderbook_training_data_filename = config['Paths']['OrderBookTrainingDataFileName']
        self._orderbook_test_data_filename = config['Paths']['OrderBookTestDataFileName']

        self._last_trades_index_filename = config['Paths']['LastTradesIndexFileName']
        self._last_trades_training_data_filename = config['Paths']['LastTradesTrainingDataFileName']
        self._last_trades_test_data_filename = config['Paths']['LastTradesTestDataFileName']

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

    def num_orderbook_samples(self):
        return len(self._orderbook_training_index) + len(self._orderbook_test_index)

    def num_last_trades_samples(self):
        return len(self._last_trades_training_index) + len(self._last_trades_test_index)

    def write_samples(self,samples, training_index, test_index, training_offset, test_offset, training_stream, test_stream):

        first_test_idx = int( len(samples) * (1.0 -self._test_data_ratio ) )

        for i in range(len(samples)):
            isTestSample = i > first_test_idx
            if isTestSample:
                test_index.append(test_offset)
                TkIO.write_to_file(test_stream, samples[i])
                test_offset = test_stream.tell()
            else:
                training_index.append(training_offset)
                TkIO.write_to_file(training_stream, samples[i])
                training_offset = training_stream.tell()
        return training_offset, test_offset        

    def flush(self):
        TkIO.write_at_path( join(self._data_path, self._orderbook_index_filename), self._orderbook_training_index )
        TkIO.append_at_path( join(self._data_path, self._orderbook_index_filename), self._orderbook_test_index )
        TkIO.write_at_path( join(self._data_path, self._last_trades_index_filename), self._last_trades_training_index )
        TkIO.append_at_path( join(self._data_path, self._last_trades_index_filename), self._last_trades_test_index )
        self._orderbook_training_data_stream.close()
        self._orderbook_test_data_stream.close()
        self._last_trades_training_data_stream.close()
        self._last_trades_test_data_stream.close()
        
    def add_samples(self, share : TkInstrument, raw_samples : list, render_callback):

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

            last_trades_sample = raw_samples[i*2+1]
            distribution, descriptor, volume = TkStatistics.trades_distribution( last_trades_sample, pivot_price, self._last_trades_width, min_price_increment * self._min_price_increment_factor )
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

        self._orderbook_training_data_offset, self._orderbook_test_data_offset = self.write_samples( 
            self._normalized_orderbook_samples,
            self._orderbook_training_index, 
            self._orderbook_test_index, 
            self._orderbook_training_data_offset, 
            self._orderbook_test_data_offset, 
            self._orderbook_training_data_stream,
            self._orderbook_test_data_stream,
        )

        self._last_trades_training_data_offset, self._last_trades_test_data_offset = self.write_samples(
            self._normalized_last_trades_samples, 
            self._last_trades_training_index, 
            self._last_trades_test_index, 
            self._last_trades_training_data_offset, 
            self._last_trades_test_data_offset, 
            self._last_trades_training_data_stream,
            self._last_trades_test_data_stream
        )

    def generate_synthetic_samples(self, render_callback):

        synthetic_sample_ratio = float(config['Autoencoders']['SyntheticSampleRatio'])
        num_samples = int( synthetic_sample_ratio * max( self.num_orderbook_samples(), self.num_last_trades_samples() ) )

        synthetic_orderbook_scheme = json.loads(config['Autoencoders']['SyntheticOrderbookScheme'])
        if not type(synthetic_orderbook_scheme) is list:
            raise RuntimeError('SyntheticOrderbookScheme is expected to be a list of lists!')

        synthetic_last_trades_scheme = json.loads(config['Autoencoders']['SyntheticLastTradesScheme'])
        if not type(synthetic_last_trades_scheme) is list:
            raise RuntimeError('SyntheticLastTradesScheme is expected to be a list of lists!')

        synthetic_sample_bias = float(config['Autoencoders']['SyntheticSampleBias'])

        self._normalized_orderbook_samples = []
        self._normalized_last_trades_samples = []

        discarded_orderbook_samples = 0
        discarded_last_trades_samples = 0

        callback_indices = [int(i / 100.0 * num_samples) for i in range(1,100)]

        for i in range(num_samples):
            orderbook_scheme = synthetic_orderbook_scheme[random.randint(0, len(synthetic_orderbook_scheme)-1)]
            distribution = np.zeros( self._orderbook_width, dtype=float)
            TkStatistics.generate_distribution( distribution, orderbook_scheme, synthetic_sample_bias )
            TkStatistics.to_cumulative_distribution(distribution)
            lsh_query = self._orderbook_lsh.query( distribution, num_results=1 )
            if len(lsh_query) == 0 or lsh_query[0][1] > self._orderbook_sample_similarity:
                self._normalized_orderbook_samples.append(distribution)
                self._orderbook_lsh.index(distribution)
            else:
                discarded_orderbook_samples = discarded_orderbook_samples + 1

            last_trades_scheme = synthetic_last_trades_scheme[random.randint(0, len(synthetic_last_trades_scheme)-1)]
            distribution = np.zeros( self._last_trades_width, dtype=float)
            TkStatistics.generate_distribution( distribution, last_trades_scheme, synthetic_sample_bias )
            lsh_query = self._last_trades_lsh.query( distribution, num_results=1 )
            if len(lsh_query) == 0 or lsh_query[0][1] > self._last_trades_sample_similarity:
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

        self._orderbook_training_data_offset, self._orderbook_test_data_offset = self.write_samples( 
            self._normalized_orderbook_samples,
            self._orderbook_training_index, 
            self._orderbook_test_index, 
            self._orderbook_training_data_offset, 
            self._orderbook_test_data_offset, 
            self._orderbook_training_data_stream,
            self._orderbook_test_data_stream,
        )

        self._last_trades_training_data_offset, self._last_trades_test_data_offset = self.write_samples(
            self._normalized_last_trades_samples, 
            self._last_trades_training_index, 
            self._last_trades_test_index, 
            self._last_trades_training_data_offset, 
            self._last_trades_test_data_offset, 
            self._last_trades_training_data_stream,
            self._last_trades_test_data_stream
        )

#------------------------------------------------------------------------------------------------------------------------

TOKEN = os.environ["TK_TOKEN"]

config = configparser.ConfigParser()
config.read( 'TkConfig.ini' )

data_path = config['Paths']['DataPath']
data_extension = config['Paths']['OrderbookFileExtension']

data_files = [filename for filename in listdir(data_path) if (data_extension in filename) and isfile(join(data_path, filename))]
random.shuffle(data_files)
# data_files = data_files[:5] # slice the rest in test purpose

print( 'Data files found:', len(data_files) )

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
    for filename in data_files:
        
        dpg.set_value("filename", filename)

        dpg.render_dearpygui_frame()
        if not dpg.is_dearpygui_running():
            break

        ticker = filename[ 0: filename.find("_") ]
        share = TkInstrument(client, config,  InstrumentType.INSTRUMENT_TYPE_SHARE, ticker, "TQBR")
        raw_samples = TkIO.read_at_path( join( data_path, filename) )

        preprocessor.add_samples(share, raw_samples, render_samples)

        total_samples = total_samples + int( len( raw_samples ) / 2 )
        files_processed = files_processed + 1

        dpg.set_value("files_processed", str(files_processed)+"/"+str(len(data_files)))
        dpg.set_value("orderbook_samples", str(preprocessor.num_orderbook_samples())+"/"+str(total_samples) )
        dpg.set_value("last_trades_samples", str(preprocessor.num_last_trades_samples())+"/"+str(total_samples) )

        dpg.render_dearpygui_frame()
        if not dpg.is_dearpygui_running():
            break

    end_time = time.time()
    print('Elapsed time:',end_time-start_time)

    dpg.set_value("filename", 'Generating synthetic samples...')
    dpg.render_dearpygui_frame()
    
    preprocessor.generate_synthetic_samples( render_samples )
    dpg.render_dearpygui_frame()

    dpg.set_value("orderbook_samples", str(preprocessor.num_orderbook_samples())+"/"+str(total_samples) )
    dpg.set_value("last_trades_samples", str(preprocessor.num_last_trades_samples())+"/"+str(total_samples) )
    dpg.set_value("filename", '...all is done!')
    dpg.render_dearpygui_frame()

    preprocessor.flush()

    while dpg.is_dearpygui_running():
        dpg.render_dearpygui_frame()

    dpg.destroy_context()