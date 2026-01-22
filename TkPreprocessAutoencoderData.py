
import os
import time
import numpy as np
import configparser
import json
import copy
import random
import math
import gc
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
        self._orderbook_width = int(_cfg['Autoencoders']['OrderbookWidth'])
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

        if ( os.path.isfile(join(self._data_path, self._orderbook_training_data_filename)) or
              os.path.isfile(join(self._data_path, self._orderbook_test_data_filename)) or
              os.path.isfile(join(self._data_path, self._last_trades_training_data_filename)) or
              os.path.isfile(join(self._data_path, self._last_trades_test_data_filename)) ):
            raise RuntimeError('Preprocessed data already exists! Delete it manually.')

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

        start_time = time.time()

        min_price_increment = quotation_to_float( share.min_price_increment() )
        raw_sample_count = int( len(raw_samples) / 2 ) # orderbook, last_trades

        orderbook_samples = []
        hasheable_orderbook_samples = []
        last_trades_samples = []
        hasheable_last_trades_samples = []

        callback_indices = [int(i / 10.0 * raw_sample_count) for i in range(1,100)]

        local_orderbook_lsh = LSHash(self._lshash_size, self._orderbook_width)
        local_last_trades_lsh = LSHash(self._lshash_size, self._last_trades_width)

        last_trades_time_threshold = None

        for i in range(raw_sample_count):

            # orderbook sample
            orderbook_sample = raw_samples[i*2]
            orderbook_tensor, hasheable_orderbook_tensor, pivot_price = TkStatistics.orderbook_to_tensor( orderbook_sample, self._orderbook_width, min_price_increment * self._min_price_increment_factor )             
            lsh_query = local_orderbook_lsh.query( hasheable_orderbook_tensor, num_results=1 )
            if len(lsh_query) == 0 or lsh_query[0][1] > self._orderbook_sample_similarity:
                orderbook_samples.append(orderbook_tensor)
                hasheable_orderbook_samples.append(hasheable_orderbook_tensor)
                local_orderbook_lsh.index(hasheable_orderbook_tensor)

            # last trades sample
            last_trades_sample = [raw_samples[i*2+1]]
            last_trades_tensor, hasheable_last_trades_tensor, num_events = TkStatistics.last_trades_to_tensor( last_trades_sample, pivot_price, self._last_trades_width, min_price_increment * self._min_price_increment_factor, last_trades_time_threshold )
            lsh_query = local_last_trades_lsh.query( hasheable_last_trades_tensor, num_results=1 )
            if len(lsh_query) == 0 or lsh_query[0][1] > self._last_trades_sample_similarity:
                last_trades_samples.append(last_trades_tensor)
                hasheable_last_trades_samples.append(hasheable_last_trades_tensor)
                local_last_trades_lsh.index(hasheable_last_trades_tensor)

            # cumulative last trades sample (this will represent time series output)
            if i + self._future_steps_count < raw_sample_count / 2:
                for j in range( 1, self._future_steps_count ):
                    last_trades_sample.append( raw_samples[(i+j+1)*2+1] )
                cumulative_last_trades_tensor, hasheable_cumulative_last_trades_tensor, num_cumulative_events = TkStatistics.last_trades_to_tensor( last_trades_sample, pivot_price, self._last_trades_width, min_price_increment * self._min_price_increment_factor, last_trades_time_threshold )                
                lsh_query = local_last_trades_lsh.query( hasheable_cumulative_last_trades_tensor, num_results=1 )
                if len(lsh_query) == 0 or lsh_query[0][1] > self._last_trades_sample_similarity:
                    last_trades_samples.append(cumulative_last_trades_tensor)
                    hasheable_last_trades_samples.append(hasheable_cumulative_last_trades_tensor)
                    local_last_trades_lsh.index(hasheable_cumulative_last_trades_tensor)
            
            # adjust minimal time for next last trades sample
            last_trades_time_threshold = orderbook_sample.orderbook_ts

            if len(callback_indices) > 0 and i >= callback_indices[0]:
                del callback_indices[0]
                if render_callback != None:
                    render_callback( orderbook_samples, last_trades_samples )

        for i in reversed(range(len(hasheable_orderbook_samples))):
            lsh_query = self._orderbook_lsh.query( hasheable_orderbook_samples[i], num_results=1 )
            if len(lsh_query) == 0 or lsh_query[0][1] > self._orderbook_sample_similarity:
                self._orderbook_lsh.index(hasheable_orderbook_samples[i])
            else:
                del orderbook_samples[i]
                del hasheable_orderbook_samples[i]

        if render_callback != None:
            render_callback( orderbook_samples, last_trades_samples )

        for i in reversed(range(len(hasheable_last_trades_samples))):
            lsh_query = self._last_trades_lsh.query( hasheable_last_trades_samples[i], num_results=1 )
            if len(lsh_query) == 0 or lsh_query[0][1] > self._last_trades_sample_similarity:
                self._last_trades_lsh.index(hasheable_last_trades_samples[i])
            else:
                del last_trades_samples[i]
                del hasheable_last_trades_samples[i]

        if render_callback != None:
            render_callback( orderbook_samples, last_trades_samples )

        if training_category:                    
            self._orderbook_training_data_offset = self.write_samples( 
                orderbook_samples,
                self._orderbook_training_index, 
                self._orderbook_training_data_offset, 
                self._orderbook_training_data_stream
            )
            self._last_trades_training_data_offset = self.write_samples(
                last_trades_samples, 
                self._last_trades_training_index, 
                self._last_trades_training_data_offset, 
                self._last_trades_training_data_stream
            )
        else:
            self._orderbook_test_data_offset = self.write_samples( 
                orderbook_samples,
                self._orderbook_test_index, 
                self._orderbook_test_data_offset, 
                self._orderbook_test_data_stream
            )
            self._last_trades_test_data_offset = self.write_samples(
                last_trades_samples, 
                self._last_trades_test_index, 
                self._last_trades_test_data_offset, 
                self._last_trades_test_data_stream
            )

        end_time = time.time()
        elapsed_time = end_time - start_time
        return raw_sample_count / elapsed_time, len(orderbook_samples), len(last_trades_samples)

    def generate_synthetic_samples(self, training_category : bool, orderbook_min_index : int, orderbook_max_index : int, num_orderbook_samples : int, num_last_trades_samples : int, render_callback):

        return # TODO: investigate possibility to generate novel complex samples

        def generate_render_callback_indices(num_samples:int):
            if num_samples < 1000:
                return [int(i / 100.0 * num_samples) for i in range(1,100)]
            elif num_samples < 10000:
                return [int(i / 1000.0 * num_samples) for i in range(1,1000)]
            else:
                return [int(i / 10000.0 * num_samples) for i in range(1,10000)]
        
        lsh_reset_threshold_time = 2.0 # TODO: configure

        synthetic_orderbook_scheme = json.loads(config['Autoencoders']['SyntheticOrderbookScheme'])
        if not type(synthetic_orderbook_scheme) is list:
            raise RuntimeError('SyntheticOrderbookScheme is expected to be a list of lists!')

        synthetic_last_trades_scheme = json.loads(config['Autoencoders']['SyntheticLastTradesScheme'])
        if not type(synthetic_last_trades_scheme) is list:
            raise RuntimeError('SyntheticLastTradesScheme is expected to be a list of lists!')

        synthetic_sample_bias = float(config['Autoencoders']['SyntheticSampleBias'])
        synthetic_orderbook_sample_central_bias = float(config['Autoencoders']['SyntheticOrderbookSampleCentralBias'])
        synthetic_last_trades_sample_max_variance = float(config['Autoencoders']['SyntheticLastTradesSampleMaxVariance'])        

        self._orderbook_samples = []
        self._last_trades_samples = []

        num_samples = num_orderbook_samples
        callback_indices = generate_render_callback_indices( num_samples )

        for i in range(num_samples):

            t0 = time.time()
            success = False
            while not success:
                orderbook_scheme = synthetic_orderbook_scheme[random.randint(0, len(synthetic_orderbook_scheme)-1)]
                distribution = np.zeros( self._orderbook_width, dtype=float)
                distribution[int(self._orderbook_width/2)-2] = random.uniform(0.0, synthetic_orderbook_sample_central_bias) # center synthetic orderbook sample to the maximal ask price
                TkStatistics.generate_distribution( distribution, orderbook_scheme, synthetic_sample_bias, orderbook_min_index, orderbook_max_index )
                TkStatistics.to_cumulative_distribution(distribution)
                lsh_query = self._orderbook_lsh.query( distribution, num_results=1 )
                if len(lsh_query) == 0 or lsh_query[0][1] > self._synthetic_orderbook_sample_similarity:
                    self._orderbook_samples.append(distribution)
                    self._orderbook_lsh.index(distribution)
                    success = True
            t1 = time.time()
            if t1 - t0 > lsh_reset_threshold_time:
                self._orderbook_lsh = LSHash(self._lshash_size, self._orderbook_width)

            if len(callback_indices) > 0 and i >= callback_indices[0]:
                del callback_indices[0]
                if render_callback != None:
                    render_callback( self._orderbook_samples, None )

        
        num_samples = num_last_trades_samples        
        callback_indices = generate_render_callback_indices( num_samples )

        for i in range(num_samples):

            t0 = time.time()
            success = False
            while not success:
                last_trades_scheme = synthetic_last_trades_scheme[random.randint(0, len(synthetic_last_trades_scheme)-1)]
                distribution = np.zeros( self._last_trades_width, dtype=float)
                TkStatistics.generate_clustered_distribution( distribution, last_trades_scheme, synthetic_sample_bias, synthetic_last_trades_sample_max_variance )
                lsh_query = self._last_trades_lsh.query( distribution, num_results=1 )
                if len(lsh_query) == 0 or lsh_query[0][1] > self._synthetic_last_trades_sample_similarity:
                    self._last_trades_samples.append(distribution)
                    self._last_trades_lsh.index(distribution)
                    success = True
            t1 = time.time()
            if t1 - t0 > lsh_reset_threshold_time:
                self._last_trades_lsh = LSHash(self._lshash_size, self._last_trades_width)

            if len(callback_indices) > 0 and i >= callback_indices[0]:
                del callback_indices[0]
                if render_callback != None:
                    render_callback( None, self._last_trades_samples )

        if render_callback != None:
            render_callback( self._orderbook_samples, self._last_trades_samples )

        if training_category:
            self._orderbook_training_data_offset = self.write_samples( 
                self._orderbook_samples,
                self._orderbook_training_index, 
                self._orderbook_training_data_offset, 
                self._orderbook_training_data_stream
            )
            self._last_trades_training_data_offset = self.write_samples(
                self._last_trades_samples, 
                self._last_trades_training_index, 
                self._last_trades_training_data_offset, 
                self._last_trades_training_data_stream
            )
        else:
            self._orderbook_test_data_offset = self.write_samples( 
                self._orderbook_samples,
                self._orderbook_test_index, 
                self._orderbook_test_data_offset, 
                self._orderbook_test_data_stream
            )
            self._last_trades_test_data_offset = self.write_samples(
                self._last_trades_samples, 
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
orderbook_width = int(config['Autoencoders']['OrderbookWidth'])
synthetic_sample_ratio = float(config['Autoencoders']['SyntheticSampleRatio'])

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
            dpg.add_text( default_value="Samples per second: " )
            dpg.add_text( tag="samples_per_second", default_value="0/0", color=[255, 254, 255])
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
            orderbook_sample_view = orderbook_samples[-1][1,:].tolist()
            TkUI.set_series("x_axis_orderbook","y_axis_orderbook","orderbook_series", orderbook_sample_view)
        if last_trades_samples:
            last_trades_sample_view = last_trades_samples[-1][1,:].tolist()
            TkUI.set_series("x_axis_last_trades","y_axis_last_trades","last_trades_series", last_trades_sample_view)
        dpg.render_dearpygui_frame()

    total_samples = 0
    files_processed = 0
    start_time = time.time()

    cumulative_samples_per_second = 0
    samples_per_second_norm = 0

    for ticker in files_by_ticker:

        share = TkInstrument(client, config,  InstrumentType.INSTRUMENT_TYPE_SHARE, ticker, "TQBR")

        num_data_sources = len(files_by_ticker[ticker])
        num_test_data_sources = max(1, int( num_data_sources * test_data_ratio ))
        num_training_data_sources = num_data_sources - num_test_data_sources

        generated_orderbook_min_index = 0
        generated_orderbook_max_index = 0
        num_orderbook_training_samples = 0
        num_last_trades_training_samples = 0
        num_orderbook_test_samples = 0
        num_last_trades_test_samples = 0

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

            samples_per_second, num_orderbook_samples, num_last_trades_samples = preprocessor.add_samples(share, raw_samples, not is_test_data_source, render_samples)            
            if is_test_data_source:
                num_orderbook_test_samples = num_orderbook_test_samples + num_orderbook_samples
                num_last_trades_test_samples = num_last_trades_test_samples + num_last_trades_samples
            else:
                num_orderbook_training_samples = num_orderbook_training_samples + num_orderbook_samples
                num_last_trades_training_samples = num_last_trades_training_samples + num_last_trades_samples
            
            cumulative_samples_per_second = cumulative_samples_per_second + samples_per_second
            samples_per_second_norm = samples_per_second_norm + 1

            total_samples = total_samples + int( len( raw_samples ) / 2 )
            files_processed = files_processed + 1

            dpg.set_value("files_processed", str(files_processed)+"/"+str(len(data_files)))
            dpg.set_value("orderbook_samples", str(preprocessor.num_orderbook_samples())+"/"+str(total_samples) )
            dpg.set_value("last_trades_samples", str(preprocessor.num_last_trades_samples())+"/"+str(total_samples) )
            dpg.set_value("samples_per_second", str( cumulative_samples_per_second/samples_per_second_norm ) )

            gc.collect()

            dpg.render_dearpygui_frame()
            if not dpg.is_dearpygui_running():
                break

        generated_orderbook_min_index = int( generated_orderbook_min_index / num_data_sources )
        generated_orderbook_max_index = int( generated_orderbook_max_index / num_data_sources )
        print(ticker, generated_orderbook_min_index, generated_orderbook_max_index)        

        if synthetic_sample_ratio > 0.0:
            dpg.set_value("filename", 'Generating synthetic samples...')
            dpg.render_dearpygui_frame()

            num_orderbook_training_samples = max( 1, int( num_orderbook_training_samples * synthetic_sample_ratio ) )
            num_last_trades_training_samples = max( 1, int( num_last_trades_training_samples * synthetic_sample_ratio ) )
            preprocessor.generate_synthetic_samples( True, generated_orderbook_min_index, generated_orderbook_max_index, num_orderbook_training_samples, num_last_trades_training_samples, render_samples )
            gc.collect()
            dpg.render_dearpygui_frame()

            num_orderbook_test_samples = max( 1, int( num_orderbook_test_samples * synthetic_sample_ratio ) )
            num_last_trades_test_samples = max( 1, int( num_last_trades_test_samples * synthetic_sample_ratio ) ) 
            preprocessor.generate_synthetic_samples( False, generated_orderbook_min_index, generated_orderbook_max_index, num_orderbook_test_samples, num_last_trades_test_samples, render_samples )
            gc.collect()
            dpg.render_dearpygui_frame()

        if not dpg.is_dearpygui_running():
            break

        preprocessor.clear_lsh()
        #break

    if dpg.is_dearpygui_running():

        dpg.set_value("orderbook_samples", str(preprocessor.num_orderbook_samples())+"/"+str(total_samples) )
        dpg.set_value("last_trades_samples", str(preprocessor.num_last_trades_samples())+"/"+str(total_samples) )
        dpg.set_value("filename", '...all is done!')
        dpg.render_dearpygui_frame()

    preprocessor.flush()

    end_time = time.time()
    print('Elapsed time:',end_time-start_time)

    while dpg.is_dearpygui_running():
        dpg.render_dearpygui_frame()

    dpg.destroy_context()

    print( "Samples per second: ", str( cumulative_samples_per_second/samples_per_second_norm ) )