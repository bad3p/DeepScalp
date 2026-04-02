
import os
import time
import numpy as np
import configparser
import json
import copy
import random
import math
from LSHash import LSHash
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
from tinkoff.invest.utils import decimal_to_quotation, quotation_to_decimal
from TkModules.TkQuotation import quotation_to_float
from TkModules.TkIO import TkIO
from TkModules.TkInstrument import TkInstrument
from TkModules.TkStatistics import TkStatistics
from TkModules.TkUI import TkUI

#------------------------------------------------------------------------------------------------------------------------

class TkTimeSeriesDataPreprocessor():

    def __init__(self, _cfg : configparser.ConfigParser):

        self._cuda = torch.device("cuda")
        
        self._min_price_increment_factor = int(_cfg['Autoencoders']['MinPriceIncrementFactor'])                

        self._data_path = _cfg['Paths']['DataPath']
        self._time_series_index_filename = _cfg['Paths']['TimeSeriesIndexFileName']
        self._time_series_training_data_filename = _cfg['Paths']['TimeSeriesTrainingDataFileName']
        self._time_series_test_data_filename = _cfg['Paths']['TimeSeriesTestDataFileName']

        self._orderbook_width = int(_cfg['Autoencoders']['OrderbookWidth'])
        self._orderbook_depth = int(_cfg['Autoencoders']['OrderbookDepth'])
        self._last_trades_width = int(_cfg['Autoencoders']['LastTradesWidth'])
        self._last_trades_depth = int(_cfg['Autoencoders']['LastTradesDepth'])

        self._test_data_ratio = float(_cfg['TimeSeries']['TestDataRatio'])
        self._lshash_size = int(_cfg['TimeSeries']['LSHashSize'])
        self._ts_sample_similatiry = float(_cfg['TimeSeries']['TSSampleSimilatiry'])
        self._ts_data_stride = int(_cfg['TimeSeries']['TSDataStride'])
        self._market_regime_steps_count = int(_cfg['TimeSeries']['MarketRegimeStepsCount'])
        self._num_market_regimes = int(_cfg['TimeSeries']['NumMarketRegimes'])
        self._prior_steps_count = int(_cfg['TimeSeries']['PriorStepsCount'])
        self._future_steps_count = int(_cfg['TimeSeries']['FutureStepsCount'])
        self._priority_tail_threshold = float(_cfg['TimeSeries']['PriorityTailThreshold'])

        if ( os.path.isfile( join(self._data_path, self._time_series_training_data_filename)) or
             os.path.isfile( join(self._data_path, self._time_series_test_data_filename)) ):
            raise RuntimeError('Preprocessed data already exists! Delete it manually.')

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

    @staticmethod
    def to_list_of_lists(l:list, item_length:int):
        result = []
        for i in range( int(len(l)/item_length) ):
            result.append([])
            for j in range( item_length ):
                result[-1].append( l[i*item_length + j] )
        return result

    def add_samples(self, share : TkInstrument, raw_samples : list, is_test_data_source : bool, render_callback):

        raw_sample_count = int( len(raw_samples) / 2 ) # [ orderbook, last_trades, .... ]

        if raw_sample_count < self._prior_steps_count + self._future_steps_count:
            print( 'Not enough sample count: ', raw_sample_count )
            return

        # discretisation may vary

        min_price_increment = TkStatistics.get_min_price_increment(raw_samples[0], quotation_to_decimal(share.min_price_increment()))
        for i in range(raw_sample_count):
            min_price_increment = min( min_price_increment, TkStatistics.get_min_price_increment(raw_samples[i*2], quotation_to_decimal(share.min_price_increment())) )
        min_price_increment = float(min_price_increment)

        # encode orderbook & last trades

        price = [0.0] * raw_sample_count
        spread = [0.0] * raw_sample_count
        orderbook_volume = [0] * raw_sample_count
        orderbook_slope = [0] * raw_sample_count
        orderbook_microprice = [0] * raw_sample_count
        last_trades_volume = [0] * raw_sample_count
        last_trades_num_events = [0] * raw_sample_count
        trade_flow_imbalance = [0] * raw_sample_count

        last_trades_time_threshold = None

        for i in range( raw_sample_count ):

            orderbook_sample = raw_samples[i*2]
            orderbook_tensor, _, pivot_price, volume, slope, microprice = TkStatistics.orderbook_to_tensor( orderbook_sample, self._orderbook_width, min_price_increment * self._min_price_increment_factor )            
            price[i] = pivot_price
            spread[i] = TkStatistics.orderbook_spread( orderbook_sample, self._orderbook_width, min_price_increment * self._min_price_increment_factor )
            orderbook_volume[i] = volume
            orderbook_slope[i] = slope
            orderbook_microprice[i] = microprice

            last_trades_samples = [ (raw_samples[i*2+1], last_trades_time_threshold) ]
            last_trades_tensor, _, num_events, volume, buy_trades, sell_trades, _ = TkStatistics.last_trades_to_tensor( last_trades_samples, pivot_price, self._last_trades_width, min_price_increment * self._min_price_increment_factor )
            last_trades_volume[i] = volume
            last_trades_num_events[i] = num_events
            trade_flow_imbalance[i] = (buy_trades - sell_trades) / max(1.0, buy_trades + sell_trades)

            # adjust minimal time for next last trades sample
            last_trades_time_threshold = orderbook_sample.orderbook_ts       


        trade_flow_imbalance_ema_norm = TkStatistics.ema_normalize( trade_flow_imbalance, half_life=self._market_regime_steps_count ).tolist()
        orderbook_slope_ema_norm = TkStatistics.ema_normalize( orderbook_slope, half_life=self._market_regime_steps_count ).tolist()
        orderbook_microprice_ema_norm = TkStatistics.ema_normalize( orderbook_microprice, half_life=self._market_regime_steps_count ).tolist()

        price_change = [0.0] * raw_sample_count
        for i in range( raw_sample_count ):
            if i == 0:
                price_change[i] = 0.0
            else:
                price_change[i] = price[i] - price[i-1]

        price_change_log_ema_norm = TkStatistics.log_ema_normalize( price_change, half_life=self._market_regime_steps_count )

        regimes = TkStatistics.price_to_market_regimes(price, self._market_regime_steps_count, self._num_market_regimes)

        order_flow_imbalance = [None] * raw_sample_count
        queue_imbalance = [None] * raw_sample_count
        queue_depletion_intensity = [None] * raw_sample_count
        queue_depletion_imbalance = [None] * raw_sample_count
        order_arrival_intensity = [None] * raw_sample_count
        order_arrival_imbalance = [None] * raw_sample_count

        for i in range( raw_sample_count ):
            if i == 0:
                order_flow_imbalance[i] = 0.0
                order_arrival_intensity[i] = 0.0
                order_arrival_imbalance[i] = 0.0
            else:
                order_flow_imbalance[i] = TkStatistics.depth_weighted_order_flow_imbalance( raw_samples[(i-1)*2], raw_samples[i*2], alpha=1.0 ) # TODO: configure alpha
                bid_intensity, ask_intensity = TkStatistics.depth_weighted_order_arrival_rate( raw_samples[(i-1)*2], raw_samples[i*2], raw_samples[i*2+1], alpha=1.0, dt = 60.0 ) # TODO: configure alpha & dt
                order_arrival_intensity[i] = bid_intensity + ask_intensity
                order_arrival_imbalance[i] = bid_intensity - ask_intensity
            queue_imbalance[i] = TkStatistics.depth_weighted_queue_imbalance( raw_samples[i*2], min_price_increment, alpha=1.0 ) # TODO: configure alpha
            bid_depletion, ask_depletion = TkStatistics.depth_weighted_queue_depletion_rate( raw_samples[i*2], raw_samples[i*2+1], alpha=1.0 ) # TODO: configure alpha
            queue_depletion_intensity[i] = bid_depletion + ask_depletion
            queue_depletion_imbalance[i] = bid_depletion - ask_depletion            

        order_flow_imbalance_log_ema_norm = TkStatistics.log_ema_normalize( order_flow_imbalance, half_life=self._market_regime_steps_count )
        cumulative_order_flow_imbalance_log_ema_norm = TkStatistics.rolling_sum( order_flow_imbalance_log_ema_norm, window=self._future_steps_count )
        queue_imbalance_ema_norm = TkStatistics.ema_normalize( queue_imbalance, half_life=self._market_regime_steps_count )

        queue_depletion_intensity_log_ema_norm = TkStatistics.log_ema_normalize( queue_depletion_intensity, half_life=self._market_regime_steps_count )
        queue_depletion_imbalance_log_ema_norm = TkStatistics.log_ema_normalize( queue_depletion_imbalance, half_life=self._market_regime_steps_count )

        order_arrival_intensity_log_ema_norm = TkStatistics.log_ema_normalize( order_arrival_intensity, half_life=self._market_regime_steps_count )
        order_arrival_imbalance_log_ema_norm = TkStatistics.log_ema_normalize( order_arrival_imbalance, half_life=self._market_regime_steps_count )
        
        price_log_ema_volatility = TkStatistics.log_vol_ema_normalize( price, half_life=self._market_regime_steps_count ).tolist()
        orderbook_log_ema_norm_volume = TkStatistics.log_ema_normalize( orderbook_volume, half_life=self._market_regime_steps_count ).tolist()
        last_trades_log_ema_norm_volume = TkStatistics.log_ema_normalize( last_trades_volume, half_life=self._market_regime_steps_count ).tolist()
        last_trades_log_ema_norm_num_events = TkStatistics.log_ema_normalize( last_trades_num_events, half_life=self._market_regime_steps_count ).tolist()
        spread_log_ema_norm = TkStatistics.log_ema_normalize( spread, half_life=self._market_regime_steps_count ).tolist()

        start_range = self._prior_steps_count - 1
        end_range = raw_sample_count - self._future_steps_count - 1
        range_len = end_range - start_range

        # encode future trades distribution

        future_trades = [None] * (raw_sample_count)
        future_trades_volume = [None] * (raw_sample_count)
        future_trades_tails = [None] * (raw_sample_count)

        for i in range( start_range, end_range+1 ):            
            ts_base_price = price[i]
            prev_orderbook_sample = raw_samples[i*2]
            last_trades_samples = [ (raw_samples[(i+1)*2+1], prev_orderbook_sample.orderbook_ts)]
            for j in range( 2, self._future_steps_count+1 ):
                prev_orderbook_sample = raw_samples[(i+j-1)*2]
                last_trades_samples.append( ( raw_samples[(i+j)*2+1], prev_orderbook_sample.orderbook_ts ) )

            future_last_trades_tensor, _, num_future_events, future_volume, future_buy_trades, future_sell_trades, future_trades_mean_tails = TkStatistics.last_trades_to_tensor( last_trades_samples, ts_base_price, self._last_trades_width, min_price_increment * self._min_price_increment_factor, force_categorical=True )
            
            future_trades[i] = future_last_trades_tensor
            future_trades_volume[i] = future_volume
            future_trades_tails[i] = future_trades_mean_tails

        # combine data samples

        callback_indices = [start_range + int(i / 10.0 * range_len) for i in range(1,10)]

        hasheable_sample_width = 37 * self._prior_steps_count # 37 == sizeof ts_input[] 
        sample_lhs = LSHash(self._lshash_size, hasheable_sample_width) 
        step = 0

        for i in range( start_range, end_range+1 ):

            step = step + 1

            ts_regime = regimes[i+1] 
            ts_input = [None] * self._prior_steps_count
            ts_base_price = price[i]

            i0 = i-self._prior_steps_count+1
            i1 = i+1
            #ts_base_orderbook_volume = sum( orderbook_volume[i0:i1] ) / self._prior_steps_count
            #ts_base_last_trades_volume = sum( last_trades_volume[i0:i1] ) / self._prior_steps_count
            #ts_base_last_trades_num_events = sum( last_trades_num_events[i0:i1] ) / self._prior_steps_count

            for j in range( self._prior_steps_count ):
                k = i-self._prior_steps_count+j+1

                ts_input[j] = []

                # slice 1 : price and volatility
                ts_input[j].append( price_change_log_ema_norm[k] )
                ts_input[j].append( price_log_ema_volatility[k] )

                # slice 2 : liquidity and spread
                ts_input[j].append( spread_log_ema_norm[k] )            
                ts_input[j].append( orderbook_log_ema_norm_volume[k] )

                # slice 3 : orderbook structure (shape + microprice)
                ts_input[j].append( orderbook_slope_ema_norm[k] )
                ts_input[j].append( orderbook_microprice_ema_norm[k] )
                ts_input[j].append( queue_imbalance_ema_norm[k] )

                # slice 4 : trade activity / trade flow intensity
                ts_input[j].append( last_trades_log_ema_norm_volume[k] )
                ts_input[j].append( last_trades_log_ema_norm_num_events[k] )
                ts_input[j].append( queue_depletion_intensity_log_ema_norm[k] )
                ts_input[j].append( order_arrival_intensity_log_ema_norm[k] )        
        
                # slice 5 : flow imbalance
                ts_input[j].append( cumulative_order_flow_imbalance_log_ema_norm[k] )                
                ts_input[j].append( trade_flow_imbalance_ema_norm[k] )
                ts_input[j].append( queue_depletion_imbalance_log_ema_norm[k] )
                ts_input[j].append( order_arrival_imbalance_log_ema_norm[k] )

                # slice 6 : price x liquidity / Spread
                ts_input[j].append( price_change_log_ema_norm[k] * spread_log_ema_norm[k] )
                ts_input[j].append( price_log_ema_volatility[k] * spread_log_ema_norm[k] )
                ts_input[j].append( price_change_log_ema_norm[k] * orderbook_log_ema_norm_volume[k] )

                # slice 7 : price x orderbook structure
                ts_input[j].append( price_change_log_ema_norm[k] * queue_imbalance_ema_norm[k] )
                ts_input[j].append( orderbook_microprice_ema_norm[k] - price_change_log_ema_norm[k] )
                ts_input[j].append( orderbook_slope_ema_norm[k] * price_change_log_ema_norm[k] )

                # slice 8 : liquidity x orderbook structure
                ts_input[j].append( spread_log_ema_norm[k] * queue_imbalance_ema_norm[k] )
                ts_input[j].append( orderbook_log_ema_norm_volume[k] * orderbook_slope_ema_norm[k] )

                # slice 9 : trade activity × liquidity
                ts_input[j].append( last_trades_log_ema_norm_volume[k] * spread_log_ema_norm[k] )
                ts_input[j].append( last_trades_log_ema_norm_num_events[k] * orderbook_log_ema_norm_volume[k] )
                ts_input[j].append( queue_depletion_intensity_log_ema_norm[k] * spread_log_ema_norm[k] )

                # slice 10 : trade Activity x orderbook structure
                ts_input[j].append( last_trades_log_ema_norm_volume[k] * queue_imbalance_ema_norm[k] )
                ts_input[j].append( order_arrival_intensity_log_ema_norm[k] * orderbook_slope_ema_norm[k] )
                ts_input[j].append( queue_depletion_intensity_log_ema_norm[k] * orderbook_microprice_ema_norm[k] )

                # slice 11 : flow imbalance x price
                ts_input[j].append( price_change_log_ema_norm[k] * trade_flow_imbalance_ema_norm[k] )
                ts_input[j].append( price_change_log_ema_norm[k] * cumulative_order_flow_imbalance_log_ema_norm[k] )

                # slice 12 : flow imbalance x orderbook
                ts_input[j].append( queue_imbalance_ema_norm[k] * trade_flow_imbalance_ema_norm[k] )
                ts_input[j].append( orderbook_microprice_ema_norm[k] * cumulative_order_flow_imbalance_log_ema_norm[k] )

                # slice 13 : flow imbalance x trade activity
                ts_input[j].append( last_trades_log_ema_norm_volume[k] * trade_flow_imbalance_ema_norm[k] )
                ts_input[j].append( order_arrival_imbalance_log_ema_norm[k] * last_trades_log_ema_norm_num_events[k] )

                # slice 14 : higher order nonlinear interactions
                ts_input[j].append( spread_log_ema_norm[k] * price_log_ema_volatility[k] * queue_depletion_intensity[k] )
                ts_input[j].append( queue_imbalance_ema_norm[k] * trade_flow_imbalance_ema_norm[k] * orderbook_microprice_ema_norm[k] )
                
                
            ts_input = list( itertools.chain.from_iterable(ts_input) )
            ts_target = future_trades[i].tolist()

            # sample similatiry check
            lsh_query = sample_lhs.query( ts_input, num_results=1 )
            if len(lsh_query) > 0 and lsh_query[0][1] <= self._ts_sample_similatiry:
                print( "Too similar sample, skipped: ", lsh_query[0][1])
                continue
            sample_lhs.index( ts_input )

            verbalize = True
            if is_test_data_source:
                ts_target_left_tail = future_trades_tails[i][0]
                ts_target_right_tail = future_trades_tails[i][1]
                is_priority_sample = ( ts_target_left_tail <= -self._priority_tail_threshold ) or ( ts_target_right_tail >= self._priority_tail_threshold )
                if (step-1) % self._ts_data_stride == 0 or is_priority_sample:
                    self._test_index.append( self._test_data_offset )
                    TkIO.write_to_file( self._test_data_stream, ts_input )
                    TkIO.write_to_file( self._test_data_stream, ts_target )
                    TkIO.write_to_file( self._test_data_stream, ts_regime )
                    self._test_data_offset = self._test_data_stream.tell()
                    verbalize = is_priority_sample
            else:
                ts_target_left_tail = future_trades_tails[i][0]
                ts_target_right_tail = future_trades_tails[i][1]
                is_priority_sample = ( ts_target_left_tail <= -self._priority_tail_threshold ) or ( ts_target_right_tail >= self._priority_tail_threshold )
                if (step-1) % self._ts_data_stride == 0 or is_priority_sample:
                    if is_priority_sample:
                        self._priority_table.append( len(self._training_index) )
                    else:
                        self._regular_table.append( len(self._training_index) )
                    self._training_index.append( self._training_data_offset )
                    TkIO.write_to_file( self._training_data_stream, ts_input )
                    TkIO.write_to_file( self._training_data_stream, ts_target )
                    TkIO.write_to_file( self._training_data_stream, ts_regime )
                    self._training_data_offset = self._training_data_stream.tell()
                    verbalize = is_priority_sample

            if len(callback_indices) > 0 and i >= callback_indices[0] and verbalize:
                del callback_indices[0]
                if render_callback != None:
                    render_callback( ts_input, ts_target )


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
market_regime_steps_count = int(config['TimeSeries']['MarketRegimeStepsCount'])
num_market_regimes = int(config['TimeSeries']['NumMarketRegimes'])

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
        ts_target_sample_view = ts_target[-1]
        TkUI.set_series("x_axis_target","y_axis_target","target_series", ts_target_sample_view)
        dpg.render_dearpygui_frame()

    total_samples = 0
    files_processed = 0
    start_time = time.time()
    for ticker in files_by_ticker:

        if not dpg.is_dearpygui_running():
            break

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