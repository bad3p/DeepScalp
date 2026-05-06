
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
import multiprocessing as mp
from queue import Empty  # for non-blocking queue reads
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
# Worker process
#------------------------------------------------------------------------------------------------------------------------

def preprocess_file(output_queue, ticker:str, is_test_data_source:bool, filename:str):

    pid = os.getpid()

    TOKEN = os.environ["TK_TOKEN"]
    with Client(TOKEN, target=INVEST_GRPC_API) as client:

        config = configparser.ConfigParser()
        config.read( 'TkConfig.ini' )

        share = TkInstrument(client, config, InstrumentType.INSTRUMENT_TYPE_SHARE, ticker, "TQBR")

        data_path = config['Paths']['DataPath']
        min_price_increment_factor = int(config['Autoencoders']['MinPriceIncrementFactor'])                

        orderbook_width = int(config['Autoencoders']['OrderbookWidth'])
        orderbook_depth = int(config['Autoencoders']['OrderbookDepth'])
        last_trades_width = int(config['Autoencoders']['LastTradesWidth'])
        last_trades_depth = int(config['Autoencoders']['LastTradesDepth'])

        test_data_ratio = float(config['TimeSeries']['TestDataRatio'])
        lshash_size = int(config['TimeSeries']['LSHashSize'])
        ts_sample_similatiry = float(config['TimeSeries']['TSSampleSimilatiry'])
        ts_data_stride = int(config['TimeSeries']['TSDataStride'])
        market_regime_steps_count = int(config['TimeSeries']['MarketRegimeStepsCount'])
        num_market_regimes = int(config['TimeSeries']['NumMarketRegimes'])
        prior_steps_count = int(config['TimeSeries']['PriorStepsCount'])
        future_steps_count = int(config['TimeSeries']['FutureStepsCount'])
        priority_tail_threshold = float(config['TimeSeries']['PriorityTailThreshold'])

        raw_samples = TkIO.read_at_path( join( data_path, filename) )

        raw_sample_count = int( len(raw_samples) / 2 ) # [ orderbook, last_trades, .... ]

        if raw_sample_count >= prior_steps_count + future_steps_count:

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
                orderbook_tensor, _, pivot_price, volume, slope, microprice = TkStatistics.orderbook_to_tensor( orderbook_sample, orderbook_width, min_price_increment * min_price_increment_factor )            
                price[i] = pivot_price
                spread[i] = TkStatistics.orderbook_spread( orderbook_sample, orderbook_width, min_price_increment * min_price_increment_factor )
                orderbook_volume[i] = volume
                orderbook_slope[i] = slope
                orderbook_microprice[i] = microprice

                last_trades_samples = [ (raw_samples[i*2+1], last_trades_time_threshold) ]
                last_trades_tensor, _, num_events, volume, buy_trades, sell_trades, _ = TkStatistics.last_trades_to_tensor( last_trades_samples, pivot_price, last_trades_width, min_price_increment * min_price_increment_factor )
                last_trades_volume[i] = volume
                last_trades_num_events[i] = num_events
                trade_flow_imbalance[i] = (buy_trades - sell_trades) / max(1.0, buy_trades + sell_trades)

                # adjust minimal time for next last trades sample
                last_trades_time_threshold = orderbook_sample.orderbook_ts       

            trade_flow_imbalance_ema_norm = TkStatistics.ema_normalize( trade_flow_imbalance, half_life=market_regime_steps_count ).tolist()
            orderbook_slope_ema_norm = TkStatistics.ema_normalize( orderbook_slope, half_life=market_regime_steps_count ).tolist()
            orderbook_microprice_ema_norm = TkStatistics.ema_normalize( orderbook_microprice, half_life=market_regime_steps_count ).tolist()

            price_change = [0.0] * raw_sample_count
            for i in range( raw_sample_count ):
                if i == 0:
                    price_change[i] = 0.0
                else:
                    price_change[i] = price[i] - price[i-1]

            price_change_log_ema_norm = TkStatistics.log_ema_normalize( price_change, half_life=market_regime_steps_count ).tolist()

            regimes = TkStatistics.price_to_market_regimes(price, market_regime_steps_count, num_market_regimes)

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

            order_flow_imbalance_log_ema_norm = TkStatistics.log_ema_normalize( order_flow_imbalance, half_life=market_regime_steps_count ).tolist()
            cumulative_order_flow_imbalance_log_ema_norm = TkStatistics.rolling_sum( order_flow_imbalance_log_ema_norm, window=future_steps_count ).tolist()
            queue_imbalance_ema_norm = TkStatistics.ema_normalize( queue_imbalance, half_life=market_regime_steps_count ).tolist()

            queue_depletion_intensity_log_ema_norm = TkStatistics.log_ema_normalize( queue_depletion_intensity, half_life=market_regime_steps_count ).tolist()
            queue_depletion_imbalance_log_ema_norm = TkStatistics.log_ema_normalize( queue_depletion_imbalance, half_life=market_regime_steps_count ).tolist()

            order_arrival_intensity_log_ema_norm = TkStatistics.log_ema_normalize( order_arrival_intensity, half_life=market_regime_steps_count ).tolist()
            order_arrival_imbalance_log_ema_norm = TkStatistics.log_ema_normalize( order_arrival_imbalance, half_life=market_regime_steps_count ).tolist()
        
            price_log_ema_volatility = TkStatistics.log_vol_ema_normalize( price, half_life=market_regime_steps_count ).tolist()
            orderbook_log_ema_norm_volume = TkStatistics.log_ema_normalize( orderbook_volume, half_life=market_regime_steps_count ).tolist()
            last_trades_log_ema_norm_volume = TkStatistics.log_ema_normalize( last_trades_volume, half_life=market_regime_steps_count ).tolist()
            last_trades_log_ema_norm_num_events = TkStatistics.log_ema_normalize( last_trades_num_events, half_life=market_regime_steps_count ).tolist()
            spread_log_ema_norm = TkStatistics.log_ema_normalize( spread, half_life=market_regime_steps_count ).tolist()

            start_range = prior_steps_count - 1
            end_range = raw_sample_count - future_steps_count - 1
            range_len = end_range - start_range

            # encode future trades distribution

            future_trades = [None] * (raw_sample_count)
            future_trades_volume = [None] * (raw_sample_count)
            future_trades_tails = [None] * (raw_sample_count)

            for i in range( start_range, end_range+1 ):            
                ts_base_price = price[i]
                prev_orderbook_sample = raw_samples[i*2]
                last_trades_samples = [ (raw_samples[(i+1)*2+1], prev_orderbook_sample.orderbook_ts)]
                for j in range( 2, future_steps_count+1 ):
                    prev_orderbook_sample = raw_samples[(i+j-1)*2]
                    last_trades_samples.append( ( raw_samples[(i+j)*2+1], prev_orderbook_sample.orderbook_ts ) )

                future_last_trades_tensor, _, num_future_events, future_volume, future_buy_trades, future_sell_trades, future_trades_mean_tails = TkStatistics.last_trades_to_tensor( last_trades_samples, ts_base_price, last_trades_width, min_price_increment * min_price_increment_factor, force_categorical=True )
            
                future_trades[i] = future_last_trades_tensor
                future_trades_volume[i] = future_volume
                future_trades_tails[i] = future_trades_mean_tails

            # combine data samples

            callback_indices = [start_range + int(i / 10.0 * range_len) for i in range(1,10)]

            hasheable_sample_width = 37 * prior_steps_count # 37 == sizeof ts_input[] 
            sample_lhs = LSHash(lshash_size, hasheable_sample_width) 
            step = 0

            for i in range( start_range, end_range+1 ):

                step = step + 1

                ts_regime = regimes[i+1] 
                ts_input = [None] * prior_steps_count
                ts_base_price = price[i]

                i0 = i-prior_steps_count+1
                i1 = i+1
                
                for j in range( prior_steps_count ):
                    k = i-prior_steps_count+j+1

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

                    # slice 9 : trade activity ? liquidity
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
                if len(lsh_query) > 0 and lsh_query[0][1] <= ts_sample_similatiry:
                    print( "Too similar sample, skipped: ", lsh_query[0][1])
                    continue
                sample_lhs.index( ts_input )
                
                ts_target_left_tail = future_trades_tails[i][0]
                ts_target_right_tail = future_trades_tails[i][1]
                is_priority_sample = ( ts_target_left_tail <= -priority_tail_threshold ) or ( ts_target_right_tail >= priority_tail_threshold )
                
                if (step-1) % ts_data_stride == 0 or is_priority_sample:                    
                    output_queue.put( (pid, [ts_input, ts_target, ts_regime], is_priority_sample, is_test_data_source, False) )
    
    output_queue.put( (pid, [0], False, is_test_data_source, True) )

    quit(0)

#------------------------------------------------------------------------------------------------------------------------    
# Main process
#------------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":

    TOKEN = os.environ["TK_TOKEN"]

    config = configparser.ConfigParser()
    config.read( 'TkConfig.ini' )

    data_path = config['Paths']['DataPath']
    time_series_index_filename = config['Paths']['TimeSeriesIndexFileName']
    time_series_training_data_filename = config['Paths']['TimeSeriesTrainingDataFileName']
    time_series_test_data_filename = config['Paths']['TimeSeriesTestDataFileName']
    
    data_extension = config['Paths']['OrderbookFileExtension']
    test_data_ratio = float(config['TimeSeries']['TestDataRatio'])

    if ( os.path.isfile( join( data_path, time_series_training_data_filename)) or os.path.isfile( join( data_path, time_series_test_data_filename)) ):
        raise RuntimeError('Preprocessed data already exists! Delete it manually.')
    
    training_index = []
    test_index = []
    training_data_offset = 0
    test_data_offset = 0
    training_data_stream = open( join( data_path, time_series_training_data_filename), 'wb+')
    test_data_stream = open( join( data_path, time_series_test_data_filename), 'wb+')

    data_files = [filename for filename in listdir(data_path) if (data_extension in filename) and isfile(join(data_path, filename))]
    print( 'Data files found:', len(data_files) )

    files_by_ticker = group_by_ticker(data_files)
    print( 'Tickers found:', len(files_by_ticker) )

    with Client(TOKEN, target=INVEST_GRPC_API) as client:

        dpg.create_context()
        dpg.create_viewport(title='Data preprocessor', width=1572, height=768)
        dpg.setup_dearpygui()

        with dpg.window(tag="primary_window", label="Preprocess data"):
            with dpg.group(horizontal=True):
                dpg.add_text( default_value="Files remaining: " )
                dpg.add_text( tag="files_remaining", default_value="0/0", color=[255, 254, 255])
            with dpg.group(horizontal=True):
                dpg.add_text( default_value="Samples processed (training/test): " )
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

        total_samples = 0
        files_processed = 0
        start_time = time.time()

        # prepare list of data sources

        data_sources = []

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

                data_sources.append( (ticker, is_test_data_source, filename) )

        print( 'Total num files:', len(data_sources) )

        # preprocess data sources

        max_num_processes = 16
        max_queue_size = 128

        output_queue = mp.Queue( maxsize=max_queue_size )
        processes = []

        def join_process(pid):
            global processes
            for i in range(len(processes) - 1, -1, -1):
                if processes[i].pid == pid:
                    processes[i].join()
                    del processes[i]

        def append_sample(sample:list, is_priority_sample:bool, is_test_data_source:bool):
            global test_index
            global test_data_offset
            global test_data_stream
            global training_index
            global training_data_offset
            global training_data_stream
            if is_test_data_source:
                test_index.append( test_data_offset )
                TkIO.write_to_file( test_data_stream, sample )
                test_data_offset = test_data_stream.tell()
            else:
                training_index.append( training_data_offset )
                TkIO.write_to_file( training_data_stream, sample )
                training_data_offset = training_data_stream.tell()
            if is_priority_sample:
                ts_input = sample[0]
                ts_target = sample[1]
                TkUI.set_series("x_axis_input","y_axis_input","input_series", ts_input)
                ts_target_sample_view = ts_target[-1]
                TkUI.set_series("x_axis_target","y_axis_target","target_series", ts_target_sample_view)                
                dpg.set_value("samples_processed", str(len(training_index)) + "/" + str(len(test_index)))

        while len(data_sources) > 0 or len(processes) > 0:
            if len(processes) < max_num_processes and len(data_sources) > 0:
                data_source = data_sources[0]
                del data_sources[0]

                ticker = data_source[0]                
                is_test_data_source = data_source[1]
                filename = data_source[2]

                process = mp.Process(target=preprocess_file, args=(output_queue, ticker, is_test_data_source, filename))
                process.start()
                processes.append(process)

                dpg.set_value("filename", filename)
                dpg.set_value("files_remaining", str(len(data_sources)))

            for step in range(max_num_processes):
                try:
                    tuple = output_queue.get_nowait()
                    pid = tuple[0]
                    sample = tuple[1]
                    is_priority_sample = tuple[2]
                    is_test_data_source = tuple[3]
                    done = tuple[4]
                    if done:
                        join_process( pid )
                    else:
                        append_sample(sample,is_priority_sample,is_test_data_source)
                except Empty:
                    time.sleep( 0.0 )
                    pass  # nothing in the queue right now

            dpg.render_dearpygui_frame()
            if not dpg.is_dearpygui_running():
                break

        print( 'Waiting for the rest of processes to complete...')

        while len(processes) > 0:
            try:
                tuple = output_queue.get_nowait()
                pid = tuple[0]
                sample = tuple[1]
                is_priority_sample = tuple[2]
                is_test_data_source = tuple[3]
                done = tuple[4]
                if done:
                    join_process( pid )
                else:
                    append_sample(sample,is_priority_sample,is_test_data_source)
            except Empty:
                pass  # nothing in the queue right now

        end_time = time.time()
        print('Elapsed time:',end_time-start_time)
    
        dpg.set_value("filename", '...all is done!')
        dpg.render_dearpygui_frame()

        # flush
        TkIO.append_at_path( join(data_path, time_series_index_filename), training_index )
        TkIO.append_at_path( join(data_path, time_series_index_filename), test_index )
        training_data_stream.close()
        test_data_stream.close()

        while dpg.is_dearpygui_running():
            dpg.render_dearpygui_frame()

        dpg.destroy_context()