
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
import multiprocessing as mp
from queue import Empty  # for non-blocking queue reads
from joblib import Parallel, delayed
from t_tech.invest.constants import INVEST_GRPC_API
from t_tech.invest import Client
from t_tech.invest import InstrumentType
from t_tech.invest import InstrumentIdType
from t_tech.invest import SecurityTradingStatus
from t_tech.invest import GetOrderBookResponse, GetLastTradesResponse
from t_tech.invest import HistoricCandle
from t_tech.invest.exceptions import RequestError
from TkModules.TkQuotation import quotation_to_float
from TkModules.TkIO import TkIO
from TkModules.TkInstrument import TkInstrument
from TkModules.TkStatistics import TkStatistics
from TkModules.TkUI import TkUI
from TkModules.TkPreprocessFile import PreprocessedData

#------------------------------------------------------------------------------------------------------------------------
# Computes transfer entropy & put the result to output queue
#------------------------------------------------------------------------------------------------------------------------

def compute_transfer_entropy(output_queue, ticker:str, filename:str, features:list, ema_norm_step_variants:list, ema_norm_base_time:float):

    pid = os.getpid()

    TOKEN = os.environ["TK_TOKEN"]
    with Client(TOKEN, target=INVEST_GRPC_API) as client:

        config = configparser.ConfigParser()
        config.read( 'TkConfig.ini' )
        
        data_path = config['Paths']['DataPath']        
        orderbook_width = int(config['Autoencoders']['OrderbookWidth'])
        last_trades_width = int(config['Autoencoders']['LastTradesWidth'])
        last_trades_discretization = float(config['Autoencoders']['LastTradesDiscretization'])
        trend_steps_count = int(config['TimeSeries']['TrendStepsCount'])
        prior_steps_count = int(config['TimeSeries']['PriorStepsCount'])
        future_steps_count = int(config['TimeSeries']['FutureStepsCount'])

        share = TkInstrument(client, config,  InstrumentType.INSTRUMENT_TYPE_SHARE, ticker, "TQBR")
        raw_samples = TkIO.read_at_path( join( data_path, filename) )
        
        preprocessed_data = PreprocessedData( share, raw_samples, orderbook_width, last_trades_width, last_trades_discretization, trend_steps_count, future_steps_count, prior_steps_count )
        
        price_history = []
        preprocessed_feature = []
        
        for i in range(int(len(raw_samples) / 2)):
            orderbook_sample = raw_samples[i*2]
            price_history.append( float(orderbook_sample.last_price) )
        
        price_history_delta = np.diff( price_history )
        
        #price_history_1 = np.array(price_history)[1:-3]
        #price_history_2 = np.array(price_history)[2:-2]
        #price_history_3 = np.array(price_history)[3:-1]
        #price_history_4 = np.array(price_history)[4:]
        #price_history_mean_next_4 = (price_history_1 + price_history_2 + price_history_3 + price_history_4) / 4.0
        #price_history_delta = price_history_mean_next_4 - price_history[:-4]
        
        price_history_delta_discrete = TkStatistics.discretize_quantiles(price_history_delta, 3)
        
        for feature in features:
        
            if feature == 'baseline':
                feature_data = getattr(preprocessed_data, random.choice(features[:-1]))
                random.shuffle(feature_data)
            else:
                feature_data = getattr(preprocessed_data, feature)
                
            for ema_norm_step_count in ema_norm_step_variants:
                ema_norm_time = ema_norm_base_time * ema_norm_step_count
                preprocessed_feature = TkStatistics.irregular_ema_normalize( feature_data, preprocessed_data.time_delta, ema_norm_time ).tolist()
                #preprocessed_feature = preprocessed_feature[:-4] # match length of price_history_delta 
                preprocessed_feature = preprocessed_feature[:-1] # match length of price_history_delta 
                preprocessed_feature_discrete = TkStatistics.discretize_quantiles(preprocessed_feature, 3)
                te = TkStatistics.transfer_entropy( preprocessed_feature_discrete, price_history_delta_discrete )
                #te = TkStatistics.transfer_entropy( price_history_delta_discrete, preprocessed_feature_discrete )
                output_queue.put( (pid, False, feature, ema_norm_step_count, te) )                

    output_queue.put( (pid, True, '', 0, 0.0) )

    quit(0)

#------------------------------------------------------------------------------------------------------------------------    
# Main process
#------------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":

    TOKEN = os.environ["TK_TOKEN"]

    config = configparser.ConfigParser()
    config.read( 'TkConfig.ini' )

    data_path = config['Paths']['DataPath']
    market_data_extension = config['Paths']['MarketDataFileExtension']

    orderbook_width = int(config['Autoencoders']['OrderbookWidth'])
    last_trades_width = int(config['Autoencoders']['LastTradesWidth'])
    last_trades_discretization = float(config['Autoencoders']['LastTradesDiscretization'])
    trend_steps_count = int(config['TimeSeries']['TrendStepsCount'])
    prior_steps_count = int(config['TimeSeries']['PriorStepsCount'])
    future_steps_count = int(config['TimeSeries']['FutureStepsCount'])
    ema_norm_base_time = 67.0

    data_files = [fileName for fileName in listdir(data_path) if (market_data_extension in fileName) and isfile(join(data_path, fileName))]
    random.shuffle(data_files)

    ema_norm_step_variants = [ int(prior_steps_count/4), int(prior_steps_count/3), int(prior_steps_count/2), prior_steps_count]

    features = [ 'price_change', 
                 'price_change_rate',
                 'volatility',
                 'trends',
                 'spread',
                 'orderbook_volume',
                 'orderbook_slope',
                 'orderbook_microprice',
                 'queue_imbalance',
                 'last_trades_volume',
                 'last_trades_rate',
                 'last_trades_num_events',
                 'last_trades_frequency',
                 'average_trade_size',
                 'queue_depletion_intensity',
                 'order_arrival_intensity',
                 'order_flow_imbalance_rate',
                 'order_flow_imbalance_shock',
                 'cost_of_liquidity_imbalance_10bps',
                 'cost_of_liquidity_imbalance_100bps',
                 'cost_of_liquidity_imbalance_500bps',
                 'trade_flow_imbalance',
                 'queue_depletion_imbalance',
                 'order_arrival_imbalance',
                 'baseline' ]

    transfer_entropy_samples = {}
    for feature in features:
        transfer_entropy_samples[feature] = {}
        for ema_norm_step_count in ema_norm_step_variants:
            transfer_entropy_samples[feature][ema_norm_step_count] = []

    print( 'Data files found:', len(data_files) )

    with Client(TOKEN, target=INVEST_GRPC_API) as client:

        dpg.create_context()
        dpg.create_viewport( title='Preview Data', width=1280, height=1280 )
        dpg.setup_dearpygui()

        max_num_plots_per_row = 6

        with dpg.window(tag="primary_window", label="Preprocess data"):
            with dpg.group(horizontal=True):
                dpg.add_text( default_value="Files processed: " )
                dpg.add_text( tag="files_processed", default_value="0/0", color=[255, 254, 255])
            with dpg.group(horizontal=True):
                dpg.add_text( default_value="Filename: " )
                dpg.add_text( tag="filename", default_value="", color=[255, 254, 255])

            feature_blocks = [features[i : i + max_num_plots_per_row] for i in range(0, len(features), max_num_plots_per_row)]
            for feature_block in feature_blocks:
                with dpg.group(horizontal=True):
                    for feature in feature_block:
                        with dpg.plot(label=feature, width=512, height=256):
                            dpg.add_plot_legend()
                            dpg.add_plot_axis(dpg.mvXAxis, tag="x_axis_"+feature )
                            dpg.add_plot_axis(dpg.mvYAxis, tag="y_axis_"+feature )
                            for ema_norm_step_count in ema_norm_step_variants:
                                dpg.add_line_series( [j for j in range(0, 1)], [random.random() for j in range(0, 1)], label=str(ema_norm_step_count), parent="x_axis_"+feature, tag="series_"+feature+str(ema_norm_step_count) )

        dpg.show_viewport()
        dpg.set_primary_window("primary_window", True)

        max_num_processes = 12
        max_queue_size = 2048
        max_queue_fetch_steps = int( max_queue_size / max_num_processes )

        files_processed = 0
        output_queue = mp.Queue( maxsize=max_queue_size )
        processes = []
        
        def join_process(pid):
            global processes
            for i in range(len(processes) - 1, -1, -1):
                if processes[i].pid == pid:
                    processes[i].join()
                    del processes[i]
        
        while len(data_files) > 0 or len(processes) > 0:

            while len(processes) < max_num_processes and len(data_files) > 0:
                filename = data_files[0]
                del data_files[0]

                if 'evening' in str.lower(filename):
                    continue

                ticker = filename[ 0: filename.find("_") ]

                process = mp.Process(target=compute_transfer_entropy, args=(output_queue, ticker, filename, features, ema_norm_step_variants, ema_norm_base_time ))
                process.start()
                processes.append(process)

                dpg.set_value("filename", filename)
                dpg.set_value("files_processed", str(files_processed))
                files_processed = files_processed + 1

            max_queue_fetch_steps = int(output_queue.qsize() / 2) + 1

            dirty_plots = set()

            for step in range(max_queue_fetch_steps):
                try:
                    tuple = output_queue.get_nowait() # output_queue.put( (pid, False, feature, te) )
                    pid = tuple[0]
                    done = tuple[1]
                    feature = tuple[2]
                    ema_norm_step_count = tuple[3]
                    te = tuple[4]
                    if done:                        
                        join_process( pid )
                    else:
                        transfer_entropy_samples[feature][ema_norm_step_count].append( te )
                        dirty_plots.add( (feature,ema_norm_step_count) )
                except Empty:
                    time.sleep( 0.0 )
                    break  # nothing in the queue right now

            for tuple in dirty_plots:
                feature = tuple[0]
                ema_norm_step_count = tuple[1]
                te_histogram, te_labels = TkStatistics.compute_histogram( transfer_entropy_samples[feature][ema_norm_step_count], 0.001 )
                te_histogram = np.log1p(te_histogram)
                te_histogram = te_histogram.tolist()
                te_labels = te_labels.tolist()
                TkUI.set_series_with_labels("y_axis_"+feature, "x_axis_"+feature, "series_"+feature+str(ema_norm_step_count), te_histogram, te_labels)

            dpg.render_dearpygui_frame()

            if not dpg.is_dearpygui_running():
                break

        while dpg.is_dearpygui_running():
            dpg.render_dearpygui_frame()

        dpg.destroy_context()

