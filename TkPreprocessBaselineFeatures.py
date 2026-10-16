
import os
import time
import numpy as np
import configparser
import json
import copy
import random
import math
import dearpygui.dearpygui as dpg
from LSHash import LSHash
from os import listdir
from os.path import isfile, join
from datetime import date, datetime, timezone, timedelta
from dateutil import parser
import itertools
import threading
import torch
import multiprocessing as mp
from queue import Empty  # for non-blocking queue reads
from collections import defaultdict
from joblib import Parallel, delayed
from t_tech.invest.constants import INVEST_GRPC_API
from t_tech.invest import Client
from t_tech.invest import InstrumentType
from t_tech.invest import InstrumentIdType
from t_tech.invest import SecurityTradingStatus
from t_tech.invest import GetOrderBookResponse, GetLastTradesResponse
from t_tech.invest import HistoricCandle
from t_tech.invest.exceptions import RequestError
from t_tech.invest.utils import decimal_to_quotation, quotation_to_decimal
from TkModules.TkQuotation import quotation_to_float
from TkModules.TkIO import TkIO
from TkModules.TkInstrument import TkInstrument
from TkModules.TkStatistics import TkStatistics
from TkModules.TkUI import TkUI
from TkModules.TkBaseline import TkBaseline

#------------------------------------------------------------------------------------------------------------------------    
# Main process
#------------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":

    TOKEN = os.environ["TK_TOKEN"]

    config = configparser.ConfigParser()
    config.read( 'TkConfig.ini' )

    data_path = config['Paths']['DataPath']
    baseline_file_name = config['Paths']['BaselineFileName']   
    data_extension = config['Paths']['MarketDataFileExtension']

    data_files = [filename for filename in listdir(data_path) if (data_extension in filename) and isfile(join(data_path, filename))]
    print( 'Data files found:', len(data_files) )

    files_by_ticker = TkInstrument.group_by_ticker(data_files)
    print( 'Tickers found:', len(files_by_ticker) )

    discretization_in_hours = int( config['Baseline']['DiscretizationInHours'] )
    resting_volume_depth = int( config['Baseline']['RestingVolumeDepth'] )
    per_feature_alpha = json.loads( config['Baseline']['PerFeatureAlpha'] )
    
    baseline = TkBaseline( discretization_in_hours, resting_volume_depth, per_feature_alpha )

    with Client(TOKEN, target=INVEST_GRPC_API) as client:

        dpg.create_context()
        dpg.create_viewport()
        dpg.setup_dearpygui()

        with dpg.window(tag="primary_window", label="Preprocess data"):
            with dpg.group(horizontal=True):
                dpg.add_text( default_value="Files processed: " )
                dpg.add_text( tag="files_processed", default_value="0/0", color=[255, 254, 255])
            with dpg.group(horizontal=True):
                dpg.add_text( default_value="Data samples: " )
                dpg.add_text( tag="data_samples", default_value="0/0/0", color=[255, 254, 255])
            with dpg.group(horizontal=True):
                dpg.add_text( default_value="Filename: " )
                dpg.add_text( tag="filename", default_value="", color=[255, 254, 255])
            with dpg.group(horizontal=True):
                with dpg.plot(label="Raw baseline feature", width=1024, height=256):
                    dpg.add_plot_legend()
                    dpg.add_plot_axis(dpg.mvXAxis, tag="x_axis_raw" )
                    dpg.add_plot_axis(dpg.mvYAxis, tag="y_axis_raw" )
                    dpg.add_line_series( [j for j in range(0, 32)], [random.random() for j in range(0, 32)], label="Raw baseline feature", parent="x_axis_raw", tag="raw_series" )
            with dpg.group(horizontal=True):
                with dpg.plot(label="EMA-normalized baseline feature", width=1024, height=256):
                    dpg.add_plot_legend()
                    dpg.add_plot_axis(dpg.mvXAxis, tag="x_axis_ema_norm" )
                    dpg.add_plot_axis(dpg.mvYAxis, tag="y_axis_ema_norm" )
                    dpg.add_line_series( [j for j in range(0, 32)], [random.random() for j in range(0, 32)], label="EMA-normalized baseline feature", parent="x_axis_ema_norm", tag="ema_norm_series" )

        dpg.show_viewport()
        dpg.set_primary_window("primary_window", True)

        total_samples = 0
        files_processed = 0
        start_time = time.time()
        feedback_time = time.time()

        # prepare list of data sources

        data_sources = []

        for ticker in files_by_ticker:

            #if ticker != 'ABIO':
            #    continue

            share = TkInstrument(client, config,  InstrumentType.INSTRUMENT_TYPE_SHARE, ticker, "TQBR")

            num_data_sources = len(files_by_ticker[ticker])

            prices = []

            for i in range(num_data_sources):

                date_and_filename = files_by_ticker[ticker][i]
                date = date_and_filename[0]
                filename = date_and_filename[1]
                print( filename )

                raw_samples = TkIO.read_at_path( join( data_path, filename) )
                raw_sample_count = int( len(raw_samples) / 2 ) # [ orderbook, last_trades, .... ]

                baseline.add_samples( share, raw_samples, False ) # True = incremental normalization
                
                raw, ema, raw_time, ema_time = baseline.extract_baseline( 'spread_to_tick_ratio', share.ticker(), 200 )

                dpg.set_value("filename", filename)
                TkUI.set_series_with_labels("x_axis_raw","y_axis_raw","raw_series", raw, raw_time)
                TkUI.set_series_with_labels("x_axis_ema_norm","y_axis_ema_norm","ema_norm_series", ema, ema_time )

                if not dpg.is_dearpygui_running():
                    break

                dpg.render_dearpygui_frame()

            if not dpg.is_dearpygui_running():
                break            

        print( 'Applying EMA normalization...:' )
        baseline.apply_ema_normalization_across_periods()
        baseline.save( join( data_path, baseline_file_name) )
        
        print( 'Total num files:', len(data_sources) )

        while dpg.is_dearpygui_running():
            dpg.render_dearpygui_frame()

        dpg.destroy_context()

