
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
from TkModules.TkPreprocessFile import preprocess_file_for_training

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
    
    data_extension = config['Paths']['MarketDataFileExtension']
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
                dpg.add_text( default_value="MPC Queue Size: " )
                dpg.add_text( tag="mpc_queue_size", default_value="0", color=[255, 254, 255])
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
        feedback_time = time.time()

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

        max_num_processes = 12
        max_queue_size = 2048
        max_queue_fetch_steps = int( max_queue_size / max_num_processes )

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
            global feedback_time
            if is_test_data_source:
                test_index.append( test_data_offset )
                TkIO.write_to_file( test_data_stream, sample )
                test_data_offset = test_data_stream.tell()
            else:
                training_index.append( training_data_offset )
                TkIO.write_to_file( training_data_stream, sample )
                training_data_offset = training_data_stream.tell()
            if is_priority_sample and (time.time() - feedback_time) > 1.0:
                feedback_time = time.time()
                ts_input = sample[0]
                ts_target = sample[1]
                TkUI.set_series("x_axis_input","y_axis_input","input_series", ts_input)
                ts_target_sample_view = ts_target[-1]
                TkUI.set_series("x_axis_target","y_axis_target","target_series", ts_target_sample_view)                
                dpg.set_value("samples_processed", str(len(training_index)) + "/" + str(len(test_index)))

        while len(data_sources) > 0 or len(processes) > 0:

            while len(processes) < max_num_processes and len(data_sources) > 0:
                data_source = data_sources[0]
                del data_sources[0]

                ticker = data_source[0]                
                is_test_data_source = data_source[1]
                filename = data_source[2]

                process = mp.Process(target=preprocess_file_for_training, args=(output_queue, ticker, is_test_data_source, filename))
                process.start()
                processes.append(process)

                dpg.set_value("filename", filename)                
                dpg.set_value("files_remaining", str(len(data_sources)))
                dpg.set_value("samples_processed", str(len(training_index)) + "/" + str(len(test_index)))

            dpg.set_value("mpc_queue_size", str(output_queue.qsize()))

            for step in range(max_queue_fetch_steps):
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
                    break  # nothing in the queue right now

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
            if not dpg.is_dearpygui_running():
                break

        for i in range(len(processes) - 1, -1, -1):
            processes[i].terminate()

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