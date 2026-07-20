
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
    num_regimes = int(config['TimeSeries']['NumMarketRegimes'])
    market_regime_steps_count = int(config['TimeSeries']['MarketRegimeStepsCount'])


    data_files = [filename for filename in listdir(data_path) if (data_extension in filename) and isfile(join(data_path, filename))]
    print( 'Data files found:', len(data_files) )

    files_by_ticker = group_by_ticker(data_files)
    print( 'Tickers found:', len(files_by_ticker) )

    with Client(TOKEN, target=INVEST_GRPC_API) as client:

        total_samples = 0
        files_processed = 0
        start_time = time.time()
        feedback_time = time.time()

        # prepare list of data sources

        data_sources = []

        print("{")

        for ticker in files_by_ticker:

            #if ticker != 'CHMK':
            #    continue

            share = TkInstrument(client, config,  InstrumentType.INSTRUMENT_TYPE_SHARE, ticker, "TQBR")

            num_data_sources = len(files_by_ticker[ticker])
            num_test_data_sources = max(1, int( num_data_sources * test_data_ratio ))
            num_training_data_sources = num_data_sources - num_test_data_sources

            prices = []

            for i in range(num_training_data_sources):

                date_and_filename = files_by_ticker[ticker][i]
                date = date_and_filename[0]
                filename = date_and_filename[1]
                raw_samples = TkIO.read_at_path( join( data_path, filename) )
                raw_sample_count = int( len(raw_samples) / 2 ) # [ orderbook, last_trades, .... ]

                for j in range( raw_sample_count ):
                    orderbook_sample = raw_samples[j*2]
                    prices.append( quotation_to_float(orderbook_sample.last_price) )

            
            regime_thresholds = TkStatistics.calculate_regime_thresholds( np.array(prices), num_regimes, market_regime_steps_count)
            
            print( '"' + ticker + '":', regime_thresholds, "," )

        print("}")


        print( 'Total num files:', len(data_sources) )

