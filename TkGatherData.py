
import os
import sys
import time
import pickle
import numpy as np
import configparser
import json
import multiprocessing
from os.path import join
from datetime import date, datetime, timedelta
from tinkoff.invest.constants import INVEST_GRPC_API
from tinkoff.invest import (
    Client,
    InstrumentType,
    InstrumentIdType,
    SecurityTradingStatus,
    GetOrderBookResponse,
)
from tinkoff.invest.schemas import CandleInterval, TradeSourceType
from tinkoff.invest.utils import now
from tinkoff.invest.exceptions import RequestError
from TkModules.TkQuotation import quotation_to_float
from TkModules.TkIO import TkIO
from TkModules.TkInstrument import TkInstrument
from TkModules.TkPersistentQueue import TkPersistentQueue

#------------------------------------------------------------------------------------------------------------------------
# Gather data iteration
#------------------------------------------------------------------------------------------------------------------------

def gather_data_iteration(ticker:str, data_path:str, orderbook_file_extension:str, orderbook_depth:int, last_trades_period_in_minutes:int):

    TOKEN = os.environ["TK_TOKEN"]

    config = configparser.ConfigParser()
    config.read( 'TkConfig.ini' )

    with Client(TOKEN, target=INVEST_GRPC_API) as client:

        try:
            today = date.today()
            filename = ticker + "_" + today.strftime("%B_%d_%Y") + "_" + ( "Day" if datetime.now().hour < 19 else "Evening" ) + orderbook_file_extension
            path = join( data_path, filename )
            share = TkInstrument(client, config, InstrumentType.INSTRUMENT_TYPE_SHARE, ticker, "TQBR")
            share_trading_status = share.trading_status()
            if share_trading_status == SecurityTradingStatus.SECURITY_TRADING_STATUS_NORMAL_TRADING: # or share_trading_status == SecurityTradingStatus.SECURITY_TRADING_STATUS_DEALER_NORMAL_TRADING:
                last_trades_end_date = now() 
                last_trades_start_date = now() - timedelta( minutes=last_trades_period_in_minutes )
                order_book = share.get_order_book(orderbook_depth)
                last_trades = share.get_last_trades( last_trades_start_date, last_trades_end_date, TradeSourceType.TRADE_SOURCE_UNSPECIFIED )
                TkIO.append_at_path( path, order_book )
                TkIO.append_at_path( path, last_trades )
                print(ticker, "gathered.")
            else:
                print(ticker, "trading is suspended.")

        except RequestError as e:            
            # 0 ok
            # > 0 request error, wait for specific time (in seconds)
            if e.metadata != None:
                sys.exit(e.metadata.ratelimit_reset + 1)
            else:
                sys.exit(0)

    # 0 ok
    # > 0 request error, wait for specific time
    sys.exit(0)

#------------------------------------------------------------------------------------------------------------------------
# Data gathering loop
# Multiprocessing environment required to workaround the issues in tinkoff.invest API, resulting in blocking RPC calls.
#------------------------------------------------------------------------------------------------------------------------

if __name__ ==  '__main__':

    config = configparser.ConfigParser()
    config.read( 'TkConfig.ini' )

    data_path = config['Paths']['DataPath']
    gather_data_queue_filename = config['Paths']['GatherDataQueueFileName']
    orderbook_file_extension = config['Paths']['OrderbookFileExtension']
    orderbook_depth = int(config['GatherData']['OrderBookDepth'])
    last_trades_period_in_minutes = int(config['GatherData']['LastTradesPeriodInMinutes'])
    max_iteration_time = int(config['GatherData']['MaxIterationTime'])
    ignore_tickers = json.loads(config['GatherData']['IgnoreTickers'])

    def init_gather_data_queue_callback():
        print( 'Initializing gather data queue...' )
        TOKEN = os.environ["TK_TOKEN"]
        with Client(TOKEN, target=INVEST_GRPC_API) as client:        
            return TkInstrument.get_instrument_tickers(client, InstrumentType.INSTRUMENT_TYPE_SHARE, "TQBR")

    gather_data_queue_path = join( data_path, gather_data_queue_filename )
    queue = TkPersistentQueue( gather_data_queue_path, init_gather_data_queue_callback )

    while True:

        ticker = queue.pop()

        if not ( ticker in ignore_tickers ):

            process = multiprocessing.Process(target=gather_data_iteration, args=(ticker, data_path, orderbook_file_extension, orderbook_depth, last_trades_period_in_minutes))
            process.daemon = True
            process.start()

            start_time = time.time()
            while process.exitcode == None:
                if time.time() - start_time > max_iteration_time:
                    print( 'Gathering process exceeds maximal timeout, will be terminated' )
                    break
                time.sleep(1.0)

            if process.exitcode == None or process.exitcode > 0:
                break

        queue.push(ticker)
        queue.flush()