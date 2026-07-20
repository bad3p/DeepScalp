
import os
import sys
import time
import pickle
import logging
import numpy as np
import random
import configparser
import json
import threading
import multiprocessing
import multiprocessing.connection as mpc
from decimal import Decimal
from os.path import join
from datetime import date, datetime, timedelta
from tinkoff.invest.constants import INVEST_GRPC_API
from tinkoff.invest import (
    Client,
    InstrumentType,
    InstrumentIdType,
    SecurityTradingStatus,
    GetOrderBookResponse,
    OrderDirection
)
from tinkoff.invest.utils import decimal_to_quotation, quotation_to_decimal, money_to_decimal
from tinkoff.invest.schemas import Quotation, TradeSourceType, OrderExecutionReportStatus, OrderStateStreamRequest, ReplaceOrderRequest, PriceType
from tinkoff.invest.utils import now
from tinkoff.invest.exceptions import RequestError
from TkModules.TkIO import TkIO
from TkModules.TkInstrument import TkInstrument
from TkModules.TkPersistentQueue import TkPersistentQueue
from TkModules.TkQuotation import quotation_to_float

#------------------------------------------------------------------------------------------------------------------------
# Helper wrapper class over persistent JSON container storing validation service state
#------------------------------------------------------------------------------------------------------------------------

class TkValidationServiceState():

    def __init__(self, _cfg : configparser.ConfigParser):

        data_path = _cfg['Paths']['DataPath']
        validation_service_state_filename = _cfg['Paths']['ValidationServiceFilename']
        self._validation_service_state_path = join(data_path, validation_service_state_filename)

        content = '{"Positions":[], "Deals":[]}'
        if os.path.exists( self._validation_service_state_path ): 
            with open( self._validation_service_state_path, "rt" ) as f:
                content = f.read()
        self._state = json.loads( content ) 

    def flush(self):
        content = json.dumps( self._state )
        with open( self._validation_service_state_path, "wt" ) as f:
            f.write(content)       

    def allocate_position(self, ticker:str, profit:float):
        self._state["Positions"].insert(0, {"ticker":ticker, "state":"BUY", "timestamp":time.time(), "profit":profit, "cost":0.0 } )

    def shift_current_position(self):
        if not self.is_empty():
            current_position = self._state["Positions"][0]
            del self._state["Positions"][0]
            self._state["Positions"].append(current_position)

    def delete_current_position(self):
        if not self.is_empty():
            del self._state["Positions"][0]

    def is_empty(self):
        result = len(self._state["Positions"]) == 0
        return result
    
    def has_ticker(self, ticker:str):
        filtered_items = [item for item in self._state["Positions"] if item["ticker"] == ticker]
        return len(filtered_items) > 0
    
    def get_current_position_ticker(self):
        return '' if self.is_empty() else self._state["Positions"][0]["ticker"]
    
    def get_current_position_action(self):
        return 0 if self.is_empty() else ( 1 if self._state["Positions"][0]["state"] == "BUY" else -1 )
    
    def set_current_position_action(self, action:int):
        if not self.is_empty():
            self._state["Positions"][0]["state"] = ('BUY' if action > 0 else 'SELL')

    def get_current_position_profit(self):
        return 0 if self.is_empty() else ( self._state["Positions"][0]["profit"] )

    def get_current_position_cost(self):
        return self._state["Positions"][0]["cost"]

    def set_current_position_cost(self, cost:float):
        self._state["Positions"][0]["cost"] = cost
    
    def get_current_position_life_time(self):
        return 0 if self.is_empty() else ( time.time() - self._state["Positions"][0]["timestamp"] )
    
    def report_deal(self, ticker, cost:float, price:float, profit:float, forecast:float):
        self._state["Deals"].insert(0, {"ticker":ticker, "cost":cost, "price":price, "profit":profit, "forecast":forecast } )

#------------------------------------------------------------------------------------------------------------------------
# Validation service iteration
#------------------------------------------------------------------------------------------------------------------------

def validation_service_iteration():

    TOKEN = os.environ["TK_TOKEN"]

    #logger = logging.getLogger(__name__)
    #logging.basicConfig(level=logging.INFO)

    config = configparser.ConfigParser()
    config.read( 'TkConfig.ini' )

    position_life_time = float(config['TradingService']['PositionLifeTime'])
    trading_fee = float(config['TradingService']['TradingFee'])
    trading_forecast_modifier = float(config['TradingService']['TradingForecastModifier'])

    with Client(TOKEN, target=INVEST_GRPC_API) as client:

        def get_account_id():
            try:
                account = client.users.get_accounts().accounts[0]
                return account.id
            except RequestError as e:
                # 0 ok
                # > 0 request error, wait for specific time
                if e.metadata != None:
                    sys.exit( e.metadata.ratelimit_reset )
                else:
                    sys.exit(0)

        try:
            time.sleep(1.0)
            validation_state = TkValidationServiceState( config )
            if not validation_state.is_empty():

                account_id = get_account_id()        

                ticker = validation_state.get_current_position_ticker()
                action = validation_state.get_current_position_action()
                profit = validation_state.get_current_position_profit()
                life_time = validation_state.get_current_position_life_time()

                share = TkInstrument(client, config, InstrumentType.INSTRUMENT_TYPE_SHARE, ticker, "TQBR")
                min_price_increment = quotation_to_decimal(share.min_price_increment())
                lot = share.lot()

                share_trading_status = share.trading_status()
                if share_trading_status == SecurityTradingStatus.SECURITY_TRADING_STATUS_DEALER_NORMAL_TRADING or share_trading_status == SecurityTradingStatus.SECURITY_TRADING_STATUS_NORMAL_TRADING:
                    print( ticker, share.figi(), action, profit, life_time )
                    if action > 0 : # SIMULATE BUYING

                        orderbook = share.get_order_book( 20 )
                        if len(orderbook.asks) > 0:
                            cost = quotation_to_float( orderbook.asks[0].price )
                            cost = round( lot * (cost + cost * trading_fee), 2 )
                            validation_state.set_current_position_cost( cost )
                            validation_state.set_current_position_action( -1 )
                        
                        validation_state.shift_current_position()
                        
                    elif action < 0 : # SIMULATE SELLING
                        
                        life_time = validation_state.get_current_position_life_time()
                        cost = validation_state.get_current_position_cost()
                        forecast = validation_state.get_current_position_profit()

                        # adjust forecast
                        forecast = forecast * trading_forecast_modifier

                        # adjust profit accordingly to position life time
                        profit_factor = life_time / position_life_time
                        profit_factor = max( 0.0, profit_factor )
                        profit_factor = min( 1.0, profit_factor )
                        profit = forecast * ( 1.0 - profit_factor )

                        last_trades_end_date = now() 
                        last_trades_start_date = now() - timedelta( seconds=round(life_time) )
                        last_trades = share.get_last_trades( last_trades_start_date, last_trades_end_date, TradeSourceType.TRADE_SOURCE_UNSPECIFIED )

                        if len(last_trades.trades) > 0:

                            max_price = quotation_to_float( last_trades.trades[0].price )
                            for trade in last_trades.trades:
                                max_price = max( max_price, quotation_to_float( trade.price ) )
                                
                            price = max_price                            

                            if ( price >= cost + cost * profit / 100.0 ) or ( life_time > position_life_time ):
                                price = round( lot * (price - price * trading_fee), 2 )
                                validation_state.report_deal( ticker, cost, price, (price - cost)/cost * 100, validation_state.get_current_position_profit() )
                                validation_state.delete_current_position()
                            else:
                                validation_state.shift_current_position()
                        else:
                            validation_state.shift_current_position()

                    validation_state.flush()
                else:
                    print("Trading suspended...")
            else:
                print("Idle...")

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
# Trading service loop
# Multiprocessing environment required to workaround the issues in tinkoff.invest API, resulting in blocking RPC calls.
#------------------------------------------------------------------------------------------------------------------------

if __name__ ==  '__main__':

    config = configparser.ConfigParser()
    config.read( 'TkConfig.ini' )

    def is_idle():
        validation_state = TkValidationServiceState(config)
        result = validation_state.is_empty()
        return result 

    data_path = config['Paths']['DataPath']
    validation_service_state_filename = config['Paths']['ValidationServiceFilename']
    validation_service_state_path = join(data_path, validation_service_state_filename)
    trading_service_address = config['IPC']['TradingServiceAddress']
    trading_service_port = int(config['IPC']['TradingServicePort'])
    trading_service_auth_key = bytes( config['IPC']['TradingServiceAuthKey'], 'ascii' )
    max_iteration_time = float( config['TradingService']['MaxIterationTime'] )

    ipc_message_queue = []

    def ipc_thread_func():
        global ipc_message_queue
        global trading_service_address
        global trading_service_port
        global trading_service_auth_key

        print('IPC thread started')
        eof_counter = 0
        cre_counter = 0
        
        while True:
            try:
                with mpc.Listener( (trading_service_address,trading_service_port), authkey=trading_service_auth_key ) as listener:
                    with listener.accept() as conn:
                        try:
                            message = conn.recv()
                            ipc_message_queue.append( message )
                            print( 'Received', message )
                        except EOFError:                
                            eof_counter = eof_counter + 1
            except ConnectionResetError:
                cre_counter = cre_counter + 1
                
    ipc_thread = threading.Thread( target=ipc_thread_func )
    ipc_thread.daemon = True
    ipc_thread.start()

    while True:

        if len(ipc_message_queue) > 0:
            message = ipc_message_queue[0]
            del ipc_message_queue[0]
            print( "Opportunity in ", message[0], " profit: ", message[1] )
            validation_state = TkValidationServiceState(config)
            if not validation_state.has_ticker( ticker=message[0] ):
                validation_state.allocate_position( ticker=message[0], profit=message[1] )
                validation_state.flush()

        if is_idle():
            time.sleep(0.001)

        else:

            process = multiprocessing.Process(target=validation_service_iteration, args=())
            process.daemon = True
            process.start()

            start_time = time.time()
            while process.exitcode == None:
                if time.time() - start_time > max_iteration_time:
                    print( 'Trading process exceeds maximal timeout, will be terminated' )
                    break
                time.sleep(0.001)

            if process.exitcode == None:
                break
            elif process.exitcode > 0:
                print( 'API exhausted, timeout for ', process.exitcode, " sec." )
                time.sleep(process.exitcode)
                break
