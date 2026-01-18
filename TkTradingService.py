
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

#------------------------------------------------------------------------------------------------------------------------
# Helper wrapper class over persistent JSON container storing trading service state
#------------------------------------------------------------------------------------------------------------------------

class TkTradingServiceState():

    def __init__(self, _cfg : configparser.ConfigParser):

        data_path = _cfg['Paths']['DataPath']
        trading_service_state_filename = _cfg['Paths']['TradingServiceStateFileName']
        self._trading_service_state_path = join(data_path, trading_service_state_filename)

        content = '[]'
        if os.path.exists( self._trading_service_state_path ): 
            with open( self._trading_service_state_path, "rt" ) as f:
                content = f.read()
        self._state = json.loads( content ) 

    def flush(self):
        content = json.dumps( self._state )
        with open( self._trading_service_state_path, "wt" ) as f:
            f.write(content)       

    def allocate_position(self, ticker:str, profit:float):
        self._state.insert(0, {"ticker":ticker, "state":"BUY", "profit":profit, "timestamp":time.time()} )

    def shift_current_position(self):
        if not self.is_empty():
            current_position = self._state[0]
            del self._state[0]
            self._state.append(current_position)

    def delete_current_position(self):
        if not self.is_empty():
            del self._state[0]

    def is_empty(self):
        result = len(self._state) == 0
        return result
    
    def has_ticker(self, ticker:str):
        filtered_items = [item for item in self._state if item["ticker"] == ticker]
        return len(filtered_items) > 0
    
    def get_current_position_ticker(self):
        return '' if self.is_empty() else self._state[0]["ticker"]
    
    def get_current_position_action(self):
        return 0 if self.is_empty() else ( 1 if self._state[0]["state"] == "BUY" else -1 )
    
    def set_current_position_action(self, action:int):
        if not self.is_empty():
            self._state[0]["state"] = ('BUY' if action > 0 else 'SELL')

    def get_current_position_profit(self):
        return 0 if self.is_empty() else ( self._state[0]["profit"] )
    
    def get_current_position_life_time(self):
        return 0 if self.is_empty() else ( time.time() - self._state[0]["timestamp"] )

#------------------------------------------------------------------------------------------------------------------------
# Trading service iteration
#------------------------------------------------------------------------------------------------------------------------

def trading_service_iteration():

    TOKEN = os.environ["TK_TOKEN"]

    #logger = logging.getLogger(__name__)
    #logging.basicConfig(level=logging.INFO)

    config = configparser.ConfigParser()
    config.read( 'TkConfig.ini' )

    position_life_time = float(config['TradingService']['PositionLifeTime'])
    relative_position_limit = float(config['TradingService']['RelativePositionLimit'])

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

        def get_portfolio(id):
            try:
                portfolio = client.operations.get_portfolio( account_id=id )
                return portfolio
            except RequestError as e:
                # 0 ok
                # > 0 request error, wait for specific time
                if e.metadata != None:
                    sys.exit( e.metadata.ratelimit_reset + 1 )
                else:
                    sys.exit(0)

        def get_currency(portfolio):
            filtered_positions = list( filter(lambda position : position.figi == 'RUB000UTSTOM', portfolio.positions) )
            if( len(filtered_positions) > 0):
                return quotation_to_decimal( filtered_positions[0].quantity )
            else:
                return Decimal(0)
            
        def find_position(portfolio, figi):
            for i in range(len(portfolio.positions)):
                if portfolio.positions[i].figi == figi:
                    return portfolio.positions[i]
            return None

        try:
            trading_state = TkTradingServiceState( config )
            if not trading_state.is_empty():

                account_id = get_account_id()
                portfolio = get_portfolio( account_id )                
                balance = get_currency(portfolio)
                position_limit = balance * Decimal(relative_position_limit) / Decimal(100.0)                

                ticker = trading_state.get_current_position_ticker()
                action = trading_state.get_current_position_action()
                profit = trading_state.get_current_position_profit()
                life_time = trading_state.get_current_position_life_time()

                share = TkInstrument(client, config, InstrumentType.INSTRUMENT_TYPE_SHARE, ticker, "TQBR")
                min_price_increment = quotation_to_decimal(share.min_price_increment())
                lot = share.lot()

                share_trading_status = share.trading_status()
                if share_trading_status == SecurityTradingStatus.SECURITY_TRADING_STATUS_DEALER_NORMAL_TRADING or share_trading_status == SecurityTradingStatus.SECURITY_TRADING_STATUS_NORMAL_TRADING:
                    print( ticker, share.figi(), action, profit, life_time )
                    if action > 0 : # BUYING
                        position = find_position(portfolio, share.figi())
                        if position == None:
                            orders = client.orders.get_orders(account_id=account_id)
                            filtered_orders = list( filter(lambda order : order.figi == share.figi() and order.direction == OrderDirection.ORDER_DIRECTION_BUY, orders.orders) )
                            if len(filtered_orders) == 0:
                                last_price = share.get_last_price()
                                quantity = int( position_limit / (last_price * lot) )
                                if quantity > 0:                                    
                                    print('Buy order: ', ticker, last_price, quantity )
                                    share.post_buy_market_order( account_id, quantity, last_price )
                                else:
                                    print('Not enough funds to buy: ', ticker )
                                    trading_state.delete_current_position()
                        else:
                            trading_state.set_current_position_action(-1)
                    elif action < 0 : # SELLING
                        position = find_position(portfolio, share.figi())
                        if position == None:
                            print( 'Closing position: ', ticker )
                            trading_state.delete_current_position()
                        else:
                            last_price = share.get_last_price()
                            profit_price = last_price + last_price * Decimal(profit) / Decimal(100.0)
                            profit_price = round( profit_price / min_price_increment ) * min_price_increment
                            delta_price = profit_price - last_price
                            sell_price = last_price + delta_price * max( Decimal(0), (Decimal(1.0) - Decimal(life_time) / Decimal(position_life_time)))
                            sell_price = round( sell_price / min_price_increment ) * min_price_increment - min_price_increment
                            print(last_price, profit_price, sell_price)

                            orders = client.orders.get_orders(account_id=account_id)
                            filtered_orders = list( filter(lambda order : order.figi == share.figi() and order.direction == OrderDirection.ORDER_DIRECTION_SELL, orders.orders) )

                            if len(filtered_orders) == 0:
                                share.post_sell_limit_order( account_id, int(position.quantity.units / lot), sell_price )
                            else:   
                                initial_order_price = Quotation(filtered_orders[0].initial_order_price.units, filtered_orders[0].initial_order_price.nano)                                
                                initial_order_price = quotation_to_decimal(initial_order_price)
                                initial_order_price = initial_order_price / (filtered_orders[0].lots_requested * lot)
                                if initial_order_price != sell_price:
                                    client.orders.cancel_order( account_id=account_id, order_id=filtered_orders[0].order_id )
                                    share.post_sell_limit_order( account_id, (filtered_orders[0].lots_requested-filtered_orders[0].lots_executed), sell_price )
                            trading_state.shift_current_position()
                    trading_state.flush()
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
        trading_state = TkTradingServiceState(config)
        result = trading_state.is_empty()
        return result 

    data_path = config['Paths']['DataPath']
    trading_service_state_filename = config['Paths']['TradingServiceStateFileName']
    trading_service_state_path = join(data_path, trading_service_state_filename)    
    trading_service_address = config['IPC']['TradingServiceAddress']
    trading_service_port = int(config['IPC']['TradingServicePort'])
    trading_service_auth_key = bytes( config['IPC']['TradingServiceAuthKey'], 'ascii' )

    max_iteration_time = float( config['TradingService']['MaxIterationTime'] )
    test_mode = config['TradingService']['TestMode'] == "True"
    test_tickers = json.loads( config['TradingService']['TestTickers'] )
    test_profit = float( config['TradingService']['TestProfit'] )

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
                        except EOFError:                
                            eof_counter = eof_counter + 1
            except ConnectionResetError:
                cre_counter = cre_counter + 1
                
    ipc_thread = threading.Thread( target=ipc_thread_func )
    ipc_thread.daemon = True
    ipc_thread.start()

    while True:

        if test_mode:
            trading_state = TkTradingServiceState(config)
            if random.uniform(0.0, 1.0) < 0.5:
                random_ticker = test_tickers[random.randint(0, len(test_tickers)-1)]
                if not trading_state.has_ticker(random_ticker):                
                    trading_state.allocate_position( random_ticker, profit=test_profit)
                    trading_state.flush()

        if len(ipc_message_queue) > 0:
            message = ipc_message_queue[0]
            del ipc_message_queue[0]
            print( "Opportunity in ", message[0], " profit: ", message[1] )
            trading_state = TkTradingServiceState(config)
            if not trading_state.has_ticker( ticker=message[0] ):
                trading_state.allocate_position( ticker=message[0], profit=message[1])
                trading_state.flush()

        if is_idle():
            time.sleep(0.001)

        else:

            process = multiprocessing.Process(target=trading_service_iteration, args=())
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
