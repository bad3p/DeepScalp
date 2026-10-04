
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
from TkModules.TkInstrument import TkInstrument, TkLastTrades
from TkModules.TkStatistics import TkStatistics
from TkModules.TkUI import TkUI
from dataclasses import dataclass

#------------------------------------------------------------------------------------------------------------------------
# Preprocessed data 
#------------------------------------------------------------------------------------------------------------------------

@dataclass
class PreprocessedData:

    all_trades:dict
    num_volatility_regimes:int
    num_trend_regimes:int
    min_price_increment:float
    regimes:list
    trends:list    
    trend_regimes:list

    # time data

    timestamp:list
    time_delta:list
    log_time_delta:list

    # unfiltered data
    
    price: list
    volatility: list        
    spread: list
    orderbook_volume: list
    orderbook_slope: list
    orderbook_microprice: list
    orderbook_bid_alpha: list
    orderbook_ask_alpha: list
    orderbook_alpha_imbalance: list
    orderbook_mean_alpha: list
    average_last_trades_volume: list
    last_trades_volume: list
    average_last_trades_num_events: list
    last_trades_num_events: list
    trade_flow_imbalance: list
    price_change: list
    price_change_rate: list
    short_term_integrated_order_flow_imbalance:list
    mid_term_integrated_order_flow_imbalance:list
    long_term_integrated_order_flow_imbalance:list
    queue_imbalance:list
    queue_depletion_intensity:list
    queue_depletion_imbalance:list
    order_arrival_intensity:list
    order_arrival_imbalance:list
    average_orderbook_volume: list
    cost_of_buying_liquidity_10bps:list
    cost_of_buying_liquidity_100bps:list
    cost_of_buying_liquidity_500bps:list
    cost_of_selling_liquidity_10bps:list
    cost_of_selling_liquidity_100bps:list
    cost_of_selling_liquidity_500bps:list
    cost_of_liquidity_imbalance_10bps:list
    cost_of_liquidity_imbalance_100bps:list
    cost_of_liquidity_imbalance_500bps:list

    # EMA-smoothed data

    smooth_volatility:list
    smooth_trends:list

    @staticmethod
    def interpolate_corrupted_datetimes(dt_list: list[datetime]) -> list[datetime]:
        n = len(dt_list)
        if n == 0:
            return []
        if n == 1:
            return dt_list.copy()

        # Step 1: Identify valid anchor points through iterative elimination
        valid_indices = list(range(n))
    
        while True:
            to_remove = set()
            m = len(valid_indices)
        
            for j in range(m):
                idx = valid_indices[j]
            
                # Rule 1: Replace if lesser than previous object
                if j > 0:
                    prev_idx = valid_indices[j-1]
                    if dt_list[idx] < dt_list[prev_idx]:
                        to_remove.add(idx)
                        continue
            
                # Rule 2: Replace if equal to the next object
                if j < m - 1:
                    next_idx = valid_indices[j+1]
                    if dt_list[idx] == dt_list[next_idx]:
                        to_remove.add(idx)
                        continue
        
            # Stop if no elements violate the rules in the current pass
            if not to_remove:
                break
            
            valid_indices = [idx for idx in valid_indices if idx not in to_remove]
        
            # Fallback if the array is fully eliminated (e.g., edge cases)
            if not valid_indices:
                valid_indices = [0]
                break

        # Initialize output array with the valid anchor points
        out = [None] * n
        for idx in valid_indices:
            out[idx] = dt_list[idx]

        # Step 2: Interpolation & Extrapolation
    
        # Handle Edge Case: Only 1 valid anchor remains
        if len(valid_indices) == 1:
            single_idx = valid_indices[0]
            # Fallback delta when interpolation is impossible
            delta = timedelta(seconds=1)
            for i in range(n):
                out[i] = dt_list[single_idx] + delta * (i - single_idx)
            return out

        # Interpolate gaps between valid anchors
        for k in range(len(valid_indices) - 1):
            i1 = valid_indices[k]
            i2 = valid_indices[k+1]
            dt1 = out[i1]
            dt2 = out[i2]
        
            delta = (dt2 - dt1) / (i2 - i1)
        
            for i in range(i1 + 1, i2):
                out[i] = dt1 + delta * (i - i1)
            
        # Extrapolate leading invalid objects backwards
        first_valid = valid_indices[0]
        if first_valid > 0:
            i1 = valid_indices[0]
            i2 = valid_indices[1]
            delta = (out[i2] - out[i1]) / (i2 - i1)
        
            for i in range(first_valid):
                out[i] = out[i1] + delta * (i - i1)
            
        # Extrapolate trailing invalid objects forwards
        last_valid = valid_indices[-1]
        if last_valid < n - 1:
            i1 = valid_indices[-2]
            i2 = valid_indices[-1]
            delta = (out[i2] - out[i1]) / (i2 - i1)
        
            for i in range(last_valid + 1, n):
                out[i] = out[i2] + delta * (i - i2)

        return out

    @staticmethod
    def fix_corrupted_datetimes(raw_samples:list):
        raw_sample_count = int( len(raw_samples) / 2 )
        orderbook_ts = [0.0] * raw_sample_count
        for i in range( 0, raw_sample_count ):
            orderbook_ts[i] = raw_samples[i*2].orderbook_ts
        fixed_orderbook_ts = PreprocessedData.interpolate_corrupted_datetimes( orderbook_ts )
        for i in range( 0, raw_sample_count ):
            raw_samples[i*2].orderbook_ts = fixed_orderbook_ts[i]    

    @staticmethod
    def select_prior_orderbooks(raw_samples:list, last_index:int, count:int):
        index = last_index
        result = []
        while count > 0 and index >= 0:
            result.insert( 0, raw_samples[index*2] )
            index = index - 1
            count = count -1
        return result

    def __init__(self, share:TkInstrument, raw_samples:list, orderbook_width:int, last_trades_width:int, last_trades_discretization:float, regime_ema_half_life:float, trend_ema_half_life:float, trend_steps_count:int, volatility_regimes:list, trend_regime_thresholds:list, future_steps_count:int, prior_steps_count:int):

        self.num_volatility_regimes = len(volatility_regimes) + 1
        self.num_trend_regimes = len(trend_regime_thresholds) + 1
        
        raw_sample_count = int( len(raw_samples) / 2 ) # [ orderbook, last_trades, .... ]

        # discretisation may vary (the exchange might occasionally change it per instrument)

        self.min_price_increment = TkStatistics.get_min_price_increment(raw_samples[0], quotation_to_decimal(share.min_price_increment()))
        for i in range(raw_sample_count):
            self.min_price_increment = min( self.min_price_increment, TkStatistics.get_min_price_increment( raw_samples[i*2], quotation_to_decimal(share.min_price_increment()) ) )
        self.min_price_increment = float(self.min_price_increment)

        # timestamps and time deltas

        PreprocessedData.fix_corrupted_datetimes(raw_samples)

        self.timestamp = [raw_samples[0].orderbook_ts.timestamp()] * raw_sample_count
        self.time_delta = [0.0] * raw_sample_count
        for i in range( 1, raw_sample_count ):
            prev_ts = raw_samples[(i-1)*2].orderbook_ts
            curr_ts = raw_samples[i*2].orderbook_ts
            self.timestamp[i] = raw_samples[i*2].orderbook_ts.timestamp()
            self.time_delta[i] = (curr_ts - prev_ts).total_seconds()            
            if self.time_delta[i] <= 0:
                raise ValueError("Invalid time_delta!")
        self.time_delta[0] = sum(self.time_delta) / len(self.time_delta)
        #print( 'Average delta time:',self.time_delta[0],' Max delta time:', max(self.time_delta))

        # Log-scaled time delta for immediate micro-burst detection
        self.log_time_delta = [math.log(td) for td in self.time_delta]

        # container of all trades             

        self.all_trades = {}

        for i in range( raw_sample_count ):
            last_trades_sample = raw_samples[i*2+1]
            for j in range( len(last_trades_sample.trades) ):
                trade_time = last_trades_sample.trades[j][2]
                trade_ts = int( trade_time.timestamp() )
                if trade_ts in self.all_trades:
                    if not last_trades_sample.trades[j] in self.all_trades[trade_ts]:
                        self.all_trades[trade_ts].append( last_trades_sample.trades[j] )
                else:
                    self.all_trades[trade_ts] = [last_trades_sample.trades[j]]

        for i in range( raw_sample_count ):
            last_trades_sample = raw_samples[i*2+1]
            for j in range( len(last_trades_sample.trades) ):
                trade_time = last_trades_sample.trades[j][2]
                trade_ts = int( trade_time.timestamp() )
                assert trade_ts in self.all_trades
                assert last_trades_sample.trades[j] in self.all_trades[trade_ts]
        

        # extract features from orderbook & last trades

        self.price = [0.0] * raw_sample_count
        self.volatility = [0.0] * raw_sample_count
        self.regimes = [0.0] * raw_sample_count
        self.spread = [0.0] * raw_sample_count
        self.orderbook_volume = [0] * raw_sample_count
        self.orderbook_slope = [0] * raw_sample_count
        self.orderbook_microprice = [0] * raw_sample_count
        self.orderbook_bid_alpha = [0] * raw_sample_count
        self.orderbook_ask_alpha = [0] * raw_sample_count
        self.orderbook_alpha_imbalance = [0] * raw_sample_count
        self.orderbook_mean_alpha = [0] * raw_sample_count
        self.last_trades_volume = [0] * raw_sample_count
        self.last_trades_rate = [0] * raw_sample_count
        self.last_trades_num_events = [0] * raw_sample_count
        self.trade_flow_imbalance = [0] * raw_sample_count        

        last_trades_time_threshold = None

        for i in range( raw_sample_count ):

            orderbook_sample = raw_samples[i*2]
            last_trades_sample = raw_samples[i*2+1]

            # orderbook_tensor, _, pivot_price, volume, slope, microprice = TkStatistics.orderbook_to_tensor( orderbook_sample, orderbook_width, min_price_increment )                        
            volume, pivot_price, microprice, slope, bid_alpha, ask_alpha = TkStatistics.orderbook_statistics( orderbook_sample, self.min_price_increment )
            vwap, vwvol, normalized_vwvol = TkStatistics.trades_statistics( last_trades_sample, last_trades_time_threshold )

            self.price[i] = vwap if vwap > 0.0 else pivot_price
            self.volatility[i] = normalized_vwvol
            self.spread[i] = TkStatistics.orderbook_spread( orderbook_sample, orderbook_width, self.min_price_increment )
            self.orderbook_volume[i] = volume
            self.orderbook_slope[i] = slope
            self.orderbook_microprice[i] = microprice
            self.orderbook_bid_alpha[i] = bid_alpha
            self.orderbook_ask_alpha[i] = ask_alpha
            self.orderbook_alpha_imbalance[i] = (bid_alpha - ask_alpha) / max(1.0, bid_alpha + ask_alpha)
            self.orderbook_mean_alpha[i] = (bid_alpha + ask_alpha) / 2.0

            last_trades_samples = [ (raw_samples[i*2+1], last_trades_time_threshold) ]
            last_trades_tensor, _, num_events, volume, buy_trades, sell_trades, _ = TkStatistics.last_trades_to_tensor( last_trades_samples, pivot_price, last_trades_width, last_trades_discretization )
            self.last_trades_volume[i] = volume
            self.last_trades_rate[i] = volume / self.time_delta[i]
            self.last_trades_num_events[i] = num_events
            self.trade_flow_imbalance[i] = (buy_trades - sell_trades) / max(1.0, buy_trades + sell_trades)

            # adjust minimal time for next last trades sample
            last_trades_time_threshold = orderbook_sample.orderbook_ts

        # normalization for the window equal to the prior steps count
        prior_steps_count_norm = [min(i+1, prior_steps_count) for i in range(raw_sample_count)]

        # normalize orderbook volume
        self.average_orderbook_volume = TkStatistics.rolling_sum( self.orderbook_volume, window=prior_steps_count ).tolist()
        self.average_orderbook_volume = [ int(a / b) for a, b in zip(self.average_orderbook_volume, prior_steps_count_norm)]
        self.orderbook_volume = [ a / b if b > 0.0 else 0.0 for a,b in zip(self.orderbook_volume, self.average_orderbook_volume) ]

        # normalize last trades volume 
        self.average_last_trades_volume = TkStatistics.rolling_sum( self.last_trades_volume, window=prior_steps_count ).tolist()
        self.average_last_trades_volume = [ int(a / b) for a, b in zip(self.average_last_trades_volume, prior_steps_count_norm)]
        self.last_trades_volume = [ a / b if b > 0.0 else 0.0 for a,b in zip(self.last_trades_volume, self.average_last_trades_volume) ]

        # normalize last trades volume 
        self.average_last_trades_num_events = TkStatistics.rolling_sum( self.last_trades_num_events, window=prior_steps_count ).tolist()
        self.average_last_trades_num_events = [ int(a / b) for a, b in zip(self.average_last_trades_num_events, prior_steps_count_norm)]
        self.last_trades_num_events = [ a / b if b > 0.0 else 0.0 for a,b in zip(self.last_trades_num_events, self.average_last_trades_num_events) ]

        # recompute last trades volume rate
        self.last_trades_rate = [ a / b for a,b in zip(self.last_trades_volume, self.time_delta) ]

        self.smooth_volatility = TkStatistics.irregular_ema_smoothing( self.volatility, self.time_delta, half_life=regime_ema_half_life ).tolist()
        for i in range( raw_sample_count ):
            self.regimes[i] = TkStatistics.volatility_to_market_regime( self.smooth_volatility[i], volatility_regimes )

        self.trends = TkStatistics.price_to_trends( self.price, trend_steps_count ).tolist()
        self.smooth_trends = TkStatistics.irregular_ema_smoothing( self.trends, self.time_delta, half_life=trend_ema_half_life ).tolist()
        self.trend_regimes = TkStatistics.trends_to_trend_regimes( self.smooth_trends, trend_regime_thresholds )

        self.price_change = [0.0] * raw_sample_count
        self.price_change_rate = [0.0] * raw_sample_count
        for i in range( raw_sample_count ):
            if i == 0:
                self.price_change[i] = 0.0
                self.price_change_rate[i] = 0.0
            else:
                self.price_change[i] = ( self.price[i] - self.price[i-1] ) / self.price[i]
                self.price_change_rate[i] = self.price_change[i] / self.time_delta[i]

        # normalize microprice offset and spread
        for i in range( raw_sample_count ):
            self.orderbook_microprice[i] = self.orderbook_microprice[i] / self.price[i]
            self.spread[i] = self.spread[i] / self.price[i]

        self.short_term_integrated_order_flow_imbalance = [0] * raw_sample_count
        self.mid_term_integrated_order_flow_imbalance = [0] * raw_sample_count
        self.long_term_integrated_order_flow_imbalance = [0] * raw_sample_count
        self.queue_imbalance = [0] * raw_sample_count
        self.queue_depletion_intensity = [0] * raw_sample_count
        self.queue_depletion_imbalance = [0] * raw_sample_count
        self.order_arrival_intensity = [0] * raw_sample_count
        self.order_arrival_imbalance = [0] * raw_sample_count

        for i in range( raw_sample_count ):
            if i == 0:
                self.order_arrival_intensity[i] = 0.0
                self.order_arrival_imbalance[i] = 0.0
            else:                    
                bid_intensity, ask_intensity = TkStatistics.depth_weighted_order_arrival_rate( raw_samples[(i-1)*2], raw_samples[i*2], raw_samples[i*2+1], alpha=1.0, dt=self.time_delta[i] ) # TODO: configure alpha
                self.order_arrival_intensity[i] = bid_intensity + ask_intensity
                self.order_arrival_imbalance[i] = bid_intensity - ask_intensity
            self.queue_imbalance[i] = TkStatistics.depth_weighted_queue_imbalance( raw_samples[i*2], self.min_price_increment, alpha=1.0 ) # TODO: configure alpha
            bid_depletion, ask_depletion = TkStatistics.depth_weighted_queue_depletion_rate( raw_samples[i*2], raw_samples[i*2+1], alpha=1.0 ) # TODO: configure alpha
            self.queue_depletion_intensity[i] = bid_depletion + ask_depletion
            self.queue_depletion_imbalance[i] = bid_depletion - ask_depletion            
            
            orderbooks = PreprocessedData.select_prior_orderbooks( raw_samples, i, 2 )
            if len(orderbooks) > 1:
                self.short_term_integrated_order_flow_imbalance[i] = TkStatistics.depth_weighted_integrated_order_flow_imbalance( orderbooks, depth = 8, decay_rate = 0.125 ) # TODO: configure depth and decay_rate
            else:
                self.short_term_integrated_order_flow_imbalance[i] = 0.0
            assert not math.isnan(self.short_term_integrated_order_flow_imbalance[i])

            orderbooks = PreprocessedData.select_prior_orderbooks( raw_samples, i, 8 )
            if len(orderbooks) > 1:
                self.mid_term_integrated_order_flow_imbalance[i] = TkStatistics.depth_weighted_integrated_order_flow_imbalance( orderbooks, depth = 16, decay_rate = 0.25 ) # TODO: configure depth and decay_rate
            else:
                self.mid_term_integrated_order_flow_imbalance[i] = 0.0            
            assert not math.isnan(self.mid_term_integrated_order_flow_imbalance[i])

            orderbooks = PreprocessedData.select_prior_orderbooks( raw_samples, i, 16 )
            if len(orderbooks) > 1:
                self.long_term_integrated_order_flow_imbalance[i] = TkStatistics.depth_weighted_integrated_order_flow_imbalance( orderbooks, depth = 32, decay_rate = 0.5 ) # TODO: configure depth and decay_rate
            else:
                self.long_term_integrated_order_flow_imbalance[i] = 0.0
            assert not math.isnan(self.long_term_integrated_order_flow_imbalance[i])

        # self.cumulative_order_flow_imbalance = TkStatistics.rolling_sum( self.order_flow_imbalance, window=future_steps_count ).tolist()

        self.cost_of_buying_liquidity_10bps = [0] * raw_sample_count
        self.cost_of_buying_liquidity_100bps = [0] * raw_sample_count
        self.cost_of_buying_liquidity_500bps = [0] * raw_sample_count
        self.cost_of_selling_liquidity_10bps = [0] * raw_sample_count
        self.cost_of_selling_liquidity_100bps = [0] * raw_sample_count
        self.cost_of_selling_liquidity_500bps = [0] * raw_sample_count
        self.cost_of_liquidity_imbalance_10bps = [0] * raw_sample_count
        self.cost_of_liquidity_imbalance_100bps = [0] * raw_sample_count
        self.cost_of_liquidity_imbalance_500bps = [0] * raw_sample_count

        for i in range( raw_sample_count ):

            average_orderbook_volume_10bps = max( 1, int(0.001 * self.average_orderbook_volume[i]) )
            average_orderbook_volume_100bps = max( 1, int(0.01 * self.average_orderbook_volume[i]) )
            average_orderbook_volume_500bps = max( 1, int(0.05 * self.average_orderbook_volume[i]) )

            self.cost_of_buying_liquidity_10bps[i] = TkStatistics.compute_cost_of_liquidity_bps( raw_samples[i*2], average_orderbook_volume_10bps, "Buy" )
            self.cost_of_buying_liquidity_100bps[i] = TkStatistics.compute_cost_of_liquidity_bps( raw_samples[i*2], average_orderbook_volume_100bps, "Buy" )
            self.cost_of_buying_liquidity_500bps[i] = TkStatistics.compute_cost_of_liquidity_bps( raw_samples[i*2], average_orderbook_volume_500bps, "Buy" )

            self.cost_of_selling_liquidity_10bps[i] = TkStatistics.compute_cost_of_liquidity_bps( raw_samples[i*2], average_orderbook_volume_10bps, "Sell" )
            self.cost_of_selling_liquidity_100bps[i] = TkStatistics.compute_cost_of_liquidity_bps( raw_samples[i*2], average_orderbook_volume_100bps, "Sell" )
            self.cost_of_selling_liquidity_500bps[i] = TkStatistics.compute_cost_of_liquidity_bps( raw_samples[i*2], average_orderbook_volume_500bps, "Sell" )

            self.cost_of_liquidity_imbalance_10bps[i] = TkStatistics.compute_liquidity_imbalance( self.cost_of_buying_liquidity_10bps[i], self.cost_of_selling_liquidity_10bps[i] )
            self.cost_of_liquidity_imbalance_100bps[i] = TkStatistics.compute_liquidity_imbalance( self.cost_of_buying_liquidity_100bps[i], self.cost_of_selling_liquidity_100bps[i] )
            self.cost_of_liquidity_imbalance_500bps[i] = TkStatistics.compute_liquidity_imbalance( self.cost_of_buying_liquidity_500bps[i], self.cost_of_selling_liquidity_500bps[i] )

            assert not math.isnan(self.cost_of_liquidity_imbalance_10bps[i])
            assert not math.isnan(self.cost_of_liquidity_imbalance_100bps[i])
            assert not math.isnan(self.cost_of_liquidity_imbalance_500bps[i])

        self.average_orderbook_volume[0] = self.average_orderbook_volume[1] # TEST

    def get_interval_trades(self, start_ts:int, end_ts:int):
        return TkLastTrades( self.all_trades, start_ts, end_ts )

    def sample_width(self):
        return 25 # sizeof quant_sample
    
    def quant_sample(self, i:int, base_timestamp:float):
        sample = []

        # slice 1 : price, volatility and trend
        sample.append( self.price_change[i] )
        sample.append( self.price_change_rate[i] )
        sample.append( self.volatility[i] )
        sample.append( self.trends[i] )

        # slice 2 : liquidity and spread
        sample.append( self.spread[i] )
        sample.append( self.orderbook_volume[i] )

        # slice 3 : orderbook structure (shape + microprice)
        sample.append( self.orderbook_slope[i] )
        sample.append( self.orderbook_microprice[i] )
        sample.append( self.queue_imbalance[i] )

        # slice 4 : trade activity / trade flow intensity
        sample.append( self.last_trades_volume[i] )
        sample.append( self.last_trades_rate[i] )
        sample.append( self.last_trades_num_events[i] )
        sample.append( self.queue_depletion_intensity[i] )
        sample.append( self.order_arrival_intensity[i] )
        
        # slice 5 : flow imbalance
        sample.append( self.short_term_integrated_order_flow_imbalance[i] )
        sample.append( self.mid_term_integrated_order_flow_imbalance[i] )
        sample.append( self.long_term_integrated_order_flow_imbalance[i] )
        sample.append( self.cost_of_liquidity_imbalance_10bps[i] )
        sample.append( self.cost_of_liquidity_imbalance_100bps[i] )
        sample.append( self.cost_of_liquidity_imbalance_500bps[i] )
        sample.append( self.trade_flow_imbalance[i] )
        sample.append( self.queue_depletion_imbalance[i] )
        sample.append( self.order_arrival_imbalance[i] )

        # slice 7: time data
        age_seconds = max(0.0, base_timestamp - self.timestamp[i])
        sample.append( self.log_time_delta[i] )
        sample.append( age_seconds )

        assert not any(math.isnan(x) for x in sample)

        return sample

#------------------------------------------------------------------------------------------------------------------------
# Try aferage pool of the duplicated elements in the given raw samples
#------------------------------------------------------------------------------------------------------------------------

def try_average_pool_duplicates(raw_samples:list):

    raw_sample_count = int( len(raw_samples) / 2 ) # [ orderbook, last_trades, .... ]

    for i in range( raw_sample_count-1, -1, -1):

        curr_orderbook = raw_samples[i*2]
        curr_trades = raw_samples[i*2+1]
        curr_ts = curr_orderbook.orderbook_ts

        prev_orderbook = raw_samples[(i-1)*2]
        prev_trades = raw_samples[(i-1)*2+1]
        prev_ts = prev_orderbook.orderbook_ts

        time_delta = (curr_ts - prev_ts).total_seconds()

        if time_delta == 0:

            if prev_orderbook != curr_orderbook:
                prev_orderbook.average_pool( curr_orderbook )

            if prev_trades != curr_trades:
                prev_trades.union_pool( curr_trades )

            del raw_samples[i*2+1]
            del raw_samples[i*2]

    num_pooled_samples = raw_sample_count - int( len(raw_samples) / 2 )

    if num_pooled_samples > 0:
        print( 'Found & pooled', num_pooled_samples, 'duplicated samples' )

            

#------------------------------------------------------------------------------------------------------------------------
# Preprocessing for training, adopted for MP
#------------------------------------------------------------------------------------------------------------------------

def preprocess_file_for_training(output_queue, ticker:str, is_test_data_source:bool, filename:str):

    pid = os.getpid()

    TOKEN = os.environ["TK_TOKEN"]
    with Client(TOKEN, target=INVEST_GRPC_API) as client:

        config = configparser.ConfigParser()
        config.read( 'TkConfig.ini' )

        share = TkInstrument(client, config, InstrumentType.INSTRUMENT_TYPE_SHARE, ticker, "TQBR")

        data_path = config['Paths']['DataPath']
        last_trades_discretization = float(config['Autoencoders']['LastTradesDiscretization'])

        orderbook_width = int(config['Autoencoders']['OrderbookWidth'])
        orderbook_depth = int(config['Autoencoders']['OrderbookDepth'])
        last_trades_width = int(config['Autoencoders']['LastTradesWidth'])
        last_trades_depth = int(config['Autoencoders']['LastTradesDepth'])        

        test_data_ratio = float(config['TimeSeries']['TestDataRatio'])
        lshash_size = int(config['TimeSeries']['LSHashSize'])
        ts_sample_similatiry = float(config['TimeSeries']['TSSampleSimilatiry'])
        ts_data_stride = int(config['TimeSeries']['TSDataStride'])        
        trend_steps_count = int(config['TimeSeries']['TrendStepsCount'])
        trend_regimes = json.loads(config['TimeSeries']['TrendRegimes'])
        volatility_regimes = json.loads(config['TimeSeries']['VolatilityRegimes'])
        prior_steps_count = int(config['TimeSeries']['PriorStepsCount'])
        future_steps_count = int(config['TimeSeries']['FutureStepsCount'])
        future_interval = float(config['TimeSeries']['FutureInterval'])
        regime_ema_half_life = float(config['TimeSeries']['RegimeEMAHalfLife'])
        trend_ema_half_life = float(config['TimeSeries']['TrendEMAHalfLife'])
        priority_tail_threshold = float(config['TimeSeries']['PriorityTailThreshold'])

        raw_samples = TkIO.read_at_path( join( data_path, filename) )
        #try_average_pool_duplicates( raw_samples )

        raw_sample_count = int( len(raw_samples) / 2 ) # [ orderbook, last_trades, .... ]

        num_priority_samples = 0
        num_ordinary_samples = 0

        if raw_sample_count >= prior_steps_count + future_steps_count:

            data = PreprocessedData( share, raw_samples, orderbook_width, last_trades_width, last_trades_discretization, regime_ema_half_life, trend_ema_half_life, trend_steps_count, volatility_regimes, trend_regimes, future_steps_count, prior_steps_count )

            start_range = prior_steps_count - 1
            end_range = raw_sample_count - future_steps_count - 1
            range_len = end_range - start_range

            # encode future trades distribution

            future_trades = [None] * (raw_sample_count)
            future_trades_volume = [None] * (raw_sample_count)
            future_trades_tails = [None] * (raw_sample_count)

            for i in range( start_range, end_range+1 ):            
                ts_base_price = data.price[i]

                #prev_orderbook_sample = raw_samples[i*2]
                #last_trades_samples = [ (raw_samples[(i+1)*2+1], prev_orderbook_sample.orderbook_ts)]
                #for j in range( 2, future_steps_count+1 ):
                #    prev_orderbook_sample = raw_samples[(i+j-1)*2]
                #    last_trades_samples.append( ( raw_samples[(i+j)*2+1], prev_orderbook_sample.orderbook_ts ) )

                start_ts = int( raw_samples[i*2].orderbook_ts.timestamp() )
                end_ts = start_ts + int( future_interval )
                interval_last_trades_samples = [ ( data.get_interval_trades( start_ts, end_ts ), raw_samples[i*2].orderbook_ts) ]                

                for j in range( 1, future_steps_count+1 ):
                    assert interval_last_trades_samples[0][0].validate( raw_samples[(i+j)*2+1], start_ts, end_ts )

                future_last_trades_tensor, _, num_future_events, future_volume, future_buy_trades, future_sell_trades, future_trades_mean_tails = TkStatistics.last_trades_to_tensor( interval_last_trades_samples, ts_base_price, last_trades_width, last_trades_discretization, force_categorical=True )
            
                future_trades[i] = future_last_trades_tensor
                future_trades_volume[i] = future_volume
                future_trades_tails[i] = future_trades_mean_tails

            # combine data samples

            callback_indices = [start_range + int(i / 10.0 * range_len) for i in range(1,10)]

            hasheable_sample_width = data.sample_width() * prior_steps_count
            sample_lhs = LSHash(lshash_size, hasheable_sample_width) 
            step = 0

            for i in range( start_range, end_range+1 ):

                step = step + 1

                ts_regime = data.regimes[i+1] 
                ts_trend_regime = data.trend_regimes[i+1]
                ts_input = [None] * prior_steps_count
                ts_base_timestamp = data.timestamp[i]
                
                for j in range( prior_steps_count ):
                    k = i-prior_steps_count+j+1
                    ts_input[j] = data.quant_sample(k, ts_base_timestamp)
                                
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

                ts_is_priority_sample = 1.0 if is_priority_sample else 0.0 
                
                if (step-1) % ts_data_stride == 0 or is_priority_sample or is_test_data_source:
                    if is_priority_sample:
                        num_priority_samples = num_priority_samples + 1
                    else:
                        num_ordinary_samples = num_ordinary_samples + 1
                    output_queue.put( (pid, [ts_input, ts_target, ts_regime, ts_trend_regime, ts_is_priority_sample], is_priority_sample, is_test_data_source, False) )
                    time.sleep( 0.0 )
    
    output_queue.put( (pid, [num_priority_samples, num_ordinary_samples, raw_sample_count], False, is_test_data_source, True) )

    quit(0)

#------------------------------------------------------------------------------------------------------------------------
# Preprocessing for inference
#------------------------------------------------------------------------------------------------------------------------

def preprocess_file_for_inference(ticker:str, filename:str):

    TOKEN = os.environ["TK_TOKEN"]
    with Client(TOKEN, target=INVEST_GRPC_API) as client:

        config = configparser.ConfigParser()
        config.read( 'TkConfig.ini' )

        share = TkInstrument(client, config, InstrumentType.INSTRUMENT_TYPE_SHARE, ticker, "TQBR")

        data_path = config['Paths']['DataPath']
        last_trades_discretization = float(config['Autoencoders']['LastTradesDiscretization'])

        orderbook_width = int(config['Autoencoders']['OrderbookWidth'])
        orderbook_depth = int(config['Autoencoders']['OrderbookDepth'])
        last_trades_width = int(config['Autoencoders']['LastTradesWidth'])
        last_trades_depth = int(config['Autoencoders']['LastTradesDepth'])

        test_data_ratio = float(config['TimeSeries']['TestDataRatio'])
        lshash_size = int(config['TimeSeries']['LSHashSize'])
        ts_sample_similatiry = float(config['TimeSeries']['TSSampleSimilatiry'])
        ts_data_stride = int(config['TimeSeries']['TSDataStride'])
        trend_steps_count = int(config['TimeSeries']['TrendStepsCount'])
        trend_regimes = json.loads(config['TimeSeries']['TrendRegimes'])
        volatility_regimes = json.loads(config['TimeSeries']['VolatilityRegimes'])
        prior_steps_count = int(config['TimeSeries']['PriorStepsCount'])
        future_steps_count = int(config['TimeSeries']['FutureStepsCount'])
        future_interval = float(config['TimeSeries']['FutureInterval'])
        regime_ema_half_life = float(config['TimeSeries']['RegimeEMAHalfLife'])
        trend_ema_half_life = float(config['TimeSeries']['TrendEMAHalfLife'])
        priority_tail_threshold = float(config['TimeSeries']['PriorityTailThreshold'])

        raw_samples = TkIO.read_at_path( join( data_path, filename) )
        #try_average_pool_duplicates( raw_samples )

        raw_sample_count = int( len(raw_samples) / 2 ) # [ orderbook, last_trades, .... ]

        if raw_sample_count >= prior_steps_count:

            data = PreprocessedData( share, raw_samples, orderbook_width, last_trades_width, last_trades_discretization, regime_ema_half_life, trend_ema_half_life, trend_steps_count, volatility_regimes, trend_regimes, future_steps_count, prior_steps_count )

            price = data.price[-prior_steps_count:]
            orderbook_volume = data.orderbook_volume[-prior_steps_count:]
            trades_volume = data.last_trades_volume[-prior_steps_count:]
            
            ts_input = [None] * prior_steps_count
            ts_base_timestamp = data.timestamp[raw_sample_count-1]
                
            for j in range( prior_steps_count ):
                k = raw_sample_count-prior_steps_count+j
                ts_input[j] = data.quant_sample(k, ts_base_timestamp)
                                
            ts_input = list( itertools.chain.from_iterable(ts_input) )

            last_trades, last_trades_descriptor, last_trades_volume = TkStatistics.trades_distribution( raw_samples[-1], data.price[-1], last_trades_width, last_trades_discretization )
                                
            return ts_input, data.price[-1], data.min_price_increment, last_trades, last_trades_descriptor, price, orderbook_volume, trades_volume
        
        return None, None, None, None, None, None, None, None