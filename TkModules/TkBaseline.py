import os
import pickle
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
from TkModules.TkInstrument import TkInstrument, TkOrderbook, TkLastTrades
from TkModules.TkStatistics import TkStatistics
from TkModules.TkUI import TkUI
from TkModules.TkPreprocessFile import PreprocessedData
from dataclasses import dataclass, field
from collections import defaultdict
from typing import List

#------------------------------------------------------------------------------------------------------------------------
# Baseline features 
# * samples discretized and average out over hours interval
# * EMA normalization applied per feature / per instrument
#------------------------------------------------------------------------------------------------------------------------

@dataclass
class TkBaseline:

    _resting_volume_depth:int
    _discretization_in_hours:int
    _ema_alpha:float

    _average_daily_volume: dict[datetime, dict[str, tuple[int, float]]] = field(default_factory=dict)
    _average_resting_volume: dict[datetime, dict[str, tuple[int, float]]] = field(default_factory=dict)
    _spread_to_tick_ratio: dict[datetime, dict[str, tuple[int, float]]] = field(default_factory=dict)    
    _relative_tick_size: dict[datetime, dict[str, tuple[int, float]]] = field(default_factory=dict)
    _minmax_price: dict[datetime, dict[str, tuple[float, float]]] = field(default_factory=dict)
    _volume_traded: dict[datetime, dict[str, tuple[int, float]]] = field(default_factory=dict)
    _amihud_illiquidity: dict[datetime, dict[str, float]] = field(default_factory=dict)
    _interarrival_time: dict[datetime, dict[str, tuple[int, float]]] = field(default_factory=dict)
    _average_trade_size: dict[datetime, dict[str, tuple[int, float]]] = field(default_factory=dict)
    _volume_bought: dict[datetime, dict[str, tuple[int, float]]] = field(default_factory=dict)
    _volume_sold: dict[datetime, dict[str, tuple[int, float]]] = field(default_factory=dict)
    _order_flow_toxicity: dict[datetime, dict[str, float]] = field(default_factory=dict)
    _sum_of_squared_log_returns: dict[datetime, dict[str, float]] = field(default_factory=dict)
    _realized_volatility: dict[datetime, dict[str, float]] = field(default_factory=dict)

    _average_daily_volume_ema_norm: dict[datetime, dict[str, tuple[int, float]]] = field(default_factory=dict)
    _average_resting_volume_ema_norm: dict[datetime, dict[str, tuple[int, float]]] = field(default_factory=dict)
    _spread_to_tick_ratio_ema_norm: dict[datetime, dict[str, tuple[int, float]]] = field(default_factory=dict)    
    _relative_tick_size_ema_norm: dict[datetime, dict[str, tuple[int, float]]] = field(default_factory=dict)
    _amihud_illiquidity_ema_norm: dict[datetime, dict[str, float]] = field(default_factory=dict)
    _interarrival_time_ema_norm: dict[datetime, dict[str, tuple[int, float]]] = field(default_factory=dict)
    _average_trade_size_ema_norm: dict[datetime, dict[str, tuple[int, float]]] = field(default_factory=dict)
    _order_flow_toxicity_ema_norm: dict[datetime, dict[str, float]] = field(default_factory=dict)
    _realized_volatility_ema_norm: dict[datetime, dict[str, float]] = field(default_factory=dict)
    
    def floor_dt(self, dt: datetime) -> datetime:
        # floor to nearest discretization interval
        hour_remainder = dt.hour % self._discretization_in_hours
        
        # Replace the components to floor the time
        return dt.replace( hour=dt.hour - hour_remainder, minute=0, second=0, microsecond=0 )

    def ceil_dt(self, dt: datetime) -> datetime:

        # Find how many hours we are past the start of the 2-hour block (0 or 1)
        hour_remainder = dt.hour % self._discretization_in_hours
    
        # Check if we are exactly on the specific boundary
        if hour_remainder == 0 and dt.minute == 0 and dt.second == 0 and dt.microsecond == 0:
            return dt
        
        # Calculate how many hours to add to reach the next 2-hour mark
        hours_to_add = self._discretization_in_hours - hour_remainder
    
        # Floor the time to the current hour, then add the remaining hours
        floored_hour = dt.replace(minute=0, second=0, microsecond=0)
        return floored_hour + timedelta(hours=hours_to_add)

    def set_value(self, container: dict[datetime, dict[str, float]], dt: datetime, ticker: str, val: float ) -> float:

        dt = self.ceil_dt(dt)

        sub_dict = container.setdefault(dt, {})
        sub_dict.setdefault(ticker, 0.0)
        sub_dict[ticker] = val

        return sub_dict[ticker]

    def accumulate_value(self, container: dict[datetime, dict[str, float]], dt: datetime, ticker: str, val: float ) -> float:

        dt = self.ceil_dt(dt)

        # Ensure the sub-dictionary for the datetime key exists
        sub_dict = container.setdefault(dt, {})
                    
        if ticker in sub_dict:
            sub_dict[ticker] = sub_dict[ticker] + val
        else:
            sub_dict[ticker] = val
        
        return sub_dict[ticker]

    def add_weighted_value(self, container: dict[datetime, dict[str, tuple[int, float]]], dt: datetime, ticker: str, weight:int, val: float ) -> tuple[int, float]:

        dt = self.ceil_dt(dt)
            
        # Ensure the sub-dictionary for the datetime key exists
        sub_dict = container.setdefault(dt, {})
            
        if ticker in sub_dict:
            current_weight, current_average = sub_dict[ticker]
            next_weight = current_weight + weight
            next_average = (current_average * current_weight + val * weight) / next_weight if next_weight > 0 else 0
            sub_dict[ticker] = (next_weight, next_average)
        else:
            sub_dict[ticker] = (weight, val)

        return sub_dict[ticker]

    def add_minmax_value(self, container: dict[datetime, dict[str, tuple[float, float]]], dt: datetime, ticker: str, minval:float, maxval: float ) -> None:
    
        dt = self.ceil_dt(dt)
                
        # Ensure the sub-dictionary for the datetime key exists
        sub_dict = container.setdefault(dt, {})
                
        if ticker in sub_dict:
            current_min, current_max = sub_dict[ticker]
            sub_dict[ticker] = ( min(current_min,minval), max(current_max,maxval) )
        else:
            sub_dict[ticker] = (minval, maxval)

        return sub_dict[ticker]

    def __init__(self, discretization_in_hours:int, resting_volume_depth:int, per_feature_alpha : dict[str, float]):

        self._discretization_in_hours = discretization_in_hours
        self._resting_volume_depth = resting_volume_depth
        self._per_feature_alpha = per_feature_alpha

        self._average_daily_volume = {}
        self._average_resting_volume = {}
        self._spread_to_tick_ratio = {}
        self._relative_tick_size = {}
        self._minmax_price = {}
        self._volume_traded = {}
        self._interarrival_time = {}
        self._average_trade_size = {}
        self._amihud_illiquidity = {}
        self._volume_bought = {}
        self._volume_sold = {}
        self._order_flow_toxicity = {}
        self._sum_of_squared_log_returns = {}
        self._realized_volatility = {}

        self._average_daily_volume_ema_norm = {}
        self._average_resting_volume_ema_norm = {}
        self._spread_to_tick_ratio_ema_norm = {}
        self._relative_tick_size_ema_norm = {}
        self._amihud_illiquidity_ema_norm = {}
        self._interarrival_time_ema_norm = {}
        self._average_trade_size_ema_norm = {}
        self._order_flow_toxicity_ema_norm = {}
        self._realized_volatility_ema_norm = {}

    def add_samples(self, share:TkInstrument, raw_samples:list, apply_incremental_ema_normalization:bool):

        share_ticker = share.ticker()

        raw_sample_count = int( len(raw_samples) / 2 ) # [ orderbook, last_trades, .... ]

        # discretisation may vary (the exchange might occasionally change it per instrument)
        
        min_price_increment = TkStatistics.get_min_price_increment(raw_samples[0], quotation_to_decimal(share.min_price_increment()))
        for i in range(raw_sample_count):
            min_price_increment = min( min_price_increment, TkStatistics.get_min_price_increment( raw_samples[i*2], quotation_to_decimal(share.min_price_increment()) ) )
        min_price_increment = float( min_price_increment )

        # datetime correction

        PreprocessedData.fix_corrupted_datetimes(raw_samples)

        # sampling

        last_trades_threshold_ts = None

        for i in range( raw_sample_count ):

            orderbook_sample = raw_samples[i*2]
            last_trades_sample = raw_samples[i*2+1]

            total_volume = 0
            resting_volume = 0
            for j in range(len(orderbook_sample.bids)):
                bid_volume = orderbook_sample.bids[j][1]
                total_volume = total_volume + bid_volume
                resting_volume = resting_volume + bid_volume if j < self._resting_volume_depth else resting_volume
            for j in range(len(orderbook_sample.asks)):
                ask_volume = orderbook_sample.asks[j][1]
                total_volume = total_volume + ask_volume
                resting_volume = resting_volume + ask_volume if j < self._resting_volume_depth else resting_volume
                    
            log_total_volume = math.log( total_volume + 1.0 )
            self.add_weighted_value(self._average_daily_volume, orderbook_sample.orderbook_ts, share_ticker, 1, log_total_volume )
                    
            log_resting_volume = math.log( resting_volume + 1.0 )
            self.add_weighted_value(self._average_resting_volume, orderbook_sample.orderbook_ts, share_ticker, 1, log_resting_volume )
                    
            if orderbook_sample.asks and orderbook_sample.bids:
                best_ask = orderbook_sample.asks[0][0]
                best_bid = orderbook_sample.bids[0][0]
                spread = best_ask - best_bid
                spread_to_tick_ratio = float(spread) / min_price_increment
                self.add_weighted_value(self._spread_to_tick_ratio, orderbook_sample.orderbook_ts, share_ticker, 1, spread_to_tick_ratio )
                midprice = float( (best_ask + best_bid) / 2 )
                self.add_weighted_value(self._relative_tick_size, orderbook_sample.orderbook_ts, share_ticker, 1, min_price_increment / midprice )
                    
            for trade_price, trade_quantity, trade_ts, trade_dir in last_trades_sample.trades:
                if last_trades_threshold_ts != None and trade_ts < last_trades_threshold_ts:
                    continue
                trade_price = float(trade_price)
                minmax_price = self.add_minmax_value(self._minmax_price, trade_ts, share_ticker, trade_price, trade_price )
                volume_traded = self.add_weighted_value(self._volume_traded, trade_ts, share_ticker, trade_quantity, trade_price)
                cumulative_volume = volume_traded[1] * volume_traded[0]
                amihud_illiquidity = (minmax_price[1] - minmax_price[0]) / minmax_price[0] / cumulative_volume if cumulative_volume > 0 else 0
                self.set_value(self._amihud_illiquidity, trade_ts, share_ticker, amihud_illiquidity )
                self.add_weighted_value(self._average_trade_size, trade_ts, share_ticker, 1, float(trade_quantity))
                    
                quantity_bought = trade_quantity if trade_dir > 0 else 0
                price_bought = trade_price if trade_dir > 0 else 0
                volume_bought = self.add_weighted_value(self._volume_bought, trade_ts, share_ticker, quantity_bought, price_bought )
                    
                quantity_sold = trade_quantity if trade_dir < 0 else 0
                price_sold = trade_price if trade_dir < 0 else 0
                volume_sold = self.add_weighted_value(self._volume_sold, trade_ts, share_ticker, quantity_sold, price_sold )
                    
                order_flow_toxicity = (volume_bought[1] * volume_bought[0]) / (volume_bought[1] * volume_bought[0] + volume_sold[1] * volume_sold[0]) if (volume_bought[1] * volume_bought[0] + volume_sold[1] * volume_sold[0]) > 0 else 0
                self.set_value(self._order_flow_toxicity, trade_ts, share_ticker, order_flow_toxicity )
                                                    
            for i in range(len(last_trades_sample.trades)-1):
                if last_trades_threshold_ts != None and last_trades_sample.trades[i][2] < last_trades_threshold_ts:
                    continue
                ts_curr = last_trades_sample.trades[i][2]
                ts_next = last_trades_sample.trades[i+1][2]
                delta_t = (ts_next - ts_curr).total_seconds()
                self.add_weighted_value(self._interarrival_time, ts_next, share_ticker, 1, delta_t )
                price_curr = float(last_trades_sample.trades[i][0])
                price_next = float(last_trades_sample.trades[i+1][0])
                log_returns_next = math.log(price_next / price_curr) if price_curr > 0 else 0
                sum_of_squared_log_returns = self.accumulate_value(self._sum_of_squared_log_returns, ts_next, share_ticker, log_returns_next * log_returns_next )
                self.set_value(self._realized_volatility, ts_next, share_ticker, math.sqrt(sum_of_squared_log_returns) )

            last_trades_threshold_ts = orderbook_sample.orderbook_ts

        if apply_incremental_ema_normalization:
            self.apply_incremental_ema_normalization_to_container( self._average_daily_volume, self._average_daily_volume_ema_norm, self._per_feature_alpha['average_daily_volume'], True)
            self.apply_incremental_ema_normalization_to_container( self._average_resting_volume, self._average_resting_volume_ema_norm, self._per_feature_alpha['average_resting_volume'], True)
            self.apply_incremental_ema_normalization_to_container( self._spread_to_tick_ratio, self._spread_to_tick_ratio_ema_norm, self._per_feature_alpha['spread_to_tick_ratio'], True)
            self.apply_incremental_ema_normalization_to_container( self._relative_tick_size, self._relative_tick_size_ema_norm, self._per_feature_alpha['relative_tick_size'], True)
            self.apply_incremental_ema_normalization_to_container( self._amihud_illiquidity, self._amihud_illiquidity_ema_norm, self._per_feature_alpha['amihud_illiquidity'], False)
            self.apply_incremental_ema_normalization_to_container( self._interarrival_time, self._interarrival_time_ema_norm, self._per_feature_alpha['interarrival_time'], True)
            self.apply_incremental_ema_normalization_to_container( self._average_trade_size, self._average_trade_size_ema_norm, self._per_feature_alpha['average_trade_size'], True)
            self.apply_incremental_ema_normalization_to_container( self._order_flow_toxicity, self._order_flow_toxicity_ema_norm, self._per_feature_alpha['order_flow_toxicity'], False)
            self.apply_incremental_ema_normalization_to_container( self._realized_volatility, self._realized_volatility_ema_norm, self._per_feature_alpha['realized_volatility'], False)

        return

    # Calculates EMA and EMA-variance to output time-agnostic Normalized Z-scores.
    # Can be executed repeatedly as new data arrives in the input_container.
    def apply_ema_normalization_to_container(
        self, 
        input_container: dict[datetime, dict[str, any]], 
        output_container: dict[datetime, dict[str, any]], 
        alpha:float,
        is_tuple: bool
    ) -> None:

        if not input_container:
            return

        # Identify all unique tickers across all time buckets in the raw data
        tickers = set()
        for dt_dict in input_container.values():
            tickers.update(dt_dict.keys())
        
        # Sort the datetime keys chronologically
        sorted_dts = sorted(input_container.keys())
        
        # Compute EMA and EMA-Variance per ticker
        for ticker in tickers:
            previous_ema_val = None
            previous_ema_var = None
            previous_ema_weight = None
            previous_ema_num = None  
            
            for dt in sorted_dts:
                if ticker not in input_container[dt]:
                    continue
                
                # Extract current raw value
                if is_tuple:
                    current_weight, current_val = input_container[dt][ticker]
                    current_num = current_weight * current_val
                else:
                    current_val = input_container[dt][ticker]
                
                if previous_ema_val is None:
                    # Initialization
                    ema_val = current_val
                    ema_var = 0.0
                    normalized_val = 0.0  # Z-score is 0 for the first point
                    
                    if is_tuple:
                        ema_weight = current_weight
                        ema_num = current_num
                else:
                    # Constant alpha scaling (treats intervals equally)
                    effective_alpha = alpha
                    
                    # Compute EMA Mean
                    if is_tuple:
                        ema_weight = effective_alpha * current_weight + (1.0 - effective_alpha) * previous_ema_weight
                        ema_num = effective_alpha * current_num + (1.0 - effective_alpha) * previous_ema_num
                        ema_val = ema_num / ema_weight if ema_weight > 0 else 0
                    else:
                        ema_val = effective_alpha * current_val + (1.0 - effective_alpha) * previous_ema_val
                    
                    # Compute EMA Variance (Welford's exponential method for numerical stability)
                    diff = current_val - previous_ema_val
                    ema_var = (1.0 - effective_alpha) * (previous_ema_var + effective_alpha * (diff ** 2))
                    
                    # Compute Z-Score Normalization
                    ema_std = math.sqrt(ema_var)
                    
                    # Add a small epsilon (1e-8) to prevent division by zero in zero-variance periods
                    normalized_val = (current_val - ema_val) / (ema_std + 1e-8)
                
                # Write to the output container
                sub_dict = output_container.setdefault(dt, {})
                
                if is_tuple:
                    # Retains the smoothed weight, but replaces the value with the Normalized Z-score
                    sub_dict[ticker] = (ema_weight, normalized_val)
                    previous_ema_weight = ema_weight
                    previous_ema_num = ema_num
                else:
                    sub_dict[ticker] = normalized_val
                
                previous_ema_val = ema_val
                previous_ema_var = ema_var


    def apply_incremental_ema_normalization_to_container(
        self, 
        input_container: dict[datetime, dict[str, any]], 
        output_container: dict[datetime, dict[str, any]], 
        alpha:float,
        is_tuple: bool
    ) -> None:

        if not input_container:
            return

        # Identify all unique tickers across all time buckets in the raw data
        tickers = set()
        for dt_dict in input_container.values():
            tickers.update(dt_dict.keys())
        
        # Sort the datetime keys chronologically
        sorted_dts = sorted(input_container.keys())
        
        # Compute EMA and EMA-Variance per ticker
        for ticker in tickers:
            
            # Find the most recent synchronization point for this ticker in the output container
            last_out_dt = None
            for dt in reversed(sorted_dts):
                if dt in output_container and ticker in output_container[dt]:
                    last_out_dt = dt
                    break
            
            previous_ema_val = None
            previous_ema_var = None
            previous_ema_weight = None
            previous_ema_num = None  
            
            # We must iterate from the beginning to maintain correct EMA state, 
            # but we only write to the output container for the last synchronized bucket and any new missing buckets.
            for dt in sorted_dts:
                if ticker not in input_container[dt]:
                    continue
                
                # Extract current raw value
                if is_tuple:
                    current_weight, current_val = input_container[dt][ticker]
                    current_num = current_weight * current_val
                else:
                    current_val = input_container[dt][ticker]
                
                if previous_ema_val is None:
                    # Initialization
                    ema_val = current_val
                    ema_var = 0.0
                    normalized_val = 0.0  # Z-score is 0 for the first point
                    
                    if is_tuple:
                        ema_weight = current_weight
                        ema_num = current_num
                else:
                    # Constant alpha scaling (treats intervals equally)
                    effective_alpha = alpha
                    
                    # Compute EMA Mean
                    if is_tuple:
                        ema_weight = effective_alpha * current_weight + (1.0 - effective_alpha) * previous_ema_weight
                        ema_num = effective_alpha * current_num + (1.0 - effective_alpha) * previous_ema_num
                        ema_val = ema_num / ema_weight if ema_weight > 0 else 0
                    else:
                        ema_val = effective_alpha * current_val + (1.0 - effective_alpha) * previous_ema_val
                    
                    # Compute EMA Variance (Welford's exponential method for numerical stability)
                    diff = current_val - previous_ema_val
                    ema_var = (1.0 - effective_alpha) * (previous_ema_var + effective_alpha * (diff ** 2))
                    
                    # Compute Z-Score Normalization
                    ema_std = math.sqrt(ema_var)
                    
                    # Add a small epsilon (1e-8) to prevent division by zero in zero-variance periods
                    normalized_val = (current_val - ema_val) / (ema_std + 1e-8)
                
                # INCREMENTAL LOGIC: 
                # Only write to the output container if it's the last synchronized bucket (update it) 
                # or if it's a newer bucket missing from the output container.
                if last_out_dt is None or dt >= last_out_dt:
                    sub_dict = output_container.setdefault(dt, {})
                    
                    if is_tuple:
                        # Retains the smoothed weight, but replaces the value with the Normalized Z-score
                        sub_dict[ticker] = (ema_weight, normalized_val)
                    else:
                        sub_dict[ticker] = normalized_val
                
                # Track state for the next iteration
                if is_tuple:
                    previous_ema_weight = ema_weight
                    previous_ema_num = ema_num
                previous_ema_val = ema_val
                previous_ema_var = ema_var    

    def apply_ema_normalization_across_periods(self) -> None:
        self.apply_ema_normalization_to_container( self._average_daily_volume, self._average_daily_volume_ema_norm, self._per_feature_alpha['average_daily_volume'], True)
        self.apply_ema_normalization_to_container( self._average_resting_volume, self._average_resting_volume_ema_norm, self._per_feature_alpha['average_resting_volume'], True)
        self.apply_ema_normalization_to_container( self._spread_to_tick_ratio, self._spread_to_tick_ratio_ema_norm, self._per_feature_alpha['spread_to_tick_ratio'], True)
        self.apply_ema_normalization_to_container( self._relative_tick_size, self._relative_tick_size_ema_norm, self._per_feature_alpha['relative_tick_size'], True)
        self.apply_ema_normalization_to_container( self._amihud_illiquidity, self._amihud_illiquidity_ema_norm, self._per_feature_alpha['amihud_illiquidity'], False)
        self.apply_ema_normalization_to_container( self._interarrival_time, self._interarrival_time_ema_norm, self._per_feature_alpha['interarrival_time'], True)
        self.apply_ema_normalization_to_container( self._average_trade_size, self._average_trade_size_ema_norm, self._per_feature_alpha['average_trade_size'], True)
        self.apply_ema_normalization_to_container( self._order_flow_toxicity, self._order_flow_toxicity_ema_norm, self._per_feature_alpha['order_flow_toxicity'], False)
        self.apply_ema_normalization_to_container( self._realized_volatility, self._realized_volatility_ema_norm, self._per_feature_alpha['realized_volatility'], False)        
        return

    def save(self, path:str) -> None:
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> 'TkBaseline':
        with open(path, 'rb') as f:
            return pickle.load(f)

    def sample_baseline_feature(self, container, dt:datetime, ticker:str, is_tuple:bool):
        dt = self.floor_dt(dt)
        if dt in container:
            if ticker in container[dt]:
                return container[dt][ticker][1] if is_tuple else container[dt][ticker]
        dt = self.ceil_dt(dt)
        if dt in container:
            if ticker in container[dt]:
                return container[dt][ticker][1] if is_tuple else container[dt][ticker]
        return 0.0

    def sample_baseline(self, ticker:str, dt:datetime):

        average_daily_volume = self.sample_baseline_feature( self._average_daily_volume_ema_norm, dt, ticker, True)
        average_resting_volume = self.sample_baseline_feature( self._average_resting_volume_ema_norm, dt, ticker, True )
        spread_to_tick_ratio = self.sample_baseline_feature( self._spread_to_tick_ratio_ema_norm, dt, ticker, True )
        relative_tick_size = self.sample_baseline_feature( self._relative_tick_size_ema_norm, dt, ticker, True )
        amihud_illiquidity = self.sample_baseline_feature( self._amihud_illiquidity_ema_norm, dt, ticker, False )
        interarrival_time = self.sample_baseline_feature( self._interarrival_time_ema_norm, dt, ticker, True )
        average_trade_size = self.sample_baseline_feature( self._average_trade_size_ema_norm, dt, ticker, True )
        order_flow_toxicity = self.sample_baseline_feature( self._order_flow_toxicity_ema_norm, dt, ticker, False )
        realized_volatility = self.sample_baseline_feature( self._realized_volatility_ema_norm, dt, ticker, False )
        
        return [
            average_daily_volume,
            average_resting_volume,
            spread_to_tick_ratio,
            relative_tick_size,
            amihud_illiquidity,
            interarrival_time,
            average_trade_size,
            order_flow_toxicity,
            realized_volatility
        ]

    def extract_baseline(self, feature_name:str, ticker:str, max_length):
        if feature_name.lower() == 'average_daily_volume':
            return self.extract_baseline_from_containers( ticker, max_length, self._average_daily_volume, self._average_daily_volume_ema_norm, True )
        elif feature_name.lower() == 'average_resting_volume':
            return self.extract_baseline_from_containers( ticker, max_length, self._average_resting_volume, self._average_resting_volume_ema_norm, True )
        elif feature_name.lower() == 'spread_to_tick_ratio':
            return self.extract_baseline_from_containers( ticker, max_length, self._spread_to_tick_ratio, self._spread_to_tick_ratio_ema_norm, True )
        elif feature_name.lower() == 'relative_tick_size':
            return self.extract_baseline_from_containers( ticker, max_length, self._relative_tick_size, self._relative_tick_size_ema_norm, True )
        elif feature_name.lower() == 'amihud_illiquidity':
            return self.extract_baseline_from_containers( ticker, max_length, self._amihud_illiquidity, self._amihud_illiquidity_ema_norm, False )
        elif feature_name.lower() == 'interarrival_time':
            return self.extract_baseline_from_containers( ticker, max_length, self._interarrival_time, self._interarrival_time_ema_norm, True )
        elif feature_name.lower() == 'average_trade_size':
            return self.extract_baseline_from_containers( ticker, max_length, self._average_trade_size, self._average_trade_size_ema_norm, True )
        elif feature_name.lower() == 'order_flow_toxicity':
            return self.extract_baseline_from_containers( ticker, max_length, self._order_flow_toxicity, self._order_flow_toxicity_ema_norm, False )
        elif feature_name.lower() == 'realized_volatility':
            return self.extract_baseline_from_containers( ticker, max_length, self._realized_volatility, self._realized_volatility_ema_norm, False )
        else:
            raise ValueError(f"Unknown feature={feature_name}")

    def extract_baseline_from_containers(self, ticker, max_length, raw_dict, ema_dict, is_tuple ):
        raw = []        

        all_dt_keys = sorted(raw_dict.keys())
        ticker_dt_keys = []
        for dt in all_dt_keys:
            if ticker in raw_dict[dt] :
                ticker_dt_keys.append(dt)

        if len(ticker_dt_keys) > max_length:
            ticker_dt_keys = ticker_dt_keys[-max_length:]

        for dt in ticker_dt_keys:
            raw.append( raw_dict[dt][ticker][1] if is_tuple else raw_dict[dt][ticker] )

        raw_time = []
        
        dt0 = None
        
        for dt in ticker_dt_keys:
            ts = (dt - dt0).total_seconds() if dt0 != None else 0
            ts = ts / (3600 * self._discretization_in_hours)
            raw_time.append( ts )
            dt0 = dt if dt0 == None else dt0

        ema = []

        all_dt_keys = sorted(ema_dict.keys())
        ticker_dt_keys = []
        for dt in all_dt_keys:
            if ticker in ema_dict[dt]:
                ticker_dt_keys.append(dt)            

        if len(ticker_dt_keys) > max_length:
            ticker_dt_keys = ticker_dt_keys[-max_length:]

        for dt in ticker_dt_keys:
            ema.append( ema_dict[dt][ticker][1] if is_tuple else ema_dict[dt][ticker] )

        ema_time = []

        dt0 = None

        for dt in ticker_dt_keys:
            ts = (dt - dt0).total_seconds() if dt0 != None else 0
            ts = ts / (3600 * self._discretization_in_hours)
            ema_time.append( ts )
            dt0 = dt if dt0 == None else dt0

        return raw, ema, raw_time, ema_time