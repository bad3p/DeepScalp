
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
from joblib import Parallel, delayed
from tinkoff.invest.constants import INVEST_GRPC_API
from tinkoff.invest import Client
from tinkoff.invest import InstrumentType
from tinkoff.invest import InstrumentIdType
from tinkoff.invest import SecurityTradingStatus
from tinkoff.invest import GetOrderBookResponse, GetLastTradesResponse
from tinkoff.invest import HistoricCandle
from tinkoff.invest.exceptions import RequestError
from TkModules.TkQuotation import quotation_to_float
from TkModules.TkIO import TkIO
from TkModules.TkInstrument import TkInstrument
from TkModules.TkStatistics import TkStatistics
from TkModules.TkUI import TkUI

TOKEN = os.environ["TK_TOKEN"]

config = configparser.ConfigParser()
config.read( 'TkConfig.ini' )

data_path = config['Paths']['DataPath']
data_extension = config['Paths']['OrderbookFileExtension']

orderbook_width = int(config['Autoencoders']['OrderbookWidth'])
last_trades_width = int(config['Autoencoders']['LastTradesWidth'])
min_price_increment_factor = int(config['Autoencoders']['MinPriceIncrementFactor'])

data_files = [fileName for fileName in listdir(data_path) if (data_extension in fileName) and isfile(join(data_path, fileName))]
random.shuffle(data_files)

print( 'Data files found:', len(data_files) )

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
            with dpg.plot(label="Orderbook", width=512, height=256):
                dpg.add_plot_legend()
                dpg.add_plot_axis(dpg.mvXAxis, tag="x_axis_orderbook" )
                dpg.add_plot_axis(dpg.mvYAxis, tag="y_axis_orderbook" )
                dpg.add_line_series( [j for j in range(0, 32)], [random.random() for j in range(0, 32)], label="Orderbook", parent="x_axis_orderbook", tag="orderbook_series" )
            with dpg.plot(label="Last trades", width=512, height=256):
                dpg.add_plot_legend()
                dpg.add_plot_axis(dpg.mvXAxis, tag="x_axis_last_trades" )
                dpg.add_plot_axis(dpg.mvYAxis, tag="y_axis_last_trades" )
                dpg.add_line_series( [j for j in range(0, 32)], [random.random() for j in range(0, 32)], label="Last trades", parent="x_axis_last_trades", tag="last_trades_series" )
        with dpg.group(horizontal=True):
            with dpg.plot(label="Price history", width=1024, height=256):
                dpg.add_plot_legend()
                dpg.add_plot_axis(dpg.mvXAxis, tag="x_axis_price" )
                dpg.add_plot_axis(dpg.mvYAxis, tag="y_axis_price" )
                dpg.add_line_series( [j for j in range(0, 1)], [random.random() for j in range(0, 1)], label="Price history", parent="x_axis_price", tag="price_series" )

    dpg.show_viewport()
    dpg.set_primary_window("primary_window", True)

    files_processed = 0
    for filename in data_files:
        dpg.set_value("filename", filename)

        ticker = filename[ 0: filename.find("_") ]
        share = TkInstrument(client, config,  InstrumentType.INSTRUMENT_TYPE_SHARE, ticker, "TQBR")
        raw_samples = TkIO.read_at_path( join( data_path, filename) )

        prev_orderbook_ts = None
        cumulative_delta_ts = 0
        min_delta_ts = 99999

        price_history = []

        for i in range(int(len(raw_samples) / 2)):

            orderbook_sample = raw_samples[i*2]

            if prev_orderbook_ts != None:
                delta_ts = (orderbook_sample.orderbook_ts - prev_orderbook_ts).total_seconds()
                if delta_ts > 0:
                    cumulative_delta_ts = cumulative_delta_ts + delta_ts
                    min_delta_ts = min( min_delta_ts, delta_ts )
            
            distribution, descriptor, volume, pivot_price = TkStatistics.orderbook_distribution( orderbook_sample, orderbook_width , quotation_to_float(share.min_price_increment()) * min_price_increment_factor )
            labels = [ math.copysign( max(abs(i[0]),abs(i[1])), 0.5*(i[0]+i[1])) for i in descriptor]
            TkUI.set_series_with_labels("x_axis_orderbook","y_axis_orderbook","orderbook_series", distribution.tolist(), labels)

            last_trades_sample = raw_samples[i*2+1]
            distribution, descriptor, volume = TkStatistics.trades_distribution( last_trades_sample, pivot_price, last_trades_width, quotation_to_float(share.min_price_increment()) * min_price_increment_factor )
            labels = [ 0.5 * (i[0] + i[1]) for i in descriptor]
            TkUI.set_series_with_labels("x_axis_last_trades","y_axis_last_trades","last_trades_series", distribution.tolist(), labels)

            prev_orderbook_ts = orderbook_sample.orderbook_ts

            price_history.append( quotation_to_float(orderbook_sample.last_price) )
            TkUI.set_series("x_axis_price","y_axis_price","price_series", price_history)

            timeout = 0.25
            while timeout > 0:
                dpg.render_dearpygui_frame()
                time.sleep( 1.0 / 30.0 )
                timeout -= 0.1

            if not dpg.is_dearpygui_running():
                break

        if not dpg.is_dearpygui_running():
            break

        avg_delta_ts = cumulative_delta_ts / int(len(raw_samples) / 2)
        print('Average interval:', avg_delta_ts, 'Minimal interval:', min_delta_ts)


    while dpg.is_dearpygui_running():
        dpg.render_dearpygui_frame()

    dpg.destroy_context()