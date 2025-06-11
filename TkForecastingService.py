
import os
import gc
import time
import numpy as np
import configparser
import json
import copy
import random
import math
import torch
import uuid
from os import listdir
from os.path import isfile, join
from datetime import date, datetime, timezone
from dateutil import parser
from timeit import default_timer
from win10toast import ToastNotifier
import dearpygui.dearpygui as dpg
import itertools
import threading
import multiprocessing
import multiprocessing.connection as mpc
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
from TkModules.TkTrainingHistory import TkTimeSeriesTrainingHistory
from TkModules.TkModel import TkModel
from TkModules.TkStackedLSTM import TkStackedLSTM
from TkModules.TkLastTradesAutoencoder import TkLastTradesAutoencoder
from TkModules.TkOrderbookAutoencoder import TkOrderbookAutoencoder
from TkModules.TkTimeSeriesForecaster import TkTimeSeriesForecaster

#------------------------------------------------------------------------------------------------------------------------
# UI
#------------------------------------------------------------------------------------------------------------------------

class TkMainPanel():

    def __init__(self, windowTag:str, numSamples:int):
        self._tag = uuid.uuid1()
        self._windowTag = windowTag
        self._numSamples = numSamples
                
        plotColor0 = (31,63,255)
        plotColor1 = (255,63,31)
        self._plotThemeTags = []
        for i in range(self._numSamples):                    
            plotThemeTag = str(uuid.uuid1())
            dpg.add_theme(tag=plotThemeTag)
            plotThemeComponentTag = str(uuid.uuid1())
            dpg.add_theme_component(dpg.mvLineSeries, tag=plotThemeComponentTag, parent=plotThemeTag)

            t = float(i) / float(numSamples)
            plotColor = ( 
                int(plotColor0[0] * (1.0-t) + plotColor1[0] * t),
                int(plotColor0[1] * (1.0-t) + plotColor1[1] * t),
                int(plotColor0[2] * (1.0-t) + plotColor1[2] * t)
            )

            dpg.add_theme_color(dpg.mvPlotCol_Line, plotColor, category=dpg.mvThemeCat_Plots, parent=plotThemeComponentTag)
            self._plotThemeTags.append(plotThemeTag)

        self._labelGroupTag = str(uuid.uuid1())
        self._labelGroup = dpg.add_group(horizontal=True, tag=self._labelGroupTag, parent=self._windowTag)
        
        self._tickerLabelTag = str(uuid.uuid1())
        dpg.add_text( default_value="Ticker: ", parent=self._labelGroupTag )
        dpg.add_text( tag=self._tickerLabelTag, default_value="XYZW", color=[255, 254, 255], parent=self._labelGroupTag)

        self._groupTag = str(uuid.uuid1())
        self._group = dpg.add_group(horizontal=True, tag=self._groupTag, parent=self._windowTag)

        self._orderbookPlotTag = str(uuid.uuid1())
        self._orderbookPlot = dpg.add_plot( label='Orderbook', width=384, height=256, tag=self._orderbookPlotTag, parent=self._group)
        dpg.add_plot_legend(parent=self._orderbookPlot)
        self._orderbookXAxisTag = str(uuid.uuid1())
        dpg.add_plot_axis(dpg.mvXAxis, tag=self._orderbookXAxisTag, parent=self._orderbookPlot)
        self._orderbookYAxisTag = str(uuid.uuid1())
        dpg.add_plot_axis(dpg.mvYAxis, tag=self._orderbookYAxisTag, parent=self._orderbookPlot)
        self._orderbookSeriesTags = []
        for i in range(self._numSamples):
            orderbookSeriesTag = str(uuid.uuid1())
            self._orderbookSeriesTags.append(orderbookSeriesTag)
            j = self._numSamples - i - 1
            dpg.add_line_series( [j for j in range(0, 32)], [random.random() for j in range(0, 32)], label='T-' + str(j), parent=self._orderbookXAxisTag, tag=orderbookSeriesTag )
            dpg.bind_item_theme(orderbookSeriesTag, self._plotThemeTags[i])

        self._lastTradesPlotTag = str(uuid.uuid1())
        self._lastTradesPlot = dpg.add_plot( label='Last trades', width=384, height=256, tag=self._lastTradesPlotTag, parent=self._group)
        dpg.add_plot_legend(parent=self._lastTradesPlot)
        self._lastTradesXAxisTag = str(uuid.uuid1())
        dpg.add_plot_axis(dpg.mvXAxis, tag=self._lastTradesXAxisTag, parent=self._lastTradesPlot)
        self._lastTradesYAxisTag = str(uuid.uuid1())
        dpg.add_plot_axis(dpg.mvYAxis, tag=self._lastTradesYAxisTag, parent=self._lastTradesPlot)
        self._lastTradesSeriesTags = []
        for i in range(self._numSamples):                    
            lastTradesSeriesTag = str(uuid.uuid1())
            self._lastTradesSeriesTags.append(lastTradesSeriesTag)
            j = self._numSamples - i - 1
            dpg.add_line_series( [j for j in range(0, 32)], [random.random() for j in range(0, 32)], label='T-' + str(j), parent=self._lastTradesXAxisTag, tag=lastTradesSeriesTag )
            dpg.bind_item_theme(lastTradesSeriesTag, self._plotThemeTags[i])

        self._orderbookVolumePlotTag = str(uuid.uuid1())
        self._orderbookVolumePlot = dpg.add_plot( label='Orderbook volume', width=384, height=256, tag=self._orderbookVolumePlotTag, parent=self._group)
        dpg.add_plot_legend(parent=self._orderbookVolumePlot)
        self._orderbookVolumeXAxisTag = str(uuid.uuid1())
        dpg.add_plot_axis(dpg.mvXAxis, tag=self._orderbookVolumeXAxisTag, parent=self._orderbookVolumePlot)
        self._orderbookVolumeYAxisTag = str(uuid.uuid1())
        dpg.add_plot_axis(dpg.mvYAxis, tag=self._orderbookVolumeYAxisTag, parent=self._orderbookVolumePlot)
        self._orderbookVolumeSeriesTag = str(uuid.uuid1())
        dpg.add_line_series( [j for j in range(0, 32)], [random.random() for j in range(0, 32)], label='Vol', parent=self._orderbookVolumeXAxisTag, tag=self._orderbookVolumeSeriesTag )

        self._lastTradesVolumePlotTag = str(uuid.uuid1())
        self._lastTradesVolumePlot = dpg.add_plot( label='Last trades volume', width=384, height=256, tag=self._lastTradesVolumePlotTag, parent=self._group)
        dpg.add_plot_legend(parent=self._lastTradesVolumePlot)
        self._lastTradesVolumeXAxisTag = str(uuid.uuid1())
        dpg.add_plot_axis(dpg.mvXAxis, tag=self._lastTradesVolumeXAxisTag, parent=self._lastTradesVolumePlot)
        self._lastTradesVolumeYAxisTag = str(uuid.uuid1())
        dpg.add_plot_axis(dpg.mvYAxis, tag=self._lastTradesVolumeYAxisTag, parent=self._lastTradesVolumePlot)
        self._lastTradesVolumeSeriesTag = str(uuid.uuid1())
        dpg.add_line_series( [j for j in range(0, 32)], [random.random() for j in range(0, 32)], label='Vol', parent=self._lastTradesVolumeXAxisTag, tag=self._lastTradesVolumeSeriesTag )

        self._pricePlotTag = str(uuid.uuid1())
        self._pricePlot = dpg.add_plot( label='Price movement', width=384, height=256, tag=self._pricePlotTag, parent=self._group)
        dpg.add_plot_legend(parent=self._pricePlot)
        self._priceXAxisTag = str(uuid.uuid1())
        dpg.add_plot_axis(dpg.mvXAxis, tag=self._priceXAxisTag, parent=self._pricePlot)
        self._priceYAxisTag = str(uuid.uuid1())
        dpg.add_plot_axis(dpg.mvYAxis, tag=self._priceYAxisTag, parent=self._pricePlot)
        self._priceSeriesTag = str(uuid.uuid1())
        dpg.add_line_series( [j for j in range(0, 32)], [random.random() for j in range(0, 32)], label='Price', parent=self._priceXAxisTag, tag=self._priceSeriesTag )

        self._forecastPlotTag = str(uuid.uuid1())
        self._forecastPlot = dpg.add_plot( label='Price forecast', width=384, height=256, tag=self._forecastPlotTag, parent=self._group)
        dpg.add_plot_legend(parent=self._forecastPlot)
        self._forecastXAxisTag = str(uuid.uuid1())
        dpg.add_plot_axis(dpg.mvXAxis, tag=self._forecastXAxisTag, parent=self._forecastPlot)
        self._forecastYAxisTag = str(uuid.uuid1())
        dpg.add_plot_axis(dpg.mvYAxis, tag=self._forecastYAxisTag, parent=self._forecastPlot)
        self._forecastSeriesTag = str(uuid.uuid1())
        dpg.add_line_series( [j for j in range(0, 32)], [random.random() for j in range(0, 32)], label='Distribution', parent=self._forecastXAxisTag, tag=self._forecastSeriesTag )

    def setTicker(self, ticker:str):
        dpg.set_value( self._tickerLabelTag, ticker )

    def setOrderbook(self, index:int, orderbook:list, labels:list):
        if index >=0 and index < len(self._orderbookSeriesTags):
            dpg.set_value( self._orderbookSeriesTags[index], [labels,orderbook])
            dpg.fit_axis_data( self._orderbookXAxisTag )
            dpg.fit_axis_data( self._orderbookYAxisTag )

    def setLastTrades(self, index:int, distribution:list, labels:list):
        if index >=0 and index < len(self._lastTradesSeriesTags):
            dpg.set_value( self._lastTradesSeriesTags[index], [labels,distribution])
            dpg.fit_axis_data( self._lastTradesXAxisTag )
            dpg.fit_axis_data( self._lastTradesYAxisTag )

    def setOrderbookVolume(self, orderbookVolume:list):
        dpg.set_value( self._orderbookVolumeSeriesTag, [[i for i in range(0, len(orderbookVolume))], orderbookVolume])
        dpg.fit_axis_data( self._orderbookVolumeXAxisTag )
        dpg.fit_axis_data( self._orderbookVolumeYAxisTag )            

    def setLastTradesVolume(self, lastTradesVolume:list):
        dpg.set_value( self._lastTradesVolumeSeriesTag, [[i for i in range(0, len(lastTradesVolume))], lastTradesVolume])
        dpg.fit_axis_data( self._lastTradesVolumeXAxisTag )
        dpg.fit_axis_data( self._lastTradesVolumeYAxisTag )

    def setPrice(self, price:list):
        dpg.set_value( self._priceSeriesTag, [[i for i in range(0, len(price))], price])
        dpg.fit_axis_data( self._priceXAxisTag )
        dpg.fit_axis_data( self._priceYAxisTag )

    def setForecast(self, distribution:list, labels:list):
        dpg.set_value( self._forecastSeriesTag, [labels, distribution])
        dpg.fit_axis_data( self._forecastXAxisTag )
        dpg.fit_axis_data( self._forecastYAxisTag )

#------------------------------------------------------------------------------------------------------------------------

class TkForecastPanel():

    _plotThemeTags = None

    def __init__(self, windowTag:str, instrument:TkInstrument, forecastHistorySize:int, labels:list):

        self._windowTag = windowTag
        self._instrument = instrument
        self._forecastHistorySize = forecastHistorySize

        if( TkForecastPanel._plotThemeTags == None):
            plotColor0 = (31,63,255)
            plotColor1 = (255,63,31)
            TkForecastPanel._plotThemeTags = []
            for i in range(forecastHistorySize):
                plotThemeTag = str(uuid.uuid1())
                dpg.add_theme(tag=plotThemeTag)
                plotThemeComponentTag = str(uuid.uuid1())
                dpg.add_theme_component(dpg.mvLineSeries, tag=plotThemeComponentTag, parent=plotThemeTag)

                t = float(i) / float(forecastHistorySize)
                plotColor = ( 
                    int(plotColor0[0] * (1.0-t) + plotColor1[0] * t),
                    int(plotColor0[1] * (1.0-t) + plotColor1[1] * t),
                    int(plotColor0[2] * (1.0-t) + plotColor1[2] * t)
                )

                dpg.add_theme_color(dpg.mvPlotCol_Line, plotColor, category=dpg.mvThemeCat_Plots, parent=plotThemeComponentTag)
                TkForecastPanel._plotThemeTags.append(plotThemeTag)

        self._groupTag = str(uuid.uuid1())
        self._group = dpg.add_group(horizontal=True, tag=self._groupTag, parent=self._windowTag)
        self._plotTag = str(uuid.uuid1())
        self._plot = dpg.add_plot( label=instrument.ticker(), width=512, height=256, tag=self._plotTag, parent=self._group)
        dpg.add_plot_legend(parent=self._plot)
        self._xAxisTag = str(uuid.uuid1())
        dpg.add_plot_axis(dpg.mvXAxis, tag=self._xAxisTag, parent=self._plot)
        self._yAxisTag = str(uuid.uuid1())
        dpg.add_plot_axis(dpg.mvYAxis, tag=self._yAxisTag, parent=self._plot)

        self._seriesTags = []
        self._series = []
        for i in range(forecastHistorySize):                    
            seriesTag = str(uuid.uuid1())
            self._seriesTags.append(seriesTag)
            self._series.append(None)
            j = forecastHistorySize - i - 1
            dpg.add_line_series( labels, [0.0 for j in range(0, len(labels))], label='T-' + str(j), parent=self._xAxisTag, tag=seriesTag )
            dpg.bind_item_theme(seriesTag, TkForecastPanel._plotThemeTags[i])

        self._infoGroupTag = str(uuid.uuid1())
        self._infoGroup = dpg.add_group(horizontal=True, tag=self._infoGroupTag, parent=self._groupTag)

        self._profitabilityGroupTag = str(uuid.uuid1())
        self._profitabilityGroup = dpg.add_group(horizontal=True, tag=self._profitabilityGroupTag, parent=self._infoGroupTag)
        self._profitabilityLabelTag = str(uuid.uuid1())
        dpg.add_text( default_value="Profit: ", parent=self._profitabilityGroupTag )
        dpg.add_text( tag=self._profitabilityLabelTag, default_value="0.0", color=[255, 254, 255], parent=self._profitabilityGroupTag)

    def close(self):
        dpg.delete_item(self._groupTag)

    def moveBeforeItem(self, item):
        dpg.move_item( item=self._groupTag, before=item._groupTag )       

    def setSeries(self, series : list, labels : list):
        for i in range(self._forecastHistorySize-1):
            self._series[i] = self._series[i+1]

        self._series[-1] = series

        for i in range(self._forecastHistorySize):
            j = self._forecastHistorySize-i-1
            if self._series[j] != None:
                dpg.set_value( self._seriesTags[j], [labels, self._series[j]])
            else:
                break

        dpg.fit_axis_data( self._xAxisTag )
        dpg.fit_axis_data( self._yAxisTag )

    def setProfit(self, value):
        dpg.set_value( self._profitabilityLabelTag, str(value) )        

    _panels = []

    @staticmethod
    def find(instrument : TkInstrument):
        for i in range(len(TkForecastPanel._panels)):
            if TkForecastPanel._panels[i]._instrument.ticker() == instrument.ticker():
                return TkForecastPanel._panels[i]
        return None        
    
    @staticmethod
    def update(instrument : TkInstrument, series:list , labels:list, forecastHistorySize:int, profit:float):
        panel = TkForecastPanel.find( instrument )
        if panel == None:
            panel = TkForecastPanel("primary_window", instrument, forecastHistorySize, labels)
            TkForecastPanel._panels.append( panel )
            
        panel.setSeries( series, labels )
        panel.setProfit( profit )

        if len( TkForecastPanel._panels ) > 1:
            del TkForecastPanel._panels[ TkForecastPanel._panels.index(panel) ]
            TkForecastPanel._panels.insert( 0, panel )
            TkForecastPanel._panels[0].moveBeforeItem( TkForecastPanel._panels[1] )

    @staticmethod
    def discard(instrument : TkInstrument):
        panel = TkForecastPanel.find( instrument )
        if panel != None:
            panel.close()
            del TkForecastPanel._panels[ TkForecastPanel._panels.index(panel) ]

#------------------------------------------------------------------------------------------------------------------------

TOKEN = os.environ["TK_TOKEN"]

cuda = torch.device("cuda")

config = configparser.ConfigParser()
config.read( 'TkConfig.ini' )

data_path = config['Paths']['DataPath']
orderbook_model_path = join( config['Paths']['ModelsPath'], config['Paths']['OrderbookAutoencoderModelFileName'] )
last_trades_model_path = join( config['Paths']['ModelsPath'], config['Paths']['LastTradesAutoencoderModelFileName'] )
ts_model_path =  join( config['Paths']['ModelsPath'], config['Paths']['TimeSeriesModelFileName'] )

orderbook_width = int(config['Autoencoders']['OrderBookWidth'])
last_trades_width = int(config['Autoencoders']['LastTradesWidth'])
min_price_increment_factor = int(config['Autoencoders']['MinPriceIncrementFactor'])

prior_steps_count = int(config['TimeSeries']['PriorStepsCount'])
input_width = int(config['TimeSeries']['InputWidth'])

ipcAddress = config['IPC']['Address']
ipcPort = int(config['IPC']['Port'])
ipcAuthKey = bytes( config['IPC']['AuthKey'], 'ascii' )

#------------------------------------------------------------------------------------------------------------------------
# IPC Thread
# Queues the name of freshly updated data file got from TkGatherData.py process
#------------------------------------------------------------------------------------------------------------------------

ipcMessageQueue = []

def ipc_thread_func():
    global ipcMessageQueue
    print('IPC thread started')
    eofCounter = 0
    creCounter = 0
    while True:
        try:
            with mpc.Listener( (ipcAddress,ipcPort), authkey=ipcAuthKey ) as listener:
                with listener.accept() as conn:
                    try:
                        message = conn.recv()
                        ipcMessageQueue.append( message )
                    except EOFError:                
                        eofCounter = eofCounter + 1
        except ConnectionResetError:
            creCounter = creCounter + 1
                

ipc_thread = threading.Thread( target=ipc_thread_func )
ipc_thread.daemon = True
ipc_thread.start()

#------------------------------------------------------------------------------------------------------------------------
# Load samples from the given path
# The samples are arranged into list of tuples [(orderbook, last_trades), ...]
#------------------------------------------------------------------------------------------------------------------------

def load_samples(path:str, num_samples:int):
    raw_indices = TkIO.index_at_path(path)
    raw_sample_count = int( len(raw_indices) / 2 )

    if raw_sample_count >= num_samples:

        data_stream = open( path, 'rb+')

        try:        
            result = []
            for i in range(num_samples):
                iid = len(raw_indices) - num_samples * 2 + i * 2
                data_stream.seek( raw_indices[iid], 0 )
                orderbook_sample = TkIO.read_from_file( data_stream )
                data_stream.seek( raw_indices[iid+1], 0 )
                last_trades_sample = TkIO.read_from_file( data_stream )
                result.append( (orderbook_sample, last_trades_sample) )
            data_stream.close()
            return result
        except:
            data_stream.close()
            print('Error loading data from stream!')
            return None
    else:
        return None

#------------------------------------------------------------------------------------------------------------------------
# Convert orderbook & last trades samples to TkTimeSeriesForecaster input format
#------------------------------------------------------------------------------------------------------------------------

def preprocess_samples(instrument:TkInstrument, samples:list, orderbook_width:int, last_trades_width:int, min_price_increment_factor:int, orderbook_autoencoder:TkOrderbookAutoencoder, last_trades_autoencoder:TkLastTradesAutoencoder, main_panel:TkMainPanel):

    global cuda

    num_samples = len(samples)
    price = [0.0] * num_samples
    orderbook_volume = [0] * num_samples
    last_trades_volume = [0] * num_samples
    orderbook = [None] * num_samples
    last_trades = [None] * num_samples

    min_price_increment = quotation_to_float( instrument.min_price_increment() )

    for i in range( num_samples ):
        orderbook_sample = samples[i][0]
        distribution, descriptor, volume, pivot_price = TkStatistics.orderbook_distribution( orderbook_sample, orderbook_width, min_price_increment * min_price_increment_factor )
        if volume > 0:
            distribution *= 1.0 / volume
        price[i] = quotation_to_float( orderbook_sample.last_price )
        orderbook_volume[i] = volume
        orderbook[i] = distribution
        if main_panel != None:
            labels = [ 0.5 * (item[0] + item[1]) for item in descriptor]
            main_panel.setOrderbook( i, distribution.tolist(), labels )

        last_trades_sample = samples[i][1]
        distribution, descriptor, volume = TkStatistics.trades_distribution( last_trades_sample, pivot_price, last_trades_width, min_price_increment * min_price_increment_factor )
        if volume > 0:
            distribution *= 1.0 / volume
        last_trades_volume[i] = volume
        last_trades[i] = distribution
        if main_panel != None:
            labels = [ 0.5 * (item[0] + item[1]) for item in descriptor]
            main_panel.setLastTrades( i, distribution.tolist(), labels )

    if main_panel != None:
        main_panel.setOrderbookVolume( orderbook_volume )
        main_panel.setLastTradesVolume( last_trades_volume )
        main_panel.setPrice( price )

    orderbook_input = torch.Tensor( np.concatenate( orderbook ) )
    orderbook_input = torch.reshape( orderbook_input, ( num_samples, 1, orderbook_width ) )
    orderbook_input = orderbook_input.cuda()
    orderbook_code = orderbook_autoencoder.encode(orderbook_input)
    orderbook_code = torch.reshape( orderbook_code, (num_samples, orderbook_autoencoder.code_layer_size() ) )
    orderbook_code = orderbook_code.tolist()

    last_trades_input = torch.Tensor( np.concatenate( last_trades ) )
    last_trades_input = torch.reshape( last_trades_input, ( num_samples, 1, last_trades_width ) )
    last_trades_input = last_trades_input.cuda()
    last_trades_code = last_trades_autoencoder.encode(last_trades_input)
    last_trades_code = torch.reshape( last_trades_code, (num_samples, last_trades_autoencoder.code_layer_size() ) )
    last_trades_code = last_trades_code.tolist()

    base_price = price[-1]
    base_orderbook_volume = sum( orderbook_volume[0:num_samples] ) / num_samples
    base_last_trades_volume = sum( last_trades_volume[0:num_samples] ) / num_samples

    result = [None] * num_samples

    for i in range( num_samples ):
        result[i] = orderbook_code[i].copy()
        result[i].extend( last_trades_code[i].copy() )

        sample_price = price[i]
        sample_price = ( sample_price / base_price - 1.0 ) * 100
        result[i].append( sample_price )

        sample_orderbook_volume = orderbook_volume[i]
        sample_orderbook_volume = ( sample_orderbook_volume / base_orderbook_volume - 1.0 ) * 100 if ( base_orderbook_volume > 0 ) else 0.0
        result[i].append( sample_orderbook_volume )

        sample_last_trades_volume = last_trades_volume[i]
        sample_last_trades_volume = ( sample_last_trades_volume / base_last_trades_volume - 1.0 ) * 100 if ( base_last_trades_volume > 0 ) else 0.0
        result[i].append( sample_last_trades_volume )

    result = list( itertools.chain.from_iterable(result) )

    return result

#------------------------------------------------------------------------------------------------------------------------
# Runs TkTimeSeriesForecaster model
#------------------------------------------------------------------------------------------------------------------------

def forecast(input:list, prior_steps_count:int, input_width:int, last_trades_width:int, ts_model:TkTimeSeriesForecaster, lt_model:TkLastTradesAutoencoder):

    global cuda

    input = torch.Tensor( input ) 
    input = torch.reshape( input, ( 1, prior_steps_count * input_width) )
    input = input.to(cuda)

    ts_output = ts_model.forward( input )
    lt_output = lt_model.decode( ts_output )
    lt_output = torch.reshape( lt_output, ( 1, last_trades_width ) )
    return lt_output

#------------------------------------------------------------------------------------------------------------------------
# Forecast profitability
# Returns desicion flag (True/False) and estimated profit in percents
#------------------------------------------------------------------------------------------------------------------------

def forecast_profitability( price_distribution : list , distribution_descriptor : list, num_modes : int, mode_threshold : float, mean_threshold : float ):

    # ignore short sells
    # suppose that only positive directions of price movement is profitable

    mean = 0.0
    for i in range(len(price_distribution)):
        avgBinPrice = 0.5 *( distribution_descriptor[i][0] + distribution_descriptor[i][1] )
        binWeight = price_distribution[i]        
        mean += avgBinPrice * binWeight
    
    if mean > mean_threshold:
        return True, mean
    
    modes = TkStatistics.get_distribution_modes( price_distribution, distribution_descriptor, num_modes )

    for i in range(num_modes):
        if modes[i][1] > mode_threshold:
            return True, modes[i][1]

    return False, 0   

#------------------------------------------------------------------------------------------------------------------------
# Main loop
#------------------------------------------------------------------------------------------------------------------------

print('Loading orderbook autoencoder...')
orderbook_autoencoder = TkOrderbookAutoencoder(config)
orderbook_autoencoder.to(cuda)
orderbook_autoencoder.load_state_dict(torch.load(orderbook_model_path))
orderbook_autoencoder.eval()

print('Loading last trades autoencoder...')
last_trades_autoencoder = TkLastTradesAutoencoder(config)
last_trades_autoencoder.to(cuda)
last_trades_autoencoder.load_state_dict(torch.load(last_trades_model_path))
last_trades_autoencoder.eval()

print('Loading time series forecaster...')
time_series_forecaster = TkTimeSeriesForecaster(config)
time_series_forecaster.to(cuda)
time_series_forecaster.load_state_dict(torch.load(ts_model_path))
time_series_forecaster.eval()

with Client(TOKEN, target=INVEST_GRPC_API) as client:

    dpg.create_context()
    dpg.create_viewport(title='Forecasting service', width=2388, height=936)
    dpg.setup_dearpygui()

    main_window = dpg.add_window(tag="primary_window", label="Forecasting service")
    main_panel = TkMainPanel("primary_window", prior_steps_count)
    toast = ToastNotifier()

    dpg.show_viewport()
    dpg.set_primary_window("primary_window", True)

    while dpg.is_dearpygui_running():
        
        if len(ipcMessageQueue) > 0:
            filename = ipcMessageQueue[0]
            del ipcMessageQueue[0]
            ticker = filename[ 0: filename.find("_") ]
            instrument = TkInstrument(client, config,  InstrumentType.INSTRUMENT_TYPE_SHARE, ticker, "TQBR")
            
            samples = load_samples( join( data_path, filename), prior_steps_count )

            if samples != None:

                last_price = quotation_to_float( samples[-1][0].last_price )
                min_price_increment = quotation_to_float(instrument.min_price_increment())
                distribution_incremental_value = (min_price_increment_factor * min_price_increment) / last_price * 100
                output_distribution_descriptor = TkStatistics.distribution_descriptor( distribution_incremental_value, int(last_trades_width / 2) )
                output_distribution_labels = [ 0.5 * (item[0] + item[1]) for item in output_distribution_descriptor]

                t0 = default_timer()
                input = preprocess_samples( instrument, samples, orderbook_width, last_trades_width, min_price_increment_factor, orderbook_autoencoder, last_trades_autoencoder, main_panel)
                preprocess_samples_time = default_timer() - t0

                t0 = default_timer()
                output = forecast(input, prior_steps_count, input_width, last_trades_width, time_series_forecaster, last_trades_autoencoder)
                forecast_time = default_timer() - t0

                output = list( itertools.chain.from_iterable( output.tolist() ) )
                main_panel.setForecast( output, output_distribution_labels )

                is_profitable, profit = forecast_profitability( output , output_distribution_descriptor, num_modes = 4, mode_threshold = 0.75, mean_threshold = 0.75)

                if is_profitable:
                    TkForecastPanel.update(instrument, output, output_distribution_labels, 3, profit)
                    if not toast.notification_active():
                        toastMessage = instrument.ticker() + ' +' + str(profit) + '%'
                        toast.show_toast( instrument.ticker(), toastMessage, duration = 10, threaded = True)
                        print( toastMessage )
                else:
                    TkForecastPanel.discard(instrument)

                main_panel.setTicker( filename + " / " + ticker + " / " + instrument.figi() + ", prep: " + str(preprocess_samples_time) + " forecast: " + str(forecast_time) )

            else:

                main_panel.setTicker( filename + " / " + ticker + " / " + instrument.figi() )

        dpg.render_dearpygui_frame()
        

    dpg.destroy_context()

