
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
import bisect
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
    _priceMinMaxThemeTag = None
    _priceAvgThemeTag = None
    _forecastMinMaxThemeTags = None
    _forecastAvgThemeTags = None

    @staticmethod
    def blendColor(plotColor0 : tuple, plotColor1 : tuple, t:float):
        return tuple( int(plotColor0[i] * (1.0-t) + plotColor1[i] * t) for i in range(len(plotColor0)) )

    @staticmethod
    def createLineSeriesPlotTheme(plotColor : tuple):
        plotThemeTag = str(uuid.uuid1())
        dpg.add_theme( tag = plotThemeTag )
        plotThemeComponentTag = str(uuid.uuid1())
        dpg.add_theme_component(dpg.mvLineSeries, tag=plotThemeComponentTag, parent=plotThemeTag)
        dpg.add_theme_color(dpg.mvPlotCol_Line, plotColor, category=dpg.mvThemeCat_Plots, parent=plotThemeComponentTag)
        return plotThemeTag

    def __init__(self, windowTag:str, instrument:TkInstrument, forecastHistorySize:int, futureStepsCount:int, descriptor:list, labels:list):

        self._windowTag = windowTag
        self._instrument = instrument
        self._forecastHistorySize = forecastHistorySize
        self._futureStepsCount = futureStepsCount
        self._descriptor = descriptor
        self._step = 0

        if( TkForecastPanel._plotThemeTags == None):
            plotColor0 = (31,63,255)
            plotColor1 = (255,63,31)
            plotColor3 = (23,47,191)
            plotColor4 = (191,47,23)

            TkForecastPanel._plotThemeTags = []            
            for i in range(forecastHistorySize):
                t = float(i) / float(forecastHistorySize)
                plotColor = TkForecastPanel.blendColor(plotColor0, plotColor1, t)
                plotThemeTag = TkForecastPanel.createLineSeriesPlotTheme( plotColor )
                TkForecastPanel._plotThemeTags.append(plotThemeTag)

            TkForecastPanel._priceAvgThemeTag = TkForecastPanel.createLineSeriesPlotTheme( (255,183,7) )
            TkForecastPanel._priceMinMaxThemeTag = TkForecastPanel.createLineSeriesPlotTheme( (127,91,3) )

            TkForecastPanel._forecastMinMaxThemeTags = []
            TkForecastPanel._forecastAvgThemeTags = []
            for i in range(futureStepsCount):
                t = float(i) / float(futureStepsCount)
                plotColor = TkForecastPanel.blendColor(plotColor0, plotColor1, t)
                plotThemeTag = TkForecastPanel.createLineSeriesPlotTheme( plotColor )
                TkForecastPanel._forecastAvgThemeTags.append(plotThemeTag)
                plotColor = TkForecastPanel.blendColor(plotColor3, plotColor4, t)
                plotThemeTag = TkForecastPanel.createLineSeriesPlotTheme( plotColor )
                TkForecastPanel._forecastMinMaxThemeTags.append(plotThemeTag)

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

        self._pricePlotGroupTag = str(uuid.uuid1())
        self._pricePlotGroup = dpg.add_group(horizontal=True, tag=self._pricePlotGroupTag, parent=self._groupTag)

        self._pricePlotTag = str(uuid.uuid1())
        self._pricePlot = dpg.add_plot( label=instrument.ticker(), width=1024, height=256, tag=self._pricePlotTag, parent=self._pricePlotGroup)

        dpg.add_plot_legend(parent=self._pricePlot)
        self._priceXAxisTag = str(uuid.uuid1())
        dpg.add_plot_axis(dpg.mvXAxis, tag=self._priceXAxisTag, parent=self._pricePlot)
        self._priceYAxisTag = str(uuid.uuid1())
        dpg.add_plot_axis(dpg.mvYAxis, tag=self._priceYAxisTag, parent=self._pricePlot)

        self._priceMinSeriesTag = str(uuid.uuid1())
        self._priceMaxSeriesTag = str(uuid.uuid1())
        self._priceAvgSeriesTag = str(uuid.uuid1())
        self._priceMinSeries = [0]
        self._priceMaxSeries = [0]
        self._priceAvgSeries = [0]
        self._priceLabels = [0]

        dpg.add_line_series( x=self._priceLabels, y=self._priceMinSeries, label='P.Min',  parent=self._priceXAxisTag, tag=self._priceMinSeriesTag )
        dpg.add_line_series( x=self._priceLabels, y=self._priceMaxSeries, label='P.Max',  parent=self._priceXAxisTag, tag=self._priceMaxSeriesTag )
        dpg.add_line_series( x=self._priceLabels, y=self._priceAvgSeries, label='P.Avg', parent=self._priceXAxisTag, tag=self._priceAvgSeriesTag )
        dpg.bind_item_theme( self._priceMinSeriesTag, TkForecastPanel._priceMinMaxThemeTag)
        dpg.bind_item_theme( self._priceMaxSeriesTag, TkForecastPanel._priceMinMaxThemeTag)
        dpg.bind_item_theme( self._priceAvgSeriesTag, TkForecastPanel._priceAvgThemeTag)

        self._forecastMinSeriesTags = []
        self._forecastMaxSeriesTags = []
        self._forecastAvgSeriesTags = []
        self._forecastMinSeries = []
        self._forecastMaxSeries = []
        self._forecastAvgSeries = []
        self._forecastLabels = []

        for i in range(futureStepsCount):
            forecastMinSeriesTag = str(uuid.uuid1())
            forecastMaxSeriesTag = str(uuid.uuid1())
            forecastAvgSeriesTag = str(uuid.uuid1())
            self._forecastMinSeriesTags.append(forecastMinSeriesTag)
            self._forecastMaxSeriesTags.append(forecastMaxSeriesTag)
            self._forecastAvgSeriesTags.append(forecastAvgSeriesTag)
            self._forecastMinSeries.append([0])
            self._forecastMaxSeries.append([0])
            self._forecastAvgSeries.append([0])
            self._forecastLabels.append([0])
            j = futureStepsCount - i - 1
            dpg.add_line_series( x=self._forecastLabels[-1], y=self._forecastMinSeries[-1], label='F.Min T-'+ str(j), parent=self._priceXAxisTag, tag=forecastMinSeriesTag )
            dpg.add_line_series( x=self._forecastLabels[-1], y=self._forecastMaxSeries[-1], label='F.Max T-'+ str(j), parent=self._priceXAxisTag, tag=forecastMaxSeriesTag )
            dpg.add_line_series( x=self._forecastLabels[-1], y=self._forecastAvgSeries[-1], label='F.Avg T-'+ str(j), parent=self._priceXAxisTag, tag=forecastAvgSeriesTag )
            dpg.bind_item_theme( forecastMinSeriesTag, TkForecastPanel._forecastMinMaxThemeTags[i])
            dpg.bind_item_theme( forecastMaxSeriesTag, TkForecastPanel._forecastMinMaxThemeTags[i])
            dpg.bind_item_theme( forecastAvgSeriesTag, TkForecastPanel._forecastAvgThemeTags[i])

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

    def setSeries(self, pivotPrice : float, priceSeries : list, priceDescriptor : list, forecastSeries : list, labels : list):

        # distributions

        for i in range(self._forecastHistorySize-1):
            self._series[i] = self._series[i+1]

        self._series[-1] = forecastSeries

        for i in range(self._forecastHistorySize):
            j = self._forecastHistorySize-i-1
            if self._series[j] != None:
                dpg.set_value( self._seriesTags[j], [labels, self._series[j]])
            else:
                break

        dpg.fit_axis_data( self._xAxisTag )
        dpg.fit_axis_data( self._yAxisTag )

        # prices

        mean = TkStatistics.get_distribution_mean( priceSeries, priceDescriptor )
        left_mean, right_mean = TkStatistics.get_distribution_tails( priceSeries, priceDescriptor, 0.001 )

        mean = pivotPrice + mean * pivotPrice / 100.0
        left_mean = pivotPrice + left_mean * pivotPrice / 100.0
        right_mean = pivotPrice + right_mean * pivotPrice / 100.0

        if self._step > 0:
            self._priceMinSeries.append(left_mean)
            self._priceMaxSeries.append(right_mean)
            self._priceAvgSeries.append(mean)
        else:
            self._priceMinSeries[-1] = left_mean
            self._priceMaxSeries[-1] = right_mean
            self._priceAvgSeries[-1] = mean

        dpg.set_value( self._priceMinSeriesTag, [self._priceLabels, self._priceMinSeries])
        dpg.set_value( self._priceMaxSeriesTag, [self._priceLabels, self._priceMaxSeries])
        dpg.set_value( self._priceAvgSeriesTag, [self._priceLabels, self._priceAvgSeries])
        dpg.fit_axis_data( self._priceXAxisTag )
        dpg.fit_axis_data( self._priceYAxisTag )

        self._priceLabels.append( self._step + 1 )

        # forecast prices

        mean = TkStatistics.get_distribution_mean( forecastSeries, self._descriptor )
        left_mean, right_mean = TkStatistics.get_distribution_tails( forecastSeries, self._descriptor, 0.001 )

        mean = pivotPrice + mean * pivotPrice / 100.0
        left_mean = pivotPrice + left_mean * pivotPrice / 100.0
        right_mean = pivotPrice + right_mean * pivotPrice / 100.0

        sliceIdx = self._step % self._futureStepsCount

        if len(self._forecastMinSeries[sliceIdx]) == 1:
            self._forecastMinSeries[sliceIdx][0] = left_mean

        if len(self._forecastMaxSeries[sliceIdx]) == 1:
            self._forecastMaxSeries[sliceIdx][0] = right_mean

        if len(self._forecastAvgSeries[sliceIdx]) == 1:
            self._forecastAvgSeries[sliceIdx][0] = mean

        for i in range(self._futureStepsCount):
            self._forecastLabels[sliceIdx].append( self._step + 1 + i )
            self._forecastMinSeries[sliceIdx].append( left_mean )
            self._forecastMaxSeries[sliceIdx].append( right_mean )
            self._forecastAvgSeries[sliceIdx].append( mean )

        dpg.set_value( self._forecastMinSeriesTags[sliceIdx], [self._forecastLabels[sliceIdx], self._forecastMinSeries[sliceIdx]])
        dpg.set_value( self._forecastMaxSeriesTags[sliceIdx], [self._forecastLabels[sliceIdx], self._forecastMaxSeries[sliceIdx]])
        dpg.set_value( self._forecastAvgSeriesTags[sliceIdx], [self._forecastLabels[sliceIdx], self._forecastAvgSeries[sliceIdx]])
            
        dpg.fit_axis_data( self._xAxisTag )
        dpg.fit_axis_data( self._yAxisTag )
        dpg.fit_axis_data( self._priceXAxisTag )
        dpg.fit_axis_data( self._priceYAxisTag )

        self._step = self._step + 1

    def setProfit(self, value):
        dpg.set_value( self._profitabilityLabelTag, str(value) )        

    def getForecastScore(self):

        def getPriceAtStep(step:int):
            labelIdx = bisect.bisect_left( self._priceLabels, step )
            if labelIdx == 0:
                return None, None, None
            else:
                if labelIdx < len(self._priceMinSeries):
                    priceMin = self._priceMinSeries[labelIdx]
                    priceMax = self._priceMaxSeries[labelIdx]
                    priceAvg = self._priceAvgSeries[labelIdx]
                    return priceMin, priceMax, priceAvg
                else:
                    return None,None,None

        minScore = 0.0
        maxScore = 0.0
        avgScore = 0.0
        norm = 0

        for slice in range(self._futureStepsCount):
            if len(self._forecastLabels[slice]) > 0:
                for label in range( int(len(self._forecastLabels[slice]) / 4) ):
                    step0 = self._forecastLabels[slice][label]
                    step1 = step0 + self._futureStepsCount
                    forecastMin = self._forecastMinSeries[slice][label]
                    forecastMax = self._forecastMaxSeries[slice][label]
                    forecastAvg = self._forecastAvgSeries[slice][label]
                
                    priceMin = None
                    priceMax = None
                    priceAvg = None
                    lnorm = 0
                
                    for step in range(step0, step1):
                        pmin, pmax, pavg = getPriceAtStep(step)
                        if pmin != None:
                            if priceMin != None:
                                priceMin = min( pmin, priceMin )
                                priceMax = max( pmax, priceMax)
                                priceAvg = priceAvg + pavg
                            else:
                                priceMin = pmin
                                priceMax = pmax
                                priceAvg = pavg
                            lnorm = lnorm + 1

                    priceAvg = priceAvg / lnorm if lnorm > 0 else priceAvg

                    if priceMin != None:
                        minScore = minScore + abs( priceMin - forecastMin ) / priceAvg
                        maxScore = maxScore + abs( priceMax - forecastMax ) / priceAvg
                        avgScore = avgScore + abs( priceAvg - forecastAvg ) / priceAvg
                        norm = norm + 1
        result = 0.0
        if norm > 0:
            result = (minScore + maxScore + avgScore) / norm
                        
        return result

    _panels = []

    @staticmethod
    def find(instrument : TkInstrument):
        for i in range(len(TkForecastPanel._panels)):
            if TkForecastPanel._panels[i]._instrument.ticker() == instrument.ticker():
                return TkForecastPanel._panels[i]
        return None        
    
    @staticmethod
    def getTotalForecastScore():
        result = 0
        norm = 0
        for i in range(len(TkForecastPanel._panels)):
            result = result + TkForecastPanel._panels[i].getForecastScore()
            norm = norm + 1
        return result / norm if norm > 0 else result
    
    @staticmethod
    def update(instrument:TkInstrument, pivotPrice:float, priceSeries:list, priceDescriptor:list, forecastSeries:list, forecastDescriptor:list, forecastLabels:list, forecastHistorySize:int, futureStepsCount:int, profit:float):        
        panel = TkForecastPanel.find( instrument )
        if panel == None:
            panel = TkForecastPanel("primary_window", instrument, forecastHistorySize, futureStepsCount, forecastDescriptor, forecastLabels)
            TkForecastPanel._panels.append( panel )

        panel.setSeries(pivotPrice, priceSeries, priceDescriptor, forecastSeries, forecastLabels)

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
future_steps_count = int(config['TimeSeries']['FutureStepsCount'])
input_width = int(config['TimeSeries']['InputWidth'])

profitability = float(config['ForecastingService']['Profitability'])
tail_mean_order = int(config['ForecastingService']['TailMeanOrder'])
discard_if_non_profitable = ( config['ForecastingService']['DiscardIfNonProfitable'] == 'True' )
event_notification = ( config['ForecastingService']['EventNotification'] == 'True' )
forecast_history_size = int(config['ForecastingService']['ForecastHistorySize'])

#------------------------------------------------------------------------------------------------------------------------
# IPC Threads
# Queues the name of freshly updated data file got from TkGatherData.py process
#------------------------------------------------------------------------------------------------------------------------

ipc_input_message_queue = []
ipc_output_message_queue = []

def ipc_input_thread_func():
    global ipc_input_message_queue
    global config

    print('IPC input thread started')
    
    ipc_input_address = config['IPC']['ForecastingServiceAddress']
    ipc_input_port = int(config['IPC']['ForecastingServicePort'])
    ipc_input_auth_key = bytes( config['IPC']['ForecastingServiceAuthKey'], 'ascii' )
    
    eofCounter = 0
    creCounter = 0

    while True:
        try:
            with mpc.Listener( (ipc_input_address, ipc_input_port), authkey=ipc_input_auth_key ) as listener:
                with listener.accept() as conn:
                    try:
                        message = conn.recv()
                        ipc_input_message_queue.append( message )
                    except EOFError:                
                        eofCounter = eofCounter + 1
        except ConnectionResetError:
            creCounter = creCounter + 1

def ipc_output_thread_func():
    global ipc_output_message_queue
    global config

    print('IPC output thread started')
    
    ipc_output_address = config['IPC']['TradingServiceAddress']
    ipc_output_port = int(config['IPC']['TradingServicePort'])
    ipc_output_auth_key = bytes( config['IPC']['TradingServiceAuthKey'], 'ascii' )

    while True:
        try:
            with mpc.Client( (ipc_output_address, ipc_output_port), authkey=ipc_output_auth_key ) as conn:
                while len(ipc_output_message_queue) > 0:
                    message = ipc_output_message_queue[0]
                    del ipc_output_message_queue[0]
                    conn.send(message)
        except ConnectionError:
            print("Reconnecting output thread...")
            ipc_output_message_queue.clear()

ipc_input_thread = threading.Thread( target=ipc_input_thread_func )
ipc_input_thread.daemon = True
ipc_input_thread.start()

ipc_output_thread = threading.Thread( target=ipc_output_thread_func )
ipc_output_thread.daemon = True
ipc_output_thread.start()

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
# 
# Ignores short sells
# Supposes that only positive direction of price movement is profitable.
#------------------------------------------------------------------------------------------------------------------------

def forecast_profitability( price_distribution : list , distribution_descriptor : list, profitability_threshold : float, tail_mean_order:int ):

    left_mean, right_mean = TkStatistics.get_distribution_tail_means( price_distribution, distribution_descriptor, tail_mean_order )
    
    if right_mean > profitability_threshold:
        return True, right_mean
    else:
        return False, 0.0

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
        
        if len(ipc_input_message_queue) > 0:
            filename = ipc_input_message_queue[0]
            del ipc_input_message_queue[0]
            ticker = filename[ 0: filename.find("_") ]
            instrument = TkInstrument(client, config,  InstrumentType.INSTRUMENT_TYPE_SHARE, ticker, "TQBR")
            
            samples = load_samples( join( data_path, filename), prior_steps_count )

            if samples != None:

                min_price_increment = quotation_to_float(instrument.min_price_increment())
                _, _, _, last_price = TkStatistics.orderbook_distribution( samples[-1][0], orderbook_width, min_price_increment * min_price_increment_factor )

                last_trades, last_trades_descriptor, last_trades_volume = TkStatistics.trades_distribution( samples[-1][1], last_price, last_trades_width, min_price_increment * min_price_increment_factor)

                if last_trades_volume > 0:
                    last_trades *= 1.0/last_trades_volume
                
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

                is_profitable, profit = forecast_profitability( output , output_distribution_descriptor, profitability_threshold=profitability, tail_mean_order=tail_mean_order )

                if is_profitable:
                    # Multiple forecasts in a row
                    if TkForecastPanel.find(instrument) != None:
                        ipc_output_message_queue.append( (instrument.ticker(), profit) )
                    TkForecastPanel.update(instrument, last_price, last_trades.tolist(), last_trades_descriptor, output, output_distribution_descriptor, output_distribution_labels, forecast_history_size, future_steps_count, profit)                    
                    if event_notification and not toast.notification_active():
                        toastMessage = instrument.ticker() + ' +' + str(profit) + '%'
                        toast.show_toast( instrument.ticker(), toastMessage, duration = 10, threaded = True)
                        print( toastMessage )
                else:
                    if discard_if_non_profitable:
                        TkForecastPanel.discard(instrument)
                    else:
                        TkForecastPanel.update(instrument, last_price, last_trades.tolist(), last_trades_descriptor, output, output_distribution_descriptor, output_distribution_labels, forecast_history_size, future_steps_count, profit)

                score = TkForecastPanel.getTotalForecastScore()
                main_panel.setTicker( filename + " / " + ticker + " / " + instrument.figi() + " / score : " + str(score) )

            else:

                score = TkForecastPanel.getTotalForecastScore()
                main_panel.setTicker( filename + " / " + ticker + " / " + instrument.figi() + " / score : " + str(score) )

        dpg.render_dearpygui_frame()
        

    dpg.destroy_context()

