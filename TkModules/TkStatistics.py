
import os.path
import numpy as np
from tinkoff.invest.schemas import Quotation
from tinkoff.invest import GetOrderBookResponse, GetLastTradesResponse
from TkModules.TkQuotation import quotation_to_float

#------------------------------------------------------------------------------------------------------------------------
# Statistics helpers
#------------------------------------------------------------------------------------------------------------------------

class TkStatistics():

    #------------------------------------------------------------------------------------------------------------------------
    # Returns descriptor of discrete cumulative distribution
    # The distribution is pivoted at 0.0
    # The discretization step is defined by 'incremental_value'
    # The discretization range is defined by 'num_bins' in both positive and negative sides
    # The result descriptor is a list of tuples, containing individual discrete intervals:
    #------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def cumulative_distribution_descriptor(incremental_value : float, num_bins : int):
        discretization = [ i * incremental_value for i in range(1, num_bins + 1)]
        positiveRanges = [(0.0, val) for val in discretization]
        negativeRanges = [(-val, 0.0) for val in discretization]
        negativeRanges.reverse()
        result = negativeRanges + positiveRanges
        return result

    #------------------------------------------------------------------------------------------------------------------------
    # Returns descriptor of discrete distribution
    # The distribution is pivoted at 0.0
    # The discretization step is defined by 'incremental_value'
    # The discretization range is defined by 'num_bins' in both positive and negative sides
    # The result descriptor is a list of tuples, containing individual discrete intervals:
    #------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def distribution_descriptor(incremental_value : float, num_bins : int):
        discretization = [ i * incremental_value for i in range(1, num_bins + 1)]
        positiveRanges = [(discretization[i], discretization[i+1]) for i in range(0,len(discretization)-1)]
        negativeRanges = [(-discretization[i+1], -discretization[i]) for i in range(0,len(discretization)-1)]
        positiveRanges.insert( 0, (0.0, discretization[0]))
        positiveRanges.insert( 0, (-discretization[0], 0.0))
        negativeRanges.reverse()
        result = negativeRanges + positiveRanges
        return result

    #------------------------------------------------------------------------------------------------------------------------
    # For the given orderbook, the method returns cumulative distrubution of order volumes,
    # * pivoted around maximal bid price
    # * with discretization proportional to given min_price_increment
    # * ask orders grouped in the positive range
    # * bid orders grouped in the negative range
    #------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def orderbook_distribution(orderbook : GetOrderBookResponse, orderbook_width : int, min_price_increment : float):

        pivot_price = 0.0
        for bid in orderbook.bids:
            pivot_price = max( pivot_price, quotation_to_float( bid.price ) )

        distribution_incremental_value = min_price_increment / pivot_price * 100
        descriptor = TkStatistics.cumulative_distribution_descriptor( distribution_incremental_value, int(orderbook_width / 2) )

        distribution = np.empty( len(descriptor), dtype=float)
        distribution.fill( 0 )

        volume = 0
        for ask in orderbook.asks:
            price = quotation_to_float( ask.price )
            price_percent = ( price / pivot_price - 1.0 ) * 100
            volume = volume + ask.quantity
            outOfBounds = True
            for i in range(len(descriptor)):
                if price_percent >= descriptor[i][0] and price_percent < descriptor[i][1]:
                    distribution[i] = distribution[i] + ask.quantity
                    outOfBounds = False
            if outOfBounds:
                distribution[-1] = distribution[-1] + ask.quantity
        for bid in orderbook.bids:
            price = quotation_to_float( bid.price )
            price_percent = ( price / pivot_price - 1.0 ) * 100
            volume = volume + bid.quantity
            outOfBounds = True
            for i in range(len(descriptor)):
                if price_percent > descriptor[i][0] and price_percent <= descriptor[i][1]:
                    distribution[i] = distribution[i] + bid.quantity
                    outOfBounds = False
            if outOfBounds:
                distribution[0] = distribution[0] + bid.quantity

        return distribution, descriptor, volume, pivot_price

    #------------------------------------------------------------------------------------------------------------------------
    # For the given list of anonymized trades, the method returns distrubution of order volumes,
    # * pivoted around given price
    # * with discretization proportional to given min_price_increment
    #------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def trades_distribution(trades : GetLastTradesResponse, pivot_price : float, distribution_width : int, min_price_increment : float):

        distribution_incremental_value = min_price_increment / pivot_price * 100
        descriptor = TkStatistics.distribution_descriptor( distribution_incremental_value, int(distribution_width / 2) )

        distribution = np.empty( len(descriptor), dtype=float)
        distribution.fill( 0 )

        volume = 0

        for trade in trades.trades:
            price = quotation_to_float( trade.price )
            price_percent = ( price / pivot_price - 1.0 ) * 100
            volume = volume + trade.quantity

            outOfBounds = True
            for i in range(len(descriptor)):
                if price_percent < 0:
                    if price_percent > descriptor[i][0] and price_percent <= descriptor[i][1]:
                        distribution[i] = distribution[i] + trade.quantity
                        outOfBounds = False
                        break
                else:
                    if price_percent >= descriptor[i][0] and price_percent < descriptor[i][1]:
                        distribution[i] = distribution[i] + trade.quantity
                        outOfBounds = False
            if outOfBounds:
                if price_percent < 0:
                    distribution[0] = distribution[0] + trade.quantity
                else:
                    distribution[-1] = distribution[-1] + trade.quantity

        return distribution, descriptor, volume