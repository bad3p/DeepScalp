
import os.path
import numpy as np
import random
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
    # * pivoted around maximal bid price / minimal ask price / last price - conditionally on availability of specific orders
    # * with discretization proportional to given min_price_increment
    # * ask orders grouped in the positive range
    # * bid orders grouped in the negative range
    #------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def orderbook_distribution(orderbook : GetOrderBookResponse, orderbook_width : int, min_price_increment : float):

        pivot_price = 0.0

        if len(orderbook.bids) > 0:
            for bid in orderbook.bids:
                pivot_price = max( pivot_price, quotation_to_float( bid.price ) )
        else:
            pivot_price = quotation_to_float( orderbook.last_price )
            for ask in orderbook.asks:
                pivot_price = min( pivot_price, quotation_to_float( ask.price ) )

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

    #------------------------------------------------------------------------------------------------------------------------
    # Accumulate another sample of anonymized trades in the given distrubution of order volumes
    #------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def accumulate_trades_distribution(distribution : np.ndarray, descriptor : list, volume : int, trades : GetLastTradesResponse, pivot_price : float):

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

        return volume

    #------------------------------------------------------------------------------------------------------------------------
    # Convert discrete distribution to cumulative form
    # * the distribution pivot is at the middle point of input array
    #------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def to_cumulative_distribution(distribution : np.ndarray):
        if distribution.size % 2 > 0:
            raise RuntimeError('Distribution size mismatch!')
        half_size = int(distribution.size/2)-1
        for i in range(1, half_size+1):
            distribution[half_size-i] = distribution[half_size-i] + distribution[half_size-i+1]
            distribution[half_size+i+1] = distribution[half_size+i+1] + distribution[half_size+i]

    #------------------------------------------------------------------------------------------------------------------------
    # Generates discrete distribution based on the given scheme
    # * the scheme is a list of mandatory modes that should form the distribution
    # * bias term determines how much magnitudes of modes could deviate from the scheme value
    #------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def generate_distribution(distribution : np.ndarray, scheme : list, bias : float):
        cumulative_distribution_weight = 0
        for i in range(len(distribution)):
            cumulative_distribution_weight = cumulative_distribution_weight + distribution[i]
        for i in range(len(scheme)):
            sample_weight = scheme[i] * (1.0 + random.uniform(-bias,bias))
            idx = random.randint(0, distribution.size-1)
            distribution[idx] = distribution[idx] + sample_weight
            cumulative_distribution_weight = cumulative_distribution_weight + sample_weight
        
        distribution *= 1.0 / cumulative_distribution_weight

    #------------------------------------------------------------------------------------------------------------------------
    # Generates clustered discrete distribution based on the given scheme
    # * the scheme is a list of mandatory modes that should form the distribution
    # * bias term determines how much magnitudes of modes could deviate from the scheme value
    # * the clustering is around random bin index and mimics normal distribution with given variance
    #------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def generate_clustered_distribution(distribution : np.ndarray, scheme : list, bias : float, variance : float):
        cumulative_distribution_weight = 0
        for i in range(len(distribution)):
            cumulative_distribution_weight = cumulative_distribution_weight + distribution[i]

        pivot_idx = random.randint(0, distribution.size-1)
        range_idx = int( distribution.size * random.uniform(0,variance) )

        for i in range(len(scheme)):
            sample_weight = scheme[i] * (1.0 + random.uniform(-bias,bias))
            idx = int(random.gauss(pivot_idx, range_idx))
            idx = min( max( 0, idx ), distribution.size-1 )
            distribution[idx] = distribution[idx] + sample_weight
            cumulative_distribution_weight = cumulative_distribution_weight + sample_weight
        
        distribution *= 1.0 / cumulative_distribution_weight        

    #------------------------------------------------------------------------------------------------------------------------
    # Return modes of the given discrete distribution
    #------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def get_distribution_modes(distribution : np.ndarray, descriptor : list, num_modes : int):
        result = []
        for i in range(0, len(distribution)):
            mode = ( distribution[i], 0.5 * (descriptor[i][0] + descriptor[i][1]) )
            if mode[0] > 0:
                for j in range(0, len(result)):
                    if result[j][0] < mode[0]:
                        result.insert(j, mode)
                        mode = None
                        break
                if mode != None and len(result) < num_modes:
                    result.append(mode)
                if len(result) > num_modes:
                    result.pop()
        return result
    
    #------------------------------------------------------------------------------------------------------------------------
    # Return nogative and positive "tails" of the given distribution
    #------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def get_distribution_tails(distribution : np.ndarray, descriptor : list, epsilon : float):
        left_tail = 0.0
        right_tail = 0.0
        for i in range(0, len(distribution)):
            if distribution[i] > epsilon:
                p = 0.5 * (descriptor[i][0] + descriptor[i][1])
                left_tail = min(left_tail, p)
                right_tail = max(right_tail, p)
            
        return left_tail, right_tail

    #------------------------------------------------------------------------------------------------------------------------
    # Return mean of the given discrete distribution
    #------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def get_distribution_mean(distribution : np.ndarray, descriptor : list):
        mean = 0.0
        for i in range(len(distribution)):
            avg_bin_price = 0.5 *( descriptor[i][0] + descriptor[i][1] )
            bin_weight = distribution[i]        
            mean += avg_bin_price * bin_weight
        return mean