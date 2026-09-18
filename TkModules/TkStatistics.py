
import sys
import os.path
import numpy as np
import random
import math
import bisect
from numpy.lib.stride_tricks import sliding_window_view
from collections import defaultdict
from decimal import Decimal
from TkModules.TkInstrument import TkOrderbook, TkLastTrades

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
    # Returns significant range for the cumulative distribution
    # Significant range is between minimal and maximal index of histogram beyond which the values repeat themselves
    #------------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def cumulative_significant_range(distribution : np.ndarray, tolerance=1e-6):
        min_index = 0
        while min_index < distribution.size - 2 and abs( distribution[min_index] - distribution[min_index+1] ) < tolerance:
            min_index = min_index + 1

        max_index = distribution.size - 1
        while max_index > 1 and abs( distribution[max_index] - distribution[max_index-1] ) < tolerance:
            max_index = max_index - 1

        return min_index, max_index

    
    #------------------------------------------------------------------------------------------------------------------------
    # For the given orderbook, the method returns its absolute spread
    #------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def orderbook_spread(orderbook : TkOrderbook, orderbook_width : int, min_price_increment : float):
        
        max_bid_price = 0.0
        if len(orderbook.bids) > 0:
            for bid in orderbook.bids:
                max_bid_price = max( max_bid_price, float( bid[0] ) )

        min_ask_price = 10e10
        if len(orderbook.asks) > 0:
            for ask in orderbook.asks:
                min_ask_price = min( min_ask_price, float( ask[0] ) )

        spread = ( min_ask_price - max_bid_price ) / min_price_increment
        return int( max( 0.0, min( spread, orderbook_width) ) ) * min_price_increment

    
    #------------------------------------------------------------------------------------------------------------------------
    # For the given orderbook, the method returns cumulative distrubution of order volumes,
    # * pivoted around maximal bid price / minimal ask price / last price - conditionally on availability of specific orders
    # * with discretization proportional to given min_price_increment
    # * ask orders grouped in the positive range
    # * bid orders grouped in the negative range
    #------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def orderbook_distribution(orderbook : TkOrderbook, orderbook_width : int, min_price_increment : float):

        pivot_price = 0.0

        if len(orderbook.bids) > 0:
            for bid in orderbook.bids:
                pivot_price = max( pivot_price, float( bid[0] ) )
        else:
            pivot_price = float( orderbook.last_price )
            for ask in orderbook.asks:
                pivot_price = min( pivot_price, float( ask[0] ) )

        distribution_incremental_value = min_price_increment / pivot_price * 100
        descriptor = TkStatistics.cumulative_distribution_descriptor( distribution_incremental_value, int(orderbook_width / 2) )

        distribution = np.empty( len(descriptor), dtype=float)
        distribution.fill( 0 )

        volume = 0
        for ask in orderbook.asks:
            price = float( ask[0] )
            price_percent = ( price / pivot_price - 1.0 ) * 100
            volume = volume + ask[1]
            outOfBounds = True
            for i in range(len(descriptor)):
                if price_percent >= descriptor[i][0] and price_percent < descriptor[i][1]:
                    distribution[i] = distribution[i] + ask[1]
                    outOfBounds = False
            if outOfBounds:
                distribution[-1] = distribution[-1] + ask[1]
        for bid in orderbook.bids:
            price = float( bid[0] )
            price_percent = ( price / pivot_price - 1.0 ) * 100
            volume = volume + bid[1]
            outOfBounds = True
            for i in range(len(descriptor)):
                if price_percent > descriptor[i][0] and price_percent <= descriptor[i][1]:
                    distribution[i] = distribution[i] + bid[1]
                    outOfBounds = False
            if outOfBounds:
                distribution[0] = distribution[0] + bid[1]

        return distribution, descriptor, volume, pivot_price

    #------------------------------------------------------------------------------------------------------------------------
    # For the given list of anonymized trades, the method returns distrubution of order volumes,
    # * pivoted around given price
    # * with discretization proportional to given min_price_increment
    #------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def trades_distribution(trades : TkLastTrades, pivot_price : float, distribution_width : int, discretization_interval : float):

        min_price_increment = pivot_price * 0.01 * discretization_interval
        distribution_incremental_value = min_price_increment / pivot_price * 100
        descriptor = TkStatistics.distribution_descriptor( distribution_incremental_value, int(distribution_width / 2) )

        distribution = np.empty( len(descriptor), dtype=float)
        distribution.fill( 0 )

        volume = 0

        for trade in trades.trades:
            price = float( trade[0] )
            price_percent = ( price / pivot_price - 1.0 ) * 100
            volume = volume + trade[1]

            outOfBounds = True
            for i in range(len(descriptor)):
                if price_percent < 0:
                    if price_percent > descriptor[i][0] and price_percent <= descriptor[i][1]:
                        distribution[i] = distribution[i] + trade[1]
                        outOfBounds = False
                        break
                else:
                    if price_percent >= descriptor[i][0] and price_percent < descriptor[i][1]:
                        distribution[i] = distribution[i] + trade[1]
                        outOfBounds = False
            if outOfBounds:
                if price_percent < 0:
                    distribution[0] = distribution[0] + trade[1]
                else:
                    distribution[-1] = distribution[-1] + trade[1]

        return distribution, descriptor, volume

    #------------------------------------------------------------------------------------------------------------------------
    # Accumulate another sample of anonymized trades in the given distrubution of order volumes
    #------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def accumulate_trades_distribution(distribution : np.ndarray, descriptor : list, volume : int, trades : TkLastTrades, pivot_price : float):

        for trade in trades.trades:
            price = float( trade[0] )
            price_percent = ( price / pivot_price - 1.0 ) * 100
            volume = volume + trade[1]

            outOfBounds = True
            for i in range(len(descriptor)):
                if price_percent < 0:
                    if price_percent > descriptor[i][0] and price_percent <= descriptor[i][1]:
                        distribution[i] = distribution[i] + trade[1]
                        outOfBounds = False
                        break
                else:
                    if price_percent >= descriptor[i][0] and price_percent < descriptor[i][1]:
                        distribution[i] = distribution[i] + trade[1]
                        outOfBounds = False
            if outOfBounds:
                if price_percent < 0:
                    distribution[0] = distribution[0] + trade[1]
                else:
                    distribution[-1] = distribution[-1] + trade[1]

        return volume

    #------------------------------------------------------------------------------------------------------------------------
    # Extract volume weighted average price and volume weighted volatility for the given collection of anonymized trades
    #------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def trades_statistics(trades : TkLastTrades, time_threshold = None ):

        total_currency_volume = 0.0
        total_volume = 0

        for trade in trades.trades:
            trade_time = trade[2]
            if time_threshold != None and trade_time < time_threshold:
                continue
            price = float( trade[0] )
            total_currency_volume = total_currency_volume + price * trade[1]
            total_volume = total_volume + trade[1]

        if total_volume == 0:
            return 0.0, 0.0, 0.0

        # volume weighted average price
        vwap = total_currency_volume / total_volume 

        # volume weighted variance
        vwvar = 0
        for trade in trades.trades:
            trade_time = trade[2]
            if time_threshold != None and trade_time < time_threshold:
                continue
            price = float( trade[0] )
            vwvar = vwvar + trade[1] * (price - vwap)**2

        vwvar = vwvar / total_volume

        # volume weighted volatility
        vwvol = math.sqrt(vwvar)
        normalized_vwvol = vwvol / vwap

        return vwap, vwvol, normalized_vwvol

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
    def generate_distribution(distribution : np.ndarray, scheme : list, bias : float, min_index : int, max_index : int):
        cumulative_distribution_weight = 0
        for i in range(len(distribution)):
            cumulative_distribution_weight = cumulative_distribution_weight + distribution[i]
        for i in range(len(scheme)):
            sample_weight = scheme[i] * (1.0 + random.uniform(-bias,bias))
            idx = random.randint(min_index, max_index)
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
    
    #------------------------------------------------------------------------------------------------------------------------
    # Return "tail means" of the given distribution
    # Let's suppose the "central mean" of a set of values is equal to its true mean.
    # Then the "right tail mean" of 1nd order is a mean of values from a given set which are larger than the "central mean".
    #------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def get_distribution_tail_means(distribution : list, descriptor : list, order: int):

        mean = 0.0
        for i in range(len(distribution)):
            mean += descriptor[i] * distribution[i]

        if order == 0:
            return float(mean), float(mean)

        right_mean = 0.0
        right_bin_weight = 0.0
        left_mean = 0.0
        left_bin_weight = 0.0

        for i in range(len(distribution)):
            if descriptor[i] < mean:
                left_mean += descriptor[i] * distribution[i]
                left_bin_weight += distribution[i]
            elif descriptor[i] > mean:
                right_mean += descriptor[i] * distribution[i]
                right_bin_weight += distribution[i]

        if left_bin_weight > 0.0:
            left_mean *= 1.0 / left_bin_weight
        else:
            left_mean = mean

        if right_bin_weight > 0.0:
            right_mean *= 1.0 / right_bin_weight
        else:
            right_mean = mean

        if order == 1:
            return float(left_mean), float(right_mean)
        
        leftmost_mean = left_mean
        leftmost_bin_weight = 0.0
        rightmost_mean = right_mean
        rightmost_bin_weight = 0.0

        for i in range(order-1):

            leftmost_mean = 0.0
            leftmost_bin_weight = 0.0
            rightmost_mean = 0.0
            rightmost_bin_weight = 0.0
            
            for j in range(len(distribution)):
                if descriptor[j] < left_mean:
                    leftmost_mean += descriptor[j] * distribution[j]
                    leftmost_bin_weight += distribution[j]
                elif descriptor[j] > right_mean:
                    rightmost_mean += descriptor[j] * distribution[j]
                    rightmost_bin_weight += distribution[j]

            if leftmost_bin_weight > 0.0:
                leftmost_mean *= 1.0 / leftmost_bin_weight
            else:
                leftmost_mean = left_mean

            if rightmost_bin_weight > 0.0:
                rightmost_mean *= 1.0 / rightmost_bin_weight
            else:
                rightmost_mean = right_mean

        return float(leftmost_mean), float(rightmost_mean)
    
    #------------------------------------------------------------------------------------------------------------------------
    # Deduces minimal price increment from orderbooks
    # Price increment for certain shares can vary over time
    #------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def get_min_price_increment(orderbook : TkOrderbook, default_value:Decimal):

        def get_min_price_diff(prices:list):
            min_diff = Decimal(99999999999)
            for i in range(len(prices) - 1):
                current_diff = prices[i+1] - prices[i]
                if current_diff < min_diff:
                    min_diff = current_diff
            return min_diff

        bid_prices = [ bid[0] for bid in orderbook.bids ]
        ask_prices = [ ask[0] for ask in orderbook.asks ]

        if len(bid_prices) == 0 and len(ask_prices) == 0:
            return default_value

        bid_prices.sort()
        ask_prices.sort()

        return min( get_min_price_diff(bid_prices), get_min_price_diff(ask_prices) )

    #------------------------------------------------------------------------------------------------------------------------
    # Converts orderbook to multi-channel tensor with following channels per level:
    # (0) delta price with pivot price == ( max bid price | min ask price | last price )
    # (1) absolute volume
    # (2) normalized volume
    # (3) imbalance
    # (4) discrete imbalance
    #------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def orderbook_to_tensor(orderbook : TkOrderbook, orderbook_width : int, min_price_increment : float):

        def almost_equal(a, b, rel_tol=1e-09, abs_tol=1e-06):
            return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)
        
        def discrete_imbalance(val):
            return 0 if almost_equal(val,0.0) else ( -1 if val < 0.0 else 1 )            

        pivot_price = 0.0

        if len(orderbook.bids) > 0:
            for bid in orderbook.bids:
                pivot_price = max( pivot_price, float( bid[0] ) )
        else:
            pivot_price = float( orderbook.last_price )
            for ask in orderbook.asks:
                pivot_price = min( pivot_price, float( ask[0] ) )

        bid_price = [ pivot_price - min_price_increment * i for i in range(int(orderbook_width/2))]
        ask_price = [ pivot_price + min_price_increment * i for i in range(1,int(orderbook_width/2)+1)]

        bid_delta_price = [ math.log( price / pivot_price ) for price in bid_price]
        ask_delta_price = [ math.log( price / pivot_price ) for price in ask_price]

        order_book_slope = 0
        total_bid_volume = 0
        weighted_bid = 0        

        bid_volume = np.empty( len(bid_price), dtype=float)
        bid_volume.fill(0)

        for bid in orderbook.bids:
            price = float( bid[0] )
            if price > pivot_price:
                #print( 'Bid overlapping asks: ', pivot_price, price )
                continue
            index = min( int( round( (pivot_price - price) / min_price_increment ) ), int(orderbook_width/2)-1 )
            assert almost_equal(price, bid_price[index]) if index < int(orderbook_width/2)-1 else True , "Bid index mismatch: " + str(price) + " : " + str(bid_price[index])
            total_bid_volume = total_bid_volume + bid[1]
            bid_volume[index] = bid_volume[index] + bid[1]
            weighted_bid = weighted_bid + price * bid[1]
            order_book_slope = order_book_slope + bid[1] * (price - pivot_price)

        total_ask_volume = 0
        weighted_ask = 0

        ask_volume = np.empty( len(ask_price), dtype=float)
        ask_volume.fill(0) 
        
        for ask in orderbook.asks:
            price = float( ask[0] )
            if price <= pivot_price:
                #print( 'Ask overlapping bids: ', pivot_price, price )
                continue
            index = min( int( round( (price - pivot_price) / min_price_increment ) - 1 ), int(orderbook_width/2)-1 )
            assert almost_equal(price, ask_price[index]) if index < int(orderbook_width/2)-1 else True, "Ask index mismatch: " + str(price) + " : " + str(ask_price[index])
            total_ask_volume = total_ask_volume + ask[1]
            ask_volume[index] = ask_volume[index] + ask[1]
            weighted_ask = weighted_ask + price * ask[1]
            order_book_slope = order_book_slope + ask[1] * (price - pivot_price)

        bid_imbalance = np.empty( len(bid_volume), dtype=float)
        bid_imbalance.fill(0)

        bid_discrete_imbalance = np.empty( len(bid_volume), dtype=float)
        bid_discrete_imbalance.fill(0)

        ask_imbalance = np.empty( len(ask_volume), dtype=float)
        ask_imbalance.fill(0) 

        ask_discrete_imbalance = np.empty( len(ask_volume), dtype=float)
        ask_discrete_imbalance.fill(0) 

        bid_total = 0
        ask_total = 0

        epsilon = np.finfo(np.float32).eps

        for i in range(int(orderbook_width/2)):
            bid_total = bid_total + bid_volume[i]
            ask_total = ask_total + ask_volume[i]
            bid_imbalance[i] = (bid_total - ask_total) / (bid_total + ask_total + epsilon)
            bid_discrete_imbalance[i] = discrete_imbalance(bid_imbalance[i])
            ask_imbalance[i] = (ask_total - bid_total) / (bid_total + ask_total + epsilon)
            ask_discrete_imbalance[i] = discrete_imbalance(ask_imbalance[i])

        bid_delta_price = np.flip( bid_delta_price )
        delta_price_tensor = np.concatenate((bid_delta_price, ask_delta_price))

        empty_volume = np.empty( len(bid_price), dtype=float)
        empty_volume.fill(0)

        bid_volume = np.flip( bid_volume )        
        bid_volume_tensor = np.concatenate((bid_volume, empty_volume))

        normalized_bid_volume_tensor = bid_volume_tensor.copy()

        if total_bid_volume > 0:
            normalized_bid_volume_tensor = normalized_bid_volume_tensor * 1.0 / total_bid_volume
            assert almost_equal( np.sum(normalized_bid_volume_tensor), 1.0), "|Normalized volume tensor| != 1.0"

        ask_volume_tensor = np.concatenate((empty_volume, ask_volume))

        normalized_ask_volume_tensor = ask_volume_tensor.copy()

        if total_ask_volume > 0:
            normalized_ask_volume_tensor = normalized_ask_volume_tensor * 1.0 / total_ask_volume
            assert almost_equal( np.sum(normalized_ask_volume_tensor), 1.0), "|Normalized volume tensor| != 1.0"

        bid_imbalance = np.flip(bid_imbalance)
        imbalance_tensor = np.concatenate((bid_imbalance, ask_imbalance))

        bid_discrete_imbalance = np.flip(bid_discrete_imbalance)
        discrete_imbalance_tensor = np.concatenate((bid_discrete_imbalance, ask_discrete_imbalance))
        
        bid_volume_tensor = np.log1p( bid_volume_tensor )
        ask_volume_tensor = np.log1p( ask_volume_tensor )

        result_tensor = np.stack([delta_price_tensor, bid_volume_tensor, normalized_bid_volume_tensor, ask_volume_tensor, normalized_ask_volume_tensor, imbalance_tensor, discrete_imbalance_tensor], axis=1)

        assert not np.isnan(result_tensor).any(), "NaNs in result tensor!"

        hasheable_tensor = np.multiply( delta_price_tensor, np.add( bid_volume_tensor, ask_volume_tensor ) )

        order_book_slope = order_book_slope / max(1, total_bid_volume + total_ask_volume)

        microprice = 0.0

        if len(orderbook.bids) > 0 and len(orderbook.asks) > 0:

            best_bid_volume = orderbook.bids[0].quantity
            best_bid_price = quotation_to_float( orderbook.bids[0].price )

            best_ask_volume = orderbook.asks[0].quantity
            best_ask_price = quotation_to_float( orderbook.asks[0].price )
        
            midprice = ( best_bid_price + best_ask_price ) / 2            
            microprice = ( best_ask_price * best_bid_volume + best_bid_price * best_ask_volume ) / ( best_bid_volume + best_ask_volume )
            microprice = midprice - microprice

        # transpose result_tensor to make it ready to be pytorch-convertible (1,C,W)
        return result_tensor.T, hasheable_tensor, pivot_price, (total_bid_volume + total_ask_volume), order_book_slope, microprice

    #------------------------------------------------------------------------------------------------------------------------
    # Extract various statistics from the orderbook
    # * volume
    # * midprice
    # * microprice_offset
    # * order_book_slope 
    # * bid_alpha, ask_alpha
    #------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def orderbook_statistics(orderbook : TkOrderbook, min_price_increment : float):
        
        def offset_price(levels:list, price_delta:float):
            for i in range(len(levels)):
                levels[i] = ( levels[i][0] + price_delta, levels[i][1] )
        
        microprice_offset = 0.0
        order_book_slope = 0.0
        bid_alpha = 1.0
        ask_alpha = 1.0

        # edge case

        if len(orderbook.bids) == 0 or len(orderbook.asks) == 0:
            print( 'Default statistics, orderbook is incomplete: ', len(orderbook.bids), len(orderbook.asks) )
            total_volume = 0
            for bid in orderbook.bids:            
                total_volume = total_volume + bid[1]
            for ask in orderbook.asks:
                total_volume = total_volume + ask[1]
            return total_volume, float(orderbook.last_price), microprice_offset, order_book_slope, bid_alpha, ask_alpha

        # LOB levels

        bids = [ ( float(level[0]), level[1]) for level in orderbook.bids]
        asks = [ ( float(level[0]), level[1]) for level in orderbook.asks]

        # correct overlapping

        if float(orderbook.bids[0][0]) >= float(orderbook.asks[0][0]):

            midprice = ( bids[0][0] + asks[0][0] ) / 2
            price_delta = bids[0][0] - midprice + min_price_increment / 2

            offset_price( bids, -price_delta )
            offset_price( asks, price_delta )

            if bids[0][0] >= asks[0][0]:
                raise RuntimeError('Overlapping correction failed.')

        midprice = ( bids[0][0] + asks[0][0] ) / 2
        pivot_price = round( midprice / min_price_increment ) * min_price_increment

        order_book_slope = 0
        total_bid_volume = 0

        for bid in bids:            
            total_bid_volume = total_bid_volume + bid[1]
            order_book_slope = order_book_slope + bid[1] * (bid[0] - midprice)

        total_ask_volume = 0

        for ask in asks:            
            total_ask_volume = total_ask_volume + ask[1]
            order_book_slope = order_book_slope + ask[1] * (ask[0] - midprice)

        total_volume = total_bid_volume + total_ask_volume

        order_book_slope = order_book_slope / max(1, total_volume)

        # orderbook microprice

        best_bid_volume = bids[0][1]
        best_bid_price = bids[0][0]

        best_ask_volume = asks[0][1]
        best_ask_price = asks[0][0]

        best_volume = ( best_bid_volume + best_ask_volume )                
        microprice = ( best_ask_price * best_bid_volume + best_bid_price * best_ask_volume ) / ( best_volume if best_volume > 1e-8  else 1.0 )
        microprice_offset = microprice - midprice

        # orderbook convexity/concavity

        def side_convexity(levels, mid_price : float, is_bid=True, min_points=3, eps=1e-8):

            # Extract arrays
            prices = np.array([p for p, _ in levels], dtype=float)
            qtys = np.array([q for _, q in levels], dtype=float)

            # Edge case: single level → no curvature info
            if len(levels) < min_points:
                return 1.0  # neutral (linear) assumption

            # Distance from mid
            if is_bid:
                distances = mid_price - prices
            else:
                distances = prices - mid_price

            # Filter valid points
            mask = (distances > eps) & (qtys > eps)
            distances = distances[mask]
            qtys = qtys[mask]

            # Not enough valid points
            if len(distances) < min_points:
                return np.nan

            # Cumulative depth
            cum_qty = np.cumsum(qtys)

            # Guard against zeros
            valid = (cum_qty > eps) & (distances > eps)
            if valid.sum() < min_points:
                return np.nan

            x = np.log(distances[valid])
            y = np.log(cum_qty[valid])

            # Degenerate case: identical distances
            if np.allclose(x, x[0]):
                return np.nan

            # Fit slope (alpha)
            try:
                alpha, _ = np.polyfit(x, y, 1)
            except np.linalg.LinAlgError:
                return np.nan

            return float(alpha)        
                
        bid_alpha = side_convexity(bids, midprice, is_bid=True)
        ask_alpha = side_convexity(asks, midprice, is_bid=False)
        if math.isnan(bid_alpha):
            print( 'Bid alpha is NaN, others are: ', midprice, microprice_offset, order_book_slope )
            bid_alpha = 1.0
        if math.isnan(ask_alpha):
            print( 'Ask alpha is NaN, others are: ', midprice, microprice_offset, order_book_slope )
            ask_alpha = 1.0

        return total_volume, pivot_price, microprice_offset, order_book_slope, bid_alpha, ask_alpha

    #------------------------------------------------------------------------------------------------------------------------
    # For the given list of anonymized trades, the method returns distrubution of order volumes,
    # * pivoted around given price
    # * with given relative discretization interval (in percents)
    # * with optional time threshold allowing to ignore events older than the certain time
    # Using "force_categorical" the method will return valid categorical distribution peaked at pivot_price if the input volume is zero.
    # Additionally the method measures the means of distribution tails, given the order "measure_distribution_tail_order"
    #------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def last_trades_to_tensor(trades_with_time_threshold : list, pivot_price : float, distribution_width : int, discretization_interval : float, force_categorical: bool = False, measure_distribution_tail_order=1):

        def almost_equal(a, b, rel_tol=1e-09, abs_tol=1e-06):
            return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)
        
        min_price_increment = pivot_price * 0.01 * discretization_interval

        bid_price = [ pivot_price - min_price_increment * i for i in range(int(distribution_width/2))]
        ask_price = [ pivot_price + min_price_increment * i for i in range(1,int(distribution_width/2)+1)]

        bid_abs_delta_price = [ ( price / pivot_price - 1.0 ) * 100 for price in bid_price]
        ask_abs_delta_price = [ ( price / pivot_price - 1.0 ) * 100 for price in ask_price]

        bid_log_delta_price = [ math.log( price / pivot_price ) for price in bid_price]
        ask_log_delta_price = [ math.log( price / pivot_price ) for price in ask_price]

        total_volume = 0
        total_event_count = 0

        bid_volume = np.empty( len(bid_price), dtype=float)
        bid_volume.fill( 0 )

        ask_volume = np.empty( len(ask_price), dtype=float)
        ask_volume.fill( 0 )

        total_buy_trades = 0
        total_sell_trades = 0

        for i in range(len(trades_with_time_threshold)):

            trades = trades_with_time_threshold[i][0]
            time_threshold = trades_with_time_threshold[i][1]

            for trade in trades.trades:
                trade_time = trade[2]
                if time_threshold != None and trade_time < time_threshold:
                    continue
                total_event_count = total_event_count + 1
                price = float( trade[0] )
                if price <= pivot_price:
                    index = max( 0, min( int( round( (pivot_price - price) / min_price_increment ) ), int(distribution_width/2)-1 ) )
                    total_volume = total_volume + trade[1]
                    bid_volume[index] = bid_volume[index] + trade[1]
                    total_sell_trades = total_sell_trades + trade[1]
                else:
                    index = max( 0, min( int( round( (price - pivot_price) / min_price_increment ) - 1 ), int(distribution_width/2)-1 ) )
                    total_volume = total_volume + trade[1]
                    ask_volume[index] = ask_volume[index] + trade[1]
                    total_buy_trades = total_buy_trades + trade[1]

        if total_volume == 0 and force_categorical:
            total_volume = 1
            bid_volume[0] = 1

        bid_log_delta_price = np.flip( bid_log_delta_price )
        log_delta_price_tensor = np.concatenate((bid_log_delta_price,ask_log_delta_price))

        bid_abs_delta_price = np.flip( bid_abs_delta_price )
        abs_delta_price_tensor = np.concatenate((bid_abs_delta_price,ask_abs_delta_price))

        bid_volume = np.flip( bid_volume )
        volume_tensor = np.concatenate((bid_volume, ask_volume))
        normalized_volume_tensor = volume_tensor.copy()
        if total_volume > 0:
            normalized_volume_tensor = normalized_volume_tensor * 1.0 / total_volume            
            assert almost_equal( np.sum(normalized_volume_tensor), 1.0), "|Normalized volume tensor| != 1.0"

            # TODO: configure
            total_nonzero = np.count_nonzero(normalized_volume_tensor)
            if total_nonzero > 16:
                normalized_volume_tensor = TkStatistics.smooth_with_given_kernel(normalized_volume_tensor, TkStatistics.generate_gaussian_kernel(9,1.0))
            elif total_nonzero > 8:
                normalized_volume_tensor = TkStatistics.smooth_with_given_kernel(normalized_volume_tensor, TkStatistics.generate_gaussian_kernel(7,1.0))
            elif total_nonzero > 4:
                normalized_volume_tensor = TkStatistics.smooth_with_given_kernel(normalized_volume_tensor, TkStatistics.generate_gaussian_kernel(5,1.0))
            elif total_nonzero > 2:
                normalized_volume_tensor = TkStatistics.smooth_with_given_kernel(normalized_volume_tensor, TkStatistics.generate_gaussian_kernel(3,1.0))


        volume_tensor = np.log1p( volume_tensor )

        result_tensor = np.stack([log_delta_price_tensor, volume_tensor, normalized_volume_tensor], axis=1)

        assert not np.isnan(result_tensor).any(), "NaNs in result tensor!"

        hasheable_tensor = np.multiply( log_delta_price_tensor, volume_tensor )

        left_tail, right_tail = TkStatistics.get_distribution_tail_means( normalized_volume_tensor, abs_delta_price_tensor, measure_distribution_tail_order )

        # transpose result_tensor to make it ready to be pytorch-convertible (1,C,W)
        return result_tensor.T, hasheable_tensor, total_event_count, total_volume, total_buy_trades, total_sell_trades, (left_tail, right_tail)

    #------------------------------------------------------------------------------------------------------------------------
    # Performs EMA normalization for the given sequence
    #------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def ema_normalize(sequence, half_life=250, eps=1e-12):

        sequence = np.asarray(sequence, dtype=np.float64)

        # EMA coefficient
        alpha = 1.0 - np.exp(-np.log(2.0) / half_life)

        mu = np.zeros_like(sequence)
        var = np.zeros_like(sequence)

        mu[0] = sequence[0]
        var[0] = 0.0

        for t in range(1, len(sequence)):
            mu[t] = alpha * sequence[t] + (1.0 - alpha) * mu[t - 1]
            diff = sequence[t] - mu[t]
            var[t] = alpha * diff * diff + (1.0 - alpha) * var[t - 1]

        norm_sequence = (sequence - mu) / np.sqrt(var + eps)

        return norm_sequence

    #------------------------------------------------------------------------------------------------------------------------
    # Performs log1p transform with subsequent EMA normalization for the given sequence
    #------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def log_ema_normalize(sequence, half_life=250, eps=1e-12):

        sequence = np.asarray(sequence, dtype=np.float64)

        # log transform
        log_sequence = np.sign(sequence) * np.log1p(abs(sequence))

        # EMA coefficient
        alpha = 1.0 - np.exp(-np.log(2.0) / half_life)

        mu = np.zeros_like(log_sequence)
        var = np.zeros_like(log_sequence)

        mu[0] = log_sequence[0]
        var[0] = 0.0

        for t in range(1, len(log_sequence)):
            mu[t] = alpha * log_sequence[t] + (1.0 - alpha) * mu[t - 1]
            diff = log_sequence[t] - mu[t]
            var[t] = alpha * diff * diff + (1.0 - alpha) * var[t - 1]

        norm_sequence = (log_sequence - mu) / np.sqrt(var + eps)

        return norm_sequence
    
    #------------------------------------------------------------------------------------------------------------------------
    # Performs log1p transform with subsequent short-term volatility extraction & EMA normalization
    #------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def log_vol_ema_normalize(sequence, half_life=250, eps=1e-12):

        sequence = np.asarray(sequence, dtype=np.float64)

        log_sequence = np.log(sequence)

        r = np.diff(log_sequence, prepend=log_sequence[0])
        r2 = r * r

        alpha = 1.0 - np.exp(-np.log(2.0) / half_life)

        ema_r2 = np.zeros_like(r2)
        ema_r2[0] = r2[0]

        for t in range(1, len(r2)):
            ema_r2[t] = alpha * r2[t] + (1 - alpha) * ema_r2[t - 1]

        # volatility estimate
        vol = np.sqrt(ema_r2 + eps)

        return vol

    #------------------------------------------------------------------------------------------------------------------------
    # Given the volatility, the method computes market volatility regime
    #------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def volatility_to_market_regime( volatility:float, regime_thresholds:list):
        return bisect.bisect_right( regime_thresholds, volatility )
    
    #------------------------------------------------------------------------------------------------------------------------
    # Given the absolute price series, the method computes market regimes based on rolling volatility 
    #------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def price_to_market_regimes(prices:list, regime_thresholds:list, rolling_window_size:int=20):

        # insert padding to the input list, using first element as a padding value
        prices = prices.copy()
        prices[0:0] = [prices[0]] * rolling_window_size

        thresholds = np.sort(np.array(regime_thresholds, dtype=float))

        log_returns = np.diff(np.log(prices))
        rolling_vol = np.full_like(prices, fill_value=np.nan, dtype=float)

        for t in range(rolling_window_size - 1, len(log_returns)):
            window_slice = log_returns[t - rolling_window_size + 1 : t + 1]
            rolling_vol[t + 1] = np.std(window_slice, ddof=0)
    
        regimes = np.searchsorted(thresholds, rolling_vol, side='right')
    
        regimes = regimes.tolist()

        # remove padding elements from output list
        regimes = regimes[rolling_window_size:] 
        
        return regimes
    
    #------------------------------------------------------------------------------------------------------------------------
    # Computes market regime thresholds for given sequence of prices
    #------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def calculate_regime_thresholds(prices: np.ndarray, num_regimes: int, rolling_window_size: int = 20) -> list:
        """
        Calculates data-driven volatility thresholds using native NumPy K-Means clustering.
        
        Parameters:
        -----------
        prices : array_like
            Sequence of historical prices.
        num_regimes : int
            The number of market regimes you want to model. Must be >= 2.
        rolling_window_size : int, optional
            Lookback window to calculate rolling volatility (default is 20 periods).
            
        Returns:
        --------
        thresholds : list
            List of size (num_regimes - 1) containing the volatility thresholds.
        """
        prices = np.asarray(prices, dtype=float)
        
        if num_regimes < 2:
            raise ValueError("Number of regimes must be at least 2.")
        if len(prices) <= rolling_window_size:
            raise ValueError("Price sequence must be longer than the rolling window.")

        # 1. Calculate rolling volatility
        log_returns = np.diff(np.log(prices))
        windows = sliding_window_view(log_returns, window_shape=rolling_window_size)
        rolling_vol = np.std(windows, axis=1, ddof=1)
        valid_vol = rolling_vol[~np.isnan(rolling_vol)]
        
        if len(valid_vol) < num_regimes:
            raise ValueError("Not enough data points to compute clusters.")

        # 2. Guardrail: ensure enough unique values exist
        unique_vols = np.unique(valid_vol)
        if len(unique_vols) < num_regimes:
            raise ValueError("Not enough unique volatility values found.")            

        # 3. Native K-Means Implementation
        # Initialize centers evenly across the min/max range to handle zero-inflation safely
        centers = np.linspace(np.min(valid_vol), np.max(valid_vol), num_regimes)
        
        max_iterations = 100
        for _ in range(max_iterations):
            # Calculate distance from each point to each center using broadcasting
            # valid_vol[:, None] is shape (N, 1), centers[None, :] is shape (1, K)
            distances = np.abs(valid_vol[:, None] - centers[None, :])
            
            # Assign each point to the closest center
            labels = np.argmin(distances, axis=1)
            
            new_centers = np.zeros(num_regimes)
            for k in range(num_regimes):
                cluster_points = valid_vol[labels == k]
                if len(cluster_points) > 0:
                    new_centers[k] = np.mean(cluster_points)
                else:
                    # If a cluster ends up empty, retain its previous center
                    new_centers[k] = centers[k]
            
            # Check for convergence (if centers stop moving, we're done)
            if np.allclose(centers, new_centers, atol=1e-8):
                break
                
            centers = new_centers

        # 4. Compute thresholds as the midpoints between sorted centers
        centers = np.sort(centers)
        thresholds = (centers[:-1] + centers[1:]) / 2.0

        return thresholds.tolist()

    #------------------------------------------------------------------------------------------------------------------------
    # Computes market regime thresholds for given sequence of volume weighted volatilities
    #------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def calculate_regime_thresholds_vw(vw_vols: np.ndarray, num_regimes: int) -> list:
        """
        Calculates data-driven volatility thresholds using native NumPy K-Means clustering.
        
        Parameters:
        -----------
        vw_vols : array_like
            Sequence of pre-calculated volume-weighted volatilities.
        num_regimes : int
            The number of market regimes you want to model. Must be >= 2.
            
        Returns:
        --------
        thresholds : list
            List of size (num_regimes - 1) containing the volatility thresholds.
        """
        vols = np.asarray(vw_vols, dtype=float)
        
        if num_regimes < 2:
            raise ValueError("Number of regimes must be at least 2.")

        # 1. Clean the input (remove any NaNs that might have resulted from external rolling calculations)
        valid_vol = vols[~np.isnan(vols)]
        
        if len(valid_vol) < num_regimes:
            raise ValueError("Not enough data points to compute clusters.")

        # 2. Guardrail: ensure enough unique values exist
        unique_vols = np.unique(valid_vol)
        if len(unique_vols) < num_regimes:
            raise ValueError("Not enough unique volatility values found.")            

        # 3. Native K-Means Implementation
        # Initialize centers evenly across the min/max range to handle zero-inflation safely
        centers = np.linspace(np.min(valid_vol), np.max(valid_vol), num_regimes)
        
        max_iterations = 100
        for _ in range(max_iterations):
            # Calculate distance from each point to each center using broadcasting
            # valid_vol[:, None] is shape (N, 1), centers[None, :] is shape (1, K)
            distances = np.abs(valid_vol[:, None] - centers[None, :])
            
            # Assign each point to the closest center
            labels = np.argmin(distances, axis=1)
            
            new_centers = np.zeros(num_regimes)
            for k in range(num_regimes):
                cluster_points = valid_vol[labels == k]
                if len(cluster_points) > 0:
                    new_centers[k] = np.mean(cluster_points)
                else:
                    # If a cluster ends up empty, retain its previous center
                    new_centers[k] = centers[k]
            
            # Check for convergence (if centers stop moving, we're done)
            if np.allclose(centers, new_centers, atol=1e-8):
                break
                
            centers = new_centers

        # 4. Compute thresholds as the midpoints between sorted centers
        centers = np.sort(centers)
        thresholds = (centers[:-1] + centers[1:]) / 2.0

        return thresholds.tolist()

    #------------------------------------------------------------------------------------------------------------------------
    # Compute depth-weighted Order Flow Imbalance between two LOB snapshots.
    # alpha : float = exponential decay parameter for depth weighting
    #------------------------------------------------------------------------------------------------------------------------

    @staticmethod    
    def depth_weighted_order_flow_imbalance(orderbook1 : TkOrderbook, orderbook2 : TkOrderbook, alpha=1.0):
    
        def build_signed_depth(bids, asks):
            S = defaultdict(float)

            for p, q in bids:
                S[p] += q

            for p, q in asks:
                S[p] -= q

            return S
        
        bids_1 = []
        for bid in orderbook1.bids:
            bids_1.append( (float( bid[0] ), bid[1]) )

        asks_1 = []
        for ask in orderbook1.asks:
            asks_1.append( (float( ask[0] ), ask[1]) )

        bids_2 = []
        for bid in orderbook2.bids:
            bids_2.append( (float( bid[0] ), bid[1]) )

        asks_2 = []
        for ask in orderbook2.asks:
            asks_2.append( (float( ask[0] ), ask[1]) )

        S1 = build_signed_depth(bids_1, asks_1)
        S2 = build_signed_depth(bids_2, asks_2)

        # union of price levels
        prices = set(S1.keys()) | set(S2.keys())

        # compute midprice (use second snapshot)
        best_bid = 0
        best_ask = 0

        if len(bids_2) > 0 and len(asks_2) > 0 :
            best_bid = max(p for p, _ in bids_2)
            best_ask = min(p for p, _ in asks_2)
        elif len(bids_2) > 0 and len(asks_2) == 0 :
            best_bid = max(p for p, _ in bids_2)
            best_ask = best_bid
        elif len(bids_2) == 0 and len(asks_2) > 0 :
            best_ask = min(p for p, _ in asks_2)
            best_bid = best_ask
        else:
            return 0.0

        mid = 0.5 * (best_bid + best_ask)

        ofi = 0.0

        for p in prices:
            s1 = S1.get(p, 0.0)
            s2 = S2.get(p, 0.0)

            delta = s2 - s1

            # exponential depth weight
            weight = math.exp(-alpha * abs(p - mid))

            ofi += weight * delta

        return ofi
    
    #------------------------------------------------------------------------------------------------------------------------
    # Returns cumulative sum over input x using given window size
    #------------------------------------------------------------------------------------------------------------------------

    @staticmethod    
    def rolling_sum(x, window:int):
        x = np.asarray(x)
        c = np.cumsum(x)
        out = c.copy()

        if window < len(x):
            out[window:] = c[window:] - c[:-window]

        return out


    #------------------------------------------------------------------------------------------------------------------------
    # Depth-weighted queue imbalance accounting for missing price levels.
    #------------------------------------------------------------------------------------------------------------------------        

    @staticmethod    
    def depth_weighted_queue_imbalance(orderbook : TkOrderbook, min_price_increment : float, depth=None, alpha=1.0):

        bids = []
        for bid in orderbook.bids:
            bids.append( (float( bid[0] ), bid[1]) )

        asks = []
        for ask in orderbook.asks:
            asks.append( (float( ask[0] ), ask[1]) )

        if not bids or not asks:
            return 0.0

        # Sort the book
        bids_sorted = sorted(bids, key=lambda x: x[0], reverse=True)
        asks_sorted = sorted(asks, key=lambda x: x[0])

        best_bid = bids_sorted[0][0]
        best_ask = asks_sorted[0][0]

        weighted_bid = 0.0
        weighted_ask = 0.0

        # ----- BIDS -----
        for price, qty in bids_sorted:

            level = round((best_bid - price) / min_price_increment)

            if depth is not None and level >= depth:
                continue

            weight = math.exp(-alpha * level)
            weighted_bid += weight * qty

        # ----- ASKS -----
        for price, qty in asks_sorted:

            level = round((price - best_ask) / min_price_increment)

            if depth is not None and level >= depth:
                continue

            weight = math.exp(-alpha * level)
            weighted_ask += weight * qty

        total = weighted_bid + weighted_ask
        if total == 0:
            return 0.0

        return (weighted_bid - weighted_ask) / total
    
    #------------------------------------------------------------------------------------------------------------------------
    # Queue depletion rate
    #------------------------------------------------------------------------------------------------------------------------            

    @staticmethod
    def depth_weighted_queue_depletion_rate(orderbook : TkOrderbook, last_trades : TkLastTrades, alpha: float):

        bids = [ ( float(bid[0]), bid[1]) for bid in orderbook.bids]
        asks = [ ( float(ask[0]), ask[1]) for ask in orderbook.asks]
        trades = [ ( float(trade[0]), trade[1]) for trade in last_trades.trades]

        # Sort and convert to mutable lists with initial depth levels: [price, qty, level]
        # Bids descending (highest price first), Asks ascending (lowest price first)
        sorted_bids = [[p, q, i] for i, (p, q) in enumerate(sorted(bids, key=lambda x: x[0], reverse=True))]
        sorted_asks = [[p, q, i] for i, (p, q) in enumerate(sorted(asks, key=lambda x: x[0]))]
    
        bid_depletion = 0.0
        ask_depletion = 0.0
    
        # Handle intersection of bids and asks (crossed book)
        while sorted_bids and sorted_asks and sorted_bids[0][0] >= sorted_asks[0][0]:
            best_bid = sorted_bids[0]
            best_ask = sorted_asks[0]
        
            match_qty = min(best_bid[1], best_ask[1])
        
            # Apply exponentially weighted depletion based on the level crossing
            bid_depletion += match_qty * math.exp(-alpha * best_bid[2])
            ask_depletion += match_qty * math.exp(-alpha * best_ask[2])
        
            if best_bid[1] == match_qty:
                sorted_bids.pop(0)
            else:
                sorted_bids[0][1] -= match_qty
            
            if best_ask[1] == match_qty:
                sorted_asks.pop(0)
            else:
                sorted_asks[0][1] -= match_qty

        # Re-index remaining uncrossed book to establish new top-of-book (level 0)
        for i, bid in enumerate(sorted_bids):
            bid[2] = i
        for i, ask in enumerate(sorted_asks):
            ask[2] = i

        best_bid_p = sorted_bids[0][0] if sorted_bids else None
        best_ask_p = sorted_asks[0][0] if sorted_asks else None

        # Helper functions to find effective dynamic depth level for any price
        def get_bid_level(price):
            level = 0
            for p, _, _ in sorted_bids:
                if p > price: level += 1
                elif p == price: return level
                else: break
            return level

        def get_ask_level(price):
            level = 0
            for p, _, _ in sorted_asks:
                if p < price: level += 1
                elif p == price: return level
                else: break
            return level

        # Process the explicit trades against the uncrossed, re-indexed book
        for trade_price, trade_qty in trades:
            if best_bid_p is not None and best_ask_p is not None:
                if trade_price >= best_ask_p:
                    level = get_ask_level(trade_price)
                    ask_depletion += trade_qty * math.exp(-alpha * level)
                
                elif trade_price <= best_bid_p:
                    level = get_bid_level(trade_price)
                    bid_depletion += trade_qty * math.exp(-alpha * level)
                
                else:
                    # Inside the spread. It operates between level 0 of bid and level 0 of ask.
                    # Because it occurs strictly before level 1, we apply the level 0 weight (exp(0) = 1)
                    spread = best_ask_p - best_bid_p
                    if spread > 0:
                        weight_ask = (trade_price - best_bid_p) / spread
                        weight_bid = (best_ask_p - trade_price) / spread
                    
                        ask_depletion += trade_qty * weight_ask * 1.0 
                        bid_depletion += trade_qty * weight_bid * 1.0
                    
            # Edge cases: Book is completely empty on one or both sides
            elif best_bid_p is not None: 
                if trade_price <= best_bid_p:
                    level = get_bid_level(trade_price)
                    bid_depletion += trade_qty * math.exp(-alpha * level)
                else:
                    ask_depletion += trade_qty * 1.0 # Assume level 0 since ask book is empty
            elif best_ask_p is not None:
                if trade_price >= best_ask_p:
                    level = get_ask_level(trade_price)
                    ask_depletion += trade_qty * math.exp(-alpha * level)
                else:
                    bid_depletion += trade_qty * 1.0 # Assume level 0 since bid book is empty
            else:
                bid_depletion += trade_qty * 0.5
                ask_depletion += trade_qty * 0.5

        assert not math.isnan(bid_depletion) , "Bid depletion is NaN!"
        assert not math.isnan(ask_depletion) , "Ask depletion is NaN!"

        return bid_depletion, ask_depletion

    #------------------------------------------------------------------------------------------------------------------------
    # Order arrival rate    
    #------------------------------------------------------------------------------------------------------------------------            

    @staticmethod
    def depth_weighted_order_arrival_rate(prev_orderbook : TkOrderbook, curr_orderbook : TkOrderbook, last_trades : TkLastTrades, alpha: float, dt: float = 1.0 ):

        #lob_prev_bids = [ ( quotation_to_decimal(bid.price), bid.quantity) for bid in prev_orderbook.bids]
        #lob_prev_asks = [ ( quotation_to_decimal(ask.price), ask.quantity) for ask in prev_orderbook.asks]
        #lob_curr_bids = [ ( quotation_to_decimal(bid.price), bid.quantity) for bid in curr_orderbook.bids]
        #lob_curr_asks = [ ( quotation_to_decimal(ask.price), ask.quantity) for ask in curr_orderbook.asks]
        trades = [ ( trade[0], trade[1]) for trade in last_trades.trades ]
    
        # Aggregate trades by price to identify consumed liquidity
        trade_vols = {}
        for price, qty in trades:
            trade_vols[price] = trade_vols.get(price, 0.0) + qty

        # Convert tuple lists to dictionaries for fast O(1) lookups
        prev_bids_dict = dict(prev_orderbook.bids)
        curr_bids_dict = dict(prev_orderbook.asks)
        prev_asks_dict = dict(curr_orderbook.bids)
        curr_asks_dict = dict(curr_orderbook.asks)

        # Extract and sort current book prices to establish the dynamic depth hierarchy
        # Bids descending (highest first), Asks ascending (lowest first)
        curr_bid_prices = sorted(curr_bids_dict.keys(), reverse=True)
        curr_ask_prices = sorted(curr_asks_dict.keys())

        # Helper function to find the effective depth level of any price
        def get_level(price, sorted_curr_prices, is_bid):
            level = 0
            for p in sorted_curr_prices:
                if is_bid and p > price:
                    level += 1
                elif not is_bid and p < price:
                    level += 1
                elif p == price:
                    return level
                else:
                    break # Optimization: stop searching once we pass the price tier
            return level

        # Helper function to compute weighted arrivals for a specific side
        def get_weighted_arrivals(prev_dict, curr_dict, sorted_curr_prices, is_bid):
            weighted_arrivals = 0.0
            
            # Evaluate all unique prices that exist in either the previous or current snapshot
            all_prices = set(prev_dict.keys()).union(curr_dict.keys())
            
            for p in all_prices:
                v_curr = curr_dict.get(p, 0.0)
                v_prev = prev_dict.get(p, 0.0)
                
                # Assume trades at this exact price consumed resting liquidity on this side
                t_qty = trade_vols.get(p, 0.0) 
                
                # Net Order Flow = Change in LOB + Executed Trades
                raw_arrival = v_curr - v_prev + t_qty
                
                # If raw_arrival > 0, new limit orders were added
                if raw_arrival > 0:
                    level = get_level(p, sorted_curr_prices, is_bid)
                    weight = math.exp(-alpha * level)
                    weighted_arrivals += raw_arrival * weight
                    
            return weighted_arrivals

        # 3. Compute the weighted arrivals for both sides
        bid_arrivals = get_weighted_arrivals(
            prev_bids_dict, curr_bids_dict, curr_bid_prices, is_bid=True
        )
        
        ask_arrivals = get_weighted_arrivals(
            prev_asks_dict, curr_asks_dict, curr_ask_prices, is_bid=False
        )
        
        # 4. Convert absolute weighted volume to intensity (Rate)
        bid_intensity = bid_arrivals / dt
        ask_intensity = ask_arrivals / dt

        assert not math.isnan(bid_intensity) , "Bid intensity is NaN!"
        assert not math.isnan(ask_intensity) , "Ask intensity is NaN!"
        
        return float(bid_intensity), float(ask_intensity)

    #------------------------------------------------------------------------------------------------------------------------
    # Computes the trend at each position using the past `window_size` prices.
    # The trend is represented by the slope of the linear regression line.
    #    
    #    Args:
    #        prices (list or np.array): Sequence of historical prices.
    #        window_size (int): Number of past values (N) to compute the trend.
    #        
    #    Returns:
    #        np.array: Sequence of trend values (slopes). 
    #                  Positive = Bullish, Negative = Bearish.
    #------------------------------------------------------------------------------------------------------------------------            

    @staticmethod
    def price_to_trends(prices:list, window_size:int):

        prices = np.array(prices)
        n = len(prices)
    
        # Initialize output array with 0.0 for periods where the window is incomplete
        trends = np.full(n, 0.0)
    
        if window_size < 2:
            raise ValueError("Window size must be at least 2 to compute a trend.")
    
        # Precompute X values (time steps) and their variance since they don't change
        x = np.arange(window_size)
        x_mean = np.mean(x)
        x_diff = x - x_mean
        x_var = np.sum(x_diff ** 2)
    
        # Slide the window across the price array
        for i in range(window_size - 1, n):
            # Extract the window of N past prices (including the current price)
            y = prices[i - window_size + 1 : i + 1]
        
            # Calculate the slope (beta) of the linear regression: Cov(x, y) / Var(x)
            y_mean = np.mean(y)
            slope = np.sum(x_diff * (y - y_mean)) / x_var
            normalized_slope = slope / y_mean
        
            trends[i] = normalized_slope
        
        return trends

    #------------------------------------------------------------------------------------------------------------------------
    # Trends to trend regimes 
    #------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def trends_to_trend_regimes(trends: np.ndarray, regime_thresholds:list):

        trend_regimes = [0] * len(trends)

        for i in range(len(trends)):
            trend_regimes[i] = len(regime_thresholds)
            for j in range(len(regime_thresholds)):
                if trends[i] <= regime_thresholds[j]:
                    trend_regimes[i] = j
                    break
    
        return trend_regimes

    #------------------------------------------------------------------------------------------------------------------------
    # Gaussian smoothing
    #------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def generate_gaussian_kernel(size, sigma):
        """
        Generates a 1D Gaussian kernel of a given size and standard deviation.
        """
        # Create an array centered around zero
        # For size=5, this creates [-2, -1, 0, 1, 2]
        center = size // 2
        x = np.arange(0, size) - center
    
        # Apply the Gaussian formula
        kernel = np.exp(-0.5 * (x / sigma) ** 2)
    
        # Normalize the kernel so that it sums to 1
        kernel = kernel / np.sum(kernel)
    
        return kernel

    @staticmethod
    def smooth_with_given_kernel(data_array, gaussian_kernel):
        """
        Applies a pre-defined Gaussian kernel to a 1D NumPy array.
        """
        # Ensure both inputs are numpy arrays
        data_array = np.asarray(data_array)
        gaussian_kernel = np.asarray(gaussian_kernel)
    
        # Normalize the kernel so the overall amplitude of the signal doesn't change
        kernel_normalized = gaussian_kernel / np.sum(gaussian_kernel)
    
        # Apply convolution
        smoothed_array = np.convolve(data_array, kernel_normalized, mode='same')
    
        return smoothed_array