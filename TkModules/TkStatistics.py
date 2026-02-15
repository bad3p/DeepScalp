
import sys
import os.path
import numpy as np
import random
import math
from decimal import Decimal
from tinkoff.invest.schemas import Quotation
from tinkoff.invest.utils import quotation_to_decimal
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
    def orderbook_spread(orderbook : GetOrderBookResponse, orderbook_width : int, min_price_increment : float):
        
        max_bid_price = 0.0
        if len(orderbook.bids) > 0:
            for bid in orderbook.bids:
                max_bid_price = max( max_bid_price, quotation_to_float( bid.price ) )                

        min_ask_price = 10e10
        if len(orderbook.asks) > 0:
            for ask in orderbook.asks:
                min_ask_price = min( min_ask_price, quotation_to_float( ask.price ) )

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
            
            for i in range(len(distribution)):
                if descriptor[i] < left_mean:
                    leftmost_mean += descriptor[i] * distribution[i]
                    leftmost_bin_weight += distribution[i]
                elif descriptor[i] > right_mean:
                    rightmost_mean += descriptor[i] * distribution[i]
                    rightmost_bin_weight += distribution[i]

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
    def get_min_price_increment(orderbook : GetOrderBookResponse, default_value:Decimal):

        def get_min_price_diff(prices:list):
            min_diff = Decimal(99999999999)
            for i in range(len(prices) - 1):
                current_diff = prices[i+1] - prices[i]
                if current_diff < min_diff:
                    min_diff = current_diff
            return min_diff

        bid_prices = [ quotation_to_decimal( bid.price ) for bid in orderbook.bids ]        
        ask_prices = [ quotation_to_decimal( ask.price ) for ask in orderbook.asks ]        

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
    def orderbook_to_tensor(orderbook : GetOrderBookResponse, orderbook_width : int, min_price_increment : float):

        def almost_equal(a, b, rel_tol=1e-09, abs_tol=1e-06):
            return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)
        
        def discrete_imbalance(val):
            return 0 if almost_equal(val,0.0) else ( -1 if val < 0.0 else 1 )            

        pivot_price = 0.0

        if len(orderbook.bids) > 0:
            for bid in orderbook.bids:
                pivot_price = max( pivot_price, quotation_to_float( bid.price ) )
        else:
            pivot_price = quotation_to_float( orderbook.last_price )
            for ask in orderbook.asks:
                pivot_price = min( pivot_price, quotation_to_float( ask.price ) )

        bid_price = [ pivot_price - min_price_increment * i for i in range(int(orderbook_width/2))]
        ask_price = [ pivot_price + min_price_increment * i for i in range(1,int(orderbook_width/2)+1)]

        bid_delta_price = [ math.log( price / pivot_price ) for price in bid_price]
        ask_delta_price = [ math.log( price / pivot_price ) for price in ask_price]

        total_bid_volume = 0        

        bid_volume = np.empty( len(bid_price), dtype=float)
        bid_volume.fill(0)
        
        for bid in orderbook.bids:
            price = quotation_to_float( bid.price )
            if price > pivot_price:
                print( 'Bid overlapping asks: ', pivot_price, price )
                continue
            index = min( int( round( (pivot_price - price) / min_price_increment ) ), int(orderbook_width/2)-1 )
            assert almost_equal(price, bid_price[index]) if index < int(orderbook_width/2)-1 else True , "Bid index mismatch: " + str(price) + " : " + str(bid_price[index])
            total_bid_volume = total_bid_volume + bid.quantity
            bid_volume[index] = bid_volume[index] + bid.quantity

        total_ask_volume = 0            

        ask_volume = np.empty( len(ask_price), dtype=float)
        ask_volume.fill(0) 
        
        for ask in orderbook.asks:
            price = quotation_to_float( ask.price )
            if price <= pivot_price:
                print( 'Ask overlapping bids: ', pivot_price, price )
                continue
            index = min( int( round( (price - pivot_price) / min_price_increment ) - 1 ), int(orderbook_width/2)-1 )
            assert almost_equal(price, ask_price[index]) if index < int(orderbook_width/2)-1 else True, "Ask index mismatch: " + str(price) + " : " + str(ask_price[index])
            total_ask_volume = total_ask_volume + ask.quantity
            ask_volume[index] = ask_volume[index] + ask.quantity

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

        result_tensor = np.stack([delta_price_tensor, bid_volume_tensor, normalized_bid_volume_tensor, ask_volume_tensor, normalized_ask_volume_tensor, imbalance_tensor, discrete_imbalance_tensor], axis=1)

        assert not np.isnan(result_tensor).any(), "NaNs in result tensor!"

        hasheable_tensor = np.multiply( delta_price_tensor, bid_volume_tensor, ask_volume_tensor )

        # transpose result_tensor to make it ready to be pytorch-convertible (1,C,W)
        return result_tensor.T, hasheable_tensor, pivot_price, (total_bid_volume + total_ask_volume)

    #------------------------------------------------------------------------------------------------------------------------
    # For the given list of anonymized trades, the method returns distrubution of order volumes,
    # * pivoted around given price
    # * with discretization proportional to given min_price_increment
    # * with optional time threshold allowing to ignore events older than the certain time
    # Additionally the method measures the means of distribution tails, given the order "measure_distribution_tail_order"
    #------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def last_trades_to_tensor(trades_with_time_threshold : list, pivot_price : float, distribution_width : int, min_price_increment : float, measure_distribution_tail_order=1):

        def almost_equal(a, b, rel_tol=1e-09, abs_tol=1e-06):
            return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

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

        for i in range(len(trades_with_time_threshold)):

            trades = trades_with_time_threshold[i][0]
            time_threshold = trades_with_time_threshold[i][1]

            for trade in trades.trades:
                trade_time = trade.time
                if time_threshold != None and trade_time < time_threshold:
                    continue
                total_event_count = total_event_count + 1
                price = quotation_to_float( trade.price )
                if price <= pivot_price:
                    index = min( int( round( (pivot_price - price) / min_price_increment ) ), int(distribution_width/2)-1 )
                    assert almost_equal(price, bid_price[index]) if index < int(distribution_width/2)-1 else True , "Bid index mismatch: " + str(price) + " : " + str(bid_price[index])
                    total_volume = total_volume + trade.quantity
                    bid_volume[index] = bid_volume[index] + trade.quantity
                else:
                    index = min( int( round( (price - pivot_price) / min_price_increment ) - 1 ), int(distribution_width/2)-1 )
                    assert almost_equal(price, ask_price[index]) if index < int(distribution_width/2)-1 else True, "Ask index mismatch: " + str(price) + " : " + str(ask_price[index])
                    total_volume = total_volume + trade.quantity
                    ask_volume[index] = ask_volume[index] + trade.quantity

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

        result_tensor = np.stack([log_delta_price_tensor, volume_tensor, normalized_volume_tensor], axis=1)

        assert not np.isnan(result_tensor).any(), "NaNs in result tensor!"

        hasheable_tensor = np.multiply( log_delta_price_tensor, volume_tensor )

        left_tail, right_tail = TkStatistics.get_distribution_tail_means( normalized_volume_tensor, abs_delta_price_tensor, measure_distribution_tail_order )

        # transpose result_tensor to make it ready to be pytorch-convertible (1,C,W)
        return result_tensor.T, hasheable_tensor, total_event_count, total_volume, (left_tail, right_tail)
    
    #------------------------------------------------------------------------------------------------------------------------
    # Performs log1p transform with subsequent EMA normalization for the given sequence
    #------------------------------------------------------------------------------------------------------------------------

    def log_ema_normalize(sequence, half_life=250, eps=1e-6):

        sequence = np.asarray(sequence, dtype=np.float64)

        # log transform
        log_sequence = np.log1p(sequence)

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

    def log_vol_ema_normalize(sequence, half_life=250, eps=1e-6):

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