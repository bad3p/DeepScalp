
from __future__ import annotations

import os
import argparse
import csv
import heapq
from os import path
import random
import datetime
from os.path import join
from pathlib import Path

from collections import Counter, OrderedDict
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Optional

import configparser
import json
from unittest import result

from networkx import volume
from t_tech.invest.constants import INVEST_GRPC_API
from t_tech.invest import (
    Client,
    InstrumentType,
    InstrumentIdType,
    SecurityTradingStatus,
    GetOrderBookResponse,
)

from TkModules.TkIO import TkIO
from TkModules.TkPersistentQueue import TkPersistentQueue
from TkModules.TkInstrument import TkOrderbook, TkLastTrades

FIELDS = (
    "NO",
    "SECCODE",
    "BUYSELL",
    "TIME",
    "ORDERNO",
    "ACTION",
    "PRICE",
    "VOLUME",
    "TRADENO",
    "TRADEPRICE",
)


# ======================================================
# MOEX TIME PARSING
# ======================================================

def parse_moex_time(
    value: str,
    precision: str = "auto",
) -> tuple[int, str]:
    """
    Convert MOEX TIME into microseconds since midnight.

    Supported formats:

        HHMMSSZZZ       milliseconds
        HHMMSSZZZXXX    microseconds

    AUTO:
        Up to 9 digits  -> milliseconds
        More than 9    -> microseconds

    Returns:
        (time_microseconds_since_midnight, actual_precision)
    """

    raw = str(value).strip()

    if not raw or not raw.isdigit():
        raise ValueError(f"Invalid MOEX TIME: {value!r}")

    if precision == "auto":

        actual_precision = (
            "microseconds" if len(raw) > 9
            else "milliseconds"
        )

    else:
        actual_precision = precision

    if actual_precision == "microseconds":

        digits = raw.zfill(12)

        if len(digits) != 12:
            raise ValueError(f"Invalid microsecond TIME: {value}")

        hh = int(digits[0:2])
        mm = int(digits[2:4])
        ss = int(digits[4:6])

        milliseconds = int(digits[6:9])
        microseconds = int(digits[9:12])

        fraction_us = milliseconds * 1000 + microseconds

    elif actual_precision == "milliseconds":

        digits = raw.zfill(9)

        if len(digits) != 9:
            raise ValueError(f"Invalid millisecond TIME: {value}")

        hh = int(digits[0:2])
        mm = int(digits[2:4])
        ss = int(digits[4:6])

        milliseconds = int(digits[6:9])

        fraction_us = milliseconds * 1000

    else:
        raise ValueError(
            f"Unsupported TIME precision: {actual_precision}"
        )

    if not (
        0 <= hh < 24
        and 0 <= mm < 60
        and 0 <= ss < 60
        and 0 <= fraction_us < 1_000_000
    ):
        raise ValueError(f"TIME out of range: {value}")

    seconds = hh * 3600 + mm * 60 + ss

    return seconds * 1_000_000 + fraction_us, actual_precision


def format_moex_time(time_us: int) -> str:

    seconds, microseconds = divmod(time_us, 1_000_000)

    hh, remainder = divmod(seconds, 3600)
    mm, ss = divmod(remainder, 60)

    return f"{hh:02d}:{mm:02d}:{ss:02d}.{microseconds:06d}"


# ======================================================
# DATA STRUCTURES
# ======================================================

@dataclass(slots=True)
class Order:

    order_no: str
    side: str
    price: Decimal
    volume: int


@dataclass(slots=True)
class Trade:
    """
    One actual economic trade.

    Both MOEX ACTION=2 records are represented by
    this single Trade object.
    """

    seccode: str
    trade_no: str

    time_us: int

    price: Decimal
    volume: int

    buy_order_no: Optional[str] = None
    sell_order_no: Optional[str] = None

    @property
    def time(self) -> str:
        return format_moex_time(self.time_us)

    @property
    def time_seconds(self) -> float:
        return self.time_us / 1_000_000


# ======================================================
# ORDER BOOK
# ======================================================

class OrderBook:

    def __init__(self, seccode: str):

        self.seccode = seccode

        # ORDERNO -> Order
        self.orders: dict[str, Order] = {}

        # Side -> Price -> OrderedDict(ORDERNO -> None)
        self.levels = {
            "B": {},
            "S": {},
        }

        # Lazy heaps for best-price retrieval.
        self.price_heaps = {
            "B": [],
            "S": [],
        }

        self.heap_queued = {
            "B": set(),
            "S": set(),
        }

    # --------------------------------------------------
    # ADD ORDER
    # --------------------------------------------------

    def add_order(
        self,
        order_no: str,
        side: str,
        price: Decimal,
        volume: int,
    ) -> None:

        if side not in ("B", "S"):
            raise ValueError(f"Invalid side: {side}")

        if price <= 0 or volume <= 0:
            return

        if order_no in self.orders:
            self.remove_order(order_no)

        if price not in self.levels[side]:

            self.levels[side][price] = OrderedDict()

            if price not in self.heap_queued[side]:

                heap_price = (
                    -price if side == "B" else price
                )

                heapq.heappush(
                    self.price_heaps[side],
                    heap_price,
                )

                self.heap_queued[side].add(price)

        self.levels[side][price][order_no] = None

        self.orders[order_no] = Order(
            order_no=order_no,
            side=side,
            price=price,
            volume=volume,
        )

    # --------------------------------------------------
    # REMOVE ORDER
    # --------------------------------------------------

    def remove_order(
        self,
        order_no: str,
    ) -> Optional[Order]:

        order = self.orders.pop(order_no, None)

        if order is None:
            return None

        price_level = self.levels[order.side][order.price]

        price_level.pop(order_no, None)

        if not price_level:
            del self.levels[order.side][order.price]

        return order

    # --------------------------------------------------
    # EXECUTE ORDER
    # --------------------------------------------------

    def execute_order(
        self,
        order_no: str,
        executed_volume: int,
    ) -> Optional[int]:

        order = self.orders.get(order_no)

        if order is None:
            return None

        if executed_volume <= 0:
            return 0

        excess = max(0, executed_volume - order.volume)

        if executed_volume >= order.volume:

            self.remove_order(order_no)

        else:

            order.volume -= executed_volume

        return excess

    # --------------------------------------------------
    # CANCEL ORDER
    # --------------------------------------------------

    def cancel_order(
        self,
        order_no: str,
    ) -> Optional[Order]:

        return self.remove_order(order_no)

    # --------------------------------------------------
    # BEST PRICE
    # --------------------------------------------------

    def best_price(
        self,
        side: str,
    ) -> Optional[Decimal]:

        heap = self.price_heaps[side]
        levels = self.levels[side]

        while heap:

            heap_price = heap[0]

            price = (
                -heap_price if side == "B"
                else heap_price
            )

            if price not in levels:

                heapq.heappop(heap)

                self.heap_queued[side].discard(price)

                continue

            return price

        return None

    # --------------------------------------------------
    # BEST ORDER
    # --------------------------------------------------

    def best_order(
        self,
        side: str,
    ) -> Optional[Order]:

        price = self.best_price(side)

        if price is None:
            return None

        queue = self.levels[side][price]

        order_no = next(iter(queue))

        return self.orders[order_no]

    # --------------------------------------------------
    # TOP N PRICE LEVELS
    # --------------------------------------------------

    def top_levels(
        self,
        side: str,
        depth: int = 5,
    ) -> list[tuple[Decimal, int, int]]:

        prices = sorted(
            self.levels[side],
            reverse=(side == "B"),
        )

        result = []

        for price in prices[:depth]:

            queue = self.levels[side][price]

            total_volume = sum(
                self.orders[order_no].volume
                for order_no in queue
            )

            result.append((
                price,
                total_volume,
                len(queue),
            ))

        return result

    # --------------------------------------------------
    # BEST BID / ASK
    # --------------------------------------------------

    def best_bid_ask(self):

        bid = self.best_order("B")
        ask = self.best_order("S")

        return {
            "bid_price": bid.price if bid else None,
            "bid_volume": bid.volume if bid else 0,

            "ask_price": ask.price if ask else None,
            "ask_volume": ask.volume if ask else 0,
        }


# ======================================================
# ORDER LOG REPLAYER
# ======================================================

class OrderLogReplayer:

    def __init__(
        self,
        trade_window_seconds: float = 120.0,
        duplicate_cache_size: int = 200_000,
        time_precision: str = "auto",
    ):

        if trade_window_seconds <= 0:
            raise ValueError("Trade window must be positive")

        self.trade_window_seconds = trade_window_seconds

        self.trade_window_us = int(
            round(trade_window_seconds * 1_000_000)
        )

        self.time_precision = time_precision

        # SECCODE -> OrderBook
        self.books: dict[str, OrderBook] = {}

        # SECCODE -> {TRADENO -> Trade}
        # Rolling window of recent unique economic trades.
        self.trade_tapes: dict[str, dict[str, Trade]] = {}

        # SECCODE -> Trade
        # Last known trade, preserved indefinitely.
        self.last_trade: dict[str, Trade] = {}

        # --------------------------------------------------
        # GLOBAL TRADE EXPIRATION HEAP
        #
        # (expiration_time_us, sequence, SECCODE, TRADENO)
        #
        # Entries expire without scanning all securities.
        # --------------------------------------------------

        self.trade_expiry_heap = []

        self.expiry_sequence = 0

        # Latest observed exchange timestamp.
        self.current_time_us: Optional[int] = None

        # Duplicate execution-record protection.
        #
        # Key = (SECCODE, TRADENO, ORDERNO)
        #
        # The two legitimate trade sides have different
        # ORDERNO values and are both processed.
        self.seen_trade_orders = OrderedDict()

        self.duplicate_cache_size = duplicate_cache_size

        self.stats = Counter()

    # --------------------------------------------------
    # GET ORDER BOOK
    # --------------------------------------------------

    def get_book(self, seccode: str) -> OrderBook:

        if seccode not in self.books:

            self.books[seccode] = OrderBook(seccode)

        return self.books[seccode]

    # --------------------------------------------------
    # ADVANCE EXCHANGE CLOCK
    # --------------------------------------------------

    def advance_time(
        self,
        event_time_us: int,
    ) -> None:
        """
        Maintain a monotonically increasing event-time
        watermark and expire old trades.

        Sequence number/order of CSV rows remains the
        authoritative order for book reconstruction.
        """

        if self.current_time_us is None:

            self.current_time_us = event_time_us

        elif event_time_us > self.current_time_us:

            self.current_time_us = event_time_us

        elif event_time_us < self.current_time_us:

            self.stats["time_regressions"] += 1

        self.expire_trades(self.current_time_us)

    # --------------------------------------------------
    # EXPIRE OLD TRADES
    # --------------------------------------------------

    def expire_trades(
        self,
        now_us: int,
    ) -> None:
        """
        Remove trades older than the configured window.

        The window is inclusive at its lower boundary:
            trade_time >= now - window
        """

        while (
            self.trade_expiry_heap
            and self.trade_expiry_heap[0][0] < now_us
        ):

            (
                expiry_us,
                _,
                seccode,
                trade_no,
            ) = heapq.heappop(self.trade_expiry_heap)

            tape = self.trade_tapes.get(seccode)

            if tape is None:
                continue

            trade = tape.get(trade_no)

            if trade is None:
                continue

            expected_expiry = (
                trade.time_us + self.trade_window_us
            )

            # Protect against an old expiration entry
            # affecting a newer trade with the same ID.
            if expected_expiry == expiry_us:

                del tape[trade_no]

                self.stats["trades_expired"] += 1

    # --------------------------------------------------
    # RECORD ONE ECONOMIC TRADE
    # --------------------------------------------------

    def record_trade(
        self,
        seccode: str,
        trade_no: str,
        order_no: str,
        side: str,
        event_time_us: int,
        price: Decimal,
        volume: int,
    ) -> None:

        if not trade_no:
            self.stats["trades_missing_trade_no"] += 1
            return

        if volume <= 0 or price <= 0:
            self.stats["trades_invalid_fields"] += 1
            return

        tape = self.trade_tapes.setdefault(seccode, {})

        # --------------------------------------------------
        # CHECK WHETHER THIS TRADE HAS ALREADY BEEN SEEN
        # --------------------------------------------------

        existing = tape.get(trade_no)

        # The trade might have expired from the rolling
        # tape but still be retained as last_trade.
        if existing is None:

            last = self.last_trade.get(seccode)

            if last is not None and last.trade_no == trade_no:
                existing = last

        # --------------------------------------------------
        # SECOND SIDE OF AN EXISTING TRADE
        # --------------------------------------------------

        if existing is not None:

            self.stats["trade_tape_second_legs"] += 1

            if existing.price != price or existing.volume != volume:
                self.stats["trade_pair_mismatches"] += 1

            if existing.time_us != event_time_us:
                self.stats["trade_pair_time_mismatches"] += 1

            was_complete = (
                existing.buy_order_no is not None
                and existing.sell_order_no is not None
            )

            if side == "B":

                if existing.buy_order_no is None:
                    existing.buy_order_no = order_no

                elif existing.buy_order_no != order_no:
                    self.stats["trade_pair_side_conflicts"] += 1

            elif side == "S":

                if existing.sell_order_no is None:
                    existing.sell_order_no = order_no

                elif existing.sell_order_no != order_no:
                    self.stats["trade_pair_side_conflicts"] += 1

            else:
                self.stats["trade_pair_invalid_sides"] += 1

            is_complete = (
                existing.buy_order_no is not None
                and existing.sell_order_no is not None
            )

            if is_complete and not was_complete:
                self.stats["trade_pairs_completed"] += 1

            return

        # --------------------------------------------------
        # FIRST SIDE OF A NEW ECONOMIC TRADE
        # --------------------------------------------------

        trade = Trade(
            seccode=seccode,
            trade_no=trade_no,
            time_us=event_time_us,
            price=price,
            volume=volume,
        )

        if side == "B":
            trade.buy_order_no = order_no

        elif side == "S":
            trade.sell_order_no = order_no

        else:
            self.stats["trade_pair_invalid_sides"] += 1

        # --------------------------------------------------
        # UPDATE LAST TRADE
        # --------------------------------------------------

        previous_last = self.last_trade.get(seccode)

        if (
            previous_last is None
            or event_time_us >= previous_last.time_us
        ):

            # This reference is deliberately NOT removed
            # when the rolling trade tape expires.
            self.last_trade[seccode] = trade

            self.stats["last_trade_updates"] += 1

        # --------------------------------------------------
        # UPDATE ROLLING TRADE TAPE
        # --------------------------------------------------

        # The persistent last_trade is updated independently
        # of rolling-window membership.

        if (
            self.current_time_us is not None
            and event_time_us
            < self.current_time_us - self.trade_window_us
        ):

            self.stats["trade_rows_outside_window"] += 1
            return

        tape[trade_no] = trade

        expiration_us = (
            event_time_us + self.trade_window_us
        )

        heapq.heappush(
            self.trade_expiry_heap,
            (
                expiration_us,
                self.expiry_sequence,
                seccode,
                trade_no,
            ),
        )

        self.expiry_sequence += 1

        self.stats["economic_trades_recorded"] += 1

    # --------------------------------------------------
    # GET RECENT TRADES
    # --------------------------------------------------

    def get_recent_trades(
        self,
        seccode: str,
        limit: Optional[int] = None,
        newest_first: bool = False,
    ) -> list[Trade]:
        """
        Return unique economic trades for one security.

        Trades are chronological by default.

        newest_first=True:
            Newest trades come first.

        limit:
            Optional maximum number of returned trades.
        """

        if limit is not None and limit < 0:

            raise ValueError("limit cannot be negative")

        if self.current_time_us is not None:

            self.expire_trades(self.current_time_us)

        tape = self.trade_tapes.get(seccode)

        if not tape or limit == 0:

            return []

        result = sorted(
            tape.values(),
            key=lambda trade: (
                trade.time_us,
                trade.trade_no,
            ),
            reverse=newest_first,
        )

        if limit is not None:

            result = result[:limit]

        return result

    # --------------------------------------------------
    # GET TRADE COUNT
    # --------------------------------------------------

    def get_recent_trade_count(
        self,
        seccode: str,
    ) -> int:

        tape = self.trade_tapes.get(seccode)

        return len(tape) if tape else 0

    # --------------------------------------------------
    # DUPLICATE TRADE-ORDER PROTECTION
    # --------------------------------------------------

    def remember_trade_order(
        self,
        key: tuple,
    ) -> bool:

        if key in self.seen_trade_orders:

            self.seen_trade_orders.move_to_end(key)

            return False

        self.seen_trade_orders[key] = None

        if (
            len(self.seen_trade_orders)
            > self.duplicate_cache_size
        ):

            self.seen_trade_orders.popitem(last=False)

        return True

    # --------------------------------------------------
    # PROCESS ONE CSV ROW
    # --------------------------------------------------

    def process(self, row: dict) -> None:

        self.stats["rows"] += 1

        seccode = row["SECCODE"]
        side = row["BUYSELL"]
        order_no = row["ORDERNO"]

        if not order_no:

            self.stats["missing_order_id"] += 1

            return

        if side not in ("B", "S"):

            raise ValueError(f"Invalid BUYSELL: {side}")

        action = int(row["ACTION"])

        try:

            event_time_us, actual_precision = parse_moex_time(
                row["TIME"],
                self.time_precision,
            )

        except ValueError as exc:

            raise ValueError(
                f"Invalid TIME at row NO={row['NO']}: {exc}"
            ) from exc

        self.stats["time_rows_parsed"] += 1
        self.stats[f"time_precision_{actual_precision}"] += 1

        self.advance_time(event_time_us)

        try:

            price = Decimal(row["PRICE"] or "0")
            volume = int(row["VOLUME"] or "0")

        except (InvalidOperation, ValueError) as exc:

            raise ValueError(
                f"Invalid price/volume at NO={row['NO']}"
            ) from exc

        book = self.get_book(seccode)

        # ==============================================
        # ACTION 1: ORDER PLACEMENT
        # ==============================================

        if action == 1:

            if price <= 0:

                self.stats["market_order_placements"] += 1

                return

            if order_no in book.orders:

                self.stats["replaced_order_ids"] += 1

            book.add_order(
                order_no=order_no,
                side=side,
                price=price,
                volume=volume,
            )

            self.stats["orders_placed"] += 1

        # ==============================================
        # ACTION 0: CANCELLATION
        # ==============================================

        elif action == 0:

            existing = book.orders.get(order_no)

            if existing is None:

                self.stats["untracked_cancellations"] += 1

                return

            # ACTION=0 reports remaining visible quantity.
            if existing.volume != volume:

                self.stats["cancel_volume_mismatches"] += 1

            book.cancel_order(order_no)

            self.stats["orders_canceled"] += 1

        # ==============================================
        # ACTION 2: TRADE EXECUTION
        # ==============================================

        elif action == 2:

            self.stats["trade_rows"] += 1

            trade_no = row["TRADENO"]

            # ------------------------------------------
            # Row-level duplicate detection.
            #
            # Do NOT deduplicate by TRADENO alone:
            # buyer and seller must both be processed.
            # ------------------------------------------

            if trade_no:

                key = (
                    seccode,
                    trade_no,
                    order_no,
                )

                if not self.remember_trade_order(key):

                    self.stats["duplicate_trade_rows"] += 1

                    return

            try:

                trade_price = Decimal(
                    row["TRADEPRICE"] or "0"
                )

            except InvalidOperation:

                trade_price = Decimal("0")

            if not trade_price.is_finite():

                trade_price = Decimal("0")

            # ------------------------------------------
            # Unique economic trade tape.
            # This happens once for each TRADENO.
            # ------------------------------------------

            self.record_trade(
                seccode=seccode,
                trade_no=trade_no,
                order_no=order_no,
                side=side,
                event_time_us=event_time_us,
                price=trade_price,
                volume=volume,
            )

            # ------------------------------------------
            # Independent order-book execution.
            #
            # Both legitimate trade rows must be
            # processed, even though the trade tape
            # stores only one Trade object.
            # ------------------------------------------

            existing = book.orders.get(order_no)

            if existing is None:

                self.stats["non_resting_trade_rows"] += 1

                return

            if existing.side != side:

                self.stats["side_mismatches"] += 1

            excess = book.execute_order(
                order_no=order_no,
                executed_volume=volume,
            )

            if excess is not None and excess > 0:

                self.stats["execution_overruns"] += 1

            self.stats["executions_applied"] += 1

        else:

            raise ValueError(f"Unknown ACTION={action}")


# ======================================================
# CSV STREAMING
# ======================================================

def iter_orderlog(
    filename: str,
    encoding: str = "utf-8-sig",
    delimiter: Optional[str] = None,
):

    with open(
        filename,
        "r",
        newline="",
        encoding=encoding,
    ) as f:

        sample = f.read(65536)

        f.seek(0)

        if delimiter is None:

            try:

                delimiter = csv.Sniffer().sniff(
                    sample,
                    delimiters=",;\t",
                ).delimiter

            except csv.Error:

                delimiter = ","

        reader = csv.reader(
            f,
            delimiter=delimiter,
        )

        first = next(reader, None)

        if first is None:
            return

        has_header = (
            tuple(x.strip().upper() for x in first)
            == FIELDS
        )

        if not has_header:

            if len(first) != len(FIELDS):

                raise ValueError("Invalid first CSV row")

            yield dict(zip(
                FIELDS,
                (x.strip() for x in first),
            ))

        for raw in reader:

            if not raw or not any(x.strip() for x in raw):
                continue

            if len(raw) != len(FIELDS):

                raise ValueError(
                    f"Invalid CSV row {reader.line_num}: "
                    f"expected {len(FIELDS)} fields, "
                    f"received {len(raw)}"
                )

            yield dict(zip(
                FIELDS,
                (x.strip() for x in raw),
            ))


# ======================================================
# REPLAY
# ======================================================

def replay_orderlog(
    filename: str,
    trade_window_seconds: float = 120.0,
    encoding: str = "utf-8-sig",
    delimiter: Optional[str] = None,
    callback_interval_mean: float = 1.0,
    callback_interval_std_dev: float = 0.0,
    callback: Optional[callable] = None,
    time_precision: str = "auto"
) -> OrderLogReplayer:

    replayer = OrderLogReplayer(
        trade_window_seconds=trade_window_seconds,
        time_precision=time_precision,
    )

    callback_time = None

    for row in iter_orderlog(
        filename,
        encoding=encoding,
        delimiter=delimiter,
    ):

        replayer.process(row)

        if not callback_time is None and replayer.current_time_us >= callback_time:
            callback( replayer )
            callback_time = None

        if callback_time is None:
            dt = random.gauss( callback_interval_mean, callback_interval_std_dev )
            dt_microseconds = int( dt * 1_000_000 )
            callback_time = replayer.current_time_us + dt_microseconds

    return replayer

# ======================================================
# FILENAME PARSER
# ======================================================

def filename_to_date(path: str):
    file_path = Path(path)
    file_stem = file_path.stem
    date_part = file_stem[-8:]
    parsed_date = datetime.datetime.strptime(date_part, "%Y%d%m")
    return parsed_date

# ======================================================
# EXAMPLE ENTRY POINT
# ======================================================

def main():

    TOKEN = os.environ["TK_TOKEN"]
    with Client(TOKEN, target=INVEST_GRPC_API) as client:

        filename = "E:\\MOEX\\OrderLog20241001.txt"
        filedate = filename_to_date(filename)
        print( filedate.strftime("%B_%d_%Y") )

        config = configparser.ConfigParser()
        config.read( 'TkConfig.ini' )

        data_path = config['Paths']['DataPath']
        gather_data_queue_filename = config['Paths']['GatherDataQueueFileName']
        market_data_file_extension = config['Paths']['MarketDataFileExtension']
        orderbook_depth = int(config['GatherData']['OrderBookDepth'])
        last_trades_period_in_minutes = int(config['GatherData']['LastTradesPeriodInMinutes'])
        ignore_tickers = json.loads(config['GatherData']['IgnoreTickers'])
        regular_iteration_time_mean = float(config['GatherData']['RegularIterationTimeMean'])
        regular_iteration_time_std_dev = float(config['GatherData']['RegularIterationTimeStdDev'])    

        encoding = "utf-8-sig"
        delimiter = ","

        def init_gather_data_queue_callback():
            raise RuntimeError('Global gather queue is missing!')
            return []

        gather_data_queue_path = join( data_path, gather_data_queue_filename )
        queue = TkPersistentQueue( gather_data_queue_path, init_gather_data_queue_callback )
        print( len(queue._queue), "tickers in gather queue" )

        def capture_sample(order_log_replayer: OrderLogReplayer) -> None:   
            global samples_captured     
            ticker = queue.pop()
            if ticker in order_log_replayer.books.keys():

                last_trades_sample = TkLastTrades(source=None)
                if ticker in order_log_replayer.trade_tapes:                        
                    trades = order_log_replayer.trade_tapes[ticker]
                    for trade_id in trades.keys():
                        if not trades[trade_id].buy_order_no is None and not trades[trade_id].sell_order_no is None:
                            dir = -1 if trades[trade_id].buy_order_no < trades[trade_id].sell_order_no else 1
                        else:
                            dir = -1 if trades[trade_id].sell_order_no is None else 1
                        time = filedate + datetime.timedelta( microseconds=trades[trade_id].time_us )
                        price = trades[trade_id].price
                        volume = trades[trade_id].volume
                        last_trades_sample.trades.append( (price, volume, time, dir) )

                book = order_log_replayer.books[ticker]
                bids = book.top_levels("B", orderbook_depth)
                asks = book.top_levels("S", orderbook_depth)                
                orderbook_sample = TkOrderbook(orderbook=None)
                orderbook_sample.bids = [ (b[0],b[1]) for b in bids]
                orderbook_sample.asks = [ (a[0],a[1]) for a in asks]
                orderbook_sample.orderbook_ts = filedate + datetime.timedelta( microseconds=order_log_replayer.current_time_us )                
                if ticker in order_log_replayer.last_trade:
                    last_trade = order_log_replayer.last_trade[ticker]
                    orderbook_sample.close_price = last_trade.price
                    orderbook_sample.close_price_ts = filedate + datetime.timedelta( microseconds=last_trade.time_us )
                    orderbook_sample.last_price = orderbook_sample.close_price
                    orderbook_sample.last_price_ts = orderbook_sample.close_price_ts
                    orderbook_sample.limit_up = last_trade.price
                    orderbook_sample.limit_down = last_trade.price
                else:
                    if len(orderbook_sample.bids) > 0:
                        orderbook_sample.close_price = orderbook_sample.bids[0][0]
                        orderbook_sample.close_price_ts = orderbook_sample.orderbook_ts
                        orderbook_sample.last_price = orderbook_sample.close_price
                        orderbook_sample.last_price_ts = orderbook_sample.close_price_ts
                        orderbook_sample.limit_up = orderbook_sample.close_price
                        orderbook_sample.limit_down = orderbook_sample.close_price
                    elif len(orderbook_sample.asks) > 0:
                        orderbook_sample.close_price = orderbook_sample.asks[0][0]
                        orderbook_sample.close_price_ts = orderbook_sample.orderbook_ts
                        orderbook_sample.last_price = orderbook_sample.close_price
                        orderbook_sample.last_price_ts = orderbook_sample.close_price_ts
                        orderbook_sample.limit_up = orderbook_sample.close_price
                        orderbook_sample.limit_down = orderbook_sample.close_price

                output_filename = ticker + "_" + filedate.strftime("%B_%d_%Y") + "_" + "Day" + market_data_file_extension
                output_path = join( data_path, output_filename )
                TkIO.append_at_path( output_path, orderbook_sample )
                TkIO.append_at_path( output_path, last_trades_sample )

                if random.random() < 0.01:
                    print(f"\rTimestamp : {orderbook_sample.orderbook_ts}, Ticker: {ticker}            ", end="", flush=True)

            queue.push(ticker)
    
        replayer = replay_orderlog(
            filename,
            encoding = encoding,
            delimiter = delimiter,
            trade_window_seconds = last_trades_period_in_minutes * 60.0,
            callback_interval_mean = regular_iteration_time_mean,
            callback_interval_std_dev = regular_iteration_time_std_dev,
            callback=capture_sample,
            time_precision = "microseconds"
        )

        print("\nReplay statistics:")

        for key, value in sorted(replayer.stats.items()):
            print(f"{key:32s}: {value}")

        print("\nReconstructed order books:")

        for seccode in sorted(replayer.books):
            book = replayer.books[seccode]
            print(f"\n{seccode}")
            print(f"Active orders: {len(book.orders)}")
            print("Bids:", book.top_levels("B", orderbook_depth))
            print("Asks:", book.top_levels("S", orderbook_depth))

if __name__ == "__main__":
    main()
