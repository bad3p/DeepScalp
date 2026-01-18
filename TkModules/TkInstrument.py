import configparser
from decimal import Decimal
from datetime import date, datetime, timezone, timedelta
from collections import defaultdict
from configparser import ConfigParser
from tinkoff.invest import Client
from tinkoff.invest import InstrumentIdType
from tinkoff.invest import InstrumentType
from tinkoff.invest import OrderDirection
from tinkoff.invest import OrderType
from tinkoff.invest.utils import decimal_to_quotation, quotation_to_decimal, money_to_decimal
from TkModules.TkIO import TkIO

#------------------------------------------------------------------------------------------------------------------------
# Investment instrument wrapper
#------------------------------------------------------------------------------------------------------------------------

class TkInstrument():

    def __init__(self, _client : Client, _cfg : ConfigParser, _type : InstrumentType, _ticker : str, _classCode : str):
        self._client = _client
        self._cfg = _cfg
        self._type = _type
        self._ticker = _ticker
        self._classCode = _classCode

        instrumentPath = _cfg['Paths']['InstrumentsPath'] + _ticker + _cfg['Paths']['InstrumentFileExtension']
        instruments = TkIO.read_at_path( instrumentPath )

        if len(instruments) == 0:
            instruments = self._client.instruments.find_instrument( query=_ticker, instrument_kind=_type ).instruments
            instruments = [instrument for instrument in instruments if instrument.ticker == _ticker]
            instruments = [instrument for instrument in instruments if instrument.class_code == _classCode]
            if len(instruments) == 0:
                raise RuntimeError('Unable to find instrument with given ticker and type: ' + _ticker + ', ' + _classCode)
            if len(instruments) > 1:
                raise RuntimeError('Multiple instruments found with given ticker and type: ' + _ticker + ', ' + _classCode)
            self._instrument = instruments[0]
            TkIO.write_at_path( instrumentPath, self._instrument )
        else:
            self._instrument = instruments[0]

        self._share = None
        self._bond = None
        self._etf = None

        if self._type == InstrumentType.INSTRUMENT_TYPE_SHARE:
            sharePath = _cfg['Paths']['InstrumentsPath'] + _ticker + _cfg['Paths']['ShareFileExtension']
            shares = TkIO.read_at_path( sharePath )
            if len(shares) == 0:
                self._share = self._client.instruments.share_by( id_type = InstrumentIdType.INSTRUMENT_ID_TYPE_FIGI, id = self._instrument.figi )
                TkIO.write_at_path( sharePath, self._share )
            else:
                self._share = shares[0]
        elif self._type == InstrumentType.INSTRUMENT_TYPE_BOND:
            bondPath = _cfg['Paths']['InstrumentsPath'] + _ticker + _cfg['Paths']['BondFileExtension']
            bonds = TkIO.read_at_path( bondPath )
            if len(bonds) == 0:
                self._bond = self._client.instruments.bond_by( id_type = InstrumentIdType.INSTRUMENT_ID_TYPE_FIGI, id = self._instrument.figi )
                TkIO.writeToFile( bondPath, self._bond )
            else:
                self._bond = bonds[0]
        elif self._type == InstrumentType.INSTRUMENT_TYPE_ETF:
            etfPath = _cfg['Paths']['InstrumentsPath'] + _ticker + _cfg['Paths']['EtfFileExtension']
            etfs = TkIO.write_at_path( etfPath )
            if len(etfs) == 0:
                self._etf = self._client.instruments.etf_by( id_type = InstrumentIdType.INSTRUMENT_ID_TYPE_FIGI, id = self._instrument.figi )
                TkIO.write_at_path( etfPath, self._etf )
            else:
                self._etf = etfs[0]

    def trading_status(self):
        statusResponse = self._client.market_data.get_trading_statuses( instrument_ids=[self._instrument.figi] ).trading_statuses[0]
        return statusResponse.trading_status

    def uid(self):        
        return self._instrument.uid
    
    def ticker(self):
        return self._ticker

    def figi(self):
        if self._type == InstrumentType.INSTRUMENT_TYPE_SHARE:
            return self._share.instrument.figi
        elif self._type == InstrumentType.INSTRUMENT_TYPE_BOND:
            return self._bond.instrument.figi
        elif self._type == InstrumentType.INSTRUMENT_TYPE_ETF:
            return self._etf.instrument.figi            
        else:
            raise RuntimeError('Unsupported instrument type')

    def sector(self):
        if self._type == InstrumentType.INSTRUMENT_TYPE_SHARE:
            return self._share.instrument.sector
        elif self._type == InstrumentType.INSTRUMENT_TYPE_BOND:
            return self._bond.instrument.sector
        else:
            raise RuntimeError('Unsupported instrument type')

    def lot(self):
        if self._type == InstrumentType.INSTRUMENT_TYPE_SHARE:
            return int(self._share.instrument.lot)
        elif self._type == InstrumentType.INSTRUMENT_TYPE_BOND:
            return int(self._bond.instrument.lot)
        else:
            raise RuntimeError('Unsupported instrument type')

    def min_price_increment(self):
        if self._type == InstrumentType.INSTRUMENT_TYPE_SHARE:
            return self._share.instrument.min_price_increment
        elif self._type == InstrumentType.INSTRUMENT_TYPE_BOND:
            return self._bond.instrument.min_price_increment
        elif self._type == InstrumentType.INSTRUMENT_TYPE_ETF:
            return self._etf.instrument.min_price_increment
        else:
            raise RuntimeError('Unsupported instrument type')

    def get_order_book(self, depth : int):
        return self._client.market_data.get_order_book( figi = self._instrument.figi, depth = depth )

    def get_last_trades(self, _from, _to, _tradeSourceType):
        return self._client.market_data.get_last_trades( figi = self._instrument.figi, from_=_from, to=_to, trade_source=_tradeSourceType)
    
    def get_last_price(self):
        result = self._client.market_data.get_last_prices(figi=[self.figi()]).last_prices[0].price
        return quotation_to_decimal(result)
    
    def post_buy_market_order(self,account_id:str, count:int, price:Decimal): 
        response = self._client.orders.post_order(
            figi=self.figi(),
            quantity=count,
            price=decimal_to_quotation(price),
            direction=OrderDirection.ORDER_DIRECTION_BUY,
            account_id=account_id,
            order_type=OrderType.ORDER_TYPE_MARKET
        )
        return response
    
    def post_sell_limit_order(self,account_id:str, count:int, price:Decimal): 
        response = self._client.orders.post_order(
            figi=self.figi(),
            quantity=count,
            price=decimal_to_quotation(price),
            direction=OrderDirection.ORDER_DIRECTION_SELL,
            account_id=account_id,
            order_type=OrderType.ORDER_TYPE_LIMIT
        )
        return response
    

    #------------------------------------------------------------------------------------------------------------------------    
    # Static helpers
    #------------------------------------------------------------------------------------------------------------------------    

    @staticmethod
    def get_instrument_tickers(client, instrument_kind, class_code):
        getAssetsResponse = client.instruments.get_assets(None)
        assetsWithInstruments = [asset for asset in getAssetsResponse.assets if len(asset.instruments) >= 1 ]
        filteredAssets = [asset for asset in assetsWithInstruments if asset.instruments[0].instrument_kind == instrument_kind and asset.instruments[0].class_code == class_code]        
        return [asset.instruments[0].ticker for asset in filteredAssets]        

    #------------------------------------------------------------------------------------------------------------------------    
    # File helpers
    # Orderbook file name convention: TICKER_Month_Day_Year_Anchor.obs
    # Anchor = {Day|Evening}
    #------------------------------------------------------------------------------------------------------------------------    

    @staticmethod
    def date_from_filename(filename:str): 
        date_str = filename[ filename.find("_") + 1: ]
        date_str = date_str[ 0 : date_str.find(".") ]
        date_anchor = date_str[ date_str.rfind("_") + 1: ]
        date_str = date_str[ 0 : date_str.rfind("_") ]
        result = datetime.strptime( date_str,'%B_%d_%Y' )
        if date_anchor == 'Evening':
            result = result + timedelta(hours=19)
        else:
            result = result + timedelta(hours=10)
        return result

    @staticmethod
    def ticker_from_filename(filename:str):
        return filename[ 0: filename.find("_") ]

    @staticmethod
    def group_by_ticker(filenames:list):
        result = defaultdict(list)
        for filename in filenames:
            ticker = TkInstrument.ticker_from_filename(filename)
            date = TkInstrument.date_from_filename(filename)
            result[ticker].append( (date, filename) )

        for key in result:
            files = result[key]
            files = sorted( files, key=lambda x: x[0] )
            result[key] = files

        return result