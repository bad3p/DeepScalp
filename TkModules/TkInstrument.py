import configparser
import jsonpickle
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
        instrumentData = TkIO.read_at_path( instrumentPath )

        if len( instrumentData ) != 0:
            self._data = instrumentData[0]
        else:
            instruments = self._client.instruments.find_instrument( query=_ticker, instrument_kind=_type ).instruments
            instruments = [instrument for instrument in instruments if instrument.ticker == _ticker]
            instruments = [instrument for instrument in instruments if instrument.class_code == _classCode]
            if len(instruments) == 0:
                raise RuntimeError('Unable to find instrument with given ticker and type: ' + _ticker + ', ' + _classCode)
            if len(instruments) > 1:
                raise RuntimeError('Multiple instruments found with given ticker and type: ' + _ticker + ', ' + _classCode)
            
            self._data = {}
            self._data['api_trade_available_flag'] = instruments[0].api_trade_available_flag
            self._data['blocked_tca_flag'] = instruments[0].blocked_tca_flag
            self._data['class_code'] = instruments[0].class_code
            self._data['figi'] = instruments[0].figi
            self._data['first_1day_candle_date'] = instruments[0].first_1day_candle_date
            self._data['first_1min_candle_date'] = instruments[0].first_1min_candle_date
            self._data['for_iis_flag'] = instruments[0].for_iis_flag
            self._data['for_qual_investor_flag'] = instruments[0].for_qual_investor_flag
            self._data['instrument_kind'] = instruments[0].instrument_kind
            self._data['instrument_type'] = instruments[0].instrument_type
            self._data['isin'] = instruments[0].isin
            self._data['lot'] = instruments[0].lot
            self._data['name'] = instruments[0].name
            self._data['position_uid'] = instruments[0].position_uid
            self._data['ticker'] = instruments[0].ticker
            self._data['uid'] = instruments[0].uid
            self._data['weekend_flag'] = instruments[0].weekend_flag

            if self._type == InstrumentType.INSTRUMENT_TYPE_SHARE:
                share = self._client.instruments.share_by( id_type = InstrumentIdType.INSTRUMENT_ID_TYPE_FIGI, id = instruments[0].figi )
                self._data['share'] = {}
                self._data['share']['asset_uid'] = share.instrument.asset_uid
                self._data['share']['buy_available_flag'] = share.instrument.buy_available_flag
                self._data['share']['country_of_risk'] = share.instrument.country_of_risk
                self._data['share']['country_of_risk_name'] = share.instrument.country_of_risk_name
                self._data['share']['currency'] = share.instrument.currency
                self._data['share']['div_yield_flag'] = share.instrument.div_yield_flag
                self._data['share']['dlong'] = share.instrument.dlong
                self._data['share']['dlong_client'] = share.instrument.dlong_client
                self._data['share']['dlong_min'] = share.instrument.dlong_min
                self._data['share']['dshort'] = share.instrument.dshort
                self._data['share']['dshort_client'] = share.instrument.dshort_client
                self._data['share']['dshort_min'] = share.instrument.dshort_min
                self._data['share']['exchange'] = share.instrument.exchange
                self._data['share']['instrument_exchange'] = share.instrument.instrument_exchange
                self._data['share']['ipo_date'] = share.instrument.ipo_date
                self._data['share']['issue_size'] = share.instrument.issue_size
                self._data['share']['issue_size_plan'] = share.instrument.issue_size_plan
                self._data['share']['klong'] = share.instrument.klong
                self._data['share']['kshort'] = share.instrument.kshort
                self._data['share']['liquidity_flag'] = share.instrument.liquidity_flag
                self._data['share']['min_price_increment'] = share.instrument.min_price_increment
                self._data['share']['nominal'] = share.instrument.nominal
                self._data['share']['otc_flag'] = share.instrument.otc_flag
                self._data['share']['real_exchange'] = share.instrument.real_exchange
                self._data['share']['sector'] = share.instrument.sector
                self._data['share']['sell_available_flag'] = share.instrument.sell_available_flag
                self._data['share']['share_type'] = share.instrument.share_type
                self._data['share']['short_enabled_flag'] = share.instrument.short_enabled_flag
                self._data['share']['trading_status'] = share.instrument.trading_status
            elif self._type == InstrumentType.INSTRUMENT_TYPE_BOND:
                bond = self._client.instruments.bond_by( id_type = InstrumentIdType.INSTRUMENT_ID_TYPE_FIGI, id = instruments[0].figi )
                self._data['bond'] = {}
                self._data['bond']['aci_value'] = bond.instrument.aci_value
                self._data['bond']['amortization_flag'] = bond.instrument.amortization_flag
                self._data['bond']['asset_uid'] = bond.instrument.asset_uid
                self._data['bond']['bond_type'] = bond.instrument.bond_type
                self._data['bond']['call_date'] = bond.instrument.call_date
                self._data['bond']['country_of_risk'] = bond.instrument.country_of_risk
                self._data['bond']['country_of_risk_name'] = bond.instrument.country_of_risk_name
                self._data['bond']['coupon_quantity_per_year'] = bond.instrument.coupon_quantity_per_year
                self._data['bond']['currency'] = bond.instrument.currency
                self._data['bond']['dlong'] = bond.instrument.dlong
                self._data['bond']['dlong_client'] = bond.instrument.dlong_client
                self._data['bond']['dlong_min'] = bond.instrument.dlong_min
                self._data['bond']['dshort'] = bond.instrument.dshort
                self._data['bond']['dshort_client'] = bond.instrument.dshort_client
                self._data['bond']['dshort_min'] = bond.instrument.dshort_min
                self._data['bond']['exchange'] = bond.instrument.exchange
                self._data['bond']['floating_coupon_flag'] = bond.instrument.floating_coupon_flag
                self._data['bond']['initial_nominal'] = bond.instrument.initial_nominal
                self._data['bond']['issue_kind'] = bond.instrument.issue_kind
                self._data['bond']['issue_size'] = bond.instrument.issue_size
                self._data['bond']['issue_size_plan'] = bond.instrument.issue_size_plan
                self._data['bond']['klong'] = bond.instrument.klong
                self._data['bond']['kshort'] = bond.instrument.kshort
                self._data['bond']['liquidity_flag'] = bond.instrument.liquidity_flag
                self._data['bond']['maturity_date'] = bond.instrument.maturity_date
                self._data['bond']['min_price_increment'] = bond.instrument.min_price_increment
                self._data['bond']['nominal'] = bond.instrument.nominal
                self._data['bond']['otc_flag'] = bond.instrument.otc_flag
                self._data['bond']['perpetual_flag'] = bond.instrument.perpetual_flag
                self._data['bond']['placement_date'] = bond.instrument.placement_date
                self._data['bond']['real_exchange'] = bond.instrument.real_exchange
                self._data['bond']['risk_level'] = bond.instrument.risk_level
                self._data['bond']['sector'] = bond.instrument.sector
                self._data['bond']['sell_available_flag'] = bond.instrument.sell_available_flag
                self._data['bond']['short_enabled_flag'] = bond.instrument.short_enabled_flag
                self._data['bond']['state_reg_date'] = bond.instrument.state_reg_date
                self._data['bond']['subordinated_flag'] = bond.instrument.subordinated_flag
                self._data['bond']['trading_status'] = bond.instrument.trading_status
            elif self._type == InstrumentType.INSTRUMENT_TYPE_ETF:
                etf = self._client.instruments.etf_by( id_type = InstrumentIdType.INSTRUMENT_ID_TYPE_FIGI, id = instruments[0].figi )
                self._data['etf'] = {}
                self._data['etf']['api_trade_available_flag'] = etf.instrument.api_trade_available_flag
                self._data['etf']['asset_uid'] = etf.instrument.asset_uid
                self._data['etf']['buy_available_flag'] = etf.instrument.buy_available_flag
                self._data['etf']['country_of_risk'] = etf.instrument.country_of_risk
                self._data['etf']['country_of_risk_name'] = etf.instrument.country_of_risk_name
                self._data['etf']['currency'] = etf.instrument.currency
                self._data['etf']['currency'] = etf.instrument.currency
                self._data['etf']['dlong'] = etf.instrument.dlong
                self._data['etf']['dlong_client'] = etf.instrument.dlong_client
                self._data['etf']['dlong_min'] = etf.instrument.dlong_min
                self._data['etf']['dshort'] = etf.instrument.dshort
                self._data['etf']['dshort_client'] = etf.instrument.dshort_client
                self._data['etf']['dshort_min'] = etf.instrument.dshort_min
                self._data['etf']['exchange'] = etf.instrument.exchange
                self._data['etf']['fixed_commission'] = etf.instrument.fixed_commission
                self._data['etf']['focus_type'] = etf.instrument.focus_type
                self._data['etf']['instrument_exchange'] = etf.instrument.instrument_exchange
                self._data['etf']['instrument_exchange'] = etf.instrument.instrument_exchange
                self._data['etf']['klong'] = etf.instrument.klong
                self._data['etf']['kshort'] = etf.instrument.kshort
                self._data['etf']['liquidity_flag'] = etf.instrument.liquidity_flag
                self._data['etf']['min_price_increment'] = etf.instrument.min_price_increment
                self._data['etf']['num_shares'] = etf.instrument.num_shares
                self._data['etf']['otc_flag'] = etf.instrument.otc_flag
                self._data['etf']['position_uid'] = etf.instrument.position_uid
                self._data['etf']['real_exchange'] = etf.instrument.real_exchange
                self._data['etf']['rebalancing_freq'] = etf.instrument.rebalancing_freq
                self._data['etf']['released_date'] = etf.instrument.released_date
                self._data['etf']['sector'] = etf.instrument.sector
                self._data['etf']['sell_available_flag'] = etf.instrument.sell_available_flag
                self._data['etf']['short_enabled_flag'] = etf.instrument.short_enabled_flag
                self._data['etf']['trading_status'] = etf.instrument.trading_status
            else:
                raise RuntimeError('Unsupported instrument: ' + _ticker + ', ' + _type + ', ' + _classCode)
            
            TkIO.write_at_path( instrumentPath, self._data )
        

    def trading_status(self):
        statusResponse = self._client.market_data.get_trading_statuses( instrument_ids=[self.figi()] ).trading_statuses[0]
        return statusResponse.trading_status

    def uid(self):        
        return self._data['uid']
    
    def ticker(self):
        return self._ticker

    def figi(self):
        return self._data['figi']

    def sector(self):
        if self._type == InstrumentType.INSTRUMENT_TYPE_SHARE:
            return self._data['share']['sector']
        elif self._type == InstrumentType.INSTRUMENT_TYPE_BOND:
            return self._data['bond']['sector']
        else:
            raise RuntimeError('Unsupported instrument type')

    def lot(self):
        return int(self._data['lot'])        

    def min_price_increment(self):
        if self._type == InstrumentType.INSTRUMENT_TYPE_SHARE:
            return self._data['share']['min_price_increment']
        elif self._type == InstrumentType.INSTRUMENT_TYPE_BOND:
            return self._data['bond']['min_price_increment']
        elif self._type == InstrumentType.INSTRUMENT_TYPE_ETF:
            return self._data['etf']['min_price_increment']
        else:
            raise RuntimeError('Unsupported instrument type')

    def get_order_book(self, depth : int):
        return self._client.market_data.get_order_book( figi = self.figi(), depth = depth )

    def get_last_trades(self, _from, _to, _tradeSourceType):
        return self._client.market_data.get_last_trades( figi = self.figi(), from_=_from, to=_to, trade_source=_tradeSourceType)
    
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