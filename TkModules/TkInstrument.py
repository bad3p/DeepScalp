import configparser
from configparser import ConfigParser
from tinkoff.invest import Client
from tinkoff.invest import InstrumentIdType
from tinkoff.invest import InstrumentType
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

    #------------------------------------------------------------------------------------------------------------------------    
    # Static helpers
    #------------------------------------------------------------------------------------------------------------------------    

    @staticmethod
    def get_instrument_tickers(client, instrument_kind, class_code):
        getAssetsResponse = client.instruments.get_assets(None)
        assetsWithInstruments = [asset for asset in getAssetsResponse.assets if len(asset.instruments) > 1 ]
        filteredAssets = [asset for asset in assetsWithInstruments if asset.instruments[0].instrument_kind == instrument_kind and asset.instruments[0].class_code == class_code]        
        return [asset.instruments[0].ticker for asset in filteredAssets]        
