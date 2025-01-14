
import os
import configparser
import numpy as np
from datetime import date, datetime, timedelta

from tinkoff.invest.constants import INVEST_GRPC_API
from tinkoff.invest import (
    Client,
    InstrumentType,
    InstrumentIdType,
    SecurityTradingStatus,
    GetOrderBookResponse
)
from tinkoff.invest.schemas import CandleInterval, TradeSourceType
from tinkoff.invest.utils import now
from tinkoff.invest.exceptions import RequestError
from TkModules.TkQuotation import quotation_to_float, float_to_quotation
from TkModules.TkInstrument import TkInstrument
from TkModules.TkIO import TkIO

TOKEN = os.environ["TK_TOKEN"]

config = configparser.ConfigParser()
config.read( 'TkConfig.ini' )

with Client(TOKEN, target=INVEST_GRPC_API) as client:
    share = TkInstrument(client, config, InstrumentType.INSTRUMENT_TYPE_SHARE, "SBERP", "TQBR")
    print( share.lot() )
    print( quotation_to_float( share.min_price_increment() ) )
    #print( share.get_order_book(50) )    
    #toDate = now() 
    #fromDate = now() - timedelta(minutes=5)
    #print( share.get_last_trades( fromDate, toDate, TradeSourceType.TRADE_SOURCE_UNSPECIFIED) )

#------------------------------------------------------------------------------------------------------------------------    

#TkIO.write_at_path( "output.bin", 'string' )
#TkIO.append_at_path( "output.bin", 123 )
#TkIO.append_at_path( "output.bin", 456.789 )
#TkIO.append_at_path( "output.bin", None )
#TkIO.append_at_path( "output.bin", (37,73) )
#TkIO.append_at_path( "output.bin", {'name':'jack','age':33} )

#print( TkIO.index_at_path("output.bin") )
#print( TkIO.read_at_path( "output.bin" ) )

#index = TkIO.index_at_path("output.bin")
#print( TkIO.read_at_path( "output.bin", index[:4] ) )
#print( TkIO.read_at_path( "output.bin", index[2:] ) )

#output = open( "output.bin", 'wb+')
#offsets = [output.tell()]
#TkIO.write_to_file(output, 'string')
#offsets.append( output.tell() )
#TkIO.write_to_file(output, 123)
#offsets.append( output.tell() )
#TkIO.write_to_file(output, None)
#offsets.append( output.tell() )
#TkIO.write_to_file(output, (37,73))
#offsets.append( output.tell() )
#TkIO.write_to_file(output, {'name':'jack','age':33})
#output.close()

#input = open("output.bin", 'rb+')
#for i in range(len(offsets)):
#    input.seek(offsets[i], 0)
#    print( i, offsets[i], TkIO.read_from_file(input) )
#input.close()