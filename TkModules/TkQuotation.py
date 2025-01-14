
from tinkoff.invest.schemas import Quotation

#------------------------------------------------------------------------------------------------------------------------
# Quotation conversion
#------------------------------------------------------------------------------------------------------------------------

def quotation_to_float(val : Quotation):
    return val.units + val.nano * 10e-10

def float_to_quotation(val : float):
    result = Quotation( 0, int(val * 10e8) )
    return result 