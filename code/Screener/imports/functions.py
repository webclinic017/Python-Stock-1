#from imports.imports import *
#from imports.functions import *
import imports

def getStock(npa):
    symbol = npa[0]
    start_=npa[1]
    end_=npa[2]
    
    data_ = imports.functions.yf.download(symbol, start=start_,end=end_)
  
    return([symbol,data_])