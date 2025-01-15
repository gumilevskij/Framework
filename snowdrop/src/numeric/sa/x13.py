# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 18:09:41 2019

@author: A.Goumilevski
"""

import os, sys
import pandas as pd
import statsmodels.api as sm

path = os.path.dirname(os.path.abspath(__file__))
working_dir = os.path.abspath(os.path.join(path,"../../.."))
lib = os.path.abspath(os.path.join(working_dir,"bin"))
if os.path.exists(lib) and not lib is sys.path:
    sys.path.append(lib)
    os.chdir(lib)
    
def x13(file_path=None,series=None,freq="Q"):
    """X13 seasonal adjustment."""
    if file_path is None:  
        df = None
    else:
        df = pd.read_excel(file_path,sheet_name="Sheet1",index_col=0,header=0)   
        series = pd.DataFrame(df.values,index=df.index)
    
    xpath = os.path.abspath(os.path.join(lib,"x13as"))
    results = sm.tsa.x13_arima_analysis(endog=series,x12path=xpath, 
                      outlier=True,freq=freq,trading=True,retspec=True)
    series_adj = pd.Series(results.seasadj.values,index=series.index.values)
    df_adj = series_adj.to_frame()
    df_adj.columns = ["Adjusted"]
    
    return (df_adj,df)
    