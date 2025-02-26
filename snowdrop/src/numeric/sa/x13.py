# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 18:09:41 2019

@author: A.Goumilevski
"""

import os
import pandas as pd
import statsmodels.api as sm
import statsmodels.tsa.x13 as x

path = os.path.dirname(os.path.abspath(__file__))
working_dir = os.path.abspath(os.path.join(path,"../../.."))
xpath = os.path.abspath(os.path.join(working_dir,"bin"))
os.environ['X13PATH'] = xpath

if os.name == 'nt':
    x._binary_names = ['x13as.exe']
else:
    x._binary_names = ['x13as']
                
def x13(file_path=None,series=None,freq="Q"):
    """X13 seasonal adjustment."""
    if file_path is None:  
        df = None
    else:
        df = pd.read_excel(file_path,sheet_name="Sheet1",index_col=0,header=0)   
        series = pd.DataFrame(df.values,index=df.index)
    
    results = sm.tsa.x13_arima_analysis(endog=series,prefer_x13=True,
                      outlier=True,freq=freq,trading=True,retspec=True)
    series_adj = pd.Series(results.seasadj.values,index=series.index.values)
    df_adj = series_adj.to_frame()
    df_adj.columns = ["Adjusted"]
    
    return (df_adj,df)
    
