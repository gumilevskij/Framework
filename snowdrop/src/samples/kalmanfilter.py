# -*- coding: utf-8 -*-
"""
Created on Wed May 6, 2020
@author: Alexei Goumilevski
Reproduces results of https://www.edx.org Monetary Policy Analysis and Forecasting course.


Please note that for time series variables the following rule applies for varible leads and lags:
                                 var(-1) -> var.shift(1) and var(+1) -> var.shift(-1)
"""
import os
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import glob

working_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),"../../.."))
os.chdir(working_dir)

from snowdrop.src.driver import importModel
from snowdrop.src.driver import kalman_filter
from snowdrop.src.graphs.util import plotTimeSeries
from snowdrop.src.utils.merge import merge
from snowdrop.src.numeric.solver.util import find_residuals
from snowdrop.src.utils.util import saveTimeSeries as dbsave


def kalmanfilter(Plot=False,save=True):
    """Run Kalman filter and smoother."""
    path_to_dir = os.path.join(working_dir,'graphs')
    meas = os.path.abspath(os.path.join(working_dir,'supplements/data/MPAF/history_new.csv'))
    fout = os.path.abspath(os.path.join(working_dir,'supplements/data/MPAF/results.csv'))     # Results are saved in this file
    
    # Read historic data
    #df = pd.read_excel(io=meas,header=0,index_col=0,parse_dates=True,infer_datetime_format=True)
    df = pd.read_csv(meas,header=0,index_col=0,parse_dates=True,infer_datetime_format=True)
    df.index = pd.to_datetime(df.index,format='#Q')
        
    # Populate dictionary with values
    d = {}
    for c in df.columns:
        d[c] = df[c]
    
    # Instantiate model  
    fname = 'model.yaml'
    file_path = os.path.abspath(os.path.join(working_dir,'supplements/models/MPAF',fname))
    # Create model object
    model = importModel(fname=file_path, Solver="Klein", Filter="Durbin_Koopman",
                        Smoother="Durbin_Koopman", Prior="Equilibrium", measurement_file_path=meas, use_cache=False)
    var_names = model.symbols["variables"]
    
    # Set time range
    simulation_range = [[1997,1,1],[2013,12,1]]
    filter_range = [[1998,1,1],[2013,12,1]]
    
    start = dt.date(*filter_range[0])
    end   = dt.date(*filter_range[1])
    
    # Make sure fiter range is inside measurement data range
    if start < df.index[0].date():
        start = df.index[0]
    if end > df.index[-1].date():
        end = df.index[-1]
    filter_range = [[start.year,start.month,start.day],[end.year,end.month,end.day]]
    
    model.options['range'] = simulation_range
    model.options['filter_range'] = filter_range
    
    # Set starting values
    model.setStartingValues(hist=meas,debug=True)
    
    var_names = model.symbols['variables']
    var_values = np.copy(model.calibration['variables'])
    
    lst = sorted([var_names[i]+"="+str(var_values[i]) for i in range(len(var_names))])
    # print(lst)
    # print()
    
    # Get filtered and smoothed values
    yy,dates,epsilonhat,etahat = kalman_filter(model=model,meas=meas,Output=True,fout=fout)
    
    #filtered = yy[0]
    #smoothed = yy[1]
    results = yy[1]
    
    rows,columns = results.shape
    start = dates[0]
 
    for j in range(columns):
        n = var_names[j]
        if "_plus_" in n or "_minus_" in n:
            continue
        data = results[:,j]
        m = min(len(data),len(dates))
        ts = pd.Series(data[:m], dates[:m])
        d[n] = ts[start:end]
       
    
    par_names = model.symbols['parameters']
    par_values = model.calibration['parameters']
    for n,v in zip(par_names,par_values):
        d[n] = v
        
    # Get shocks and residuals
    shocks = model.symbols['shocks']
    n_shk = len(shocks)
    res = find_residuals(model,results)
    if etahat is None:
        for j in range(n_shk):
            n = shocks[j]
            data = res[:,j]
            m = min(len(dates),len(data))
            ts = pd.Series(data[:m],dates[:m])
            d[n] = ts[start:end]
            ts2 = pd.Series(np.zeros(m),dates[:m])
            d[n+"_other"] = ts2[start:end]
    else:
        for j in range(n_shk):
            n = shocks[j]
            data = etahat[:,j]
            m = min(len(dates),len(data))
            ts = pd.Series(data[:m],dates[:m])
            d[n] = ts[start:end]
            data = res[:,j]
            ts2 = pd.Series(data[:m],dates[:m])
            d[n+"_other"] = (ts2-ts)[start:end]
   
    
    ## Save the database
    file_path = os.path.abspath(os.path.join(working_dir,'data/MPAF/kalm_his_new.csv'))
    dbsave(fname=file_path,data=d)
    

    ################################################################### Graphs
    if Plot:
        ### Observed and Trends
        header = 'Observed and Trends'
        titles = ['GDP',
                  'Real Interest Rate',
                  'Foreign Real Interest Rate',
                  'Real Exchange Rate',
                  'Change in Eq. Real Exchange rate',
                  'Risk Premium']
 
        series = [[d["L_GDP"],d["L_GDP_BAR"]],
                  [d["RR"],d["RR_BAR"]],
                  [d["RR_RW"],d["RR_RW_BAR"]],
                  [d["L_Z"],d["L_Z_BAR"]],
                  [d["DLA_Z_BAR"]],
                  [d["PREM"]]
                 ]
        labels=[["Actual","Trend"],["Actual","Trend"],["Actual","Trend"],["Actual","Trend"]]
        plotTimeSeries(path_to_dir=path_to_dir,header=header,titles=titles,labels=labels,series=series,sizes=[2,3],save=save)
    
        ### Gaps
        header = 'Gaps'
        titles = ['Inflation',
                  'Marginal Cost',
                  'GDP GAP',
                  'Monetary Conditions',
                  'Real Interest Rate Gap',
                  'Real Exchange Rate Gap',
                  'Foreign GDP Gap',
                  'Foreign inflatio',
                  'Foreign interest rates']
 
        series = [[d["DLA_CPI"],d["D4L_CPI_TAR"]],
                  [d["RMC"]],
                  [d["L_GDP_GAP"]],
                  [d["MCI"]],
                  [d["RR_GAP"]],
                  [d["L_Z_GAP"]],
                  [d["L_GDP_RW_GAP"]],
                  [d["DLA_CPI_RW"]],
                  [d["RS_RW"]]
                 ]
        labels=[[]]
        plotTimeSeries(path_to_dir=path_to_dir,header=header,titles=titles,labels=labels,series=series,sizes=[3,3],save=save)
    
 
        ### Shocks  
        header = 'Shocks'
        titles = ['Inflation (cost-push)',
                  'Output gap',
                  'Interest Rate',
                  'Exchange Rate',
                  'Trend Real Interest Rate',
                  'Trend Real Exchange Rate']
        series = [[d["SHK_DLA_CPI"]],
                  [d["SHK_L_GDP_GAP"]],
                  [d["SHK_RS"]],
                  [d["SHK_L_S"]],
                  [d["SHK_RR_BAR"]],
                  [d["SHK_DLA_Z_BAR"]],
                 ]
        labels=[[]]
        plotTimeSeries(path_to_dir=path_to_dir,header=header,titles=titles,labels=labels,series=series,sizes=[2,3],save=save)
    
   
        ### Interest rate and exchange rate
        header = 'Interest rate and exchange rate'
        titles = ['Nominal interest rate',
                  'Real Interest Rate Gap',
                  'Inflation qoq',
                  'Nominal exchange rate',
                  'Real Exchange Rate Gap',
                  'Nominal exchange rate depreciation',
                  'Inflation differential',
                  'Interest rate differential',
                  'Exchange rate shock']
        series = [[d["RS"]],
                  [d["RR_GAP"]],
                  [d["DLA_CPI"]],
                  [d["S"]],
                  [d["L_Z_GAP"]],
                  [d["DLA_S"],d["D4L_S"]],
                  [d["DLA_CPI"],d["DLA_CPI_RW"]],
                  [d["RS"],d["RS_RW"]],
                  [d["SHK_L_S"]],
                 ]
        labels=[[],[],[],[],[],['qoq','yoy'],['domestic inflation','foreign inflation'],['domestic IR','foreign IR']]
        plotTimeSeries(path_to_dir=path_to_dir,header=header,titles=titles,labels=labels,series=series,sizes=[3,3],save=save)
    
       
        ### Inflation
        header = 'Inflation'
        titles = ['Inflation qoq, percent',
                  'Inflation and RMC, percent',
                  'Marginal cost decomposition, pp']
        series = [[d["DLA_CPI"],d["DLA_CPI"]-d["SHK_DLA_CPI"]],
                  [d["DLA_CPI"]-d["D4L_CPI_TAR"],d["RMC"]],
                  [d["a3"]*d["L_GDP_GAP"], (1-d["a3"])*d["L_Z_GAP"],d["RMC"]]
                 ]
        labels=[['Actual','Predicted'],['Inflation (deviation from the target)','RMC'],['Output gap','RER gap','RMC']]
        plotTimeSeries(path_to_dir=path_to_dir,header=header,titles=titles,labels=labels,series=series,sizes=[3,1],save=save)
    
        
        # DLA_CPI = a1*DLA_CPI(-1) + (1-a1)*DLA_CPI(+1) + a2*RMC + SHK_DLA_CPI
        ### Inflation decomposition
        header = 'Inflation decomposition'
        titles = ['Inflation decomposition, qoq percent']
        series = [[d["a1"]*d["DLA_CPI"].shift(1),(1-d["a1"])*d["E_DLA_CPI"], d["a2"]*d["a3"]*d["L_GDP_GAP"], d["a2"]*(1-d["a3"])*d["L_Z_GAP"], d["SHK_DLA_CPI"],d["SHK_DLA_CPI_other"],
                  d["DLA_CPI"]]
                 ]
        labels=[['Persistency','Expectations','Output Gap','RER Gap','Shock','Other Shocks','Inflation']]
        plotTimeSeries(path_to_dir=path_to_dir,header=header,titles=titles,labels=labels,series=series,sizes=[1,1],save=save)
    
        
        ### Output gap
        header = 'Output gap'
        titles = ['Output gap, percent',
                  'Output gap decomposition, pp']
        series = [[d["L_GDP_GAP"],d["L_GDP_GAP"]-d["SHK_L_GDP_GAP"]],
                  [d["b1"]*d["L_GDP_GAP"].shift(1), -d["b2"]*d["b4"]*d["RR_GAP"], d["b2"]*(1-d["b4"])*d["L_Z_GAP"], d["b3"]*d["L_GDP_RW_GAP"], d["SHK_L_GDP_GAP"],d["SHK_L_GDP_GAP_other"],
                   d["L_GDP_GAP"]]
                 ]
        labels=[['Actual','Predicted'],['Lag','RIR gap','RER gap','Foreign gap','Shock','OtherShock','Output Gap']]
        plotTimeSeries(path_to_dir=path_to_dir,header=header,titles=titles,labels=labels,series=series,sizes=[2,1],save=save)
    
        ### Decomposition          
        header = 'Decomposition'    
        titles = ['MCI decomposition, pp']
        series = [[d["b4"]*d["RR_GAP"], (1-d["b4"])*(-d["L_Z_GAP"]),
                  d["MCI"]]
                 ]
        labels=[['RIR gap','RER gap','MCI']]
        plotTimeSeries(path_to_dir=path_to_dir,header=header,titles=titles,labels=labels,series=series,sizes=[2,1],save=save)     
        
        if save:
            # image list is the list with all image file names
            lst = ["Observed and Trends","Gaps","Shocks","Interest rate and exchange rate",
                   "Inflation","Inflation decomposition","Output gap","Decomposition"]
            
            outputFile = os.path.abspath(os.path.join(working_dir,"results/MPAF_Filtration.pdf"))
            files = []
            for f in lst:
                files.append(working_dir+"/graphs/"+f+".pdf")
            merge(outputFile,files)

    print('Done!')

        
if __name__ == '__main__':
    """
    The main program
    """
    kalmanfilter(Plot=True,save=True)
    

