# -*- coding: utf-8 -*-
"""
Created on Tue May 8, 2020

@author: Alexei Goumilevski

Reproduces results of  https://www.edx.org/  MPAF course.
"""
import os
import pandas as pd
from datetime import datetime as dt
from datetime import date
from dateutil import relativedelta as rd

working_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),"../../.."))
os.chdir(working_dir)

from snowdrop.src.driver import importModel
from snowdrop.src.driver import run
from snowdrop.src.graphs.util import plotTimeSeries
from snowdrop.src.utils.util import simulationRange


def in_sample(Plot=False,save=True):
    """In sample forecast of endogenous variables given user judgements."""
    path_to_dir = os.path.abspath(os.path.join(working_dir,"graphs"))
    fname = 'model.yaml'
    file_path = os.path.abspath(os.path.join(working_dir,'supplements/models/MPAF/'+fname))
    # Create model
    model = importModel(file_path,Solver="Klein")
    model.anticipate = True
    var_names = model.symbols["variables"]

    ## Time span
    rtime = '2005-1-1'  #start of sample
    stime = '2005-6-1'  #starting point of the first simulation 
                        #stime is the initial state
                        #stime+1 is the first simulated time period
    etime = '2013-12-1' #the end of the known history
    e_time = dt.strptime(etime,"%Y-%m-%d")
    
    ## Selection of historical time series for computing model's forecasting properties:
    list_xnames = ['DLA_CPI','D4L_CPI','L_GDP_GAP','D4L_GDP','L_S']
    list_headers = ['CPI Inflation QoQ annualized (in % pa)','CPI Inflation YoY (in % pa)','Output Gap (in %)',
                   'Real GDP Growth YoY (in % pa)','Nominal Exchange Rate (LCY/FCY, 100*log)']
    
    ## Database preparation
    # Load quarterly data
    skip_rows = 6
    file_path = os.path.abspath(os.path.join(working_dir,"supplements/data/MPAF/kalm_his_new.csv"))
    df = pd.read_csv(filepath_or_buffer=file_path,sep=',',header=0,index_col=0,parse_dates=True,infer_datetime_format=True)
    df = df.iloc[skip_rows:].astype(float)
    df.index = pd.to_datetime(df.index)  # Quartely data frequency
    
    d = {} 
    for c in df.columns:
        d[c] = df[c][rtime:etime]
        d[c].columns = [c]
            
    
    shock_values = model.calibration["shocks"]
    n_shocks = len(shock_values)
    
    
    ## Simulations ... (done by a "loop")
    # Beginning of the "loop" ...
    sim_rng = pd.date_range(start=stime,freq='QS',end=etime)
    lst = []
    for k in range(len(sim_rng)):
        t = sim_rng[k]
        print('-'*80)
        print('The starting time period of this projection round is : {}'.format(t))
        # Simulation range is eight quarters or two years.
        f_time = t + rd.relativedelta(months=8*3)
        f_time = min(e_time,f_time)
        rng = [[t.year,t.month,t.day],[f_time.year,f_time.month,f_time.day]]
        # Set simulation range
        model.options['range'] = rng
        simulationRange(model)
        # Reset shocks
        model.options['shock_values'] = [[0]*n_shocks]
        
        # Set starting values
        model.setStartingValues(hist=file_path,skip_rows=skip_rows)

        start = date(year=t.year, month=t.month, day=t.day)  + rd.relativedelta(months=3)
        end   = date(year=f_time.year, month=f_time.month, day=f_time.day)
        
        if (start<end):
            
            model.mapSwap = {}
            alt_rng  = pd.date_range(start=start,end=end,freq='QS')
            
            m = {}; shock_names = []
            
            # condition the forecast on the inflation target        
            m['D4L_CPI_TAR'] = df['D4L_CPI_TAR'][alt_rng]
            shock_names.append('SHK_D4L_CPI_TAR')
            
            # condition the forecast on the foreign output gap
            m['L_GDP_RW_GAP'] = df['L_GDP_RW_GAP'][alt_rng] 
            shock_names.append('SHK_L_GDP_RW_GAP')
            
            # condition the forecast on the foreign nominal interest rate
            m['RS_RW'] = df['RS_RW'][alt_rng]   
            shock_names.append('SHK_RS_RW') 
            
           # condition the forecast on the foreign real equilibrium interest rate
            m['RR_RW_BAR'] = df['RR_RW_BAR'][alt_rng]
            shock_names.append('SHK_RR_RW_BAR')
            
            # condition the forecast on the foreign inflation rate
            m['DLA_CPI_RW'] = df['DLA_CPI_RW'][alt_rng]
            shock_names.append('SHK_DLA_CPI_RW')
            
            # swap endogenous and exogenous variables
            model.swap(var1=m,var2=shock_names)
            
        else:
            model.mapSwap = None
        
        
        # Simulate the alternative scenario
        y,dates = run(model=model)
        
        rows,columns = y.shape
        results = {}
        for j in range(columns):
            n = var_names[j]
            results[n] = pd.Series(y[:,j][:len(dates)], dates)
        lst.append(results)
   
    all_data = {}
    for e in lst:
        for k in e:
            if not k in all_data:
                all_data[k] = []
                if k in d:
                    all_data[k].append(d[k])
            all_data[k].append(e[k])
            
    ############################################  Graphs
    if Plot:
        for i,n in enumerate(list_xnames):
            header = n
            series = all_data[n]
            titles = [list_headers[i]]*len(series)
            labels = None
            plotTimeSeries(path_to_dir=path_to_dir,header=header,titles=titles,labels=labels,series=series,save=save)
            
    
    if save:
        from snowdrop.src.utils.merge import merge
        # imagelist is the list with all image filenames
        outputFile = os.path.abspath(os.path.join(working_dir,"results/MPAF_in_sample.pdf"))
        files = []
        for f in list_xnames:
            files.append(working_dir+"/graphs/"+f+".pdf")
        merge(outputFile,files)
        
    print('Done with in_sample script!!!')
    
   
if __name__ == '__main__':
    """
    The main program
    """
    in_sample(Plot=True,save=True)
