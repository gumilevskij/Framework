"""Forecast time series.

Created on Tue Apr 23 14:14:37 2019

@author: A.Goumilevski
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime as dt

working_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),"../../.."))
os.chdir(working_dir)

from snowdrop.src.driver import run
from snowdrop.src.driver import importModel
from snowdrop.src.graphs.util import plotTimeSeries
from snowdrop.src.numeric.solver.linear_solver import stochastic_simulations

def forecast(Plot=False,save=True):
    """Forecast endogenous variables with user's variables path tuning."""
    
    shocks = {'Baseline':'Baseline', 'CPI inflation (cost-push)':'SHK_DLA_CPI',                                 
              'Exchange rate (UIP)': 'SHK_L_S', 'Foreign nominal interest rate':'SHK_RS_RW'}                                      
    shk_mean = {'Baseline':0, 'SHK_DLA_CPI':0.2, 'SHK_L_S':1, 'SHK_RS_RW':1}
    scenario_names = list(shocks.keys())
    
    path_to_dir = os.path.abspath(os.path.join(working_dir,"graphs"))
    fname = 'model.yaml'
    file_path = os.path.abspath(os.path.join(working_dir,'supplements/models/MPAF/'+fname))
    
    # Create model
    model = importModel(file_path,Solver="Klein")
    model.anticipate = False
    var_names = model.symbols["variables"]
    var_name = "L_GDP_GAP"
    var_ind = var_names.index(var_name)
    shock_names = model.symbols["shocks"]
    n_shocks = len(shock_names)
  
    ## Defines the time range of the forecast
    start_fcast = '2013-12-1'
    end_fcast   = '2020-1-1'
    start = dt.strptime(start_fcast,"%Y-%m-%d")
    end = dt.strptime(end_fcast,"%Y-%m-%d")
    fcast_range = pd.date_range(start=start,end=end,freq='QS')
    
    # Set simulation range
    model.options['range'] = [start_fcast,end_fcast]
    
    # Load Kalman filter data
    kf_hist = os.path.abspath(os.path.join(working_dir,"output/data/MPAF/kalm_hist.csv"))
    df = pd.read_csv(kf_hist,header=0,index_col=0,parse_dates=True)
    df = df[:start_fcast]
    obs = df[var_name][:start_fcast]
    hist_start = df.index[0].strftime("%Y-%m-%d")
    hist_end = df.index[-1].strftime("%Y-%m-%d")
    
    columns = list(df.columns)
    df.index = pd.to_datetime(df.index)
    df.dropna()

    # Set variables starting values
    model.setStartingValues(hist=df)


    ### ------------- Deterministic shocks block ----------------
    # Run scenario simulations
    results = []
    for i,k in enumerate(shocks):
        shk = shocks[k]
        values = np.zeros(n_shocks)
        if shk in shock_names:
            ind = shock_names.index(shk)
            values[ind] = shk_mean[shk]
        model.options["shock_values"] = values
        
        y,rng_date = run(model=model)
        results.append(list(y[:,var_ind][2:-1]))
        
    # Put together historical database 'data' and results of the simulation.
    dates = list(df.index.date) + list(rng_date.date[1:])
    series = []
    for i in range(len(scenario_names)):
        data = list(obs.values) + results[i]
        ts = pd.Series(data, dates)
        series.append(ts)
        
    if Plot:
        header = 'Output Gap: Policy Scenarios'
        titles = [" "] * len(series)
        plotTimeSeries(path_to_dir=path_to_dir,header=header,titles=titles,labels=scenario_names,series=series,highlight=[hist_start,hist_end],fig_sizes=(6,4),save=save,stacked=False,ext='pdf')

   
    ### -------------- Stochastic shocks block ---------------------
    quantiles = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

    if "periods" in model.options: 
        del model.options["periods"]
    if "shock_values" in model.options:    
        del model.options["shock_values"] 
            
    model.options["T"] = 2+len(fcast_range)
            
    series = stochastic_simulations(model=model,var_name=var_name,hist=kf_hist,fcast_range=fcast_range,quantiles=quantiles)
        
    if Plot:
        import matplotlib.pyplot as plt
        
        nc = int(np.floor(len(quantiles)/2))
        title = 'Output Gap (YoY)'
        fig = plt.figure(figsize=(6,4))
        plt.title(title,loc='center',wrap=True)
        plt.xlabel("Year")
        plt.ylabel("Percentage (%)")
        plt.plot(series[nc],lw=3,color='k')
        ser = list(obs) + list(series[i].values) 
        for i in range(2,len(quantiles)): 
            plt.plot(dates,ser,lw=2,color='grey',alpha=0.1)   
        for i in range(2,len(series)): 
            ser2 = list(obs) + list(series[i-1].values)           
            plt.fill_between(dates,ser2,ser,lw=2,color='green',alpha=0.5-0.1*abs(nc-i+1))          
        plt.figtext(0.75,0.8,"Projection -->",fontsize=12,fontstyle='italic',color="k")
        plt.xlim(df.index[0],end)
        plt.axvspan(df.index[0],start,color='lightgray',alpha=0.2)
        plt.axvspan(start,end,color='lightgreen',alpha=0.1)
        plt.grid(True)
        plt.tight_layout()
        
        if save:
            plt.savefig(path_to_dir+"/"+title+'.pdf')
        
        plt.show(block=False)
        plt.close(fig)
    
    print('Done.')


if __name__ == '__main__':
    """
    The main program
    """
    forecast(Plot=True,save=True)
