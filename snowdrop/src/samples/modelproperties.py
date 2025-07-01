# -*- coding: utf-8 -*-
"""
Created on Thursday May 7, 2020
@author: Alexei Goumilevski

Reproduces results of  https://www.edx.org Monetary Policy Analysis and Forecasting course.
"""

import os
import pandas as pd
import numpy as np

working_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),"../../.."))
os.chdir(working_dir)

from snowdrop.src.driver import run as simulate
from snowdrop.src.driver import importModel
from snowdrop.src.graphs.util import plotTimeSeries


def modelproperties(Plot=False,save=True):
    """
    Simulates a set of basic shocks. 
    """
    fname = 'model.yaml'
    file_path = os.path.abspath(os.path.join(working_dir,'supplements/models/MPAF/'+fname))

    # Create model
    model = importModel(file_path,Solver="Klein")
    
    var_names = model.symbols["variables"]
    # Set starting values
    hist = os.path.abspath(working_dir + "/supplements/data/MPAF/history.xlsx")
    model.setStartingValues(hist=hist)
    
    path_to_dir = os.path.abspath(os.path.join(working_dir,"graphs"))
    # List of shocks
    # One period unexpected shocks: inflation, output, exchange rate, interest rate
    list_shocks = ['SHK_DLA_CPI','SHK_L_GDP_GAP','SHK_L_S','SHK_RS']
    list_headers = ['Inflationary (cost-push) Shock','Aggregate Demand Shock', 
                   'Exchange Rate Shock', 'Interest Rate (monetary policy) Shock']

    
    # Sets the time frame for the simulation 
    model.options["periods"] = [1]
    #model.options["range"] = [[2000,1,1],[2005,1,1]] # fiver-years simulation horizon
    shock_names = model.symbols["shocks"]
    n_shocks = len(shock_names)
    
    # Fills the respective databases with the shocks' values for the starting
    # point of the simulation (startsim). For simplicity, all shocks are set to
    # 1 percent
    ################################################################### Graphs
    
    
    titles = ['CPI Inflation QoQ (% ar)',
              'Nominal Interest Rate (% ar)',
              'Nominal ER Deprec. QoQ (% ar)',
              'Output Gap (%)',
              'Real Interest Rate Gap (%)',
              'Real Exchange Rate Gap (%)',
              'Real Monetary Condition Index (%)']
        
    shocks = [1,1,1,1]
    num_shocks = len(list_shocks)
    for i in range(num_shocks):
        # Set shocks
        shock_name = list_shocks[i]
        ind = shock_names.index(shock_name)
        shock_values = np.zeros(n_shocks)
        shock_values[ind] = shocks[i]
        model.options["shock_values"] = shock_values
        
        # Find solution
        results,dates = simulate(model=model,irf=True)
        rows,columns = results.shape
        
        d = {}
        for j in range(columns):
            n = var_names[j]
            data = results[:,j] 
            ts = pd.Series(data[1:-1], dates)
            ts = ts[:'2004-1-1']
            d[n] = ts
    
    
        # Create separate page with IRFs for each shock
        series = [[d["DLA_CPI"]],
                  [d["RS"]],
                  [d["DLA_S"]],
                  [d["L_GDP_GAP"]],
                  [d["RR_GAP"]],
                  [d["L_Z_GAP"]],
                  [d["MCI"]]
                 ]
        
        header = list_headers[i]
        labels=[[]]
        plotTimeSeries(path_to_dir=path_to_dir,header=header,titles=titles,labels=labels,series=series,sizes=[3,3],save=save)
    
    
    if save:
        from snowdrop.src.utils.merge import merge
        outputFile = os.path.abspath(os.path.join(working_dir,"results/MPAF_Report_Shocks.pdf"))
        files = []
        for f in list_headers:
            files.append(os.path.abspath((os.path.join(working_dir,"graphs/"+f+".pdf"))))
        merge(outputFile,files)

    print('Done!')


if __name__ == '__main__':
    """
    The main program
    """
    modelproperties(Plot=True,save=True)
