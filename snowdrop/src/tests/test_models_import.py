# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 14:14:16 2019
@author: AGoumilevski

Simulates a set of basic shocks.  
"""

import os
import sys
import pandas as pd
import numpy as np

path = os.path.dirname(os.path.abspath(__file__))
working_dir = os.path.abspath(path+"/../../..")
sys.path.append(working_dir)
os.chdir(working_dir)

from snowdrop.src.driver import run
from snowdrop.src.model.factory import import_model
from snowdrop.src.graphs.util import plotTimeSeries


def test_model_import(file_path, list_variables, list_shocks, shocks, list_titles, irf = True,calibration={}, freq=None, rng=None, sizes = [1,1]):
    """Test import of model files. """
    
    # Create model
    model = import_model(file_path,return_interface=False,calibration=calibration)
    
    path_to_dir = os.path.abspath(os.path.join(working_dir,"graphs"))
    
    if not freq is None:
        model.options["frequency"] = freq
    if not rng is None:
        model.options["range"] = rng
    model.options["periods"] = [[2025,7,1]]
    #model.options["periods"] = [1]
    var_names = model.symbols["variables"]
    shock_names = model.symbols["shocks"]
    n_shocks = len(shock_names)
    num_shocks = len(list_shocks)
    
    shock_values = np.zeros(n_shocks)
    for i in range(num_shocks):
        # Set shocks
        shock_name = list_shocks[i]
        if shock_name in shock_names:
            ind = [i for i,x in enumerate(shock_names) if x==shock_name][0]
            shock_values[ind] = shocks[i]
        model.options["shock_values"] = shock_values
        
        # Find solution
        results,rng_date = run(model=model,irf=irf)
        rows,columns = results.shape
        
        d = {}
        for j in range(columns):
            n = var_names[j]
            data = results[:,j] 
            ts = pd.Series(data[1:-1], rng_date)
            d[n] = ts
    
        header = list_titles[i]
        if len(list_variables) == 1:
            titles = ['Nominal Interest Rate']
        else:
            titles = list_variables
        series = [[d[k]] for k in list_variables]
        labels = None
        plotTimeSeries(path_to_dir=path_to_dir,header=header,titles=titles,labels=labels,series=series,sizes=sizes)
    

if __name__ == '__main__':
    """
    Main program.
    """
    rng =  ["2025-1-1","2035-1-1"]; freq = 1 # Quarterly
    
    # 1. Yaml model file
    # Unit shock to nominal interest rate
    file_path = os.path.abspath(os.path.join(working_dir,'snowdrop/models/MPAF/model.yaml'))
    test_model_import(file_path=file_path,list_variables=['RS'],list_shocks=['SHK_RS'],shocks=[1],list_titles=['Policy Interest Rate'],freq=freq,rng=rng)
       
    # 2. Dynare model file
    # Unit shock to nominal interest rate
    file_path = os.path.abspath(os.path.join(working_dir,'snowdrop/models/MPAF/model.mod'))
    test_model_import(file_path=file_path,list_variables=['RS'],list_shocks=['SHK_RS'],shocks=[1],list_titles=['Policy Interest Rate'],freq=freq,rng=rng)
     
    # 3. Iris model file
    # Shock to total factor productivity
    calib_params = { "beta": 0.99, "delta": 0.03, "gamma":0.5, "rho":0.8, "Y":1.25, "C":0.78, "r":0.01, "K":15.55, "a":0.1, "A":0.1}
    file_path = os.path.abspath(os.path.join(working_dir,'snowdrop/models/TOY/RBC.model'))
    test_model_import(file_path=file_path,list_variables=['Y','C','K','r','A'],list_shocks=['ea'],shocks=[0.1],list_titles=['Macroeconomic Variables'],
                      irf=False,calibration=calib_params,freq=freq,rng=rng,sizes=[3,2])
