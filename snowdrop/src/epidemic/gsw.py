#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 00:25:58 2021

@author: A.Goumilevski
"""
import os

working_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),"../../.."))
os.chdir(working_dir)

#from snowdrop.src.utils.util import simulationRange

ESTIMATE = False

if __name__ == '__main__':
    """
    The main test program.
    """
    from snowdrop.src.driver import importModel, run

    fname = "COVID19/gsw_model.yaml" # GSW model"
    # Path to model file
    file_path = os.path.abspath(os.path.join(working_dir, 'supplements/models', fname))
    
    # Path to data
    meas = os.path.abspath(os.path.join(working_dir, 'supplements/data/COVID19/country_data.xlsx'))
    
    output_variables = ['y','kpf','lab','c','unempl','inve','pinf','r','labstar'] 
    decomp = ['y','lab','inve','pinf']
    
    
    if ESTIMATE:
        
        # Create model object
        model = importModel(fname=file_path,Solver="Klein",Filter="Durbin_Koopman",Smoother="Durbin_Koopman",Prior="Diffuse",model_info=False)
        model.anticipate = True
        
        model.options["range"] = [[2008,1,1],[2023,1,1]]
        model.options["filter_range"] = [[2008,1,1],[2019,12,31]]
      
        ###   Estimate model parameters
        yy,dates,epsilonhat,etahat = run(model=model,meas=meas,Plot=True,Output=True,sample=False,estimate=True,output_variables=output_variables,decomp_variables=decomp)

        # Disable KF
        model.symbolic.measurement_file_path = None 

        # Save model parameters
        params = model.calibration["parameters"]


    # Create model object with perfect foresight solver
    model = importModel(fname=file_path,Solver="Klein",model_info=False)
    model.anticipate = True
    
    if ESTIMATE:
        model.calibration["parameters"] = params
        
    par_names  = model.symbols["parameters"]
    par_values = model.calibration["parameters"]
    par = dict(zip(par_names,par_values))
        
    model.options["periods"] = [1]
    model.options["shock_values"] = [90,0,0,0,0,0,0,0,0]

    fout = 'data/COVID19/US_Lockdown_1Q.csv' 
    run(model=model,meas=None,output_variables=output_variables,decomp_variables=decomp,fout=fout,Output=True,Plot=True,Tmax=9)
    
    # Two periods shocks
    model.options["periods"] = [1,2]
    model.options["shock_values"] = [[90,0,0,0,0,0,0,0,0],[90,0,0,0,0,0,0,0,0]]

    fout = 'data/COVID19/US_Lockdown_2Q.csv' 
    run(model=model,meas=None,output_variables=output_variables,decomp_variables=decomp,fout=fout,Output=True,Plot=True,Tmax=9)
   