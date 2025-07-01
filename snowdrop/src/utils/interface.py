# -*- coding: utf-8 -*-
"""
Created on Tue May 15 11:58:35 2018
@author: agoumilevski

This module replicates some TROLL functionality.  In TROLL user may load a steady-state and a dynamic models,
set starting values of endogenous variables, preform simulations, save results in a database, and plot
these results. These operations can be run as a sequence of commands in a terminal window.
"""

import os
import dill as pickle
import numpy as np
from snowdrop.src.model.model import Model

path = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(path+"\\..")


attributes = ['functions','functions_src','name','infos','autodiff','jaxdiff',
              'lead_lag_incidence','linear_model','symbols','symbolic',
              'options','covariances','distribution',
              'eqLabels','solved','eq_vars','eqs_number','bSparse',
              'priors','max_lead','min_lag','n_fwd_looking_var',
              'numberOfNewEqs','terminal_values','var_lag','var_lead',
              'variables','state_vars', 'ss','steady_state','ev',
              'topology','lead_lag_incidence','n_fwd_looking_var',
              'n_bkwd_looking_var','isLinear','anticipate',
              'SOLVER','FILTER','SMOOTHER','PRIOR','INITIAL_CONDITION']

# all_attributes = ['COMPLEMENTARITY_CONDITIONS', 'FILTER', 'GENERATE_CPP_CODE', 
#               'INITIAL_CONDITION', 'INIT_COND_CORRECTION', 'PRIOR', 
#               'SAMPLING_ALGORITHM', 'SMOOTHER', 'SOLVER', 'T', 'Topology', 
#               'anticipate', 'autodiff', 'jaxdiff', 'bSparse', 'calibration', 'calibration_dict', 
#               'condShocks', 'count', 'covariances', 'data_sources', 'date_range', 
#               'distribution', 'eqLabels', 'eq_vars', 'eqs_number', 'estimate', 'ev', 
#               'functions', 'functions_src', 'infos',  'isLinear', 
#               'lead_lag_incidence', 'linear_model', 'mapSwap', 'markov_chain', 
#               'max_lead', 'max_lead_shock', 'min_lag', 'min_lag_shock', 
#               'nUnit', 'n_bkwd_looking_shocks', 'n_bkwd_looking_var', 
#               'n_fwd_looking_shocks', 'n_fwd_looking_var', 'name', 
#               'nonstationary', 'numberOfNewEqs', 'options', 'order', 
#               'priors', 'solved', 'stable', 'state_vars', 'stationary', 
#               'steady_state', 'symbolic', 'symbols', 'terminal_values', 
#               'topology', 'total_nmbr_shocks', 'unstable', 'var_lag', 'var_lead', 
#               'var_rows_incidence']

   
def loadAll(filename):  
    """Read and deserialize objects from a file."""
    with open(filename, "rb") as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break


def saveModel(file_path,model):    
    """Save (serializes) model content and saves it into a file."""
    pickle.dump(model,open(file_path,'wb'))
    
    
def __saveModel(file_path,model):    
    """Save (serializes) model content and saves it into a file."""
    attributes = [attr for attr in dir(model) if not callable(getattr(model, attr)) and not attr.startswith("__")]
    
    with open(file_path, 'wb') as f:
        for a in attributes:
            try:
                if hasattr(model,a):
                    attr = getattr(model,a)
                    m = [a,attr]
                    #print(m)
                    data = pickle.dumps(m)
                    f.write(data)
            except RuntimeWarning:
                pass
        
        
def loadModel(file_path,shocks_file_path=None,steady_state_file_path=None,calibration_file_path=None,calibration=None):
    """
    Deserializes model object from a file.
    """
    from snowdrop.src.utils.load import loadFile
    
    model = pickle.load(open(file_path,'rb')) 
    
    if not shocks_file_path is None:
        model.options['shock_values'],model.options['periods'] = loadFile(shocks_file_path,model.calibration,names=model.symbols['shocks'],bShocks=True)
       
    if not steady_state_file_path is None:
        calibration = loadFile(steady_state_file_path,model.calibration,names=model.symbols['variables'],bShocks=False)
        model.calibration = {**model.calibration, **calibration}
        
    if not calibration_file_path is None:
        calibration = loadFile(calibration_file_path,model.calibration,names=model.symbols['parameters']+model.symbols['variables']+model.symbols['shocks'],bShocks=False)
        model.calibration = {**model.calibration, **calibration}
        
    if calibration:
        var_names = model.symbols["variables"]
        var_values = model.calibration["variables"]
        par_names = model.symbols["parameters"]
        par_values = model.calibration["parameters"]
        for  k in calibration:
            if k in var_names:
                ind = var_names.index(k)
                var_values[ind] = calibration[k]
            if k in par_names:
                ind = par_names.index(k)
                par_values[ind] = calibration[k]
                
        model.calibration["variables"] = var_values
        model.calibration["parameters"] = par_values
        
    return model
    

def __loadModel(file_path,shocks_file_path=None,steady_state_file_path=None,calibration_file_path=None):
    """
    Deserializes model object from a file.
    """
    from snowdrop.src.utils.load import loadFile
    
    items = loadAll(file_path)
    m = {}
    for item in items:
        name,attr = item 
        if name == 'functions_src':
            d = {}
            for k in attr:
                txt = attr[k]
                if k.startswith('f_') and not txt is None:
                    filename = os.path.abspath(src_dir + "\\preprocessor\\" + k + ".py")
                    with open(filename, "w") as f:
                        f.writelines(txt)
                    f = {}
                    exec(txt, f)
                    d[k] = f[k]
            m['functions'] = d  
        else:
            m[name] = attr
        
    infos = {
        'name' : m['name'],
        'filename' : file_path
    }
        
    smodel = m['symbolic'] 
    variables_names = smodel.symbols['variables']
    parameters_names = smodel.symbols['parameters']
    shocks_names = smodel.symbols['shocks']
       
    if not steady_state_file_path is None:
        ms = loadFile(path=steady_state_file_path,names=variables_names) 
        steady_states = {}
        for n in variables_names:
            if n in ms:
                steady_states[n] = ms[n]
            else:
                steady_states[n] = 0
        smodel.steady_state = steady_states
                
    calibration_dict = {}
    if not shocks_file_path is None:
        calibration_dict = loadFile(shocks_file_path,calibration_dict,shocks_names)
    
    if not calibration_file_path is None: 
        if isinstance(calibration_file_path,str):
                calibration_dict = loadFile(calibration_file_path,calibration_dict,names=parameters_names+variables_names+shocks_names)
        elif isinstance(calibration_file_path,list):
            for f in calibration_file_path:
                calibration_dict = loadFile(f,calibration_dict,names=parameters_names+variables_names+shocks_names)
    
    model =  Model(symbolic_model=smodel,m=m,infos=infos)
  
    
    def f_static(y,p,e):
        if np.ndim(p) == 2:
            p = p[:,0]
        z = np.concatenate([y,y,y,e])    
        f = func(z,p,order=0)           
        return f
    
    if model.autodiff or model.jaxdiff:
        func = model.functions["f_func"]
    else:
        func = model.functions["f_dynamic"]
        
    model.functions["f_static"] = f_static
 
    return model

          
   