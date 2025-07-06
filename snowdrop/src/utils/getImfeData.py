# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 17:53:30 2024

@author: A.Goumilevski
"""
import os, sys
import re
import json
import brotli
import zipfile
import numpy as np
import pandas as pd
from pathlib import Path
from io import StringIO
from datetime import date

path = os.path.dirname(os.path.abspath(__file__))
working_dir = os.path.abspath(path + "/../../..")
sys.path.append(working_dir)
os.chdir(working_dir)

#from snowdrop.src.numeric.solver.util import UnionFind
from snowdrop.src.misc.termcolor import cprint
from snowdrop.src.utils.equations import build_func
from snowdrop.src.utils.equations import transformEq
#from snowdrop.src.utils.equations import aggregateEqs
from snowdrop.src.model.factory import getModel

bOneVarEquation = True

delimiters = " ", ",", ";", "*", "/", ":", "+", "-", "^", "=", "{", "}", "(", ")", "[", "]"
regexPattern = "|".join(map(re.escape, delimiters))
        
data_range_start = data_range_end = forecast_range_start = forecast_range_end = None
today = date.today()
year = str(today.year)

def summary(all_sizes):
    import numpy as np
    sizes = sorted(set(all_sizes))
    sizes = np.array(sizes)   
    for x in sizes:
        n = sum(all_sizes==x)
        if n == 1:
            print(f"{n} component of size {x}")
        else:
            print(f"{n} components of size {x}")
           
def getEqsInfo(eqs,variables_names,debug=False):
    """
    Display information on model equations blocks. 

    Parameters
    ----------
    eqs : list.
        Model equations.
    variables_names : list.
        Names of variables.

    Returns
    -------
    Graph object.
    """
    m = dict()
    delimiters = "+","-","*","/","**","^", "(",")","="," "
    regexPattern = '|'.join(map(re.escape, delimiters))
    for i,eq in enumerate(eqs):
        ind   = eq.index("=")
        left  = eq[:ind].strip()
        arr2   = re.split(regexPattern,left)
        arr2   = list(filter(None,arr2))
        for x in arr2:
            if x in variables_names:
                left = x
                break
        right = eq[1+ind:].strip()
        arr   = re.split(regexPattern,right)
        arr   = list(filter(None,arr))
        m[left] = [x for x in arr if x in variables_names]
                
    mv = dict((v,k) for k,v in enumerate(variables_names))
    
    import networkx as nx
    G = nx.DiGraph()
    G.add_nodes_from(mv)
    for k in m:
        nds = m[k]
        for k2 in nds:
            G.add_edge(k,k2)
            
            
    if debug:
        sc = [len(c) for c in sorted(nx.strongly_connected_components(G),key=len,reverse=False)]
        print(f"\nNumber of strongly connected components {len(sc)}")
        summary(sc)
        
        cn = [len(c) for c in sorted(nx.connected_components(G.to_undirected()),key=len,reverse=False)]
        print(f"\nNumber of connected components {len(cn)}")  
        summary(cn)
        
        ac = [len(c) for c in sorted(nx.attracting_components(G),key=len,reverse=False)]
        print(f"\nNumber of attracting components {len(ac)}")
        summary(ac)
        
        
        bc = [len(c) for c in sorted(nx.biconnected_components(G.to_undirected()),key=len,reverse=False)]
        print(f"\nNumber of biconnected components {len(bc)}")
        summary(bc)
        
        wc = [len(c) for c in sorted(nx.weakly_connected_components(G),key=len,reverse=False)]
        print(f"\nNumber of weekly connected components {len(wc)}")
        summary(wc)
        
    return G 


def getData(json_path=None,xl_path=None,fpe_file=None,save=False):
    """
    Read and decompress IMFE data.

    Parameters
    ----------
    json_path : str, optional
        Path to json files directory.
    xl_path : str, optional
        Path to excel files directory.
    fpe_file : str, optional
        Path to *.fpe file.
    save : bool, optional
        If True saves data to excel file.

    Returns
    -------
    m : dict
        Dictionary containing information for variables, constants, equations, etc...

    """
    
    m = dict()
    if not json_path is None:
        for subdir, dirs, files in os.walk(json_path):
            for file in files:
                fpath = Path(json_path+"/"+file)
                data = fpath.read_bytes()
                df = _getData(file,data,decompress=True)
                if not df is None and len(df.columns) > 1:
                    if save:
                        df.to_csv(xl_path,index=False)
                    m[file[:-5]] = df 
    elif not xl_path is None:
        for subdir, dirs, files in os.walk(xl_path):
            for file in files:
                fpath = Path(xl_path+"/"+file)
                data = fpath.read_bytes()
                df = _getData(file,data,decompress=False)
                if not df is None and len(df.columns) > 1:
                    if save:
                        df.to_csv(xl_path,index=False)
                    m[file[:-5]] = df             
    elif not fpe_file is None:    
        archive = zipfile.ZipFile(fpe_file, 'r')
        for name in archive.namelist():
            if not os.path.isdir(name):
                with archive.open(name) as f:
                    data = f.read()
                    fname = f.name
                    df = _getData(fname,data,decompress=True)
                    if not df is None and len(df.columns) > 1:
                        if save and not df is None:
                            df.to_csv(xl_path,index=False)
                        name = fname[:-5].split("/")[1]
                        m[name] = df
    return m
            
def _getData(attr,data,decompress):
    """
    Read and decompress IMFE data.

    Parameters
    ----------
    attr : str
        Name of json attribute.
    data : json 
        Formated json data.
    decompress : bool
        If True decompresses data with Brotli algorithm.

    Returns
    -------
    df : dataframe
        DESCRIPTION.

    """
    if decompress and not "framework" in attr:
        try:
            json_data = brotli.decompress(data)
        except:
            return None  
    else:
        json_data = data
    
    if "assumptions" in attr:
        json_data = StringIO(json_data.decode('utf-8'))
        mp = json.load(json_data)
        if not "assumptions" in mp:
            mp = mp[0]
        data = list()
        assumptions = mp["assumptions"] if "assumptions" in mp else {}
        scenario = mp["scenario"] if "scenario" in mp else ""
        periods = []
        for assumption in assumptions:
            alias = assumption["alias"]
            desc = assumption["desc"]
            sector = assumption["sector"]
            annual = assumption["data"]["annual"]
            #monthly = assumption["data"]["monthly"]
            #quarterly = assumption["data"]["quarterly"] 
            periods += annual["periods"]
            values = annual["values"]
            default = annual["default"]
            default_forecast = annual["default_forecast"]
            data.append([alias,desc,sector,default,default_forecast,scenario] + values)
        periods = sorted(list(set(periods)))
        columns = ["alias","desc","sector","default","default_forecast","scenario"] + periods
        df = pd.DataFrame(data=data,columns=columns)  
                        
    elif "data" in attr:
        json_data = StringIO(json_data.decode('utf-8'))
        mp = json.load(json_data)
        periods = list(); data_ = list()
        for x in mp:
            periods += x["data"]["annual"]["periods"]
        periods = list(set(periods))
        periods = sorted(periods)
        columns = ["alias","code","desc","frequencies","sector","db_source"] + periods
        for x in mp:
            alias = x["alias"]
            code = x["code"]
            desc = x["desc"]
            db_source = x["db_source"]
            frequencies = x["frequencies"]
            if isinstance(frequencies,list):
                frequencies = frequencies[0]
            sector = x["sector"]
            annual = x["data"]["annual"]
            #monthly = x["data"]["monthly"]
            #quarterly = x["data"]["quarterly"] 
            lper = annual["periods"]
            lval = annual["values"]
            values = [lval[lper.index(x)] if x in lper else None for i,x in enumerate(periods)]
            data_.append([alias,code,desc,frequencies,sector,db_source] + values)
        df = pd.DataFrame(data=data_,columns=columns)    
            
    else:   
        x = str(json_data,'utf-8')
        x = pd.json_normalize(json.loads(x))
        df = pd.DataFrame(x)
    
    return df

def get_frequencies(json_path=None,xl_path=None,fpe_file=None):
     """
     Return list of frequencies. 

     Parameters
     ----------
     json_path : str, optional
         Path to json files directory.
     xl_path : str, optional
         Path to excel files directory.
     fpe_file : str, optional
         Path to *.fpe file.
     
     Returns
     -------
     A/Q/M frequencies.
     
     """
     mv = {}; m_aggr = {}; aggr_variables = {}
     m = getData(json_path=json_path,xl_path=xl_path,fpe_file=fpe_file)
     df = m["variables"]
     var_aggr = df["aggregation"].to_list()
     var_freq = df["frequencies"].to_list()
     var_alias = df["alias"].to_list()
     m_aggr = dict(zip(var_alias,var_aggr))
     
     # Flatten list of list
     frequencies = [x for xs in var_freq for x in xs]
     frequencies = sorted(list(set(frequencies)))
     if len(frequencies) == 3:
         frequencies = [frequencies[i] for i in [0,2,1]]
     
     for freq in frequencies:
         lst = []
         for v,flist in zip(var_alias,var_freq):
             if freq in flist:
                 lst.append(v)
         mv[freq] = lst
   
     # Find variables that have multiple frequencies
     if len(frequencies) > 1:
         for i in range(1,len(frequencies)):
               freq = frequencies[i]
               aggr_freq = frequencies[i-1]
               var1 = mv[aggr_freq]
               var2 = mv[freq]
               aggr_variables[aggr_freq] = [v for v in var1 if v in var2]
        
     return frequencies,aggr_variables,m_aggr

   
def getModelSpecifications(json_path=None,xl_path=None,fpe_file=None,bHist=False,freq=None,aggr_freq=None,m_aggr=None,save=False,debug=False):
    """
    Read IMFE data required to create model object. 

    Parameters
    ----------
    json_path : str, optional.
        Path to json files directory.
    xl_path : str, optional.
        Path to excel files directory.
    fpe_file : str, optional.
        Path to *.fpe file.
    bHist : bool, optional.
        If True creates model for historic range, otherwise - for forecasting range. The default is False.
    freq : str, optional.
        Frequency (A/Q/M).
    aggr_freq : str, optional.
        Aggregation frequency (A/Q/M).
    m_aggr : dict, optional.
        Interpolated results of simulations at lower frequency.
    save : bool, optional.
        If set saves exogenous series and model calibration parameters into excel files. The default is False.   
    debug : bool, optional.
        If True prints model info. The default is False.

    Returns
    -------
    Model specifications

    """
    lfunc=[]; lvars=[]; largs=[]; leqs = []
    
    global data_range_start,data_range_end,forecast_range_start,forecast_range_end

    
    m = getData(json_path=json_path,xl_path=xl_path,fpe_file=fpe_file,save=save)
    
    if "framework" in m:
        df = m["framework"]
        data_range_start = int(df['frequencies.annual.data_range.start'].values[0])
        data_range_end = int(df['frequencies.annual.data_range.end'].values[0])
        forecast_range_start = int(df['frequencies.annual.forecast_range.start'].values[0])
        forecast_range_end = int(df['frequencies.annual.forecast_range.end'].values[0])
        
    ### Parameters
    if "constants" in m:
        df = m["constants"]
        params = df["alias"].to_list()
        param_values = df["value"].to_list()
        param_sector = df["sector"].to_list()
        #param_labels = dict(zip(params,df["desc"].to_list()))
        calib = dict(zip(params,param_values))
    else: 
        params = list()
        param_values = list()
        param_sector = list()
        #param_labels = dict()
        calib = dict()
        
    ### Variables
    df = m["variables"]
    var_freq = df["frequencies"].to_list()
    bFreq = [freq in x for x in var_freq]
    var_freq = df["frequencies"][bFreq].to_list()
    var_names_alias = df["alias"].to_list()
    var_alias = df["alias"][bFreq].to_list()
    #var_code = df["code"][bFreq].to_list()
    #var_export_code = df["export_code"][bFreq].to_list()
    var_descr = df["desc"][bFreq].to_list()
    #var_scale = np.array(df["scale"][bFreq].to_list())
    #var_title = df["title"][bFreq].to_list()
    #var_units = df["units"][bFreq].to_list()
    #var_aggr = df["aggregation"][bFreq].to_list()
    #var_transform = df["transformation"][bFreq].to_list()
    #var_interpolation = df["interpolation"][bFreq].to_list()
    #var_descr = [var_title[i].upper() if x is None else x for i,x in enumerate(var_descr)] 
    var_labels = dict(zip(var_alias,var_descr))
    #var_sector = df["sector"].to_list()
    
    ### Raw data
    exog_data = dict(); total_range = []; exogenous = []; df_data = None
    if "hard_data" in m:
        df_data = m["hard_data"]
        exogenous = df_data["alias"].to_list()
        #data_freq = df_data["frequencies"].to_list()
        #m_data = dict(zip(exogenous,data_freq))
        #data_code = df_data["code"].to_list()
        #data_labels = df_data["desc"].to_list()
        #data_sector = df_data["sector"].to_list()
        columns = df_data.columns
        data = df_data.values
        ind = [i for i,x in enumerate(columns) if x.isdigit()]
        total_range = sorted(columns[ind])
        for i in range(len(data)):
            z = data[i,ind]
            if sum([np.isnan(s) for s in z]) == len(z):
                z = np.zeros(len(z))
            x = exogenous[i]
            ser = pd.Series(z,pd.to_datetime(total_range))
            ser = ser.infer_objects() 
            ser = ser.bfill().ffill()
            exog_data[x] = ser
        
    if data_range_start is None:
        hist_range = [x for x in total_range if data_range_start <= x < year]
    else:
        hist_range = [str(x) for x in range(data_range_start,1+data_range_end)]
        
    ### Equations
    eqs_expressions = list(); eqs_ids = list(); eqs_labels = list(); eqs_sector = list()
    df = m["equations"]
    eqs_freq = df["frequencies"].to_list()
    bFreq = [freq in x for x in eqs_freq]
    eqs_freq = df["frequencies"][bFreq].to_list()
    ids = df["id"][bFreq].to_list()
    #m_eqs = dict(zip(ids,eqs_freq))
    sector = df["sector"][bFreq].to_list()
    #all_expressions = df["expression"].to_list()
    expressions = df["expression"][bFreq].to_list()
    m_eqs_freq = dict(zip(expressions,eqs_freq))
    for i,eqtn in enumerate(expressions):
        eq = eqtn.replace(" ","")
        if not eq in eqs_expressions:
            new_eq = transformEq(eqtn=eq,var=var_alias,formulas=["pch","pchy"])
            eqs_expressions.append(new_eq)
            eqs_ids.append(ids[i])
            #eqs_labels.append(df["title"][i])
            eqs_labels.append(1+i)
            eqs_sector.append(sector[i])
            
    ### Match variables to equations
    #var_alias = mapVariables(eqs_expressions,var_names_alias,max_match=True)
        
    ### Assumptions
    assumpt_data = dict(); assumpt_hist_data = dict(); assumpt_forecast_data = dict()
    df = m["solver_assumptions"]
    assumption_alias = df["alias"].to_list()
    #assumption_labels = df["desc"].to_list()
    assumption_sector = df["sector"].to_list()
    columns = list(df.columns)
    data = df.values
    ind = [i for i,x in enumerate(columns) if x.isdigit()]
    rng = sorted([columns[i] for i in ind])
    tmp = df["default_forecast"].to_list()
    assumptions_all_forecast = {k:v for k,v in zip(assumption_alias,tmp) if not v is None}
    assumptions = df[rng].values
    
    # Get forecast periods
    if forecast_range_start is None:
        forecast_range = sorted([x for x in rng if not x in hist_range])
    else:
        forecast_range = [str(x) for x in range(forecast_range_start,1+forecast_range_end)]
    
    # Get historic and forecast assumption time series     
    h_rng = [x for x in rng if x in hist_range]     
    f_rng = [x for x in rng if x in forecast_range]
    for i in range(len(data)):
        z = data[i,ind]
        n = sum([isinstance(a,str) or str(a) in ["nan","None"] for a in z])
        if not len(z) == n:
            z = [default_value if v is None or isinstance(v,str)  else v for v in z] 
            assumpt_data[assumption_alias[i]] = pd.Series(z,pd.to_datetime(rng))
        
        z = data[i,ind]    
        n = sum([isinstance(a,str) or str(a) in ["nan","None"] for a in z[:len(h_rng)]])
        if not len(z[:len(h_rng)]) == n:   
            z = [default_value if v is None or isinstance(v,str) else v for v in z]
            assumpt_hist_data[assumption_alias[i]] = pd.Series(z[:len(h_rng)],pd.to_datetime(h_rng))
        
        z = data[i,ind]
        n = sum([isinstance(a,str) or str(a) in ["nan","None"] for a in z[-len(f_rng):]])
        if not len(z[-len(f_rng):]) == n:   
            z = [default_value if v is None or isinstance(v,str)else v for v in z]
            assumpt_forecast_data[assumption_alias[i]] = pd.Series(z[-len(f_rng):],pd.to_datetime(f_rng))                        
         
    # Analyze default assumptions
    assumption_default = df["default"]
    b = assumption_default.isna()
    # check if assumptions data row has a formula or a reference to a cell
    bf = [any([isinstance(s,float) and not np.isnan(s) for s in x if not s is None]) for x in assumptions]
    
    keys = [assumption_alias[i] for i in range(len(assumption_default)) if not b[i]]
    tmp = assumption_default[~b].to_list()
    tmp = [x.replace("{","(").replace("}",")") if isinstance(x,str) else x for x in tmp]
    assumption_default = dict(zip(keys,tmp))

    # Assumption equations
    assumption_expression = []; assumption_expression2 = []
    for k in assumptions_all_forecast:
        eq = f"{k}={assumptions_all_forecast[k]}"
        new_eq = transformEq(eqtn=eq,var=var_alias,formulas=["pch","pchy"])
        assumption_expression.append(new_eq)
        assumption_expression2.append(eq)
   
    # Update parameters if default assumption specifies numeric value
    if bHist:
        params += [x for i,x in enumerate(assumption_alias) if x in calib and not b[i]]
        param_sector += [x for i,x in enumerate(assumption_sector) if x in calib and not b[i]]
    else:
       params += [x for i,x in enumerate(assumption_alias) if x in calib and not b[i] and bf[i]]
       param_sector += [x for i,x in enumerate(assumption_sector) if x in calib and not b[i] and bf[i]]
        
    for k in assumpt_data:
        ser = assumpt_data[k]
        ind = (ser != default_value)
        calib[k] = np.nanmean(ser[ind])
        
    for k in assumption_default:
        v = assumption_default[k]
        if isinstance(v,int) or isinstance(v,float):
            calib[k] = v
            params.append(k)
            if k in exogenous:
                exogenous.remove(k)
                del exog_data[k]
        elif isinstance(v,str):
            v = v.replace("{","(").replace("}",")")
            expressions.append(f"{k}={v}")
            eqs_expressions.append(f"{k}={v}")
            m_eqs_freq[f"{k}={v}"] = [freq]
            arr = re.split(regexPattern,v)
            arr = list(filter(None,arr))
            lst = [x for x in arr if x in var_names_alias]
            var_alias.extend(lst)
            if k in params: 
                params.remove(k)
        
    s_var_alias = set(var_alias) 
    s_params = set(params) 
    var_alias = list(s_var_alias) 
    param_names = list(s_params) 
    
    # Replace nan with default value
    #calib = {k:default_value if np.isnan(v) else v for k, v in calib.items()}
    # Set precentages to zero
    calib = {k:0 if k.endswith("_PCH") and not k in assumption_default else v for k, v in calib.items()}
      
    ################################################################# Historic section
    if bHist:
        cprint("\nHistoric run:","blue")
        
        for k in assumpt_hist_data:
            ser = assumpt_hist_data[k]
            if k in exog_data:
                x = exog_data[k]
                ind = (ser != default_value)
                x.update(ser[ind])
                exog_data[k] = x
            else:
                exog_data[k] = ser
        
        s_exogenous = set(df_data["alias"].to_list()) if not df_data is None else set()
        s_var = s_var_alias-s_params-s_exogenous-set(assumption_default.keys()) 
        var = list(s_var)
        #s_exog = s_var_alias.intersection(s_exogenous)
        
        # Build list of all variables in equations
        variables = list(); var_names = list()
        eqs = list();  one_var_eqs = list(); one_var = list()
        i = -1
        for eq,eq2 in zip(eqs_expressions,expressions):
            i += 1
            arr = re.split(regexPattern,eq)
            arr = list(filter(None,arr))
            x = eq.split("=")[0].strip()
            arr2 = re.split(regexPattern,x)
            arr2 = list(filter(None,arr2))
            #print(arr2)
            for x in arr2:
                if x in var:
                    break
            if not x in exogenous and not x in param_names and not x in variables:
                eqs.append(eq)
                variables.append(x)
                var_names.append(x)
            lst = list(set([x for x in arr if x in var]))
            if len(lst) == 1:
                one_var.append(x)
                eq2 = transformEq(eqtn=eq2,var=var+exogenous,b=True)
                if bOneVarEquation:
                    one_var_eqs.append(eq2)
            var_names.extend(lst)
            
        
        variables = list(set(variables))
        var_names = list(set(var_names))
        extra_var = list( set(var_names)-set(variables))

        #print(len(var_names),len(variables),len(eqs))
        
        #  Set initial values
        for v in var_names:
            if not v in calib:
                calib[v] = default_value
        
        new_exog_var = [f" {x} - " + var_labels[x] if x in var_labels and not var_labels[x] is None else " " + x for x in extra_var]
        if len(new_exog_var) > 0:
            cprint(f"The system is underdetermined - {len(new_exog_var)} extra variable(s):","red")
            cprint('\n'.join(new_exog_var[:10]),'blue')
            print("...")
            for v in extra_var:
                eqs.append(f"{v}={v}(-1)")
                if bOneVarEquation:
                    one_var.append(v)
                    one_var_eqs.append(f"{v}={v}__m1_")
            
        # Replace nan with default value
        calib = {k:default_value if np.isnan(v) and not k in assumption_default else v for k, v in calib.items()}
            
        ma = {}
        for k in assumpt_forecast_data:
            v = assumpt_forecast_data[k]
            try:
                ma[k] = float(v.iloc[0])
                if k in calib:
                    calib[k] = float(v.iloc[0])
            except:
                pass
        
        for k in assumptions_all_forecast:
            v = assumptions_all_forecast[k]
            try:
                ma[k] = float(v)
                if k in calib:
                    calib[k] = float(v)
            except:
                pass

        tot_vars = len(var_alias)
        n_params = len(param_names)
        n_data = len(exog_data)
        n_eqs = len(eqs)
        n_vars = len(var_names)
        n_one_var = len(one_var_eqs)
                
        print(f"\nFrequency: {freq}")
        print(f"   Total # of variables: {tot_vars}")     
        print(f"   # of parameters: {n_params}, # of data series: {n_data}")
        print(f"   # of endogenous variables: {n_vars}, # of equations: {n_eqs}")
        print(f"   # of one variable equations: {n_one_var}")

        # Get one variable functions
        lfunc,lvars,largs,leqs = build_func(var=one_var,exog=exogenous,params=param_names,eqs=one_var_eqs)
                   
        return eqs,var_names,param_names,exogenous,exog_data,\
               None,calib,var_labels,eqs_labels,lfunc,lvars,largs,leqs, \
               one_var,hist_range,ma
               
    ################################################################# Forecast section
    else:    
        
        cprint("\nForecast run:","blue")
                
        # Ensure exogenous data contain only foreast range
        for k in exog_data:
            ser = exog_data[k]
            index = ser.index
            b = index >= forecast_range[0]
            if any(b):
                exog_data[k] = ser[b]
            else:
                exogenous.remove(k)
                var_alias.append(k)
        exog_data = {k:exog_data[k] for k in exog_data if k in exogenous}
                
        # If variable was defined by previos period of hard data, then use this value as a starting value
        for v in var_alias:
            if v in exogenous:
                ser = exog_data[v]
                ind = ser.notna()[::-1].idxmax()
                calib[v] = ser[ind]
            if not v in calib:
                calib[v] = default_value
                
        #exog_data = {k:exog_data[k] for k in exogenous}
        exog_data = {**exog_data,**assumpt_forecast_data}
        exogenous = list(exog_data.keys())
            
        s_assumption_alias = set(assumpt_forecast_data.keys())
        s_var = s_var_alias-s_params-s_assumption_alias; 
        var = list(s_var)
        # Add equations and assumption expressions
        equations = []; equations2 = []; m_dpl = dict()
        eqs = assumption_expression + eqs_expressions
        eqs2 = assumption_expression + expressions
        for i,eq in enumerate(eqs):
            arr = eq.split("=")
            k = arr[0].strip()
            arr2 = re.split(regexPattern,k)
            arr2 = list(filter(None,arr2))
            for k in arr2:
                if k in var:
                    break
            if k in var:
                if k in m_dpl:
                    m_dpl[k] += [i]
                else:
                    m_dpl[k] = [i]
                    equations.append(eq)
                    eq2 = transformEq(eqtn=eqs2[i],var=var+exogenous,b=True)
                    equations2.append(eq2)
                    
        # # Check for duplicate equations
        # duplicate_eqs = list()
        # for k in m_dpl:
        #     arr = m_dpl[k]
        #     if len(arr) > 1:
        #         x = [eqs[i] for i in arr]
        #         duplicate_eqs.append(" -->  ".join(x))
        # if len(duplicate_eqs) > 0:
        #     duplicate_eqs = "\n".join(duplicate_eqs)
        #     cprint(f"Duplicate equations:\n {duplicate_eqs}")
        
        # Build list of all variables in equations
        variables = list(); var_names = list(); indx = list() # index of one variable equations
        eqs = list(); one_var_eqs = list(); one_var = list()
        i = -1
        for eq,eq2 in zip(equations,equations2):
            i += 1
            arr = re.split(regexPattern,eq)
            arr = list(filter(None,arr))
            x = eq.split("=")[0].strip()
            arr2 = re.split(regexPattern,x)
            arr2 = list(filter(None,arr2))
            for x in arr2:
                if x in var:
                    break
            eqs.append(eq)
            variables.append(x)
            var_names.append(x)
            lst = list(set([x for x in arr if x in var]))
            if len(lst) == 1:
                indx.append(i)
                if bOneVarEquation:
                    one_var_eqs.append(eq2)
                    one_var.append(x)
            var_names.extend(lst)
        variables = list(set(variables))
        var_names = list(set(var_names))
        
        # Add equations for extra variables
        extra_var = list( set(var_names)-set(variables))
        new_exog_var = [f" {x} - " + var_labels[x] if x in var_labels and not var_labels[x] is None else " " + x for x in extra_var]
        if len(new_exog_var) > 0:
            cprint(f"The system is underdetermined - {len(new_exog_var)} extra variables:","red")
            cprint('\n'.join(new_exog_var[:10]),'blue')
            print("...")
        
        i = len(equations)
        for v in extra_var:
            indx.append(i)
            i += 1
            eqs.append(f"{v}={v}(-1)")
            if bOneVarEquation:
                one_var.append(v)
                one_var_eqs.append(f"{v}={v}__m1_")
            if v in assumpt_forecast_data:
                try:
                    calib[v] = float(assumpt_forecast_data[v])
                except:
                    calib[v] = default_value
            elif not v in assumption_default:    
                calib[v] = default_value
                
        # One equation variables are treated as exogenous
        if bOneVarEquation:
            exogenous.extend(extra_var)
        
        # Replace nans with default value
        calib = {k:default_value if np.isnan(v) else v for k, v in calib.items()}
        
        # Replace zeros with default value value
        calib = {k:default_value if abs(v)<1.e-8 else v for k, v in calib.items()}
        for k in assumptions_all_forecast:
            v = assumptions_all_forecast[k]
            try:
                calib[k] = float(v)
                params.append(k)
                if bOneVarEquation:
                    var_names.remove(k)
            except:
                pass

        param_names = [x for x in params if x in calib ]
        
        # Get one variable functions. These equations are solved directy
        if bOneVarEquation:
            lfunc,lvars,largs,leqs = build_func(var=one_var,exog=exogenous,params=param_names,eqs=one_var_eqs)
        
        f_range = pd.to_datetime(forecast_range)
        for v in lvars:
            if v in exog_data:
                try:
                    ser = exog_data[v][f_range]
                except:
                    ser = pd.Series([exog_data[v].iloc[-1]]*len(f_range),f_range)
                assumpt_forecast_data[v] = ser
            
        # Exclude one variable equations
        if bOneVarEquation:
            var_names = [x for x in var_names if not x in lvars]
            exogenous.extend(lvars)
            eqs = [eqs[i] for i in range(len(eqs)) if not i in indx]
        
        
        n_eqs = len(eqs)
        n_assumpt = len(assumptions_all_forecast)
        tot_vars = len(var_alias)
        n_vars = len(var_names)
        n_params = len(param_names)
        n_data = len(exog_data)
        n_one_var = len(one_var_eqs)

        print(f"\nTotal # of variables: {tot_vars}")     
        print(f"# of endogenous variables: {n_vars}, # of equations: {n_eqs}")
        print(f"# of parameters: {n_params}, # of assumptions: {n_assumpt}, # of data series: {n_data}")
        print(f"# of one variable equations: {n_one_var}")
        
        # cprint("\nOverwritten equations by imposed assumptions:","blue")
        # cprint("Assumptions          -->            Core","blue") 
        # print(f"{duplicate_eqs}")  
        
        # lst = "\n".join(one_var_eqs)
        # cprint(f"\nList of one variable equations:\n{lst}","green")
        
        if save:
            df = pd.DataFrame(exog_data)
            df.to_csv(working_dir+"/data/IMFE/exog.csv")
            df = pd.DataFrame({k:[v] for k,v in calib.items()})
            df.to_csv(working_dir+"/data/IMFE/calib.csv",index=False)
            
        #exogenous = list(set(exogenous)-set(var_names))  
        return eqs,var_names,param_names,exogenous,exog_data,assumpt_forecast_data, \
               calib,var_labels,eqs_labels,lfunc,lvars,largs,leqs, \
               one_var,forecast_range,None
        
def get_model(json_path=None,xl_path=None,fpe_file=None,bHist=False,freq=None,aggr_freq=None,m_aggr=None,f_calib=None,bCompileAll=True,debug=False):
    """
    Construct model object.

    Parameters
    ----------
    json_path : str, optional
        Path to json files directory.
    xl_path : str, optional
        Path to excel files directory.
    fpe_file : str, optional
        Path to *.fpe file.
    bHist : bool, optional
        If True creates model for historic range, otherwise - for forecasting range. The default is False.
    freq : str, optional.
        Frequency (A/Q/M).
    aggr_freq : str, optional.
        Aggregation frequency (A/Q/M).
    m_aggr : dict, optional.
        Results of simulations at lower frequency.
    f_calib : str, optional.
        Path to excel files with containing calibrated parameters.
    bCompileAll : bool, optional
        If True compiles functions source code.
    debug : bool, optional
        If True prints model info. The default is False.

    Returns
    -------
    model : `Model'
        Model object.
    lfunc : list
        Compiled functions.
    lvars : list
        Variables on the left side of equation.
    largs : list
        Functions arguments.
    one_var : list
        DESCRIPTION.
    params : list
        Equations parameters.
    one_var_eqs : list
        DESCRIPTION.

    """
    #from snowdrop.src.utils.equations import getRHS
    from snowdrop.src.utils.equations import fixEquations
    from snowdrop.src.utils.equations import getMaxLeadsLags
    from snowdrop.src.preprocessor.function_compiler_sympy import compile_higher_order_function
 
    
    eqs,var,params,exogenous,exog_data,assumpt_data, \
    calib,var_labels,eqs_labels,lfunc,lvars,largs,one_var_eqs, \
    one_var,rng,m_assumpt = \
        getModelSpecifications(json_path=json_path,xl_path=xl_path,fpe_file=fpe_file,
                               bHist=bHist,freq=freq,aggr_freq=aggr_freq,m_aggr=m_aggr,debug=debug)
    
    #Sanity check: mv = dict(zip(var,[calib[x] for x in var])); mp = dict(zip(params,[calib[x] for x in params]))   
    if debug:
        getEqsInfo(eqs,var,debug=debug)
        
    if not json_path is None:
        name,ext = os.path.splitext(os.path.basename(os.path.abspath(json_path)))
    elif not fpe_file is None:
        name,ext = os.path.splitext(os.path.basename(os.path.abspath(fpe_file)))
    else:
        name = "IMFE"
        
    if not bCompileAll: 

        new_eqs,new_endog,_,leads,lags = fixEquations(eqs=eqs,endog=var,params=params)
        if len(new_endog) > 0:
            eqs = new_eqs
            var += new_endog
            for v in new_endog:
                calib[v] = default_value
                
        equations = []
        for eq in eqs:
            arr = str.split(eq,'=')
            if len(arr) == 2:
                s = '{} - ({})'.format(arr[0].strip(),arr[1].strip())
                s = str.strip(s)
            else:
                s = eq
            equations.append(s)
            
        # Get equations right-hand-side
        #rhs_eqs,eq_vars = getRHS(eqs,var)
        
        # Constructs arguments of function f([y(1),y,y(-1)],p)
        syms = [(v,1) for v in var] + [(v,0) for v in var] + [(v,-1) for v in var]
        
        # Get maximum of leads and minimum of lags of equations exogenous variables
        max_lead_exog,min_lag_exog,n_fwd_looking_exog,n_bkwd_looking_exog,exog_lead,exoglag = \
            getMaxLeadsLags(eqs=eqs,variables=exogenous)
        
        
        syms_exog = []
        for i in range(min_lag_exog,1+max_lead_exog):
            syms_exog += [(s,i) for s in exogenous]
            
        # Build a list of variables in each of the equations
        from snowdrop.src.preprocessor.symbolic import stringify
        delimiters = " ", ",", ";", "*", "/", ":", "=", "(", ")", "+", "-"
        regexPattern = '|'.join(map(re.escape, delimiters))
        eq_vars = []
        for i,eq in enumerate(equations):
            e = eq.replace(' ','')
            arr = re.split(regexPattern,e)
            arr = list(filter(None,arr))
            ind = -1
            lst = []
            for v in arr:
                ind = e.find(v)
                e = e[ind+len(v):]
                if v in var:
                    if len(e) > 0 and e[0] == '(':
                        ind2 = e.find(')')
                        if ind2 > 0:
                            lead_lag = e[1:ind2]
                            match = re.match(r"[-+]?\d+",lead_lag)
                            if not match is None:
                                try:
                                    i_lead_lag = int(lead_lag)
                                    lst.append(stringify((v,i_lead_lag)))
                                except:
                                    pass
                        else:
                            lst.append(v+'__')
                    else:
                        lst.append(v+'__')
            eq_vars.append(lst) 
        
        functions = {}; functions_src = {}
        if len(equations) > 0:
            func,_,_,_,_,src = compile_higher_order_function(equations=equations,syms=syms,params=params,syms_exog=syms_exog,
                                                                  eq_vars=eq_vars,order=1,function_name='f_dynamic',
                                                                  out='f_dynamic',model_name=name)
            functions['f_dynamic'] = func 
            functions_src['f_dynamic_src'] = src
          
    model = instantiate_model(name=name,eqs=eqs,var=var,params=params,exogenous=exogenous,
                              exog_data=exog_data,assumpt_data=assumpt_data,calib=calib,
                              var_labels=var_labels,eqs_labels=eqs_labels,frequency=freq,
                              rng=rng,bHist=bHist,bCompileAll=bCompileAll and len(eqs)>0)
    
    if not bCompileAll:
        
        def f_static(y,p,e):
            from snowdrop.src.numeric.solver.util import getExogenousData         
            z = np.concatenate([y,y,y]) 
            # Get exogenous time series
            exog = getExogenousData(model)
            exog = np.array(exog)
            exog[exog==0] = 1.e-8
            bHasAttr  = hasattr(func,"py_func")   
            if bHasAttr:
                f = func.py_func(z,p,exog=exog,order=0)  
            else:
                f = func(z,p,exog=exog,order=0)  
            return f
        
        functions['f_static'] = f_static
        functions['f_steady'] = f_static
        
        model.max_lead_exog = max_lead_exog
        model.min_lag_exog = min_lag_exog
        model.functions = functions
        model.functions_src = functions_src  
    
    if not f_calib is None:
        param_names = model.symbols["parameters"] 
        param_values = model.calibration["parameters"].copy()
        df = pd.read_excel(f_calib,sheet_name="Constants")   
        par = list(df["Alias"]); val = list(df["Value"]) #; descr = list(df["Description"])
        for p,v in zip(par,val):
            if p in param_names:
                ind = param_names.index(p)
                param_values[ind] = float(v)
        model.calibration["parameters"] = param_values        
    
    if bool(m_assumpt):
        terminal_values = dict()
        for k in m_assumpt:
            if k in var:
                terminal_values[k] = m_assumpt[k]
        model.terminal_values = terminal_values
        
    if debug:
        cprint("\n\nHistoric Periods" if bHist else "\n\nForecast Periods","blue")
        print(model)
    #model.isLinear = False
    
    # var_names = model.symbols["variables"] 
    # var_values = model.calibration["variables"]
    # mv = dict(zip(var_names,var_values))
    
    
    return model,lfunc,lvars,largs,one_var,exogenous,params,one_var_eqs,rng


def instantiate_model(name,eqs,var,params,exogenous,exog_data,assumpt_data,calib,
                      var_labels={},eqs_labels=[],frequency="annual",
                      rng=None,bHist=False,bCompileAll=False):
    """
    Create model object.

    Parameters
    ----------
    name : str
        Model name.
    eqs : list
        Equations.
    var : list
        List of variables.
    params : list
        List of parameters.
    exogenous : list
        List of exogenous variables.
    exog_data : dict
        Exogenous variables time series.
    assumpt_data : dict
        Assumption data.
    calib : dict
        Dictionary of endogenous variables starting values and parameters vaues.
    var_labels : dict, optional
        Variables labels. The default is {}.
    eqs_labels : list, optional
        Equations labels. The default is [].
    frequency : str, optional
        Frequency. The default is 'annual'.       
    bHist : bool, optional
        If True creates model for history and if False - for forecasting. The default is False.
    bCompileAll : bool, optional
        If True compiles functions source code.
    
    Returns
    -------
    model : `Model'
        Model.

    """
    if assumpt_data is None:
        df = pd.DataFrame(exog_data)
        data = exog_data
    else:
        df = pd.DataFrame(assumpt_data)
        data = {**exog_data,**assumpt_data}
    if frequency == "annual":
        freq = 0
        alias = "YS"
    elif frequency == "quarterly":
        freq = 1
        alias = "QS"
    elif frequency == "monthly":
        freq = 2
        alias = "MS"
    elif frequency == "daily":
        freq = 3
        alias = "D"
    else:
        freq = 0
        alias = "Y"
    if bHist: 
        data_start = date(year=data_range_start,month=1,day=1)
        data_end = date(data_range_end,month=10,day=1)
        if not data_range_start is None and not data_range_end is None:
            data_rng = pd.date_range(start=data_start,end=data_end,freq=alias)
            T = 2 + len(data_rng)
            periods = list(range(1,1+T))
            rng = [[data_range_start,1,1],[data_range_end,10,1]]
            options = {"T":T,"periods":periods,"frequency":freq,"range":rng}
        elif len(df) > 0:
            dates = df.index if rng is None else rng
            start = min(dates)
            end = max(dates)
            rng = [[max(data_range_start,start.year),start.month,start.day],[min(data_range_end,end.year),end.month,end.day]]
            T = 3 + max(data_range_start,start.year) - min(data_range_end,end.year)
            periods = list(range(1,1+T))
            options = {"T":T,"periods":periods,"frequency":freq,"range":rng}
    else:
        forecast_start = date(year=forecast_range_start,month=1,day=1)
        forecast_end = date(forecast_range_end,month=10,day=1)
        if not forecast_range_start is None and not forecast_range_end is None:
            forecast_rng = pd.date_range(start=forecast_start,end=forecast_end,freq=alias)
            T = 3 + len(forecast_rng)
            rng = [[forecast_range_start,1,1],[forecast_range_end+2,10,1]]
            periods = list(range(1,1+T))
            options = {"T":T,"periods":periods,"frequency":freq,"range":rng}
        elif len(df) > 0:
            dates = df.index if rng is None else rng
            start = min(dates)
            end = max(dates)
            rng = [[max(forecast_range_start,start.year),start.month,start.day],[min(forecast_range_end,end.year),end.month,end.day]]
            T = 3 + max(forecast_range_start,start.year) - min(forecast_range_end,end.year)
            periods = list(range(1,1+T))
            options = {"T":T,"periods":periods,"frequency":freq,"range":rng}
        
    # Instantiate model
    model = getModel(name=name,eqs=eqs,variables=var,parameters=params,shocks=[],
                     exogenous=exogenous,exog_data=data,calibration=calib,var_labels=var_labels,
                     eqs_labels=eqs_labels,options=options,return_interface=False,check=False,bCompileAll=bCompileAll)
    
    model.T = T
    return model

def mapVariables(equations,var,max_match=False):
    """
    Bipartite match of variables to equations.

    Parameters
    ----------
    equations : list
        Equations.
    var : list
        Variables.
    max_match : bool
        If true finds maximum bipartite matching in a graph .

    Returns
    -------
    variables : list
        Variables.
    """


    mp = dict(); variables = list(); ind = list()
    
    if max_match:
        from scipy import sparse
        import scipy.sparse.csgraph as csgraph
        
        endog = []
        neq = len(equations)
        nv = len(var)
        # Create a biadjacency matrix
        matrix = np.zeros((neq,nv))
        
        for i,eq in enumerate(equations):
            arr = re.split(regexPattern,eq)
            arr = list(filter(None,arr))
            lst = [i for i,x in enumerate(arr) if x in var]
            matrix[i][lst] = 1
            
        biadjacency_matrix = sparse.csr_matrix(matrix)

        # Find the maximum matching
        try:
            matching = csgraph.maximum_bipartite_matching(biadjacency_matrix)
          
            for k in matching:
                if k >= 0:
                    endog.append(var[k])
        except:
            cprint("Cannot match variables to equations.\n","red")
        if len(endog) < neq:
            return var
        else:    
            return endog
      
    else:
        import networkx as nx
        from networkx.algorithms import bipartite
        
        for i,eq in enumerate(equations):
            arr = re.split(regexPattern,eq)
            arr = list(filter(None,arr))
            x = eq.split("=")[0]
            arr2 = re.split(regexPattern,x)
            arr2 = list(filter(None,arr2))
            for x in arr2:
                if x in var:
                    break
            variables.append(x)
        variables = list(set(variables))
        n_var = len(variables)
        
        if n_var == len(equations):
            return variables
        else:   
            rest_eqs = equations[ind]
            rest_var = list(); top = list(); endog = list()
            for i,eq in enumerate(rest_eqs):
                arr = re.split(regexPattern,eq)
                arr = list(filter(None,arr))
                lst = list(set([x for x in arr if x in var and not x in variables]))
                mp[i] = lst
                top.append(i)
                rest_var.extend(lst)
            rest_var = list(set(rest_var))
                
            G = nx.Graph()           
            G.add_nodes_from(mp, bipartite=0)
            G.add_nodes_from(rest_var, bipartite=1) 
            for i in mp:
                for k in mp[i]:
                    if k in rest_var:
                        G.add_edge(i,k)
            
            bp = bipartite.is_bipartite(G)
            print(f"Graph is bipartite: {bp}")
            # Obtain the minimum weight full matching (aka equations to variables perfect matching)
            try:
                matching = bipartite.matching.minimum_weight_full_matching(G,top,"weight")
                for k in top:
                    i = matching[k]
                    endog.append(k)
            except:
                cprint(f"Cannot match variables {','.join(rest_var)} to Equations \n {','.join(rest_eqs)}","red")
          
            variables += endog
        
    return variables
    
    
def getModelHash(json_path=None,xl_path=None,fpe_file=None):
    """
    Build hash of model variables, parameters, equations and hard data.

    Parameters
    ----------
    json_path : str, optional
        DESCRIPTION. Path to json files directory.
    xl_path : str, optional
        DESCRIPTION. Path to excel files directory.
    fpe_file : str, optional
        DESCRIPTION. Path to *.fpe file.

    Returns
    -------
    h : TYPE
        DESCRIPTION.

    """
    if not json_path is None:
        fdir,file = os.path.split(os.path.abspath(json_path))
        ind = file.index(".")
        f = file[:ind]+".hash"
    elif not fpe_file is None:
        fdir,file = os.path.split(os.path.abspath(fpe_file))
        ind = file.index(".")
        f = file[:ind]+".hash"
    elif not xl_path is None:
        fdir = os.path.abspath(xl_path)
        f = "imfe.hash"
    fpath = os.path.abspath(os.path.join(fdir,f))
    
    if os.path.exists(fpath):
        with open(fpath,'r') as f:
            txt = f.read()
            old_h = int(txt)
    else:
        old_h = -1
    
    m = getData(json_path=json_path,xl_path=xl_path,fpe_file=fpe_file)
    h = 0
    for k in ["constants","variables","equations","hard_data"]:
        if k in m:
            df = m[k]
            columns = list(df.columns)
            for col in columns:
                x = df[col]
                if isinstance(x[0],list):
                    df.drop(columns=col,inplace=True)
            try:
                h += np.sum(pd.util.hash_pandas_object(df))
            except:
                pass
            
    h = int(h)
    if old_h == h:
        return False
    else:  
        with open(fpath,'w') as f:
            f.write(str(h))
        return True
       
        
def computeShocks(new_eqs,m_var,m_aggr):
    """Compute shocks as residuals of aggregate equations."""
    delimiters = " ", ",", ";", "*", "/", ":", "+", "-", "=", "(", ")","^"
    regexPattern = '|'.join(map(re.escape, delimiters))
    
    n = len(new_eqs)
    k = next(iter(m_var))
    T = len(m_var[k])
    shocks = np.zeros((T,n))
    for i,eqtn in enumerate(new_eqs):
        ind = eqtn.index("=")
        #lhs = eqtn[:ind]
        rhs = eqtn[1+ind:]
        arr = re.split(regexPattern,rhs)
        arr = list(filter(None,arr))
        arr1 = [x[:x.index("_AGGREGATED")] for x in arr if "_AGGREGATED" in x]
        v = arr1[0]
        ser1 = m_aggr[v].values
        arr2 = [x[:x.index("[")] for x in arr if "[" in x and x[:x.index("[")] in m_var]
        x = arr2[0]
        ser2 = m_var[x].values
        mp = {f"{x}":ser2}
        
        for t in range(T):
            mp["t"] = t
            eq = rhs.replace(v+"_AGGREGATED[t]",str(ser1[t]))
            shocks[t,i] = eval(eq,mp)
            
    return shocks


default_value = 1.e0

if __name__ == '__main__':
    """The main program."""
    
    #xl_path = os.path.join(working_dir,"snowdrop/data/IMFE/xl")
    #json_path = os.path.join(working_dir,"snowdrop/data/IMFE/json")
    fpe_file = os.path.join(working_dir,"snowdrop/models/IMFE/GUY.fpe")
    #fpe_file = os.path.join(working_dir,"snowdrop/models/IMFE/Fin.fpe")
    #fpe_file = os.path.join(working_dir,"snowdrop/models/IMFE/ALB.fpe")
    #fpe_file = os.path.join(working_dir,"snowdrop/models/IMFE/Fin.fpe")
                     
    model,lfunc,lvars,largs,one_var,exog,one_params,one_var_eqs,frcst_rng  =  \
        get_model(fpe_file=fpe_file,bHist=False,freq="yearly",aggr_freq="quarterly",debug=False)
        
    i = 0
    for var,args,eq in zip(lvars,largs,one_var_eqs):
        i += 1
        print(i,": ",eq," \t\t   Equation Arguments: ",args)
        

    