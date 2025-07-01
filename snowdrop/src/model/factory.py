### Factory produces `Model' class instances

import os,sys,re
import numpy as np
import pandas as pd
from snowdrop.src.utils.load import read_file_or_url
from snowdrop.src.model.model import Model
from snowdrop.src.model.interface import Interface


def import_model(fname,order=1,return_interface=True,check=True,hist=None,boundary_conditions_path=None,
                 exogenous=None,shocks_file_path=None,steady_state_file_path=None,measurement_file_path=None,
                 calibration_file_path=None,calibration={},tag_variables=None,tag_shocks=None,
                 tag_parameters=None,tag_equations=None,tag_measurement_variables=None,
                 tag_measurement_equations=None,options={},conditions={},debug=False):
    """
    Parse a model file and create a model object.
    
    Parameters:
        :param fname: Path to model file.
        :type fname: str.
        :param order: Approximation order of solution of the non-linear system of equations.
        :type order: int.
        :param return_interface: if True returns Interface object.
        :type return_interface: bool.
        :param check: If True checks syntax of model file.
        :type check: bool.
        :param hist: Path to history file.
        :type hist: str.
        :param boundary_conditions_path: Path to the boundary conditions excel file.  This file contains initial and terminal conditions.
        :type boundary_conditions_path: str.
        :param exogenous: Exogenous variables list.
        :type exogenous: list.
        :param shocks_file_path: Path to a file containing shock values.
        :type shocks_file_path: str.
        :param steady_state_file_path: Path to a file containing steady-state values.
        :type steady_state_file_path: str.
        :param measurement_file_path: Path to a file with measurement data.
        :type measurement_file_path: str.
        :param calibration_file_path: Path to calibration files or a file.
        :type calibration_file_path: list or str.
        :param calibration: Map with values of calibrated parameters and starting values of endogenous variables.
        :type calibration: dict.
        :param variables: Tag for endogenous variables section.
        :type variables: list, optional.
        :param tag_shocks: Tag for shock variables section.
        :type tag_shocks: str, optional.
        :param tag_parameters: Tag for parameters section.
        :type tag_parameters: str, optional.
        :param tag_equations: Tag for equations section.
        :type tag_equations: str, optional.
        :param tag_measurement_variables: Tag for measurement variables section.
        :type tag_measurement_variables: str, optional.
        :param tag_measurement_equations: Tag for measurement equations section.
        :type tag_measurement_equations: str, optional.
        :param options: Dictionary of options.
        :type options: dict, optional.
        :param conditions: Dictionary of preprocessing directives.
        :type conditions: dict, optional.
        :param debug: If set to True prints information on Iris model file sections. The default is False.
        :type debug: bool, optional.
            
    Returns:
        model : Interface.
            `Interface' object.
    """
    name = os.path.basename(fname)
    ext = os.path.splitext(fname)[1]
    infos = {'name': name,'filename' : fname} 
    
    #print(f"Model file: {fname}")
    
    if ext == ".mod":        # Dynare model file
        from snowdrop.src.utils.getDynareData import readDynareModelFile
        
        eqs,variables,variables_values,shocks,shock_values,params,mapCalibration,labels = \
            readDynareModelFile(file_path=fname,conditions=conditions,bFillValues=False)
        
        calibration = {**calibration, **mapCalibration}
        # Set default values for missing calibration parameters
        for k in params:
            if not k in calibration:
                calibration[k] = 1.0
        
        for k,v in zip(variables,variables_values):
            calibration[k] = v
            
        # Set default values for missing starting values of endogenous variables
        for k in variables:
            if not k in calibration:
                calibration[k] = 0.0
                
        # Variables labels
        if bool(labels):
            var_labels = dict()
            for var,lbl in zip(variables,labels):
                var_labels[var] = lbl
        else:
            var_labels = {}
            
        interface = getModel(name=name,eqs=eqs,variables=variables,parameters=params,shocks=shocks,
                          shocks_file_path=shocks_file_path,calibration=calibration,var_labels=var_labels,
                          return_interface=return_interface,options=options,infos=infos)
        
        
    elif ext == ".model":    # Iris model file
        from snowdrop.src.utils.getIrisData import readIrisModelFile
        
        eqs,measEqs,params,variables,measVar,measEqs,shocks,measShocks,ss,labels = \
            readIrisModelFile(file_path=fname,bFillValues=False,
                              strVariables=["!variables","!transition_variables"],strShocks =  ["!shocks","!transition_shocks"],
                              strParameters = "!parameters",strEquations = ["!equations","!transition_equations"],
                              strMeasurementVariables="!measurement_variables",
                              strMeasurementEquations="!measurement_equations",
                              strMeasuarementShocks="!measurement_shocks")  
        
        # Set default values for missing calibration parameters
        for k in params:
            if not k in calibration or np.isnan(calibration[k]):
                calibration[k] = 1.0
            
        # Set default values for missing starting values of endogenous variables
        for k in variables:
            if not k in calibration:
                calibration[k] = 0.0
            
        var_labels = dict()
        for var,lbl in zip(variables,labels):
            var_labels[var] = lbl
            
        interface = getModel(name=name,eqs=eqs,meas_eqs=measEqs,variables=variables,
                         parameters=params,shocks=shocks,meas_shocks=measShocks,
                         shocks_file_path=shocks_file_path,meas_variables=measVar,
                         calibration=calibration,var_labels=var_labels,
                         return_interface=return_interface,options=options,infos=infos)
            
    elif ext == ".xml":      # Sirius model file
        from snowdrop.src.utils.getXmlData import readXmlModelFile
        
        eqs,variables,variables_values,shocks,params,params_values,mapCalibration,labels,freq = readXmlModelFile(file_path=fname,bFillValues=False)
        
        if not bool(options):
            options = {}
        options["frequency"] = freq
        
        calibration = {**calibration, **mapCalibration}
        
        # Set default values for missing starting values of endogenous variables
        for k in variables:
            if not k in calibration:
                calibration[k] = 0.0
                
        var_labels = dict()
        for var,lbl in zip(variables,labels):
            var_labels[var] = lbl
                
        interface = getModel(name=name,eqs=eqs,variables=variables,parameters=params,shocks=shocks,
                         shocks_file_path=shocks_file_path,calibration=calibration,var_labels=var_labels,
                         return_interface=return_interface,options=options,infos=infos)
        
    elif ext == ".inp":      # Troll model file
        from snowdrop.src.utils.getTrollData import readTrollModelFile
        
        eqs,variables,variables_values,exogVars,params,param_names,shocks,labels,eqLabels,comments,mapCalibration,undefinedParameters \
                    = readTrollModelFile(file_path=fname,bFillValues=False)         
        calibration = {**calibration, **mapCalibration}
        
        # Set default values for missing starting values of endogenous variables
        for k in variables:
            if not k in calibration or np.isnan(calibration[k]):
                calibration[k] = 0.0
                
        interface = getModel(name=name,eqs=eqs,variables=variables,parameters=params,shocks=shocks,
                             shocks_file_path=shocks_file_path,calibration=calibration,
                             return_interface=return_interface,options=options,infos=infos)
        
        interface.eqLabels = eqLabels
        
    elif ext == ".yaml":     # YAML model file
        txt,labels = read_file_or_url(fname,conditions=conditions)
        
        # if not calibration_file_path is None and os.path.exists(calibration_file_path):
        #     txt2 = txt + "\ncalibration:\n"
        #     with open(calibration_file_path) as f:
        #        for line in f:
        #            txt2 += "  " + line.strip() + "\n"
        if "file" in txt:
            arr = txt.split("\n")
            txt2 = []
            dir_name = os.path.dirname(fname)
            for line in arr:
                if line.strip().startswith("#"):
                    txt2.append(line)
                elif "file" in line:
                    ind = line.index("file")
                    ln = line[ind+4:]
                    if ":" in ln:
                        ind = ln.index(":")
                        fn = ln[ind+1:].replace("\n","").replace("'","").replace('"','').strip()
                        fn = os.path.abspath(os.path.join(dir_name,fn))
                        if os.path.exists(fn):
                            incl,_ = read_file_or_url(fn)
                            txt2.append(incl)
                else:
                    txt2.append(line)
            txt2 = "".join(txt2)
        else:
            txt2 = txt
            
        txt = txt.replace(",,",",")
            
        with open(fname+".txt", "w") as f:
            f.write(txt)
            
       
        if check:
            from snowdrop.src.misc.linter import lint
            output = lint(txt2)
            if len(output) > 0:
                print("\nModel file exceptions:")
                print(output)
                print()
                raise Exception("Model file exception.")
                
        txt = txt.replace('^', '**')
     
        interface = instantiate_model(txt=txt,order=order,return_interface=return_interface,
                                      filename=fname,hist=hist,labels=labels,calibration=calibration,
                                      boundary_conditions_path=boundary_conditions_path,
                                      exogenous=exogenous,shocks_file_path=shocks_file_path,
                                      steady_state_file_path=steady_state_file_path,
                                      calibration_file_path=calibration_file_path,
                                      measurement_file_path=measurement_file_path)

    return interface


def instantiate_model(txt='',order=1,data=None,exog_data={},return_interface=False,filename='',hist=None,
                      boundary_conditions_path=None,exogenous=None,calibration={},shocks_file_path=None,
                      exog_file_path=None,steady_state_file_path=None,calibration_file_path=None,labels={},
                      measurement_file_path=None,infos={}):
    """
    Parse a model file and create an instance of a  model.
    
    Parameters:
        :param txt: Content of model file.
        :type txt: str.
        :param order: Approximation order of solution of the non-linear system of equations.
        :type order: int.
        :param data: Dictionary of model file.
        :type data: dict.
        :param exog_data: Exogenous time series.
        :type exog_data: dict.
        :param return_interface: If True returns `Interface' object.
        :type return_interface: bool.
        :param filename: Model file name.
        :type filename: str.
        :param hist: Path to history file.
        :type hist: str.
        :param boundary_conditions_path: Path to the boundary conditions excel file.  This file contains initial and terminal conditions.
        :type boundary_conditions_path: str.
        :param exogenous: Exogenous variables list.
        :type exogenous: list.
        :param shocks_file_path: Path to shock file.
        :type shocks_file_path: str.
        :param exog_file_path: Path to exogenous series data file.
        :type exog_file_path: str.
        :param steady_state_file_path: Path to a steady-state file.
        :type steady_state_file_path: str.
        :param measurement_file_path: Path to a file with measurement data.
        :type measurement_file_path: str.
        :param calibration_file_path: Path to calibration files.
        :type calibration_file_path: list.
        
    Returns:
        model : Model.
            Model object.
    """
    from snowdrop.src.utils.equations import fixEquations
    from snowdrop.src.utils.equations import get_steady_state_equations
    from snowdrop.src.utils.load import loadYaml,loadFile
    from snowdrop.src.misc.termcolor import cprint
    from os import path
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
 
    model_path = path.dirname(filename)
    if data is None:
        data = loadYaml(model_path,txt)
        
    symbols = data['symbols'] 
    if not exogenous is None:
        symbols['exogenous'] = exogenous
    elif 'exogenous' in symbols:
        symbols['exogenous'] = symbols.pop('exogenous')
    else:
        symbols['exogenous'] = []
    if 'endogenous' in symbols and not 'variables' in symbols:
        symbols['variables'] = symbols.pop('endogenous')
    if 'log_variables' in symbols:
        log_variables = symbols['log_variables']
        symbols['variables'] += log_variables 
    else:
        log_variables = []
    if not 'shocks' in symbols:
        symbols['shocks'] = []
    if 'steady_state' in data:
        steady_state = data['steady_state']
    else:
        steady_state = None  
    if not 'parameters' in symbols:
        symbols['parameters'] = []
    if not 'measurement_parameters' in symbols:
        symbols['measurement_parameters'] = []
    if "labels" in data:
        symbols["variables_labels"] = {**labels,**data["labels"]}
    else:
        symbols["variables_labels"] = labels
    if "eqComments" in data:
        eqs_comments = data["eqComments"]
    else:
        eqs_comments = []


    # Get definitions, parameters, and options
    eqs = data["equations"]
    endog = symbols['variables']
    symbols['endogenous'] = endog
    definitions = data.get('definitions', {})
    if not bool(calibration):
        calibration = data.get('calibration', {})
    param_names = symbols['parameters']
    options = data.get('options', {})
    data_sources = data.get('data_sources', {})
    
    if 'file' in symbols['shocks']: 
        options['shock_values'],options['periods'] = loadFile(model_path+"\\"+symbols['shocks']['file'],calibration,names=symbols['shocks'],bShocks=True)
    if not shocks_file_path is None:
        options['shock_values'],options['periods'] = loadFile(shocks_file_path,calibration,names=symbols['shocks'],bShocks=True)
        
    if not calibration_file_path is None: 
        if isinstance(calibration_file_path,str):
            calibration = loadFile(calibration_file_path,calibration,names=symbols['parameters']+symbols['variables']+symbols['shocks'])
        elif isinstance(calibration_file_path,list):
            for f in calibration_file_path:
                calibration = loadFile(f,calibration,names=symbols['parameters']+symbols['variables']+symbols['shocks'])

        
    if 'file' in options: 
        options = loadFile(model_path + "\\" + options.pop('file'),options)
    
    for exo in symbols['exogenous']:
        if exo in calibration:
            ser = calibration[exo]
            if not isinstance(ser,pd.Series):
                ser = pd.Series(data=ser,index=[i+1 for i in range(len(ser))])
            exog_data[exo] = ser
            del calibration[exo]
        
    # Extract equation labels
    new_eqs = []; eqs_labels = []
    for eq in eqs:
        if isinstance(eq,dict):
            for k in eq:
                new_eqs.append(eq[k])
                k = k.strip()
                arr = k.split(" ")
                if len(arr) > 1:
                    eqs_labels.append(arr[-1]) 
                else:    
                    eqs_labels.append(k) 
        else:
            new_eqs.append(eq)
            
            
    # Build steady state equations
    ss_equations = get_steady_state_equations(new_eqs,endog,param_names,symbols["shocks"],symbols['exogenous'])
        
    #Add new equations and new variables if variables leads/lags greater than one.
    map_new_endog = {}
    num_new_eqs = 0
    if 'variables' in symbols:
        params = symbols.get('parameters', {})
        mod_eqs, new_endog, map_new_endog, leads, lags = fixEquations(eqs=new_eqs,endog=endog,params=params,tagBeg='(',tagEnd=')',b=False)
        if len(new_endog) > 0:
            num_new_eqs = len(mod_eqs) - len(new_eqs)
            data['equations'] = mod_eqs
            var = list(set(endog + new_endog))
            symbols['new_variables'] = new_endog
            symbols['variables'] = var
        else:
            data['equations'] = new_eqs
    else:
        data['equations'] = new_eqs
 
    name = data['name']
    equations = data['equations']
    
    delimiters = " ", ",", ";", "*", "/", ":", "+", "-", "=", "(", ")"
    regexPattern = '|'.join(map(re.escape, delimiters))
    
    # Build equations labels
    m = {}; mm = {}
    for i in range(len(eqs_labels),len(equations)):
        eq  = equations[i] 
        arr = re.split(regexPattern,eq)
        m[i] = [x for x in arr if x in endog]
        if '=' in eq:
            ind  = eq.index('=')
            left = eq[:ind]
            mm[i] = eq[1+ind:]
            arr2 = re.split(regexPattern,left)
            arr2 = [x for x in arr2 if x in endog]
            label = arr2[0].strip() if len(arr2) > 0 else str(i)
            # if not "_minus_" in label and not "_plus_" in label:
            #     arr2 = [x for x in arr2 if x in endog]
            #     label = arr2[0].strip() if len(arr2) > 0 else str(i)
            # if len(arr2) > 1:
            #     eqs_labels.append(str(i))
            if label in eqs_labels:
                eqs_labels.append(str(i))
            else:
                eqs_labels.append(label)
        else:
            eqs_labels.append(str(i))
            m[i] = eq
            
    undefined = [int(x) for x in eqs_labels if x.isdigit()]
        
    # Check if equations labels are unique
    if len(undefined) > 0:    
        import networkx as nx
        from networkx.algorithms import bipartite
        
        lbls = [x for i,x in enumerate(eqs_labels) if not i in undefined]
        m = {k:m[k] for k in m if k in undefined}
        top_nodes = list(m.keys())  
        bottom_nodes = list(set(endog)-set(lbls))
        G = nx.Graph()           
        G.add_nodes_from(top_nodes, bipartite=0)
        G.add_nodes_from(bottom_nodes, bipartite=1) 
        for k in m:
            for k2 in m[k]:
                if k2 in bottom_nodes:
                    G.add_edge(k,k2)
     
        try:
            if len(m) > 0:
                # Obtain the minimum weight full matching (aka equations to variables perfect matching)
                matching = bipartite.matching.minimum_weight_full_matching(G,top_nodes,"weight")
                for k in top_nodes:
                    eqs_labels[k] = matching[k]
                        
        except:
            try:
                m = {}
                for i in range(len(equations)):
                    eq  = equations[i] 
                    arr = re.split(regexPattern,eq)
                    m[i] = [x for x in arr if x in endog]
                    
                top_nodes = list(m.keys())  
                bottom_nodes = endog
                G = nx.Graph()           
                G.add_nodes_from(top_nodes, bipartite=0)
                G.add_nodes_from(bottom_nodes, bipartite=1) 
                for k in m:
                    for k2 in m[k]:
                        if k2 in bottom_nodes:
                            G.add_edge(k,k2)
                # Obtain the minimum weight full matching (aka equations to variables perfect matching)
                matching = bipartite.matching.minimum_weight_full_matching(G,top_nodes,"weight")
                for k in top_nodes:
                    if k in matching:
                        eqs_labels[k] = matching[k]
                    else:
                        #cprint(f"Label for equation {k} is missing","red")
                        eqs_labels[k] = str(k)
            except:
                cprint("Unable to match variables to equations","red")
                
    symbols['equations_comments'] = eqs_comments
    
    if not steady_state_file_path is None:
        ms = loadFile(path=steady_state_file_path,names=symbols['variables'])
        steady_state = {}
        for n in symbols['variables']:
            if n in ms:
                steady_state[n] = ms[n]
            else:
                steady_state[n] = 0
    
    if 'measurement_equations' in data:
        meas_eqs = measurement_equations = data['measurement_equations']
        meas_params = symbols.get('measurement_parameters', {})
        delimiters = " ", ",", ";", "*", "/", ":", "+", "-", "^", "{", "}", "(", ")", "="
        regexPattern = '|'.join(map(re.escape, delimiters))        
        # Re-arrange measurement variables order according to equations 
        if 'measurement_variables' in symbols: 
            vm = symbols['measurement_variables']
            meas_var = []
            for eq in meas_eqs:
                arr = re.split(regexPattern,eq)
                arr = list(filter(None,arr))
                bExist = False
                for v in arr:
                    if v in vm+meas_params:
                        meas_var.append(v.strip())
                        bExist = True
                        break
                if not bExist:
                    cprint("Variable {0} is not present in the list of measurement variables or parameters. Please correct measurement variables.".format(v),"red")
                    sys.exit(-1)
                    
            ### Filter out extra observation variables and measurement equations.
            # We check if observations are available in a data file and if not we apply filter.
            if not measurement_file_path is None and os.path.exists(measurement_file_path):
                ext = measurement_file_path.split(".")[-1].lower()
                if ext == 'xlsx' or ext == 'xls':
                    meas_df = pd.read_excel(measurement_file_path,header=0,index_col=0,parse_dates=True)
                else:
                    meas_df = pd.read_csv(filepath_or_buffer=measurement_file_path,sep=',',header=0,index_col=0,parse_dates=True,infer_datetime_format=True)
                
                meas_var = [x for x in meas_var if x in meas_df.columns]
                measurement_equations = []
                for eq in meas_eqs:
                    arr = re.split(regexPattern,eq)
                    arr = list(filter(None,arr))
                    b_meas_eq = False
                    for v in meas_var:
                        if v in arr:
                            #print(v,": ",eq)
                            b_meas_eq = True
                    if b_meas_eq:
                        measurement_equations.append(eq) 
                if not bool(measurement_equations):
                    cprint("None of the observations is available in a data file.  Please update list of measurement variables.","red")
                    sys.exit(-1)
                    
            symbols['measurement_variables'] = meas_var
            # Assign measurement variables to zero
            for v in vm:
                if not v in calibration:
                    if v.lower().startswith("obs_"):
                        vname = v[4:]
                        calibration[v] = calibration.get(vname,0)
                    elif v.lower().endswith("_meas"):
                        vname = v[:-4]
                        calibration[v] = calibration.get(vname,0)
                    else:
                        calibration[v] = 0
                
        symbols['measurement_parameters'] += meas_params
    else:
        measurement_equations = None
        meas_params = []
  
    # Build a list of variables in each of the equations
    from snowdrop.src.preprocessor.symbolic import stringify
    delimiters = " ", ",", ";", "*", "/", ":", "=", "(", ")", "+", "-"
    regexPattern = '|'.join(map(re.escape, delimiters))
    symb = symbols['variables'] + symbols['shocks']
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
            if v in symb:
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
        
    if not bool(infos):
        infos = {'filename': filename,'name': name}
 
    # all symbols are initialized to nan
    # except shocks and values which are initialized to 0
    initial_values = {
        'shocks': 0,
        'measurement_shocks': 0,
        'expectations': 0,
        'values': 0,
        'new_variables': 0,
        'states': float('nan')
    }
 
    # variables defined by a model equation default to using these definitions
    initialized_from_model = {
        'variables': 'variables', 
        'shocks': 'shocks', 
        'parameters': 'parameters', 
        'measurement_parameters': 'measurement_parameters', 
        'variables_labels': 'variables_labels', 
        'endogenous' : 'endogenous', 
        'shock_values': 'shock_values',
        'equations_comments': 'equations_comments',
        'equations_labels': 'equations_labels'     
    }
 
    for k in symbols['shocks']:
        calibration[k] = 0
    
    for v in log_variables:
        if v in calibration:
            calibration[v] = np.exp(calibration[v])
        
    for k, v in definitions.items():
        if k not in calibration:
            calibration[k] = v
 
    for symbol_group in symbols:
        if symbol_group not in initialized_from_model:
            if symbol_group in initial_values:
                default = initial_values[symbol_group]
            else:
                default =  float('nan')
            for s in symbols[symbol_group]:
                if s not in calibration:
                    #print(symbol_group,default)
                    calibration[s] = default
           
    # Assign starting values for new variables
    for k in map_new_endog:
        new_vars =  map_new_endog[k]
        if k in calibration:
            for new_var in new_vars:
                calibration[new_var] = calibration[k] 
                
    # Set starting and terminal values from a history file
    # The first line of this file contains starting values 
    # and the last - terminal values
    term_values = None
    if not boundary_conditions_path is None:
        name, ext = os.path.splitext(boundary_conditions_path)
        if ext.lower() == ".csv":
            i = 0
            with open(boundary_conditions_path, "r") as f:
                for line in f:
                    i += 1
                    if i == 1:
                        header = line.replace('\n','').replace('"','').split(",")
                    elif i == 2:
                        start_values = line.replace('\n','').replace('"','').split(",")
                    else:
                        pass
                final_values = line.replace('\n','').replace('"','').split(",")
                
        elif ext.lower() == ".xlsx" or ext.lower() == ".xls":
            df = pd.read_excel(boundary_conditions_path,header=0,index_col=0,parse_dates=True) 
            header = list(df.columns.values)
            start_values = df.iloc[0].values
            final_values = df.iloc[-1] .values
            
        else:
            cprint("Only files with extension csv, xls and xlsx are supprted for history","red")
            
        keys = calibration.keys()
        term_values = {}
        for k in keys:
            v = calibration[k]
            if k in header:
                ind = header.index(k)
                val = str(start_values[ind])
                if not val is None and len(val) > 0 and not val.upper == "NONE":
                    calibration[k] = float(val)
                    ls = [x for x in keys if x.startswith(k+"_m_") or x.startswith(k+"_p_")]
                    for e in ls:
                        calibration[e] = float(val)
                val = str(final_values[ind])
                if not val is None and len(val) > 0 and not val.upper == "NONE":
                    term_values[k] = float(val)
                    ls = [x for x in keys if x.startswith(k+"_m_") or x.startswith(k+"_p_")]
                    for e in ls:
                        term_values[e] = float(val)                   
                                
    # Read exogenous values
    param_names = symbols.get('parameters', {})
    if not exog_file_path is None:
        df = pd.read_csv(exog_file_path,sep=',',header=0,parse_dates=True,infer_datetime_format=True)
        column_names = list(df)
        names = [x for x in column_names if x in param_names]
        for n in names:
            values = list(df[n])
            calibration[n] = values
            
    # Read parameter prior distribution
    priors = None
    if 'estimated_parameters' in data:
        priors = {}
        est_params = data['estimated_parameters']
        if not est_params is None:
            for p in est_params:
                if "#" in p:
                    ind = p.index("#")
                    p = p[:ind]
                arr = p.split(',')
                n,initial,lower,upper,distr = p.split(',')[:5]
                params = [initial,lower,upper]+arr[5:]
                params = np.array(params,dtype=float)
                priors[n] = {'distribution': distr.strip(), 'parameters': params}
            
            
    if not hist is None:
        from snowdrop.src.model.util import getStartingValues
        var_names = symbols['variables']
        var_values = [calibration[x] for x in var_names]
        var_values,calib,missing = getStartingValues(hist=hist,var_names=var_names,orig_var_values=var_values,options=options)
        calibration = {**calibration, **calib}
                
    arr = []    
    be = data.get('bellman_equation', {}) 
    if bool(be):
        be_eqs = be.get("equations", None)
        utilities = be.get("utilities", None)
        value_functions = be.get("value_functions",None)
        control_variables = be.get("control_variables", None)
        lower_boundary = be.get("lower_boundary", None)
        upper_boundary = be.get("upper_boundary", None)
        d = {
            "utilities":utilities, 
             "equations":be_eqs,
             "value_functions":value_functions,
             "control_variables":control_variables, 
             "lower_boundary":lower_boundary, 
             "upper_boundary":upper_boundary
             }
        arr.append(d)
            
    bellman = arr if bool(arr) else None
            
        
    interface = Interface(name,symbols,equations,ss_equations,calibration,exog_data,order=order,eq_vars=eq_vars,
                          steady_state=steady_state,measurement_equations=measurement_equations,bellman=bellman,
                          terminal_values=term_values,options=options,data_sources=data_sources,
                          measurement_file_path=measurement_file_path,definitions=definitions,priors=priors)
 
    interface.eqLabels = eqs_labels
    interface.numberOfNewEqs = num_new_eqs
    
    if return_interface:
        return interface
 
    model = Model(interface, infos=infos)
    
    
    return model
 
    
def getModel(name,eqs,variables,parameters,shocks,exogenous=[],exog_data={},shocks_file_path=None,
             Solver=None,ss=None,meas_eqs=[],meas_variables=[],var_labels={},meas_shocks=[],
             meas_parameters=[],options={},eqs_labels=[],calibration={},return_interface=False,
             infos={},definitions={},check=True,bCompileAll=True):
    """
    Instantiate a model object based on passed parameters and calibration dictionary.
    
    Parameters:
        :param name: Model name.
        :type name:  str.
        :param eqs: Transition equations.
        :type eqs:  list.
        :param variables:  Endogenous variables.
        :type variables:  list.
        :param parameters: Model parameters.
        :type parameters:  list.
        :param shocks: Shock names.
        :type shocks:  list.
        :param exogenous:  Exogenous variables.
        :type exogenous:  list.
        :param exog_data:  Exogenous data.
        :type exog_data:  dict.
        :param shocks_file_path: Path to shock file.
        :type shocks_file_path: str.
        :param Sover: Solver name.
        :type Silver:  str.
        :param ss: Map with variables names as a key and steady states as values.
        :type ss: dict.
        :param meas_eqs: Measurement equations.
        :type meas_eqs:  list.
        :param meas_variables: Measurement variables. The default is an empty list.
        :type meas_variables:  list.
        :param var_labels: Labels of endogenous variables. The default is an empty dict.
        :type var_labels:  dict.
        :param meas_shocks: Measurement shocks. The default is an empty list.
        :type meas_shocks:  list.
        :param meas_parameters: Measurement parameters. The default is an empty list.
        :type meas_parameters:  list.
        :param options: Model options.
        :type options:  dict.
        :param eqs_labels: Equation labels. The default is an empty list.
        :type eqs_labels:  list.
        :param calibration: Calibration values. The default is an empty dictionary.
        :type calibration:  dict, optional.
        :param return_interface: if True returns `Interface' object.
        :type return_interface: bool.
        :param infos: Brief information on model. The default is an empty dictionary.
        :type infos:  dict, optional.
        :param definitions: Variables definitions. The default is an empty dictionary.
        :type definitions:  dict, optional.
        :param check: If True checks for errors of passed parameters. The default is False.
        :type check:  bool, optional.
        :param bCompileAll: If True compiles all functions source code.
        :type bCompileAll:  bool, optional.

    Returns:
        `Interface` object.

    """
    from snowdrop.src.utils.equations import fixEquations
    from snowdrop.src.utils.equations import get_steady_state_equations
    from snowdrop.src.model.settings import SolverMethod
    
    if not bool(eqs_labels):
        eqs_labels = [str(1+i) for i in range(len(eqs))]
        
    if not bool(infos):
        infos = {'name': name,'filename' : name}   
        
    n_eqs = len(eqs)
    n_meas_eqs = len(meas_eqs)
    n_var = len(variables)
    n_meas_var = len(meas_variables)
    # # Add measurement variables and equations.
    # eqs       += meas_eqs    
    # variables += meas_variables  
   
    if check: # Check syntax of model parameters.
        
        from snowdrop.src.misc.linter import check_all
        symbols = {"variables":variables,"parameters":parameters,"shocks":shocks,"exogenous":exogenous}
        data = {"name":name,"equations":eqs,"symbols":symbols,"options":options,"definitions":definitions}
        check_all(data)
        interface = instantiate_model(data=data,exog_data=exog_data,return_interface=True,calibration=calibration,shocks_file_path=shocks_file_path,infos=infos)
        
    else: # No check assumes that all passed parameters are valid...  Do it on your own risk!
        
        # Build steady state equations
        ss_eqs = get_steady_state_equations(eqs,variables,parameters,shocks,exogenous)
        ss_variables = variables.copy()
        
         #Adds new equations and new variables if variables leads/lags greater than one.
        mod_eqs, new_variables, map_new_variables, leads, lags = fixEquations(eqs=eqs,endog=variables,params=parameters,tagBeg='(',tagEnd=')')
        if len(new_variables) > 0:
            num_new_eqs = len(mod_eqs) - len(eqs)
            eqs = mod_eqs
            variables += new_variables
            for v in new_variables:
                if "_minus_" in v:
                    ind = v.index("_minus_")
                    vv = v[:ind]
                    if vv in calibration:
                       calibration[v] = calibration[vv] 
                elif "_plus_" in v:
                    ind = v.index("_plus_")
                    vv = v[:ind]
                    if vv in calibration:
                       calibration[v] = calibration[vv] 
                else:
                    calibration[v] = 1
        else:
            num_new_eqs = 0
            
        symbols = {"parameters":parameters,"variables":variables,"endogenous":ss_variables,"new_variables":new_variables,
                   "exogenous":exogenous,"measurement_variables":meas_variables,"variables_labels":var_labels,"shocks":shocks, 
                   "measurement_shocks":meas_shocks,"measurement_parameters":meas_parameters,
                   "equations_labels":eqs_labels}
        
        
        from snowdrop.src.preprocessor.symbolic import stringify
        delimiters = " ", ",", ";", "*", "/", ":", "=", "(", ")", "+", "-"
        regexPattern = '|'.join(map(re.escape, delimiters))
        eq_vars = []
        symb = variables + shocks
        for i,eq in enumerate(eqs):
            e = eq.replace(' ','')
            arr = re.split(regexPattern,e)
            arr = list(filter(None,arr))
            ind = -1
            lst = []
            for v in arr:
                ind = e.find(v)
                e = e[ind+len(v):]
                if v in symb:
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
            
            
        interface = Interface(model_name=name,symbols=symbols,equations=eqs,ss_equations=ss_eqs,calibration=calibration,ss=ss,exogenous=exogenous, 
                              exog_data=exog_data,eq_vars=eq_vars,measurement_equations=meas_eqs,options=options,definitions=definitions)  
        interface.numberOfNewEqs = num_new_eqs
    
    if not bool(interface.bellman):
        assert (n_eqs==n_var),"Number of transient equations {0} and number of endogenous variables {1} is different!".format(n_eqs,n_var)
        assert (n_meas_eqs==n_meas_var),"Number of measurement equations {0} and number of measurement variables {1} is different!".format(n_meas_eqs,n_meas_var)

    interface.eqLabels = eqs_labels
    
    if Solver is None:
        interface.SOLVER = SolverMethod.LBJ
    elif Solver.upper() == "SIMS":
        interface.SOLVER = SolverMethod.Sims
    elif Solver.upper() == "VILLEMOT":
        interface.SOLVER = SolverMethod.Villemot
    elif Solver.upper() == "KLEIN":
       interface.SOLVER = SolverMethod.Klein
    elif Solver.upper() == "ANDERSONMOORE":
       interface.SOLVER = SolverMethod.AndersonMoore
    elif Solver.upper() == "BINDERPESARAN":
        interface.SOLVER = SolverMethod.BinderPesaran
    elif Solver.upper() == "BENES":
        interface.SOLVER = SolverMethod.Klein
    elif Solver.upper() == "LBJ":
        interface.SOLVER = SolverMethod.LBJ
    elif Solver.upper() == "LBJAX":
        interface.SOLVER = SolverMethod.LBJax
    elif Solver.upper() == "ABLR":
       interface.SOLVER = SolverMethod.ABLR
    elif Solver.upper() == "FAIRTAYLOR":
        interface.SOLVER = SolverMethod.FairTaylor
    elif Solver.upper() == "BLOCK":
        interface.SOLVER = SolverMethod.Block
    elif Solver.upper() == "GMRES":
        interface.SOLVER = SolverMethod.GMRes
    else:
        interface.SOLVER = SolverMethod.LBJ
    
    if return_interface:
        return interface
    else:
        model = Model(interface, infos=infos, bCompileAll=bCompileAll)
        # if not "shock_values" in model.options or len(model.options["shock_values"])==0:
        #     model.options["shock_values"] = np.zeros(len(model.symbols["shocks"]))
        if np.isnan(model.calibration["shocks"]).any():
            model.calibration["shocks"] = np.zeros(len(model.symbols["shocks"]))
        
        return model

 
if __name__ == "__main__":
    """Main entry point."""
    from snowdrop.src.driver import run as simulate
    
    path = os.path.dirname(os.path.abspath(__file__))    
    file_path = os.path.abspath(os.path.join(path,'../../models/FPAS/model.model'))
    name = os.path.basename(file_path)
 
    interface = import_model(file_path,return_interface=True)        
    infos = {'name' : name,'filename' : file_path}
    model = Model(interface=interface, infos=infos)
    
    print('\nVariables:')
    variables = model.symbols['variables']
    var_values = model.calibration['variables']
    print(variables)
    print('\nParameters:')
    par_names = model.symbols['parameters']
    par_values = model.calibration['parameters']
    pv = [n+"="+str(v) for n,v in zip(par_names,par_values)]
    print(pv)
    
    print(model)
    
    simulate(model=model,Plot=True)
    
    
    
