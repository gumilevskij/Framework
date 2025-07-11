"""
Utility module.

@author: A.Goumilevski
"""
import os
import sys
import pandas as pd
#import datetime as dt
import numpy as np
import ruamel.yaml as yaml
from dataclasses import dataclass
from snowdrop.src.misc.termcolor import cprint
from snowdrop.src.numeric.optimization.util import expand

fpath = os.path.dirname(os.path.abspath(__file__))
working_dir = os.path.abspath(os.path.join(os.path.abspath(fpath + "../..")))

version = yaml.__version__
eqs_labels = []

@dataclass
class Data:
    success: bool
    x: float
    fun: float
    nfev: int = 0
    
def replace_all(old,new,expr):
    while old in expr:
        expr = expr.replace(old,new)
    return expr

def loadLibrary(lib="libpath"):
    """ 
    Simple example of loading and using the system C library from Python.
    """
    import platform
    import ctypes, ctypes.util
    
    basepath = os.path.dirname(os.path.abspath(__file__))
    lib_dir = os.path.abspath(os.path.join(basepath, '../../../bin'))
    if os.path.exists(lib_dir) and not lib_dir in sys.path:
        sys.path.append(lib_dir)
    
    # Get the path to the system C library.
    # If library is not found on the system path, set path explicitely.
    if platform.system() == "Windows":
        path_libc = ctypes.util.find_library(lib)
        if path_libc is None:
            path_libc = os.path.join(lib_dir, lib+".dll")
    else:
        path_libc = ctypes.util.find_library(lib)
        if path_libc is None:
            path_libc = os.path.join(lib_dir, lib+".so")
           
    path_dep = path_libc.replace('libpath','libpath50')
    
    # Get a handle to the sytem C library
    try:
        ctypes.CDLL(name=path_dep, mode=ctypes.RTLD_GLOBAL)
        libc = ctypes.CDLL(path_libc)
    except OSError as ex:
        cprint(f"\n{ex}\nUnable to load the C++ library {lib}!  Exitting...","red")
        sys.exit()
    
    cprint(f'Succesfully loaded the system C library from "{path_libc}"',"green")
        

    # Set the argument and result types of function.
    libc.path_solver.restype  = ctypes.c_long
    libc.path_solver.argtypes = [ ctypes.c_int, ctypes.c_int,
                                  np.ctypeslib.ndpointer(dtype=np.double),
                                  np.ctypeslib.ndpointer(dtype=np.double),
                                  np.ctypeslib.ndpointer(dtype=np.double),
                                  np.ctypeslib.ndpointer(dtype=np.double),
                                  np.ctypeslib.ndpointer(dtype=np.double)
                                ]
    return libc

def getIndex(e,ind):    
    """
    Find the first matching occurance of open bracket.

    Parameters:
        :param e: Expression.
        :type e: str.
        :param ind: Starting index.
        :type ind: int.
        :returns: Index of the matching open bracket.
    """
    ind1 = [i for i in range(len(e)) if i>ind and e[i]=="("]
    ind2 = [i for i in range(len(e)) if i>ind and e[i]==")"]
    index = sorted(ind1+ind2)
    s = 0
    for i in index:
        if i in ind1:
            s += +1
        elif i in ind2:
            s += -1
        if s == 0:
            index = 1+i
            break
       
    return index
        


 
def fix(eqs,model_eqs):
    """
    Get equations, labels of equations and complementarity conditions.

    Parameters:
        :param eqs: Equations.
        :type eqs: list.
        :param model_eqs: Model equations to solve.
        :type model_eqs: list.
        :returns: List of equations and complementarity conditions.
    """
    from collections import OrderedDict
    global eqs_labels
    
    arr = []; names = []; cond = []
    complementarity = OrderedDict()
    for x in model_eqs:
        z = x.split(".")
        names.append(z[0])
        if len(z) > 1:
            cond.append(z[1])
        else:
            cond.append(None)
            
    for i,e in enumerate(eqs):
        if isinstance(e,dict):
            for k in e:
                if "(" in k:
                    ind = k.index("(")
                    lbl = k[:ind]
                else:
                    lbl = k
                if bool(names):
                    if lbl in names:
                        eqs_labels.append(k)
                        arr.append(e[k])
                        ind = names.index(lbl)
                        complementarity[lbl] = cond[ind]
                else:
                    eqs_labels.append(k)
                    arr.append(e[k])
        else:
            eqs_labels.append(str(1+i))
            arr.append(e)
        
    return arr,complementarity   
        
def getLabels(keys,m):
    
    labels = {}
    for k in m:
        if "(" in k:
            ind = k.index("(")
            key = k[:ind].strip()
            for x in keys:
                if x.startswith(key+"_"):
                    labels[x] = m[k]
        else:
            labels[k] = m[k]
            
    return labels    
        
def importModel(fpath):
    """
    Parse a model file and create a model object.
    
    Parameters:
        :param fpath: Path to model file.
        :type fpath: str.
        
    """
    global eqs_labels
    import re
    from snowdrop.src.model.interface import Interface
    from snowdrop.src.model.model import Model
    
    name = "Model"
    solver = None; method = None
    symbols = {}; calibration = {}; constraints = {}; obj = {}; labels = {}; options = {}
    variables = []; parameters = []; equations = []
    
    with open(fpath,  encoding='utf8') as f:
        txt = f.read()
        txt = txt.replace('^', '**')
        if version >= '0.18':
            data = yaml.YAML(typ='safe').load(txt)
        else:
            data = yaml.load(txt, Loader=yaml.Loader)
        # Model name
        name = data.get('name','Model')
        # Model equations to solve
        model_eqs = data.get('Model',[])
        # Solver
        solver = data.get('Solver',None)
        # Method
        method = data.get('Method',None)
        # Sets section
        _sets = data.get('sets',{})
        indices = [x.split(" ")[-1].split("(")[0].strip() for x in _sets.keys()]
        sets = {}
        for k in _sets:
            arr = list(filter(None,k.split(" ")))
            k1 = k[:-len(arr[-1])].strip()
            indx = arr[-1].strip()
            if "(" in indx and ")" in indx:
                ind1 = indx.index("(")
                ind2 = indx.index(")")
                k2 = indx[1+ind1:ind2].strip()
                k3 = indx[:ind1].strip()
            else:
                k2 = None
                k3 = indx
            if isinstance(_sets[k],str) and _sets[k] in sets:
                sets[k3] = sets[_sets[k]]
            else:
                sets[k3] = _sets[k]
            # Check that all elements of map for key=k3 are subset of elements of this map for key=k2
            if not k2 is None:
                diff = set(sets[k3]) - set(sets[k2])
                if len(diff) > 0:
                    diff = ",".join(diff)
                    cprint(f"\nMisspecified elements of set '{k1}': extra elements - {diff}.","red")
                    sys.exit()
                
        # Symbols section
        symbols = data.get('symbols',{})
        variables = symbols.get('variables',[])
        parameters = symbols.get('parameters',[])
        
        # Equations section
        eqs = data.get('equations',[])
        equations,complementarity = fix(eqs,model_eqs)
        if not len(eqs) == len(equations):
            cprint(f"\nNumber of model equations is {len(equations)} out of original {len(eqs)}.","red")
            
        # Calibration section
        calibration = data.get('calibration',{})
        # Constraints section
        constr = data.get('constraints',{})
        # Take subset of constraints that are defined in complementarity conditions
        constraints = []; model_constraints = complementarity.values()
        for c in constr:
            if "(" in c:
                ind = c.index("(")
                k = c[:ind]
                if bool(complementarity):
                    if k in model_constraints:
                        constraints.append(c)
                else:
                    constraints.append(c)
            else:
                constraints.append(c)
                
        # Print number of equations and variables
        cprint(f"\nNumber of declared equations: {len(equations)}, variables: {len(variables)}, constraints: {len(constraints)}","blue")
        
        # Objective function section
        obj = data.get('objective_function',{})
        # Labels section
        _labels = data.get('labels',{})
        # Optional section
        options = data.get('options',{})

        # Expand expressions
        if bool(obj):
            obj     = expand(sets,indices,obj,objFunc=True)[0]
        variables   = expand(sets,indices,variables)
        parameters  = expand(sets,indices,parameters)
        equations   = expand(sets,indices,equations,loop=True)        
        
        # Check number of equations and variables
        if not len(equations) == len(variables) and not method in ["Minimize","minimize","Maximize","maximize"]:
            cprint(f"\nNumber of equations {len(equations)} and variables {len(variables)} must be the same!  \nPlease correct the model file. Exitting...","red")
            sys.exit()
        else:
            cprint(f"Number of expanded equations: {len(equations)}, parameters: {len(parameters)}","blue")
               
        calibration = expand(sets,indices,calibration)
        klist = []
        for k in calibration:
            if "(" in k and ")" in k:
                k2 = k.replace(")(","_").replace("(","_").replace(")","_")
                if k2[-1] == "_":
                    k2 = k2[:-1]
                if k2 in calibration:
                    calibration[k2] = calibration[k]
                    klist.append(k)
         
        for k in klist:
            del calibration[k]         
        
        constraints = expand(sets,indices,constraints)
        # Labels
        labels_keys = expand(sets,indices,list(_labels.keys()))
        labels      = getLabels(labels_keys,_labels)
        eqs_labels  = expand(sets,indices,eqs_labels)
        equations_labels = []
        for x in eqs_labels:
            if x in labels:
                equations_labels.append(x + "   -  " + labels[x])
            else:
                equations_labels.append(x)
                      
        # Read calibration values from excel file
        options = data.get('options',{})
        if "file" in options:
            fname = options["file"]
            del options["file"]
            file_path = os.path.abspath(os.path.join(working_dir, "../..", fname))
            if not os.path.exists(file_path):
                cprint(f"\nFile {file_path} does not exist!\n","red")
            xl = pd.ExcelFile(file_path)
            sheet_names = xl.sheet_names
            if "sheets" in options:
                sheets = [ x for x in options["sheets"] if x in sheet_names]
                del options["sheets"]
            else:
                sheets = sheet_names
            for sh in sheets:
                df = xl.parse(sh)
                symbols = df.values[:,1:-1]
                values = df.values[:,-1]
                for x,y in zip(symbols,values):
                    symb = sh+"_"+"_".join(x)
                    calibration[symb] = y
        
    delimiters = " ",",","^","*","/","+","-","(",")","<",">","=","max","min"
    regexPattern = '|'.join(map(re.escape, delimiters))
    regexFloat = r'[+-]?[0-9]+\.[0-9]+'
    # Resolve calibration references
    nprev_str = 1; n_str = i = 0; m = {}
    cal = calibration.copy()
    while i < 2 or not nprev_str == n_str:
        i += 1
        nprev_str = n_str 
        n_str = 0
        for k in calibration:
            val = cal[k]
            if isinstance(val,str):
                arr = re.split(regexPattern,val)
                arr = list(filter(None,arr))
                for x in arr:
                    if not x in m and not x.isdigit() and not re.search(regexFloat,x):
                        if x in variables and not x in cal:
                            cal[x] = 0                       
                        elif x in parameters and not x in cal:
                            cal[x] = 0
                        elif not x in parameters:
                            m[x] = 0
                try:
                    val = eval(val,m,cal)
                    cal[k] = float(np.real(val))
                except:
                    n_str += 1
                
    calibration = cal
    if len(variables) < 10:
        order = 2
    else:
        order = 1
        
    symbols = {'variables': variables,'endogenous': variables,'parameters': parameters, 'shocks': [], 'variables_labels': labels, 'equations_labels' : equations_labels}
    smodel = Interface(model_name=name,symbols=symbols,equations=equations,ss_equations=equations,calibration=calibration,constraints=constraints,objective_function=obj,definitions=[],order=order,options=options)
    smodel.SOLVER = solver
    smodel.METHOD = method
    smodel.COMPLEMENTARITY_CONDITIONS = complementarity

    infos = {'name': name,'filename': fpath}
    model = Model(smodel, infos=infos)
    model.eqLabels = eqs_labels
    
    return model
        
def getLimits(var_names,constraints,cal):
    """Find variables upper and lower limits."""
    Il, Iu = None,None
    N = 0
    lw = {}; up = {}
    
    for v in var_names:
        for c in constraints:
            if c.startswith(v):
                if '.le.' in c or '<=' in c:
                    Iu = True
                    if '.le.' in c:
                        ind = c.index('.le.')
                        s = c[4+ind:].strip()
                    else:
                        ind = c.index('<=')
                        s = c[2+ind:].strip()
                    if s in cal:
                        val = cal[s]
                    else:
                        try:
                            val = float(s)
                        except:
                            val = np.inf
                    up[v] = float(val)
                    
                elif '.lt.' in c or '<' in c:
                    Iu = True
                    if '.lt.' in c:
                        ind = c.index('.lt.')
                        s = c[4+ind:].strip()
                    else:
                        ind = c.index('<')
                        s = c[1+ind:].strip()
                    if s in cal:
                        val = cal[s]
                    else:
                        try:
                            val = float(s)
                        except:
                            val = np.inf
                    up[v] = float(val)
                    
                elif '.ge.' in c or '>=' in c:
                    Il = True
                    if '.ge.' in c:
                        ind = c.index('.ge.')
                        s = c[4+ind:].strip()
                    else:
                        ind = c.index('>=')
                        s = c[2+ind:].strip()
                    if s in cal:
                        val = cal[s]
                    else:
                        try:
                            val = float(s)
                        except:
                            val = -np.inf
                    lw[v] = float(val)
                    
                elif '.gt.' in c or '>' in c:
                    Il = True
                    if '.gt.' in c:
                        ind = c.index('.gt.')
                        s = c[4+ind:].strip()
                    else:
                        ind = c.index('>')
                        s = c[1+ind:].strip()
                    if s in cal:
                        val = cal[s]
                    else:
                        try:
                            val = float(s)
                        except:
                            val = -np.inf
                    lw[v] = float(val)
                    
                elif '.eq.' in c or '=' in c:
                    if '.eq.' in c:
                        ind = c.index('.eq.')
                        s = c[4+ind:].strip()
                    else:
                        ind = c.index('=')
                        s = c[1+ind:].strip()
                    if s in cal:
                        val = cal[s]
                        lb = ub = val
                    else:
                        try:
                            val = float(s)
                        except:
                            val = None
                        lw[v] = float(val)
                        up[v] = float(val)

    upper = []; lower = []
    for v in var_names:
        if v in lw and v in up:
            N += 1
            lower.append(lw[v])
            upper.append(up[v])
        elif v in lw:
            N += 1
            lower.append(lw[v])
            upper.append(np.inf)    
        elif v in up:
            N += 1
            lower.append(-np.inf)
            upper.append(up[v])
        else:
            lower.append(-np.inf)
            upper.append(np.inf)
            
        
    return Il,Iu,N,np.array(lower),np.array(upper)


def getConstraints(n,constraints,cal,eqLabels,jacobian):
    """Build linear constraints."""
    A = np.zeros((n,n))
    lb = np.zeros(n) - np.inf
    ub = np.zeros(n) + np.inf
    for c in constraints:
        if '.le.' in c or '<=' in c:
            if '.le.' in c:
                ind = c.index('.le.')
                label = c[:ind]
                shift = 4
            else:
                ind = c.index('<=')
                label = c[:ind]
                shift = 2
            if label in eqs_labels:
                i = eqs_labels.index(label)
                A[i] = jacobian[i]
                s = c[shift+ind:].strip()
                if s in cal:
                    val = cal[s]
                else:
                    try:
                        val = float(s)
                    except:
                        val = np.inf
                ub[i] = float(val)
        elif '.lt.' in c or '<' in c:
            if '.lt.' in c:
                ind = c.index('.lt.')
                label = c[:ind]
                shift = 4
            else:
                ind = c.index('<')
                label = c[:ind]
                shift = 1
            if label in eqs_labels:
                i = eqs_labels.index(label)
                A[i] = jacobian[i]
                s = c[shift+ind:].strip()
                if s in cal:
                    val = cal[s]
                else:
                    try:
                        val = float(s)
                    except:
                        val = np.inf
                ub[i] = float(val)-1.e-10
        elif '.ge.' in c or '>=' in c:
            if '.ge.' in c:
                ind = c.index('.ge.')
                label = c[:ind]
                shift = 4
            else:
                ind = c.index('>=')
                label = c[:ind]
                shift = 2
            if label in eqs_labels:
                i = eqs_labels.index(label)
                A[i] = jacobian[i]
                s = c[shift+ind:].strip()
                if s in cal:
                    val = cal[s]
                else:
                    try:
                        val = float(s)
                    except:
                        val = -np.inf
            lb[i] = float(val)
        elif '.gt.' in c or '>' in c:
            if '.gt.' in c:
                ind = c.index('.gt.')
                label = c[:ind]
                shift = 4
            else:
                ind = c.index('>')
                label = c[:ind]
                shift = 1
            if label in eqs_labels:
                i = eqs_labels.index(label)
                A[i] = jacobian[i]
                s = c[shift+ind:].strip()
                if s in cal:
                    val = cal[s]
                else:
                    try:
                        val = float(s)
                    except:
                        val = -np.inf
            lb[i] = float(val)+1.e-10
        elif '.eq.' in c or '=' in c:
            if '.eq.' in c:
                ind = c.index('.eq.')
                label = c[:ind]
                shift = 4
            else:
                ind = c.index('=')
                label = c[:ind]
                shift = 1
            if label in eqs_labels:
                s = c[shift+ind:].strip()
                if s in cal:
                    val = cal[s]
                else:
                    try:
                        val = float(s)
                    except:
                        val = None
                ub[i] = lb[i] = val
                
    return A,lb,ub

def getNonlinearConstraints(constraints, labels, calib):
    """Builds non-linear constrain."""
    Upper = np.empty(len(labels)); Upper[:] = np.inf
    Lower = np.empty(len(labels)); Lower[:] = -np.inf
    for i,x in enumerate(labels):
        b = False
        for cnstr in constraints:
            if '.' in cnstr or '=' in cnstr or '<' in cnstr or '>' in cnstr:
                if '.' in cnstr:
                    ind = cnstr.index('.')
                    shift = 1
                elif '<=' in cnstr:
                    ind = cnstr.index('<=')
                    shift = 0
                elif '<' in cnstr:
                    ind = cnstr.index('<')
                    shift = 0
                elif '>=' in cnstr:
                    ind = cnstr.index('>=')
                    shift = 0
                elif '>' in cnstr:
                    ind = cnstr.index('>')
                    shift = 0
                elif '=' in cnstr:
                    ind = cnstr.index('=')
                    shift = 0
                lhs = cnstr[:ind]
                if lhs == x:
                    rhs = cnstr[shift+ind:]
                    shift2 = 1
                    if '.' in rhs:
                        ind2 = rhs.index('.')
                        shift2 = 1
                    elif '>=' in rhs:
                        ind2 = rhs.index('>=')
                        shift2 = 2
                    elif '>' in rhs:
                        ind2 = rhs.index('>')
                        shift2 = 1 
                    elif '<=' in rhs:
                        ind2 = rhs.index('<=')
                        shift2 = 2  
                    elif '<' in rhs:
                        ind2 = rhs.index('<')
                        shift2 = 1
                    elif '=' in rhs:
                        ind2 = rhs.index('=')
                        shift2 = 1
                    op = rhs[:shift2+ind2].strip()
                    v = rhs[shift2+ind2:].strip()
                    #print(cnstr,op,v)
                    try:
                        v = eval(v,calib,calib)
                        b = True
                    except:
                        pass
                    break
        if b:
            if op in ['le','lt','<]','<']:
                Upper[i] = v
                Lower[i] = -np.inf
            elif op in ['ge','et','>=','>']:
                Upper[i] = np.inf
                Lower[i] = v
            elif op in ['eq','=']:
                Upper[i] = v
                Lower[i] = v
        else:
            print(cnstr)
            
            
    return Lower, Upper

def print_path_solution_status(status):
    if status == 1:
        cprint("A solution to the problem was found.","green")
    elif status == 2:
        cprint("Algorithm could not improve upon the current iterate.","red")
    elif status == 3:
        cprint("An iteration limit was reached.","red")
    elif status == 4:
        cprint("The minor iteration limit was reached.","red")
    elif status == 5:
        cprint("Time limit was exceeded.","red")
    elif status == 6:
        cprint("The user requested that the solver stop execution.","red")
    elif status == 7:
        cprint("The problem is infeasible because lower bound is greater than upper bound for some components.","red")
    elif status == 8:
        cprint("A starting point where the function is defined could not be found.","red")
    elif status == 9:
        cprint("The preprocessor determined the problem is infeasible.","red")
    elif status == 10:
        cprint("An internal error occurred in the algorithm.","red")  
   
def firstNonZero(vals):
    """
    Return first non-zero value.
    
    Args:
        vals : list
            List of values.

    Returns:
        First non nan occurence of an element in a list.
    """
    for i,v in enumerate(vals):
        if v is None or np.isnan(v):
            continue
        else:
            return i,v
        
    return -1,np.nan     

def lastNonZero(vals):
    """
    Return last nonzero value.
    
    Args:
        vals : list
            List of values.

    Returns:
        Last non nan occurence of an element in a list.
    """
    i,v = firstNonZero(vals[::-1])
    k = len(vals) - i - 1
    
    return k,v 

def getStartingValues(hist,var_names,orig_var_values,options,skip_rows=0,debug=False):
    """    
    Get starting values for current and lagged endogenous variables.
        
    Parameters:
        :param hist: Path to historical data file.
        :type hist: str.
        :param orig_var_values: Values of endogenous variables.
        :type orig_var_values: list.
        :param var_names: Names of endogenous variables.
        :type var_names: list
        :param options: Model options.
        :type options: dict.
        :param skip_rows: NUmber of rows to skip.
        :type skip_rows: int.
    """
    from dateutil import relativedelta as rd
    from snowdrop.src.utils.util import findVariableLag
    from snowdrop.src.utils.util import findVariableLead
    from snowdrop.src.utils.util import getDate
    
    calib = {}
    var_values = np.copy(orig_var_values)
    
    if isinstance(hist,str):
        name, ext = os.path.splitext(hist)
        if ext.lower() == ".xlsx" or ext.lower() == ".xls":
            df = pd.read_excel(hist,header=0,index_col=0,parse_dates=True)
        elif ext.lower() == ".csv":
            df = pd.read_csv(filepath_or_buffer=hist,sep=',',header=0,index_col=0,parse_dates=True,infer_datetime_format=True)
        df = df.apply(pd.to_numeric,errors='coerce')
        b = np.array([all(np.isnan(df.iloc[i].values)) for i in range(len(df))])
        df = df.iloc[~b]
        df = df.astype(float)
        df.index = pd.to_datetime(df.index)
    else:
        df = pd.DataFrame(hist)

    missing = []
    if "range" in options:
        start,end = options["range"]
        start = getDate(start)
        end = getDate(end)
        if "frequency" in options:
            freq = options["frequency"]
        else:
            freq = 0
        # Set starting values of endogenous variables 
        for i,var in enumerate(var_names):
            
            bSet = False
            # Set starting values of lagged variables 
            if '_minus_' in var:
                ind = var.index('_minus')
                v = var[:ind]
                if v in df.columns:
                    vname = v
                elif "OBS_" + v in df.columns:
                    vname = "OBS_" + v
                elif v + "_meas" in df.columns:
                    vname = v + "_meas"
                else:
                    vname = None
                if not vname is None:
                    lag = findVariableLag(var)
                    if freq == 0:
                        t = start + rd.relativedelta(months=12*lag)
                    elif freq == 1:
                        t = start + rd.relativedelta(months=3*lag)
                    elif freq == 2:
                        t = start + rd.relativedelta(months=1*lag)
                    elif freq == 3:
                        t = start + rd.relativedelta(weeks=1*lag)
                    elif freq == 4:
                        t = start + rd.relativedelta(days=1*lag)
                    if t in df.index:
                        values = df[vname]
                        val = values[t]
                        if not np.isnan(val):
                            calib[var] = val
                            bSet = True
                        else:
                            mask1  = df.index >= t
                            mask2  = df.index <= start
                            mask   = mask1 & mask2
                            k,val = firstNonZero(values[mask])
                            if not np.isnan(val):
                                calib[var] = val
                                bSet = True
                    else:
                        values = df[vname]
                        mask1  = df.index >= t
                        mask2  = df.index <= start
                        mask   = mask1 & mask2
                        k,val = firstNonZero(values[mask])
                        # Check time difference between the start date and the data latest available date
                        t_delta = (start - values.index[k]).days
                        b = False
                        if freq == 0:
                            b = t_delta <= 365
                        elif freq == 1:
                            b = t_delta <= 91
                        elif freq == 2:
                            b = t_delta <= 30
                        elif freq == 3:
                            b = t_delta <= 7
                        elif freq == 4:
                            b = t_delta <= 1
                        if not np.isnan(val) and b:
                            calib[var] = val
                            bSet = True
                            
            # Set starting values of lead variables 
            elif '_plus_' in var:
                ind = var.index('_plus')
                v = var[:ind]
                if v in df.columns:
                    vname = v
                elif "OBS_" + v in df.columns:
                    vname = "OBS_" + v
                elif v + "_meas" in df.columns:
                    vname = v + "_meas"
                else:
                    vname = None
                if not vname is None:
                    lead = findVariableLead(var)
                    if freq == 0:
                        t = start + rd.relativedelta(months=12*lead)
                    elif freq == 1:
                        t = start + rd.relativedelta(months=3*lead)
                    elif freq == 2:
                        t = start + rd.relativedelta(months=1*lead)
                    elif freq == 3:
                        t = start + rd.relativedelta(weeks=1*lead)
                    elif freq == 4:
                        t = start + rd.relativedelta(days=1*lead)
                    if t in df.index:
                        values = df[vname]
                        val = values[t]
                        if not np.isnan(val):
                            calib[var] = val
                            bSet = True
                        else:
                            mask1  = df.index >= start
                            mask2  = df.index <= t
                            mask   = mask1 & mask2
                            k,val = lastNonZero(values[mask])
                            if not np.isnan(val):
                                calib[var] = val
                                bSet = True
                    else:
                        values = df[vname]
                        mask   = df.index <= t
                        k,val  = lastNonZero(values[mask])
                        # Check time difference between the start date and the data latest available date
                        t_delta = (t - values.index[k]).days
                        b = False
                        if freq == 0:
                            b = t_delta <= 365
                        elif freq == 1:
                            b = t_delta <= 91
                        elif freq == 2:
                            b = t_delta <= 30
                        elif freq == 3:
                            b = t_delta <= 7
                        elif freq == 4:
                            b = t_delta <= 1
                        if not np.isnan(val) and b:
                            calib[var] = val
                            bSet = True
                            
            # Set starting values of current variables 
            else:
                if var in df.columns:
                    vname = var
                elif "OBS_" + var in df.columns:
                    vname = "OBS_" + var
                elif var + "_meas" in df.columns:
                    vname = var + "_meas"
                else:
                    vname = None
                if vname in df.columns:
                    if start in df.index:
                        val = df[vname][start]
                        if not np.isnan(val):
                            calib[var] = val
                            bSet = True
                    else:
                        values  = df[vname]
                        mask    = df.index <= start
                        k,val = lastNonZero(values[mask])
                        val     = values.iloc[k]
                        t_delta = (start - values.index[-1]).days
                        # Check time difference between the start date and the data latest available date
                        b = False
                        if freq == 0:
                            b = t_delta <= 365
                        elif freq == 1:
                            b = t_delta <= 91
                        elif freq == 2:
                            b = t_delta <= 30
                        elif freq == 3:
                            b = t_delta <= 7
                        elif freq == 4:
                            b = t_delta <= 1
                            
                        if b and not np.isnan(val):
                            calib[var] = val
                            bSet = True
                            
            # If data are missing then look at the first non-empty value
            if not bSet:
                if var in df.columns:
                    vname = var
                elif "OBS_" + var in df.columns:
                    vname = "OBS_" + var
                elif var + "_meas" in df.columns:
                    vname = var + "_meas"
                else:
                    vname = None
                if vname in df.columns:
                    values = df[vname][start:]
                    k,val = firstNonZero(values)
                    if not np.isnan(val):
                        calib[var] = val
                        bSet = True
                
            # If value is missing and variable is forward looking then, set it to current value
            if not bSet:
                if '_plus_' in var:
                    ind = var.index('_plus_')
                    var_current = var[:ind]
                    if var_current in calib:
                        val = calib[var_current]
                        if not np.isnan(val):
                            calib[var] = val
                            bSet = True
                    
            if not bSet:
                missing.append(var)
                if debug and not "_plus_" in var and not "_minus_" in var:
                    ind = var_names.index(var)
                    print(f'Variable "{var}" was not set from historical data - keeping original value {var_values[ind]}')

        # Reset missing starting values of lead and lag variables
        for i,var in enumerate(missing): #missing var_names
            if '_minus_' in var:
                ind = var.index('_minus')
                v = var[:ind]
            elif '_plus_' in var:
                ind = var.index('_plus')
                v = var[:ind]
            else:
                v = None
            if v in var_names and v in calib:
                val = calib[v]
                calib[var] = val
            
    for i,var in enumerate(var_names):
        if var in calib:
            var_values[i] = calib[var]
                         
    # x = dict(zip(var_names,var_values))
    return var_values,calib,missing

# def setShocks(model,d,start=None,reset=False):
#     """
#     Set shocks values given the time of their appearance.

#     Args:
#         model : Model
#             model object.
#         d : dict
#             Map of shock name and shock values.
#         start : datetime.
#             Start date of simulations.  Default is None.

#     Returns:
#         None.

#     """
#     shock_names = model.symbols["shocks"]
#     if reset:
#         shock_values = np.zeros(len(shock_names))
#     else:    
#         shock_values = model.calibration["shocks"]
    
#     calib,startingDate,interval = setValues(model=model,d=d,names=shock_names,values=shock_values,start=start,isShock=True)
        
#     model.calibration["shocks"] = calib.T
#     model.options["shock_values"] = calib.T
    
#     n_shk,n_t = calib.shape
    
#     return
  
# def setCalibration(model,param_name,param_value):
#     """
#     Set calibration dictionary values given the time of their appearance.

#     Args:
#         model : Model
#             model object.
#         param_name : str
#             Parameter name.
#         param_value : numeric.
#             Parameter value.

#     Returns:
#         None.

#     """
#     param_names = model.symbols["parameters"]
#     param_values = model.calibration["parameters"].copy()
#     ind = param_names.index(param_name)
#     value = param_values[ind]
#     if np.isscalar(value):
#         param_values[ind] = param_value
#     else:
#         param_values[ind] = np.zeros(len(value)) + param_value
        
#     model.calibration["parameters"] = param_values
    
# def setParameters(model,d,start=None):
#     """
#     Set parameters values given the time of their appearance.

#     Args:
#         model : Model
#             model object.
#         d : dict
#             Map of parameters name and parameters values.
#         start : datetime.
#             Start date of simulations.  Default is None.

#     Returns:
#         None.

#     """
#     param_names = model.symbols["parameters"]
#     param_values = model.calibration["parameters"]
    
#     calib,_,_ = setValues(model=model,d=d,names=param_names,values=param_values,start=start,isShock=False)
    
#     model.calibration["parameters"] = calib
      
def setValues(model,d,names,values,start=None,isShock=True):
    """
    Set shocks values given the time of their appearance.

    Args:
        model : Model
            model object.
        d : dict
            Map of variables name and variable values.
        start : datetime.
            Start date of simulations.  Default is None.
        isShock: bool
            True if shocks and False if parameters.

    Returns:
        New calibration dictionary.

    """
    from dateutil import relativedelta as rd
    from snowdrop.src.misc.termcolor import cprint
    from snowdrop.src.utils.util import getDate
    
    options = model.options

    if start is None and "range" in options:
        start,end = options["range"]
        start = getDate(start)
        # if len(start)==3:
        #     start = dt.datetime(start[0],start[1],start[2])
        # elif len(start)==2:
        #     start = dt.datetime(start[0],start[1],1)
        # elif len(start)==1:
        #     start = dt.datetime(start[0],1,1)
        
    if "frequency" in options:
        freq = options["frequency"]
        if freq == 0:
            interval = rd.relativedelta(months=12)
        elif freq == 1:
            interval = rd.relativedelta(months=3)
        elif freq == 2:
            interval = rd.relativedelta(months=1)
        elif freq == 3:
            interval = rd.relativedelta(weeks=1)
        elif freq == 4:
            interval = rd.relativedelta(days=1)
    else:
        interval = rd.relativedelta(months=12)
        
    max_size = 0
    # Get maximum length of values.
    for k in d:
        if k in names:
            i = names.index(k)
            if isinstance(d[k],(int,float)):
                e = d[k]
                if isinstance(e,tuple):
                    jj,x = e
                    max_size = max(max_size,jj+1)
                if np.isscalar(values[i]):
                    max_size = 1
                else:
                    max_size = max(max_size,len(values[i]))
            elif isinstance(d[k],list) or isinstance(d[k],np.ndarray):
                for e in d[k]:
                    if isinstance(e,tuple):
                        jj,x = e
                        max_size = max(max_size,jj+1)
                if np.isscalar(values[i]):
                    max_size = max(max_size,len(d[k]))
                else:
                    max_size = max(max_size,len(d[k]),len(values[i]))
            elif isinstance(d[k],pd.Series):
                t = start
                index = d[k].index
                j = 0
                while t <= index[-1]:
                    t += interval
                    j += 1
                max_size = max(max_size,j)
                if np.isscalar(values[i]):
                    max_size = max(max_size,len(d[k]))
                else:
                    max_size = max(max_size,len(d[k]),len(values[i]))
            else:
                if np.isscalar(values[i]):
                    max_size = max(max_size,1)
                else:
                    max_size = max(max_size,len(values[i]))
          
    # Reset the size to the number of simulations periods
    max_size = max(max_size,model.T+2)
    calib = list()
    ndim = np.ndim(values)
    for i,key in enumerate(names):
        if ndim == 1:
            calib_value = values[i]
        else:
            # Two dimensional matrices of shocks and parameters have different structure:
            # Shock's first index is time and the second index is shock number.
            # Parameter's first index is parameter number and the second index is time.
            calib_value = values[:,i] if isShock else values[i]
        if isinstance(calib_value,np.ndarray):
            calib_value = list(calib_value)
        elif np.isscalar(calib_value):
            calib_value = [calib_value]
                    
        # Fill in the last values to make size of array equal to  max_size.
        # For shocks fill in with zeros values, for parameters fill in with the last element.
        if isShock:
            calib_value += [0]*(max_size-len(calib_value))
        else:
            calib_value += [calib_value[-1]]*(max_size-len(calib_value))
            
        if key in d:
            v = d[key]
            if isinstance(v,pd.Series):
                index = v.index
                dates = []
                for i in range(2+max_size):
                    dates.append(start+i*interval)
                for ind in index:
                    val = v[ind]
                    if ind in dates:
                        j = dates.index(ind)
                    else:
                        for j in range(max_size):
                            if ind >= dates[j] and ind < dates[j+1]:
                                break
                    # Shocks enter equations at time: t+1,t+2,...   So, subtract one period.        
                    if isShock:
                        j = max(0,j-1) #? j :  Y[t+1] = ... + R*shock[t]
                    if j < len(calib_value):
                        calib_value[j] = val
                    else:
                        calib_value += [calib_value[-1]]*(j-1-len(calib_value)) + [val]
                
            elif np.isscalar(v):
                calib_value = list(v + np.zeros(max_size))
            else:
                # Fill in the last values to make size of array equal to  max_size.
                # For shocks fill in with zeros values, for parameters fill in with the last element of vector.
                ve = v[-1]
                if isinstance(ve,tuple):
                    _,ve = ve
                if isShock:
                    calib_value += [0]*(max_size-len(calib_value))
                else:
                    calib_value += [ve]*(max_size-len(calib_value))
                for k,e in enumerate(v):
                    if isinstance(e,tuple):
                        ii,x = e
                        if ii  < len(calib_value):
                            calib_value[ii] = x
                        else:
                            cprint("Index {0} exceeds array size {1}".format(ii,len(calib_value)),"yellow")
                            break
                    elif np.isscalar(e):
                        if k  < len(calib_value):
                            calib_value[k] = e
                        else:
                            cprint("Index {0} exceeds array size {1}".format(k,len(calib_value)),"yellow") 
                            break
        else:        
            if isinstance(calib_value,list):
                # Fill in the last values to make size of array equal to  max_size
                calib_value += [calib_value[-1]]*(max_size-len(calib_value))
            else:
                cprint("Failed to set calibrated value {}".format(calib_value))
                
        calib.append(calib_value)
                    
    calib = np.array(calib)
    
    return calib,start,interval
         
def getEquationsLables(model):
    """
    Build partite graph relating equations numbers to endogenous variables.
    
    In other words, match equation numbers to endogenous variables.
    
    Parameters:
        :param model: The Model object.
        :type model: Instance of class Model.
    """
    import re
    import networkx as nx
    from networkx.algorithms import bipartite
    
    m = dict(); exclude = list()
    delimiters = "+","-","*","/","**","^", "(",")","="," "
    regexPattern = '|'.join(map(re.escape, delimiters))
      
    eqs = model.symbolic.equations
    variable_names = model.symbols["variables"]
    eqsLabels = model.eqLabels
    n = len(eqs)
        
    # Try matching subset of equations first
    for i,eq in enumerate(eqs):
        label = eqsLabels[i]
        if label.isdigit():
            arr = re.split(regexPattern,eq)
            arr = list(filter(None,arr))
            ls = list(set([x for x in arr if x in variable_names and not x==label]))
            exclude.append(label)  
            if len(ls) > 0:
                m[label] = ls        
                      
    if len(exclude) > 0:        
        top_nodes = list(m.keys())  
        bottom_nodes = list(set(variable_names)-set(eqsLabels))
        G = nx.Graph()           
        G.add_nodes_from(top_nodes, bipartite=0)
        G.add_nodes_from(bottom_nodes, bipartite=1) 
        for k in m:
            for k2 in m[k]:
                if k2 in bottom_nodes:
                    G.add_edge(k,k2)
                        
        if len(m) > 0:  
            #Obtain the minimum weight full matching (aka equations to variables perfect matching)
            matching = bipartite.matching.minimum_weight_full_matching(G,top_nodes,"weight")
            for k in top_nodes:
                if k in eqsLabels:
                    ind = eqsLabels.index(k)
                    eqsLabels[ind] = matching[k]
                    
    if n > len(set(eqsLabels)):
        # If it fails then use a full set of equations
        top_nodes = list()
        for i,eq in enumerate(eqs):
            arr = re.split(regexPattern,eq)
            arr = list(filter(None,arr))
            ls = list(set([x for x in arr if x in variable_names]))
            m[i] = ls
            top_nodes.append(i)

        G = nx.Graph()           
        G.add_nodes_from(top_nodes, bipartite=0)
        G.add_nodes_from(variable_names, bipartite=1) 
        for k in m:
            nodes = m[k]
            for k2 in nodes:
                G.add_edge(k,k2)
                
        if len(m) > 0:  
            #Get perfect matching for a full set of equations
            matching = bipartite.matching.minimum_weight_full_matching(G,top_nodes,"weight")
            for i in top_nodes:
                eqsLabels[i] = matching[i]  
                
    model.eqLabels = eqsLabels
               
if __name__ == '__main__':
    """
    The test program.
    
    eq1: f1(x1,x3)
    eq2: f2(x2)
    eq3: f3(x1,x2)
    eq4: f(x1,x3,x6)
    """
    import networkx as nx
    from networkx.algorithms import bipartite
    
    top_nodes = ["eq1","eq2","eq3","eq4"]
    bottom_nodes = ["x1","x2","x3","x4","x5","x6"]
    nodes = top_nodes + bottom_nodes
    m = {"eq1":["x1","x3"], "eq2":["x2"], "eq3":["x1","x2"], "eq4":["x1","x3","x6"]}
                                                
    G = nx.Graph()   
    G.add_nodes_from(top_nodes, bipartite=0)
    G.add_nodes_from(bottom_nodes, bipartite=1) 
    for k in m:
        for k2 in m[k]:
            G.add_edge(k,k2)

    #Obtain variables to equations matching
    matching = bipartite.matching.minimum_weight_full_matching(G, top_nodes, "weight")    
    for k in top_nodes:
        print(f"{matching[k]} -> {k}")
