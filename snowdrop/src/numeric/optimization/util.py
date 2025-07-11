# -*- coding: utf-8 -*-
"""
Generic optimization module.
Developed to replicate GAMS PATH Solver.
@author: A.Goumilevski
"""
import os, sys
import numpy as np
import pandas as pd
import ruamel.yaml as yaml
from dataclasses import dataclass
from snowdrop.src.misc.termcolor import cprint

fpath = os.path.dirname(os.path.abspath(__file__))
working_dir = os.path.abspath(os.path.join(os.path.abspath(fpath + "../..")))

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


def loadLibrary():
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
        path_libc = ctypes.util.find_library("libpath")
        if path_libc is None:
            path_libc = os.path.join(lib_dir, 'libpath.dll')
    else:
        path_libc = ctypes.util.find_library("libpath")
        if path_libc is None:
            path_libc = os.path.join(lib_dir, 'libpath.so')
           
    path_dep = path_libc.replace('libpath','libpath50')
    
    # Get a handle to the sytem C library
    try:
        ctypes.CDLL(name=path_dep, mode=ctypes.RTLD_GLOBAL)
        libc = ctypes.CDLL(path_libc)
    except OSError as ex:
        cprint(f"\n{ex}\nUnable to load the system C++ library!  Exitting...","red")
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

def expand_obj_func(categories,sub,expr,symb="+"):
    """
    Iterates thru a list of indices and categories, 
    substitutes an index in a variable name with a corresponding category, 
    and builds a list of new variables.

    Parameters:
        :param categories: Categories.
        :type categories: list.
        :param sub: Sub-string to replace.
        :type sub: str.
        :param expr: Text.
        :type expr: str.
        :returns: Text representation of sum or product operation.
    """
    arr = []
    # Loop over items in a set
    for c in categories:
        arr.append(replace_all(sub,"_"+c,expr))
     
    out = symb.join(arr)
    return out

def expand_sum(sets,indices,txt): 
    return expand_sum_or_prod(sets,indices,txt,symb="+")

def expand_prod(sets,indices,txt): 
    return expand_sum_or_prod(sets,indices,txt,symb="*")
    
def expand_sum_or_prod(sets,indices,txt,symb): 
    """ 
    Iterates thru a list of indices and categories,  
    substitutes an index in a variable name with a corresponding category,  
    and builds a list of new variables. 
 
    Parameters: 
        :param sets: Dictionary of categories. 
        :type sets: dict. 
        :param indices: List of indeces. 
        :type indices: list. 
        :param txt: Text. 
        :type txt: str. 
        :param symb: Symbol "+" or "*". 
        :type txt: str. 
        :returns: Text representation of summation or product operation. 
    """ 
    ind = txt.index(",") 
    args = txt[:ind].split(";") 
    expr = txt[1+ind:].strip() 
    arr = [] 
    # Loop over indices 
    for i,index in enumerate(indices): 
        if index in args: 
            cat = sets[index] 
            for c in cat: 
                arr.append(replace_all("("+index+")","_"+c,expr)) 
      
    out = symb.join(arr) 
     
    return "(" + out + ")" 

def expand_minmax(b,sets,indices,txt):    
    """
    Iterates thru a list of indices and categories, 
    substitutes an index in a variable name with a corresponding category, 
    and builds a list of new variables.

    Parameters:
        :param b: True if minimum and False if maximum.
        :type b: bool.
        :param sets: Dictionary of categories. 
        :type sets: dict. 
        :param indices: List of indeces. 
        :type indices: list. 
        :param txt: Text. 
        :type txt: str. 
        :returns: Text representation of min/max operation.
    """    
    ind = txt.index(",") 
    args = txt[:ind].split(";") 
    expr = txt[1+ind:].strip() 
    arr = [] 
    # Loop over indices 
    for i,index in enumerate(indices): 
        if index in args: 
            cat = sets[index] 
            for c in cat: 
                arr.append(replace_all("("+index+")","_"+c,expr)) 
                
    out = ", ".join(arr) 
    
    if b:
        return "min(" + out + ")"
    else:
        return "max(" + out + ")"        
    
def expand_loop(categories,sub,expr):    
    """
    Iterates thru a list of indices and categories, 
    substitutes an index in a variable name with a corresponding category, 
    and builds a list of new variables.

    Parameters:
        :param categories: Categories.
        :type categories: list.
        :param sub: Sub-string to replace.
        :type sub: str.
        :param expr: Text.
        :type expr: str.
        :returns: Text representation of sum operation.
    """
    
def expand_list(sets,indices,arr,objFunc=False,loop=False):
    """
    Iterates thru a list of indices and categories, 
    substitutes an index of a variable name with a corresponding category, 
    and builds a list of new variables.

    Parameters:
        :param sets: Dictionary of categories.
        :type sets: dict.
        :param indices: List of indeces.
        :type indices: list.
        :param arr: List of indeces.
        :type arr: list.
        :param: objFunc: True if expanding expression for objective function.
        :type objFunc: bool.
        :param loop: True if expanding expression for objective function.
        :type loop: bool.
        :returns: list object.
    """
    if len(indices) == 0:
        return arr
    
    out = [] 
    if objFunc:
        # Loop over items in array
        for ieq,eq in enumerate(arr):
            e = eq.replace(" ","")
            
            if "sum(" in e or "prod(" in e:
                # Loop over indices
                ind1 = e.index("sum(") if "sum(" in e else  e.index("prod(") 
                ind2 = getIndex(e,ind1)
                op = e[ind1+4:ind2-1]
                ind = op.index(",")
                txt = op[1+ind:]
                for i,index in enumerate(indices):
                    sub = "("+index+")"
                    if sub in e:
                        txt = expand_obj_func(sets[index],sub,txt,"+" if "sum(" in e else "*")
                        for c in sets[index]:
                            n = replace_all(sub,"_"+c,txt)
                arr[ieq] = e[:ind1] + "(" + txt + ")" + e[2+ind2:]
                        
           
        for i,index in enumerate(indices): 
            sub = "("+index+")" 
            for e in arr: 
                e = e.replace(" ","") 
                if sub in e:   
                    for c in sets[index]: 
                        n = replace_all(sub,"_"+c,e) 
                        if not n in out: 
                            out.append(n) 
                else: 
                    if not e in out: 
                        out.append(e) 
            arr = out 
                  
    else:
        
        # Expand loop statements
        if loop:
            lst = []
            for e in arr: 
                e = e.replace(" ","") 
                if "loop(" in e:
                    ind1 = e.index("loop(") 
                    ind2 = getIndex(e,ind1)
                    op = e[ind1+5:ind2-1] 
                    stmts = expand_loop(sets,indices,op) 
                    lst.extend(stmts)
                else:
                    lst.append(e)
            arr = lst
            
        # Loop over indices 
        for i,index in enumerate(indices): 
            sub = "("+index+")" 
            for j,e in enumerate(arr): 
                e = e.replace(" ","") 
                if sub in e:                     
                    if "MIN(" in e: 
                        ind1 = e.index("MIN(") 
                        ind2 = getIndex(e,ind1)
                        op = e[ind1+4:ind2-1] 
                        txt = expand_minmax(True,sets,indices,op) 
                        e = e[:ind1] + txt + e[ind2:]
                    elif "MAX(" in e: 
                        ind1 = e.index("MAX(") 
                        ind2 = getIndex(e,ind1)
                        op = e[ind1+4:ind2-1] 
                        txt = expand_minmax(False,sets,indices,op) 
                        e = e[:ind1] + txt + e[ind2:]
                    while "sum(" in e: 
                        ind1 = e.index("sum(") 
                        ind2 = getIndex(e,ind1)
                        op = e[ind1+4:ind2-1] 
                        txt = expand_sum(sets,indices,op) 
                        e = e[:ind1] + txt + e[ind2:] 
                    while "prod(" in e: 
                        ind1 = e.index("prod(") 
                        ind2 = getIndex(e,ind1) 
                        op = e[ind1+5:ind2-1] 
                        txt = expand_prod(sets,indices,op) 
                        e = e[:ind1] + txt + e[ind2:] 
                    for c in sets[index]: 
                        n = replace_all(sub,"_"+c,e) 
                        if not n in out: 
                            out.append(n) 
                else: 
                    if not e in out: 
                        out.append(e) 
            arr = out 
        
    # Clean left over indices that might be left.
    arr = []
    for i,x in enumerate(out):
        b = True
        for index in indices:
            sub = "("+index+")"
            if sub in x:
                b = False
                break
        if b:
            arr.append(x)
            
    # Treat any hard-coded indices
    for i,x in enumerate(arr):
        x = x.replace(')(','_')
        x = x.replace('(','_')
        x = x.replace(')','_')
        if x[-1] == "_":
            x = x[:-1]
        if x[0] == '_':
            x = x[1:]
    
    return arr

   
def expand(sets,indices,expr,objFunc=False,loop=False):
    """
    Expands expression.

    Parameters:
        :param sets: Dictionary of categories.
        :type sets: dict.
        :param indices: List of indeces.
        :type indices: list.
        :param expr: Object.
        :type expr: list or dict.
        :returns: objFunc: True if expanding expression for objective function.
        :type objFunc: bool.
        :returns: Expanded expression.
    """
    if isinstance(expr,list):
        return expand_list(sets,indices,expr,objFunc=objFunc,loop=loop)
    elif isinstance(expr,dict):
        return expand_map(sets,indices,expr)

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
    
    
def expand_map(sets,indices,m):
    """
    Iterates thru a list of indices and categories, 
    substitutes an index in a variable name with a corresponding category, 
    and builds a list of new variables.

    Parameters:
        :param sets: Dictionary of categories.
        :type sets: dict.
        :param indices: List of indeces.
        :type indices: list.
        :param m: Map.
        :type m: dict.
        :returns: Dictionary object.
    """
    if len(indices) == 0:
        return m
    
    out = {}
    # Loop over indices
    for i,index in enumerate(indices):
        for k in m:
            values = m[k]
            sub = "("+index+")"
            if sub in k or sub in str(values):
                for j,c in enumerate(sets[index]):
                    key = replace_all(sub,"_"+c,k)
                    if isinstance(values,list) and j < len(values):
                        out[key] = values[j]
                    elif isinstance(values,str):
                        values = values.replace(" ","")
                        if "MIN(" in values: 
                            ind1 = values.index("MIN(") 
                            ind2 = getIndex(values,ind1)
                            op = values[ind1+4:ind2-1] 
                            txt = expand_minmax(True,sets,indices,op) 
                            values = values[:ind1] + txt + values[ind2:]
                        elif "MAX(" in values: 
                            ind1 = values.index("MAX(") 
                            ind2 = getIndex(values,ind1)
                            op = values[ind1+4:ind2-1] 
                            txt = expand_minmax(False,sets,indices,op) 
                            values = values[:ind1] + txt + values[ind2:]
                        while "sum(" in values: 
                            ind1 = values.index("sum(") 
                            ind2 = getIndex(values,ind1) 
                            op = values[ind1+4:ind2-1] 
                            txt = expand_sum(sets,indices,op) 
                            values = values[:ind1] + txt + values[ind2:] 
                        while "prod(" in values: 
                            ind1 = values.index("prod(") 
                            ind2 = getIndex(values,ind1) 
                            op = values[ind1+5:ind2-1] 
                            txt = expand_prod(sets,indices,op) 
                            values = values[:ind1] + txt + values[ind2:] 
                        out[key] = replace_all(sub,"_"+c,values)
                    else:
                        if not key in out: 
                            out[key] = values 
                        
            else:
                if not k in out: 
                    out[k] = m[k]
        m = out.copy()
    
    # Clean left over
    out = {}
    for k in m:
        b = True
        for index in indices:
            sub = "("+index+")"
            if sub in k:
                b = False
                break
        if b:
            out[k] = m[k]
            
    return out
 
 
    
def fix(eqs,model_eqs):
    """
    Get equations, label of equations and complemtarity conditions.

    Parameters:
        :param eqs: Equations.
        :type eqs: list.
        :param model_eqs: Model equations to solve.
        :type model_eqs: list.
        :returns: List of equations and complemtarity conditions.
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
        

def getLimits(var_names,constraints,cal):
    """Find variables upper and lower limits."""
    Il, Iu = None,None
    lower = []; upper = []
    for v in var_names:
        arr = []
        for c in constraints:
            lb = ub = None
            if v in c:
                if '.lt.' in c:
                    Iu = True
                    ind = c.index('.lt.')
                    s = c[4+ind:].strip()
                    if s in cal:
                        val = cal[s]
                    else:
                        try:
                            val = float(s)
                        except:
                            val = np.inf
                    ub = float(val)-1.e-10
                elif '.le.' in c:
                    Iu = True
                    ind = c.index('.le.')
                    s = c[4+ind:].strip()
                    if s in cal:
                        val = cal[s]
                    else:
                        try:
                            val = float(s)
                        except:
                            val = np.inf
                    ub = float(val)
                elif '.gt.' in c:
                    Il = True
                    ind = c.index('.gt.')
                    s = c[4+ind:].strip()
                    if s in cal:
                        val = cal[s]
                    else:
                        try:
                            val = float(s)
                        except:
                            val = -np.inf
                    lb = float(val)+1.e-10
                elif '.ge.' in c:
                    Il = True
                    ind = c.index('.ge.')
                    s = c[4+ind:].strip()
                    if s in cal:
                        val = cal[s]
                    else:
                        try:
                            val = float(s)
                        except:
                            val = -np.inf
                    lb = float(val)
                elif '.eq.' in c:
                    ind = c.index('.eq.')
                    s = c[1+ind:].strip()
                    if s in cal:
                        val = cal[s]
                        lb = ub = val
                    else:
                        try:
                            val = float(s)
                        except:
                            val = None
            arr.append([lb, ub])
            
        lb = ub = None
        for x in arr:
            if not x[0] is None and lb is None:
                lb = x[0]
            if not x[1] is None and ub is None:
                ub = x[1]
        if lb is None:
            lb = -np.inf
        if ub is None:
            ub = np.inf
            
        lower.append(lb)
        upper.append(ub)
        
    return Il,Iu,np.array(lower),np.array(upper)


def getConstraints(n,constraints,cal,eqLabels,jacobian):
    """Build linear constraints."""
    A = np.zeros((n,n))
    lb = np.zeros(n) - np.inf
    ub = np.zeros(n) + np.inf
    for c in constraints:
        if '.lt.' in c:
            ind = c.index('.lt.')
            label = c[:ind]
            if label in eqs_labels:
                i = eqs_labels.index(label)
                A[i] = jacobian[i]
                s = c[4+ind:].strip()
                if s in cal:
                    val = cal[s]
                else:
                    try:
                        val = float(s)
                    except:
                        val = np.inf
                ub[i] = float(val)-1.e-10
        elif '.le.' in c:
            ind = c.index('.le.')
            label = c[:ind]
            if label in eqs_labels:
                i = eqs_labels.index(label)
                A[i] = jacobian[i]
                s = c[4+ind:].strip()
                if s in cal:
                    val = cal[s]
                else:
                    try:
                        val = float(s)
                    except:
                        val = np.inf
                ub[i] = float(val)
        elif '.gt.' in c:
            ind = c.index('.gt.')
            label = c[:ind]
            if label in eqs_labels:
                i = eqs_labels.index(label)
                A[i] = jacobian[i]
                s = c[4+ind:].strip()
                if s in cal:
                    val = cal[s]
                else:
                    try:
                        val = float(s)
                    except:
                        val = -np.inf
            lb[i] = float(val)+1.e-10
        elif '.ge.' in c:
            ind = c.index('.ge.')
            label = c[:ind]
            if label in eqs_labels:
                i = eqs_labels.index(label)
                A[i] = jacobian[i]
                s = c[4+ind:].strip()
                if s in cal:
                    val = cal[s]
                else:
                    try:
                        val = float(s)
                    except:
                        val = -np.inf
            lb[i] = float(val)
        elif '.eq.' in c:
            ind = c.index('.eq.')
            label = c[:ind]
            if label in eqs_labels:
                s = c[4+ind:].strip()
                if s in cal:
                    val = cal[s]
                else:
                    try:
                        val = float(s)
                    except:
                        val = None
                ub[i] = lb[i] = val
                
    return A,lb,ub


def plot(var,var_names,par=None,par_names=None,title="",symbols=None,xLabel=None,yLabel=None,plot_variables=False,relative=False,sizes=None,fig_sizes=(8,6)):
    """Plot bar graphs."""
    
    from graphs.util import barPlot
    path_to_dir = os.path.abspath(os.path.join(fpath,'../../../graphs'))
    
    if plot_variables:
        title = []
        labels = []; data = []
        for v in var_names:
            arr = []; lbls = []
            for k in var:
                if k.startswith(v+"_"):
                    s = k[1+len(v):]
                    lbls.append(s.upper().replace("_","\n"))
                    arr.append(var[k])
            if len(lbls)>0:
                labels.append(lbls)
                t = symbols[v] if v in symbols else v
                title.append(t)
            if len(arr)>0:
                data.append(arr)
        data = np.array(data)
        
        if len(labels) > 0:
            barPlot(path_to_dir,title,data,labels,xLabel,yLabel,sizes=sizes,plot_variables=plot_variables,fig_sizes=fig_sizes,save=True,show=True,ext='png')
     
    else:
        vom_parameters = [x for x in par_names if x.startswith("vom_")]
        Y   = [x for x in var_names if x.startswith("Y_")]
        if len(vom_parameters) == len(Y):
            welfare = {}
            for p in vom_parameters:
                s = p[4:]
                if "Y_"+s in Y:
                    welfare[s.upper().replace("_","\n")] = par[p] * var["Y_"+s]
                    
            if bool(welfare):
                labels = list(welfare.keys())
                if relative:
                    data = np.array([100*(welfare[k][1]/welfare[k][0]-1) for k in labels])
                else:
                    data = np.array([welfare[k] for k in labels])
                barPlot(path_to_dir,title,data,labels,xLabel,yLabel,sizes=sizes,fig_sizes=fig_sizes,save=True,show=True,ext='png')
     
    
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