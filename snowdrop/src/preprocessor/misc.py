"""Miscellaneous code."""
from collections import OrderedDict
import numpy


def calibration_to_vector(symbols, calibration_dict):
    """
    Build list of dictionary values.

    Parameters:
        symbols : list
            Symbols.
        calibration_dict : dict
            Mapping of names and values.

    Returns:
        calibration : list
            Values list.

    """
    from numpy import nan
    from snowdrop.src.preprocessor.eval_solver import evaluate
    
    sol = evaluate(calibration_dict)
    calibration  = OrderedDict()
    for group in symbols:
        symb = symbols[group]
        max_size = 0
        for s in symb:
            obj = sol.get(s, nan)
            if isinstance(obj,list):
                max_size = max(max_size,len(obj))  
                
        if max_size == 0: # List does not contain sub-lists
            values = [sol.get(s, nan) for s in symb]
            t = numpy.array(values, dtype=float)
        else:
            values = []        
            for s in symb:
                obj = sol.get(s, nan)
                if isinstance(obj,list):
                    size = len(obj)
                    e = obj + [obj[-1]]*(max_size-size)
                    values.append(e)
                else:
                    e = [obj]*max_size
                    values.append(e)
                    
            t = numpy.array(values, dtype=float)
            
        calibration[group] = t

    return calibration


def calibration_to_dict(symbols, calib):
    """
    Build `OrderedDict` from `dict`.

    Parameters:
        symbols : list
            Symbols.
        calibration_dict : dict
            Mapping of names and values.

    Returns:
        calibration : OrderedDict
            Ordered dictionary.

    """
    from collections import OrderedDict
    if not isinstance(symbols, dict):
        symbols = symbols.symbols

    d = OrderedDict()
    for group, values in calib.items():
        if group == 'covariances':
            continue
        syms = symbols[group]
        for i, s in enumerate(syms):
            d[s] = values[i]

    return d


import copy


class CalibrationDict(OrderedDict):
    """
    Dictionary that holds model calibration names and values.
    
    Parameters:
        OrderedDict:
            Ordered dictionary

    Usage examples:
        cb = CalibrationDict(symbols, calib)
        
    """

    def __init__(self, symbols=None, calib=None):
        superclass = super()
        if symbols is None or calib is None:
            return
        calib = copy.deepcopy(calib)
        for v in calib.values():
            v.setflags(write=False)
        superclass.__init__(calib)
        self.symbols = symbols
        self.flat = calibration_to_dict(symbols, calib)
        self.grouped = calib


    def __getitem__(self, p):
        """
        """
        if isinstance(p,tuple):
            return [self[e] for e in p]
        if p in self.symbols.keys():
            return super().__getitem__(p)
        else:
            return self.flat[p]


def allocating_function(inplace_function, size_output):
    """
    """
    def new_function(*args, **kwargs):
        val = numpy.zeros(size_output)
        nargs = args + (val,)
        inplace_function( *nargs )
        if 'diff' in kwargs:
            return numdiff(new_function, args)
        return val

    return new_function


def numdiff(fun, args):
    """Vectorized numerical differentiation."""
    epsilon = 1e-8
    args = list(args)
    v0 = fun(*args)
    N = v0.shape[0]
    l_v = len(v0)
    dvs = []
    for i,a in enumerate(args):
        l_a = (a).shape[1]
        dv = numpy.zeros( (N, l_v, l_a) )
        nargs = list(args) #.copy()
        for j in range(l_a):
            xx = args[i].copy()
            xx[:,j] += epsilon
            nargs[i] = xx
            dv[:,:,j] = (fun(*nargs) - v0)/epsilon
        dvs.append(dv)
    return [v0] + dvs

