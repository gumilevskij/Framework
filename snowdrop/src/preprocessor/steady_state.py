from snowdrop.src.numeric.solver.util import getParameters

def residuals(model):
    """
    Returns residuals of model equations for a static solution.
    """
 
    y = model.calibration['variables']
    p = getParameters(model=model)
    e = model.calibration['shocks'] 
    
    f_static = model.functions['f_static']
    bHasAttr  = hasattr(f_static,"py_func")
    if bHasAttr:
        res = f_static.py_func(y,p,e)
    else:
        res = f_static(y,p,e)
    
     
    return {'equations': res}