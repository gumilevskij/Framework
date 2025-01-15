from snowdrop.src.preprocessor.symbolic import stringify, parse_string, list_variables, stringify_symbol
from snowdrop.src.preprocessor.codegen import to_source
from snowdrop.src.preprocessor.misc import CalibrationDict
from snowdrop.src.preprocessor.util import IfThen, IfThenElse, Positive, Negative
from numpy import log, exp


def eval_formula(expr: str, dataframe=None, context=None):
    """
    Evaluate expression.
        
    Args:
        expr: string
            Symbolic expression to evaluate.
            Example: `k(1)-delta*k(0)-i`
        table: (optional) pandas dataframe
            Each column is a time series, which can be indexeds.
        context: dict or CalibrationDict
            Context
        
    """
    print("Evaluating: {}".format(expr))
    if context is None:
        dd = {}  # context dictionary
    elif isinstance(context, CalibrationDict):
        dd = context.flat.copy()
    else:
        dd = context.copy()

    # compat since normalize form for parameters doesn't match calib dict.
    for k in [*dd.keys()]:
        dd[stringify_symbol(k)] = dd[k]

    expr_ast = parse_string(expr).value
    variables = list_variables(expr_ast)
    nexpr = stringify(expr_ast)
    print(expr)
    print(variables)

    dd['log'] = log
    dd['exp'] = exp
    dd['IfThen'] = IfThen 
    dd['IfThenElse'] = IfThenElse 
    dd['Positive'] = Positive 
    dd['Negative'] = Negative 

    if dataframe is not None:
        import pandas as pd
        for (k, t) in variables:
            dd[stringify_symbol((k, t))] = dataframe[k].shift(t)
        dd['t'] = pd.Series(dataframe.index, index=dataframe.index)

    expr = to_source(nexpr)
    print(expr)
    print(dd.keys())
    res = eval(expr, dd)

    return res
