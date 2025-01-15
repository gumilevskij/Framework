from __future__ import division
    
try:
    import symengine as sy
    from symengine import Abs
    SYMENGINE = True
except:
    import sympy as sy  
    from sympy.functions.elementary.complexes import Abs
    SYMENGINE = False
    
import ast, os
import numpy as np
#from scipy.special import lambertw as LambertW
#from snowdrop.src.utils.equations import check_presence
from snowdrop.src.misc.termcolor import cprint

from snowdrop.src.preprocessor.util import IfThen, IfThenElse, Positive, Negative, myzif
from sympy import  Min, Max, LambertW

skip = False

# Add symbols that are not functions to local name space
not_to_be_treated_as_functions = ['alpha','beta','gamma','zeta','Chi','div']
ns = {v: sy.Symbol(v) for v in not_to_be_treated_as_functions}

# Add user defined functions to local name space
ns["IfThen"] = IfThen
ns["IfThenElse"] = IfThenElse
ns["Positive"] = Positive
ns["Negative"] = Negative
ns["abs"] = Abs
ns["myzif"] = myzif
ns["LambertW"] = LambertW
ns["Min"] = Min
ns["Max"] = Max


def ast_to_sympy(expr):
    """
    Convert an AST to a sympy expression.
    
    Parameters:
        :param expr: AST expression.
        :type expr: ast.
        :returns:  Sympy expression.
    """
    from snowdrop.src.preprocessor.codegen import to_source
    try:
        s = to_source(expr)
    except:
        cprint("Cannot convert AST expression to string: {}\n".format(expr),"red")
        raise
    try:
        if SYMENGINE:
            e = sy.sympify(s)
        else:
            e = sy.sympify(s,locals=ns)
        return e
    except:
        cprint("Cannot convert string to SymPy expression: {}\n".format(s),"red")
        raise
        

#from functools import lru_cache
#@lru_cache(maxsize=2048)
def non_decreasing_series(n, size):
    """
    List all combinations of 0,...,n-1 in increasing order.
    
    Parameters:
        :param n: Defines maximum number of integers in all combinations.
        :type n: int.
        :param size: Size of all combinations.
        :type size: int.
        :returns:  List of combination of 0,...,n-1 of size *size*.
    """
    if size == 1:
        return [[a] for a in range(n)]
    else:
        lc = non_decreasing_series(n, size-1)
        ll = []
        for l in lc:
            last = l[-1]
            for i in range(last, n):
                e = l + [i]
                ll.append(e)
        return ll


def higher_order_diff(eqs, symbols, eq_vars=[], order=1):
    """
    Take higher order derivatives of a list of equations w.r.t a list of symbols.
    
    Parameters:
        :param eqs: List of equations.
        :type eqs: list.
        :param symbols: List of symbols.
        :type symbols: list.
        :param eq_vars: List of variables in each equation.
        :type eq_vars: list.
        :param order: Order of partial derivatives of Jacobian.
        :type order: int.
        :returns:  Matrix of partial derivatives.
    """
#    for e in equations:
#        print(e)
#        eq = sy.sympify(e,locals=ns)
#        print(eq)
#        print('-')

    # Convert equations and symbols to SumPy types that can be used in Sumpy differentiation of mathematical expression with variables method.
    # if SYMENGINE:
    #     eqs = list([sy.sympify(eq) for eq in eqs]) 
    # else:
    #     eqs = list([sy.sympify(eq,locals=ns) for eq in eqs])
    syms = list([sy.sympify(s) for s in symbols])

    neq = len(eqs)
    p = len(syms)

    B = [np.arange(0,neq)]
    D = [np.array(eqs)]
    
    eq_nmbrs = []
    if not skip and bool(eq_vars):
        for i in range(neq):
            eq_n = []
            for j,var_name in enumerate(symbols):
                if var_name in eq_vars[i]:
                    eq_n.append(j)
            eq_nmbrs.append(eq_n)
        
    for i in range(1,order+1):

        par = D[i-1]
        mat = np.empty([neq] + [p]*i, dtype=object)
        eqs_numbers = B[i-1]
        mat_numbers = np.empty([neq] + [p]*i, dtype=object)
        nds = non_decreasing_series(p,i)
        
        for ind in nds:

            ind_parent = ind[:-1]
            k = ind[-1]
            var = syms[k]
            var_name = symbols[k]

            for line in range(neq):
                
                ii = tuple([line] + ind)
                iid = tuple([line] + ind_parent)
                eeq = par[iid]
                eq_number = eqs_numbers[iid]
                mat_numbers[ii] = eq_number
                #mat[ii] = eeq.diff(var)
                if bool(eq_vars) and not skip: 
#                    if not str(mat[ii]) == "0" and not str(var) in eq_vars[eq_number]:
#                        print(line,iid,eq_number,var)
#                        print(eeq)   
                    if k in eq_nmbrs[eq_number]:
                        if isinstance(eeq,int):
                            mat[ii] = 0
                        else:
                            mat[ii] = eeq.diff(var)
                    else:
                        mat[ii] = 0
                else:
                    mat[ii] = eeq.diff(var)

        D.append(mat)
        B.append(mat_numbers)

    return D
       
               
def compile_higher_order_function(equations,syms,params,syms_exog=[],eq_vars=[],order=1,function_name='anonymous',out='f_dynamic',b=False,bSparse=False,model_name='',log_variables=[]):
    """
    From a list of equations and variables, define a multivariate functions with higher order derivatives.
    
    Parameters:
        :param equations: List of equations.
        :type equations: list.
        :param syms: Symbols.
        :type syms: list.
        :param params: Parameters.
        :type params: list.
        :param syms_exog: Exogenous variables symbols.
        :type syms_exog: list
        :param eq_vars: List of variables in each equation.
        :type eq_vars: list.
        :param order: Order of partial derivatives of Jacobian.
        :type order: int.
        :param function_name: Function name.
        :type function_name: str.
        :param out: Name of file that contains this function source code.
        :type out: str.
        :param b: If True use cupy package, otherwise - numpy.
        :type b: bool.
        :param bSparse: True if sparse algebra is used.
        :type bSparse: bool.
        :param model_name: Model name.
        :type model_name; str.
        :returns:  Function and matrices of partial derivatives.
    """
    import re
    from snowdrop.src.preprocessor.symbolic import stringify
#    from snowdrop.src.utils.equations import fixEquation

    variables = [s[0] for s in syms]
    exogenous = [s[0] for s in syms_exog]
#    # TEMP: compatibility fix when eqs is an Odict:
#    eqs = [eq for eq in equations]
#
#    if isinstance(eqs[0], str):
#    # elif not isinstance(eqs[0], sy.Basic):
#    # assume we have ASTs
    
#       from snowdrop.src.preprocessor.symbolic import normalize
#        count = 0
#        new_eqs = []
#        for eq in eqs:
#            count += 1
#            try:
#                new_eqs.append(ast.parse(eq).body[0])
#            except:
#                print()
#                print(f"Error parsing equation #{count}: {eq}")
#        #new_eqs = list([ast.parse(eq).body[0] for eq in eqs])
#        eqs_std = list( [normalize(eq, variables=variables) for eq in new_eqs] )
#        eqs_sym = list( [ast_to_sympy(eq) for eq in eqs_std] )
#    else:
#        eqs_sym = eqs
        
    if bool(log_variables):  
        eqs_sym = log_normalize_equations(equations,variables+exogenous,log_variables,True)
    else:
        eqs_sym = normalize_equations(equations,variables+exogenous,True)

    symsd = [stringify((a,b)) for a,b in syms]
    exogd = [stringify((a,b)) for a,b in syms_exog]
    paramsd = [stringify(a) for a in params]
    D = higher_order_diff(eqs_sym, symsd, eq_vars, order=0 if order is None else order)
    eqs = " ".join([str(x) for x in D[0]])

    txt_eq = ""; txt_der = ""; txt_der2 = ""; txt_der3 = ""
    
    txt = """from numba import njit
    
@njit
"""    

    txt += "def {function_name}(x, p, exog=[0], order={order}, ind=None):".format(function_name=function_name,order=order)
    txt += """
        
    ### This code was generated by Python.
    ### {model_name}
    
    # First order derivatives are employed in most of the models to compute Jacobian.
    # Higher order derivatives are used in nonlinear rational expectations models.
    
    from scipy.special import lambertw as LambertW
    from snowdrop.src.preprocessor.functions import Heaviside,Max,Min,Abs,DiracDelta
    from snowdrop.src.preprocessor.condition import IfThenElse,IfThen,Derivative,Subs,Positive,Negative,myzif
""".format(model_name=model_name)

    if b:
        txt += """
    import cupy as np
    from cupy import exp, sin, cos, tan, sqrt, sign, log
    
"""
    else:
        txt += """
    import numpy as np
    from numpy import exp, sin, cos, tan, sqrt, sign, log
        
"""
    n_syms = len(syms)
    n_params = len(params)
    n_syms_exog = len(syms_exog)
    n_equations = len(equations)
    
    delimiters = " ", ",", ";", "*", "/", ":", "+", "-", "^", "{", "}", "(", ")","[","]"
    regexPattern = '|'.join(map(re.escape, delimiters))
    arr_eqs = re.split(regexPattern,eqs)
    arr_eqs = list(filter(None,arr_eqs))
    
    txt += "    # Initialize variables\n"
    txt += "    _xi_1 = 0 \n"
    txt += "    _xi_2 = 0 \n"
    txt += "    _xi_3 = 0 \n"
    for i in range(n_syms):
        if symsd[i] in arr_eqs:
            txt += f"    {symsd[i]} = x[{i}]\n"

    txt += "\n    # Set parameters\n"

    for i in range(n_params):
        txt += f"    {paramsd[i]} = p[{i}]\n"

    if n_syms_exog > 0:
        txt += "\n    # Set exogenous variables\n"
        for i in range(n_syms_exog):
            if exogd[i] in arr_eqs:
                txt += f"    {exogd[i]} = exog[{i}]\n"

    txt += "\n    # Function:"
    txt += f"\n    function = np.zeros({n_equations})"
    txt += "\n"

    for i in range(n_equations):
        val = D[0][i]
        txt_eq += f"    function[{i}] = {val}\n"

    txt += txt_eq
    txt += """
    if order == 0:
        return function
    """ 
    
    if order >= 1:
        # Jacobian
        txt += "\n    # Jacobian: \n"
        if bSparse:
            txt += "    row_ind = []; col_ind = []; jacobian = []\n"
        else: 
            txt += f"    jacobian = np.zeros(({n_equations},{n_syms}))\n"

        for i in range(n_equations):
            for j in range(n_syms):
                val = D[1][i,j]
                if val != 0:
                    if bSparse:
                        txt_der += f"    row_ind.append({i}); col_ind.append({j}); jacobian.append({val})\n"
                    else:
                        txt_der += f"    jacobian[{i},{j}] = {val}\n"

        txt += txt_der
        txt += "\n    if order == 1:\n"
        if bSparse:        
            txt += "        return [function, jacobian, row_ind, col_ind]\n"
        else:
            txt += "        return [function, jacobian]\n"


    if order >= 2:
        # Hessian
        txt += "\n    # Hessian: \n"
        if bSparse: 
            txt += "    row_ind = []; col_ind = []; mat_ind = []; hessian = []\n"
        else:                
            txt += f"    hessian = np.zeros(({n_equations},{n_syms},{n_syms}))\n"

        for n in range(n_equations):
            for i in range(n_syms):
                for j in range(n_syms):
                    val = D[2][n,i,j]
                    if bSparse:
                        if val is not None:
                            if val != 0:
                                txt_der2 += f"    row_ind.append({i}); col_ind.append({j}); mat_ind.append({n}); hessian.append({val})\n"
                        else:
                            i1, j1 = sorted( (i,j) )
                            if D[2][n,i1,j1] != 0:
                                txt_der2 += f"    row_ind.append({i1}); col_ind.append({j1}); mat_ind.append({n}); hessian.append({val})\n"
                    else:
                        if val is not None:
                            if val != 0:
                                txt_der2 += f"    hessian[{n},{i},{j}] = {val}\n"
                        else:
                            i1, j1 = sorted( (i,j) )
                            if D[2][n,i1,j1] != 0:
                                txt_der2 += f"    hessian[{n},{i},{j}] = hessian[{n},{i1},{j1}]\n"

        txt += txt_der2
        txt += "\n    if order == 2:\n"
        if bSparse:        
            txt += "        return [function, jacobian, hessian, mat_ind, row_ind, col_ind]"
        else:
            txt += "        return [function, jacobian, hessian]"


    if order >= 3:
        # Hessian
        txt += "\n\n\n    # Third order partial derivatives: \n"
        if bSparse: 
            txt +=  "    row_ind = []; col_ind = []; sym_ind = []; mat_ind = []; derivative_of_hessian = []\n"
        else:                
            txt += f"    derivative_of_hessian = np.zeros(({n_equations},{n_syms},{n_syms},{n_syms}))\n"

        for n in range(n_equations):
            for i in range(n_syms):
                for j in range(n_syms):
                    for k in range(n_syms):
                        val = D[3][n,i,j,k]
                        if bSparse:
                            if val is not None:
                                if val != 0:
                                    txt_der3 += f"    row_ind.append({i}); col_ind.append({j}); sym_ind.append({k}); mat_ind.append({n}); derivative_of_hessian.append({val})\n" 
                            else:
                                i1, j1, k1 = sorted( (i,j,k) )
                                if D[3][n,i1,j1,k1] != 0:                                       
                                    txt_der3 += f"    row_ind.append({i1}); col_ind.append({j1}); sym_ind.append({k1}); mat_ind.append({n}); derivative_of_hessian.append({val})\n" 
  
                        else:
                            if val is not None:
                                if val != 0:
                                    txt_der3 += f"    derivative_of_hessian[{n},{i},{j},{k}] = {val}\n"
                            else:
                                i1, j1, k1 = sorted( (i,j,k) )
                                if D[3][n,i1,j1,k1] != 0:
                                    txt_der3 += f"    derivative_of_hessian[{n},{i},{j},{k}] = derivative_of_hessian[{n},{i1},{j1},{k1}]\n"

        txt += txt_der3
        txt += "\n    if order == 3:\n"
        if bSparse:        
            txt += "        return [function, jacobian, hessian, derivative_of_hessian, mat_ind, row_ind, col_ind, sym_ind]"
        else:
            txt += "        return [function, jacobian, hessian, derivative_of_hessian]"

    #print('Function:')
    #print(txt)
    
    if not out is None:
        path = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.abspath(os.path.join(path, out+'.py'))
        try:
            with open(file_path, 'w') as f:
                f.write(txt)
        except:
            pass


    m = {}
    m['division'] = division

    exec(txt, m)
    fun = m[function_name]
    
    if txt_eq.endswith("\n"):   txt_eq   = txt_eq[:-1]
    if txt_der.endswith("\n"):  txt_der  = txt_der[:-1]
    if txt_der2.endswith("\n"): txt_der2 = txt_der2[:-1]
    if txt_der3.endswith("\n"): txt_der3 = txt_der3[:-1]
    if txt.endswith("\n"):      txt      = txt[:-1]

    return fun,txt_eq,txt_der,txt_der2,txt_der3,txt


def compile_function(equations,syms,params,syms_exog=[],eq_vars=[],function_name='func',out='f_func',b=False,model_name='',log_variables=[]):
    """
    From a list of equations and variables, define a multivariate functions with higher order derivatives.
    
    Parameters:
        :param equations: List of equations.
        :type equations: list.
        :param syms: Symbols.
        :type syms: list.
        :param params: Parameters.
        :type params: list.
        :param syms_exog: Exogenous variables symbols.
        :type syms_exog: list
        :param eq_vars: List of variables in each equation.
        :type eq_vars: list.
        :param function_name: Function name.
        :type function_name: str.
        :param out: Name of file that contains this function source code.
        :type out: str.
        :param b: If True use JAX package, otherwise - numpy.
        :type b: bool.
        :param model_name: Model name.
        :type model_name: str.
        :returns:  Compiled function.
    """
    import re
    from snowdrop.src.preprocessor.symbolic import stringify
    
    n_equations = len(equations)
    n_syms = len(syms)
    n_params = len(params)
    n_syms_exog = len(syms_exog)
    variables = [s[0] for s in syms]
    exogenous = [s[0] for s in syms_exog]
    
    if bool(log_variables):  
        eqs_sym = log_normalize_equations(equations,variables+exogenous,log_variables,True)
    else:
        eqs_sym = normalize_equations(equations,variables+exogenous,True)

    symsd = [stringify((a,b)) for a,b in syms]
    exogd = [stringify((a,b)) for a,b in syms_exog]
    paramsd = [stringify(a) for a in params]
    D = higher_order_diff(eqs_sym, symsd, eq_vars, order=0)    
    eqs = " ".join([str(x) for x in D[0]])

    delimiters = " ", ",", ";", "*", "/", ":", "+", "-", "^", "{", "}", "(", ")","[","]"
    regexPattern = '|'.join(map(re.escape, delimiters))
    arr_eqs = re.split(regexPattern,eqs)
    arr_eqs = list(filter(None,arr_eqs))
    
    if b:
        txt = """from jax import jit
    
@jit
""" 
    else:
        txt = """from numba import njit
    
@njit
"""  
    txt += "def {function_name}(x,p,exog=[0]):".format(function_name=function_name)
    txt += """
    
    ### This code was generated by Python.
    ### {model_name}
    
    from sympy import DiracDelta
    from snowdrop.src.preprocessor.condition import IfThenElse,IfThen,Derivative,Subs,Positive,Negative,myzif
    from snowdrop.src.preprocessor.functions import Heaviside
""".format(model_name=model_name)

    if b:
        txt += """
    import jax.numpy as np
    from jax.numpy import log,exp,sin,cos,tan,sqrt,sign
    from jax.numpy import maximum as Max, minimum as Min, abs as Abs
    
"""
    else:
        txt += """
    import numpy as np
    from numpy import log,exp,sin,cos,tan,sqrt,sign
    from numpy import maximum as Max, minimum as Min, abs as Abs
    
"""
    txt += "    # Initialize variables\n"
    for i in range(n_syms):
        if symsd[i] in arr_eqs:
            txt += f"    {symsd[i]} = x[{i}]\n"

    txt += "\n"

    txt += "\n    # Set parameters\n"
    for i in range(n_params):
        txt += f"    {paramsd[i]} = p[{i}]\n"
        
    if n_syms_exog > 0:
        txt += "\n    # Set exogenous variables\n"
        for i in range(n_syms_exog):
            if exogd[i] in arr_eqs:
                txt += f"    {exogd[i]} = exog[{i}]\n"

    txt += "\n    # Function:\n"
    
    txt += "\n    function = list()\n"
    for i in range(n_equations):
        txt += f"    function.append({D[0][i]})\n"

    txt += """
    return np.array(function)
""" 
    if not out is None:
        path = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.abspath(os.path.join(path, out+'.py'))
        try:
            with open(file_path, 'w') as f:
                f.write(txt)
        except:
            pass
            
    m = {}
    m['division'] = division

    exec(txt, m)
    fun = m[function_name]

    return fun,txt


def compile_jacobian(equations,syms,params,syms_exog=[],eq_vars=[],function_name='jacob',out = 'f_jacob',bSparse=False,b=False,model_name='',log_variables=[]):
    """
    From a list of equations and variables, define a multivariate functions with higher order derivatives.
    
Parameters:
    :param equations: List of equations.
    :type equations: list.
    :param syms: Symbols.
    :type syms: list.
    :param params: Parameters.
    :type params: list.
    :param syms_exog: Exogenous variables symbols.
    :type syms_exog: list
    :param eq_vars: List of variables in each equation.
    :type eq_vars: list.
    :param order: Order of partial derivatives of Jacobian.
    :type order: int.
    :param function_name: Function name.
    :type function_name: str.
    :param out: Name of file that contains this function source code.
    :type out: str.
    :param b: If True use cupy package, otherwise - numpy.
    :type b: bool.
    :param bSparse: True if sparse algebra is used.
    :type bSparse: bool.
    :param model_name: Model name.
    :type model_name; str.
    :returns:  Compiled jacobian.
    """
    import re
    from snowdrop.src.preprocessor.symbolic import stringify
    
    n_equations = len(equations)
    n_syms = len(syms)
    n_params = len(params)
    n_syms_exog = len(syms_exog)
    variables = [s[0] for s in syms]
    exogenous = [s[0] for s in syms_exog]
    
    if bool(log_variables):  
        eqs_sym = log_normalize_equations(equations,variables+exogenous,log_variables,True)
    else:
        eqs_sym = normalize_equations(equations,variables+exogenous,True)

    symsd = [stringify((a,b)) for a,b in syms]
    exogd = [stringify((a,b)) for a,b in syms_exog]
    paramsd = [stringify(a) for a in params]
    D = higher_order_diff(eqs_sym, symsd, eq_vars, order=1)    
    eqs = " ".join([str(x) for x in D[0]])

    delimiters = " ", ",", ";", "*", "/", ":", "+", "-", "^", "{", "}", "(", ")","[","]"
    regexPattern = '|'.join(map(re.escape, delimiters))
    arr_eqs = re.split(regexPattern,eqs)
    arr_eqs = list(filter(None,arr_eqs))
    
    txt = """from numba import njit
    
@njit
""" 
    txt += "def {function_name}(x,p,exog=[0]):".format(function_name=function_name)
    txt += f"""
    
    ### This code was generated by Python.
    ### {model_name}
    
    from sympy import DiracDelta
    from snowdrop.src.preprocessor.condition import IfThenElse,IfThen,Derivative,Subs,Positive,Negative,myzif
    from snowdrop.src.preprocessor.functions import Heaviside
""".format(model_name=model_name)
  
    if b:
        txt += """
    import jax.numpy as np
    from jax.numpy import log,exp,sin,cos,tan,sqrt,sign
    from jax.numpy import maximum as Max, minimum as Min, abs as Abs
    
"""
    else:
        txt += """
    import numpy as np
    from numpy import log,exp,sin,cos,tan,sqrt,sign
    from numpy import maximum as Max, min as Min, abs as Abs
     
"""   
    txt += "    # Initialize variables\n"
    txt += "    _xi_1 = 0 \n"
    txt += "    _xi_2 = 0 \n"
    txt += "    _xi_3 = 0 \n"
    
    for i in range(n_syms):
        if symsd[i] in arr_eqs:
            txt += f"    {symsd[i]} = x[{i}]\n"

    txt += "\n"

    txt += "\n    # Set parameters\n"
    for i in range(n_params):
        txt += f"    {paramsd[i]} = p[{i}]\n"
        
    if n_syms_exog > 0:
        txt += "\n    # Set exogenous variables\n"
        for i in range(n_syms_exog):
            if exogd[i] in arr_eqs:
                txt += f"    {exogd[i]} = exog[{i}]\n"
  
    # Jacobian
    txt += "\n    # Jacobian: \n"
    if bSparse:
        txt += "    row_ind = []; col_ind = []; function = []\n"
    else: 
        txt += f"    jacobian = np.zeros(({n_equations},{n_syms}))\n"

    for i in range(n_equations):
        for j in range(n_syms):
            val = D[1][i,j]
            if val != 0:
                if bSparse:
                    txt += f"    row_ind.append({i}); col_ind.append({j}); jacobian.append({val})\n"
                else:
                    txt += f"    jacobian[{i},{j}] = {val}\n"

    if bSparse:        
        txt += "\n    return jacobian,row_ind,col_ind \n"
    else:
        txt += "\n    return jacobian \n"
    
        
    if not out is None:
        path = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.abspath(os.path.join(path, out+'.py'))
        try:
            with open(file_path, 'w') as f:
                f.write(txt)
        except:
            pass
            
    m = {}
    m['division'] = division

    exec(txt, m)
    fun = m[function_name]

    return fun,txt
    

def normalize_equations(equations,variables,bAST=True,debug=True):
    """
    Modify equations by replacing lead/lag variables with the dated variables.
    
    For example, replaces variable:
        1. var(-1) -> var__m1_
        2. var(1) -> var__p1_
        3. var -> var__
        
    Parameters:
        :param equations: List of equations.
        :type equations: list.
        :param variables: List of variables in each equation.
        :type variables: list.
        :param bAST: If True use AST tree to parse equations, stringify variables and convert to sumpy expressions.
                     Otherwise parse equations with the aid of regular expressions of ReGex module. 
        :type bAST: bool.
    """
    from snowdrop.src.preprocessor.symbolic import stringify,normalize
    
    if bAST:
        
        eqs = [eq for eq in equations] # TEMP: compatibility fix when eqs is an Odict:
        if isinstance(eqs[0], str):
            if debug:
                count = 0
                new_eqs = []
                for eq in eqs:
                    count += 1
                    try:
                        new_eqs.append(ast.parse(eq).body[0])
                    except:
                        print()
                        cprint(f"Error parsing equation #{count}: {eq}","red")
            else:
                new_eqs = [ast.parse(eq).body[0] for eq in eqs]
            eqs_std = [normalize(eq, variables=variables) for eq in new_eqs]
            eqs_sym = [ast_to_sympy(eq) for eq in eqs_std]
        else:
            # assume we have ASTs
            eqs_sym = eqs
        
    else:
        
        import re
        delimiters = " ", ",", ";", "*", "/", ":", "=", "+", "-", "(", ")"
        regexPattern = '|'.join(map(re.escape, delimiters))
        eqs_sym = []
        for i,eqtn in enumerate(equations):
            eq = ''
            e = eqtn.replace(' ','')
            arr = re.split(regexPattern,e)
            arr = [x for x in arr if not x.isdigit()]
            arr = list(filter(None,arr))
            for v in arr:
                if len(e) == 0:
                    break
                if debug:
                    print()
                    print('eq:',eq)
                    print('e:',e)
                    print(v)
                if v in variables:
                    ind = max(0,e.find(v))
                    eq += e[:ind]
                    e = e[ind:]
                    if len(e) > len(v) and e[len(v)] == '(':
                        ind2 = e.find(')')
                        if ind2 > 0:
                            lead_lag = e[1+len(v):ind2]
                            match = re.match(r"[-+]?\d+",lead_lag)
                            if not match is None:
                                try:
                                    i_lead_lag = int(lead_lag)
                                    eq += stringify((v,i_lead_lag))
                                except:
                                    eq += stringify((v,0))
                            e = e[1+ind2:]
                        else:
                            cprint("{0}:  Equation {1} - variable {2} doesn't have a closing parenthesis: ".format(i,eqtn,v),"red")
                            eq += v+'__'
                            e = e[len(v):]
                    elif len(e) >= len(v):
                        eq += v+'__'
                        e = e[len(v):]
                else:
                    ind = max(0,e.find(v))
                    if len(e) > len(v)+ind:
                        eq += e[:len(v)+ind]
                        e = e[len(v)+ind:]
            eq += e
            eqs_sym.append(eq) 
            
        if SYMENGINE:
            eqs_sym = [sy.sympify(eq) for eq in eqs_sym]
        else:
            eqs_sym = [sy.sympify(eq,locals=ns) for eq in eqs_sym]
        
    return eqs_sym


def log_normalize_equations(equations,variables,log_variables=[],bAST=True,debug=False):
    """
    Modify equations by replacing variables with their logs with time shift.
    
    For example, replaces variable:
        1. var(-1) -> log(var__m1_)
        2. var(1) -> log(var__p1_)
        3. var -> log(var__)
        
    Parameters:
        :param equations: List of equations.
        :type equations: list.
        :param variables: List of variables in each equation.
        :type variables: list.
        :param bAST: If True use AST tree to parse equations, stringify variables and convert to sumpy expressions.
                     Otherwise parse equations with the aid of regular expressions of ReGex module. 
        :type bAST: bool.
    """
    from snowdrop.src.preprocessor.symbolic import stringify_variable 
    from snowdrop.src.preprocessor.symbolic import log_stringify_variable 
    from snowdrop.src.preprocessor.symbolic import log_normalize
    
    if bAST:
        
        eqs = [eq for eq in equations] # TEMP: compatibility fix when eqs is an Odict:
        if isinstance(eqs[0], str):
            if debug:
                count = 0
                new_eqs = []
                for eq in eqs:
                    count += 1
                    try:
                        new_eqs.append(ast.parse(eq).body[0])
                    except:
                        print()
                        cprint(f"Error parsing equation #{count}: {eq}","red")
            else:
                new_eqs = list([ast.parse(eq).body[0] for eq in eqs])
            eqs_std = list([log_normalize(eq, variables=variables, log_variables=log_variables) for eq in new_eqs])
            eqs_sym = list([ast_to_sympy(eq) for eq in eqs_std])
        else:
            # assume we have ASTs
            eqs_sym = eqs
        
    else:
        
        import re
        delimiters = " ", ",", ";", "*", "/", ":", "=", "+", "-", "(", ")"
        regexPattern = '|'.join(map(re.escape, delimiters))
        eqs_sym = []
        for i,eqtn in enumerate(equations):
            eq = ''
            e = eqtn.replace(' ','')
            arr = re.split(regexPattern,e)
            arr = [x for x in arr if not x.isdigit()]
            arr = list(filter(None,arr))
            for v in arr:
                if len(e) == 0:
                    break
                if debug:
                    print()
                    print('eq:',eq)
                    print('e:',e)
                    print(v)
                if v in variables:
                    ind = max(0,e.find(v))
                    eq += e[:ind]
                    e = e[ind:]
                    if len(e) > len(v) and e[len(v)] == '(':
                        ind2 = e.find(')')
                        if ind2 > 0:
                            lead_lag = e[1+len(v):ind2]
                            match = re.match(r"[-+]?\d+",lead_lag)
                            if not match is None:
                                if v in log_variables:
                                    try:
                                        i_lead_lag = int(lead_lag)
                                        eq += log_stringify_variable((v,i_lead_lag))
                                    except:
                                        eq += log_stringify_variable((v,0))
                                else:
                                    try:
                                        i_lead_lag = int(lead_lag)
                                        eq += stringify_variable((v,i_lead_lag))
                                    except:
                                        eq += stringify_variable((v,0))
                                    
                            e = e[1+ind2:]
                        else:
                            cprint("{0}:  Equation {1} - variable {2} doesn't have a closing parenthesis: ".format(i,eqtn,v),"red")
                            eq += v+'__'
                            e = e[len(v):]
                    elif len(e) >= len(v):
                        eq += v+'__'
                        e = e[len(v):]
                else:
                    ind = max(0,e.find(v))
                    if len(e) > len(v)+ind:
                        eq += e[:len(v)+ind]
                        e = e[len(v)+ind:]
            eq += e
            eqs_sym.append(eq) 
        
        if SYMENGINE:
            eqs_sym = [sy.sympify(eq) for eq in eqs_sym]
        else:
            eqs_sym = [sy.sympify(eq,locals=ns) for eq in eqs_sym]
        
    return eqs_sym

    
def get_indices(equations,syms,eq_vars=[]):
    """
    From a list of equations and variables, define a multivariate functions with higher order derivatives.
    
    Parameters:
        :param equations: List of equations.
        :type equations: list.
        :param syms: Symbols.
        :type syms: list.
        :returns:  row and column indices.
    """
    from snowdrop.src.preprocessor.symbolic import stringify
    
    n_equations = len(equations)
    n_syms = len(syms)
    variables = [s[0] for s in syms]
    eqs_sym = normalize_equations(equations,variables,True)

    symsd = [stringify((a,b)) for a,b in syms]
    D = higher_order_diff(eqs_sym, symsd, eq_vars, order=1) 
    
    ind = list(); row_ind = list(); col_ind = list()
    for i in range(n_equations):
        for j in range(n_syms):
            val = D[1][i,j]
            if val != 0:
                ind.append([i,j])
                row_ind.append(i)
                col_ind.append(j)
                
    return row_ind, col_ind
    
    
def test_deriv():
    """Test derivatives of equations."""
    # list of equations
    eqs = ['(a*x + 2*b)**2', 'y + exp(a + 2*b)*c(1)']

    # list of variables (time symbols)
    syms = [('a',0),('b',0),('c',1)]
    eq_vars = [['a__','b__'],['a__','b__','c__p1_']]

    # list of parameters
    params = ['x','y']

    # compile a function and its derivatives
    fun,eqs,der,der2,der3,txt = compile_higher_order_function(equations=eqs,syms=syms,params=params,syms_exog=[],eq_vars=eq_vars,order=3)

    # evaluate the function
    import numpy as np
    v = np.array([0.0, 0.0, 0.0])
    p = np.array([1.0, 0.5])

    from f_dynamic import anonymous as func
    f0,f1,f2,f3 = fun.py_func(v, p, order=3)

    assert(f0[1]==0.5)
    assert(f1[1,2]==1)
    assert(f2[0,1,1]==8)
    assert(f2[0,1,0]==f2[0,0,1])

    
def test_deriv2():
    """Test derivatives of equations."""
    
    eq = '(a*x + 2*b)**2 + IfThenElse(1,x,exp(a + 2*b)*y(1))'

    # list of variables (time symbols)
    syms = [('x',0),('y',0),('y',1)]
    eq_vars = []
    eq_vars.append(['x__','y__','y__1_'])

    # list of parameters
    params = ['a','b']

    # compile a function with its derivatives
    fun,eqs,der,der2,der3,txt = compile_higher_order_function(equations=[eq],syms=syms,params=params,syms_exog=[],eq_vars=eq_vars,order=1)
    #print("Function:\n",txt)
    #print("Derivatives:\n",der)

    
if __name__ == '__main__':
    """
    Main entry point
    """
    # equations = ['p_pdot1*PDOT(+1) + (1-p_pdot1)*PDOT(-1) + p_pdot2*(g**2/(g-Y) - g) + p_pdot3*(g**2/(g-Y(-1)) - g)', 'RS - p_pdot1*PDOT(+1) - (1-p_pdot1)*PDOT(-1)', 'p_rs1*PDOT + Y', 'p_y1*Y(-1) - p_y2*RR - p_y3*RR(-1) + e']
    # variables = ['PDOT','RR','RS','Y']
    # eqs = normalize_equations(equations=equations,variables=variables,bAST=True)
    # print(eqs)
    
    # from snowdrop.src.preprocessor.symbolic import normalize
    # variables = ["E_ISR_TAXL_RAT","E_WRL_PRODOIL_R","ISR_ACT_R","ISR_BREVAL_N","ISR_B_N","ISR_B_RAT","ISR_CFOOD_R","ISR_CNCOM_R","ISR_COIL_R","ISR_COM_FE_R","ISR_COM_RK_P","ISR_CPINCOM_P","ISR_CPIX_P","ISR_CPI_P","ISR_CURBAL_N","ISR_C_LIQ_R","ISR_C_OLG_R","ISR_C_R","ISR_C_RAT","ISR_DELTA","ISR_EPS","ISR_FACTFOOD_R","ISR_FACT_R","ISR_FXPREM","ISR_GC_N","ISR_GC_R","ISR_GC_RAT","ISR_GDEF_N","ISR_GDEF_RAT","ISR_GDEF_TAR","ISR_GDPINC_N","ISR_GDPSIZE","ISR_GDP_FE_R","ISR_GDP_N","ISR_GDP_R","ISR_GE_N","ISR_GISTOCK_R","ISR_GI_N","ISR_GI_R","ISR_GI_RAT","ISR_GNP_R","ISR_GOVCHECK","ISR_GSUB_N","ISR_GTARIFF_N","ISR_G_R","ISR_IFOODA_R","ISR_IFOOD_R","ISR_IMETALA_R","ISR_IMETAL_R","ISR_IM_R","ISR_INFCPI","ISR_INFCPIX","ISR_INFEXP","ISR_INFL","ISR_INFPIM","ISR_INFWAGE","ISR_INFWAGEEFF","ISR_INFWEXP","ISR_INT","ISR_INT10","ISR_INTC","ISR_INTCORP","ISR_INTCOST_N","ISR_INTCOST_RAT","ISR_INTGB","ISR_INTMP","ISR_INTMPU","ISR_INTNFA","ISR_INTRF","ISR_INTRF10","ISR_INTXM10","ISR_INVESTP_R","ISR_INVEST_R","ISR_INVEST_RAT","ISR_IOILA_R","ISR_IOIL_R","ISR_IT_R","ISR_IT_RAT","ISR_J","ISR_KG_R","ISR_K_R","ISR_LABH_FE_R","ISR_LABH_R","ISR_LAB_FE_R","ISR_LAB_R","ISR_LF_FE_R","ISR_LF_R","ISR_LSTAX_RAT","ISR_MET_RK_P","ISR_MKTPREM","ISR_MKTPREMSM","ISR_MPC","ISR_MPCINV","ISR_NFAREVAL_N","ISR_NFA_D","ISR_NFA_RAT","ISR_NPOPB_R","ISR_NPOPH_R","ISR_NPOP_R","ISR_NTRFPSPILL_FE_R","ISR_OILRECEIPT_N","ISR_OILSUB_N","ISR_PART","ISR_PARTH","ISR_PARTH_DES","ISR_PARTH_FE","ISR_PARTH_W","ISR_PART_DES","ISR_PART_FE","ISR_PCFOOD_P","ISR_PCOIL_P","ISR_PCW_P","ISR_PC_P","ISR_PFM_P","ISR_PFOOD_P","ISR_PGDP_P","ISR_PGDP_P_AVG","ISR_PG_P","ISR_PIMADJ_P","ISR_PIMA_P","ISR_PIM_P","ISR_PIT_P","ISR_PI_P","ISR_PMETAL_P","ISR_POIL_P","ISR_POIL_P_SUB","ISR_PRIMSUR_N","ISR_PRIMSUR_TAR","ISR_PRODFOOD_R","ISR_PSAVING_N","ISR_PXMF_P","ISR_PXMUNADJ_P","ISR_PXM_P","ISR_PXT_P","ISR_Q_P","ISR_R","ISR_R10","ISR_RC","ISR_RC0_SM","ISR_RC0_WM","ISR_RCI","ISR_RCORP","ISR_REER","ISR_RK_P","ISR_RPREM","ISR_R_NEUT","ISR_SOVPREM","ISR_SOVPREMSM","ISR_SUB_OIL","ISR_TAU_C","ISR_TAU_K","ISR_TAU_L","ISR_TAU_OIL","ISR_TAXC_N","ISR_TAXC_RAT","ISR_TAXK_N","ISR_TAXK_RAT","ISR_TAXLH_N","ISR_TAXL_N","ISR_TAXL_RAT","ISR_TAXOIL_N","ISR_TAX_N","ISR_TAX_RAT","ISR_TB_N","ISR_TFPEFFECT_R","ISR_TFPKGSPILL_FE_R","ISR_TFPSPILL_FE_R","ISR_TFP_FE_R","ISR_TFP_FE_R_AVG","ISR_TFP_R","ISR_TM","ISR_TPREM","ISR_TRANSFER_LIQ_N","ISR_TRANSFER_OLG_N","ISR_TRANSFER_RAT","ISR_TRANSFER_TARG_N","ISR_TRANSFER_TARG_RAT","ISR_TRFPSPILL_FE_R","ISR_UFOOD_R","ISR_UNR","ISR_UNRH","ISR_UNRH_FE","ISR_UNR_FE","ISR_USA_SM","ISR_USA_WM","ISR_WAGEEFF_N","ISR_WAGEH_N","ISR_WAGE_N","ISR_WF_R","ISR_WH_R","ISR_WK_N","ISR_WO_R","ISR_W_R","ISR_W_R_AVG","ISR_XFOOD_R","ISR_XMA_R","ISR_XM_R","ISR_XT_R","ISR_XT_RAT","ISR_YCAP_N","ISR_YD_R","ISR_YLABH_N","ISR_YLAB_N","ISR_Z","ISR_Z_AVG","ISR_Z_NFA","PFOOD_P","PMETAL_P","POIL_P","RC0_ACT_R","RC0_BREVAL_N","RC0_B_N","RC0_B_RAT","RC0_CFOOD_R","RC0_CNCOM_R","RC0_COIL_R","RC0_COM_FE_R","RC0_COM_RK_P","RC0_CPINCOM_P","RC0_CPIX_P","RC0_CPI_P","RC0_CURBAL_N","RC0_C_LIQ_R","RC0_C_OLG_R","RC0_C_R","RC0_C_RAT","RC0_DELTA","RC0_EPS","RC0_FACTFOOD_R","RC0_FACTMETAL_R","RC0_FACTOIL_R","RC0_FACT_R","RC0_FXPREM","RC0_GC_N","RC0_GC_R","RC0_GC_RAT","RC0_GDEF_N","RC0_GDEF_RAT","RC0_GDEF_TAR","RC0_GDPINC_N","RC0_GDPSIZE","RC0_GDP_FE_R","RC0_GDP_N","RC0_GDP_R","RC0_GE_N","RC0_GISTOCK_R","RC0_GI_N","RC0_GI_R","RC0_GI_RAT","RC0_GNP_R","RC0_GOVCHECK","RC0_GSUB_N","RC0_GTARIFF_N","RC0_G_R","RC0_IFOODA_R","RC0_IFOOD_R","RC0_IMETALA_R","RC0_IMETAL_R","RC0_IM_R","RC0_INFCPI","RC0_INFCPIX","RC0_INFEXP","RC0_INFL","RC0_INFPIM","RC0_INFWAGE","RC0_INFWAGEEFF","RC0_INFWEXP","RC0_INT","RC0_INT10","RC0_INTC","RC0_INTCORP","RC0_INTCOST_N","RC0_INTCOST_RAT","RC0_INTGB","RC0_INTMP","RC0_INTMPU","RC0_INTNFA","RC0_INTRF","RC0_INTRF10","RC0_INTXM10","RC0_INVESTP_R","RC0_INVEST_R","RC0_INVEST_RAT","RC0_IOILA_R","RC0_IOIL_R","RC0_ISR_SM","RC0_ISR_WM","RC0_IT_R","RC0_IT_RAT","RC0_J","RC0_KG_R","RC0_K_R","RC0_LABH_FE_R","RC0_LABH_R","RC0_LAB_FE_R","RC0_LAB_R","RC0_LF_FE_R","RC0_LF_R","RC0_LSTAX_RAT","RC0_MET_RK_P","RC0_MKTPREM","RC0_MKTPREMSM","RC0_MPC","RC0_MPCINV","RC0_MROYALTIES_N","RC0_MROYALTY","RC0_NFAREVAL_N","RC0_NFA_D","RC0_NFA_RAT","RC0_NPOPB_R","RC0_NPOPH_R","RC0_NPOP_R","RC0_NTRFPSPILL_FE_R","RC0_OILPAY_N","RC0_OILRECEIPT_N","RC0_OILSHARF","RC0_OILSUB_N","RC0_PART","RC0_PARTH","RC0_PARTH_DES","RC0_PARTH_FE","RC0_PARTH_W","RC0_PART_DES","RC0_PART_FE","RC0_PCFOOD_P","RC0_PCOIL_P","RC0_PCW_P","RC0_PC_P","RC0_PFM_P","RC0_PFOOD_P","RC0_PGDP_P","RC0_PGDP_P_AVG","RC0_PG_P","RC0_PIMADJ_P","RC0_PIMA_P","RC0_PIM_P","RC0_PIT_P","RC0_PI_P","RC0_PMETAL_P","RC0_POIL_P","RC0_POIL_P_SUB","RC0_PRIMSUR_N","RC0_PRIMSUR_TAR","RC0_PRODFOOD_R","RC0_PRODMETAL_R","RC0_PRODOIL_R","RC0_PSAVING_N","RC0_PXMF_P","RC0_PXMUNADJ_P","RC0_PXM_P","RC0_PXT_P","RC0_Q_P","RC0_R","RC0_R10","RC0_RC","RC0_RCI","RC0_RCORP","RC0_REER","RC0_RK_P","RC0_ROYALTIES_N","RC0_ROYALTIES_RAT","RC0_ROYALTY","RC0_RPREM","RC0_R_NEUT","RC0_SOVPREM","RC0_SOVPREMSM","RC0_SUB_OIL","RC0_TAU_C","RC0_TAU_K","RC0_TAU_L","RC0_TAU_OIL","RC0_TAXC_N","RC0_TAXC_RAT","RC0_TAXK_N","RC0_TAXK_RAT","RC0_TAXLH_N","RC0_TAXL_N","RC0_TAXL_RAT","RC0_TAXOIL_N","RC0_TAX_N","RC0_TAX_RAT","RC0_TB_N","RC0_TFPEFFECT_R","RC0_TFPKGSPILL_FE_R","RC0_TFPSPILL_FE_R","RC0_TFP_FE_R","RC0_TFP_FE_R_AVG","RC0_TFP_R","RC0_TM","RC0_TPREM","RC0_TRANSFER_LIQ_N","RC0_TRANSFER_N","RC0_TRANSFER_OLG_N","RC0_TRANSFER_RAT","RC0_TRANSFER_TARG_N","RC0_TRANSFER_TARG_RAT","RC0_TRFPSPILL_FE_R","RC0_UFOOD_R","RC0_UMETAL_R","RC0_UNR","RC0_UNRH","RC0_UNRH_FE","RC0_UNR_FE","RC0_UOIL_R","RC0_USA_SM","RC0_USA_WM","RC0_WAGEEFF_N","RC0_WAGEH_N","RC0_WAGE_N","RC0_WF_R","RC0_WH_R","RC0_WK_N","RC0_WO_R","RC0_W_R","RC0_W_R_AVG","RC0_XFOOD_R","RC0_XMA_R","RC0_XMETAL_R","RC0_XM_R","RC0_XOIL_R","RC0_XT_R","RC0_XT_RAT","RC0_YCAP_N","RC0_YD_R","RC0_YLABH_N","RC0_YLAB_N","RC0_Z","RC0_Z_AVG","RC0_Z_NFA","RPFOOD","RPMETAL","RPMETAL_ADJ","RPMETAL_AVG","RPOIL","RPOIL_ADJ","RPOIL_AVG","USA_ACT_R","USA_BREVAL_N","USA_B_N","USA_B_RAT","USA_CFOOD_R","USA_CNCOM_R","USA_COIL_R","USA_COM_FE_R","USA_COM_RK_P","USA_CPINCOM_P","USA_CPIX_P","USA_CPI_P","USA_CURBAL_N","USA_C_LIQ_R","USA_C_OLG_R","USA_C_R","USA_C_RAT","USA_DELTA","USA_EPS","USA_FACTFOOD_R","USA_FACTMETAL_R","USA_FACTOIL_R","USA_FACT_R","USA_FXPREM","USA_GC_N","USA_GC_R","USA_GC_RAT","USA_GDEF_N","USA_GDEF_RAT","USA_GDEF_TAR","USA_GDPINC_N","USA_GDPSIZE","USA_GDP_FE_R","USA_GDP_N","USA_GDP_R","USA_GE_N","USA_GISTOCK_R","USA_GI_N","USA_GI_R","USA_GI_RAT","USA_GNP_R","USA_GOVCHECK","USA_GSUB_N","USA_GTARIFF_N","USA_G_R","USA_IFOODA_R","USA_IFOOD_R","USA_IMETALA_R","USA_IMETAL_R","USA_IM_R","USA_INFCPI","USA_INFCPIX","USA_INFEXP","USA_INFL","USA_INFPIM","USA_INFWAGE","USA_INFWAGEEFF","USA_INFWEXP","USA_INT","USA_INT10","USA_INTC","USA_INTCORP","USA_INTCOST_N","USA_INTCOST_RAT","USA_INTGB","USA_INTMP","USA_INTMPU","USA_INTNFA","USA_INTRF","USA_INTRF10","USA_INTXM10","USA_INVESTP_R","USA_INVEST_R","USA_INVEST_RAT","USA_IOILA_R","USA_IOIL_R","USA_ISR_SM","USA_ISR_WM","USA_IT_R","USA_IT_RAT","USA_J","USA_KG_R","USA_K_R","USA_LABH_FE_R","USA_LABH_R","USA_LAB_FE_R","USA_LAB_R","USA_LF_FE_R","USA_LF_R","USA_LSTAX_RAT","USA_MET_RK_P","USA_MKTPREM","USA_MKTPREMSM","USA_MPC","USA_MPCINV","USA_MROYALTIES_N","USA_MROYALTY","USA_NFAREVAL_N","USA_NFA_D","USA_NFA_RAT","USA_NPOPB_R","USA_NPOPH_R","USA_NPOP_R","USA_NTRFPSPILL_FE_R","USA_OILPAY_N","USA_OILRECEIPT_N","USA_OILSHARF","USA_OILSUB_N","USA_PART","USA_PARTH","USA_PARTH_DES","USA_PARTH_FE","USA_PARTH_W","USA_PART_DES","USA_PART_FE","USA_PCFOOD_P","USA_PCOIL_P","USA_PCW_P","USA_PC_P","USA_PFM_P","USA_PFOOD_P","USA_PGDP_P","USA_PGDP_P_AVG","USA_PG_P","USA_PIMADJ_P","USA_PIMA_P","USA_PIM_P","USA_PIT_P","USA_PI_P","USA_PMETAL_P","USA_POIL_P","USA_POIL_P_SUB","USA_PRIMSUR_N","USA_PRIMSUR_TAR","USA_PRODFOOD_R","USA_PRODMETAL_R","USA_PRODOIL_R","USA_PSAVING_N","USA_PXMF_P","USA_PXMUNADJ_P","USA_PXM_P","USA_PXT_P","USA_Q_P","USA_R","USA_R10","USA_RC","USA_RC0_SM","USA_RC0_WM","USA_RCI","USA_RCORP","USA_REER","USA_RK_P","USA_ROYALTIES_N","USA_ROYALTIES_RAT","USA_ROYALTY","USA_RPREM","USA_R_NEUT","USA_SOVPREM","USA_SOVPREMSM","USA_SUB_OIL","USA_TAU_C","USA_TAU_K","USA_TAU_L","USA_TAU_OIL","USA_TAXC_N","USA_TAXC_RAT","USA_TAXK_N","USA_TAXK_RAT","USA_TAXLH_N","USA_TAXL_N","USA_TAXL_RAT","USA_TAXOIL_N","USA_TAX_N","USA_TAX_RAT","USA_TB_N","USA_TFPEFFECT_R","USA_TFPKGSPILL_FE_R","USA_TFPSPILL_FE_R","USA_TFP_FE_R","USA_TFP_FE_R_AVG","USA_TFP_R","USA_TM","USA_TPREM","USA_TRANSFER_LIQ_N","USA_TRANSFER_N","USA_TRANSFER_OLG_N","USA_TRANSFER_RAT","USA_TRANSFER_TARG_N","USA_TRANSFER_TARG_RAT","USA_TRFPSPILL_FE_R","USA_UFOOD_R","USA_UMETAL_R","USA_UNR","USA_UNRH","USA_UNRH_FE","USA_UNR_FE","USA_UOIL_R","USA_WAGEEFF_N","USA_WAGEH_N","USA_WAGE_N","USA_WF_R","USA_WH_R","USA_WK_N","USA_WO_R","USA_W_R","USA_W_R_AVG","USA_XFOOD_R","USA_XMA_R","USA_XMETAL_R","USA_XM_R","USA_XOIL_R","USA_XT_R","USA_XT_RAT","USA_YCAP_N","USA_YD_R","USA_YLABH_N","USA_YLAB_N","USA_Z","USA_Z_AVG","USA_Z_NFA","WRL_GDP_FE_METAL_R","WRL_GDP_FE_OIL_R","WRL_GDP_FE_R","WRL_GDP_METAL_R","WRL_GDP_OIL_R","WRL_GDP_R","WRL_PRODFOOD_R","WRL_PRODMETAL_R","WRL_PRODOIL_R","WRL_XFOOD_R","WRL_XMETAL_R","WRL_XOIL_R","WTRADE_FOOD_N","WTRADE_FOOD_R","WTRADE_METAL_N","WTRADE_METAL_R","WTRADE_M_N","WTRADE_M_R","WTRADE_OIL_N","WTRADE_OIL_R"]
    # eq = "((-(USA_XOIL5)) * (LambertW((((-(USA_XOIL_R(-(1)))) * ((USA_XOIL5) - (1))) * (exp((((CONSTANT_USA_XOIL_R) * (USA_GDP_R_BAR) + (E_USA_XOIL_R) * (USA_XOIL_R_BAR) + (FILTER_USA_XOIL_R) * (USA_GDP_R_BAR)) - (USA_UOIL_R_BAR) + USA_UOIL_R + (USA_XOIL5) * ((((((((-(CONSTANT_USA_XOIL_R)) * (USA_GDP_R_BAR) + (CONSTANT_USA_XOIL_R) * (USA_XOIL2) + (E_USA_XOIL_R) * (USA_XOIL2)) - ((E_USA_XOIL_R) * (USA_XOIL_R_BAR))) - ((FILTER_USA_XOIL_R) * (USA_GDP_R_BAR)) + (FILTER_USA_XOIL_R) * (USA_XOIL2) + USA_UOIL_R_BAR) - (USA_UOIL_R) + (USA_XOIL1) * (log(RPOIL))) - ((USA_XOIL1) * (log(RPOIL(-(1))))) + ((USA_XOIL2) * (USA_XOIL3)) * (log(RPOIL(-(1)))) + (USA_XOIL2) * (log((USA_FACTOIL_R(-(1))) / (USA_GREAL_SS)))) - ((USA_XOIL2) * (log((USA_XOIL_R(-(1))) / (USA_GREAL_SS)))) + (USA_XOIL4) * (log(USA_FACTOIL_R))) - ((USA_XOIL4) * (log(USA_FACTOIL_R(-(1))))))) / (USA_XOIL5)))) / (USA_XOIL5)))) / ((USA_XOIL5) - (1))"
    # eq_ast = ast.parse(eq).body[0]
    # eq_std = normalize(eq_ast,variables=variables)
    # eq_sym = ast_to_sympy(eq_std)
            
    test_deriv()
    test_deriv2()
    
#    from time import time
#    t0 = time()
#    n = 1000
#    size = 2
#    series=non_decreasing_series(n, size)  
#    #print(series)
#    elapsed = time() - t0
#    print(elapsed,len(series))

    
