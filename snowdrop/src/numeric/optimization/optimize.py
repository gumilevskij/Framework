# -*- coding: utf-8 -*-
"""
Generic optimization module.

@author: A.Goumilevski
"""
import os
from time import time
import numpy as np
from warnings import filterwarnings
#from numba import njit
from scipy.optimize import minimize, root, Bounds
from scipy.optimize import LinearConstraint
from scipy.optimize import NonlinearConstraint
from dataclasses import dataclass
from snowdrop.src.misc.termcolor import cprint
from snowdrop.src.model.util import importModel
#from snowdrop.src.model.util import loadLibrary
from snowdrop.src.model.util import getLimits, getConstraints
#from snowdrop.src.model.util import print_path_solution_status
from snowdrop.src.graphs.util import bar_plot
from snowdrop.src.preprocessor.function import get_function_and_jacobian as fun
from snowdrop.src.utils.prettyTable import PrettyTable

fpath = os.path.dirname(os.path.abspath(__file__))
it = 0

@dataclass
class Data:
    success: bool; x: float; fun: float; nfev: int = 0; message: str = ""

#@njit
def solver(model):
    """
    Find solution of linear constraint optimization problem.

    Parameters:
            :param model: Model object.
            :type model: Model.
            :returns: solution of optimization problem.
    """
    filterwarnings("ignore")

    MAXITER = 1000
    
    solver = model.symbolic.SOLVER
    method = model.symbolic.METHOD
    if method is None:
        sign = 1
    else:
        method = method.lower()
        if method == "minimize":
            sign = 1 
        elif method == "maximize":
            sign = -1
        else:
            cprint("\nPlease choose either Maximize or Minimize method.\n","red")
            raise ValueError(f"Method {method} is not implemented.")

    constraints = model.symbolic.constraints
    obj_func    = model.symbolic.objective_function
    var_names   = model.symbols['variables']
    var_values  = model.calibration['variables']
    #var        = dict(zip(var_names,var_values))
    par_names   = model.symbols['parameters']
    par_values  = model.calibration['parameters']
    cal         = dict(zip(par_names,par_values))
    n           = len(var_names)
    #n_par      = len(par_names)
    eqs_labels  = model.eqLabels
    
    # Assign missing parameters to zero.
    for k in cal:
        val = cal[k]
        if np.isnan(val):
            cal[k] = 0
  
    x0 = np.copy(var_values)

    ### Objective function
    def mcp_fobj(x):
        global it
        it += 1
        f = func(x)
        return sign*np.sum(f**2)
    
    ### Objective function
    def fobj(x):
        global it
        it += 1
        loc = cal.copy()
        for i,v in enumerate(var_names):
            loc[v] = x[i]
        f = sign*eval(obj_func,{},loc)

        return f
    
    # Function and Jacobian
    def func_jac(x):
        global it
        it += 1
        I = np.eye(n)
        f ,jacob = fun(model=model,y=np.vstack((x,x,x)),params=par_values,order=1)
        jac = jacob[:,n:2*n]
        if bool(Il) and bool(Iu):
            a1 = np.array([upper[i]-x[i] if not np.isinf(upper[i]) else -x[i] for i in range(n)])
            b1 = -f
            b = map_func(a1,b1)
            a = np.array([x[i]-lower[i] if not np.isinf(lower[i]) else x[i] for i in range(n)])
            z = map_func_der(a1,b1)*I + map_func_der(b1,a1)*jac
            y = map_func_der(a,b)*I - map_func_der(b,a)*z
        elif bool(Il):
            a = np.array([x[i]-lower[i] if not np.isinf(lower[i]) else x[i] for i in range(n)])
            b = f
            y = map_func_der(a,b)*I + map_func_der(b,a)*jac
        elif bool(Iu):
            a = np.array([upper[i]-x[i] if not np.isinf(upper[i]) else -x[i] for i in range(n)])
            b = -f
            y = map_func_der(a,b)*I + map_func_der(b,a)*jac
        else:
            y = -jac
        return f,y
    
    # Function
    def func(x):
        f = fun(model=model,y=np.vstack((x,x,x)),params=par_values,order=0)
        return f
    
    # Mapping function
    def map_func(a,b):
        return np.sqrt(a*a+b*b)-a-b
     
    # Mapping function derivatives
    def map_func_der(a,b):
        return a/np.sqrt(a*a+b*b)-1
         
    # MCP Squared Function
    def path_scalar_func(x):
        f = path_func(x)
        f2 = np.sum(f*f)
        if np.isnan(f2):
            return 1.e10
        else:
            return f2
        
    # MCP Function
    def path_func(x):
        f = func(x)
        if bool(Il) and bool(Iu):
            a1 = np.array([upper[i]-x[i] if not np.isinf(upper[i]) else -x[i] for i in range(n)])
            b1 = -f
            b = map_func(a1,b1)
            a = np.array([x[i]-lower[i] if not np.isinf(lower[i]) else x[i] for i in range(n)])
            y = map_func(a,b)
        elif bool(Il):
            a = np.array([x[i]-lower[i] if not np.isinf(lower[i]) else x[i] for i in range(n)])
            b = f
            y = map_func(a,b)
        elif bool(Iu):
            a = np.array([upper[i]-x[i] if not np.isinf(upper[i]) else -x[i] for i in range(n)])
            b = -f
            y = -map_func(a,b)
        else:
            y = -f
        return y
        
    # MCP Jacobian
    def path_jacob(x):
        f,jac = func_jac(x)
        return jac
    
    # Jacobian matrix
    def jacob(x):
        f,jac = FuncJacob(x)
        return jac
            
    # Jacobian matrix
    def FuncJacob(x):
        f,jac = fun(model=model,y=np.vstack((x,x,x)),params=par_values,order=1)
        return f,jac[:,n:2*n]
    
    # Hessian matrix
    def hess(x,v):
        _,_,hessian = fun(model=model,y=np.vstack((x,x,x)),params=par_values,order=2)
        #print(hessian)
        y = 0
        for i in range(len(v)):
            y += v[i] * hessian[i][n:2*n,n:2*n]
        return y
    

    # Get variables constraints
    Il,Iu,lower,upper = getLimits(var_names,constraints,cal)
    
    for i in range(n):
        x0[i] = min(upper[i],max(lower[i],x0[i]))
        
    # Get bounds
    bounds = Bounds(lower,upper)
    
    if not method is None and (method.lower() == "minimize" or method.lower() == "maximize"):
        if model.isLinear:
            #Get jacobian.  It is constant for linear problems.
            jacobian = jacob(x=x0)
            # Build linear constraints matrix
            A,lb,ub = getConstraints(n,constraints,cal,eqs_labels,jacobian)
            constraint = LinearConstraint(A,lb,ub)
        else:
            ind = [True]*n
            for i in range(n):
                if upper[i] == np.inf and lower[i] == -np.inf:
                    ind[i] = False
            Lower = lower[ind]; Upper = upper[ind]
            if model.order == 2:
                constraint = NonlinearConstraint(func,Lower,Upper,jac=jacob,hess=hess)
            elif model.order == 1:
                constraint = NonlinearConstraint(func,Lower,Upper,jac=jacob)
            elif model.order == 0:
                constraint = NonlinearConstraint(func,Lower,Upper)
        
        if solver in ['trust-constr','SLSQP']:            
            results = minimize(fun=fobj,x0=x0,method=solver,bounds=bounds,constraints=constraint,tol=1.e-10,options={'disp':False,'maxiter':MAXITER})
        else:
            cprint("\nOnly 'trust-constr' and 'SLSQP' methods are implemented...","red")
            cprint("'trust-constr' method.","red")
            results = minimize(fun=fobj,x0=x0,method="trust-constr",bounds=bounds,constraints=constraint,tol=1.e-10,options={'disp':False,'maxiter':MAXITER})
        
        if not results.success:
            cprint("\nConstrained minimization failed.","red")
            cprint("Running un-constrained minimization...","red")
            results = minimize(fun=fobj,x0=x0,method='Powell',bounds=None,options={'disp':False,'maxiter':1000})
        
    else:
        if solver in ['trust-constr','SLSQP']:            
            results = minimize(fun=fobj,x0=x0,method=solver,bounds=bounds,constraints=constraint,tol=1.e-10,options={'disp':False,'maxiter':MAXITER})
            
        elif solver in ['MCP','CONSTRAINED_OPTIMIZATION','PATH','ROOT']:
            cprint(f"\n{solver} solver","green")
            # if solver == 'PATH':
            #     x = x0
            #     f = np.zeros(n) 
            #     os.environ['PATH_LICENSE_STRING'] = '2830898829&Courtesy&&&USR&45321&5_1_2021&1000&PATH&GEN&31_12_2025&0_0_0&6000&0_0'
            #     # Load path library
            #     libc = loadLibrary()
            #     # Call path solver C++ function
            #     status = libc.path_solver(n,n_par,x,par_values,f,lower,upper)
            #     print_path_solution_status(status)
            #     results = Data(status,x,np.sqrt(np.sum(f*f)),"Success")
                
            if solver == 'CONSTRAINED_OPTIMIZATION':
                results = minimize(fun=path_scalar_func,x0=x0,method="trust-constr",bounds=bounds,tol=1.e-8,options={'disp':False,'maxiter':MAXITER})
            
            elif solver == 'MCP': 
                from compecon import MCP 
                try:
                    F = MCP(f=func_jac,a=lower,b=upper,x0=x0,maxit=1500)
                    x = F.zero(transform='minmax')
                    f = func_jac(x)[0]
                    results = Data(success=True,x=x,fun=np.sqrt(np.sum(f*f)),nfev=it)
                except np.linalg.LinAlgError:
                    cprint("\nCompecon MCP solver failed.","red")
                    cprint("Running constrained minimization solver...\n","red")
                    results = minimize(fun=path_scalar_func,x0=x0,method="trust-constr",bounds=bounds,tol=1.e-6,options={'disp':False,'maxiter':MAXITER})

            elif solver == 'ROOT':
                results = root(fun=path_func,jac=path_jacob,x0=x0,method="lm",tol=1.e-8,options={'disp':False,'maxiter':MAXITER})
            
            else:
                import sys
                cprint(f"\n{solver} is not implemented. Exitting...\n","red")
                sys.exit()
                
        else:
            cprint("\nOnly 'MCP', 'CONSTRAINED_OPTIMIZATION', 'ROOT' and 'PATH' solvers are implemented...","red")
            cprint("'CONSTRAINED_OPTIMIZATION' solver.","red")
            results = minimize(fun=path_scalar_func,x0=x0,method="trust-constr",bounds=bounds,tol=1.e-8,options={'disp':False,'maxiter':MAXITER})
        
        if not results.success:
            cprint("\nConstrained solver failed.","red")
            cprint("Running un-constrained solver...\n","red")
            results = minimize(fun=path_scalar_func,x0=x0,method="trust-constr",tol=1.e-10,options={'disp':False,'maxiter':MAXITER})
         
    return results
            

def run(fpath=None,fout=None,Output=False,plot_variables=None,model_info=False):
    """
    Call main driver program.
    
    Runs model simulations.
    
    Parameters:
        :param fpath: Path to model file.
        :type fpath: str.      
        :param fout: Path to output excel file.
        :type fout: str.
        :param Output: If True save results in excel file.
        :type Output: bool.
        :param plot_variables: Plot variables.
        :type plot_variables: list.
        :param model_info: If True creates a pdf/latex model file.
        :type model_info: bool.
        :returns: Optimization results.
    """
    global it
   
    model = importModel(fpath)
    if Output:
        print(model)
    
    var_names = model.symbols['variables']
    var_values = model.calibration['variables']
    par_names = model.symbols['parameters']
    par_values = model.calibration['parameters']
    # par = dict(zip(par_names,par_values))
    # co2_parameters = [x for x in par_names if x.startswith("PCARB")]
    shock_names = []; shock_values = []
    T = 1
    
    if model.symbolic.SOLVER == "PATH":
        from snowdrop.src.utils.util import create_config_file
        create_config_file(T,var_names,var_values,shock_names,shock_values,par_names,par_values,model.options)
    
    t0 = time()
    
    # Run baseline scenario
    results = solver(model=model)
    y       = results.x
    n       = len(y)
    var     = dict(zip(var_names,y))
    func    = results.fun
    nfev    = results.nfev
    status  = "success" if results.success else "failure"
    
    elapsed = time() - t0
    model.calibration['variables'] = y
    
    if bool(model.symbolic.objective_function):
        #cprint(f"\nObjective function:\n {model.symbolic.objective_function}","blue")
        cprint("\nObjective function value: {:.2e}".format(func),"green")
    
    cprint(f"Solution status: {status}","green")
    cprint(f"Number of function calls: {nfev}","green")
    cprint("Elapsed time: {:.2f} (seconds) \n".format(elapsed),"green")
    
    # Plot bar graphs
    # if bool(co2_parameters):
        # # Set CO2 price to zero
        # cal = model.calibration
        # for p in co2_parameters:
        #     cal[p] = 0
         
        # model.calibration = cal   
        # # Run scenrio 
        # results = solver(model=model)
        # var2 = dict(zip(var_names,model.calibration['parameters']))
        # arr  = [np.array([var[x],var2[x]]) for x in var_names]
        # var2 = dict(zip(var_names,arr))plot(var=var,var_names=var_names,par=par,par_names=par_names,fig_sizes=(8,6),title="Equivalent Variation")
        # plotBar(var=var2,var_names=var_names,par=par,par_names=par_names,yLabel="percent",fig_sizes=(8,6),title="Relative Impact on Welfare",relative=True)
        # plot(var=var,var_names=var_names,par=par,par_names=par_names,fig_sizes=(8,6),title="Equivalent Variation")

    
    if not plot_variables is None:
        bar_plot(var=var,var_names=plot_variables,plot_variables=True,symbols=model.symbols["variables_labels"],sizes=(3,3),fig_sizes=(10,8),title="Variables")
        
    if model_info:
        from snowdrop.src.misc.text2latex import saveDocument
        saveDocument(model)
        
    if Output:
        # Save results in excel file
        fdir = os.path.dirname(fpath)
        name, ext = os.path.splitext(fpath)
        if fout is None:
            fout = os.path.abspath(os.path.join(fdir,'../../../supplements/data/OPT/'+ os.path.basename(name) + '.csv'))
        with open(fout, 'w') as f:
            f.writelines(','.join(var_names) + '\n')
            f.writelines(','.join(str(x) for x in y)  +'\n')
            
        pTable = PrettyTable(['Var Name','Var Value','Var Name ','Var Value '])
        for i in range(n):
            if i+1 < n:
                row = [var_names[i], y[i], var_names[1+i], y[1+i]]
            else:
                row = [var_names[i], y[i], "", ""]
            pTable.add_row(row)
        pTable.float_format = "4.1"
        print(pTable)
        
    if not results.success:
        cprint(f"\nConstrained solver failed: {results.message}","red")
              
    return
 
