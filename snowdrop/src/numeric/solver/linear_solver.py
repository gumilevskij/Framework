""" 
The linear solver. It uses direct and iterative solvers that are 
applicable to equations with maximum of one lead and one lag variables.

.. note::
    This solver was designed to solve system of linear equations with shocks.
    To include exogenous processes in addition, please use Villemot solver.
""" 

from __future__ import division

import numpy as np
from time import time
import scipy.linalg as la
from snowdrop.src.model.settings import SolverMethod
from snowdrop.src.numeric.solver.util import getAllShocks
from snowdrop.src.numeric.solver.util import getParameters
from snowdrop.src.numeric.solver.Klein import solve_klein
from snowdrop.src.numeric.solver.Villemot import solve_villemot
from snowdrop.src.numeric.solver.AndersonMoore import solve_am
from snowdrop.src.numeric.solver.BinderPesaran import solve_quadratic_determinantal_equation

it = 0

def solve(model,p=None,steady_state=None,suppress_warnings=False):
    """
    Select solver.
    
    This is a convenience method to choose model equations solver.
    """
    global it
    if steady_state is None:
        n = len(model.symbols['variables'])
        steady_state = np.zeros(n)
        
    if not model.solved:
        if model.SOLVER.value == SolverMethod.Villemot.value:
            model = solve_villemot(model=model,steady_state=steady_state,p=p,suppress_warnings=suppress_warnings)
        elif model.SOLVER.value == SolverMethod.Klein.value:
            model = solve_klein(model=model,steady_state=steady_state,p=p,suppress_warnings=suppress_warnings)
        elif model.SOLVER.value == SolverMethod.AndersonMoore.value:
            model = solve_am(model=model,steady_state=steady_state,p=p,suppress_warnings=suppress_warnings)
        elif model.SOLVER.value == SolverMethod.BinderPesaran.value:
            model = solve_quadratic_determinantal_equation(model=model,steady_state=steady_state,p=p,suppress_warnings=suppress_warnings)
        else:           
            model = solve_villemot(model=model,steady_state=steady_state,p=p,suppress_warnings=suppress_warnings)
            # if it == 0:
            #     it += 1
            #     from snowdrop.src.misc.termcolor import cprint
            #     cprint("Linear models solver was not defined: using Binder and Pesaran solver ...","green")
            # model = solve_quadratic_determinantal_equation(model=model,steady_state=steady_state,p=p,suppress_warnings=suppress_warnings)
        model.solved = True
    return model
   
    
def simulate(model,T,periods,y0,steady_state,parameters=None,Npaths=1):
    """
    Find first-order accuracy approximation solution.
    
    For details on an algorithm to select anticipated and un-anticipated shocks to bring the level of
    endogenous variables to the desired path please see:
    https://michalandrle.weebly.com/uploads/1/3/9/2/13921270/iris_simulate.pdf
    
    Parameters:
        :param model: Model object.
        :type model: `Model'.
        :param T: Time span.
        :type T: int.
        :param periods: Simulation periods.
        :type periods: numpy.ndarray.
        :param y0: Starting values of endogenous variables.
        :type y0: numpy.ndarray
        :param steady_state: Steady state values of endogenous variables.
        :type steady_state: numpy.ndarray
        :param parameters: Values of parameters.
        :type parameters: list.
        :param Npaths: Number of simulation paths. This is the number of paths of stochastic shocks.
        :type Npaths: int.
        :returns: Numerical solution.
    """
    from snowdrop.src.numeric.solver.tunes import hasImaginaryShocks,hasImaginaryValues,forecast_with_tunes
            
    t0 = time()
    n = len(y0)
    M = None
                
    shock_var = model.symbols['shocks']
    n_shocks = len(shock_var)
    
    parameters = getParameters(parameters=parameters,model=model)
    all_shocks = getAllShocks(model,periods,n_shocks,Npaths,T)
    #print(all_shocks)
    
    # Solve linear model at steady state
    solve(model=model,p=parameters,steady_state=np.zeros(n))
    # State transition matrix
    F = np.real(model.linear_model["A"])
    # Array of constants
    C = np.real(model.linear_model["C"])
    # Matrix of coefficients of shocks
    R = np.real(model.linear_model["R"])
    U = np.real(model.linear_model.get("U",None))
    Z = np.real(model.linear_model["Z"])
      
    N = len(F)
    # Auxiliary matrices
    if model.SOLVER.value in [SolverMethod.Klein.value,SolverMethod.Villemot.value]:
        Xa = -model.linear_model["Xa"]
        Ru = model.linear_model["Ru"]
        J  = model.linear_model["J"]
        M  = [np.copy(R)]
        M.append(U @ Xa @ Ru)
        temp = J @ Ru
        for t in range(T+2):
            M.append(U @ Xa @ temp)
            temp = J @ temp
    elif model.SOLVER.value in [SolverMethod.BinderPesaran.value,SolverMethod.AndersonMoore.value]:
        M = [R]
        if not Z is None:
            temp = np.copy(R)
            for t in range(T+2):
                temp = Z @ temp
                M.append(temp)
       
        
    lst_exog = []
    # Compute contribution of exogenous terms
    exogenous = model.symbols.get("exogenous",[])
    
    if bool(exogenous) and not Z is None:
        from snowdrop.src.numeric.solver.util import getExogenousData
        from snowdrop.src.utils.equations import getExogMatrix
        
        eqs = model.symbolic.equations
        exog = model.symbols["exogenous"]
        W = getExogMatrix(eqs,exog)
        
        for t in range(T+2):
            exo = getExogenousData(model,1+t)
            x = Z @ W @ np.array(exo)
            lst_exog.append(x)

    ### Find solution.
    y = np.zeros((T+2,N))
    for path in range(Npaths):
        yIter = []
        shocks = np.array(all_shocks[path])
        temp = np.copy(y0)
        y[0] = temp
        
        ### Correct solution for a 'fixed path' of endogenous variables.
        imaginary_shocks = hasImaginaryShocks(shocks) 
        if bool(model.mapSwap):
            imaginary_values = hasImaginaryValues(model.mapSwap)
        else:
            imaginary_values = False
        
        if bool(model.mapSwap) or bool(model.condShocks) or imaginary_shocks or imaginary_values:  
            
            y,adjusted_shocks,model = forecast_with_tunes(model=model,Nperiods=T,y=y,T=F,Re=M,C=C,shocks=shocks,
                                          has_imaginary_shocks=imaginary_shocks,has_imaginary_values=imaginary_values)
        
        
            # Add padding to shocks since we removed shocks at t=0.  This is the starting values of shocks and can be disregarded.
            new_shocks = np.zeros((len(adjusted_shocks)+1,n_shocks))
            new_shocks[1:] = adjusted_shocks
            model.calibration["adjusted_shocks"] = new_shocks
    
        else:
            
            for t in range(T+1):
                # Solution for un-anticipated shock.
                y[t+1] = F @ y[t] + C + R @ shocks[t]
                if bool(lst_exog):
                    y[t+1] += lst_exog[t]
                # These are extra terms that arise when future shocks are anticipated.
                if model.anticipate:
                    for j in range(t+1,T+1):
                        y[t+1] += M[j-t] @ shocks[j]
            
        sol = y[:,:n]
        y = sol
    
    model.y = y;
    yIter.append(np.copy(y))    
    elapsed = time() - t0
    count = 1
    max_f = 0
    
    return (count,y,yIter,max_f,elapsed)

    
def find_steady_state(model,method="iterative",debug=False):
    """
    Find steady state solution.
     
    Parameters:
        :param model: Model object.
        :type model: `Model'.
        :param method: Find steady state solution by iterations, by minimizing square of equations error, etc...
        :type model: str.
        :returns: Array of endogenous variables steady states and their growth.
    """
    #from snowdrop.src.numeric.solver.util import getStableUnstableRowsColumns
    from snowdrop.src.numeric.solver.util import getStableUnstableRowsColumns
    
    var = model.symbols["variables"]
    n = len(var)
    
    I = np.eye(n)
    steady_state = np.zeros(n)
    growth = np.zeros(n)
    
    if not "A" in model.linear_model:
        solve(model)

    # State transition matrix
    T = model.linear_model["A"][:n,:n]
    # Array of constants
    K = model.linear_model["C"][:n]
    ev,ew = la.eig(T)
    nev = sum(abs(e) > 1 for e in ev)
        
    rowStable,colStable,rowUnstable,colUnstable = getStableUnstableRowsColumns(model)
    n_stable = len(colStable)
    
    if nev == 0:
        # Model is stationary...
        # So, apply direct method to find steady state solution.
        steady_state = la.pinv(I-T) @ K
        
    else:
        
        # stable = [True]*n
        # for i in range(n):
        #     for j in range(n):
        #         if  abs(ev[j]) > 1 and abs(ew[i,j] > 1.e-12):
        #             stable[i] = False 
        #             break
        
        if method=="iterative":
            z = zprev = np.zeros(n)
            for j in range(1000):
                zn = T @ z + K
                #Stop if growth of endogenous variables is constant.
                if abs(la.norm(zn-2*z+zprev)) < 1.e-15:
                    break
                zprev = np.copy(z)
                z = np.copy(zn)
                
            growth = z - zprev
            growth = [growth[i] if abs(growth[i])>1.e-10 else 0 for i in range(n)]
            steady_state = [zn[i] if growth[i]==0 else 0 for i in range(n)]
            steady_state = np.round(steady_state,10)
        
        elif method=="root":
            from scipy.optimize import root
            
            T_stable = T[np.ix_(rowStable,colStable)]
            
            # Define objective function.
            def fobj(x):
                y = np.zeros(n)
                y[rowStable] = x
                func = f_dynamic(np.concatenate([y,y,y,e]),p,order=0)
                return func[rowStable]
            
            # Define jacobian.
            def fjac(x):
                # Jacobian is constant for linear models.
                return T_stable
            
            f_dynamic = model.functions["f_dynamic"]
            x0 = np.zeros(len(colStable))
            p = getParameters(model=model)
            e = np.zeros(n)
            
            sol = root(fobj,x0,method='lm',tol=1e-12,options={"maxiter":1000})
            print(f"Number of function evaluations: {sol.nfev}")
            steady_state[rowStable] = -sol.x
            growth = (T-I)@steady_state + K
            
        else:
            T_stable = T[np.ix_(rowStable,colStable)]
            K_stable = K[rowStable]
            sol = la.pinv(np.eye(n_stable)-T_stable) @ K_stable
            steady_state[rowStable] = sol
            growth = (T-I)@steady_state + K
            
    # # Write the steady-state equations at two different times: t and t+d.
    # d    = 10
    # Ta   = T[np.ix_(rowStable,colStable)]
    # Ka   = K[rowStable]
    # tmp1 = np.concatenate((np.eye(n_stable), np.zeros((n_stable,n_stable))),axis=1)
    # tmp2 = np.concatenate((np.eye(n_stable), d*np.eye(n_stable)),axis=1)
    # E1   = np.concatenate((tmp1,tmp2),axis=0)
    # tmp1 = np.concatenate((Ta, -Ta),axis=1)
    # tmp2 = np.concatenate((Ta, (d-1)*Ta),axis=1)
    # E2   = np.concatenate((tmp1,tmp2),axis=0)
    # tmp  = np.concatenate((Ka,Ka),axis=0)
    # dx   = la.pinv(E1-E2) @ tmp
    # growth[rowUnstable]  = T[np.ix_(rowUnstable,colStable)] @ dx[:n_stable]
    
    
    if debug:
        if model.count == -1:
            print('Steady-State Solution:')
            sv = [var[i]+"="+str(round(steady_state[i],4)) for i in colStable if not "_minus_" in var[i] and not "_plus_" in var[i]]
            sv.sort()
            print(sv)
            print()
    
        for k,g,ss in zip(var,growth,steady_state):
            if k.endswith("_GAP"):
                print(f"{k}: \t{g:.2f} , \t{ss:.2f}")

    # Sanity check
    for i,v in enumerate(var):
        if "_minus_" in v:
            ind = v.index("_minus_")
            var_name = v[:ind]
        elif "_plus_" in v:
            ind = v.index("_plus_")
            var_name = v[:ind]
        else:
            var_name = None
        if var_name in var:
            k = var.index(var_name)
            steady_state[i] = steady_state[k]
                
    return steady_state, growth
    
