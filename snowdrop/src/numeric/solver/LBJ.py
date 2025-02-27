# -*- coding: utf-8 -*-
"""
Created on Fri Nov 1, 2019

@author: A.Goumilevski

ALGORITHM
  Laffargue, Boucekkine, Juillard (LBJ)
  
  see Juillard (1996) Dynare: A program for the resolution and
  simulation of dynamic models with forward variables through the use
  of a relaxation algorithm. CEPREMAP. Couverture Orange. 9602.
  
"""
from __future__ import division

from numba import njit
#from numba import jit

try:
    import cupy as np
    import cupyx.scipy.linalg as la
    from cupyx.scipy.sparse import linalg as sla
    from cupyx.scipy.sparse import csc_matrix,identity
    bCupy = True
except:
    import numpy as np
    import scipy.linalg as la
    from scipy.sparse import linalg as sla
    from scipy.sparse import csc_matrix,identity
    bCupy = False

    
from snowdrop.src.model.settings import BoundaryConditions,BoundaryCondition
from snowdrop.src.preprocessor.function import get_function_and_jacobian
from snowdrop.src.misc.termcolor import cprint
from snowdrop.src.model.model import Model

count = 1

#@cuda.jit
#@jit(parallel=True,fastmath=True,nopython=True,target="cpu")
@njit
def dense_system_solver(model:Model,T:int,n:int,y:np.array,params:list=None,shocks:list=None)->np.array:
    """
    Solve nonlinear model by LBJ method for dense matrices.
    
    Solver employs Newton's method and LBJ solver to update the solution.
    It uses backward substitution method.
        
    Parameters:
        :param model: Model object.
        :type model: Model.
        :param T: Time span.
        :type T: int.
        :param n: Number of endogenous variables.
        :type n: int.
        :param y: Array of values of endogenous variables for current iteration.
        :type y: numpy.ndarray.
        :param params: Array of parameters.
        :type params: numpy.ndarray.
        :param shock: Array of shock values.
        :type shocks: numpy.ndarray.
        :returns: Numerical solution.
    """    
    global count
    if count == 1:
        count += 1
        cprint("LBJ solver","blue")
        cprint("Dense matrices algebra","blue")
        cprint("Using GPU cores\n" if bCupy else "Using CPU cores\n","blue")
        
    n_shk = len(model.symbols['shocks'])
    # Allocate memory for arrays
    I = np.eye(n,n)
    M = np.empty((T,n,n))
    d = np.empty((T,n))
    var = model.symbols["variables"]
    terminal_values = model.terminal_values
    # mv = dict(zip(var,model.calibration["variables"]))
    # mp = dict(zip(model.symbols["parameters"],model.calibration["parameters"]))
    
    nd = np.ndim(params)
    
    if shocks is None:
        shocks = np.zeros(n_shk)
        
    # Equation:  L(t)*dy(t-1) + C(t)*dy(t) + F(t)*dy(t+1) = - Fn(t)
    # Starting values
    t = 0
    shk = shocks[t]
    par = params if nd == 1 else params[:,t]
    Fn,jacob = get_function_and_jacobian(model=model,t=t,y=y[t:t+3],params=par,shock=shk)
    
    if bCupy:
        jacob = np.asarray(jacob)
        Fn = np.asarray(Fn)

    F = jacob[:,0:n]
    C = jacob[:,n:2*n]
    L = jacob[:,2*n:3*n]
    lu, piv = la.lu_factor(C)
    M[0] = la.lu_solve((lu, piv),F)
    d[0] = -la.lu_solve((lu, piv),Fn)
    # M[0] = la.solve(C,F)
    # d[0] = -la.solve(C,Fn)
    # Forward substitution
    # M(t) = (C(t)-L(t)*M(t-1))^-1 * F(t)
    # d(t) = -(C(t)-L(t)*M(t-1))^-1 * (Fn(t)+L(t)*d(t-1))
    for t in range(1,T):
        shk = shocks[t]
        par = params if nd == 1 else params[:,min(t,params.shape[1]-1)]
        
        Fn,jacob = get_function_and_jacobian(model=model,t=t,y=y[t:t+3],params=par,shock=shk)

        if bCupy:
            Fn = np.asarray(Fn)
            jacob = np.asarray(jacob)

        F = jacob[:,0:n]
        C = jacob[:,n:2*n]
        L = jacob[:,2*n:3*n]
        temp = C - L@M[t-1]
        lu, piv = la.lu_factor(temp)
        M[t] = la.lu_solve((lu, piv),F)
        d[t] = -la.lu_solve((lu, piv),Fn + L@d[t-1])
        # M[t] = la.solve(temp,F)
        # d[t] = -la.solve(temp,Fn + L@d[t-1])
                
    ### Terminal conditions
    dy = np.zeros((T+2,n))
    if terminal_values is None:             
        if bCupy:
            temp = I + M[T-1]
            lu, piv = la.lu_factor(temp)
            dy[T+1] = la.lu_solve((lu, piv),d[T-1])
        else:
            dy[T+1] = la.solve(I + M[T-1],d[T-1])

    else:
        ind2 = [var.index(k) for k in terminal_values if k in var]     
        ind1 = [k for k in range(len(var)) if not k in ind2]   
        dy[T+1][ind2] = 0   
        if bCupy:
            temp = I + M[T-1]
            lu, piv = la.lu_factor(temp[np.ix_(ind1,ind1)])
            dy[T+1][ind1] = la.lu_solve((lu, piv),d[T-1][ind1])
        else:
            dy[T+1][ind1] = la.solve((I+M[T-1])[np.ix_(ind1,ind1)],d[T-1][ind1])
        
        
    # Backward substitution
    for t in range(T-1,-1,-1):
        dy[t+1] = d[t] - M[t]@dy[t+2]
        
    # New solution
    if bCupy:
       y = np.asarray(y) + dy
       y = np.asnumpy(y)
    else:
       y += dy
    
    del M,F,C,L,Fn,I,jacob,d,dy
    
    return y
 
    
@njit
def sparse_system_solver(model:Model,T:int,n:int,y:np.array,params:list=None,shocks:list=[]) -> np.array:
    """
    Solve nonlinear model by LBJ method for sparse matrices.
    
    Solver employs Newton's method and LBJ solver to update the solution.
    It uses backward substitution method.
    
    Parameters:
        :param model: Model object.
        :type model: Model.
        :param T: Time span.
        :type T: int.
        :param n: Number of endogenous variables.
        :type n: int.
        :param y: Array of values of endogenous variables for current iteration.
        :type y: numpy.ndarray.
        :param params: Array of parameters.
        :type params: numpy.ndarray.
        :param shock: Array of shock values.
        :type shocks: numpy.ndarray.
        :returns: Numerical solution.
    """        
    try:
        import pypardiso as slap
        bPardiso = True
    except:
        bPardiso = False
    from snowdrop.src.numeric.solver.util import getIndicesAndData

    global count
    if count == 1:
        count += 1
        cprint("LBJ solver","blue")
        cprint("Sparse matrices algebra","blue")
        cprint("Using GPU cores\n" if bCupy else "Using CPU cores\n","blue")
        
    nd = np.ndim(params)
    var = model.symbols["variables"]
    terminal_values = model.terminal_values
    
    # Allocate memory for arrays
    I = identity(n)
    M = np.empty((T,n,n))
    d = np.empty((T,n))
    
    # Equation:  L(t)*dy(t-1) + C(t)*dy(t) + F(t)*dy(t+1) = - Fn(t)
    # Starting values
    t = 0
    shk = shocks[t]
    par = params if nd == 1 else params[:,t]
    Fn,jacob,row_ind,col_ind = get_function_and_jacobian(model=model,t=t,y=y[t:t+3],params=par,shock=shk,bSparse=True)
    lead_row,lead_col,lead_data,current_row,current_col,current_data,lag_row,lag_col,lag_data = \
        getIndicesAndData(n,row_ind,col_ind,jacob)        

    if bCupy:
        lead_row_ind    = np.asarray(lead_row)
        lead_col_ind    = np.asarray(lead_col)
        current_row_ind = np.asarray(current_row)
        current_col_ind = np.asarray(current_col)
        lag_row_ind     = np.asarray(lag_row)
        lag_col_ind     = np.asarray(lag_col)
        lead_data       = np.asarray(lead_data)
        # current_data    = np.asarray(current_data)
    else:
        lead_row_ind    = lead_row
        lead_col_ind    = lead_col
        current_row_ind = current_row
        current_col_ind = current_col
        lag_row_ind     = lag_row
        lag_col_ind     = lag_col
        
       
    F = csc_matrix((lead_data,(lead_row_ind,lead_col_ind)),shape=(n,n))
    C = csc_matrix((current_data,(current_row_ind,current_col_ind)),shape=(n,n))    
    if bPardiso:
        M[0] = slap.spsolve(C,F.todense())
        d[0] = -slap.spsolve(C,Fn)
    else:
        B = sla.splu(C,permc_spec='COLAMD')
        M[0] = B.solve(F.todense())
        d[0] = -B.solve(Fn)
     
    # Forward substitution
    # M(t) = (C(t)-L(t)*M(t-1))^-1 * F(t)
    # d(t) = -(C(t)-L(t)*M(t-1))^-1 * (Fn(t)+L(t)*d(t-1))
    for t in range(1,T):
        shk = shocks[t]
        par = params if nd == 1 else params[:,min(t,params.shape[1]-1)]
        
        Fn,jacob,row_ind,col_ind = get_function_and_jacobian(model=model,t=t,y=y[t:t+3],params=par,shock=shk,bSparse=True)
        lead_row,lead_col,lead_data,current_row,current_col,current_data,lag_row,lag_col,lag_data = \
            getIndicesAndData(n,row_ind,col_ind,jacob)

        if bCupy:
            lead_data       = np.asarray(lead_data)
            current_data    = np.asarray(current_data)
            # lag_data        = np.asarray(lag_data)
                    
        F = csc_matrix((lead_data,(lead_row_ind,lead_col_ind)),shape=(n,n))
        C = csc_matrix((current_data,(current_row_ind,current_col_ind)),shape=(n,n))
        L = csc_matrix((lag_data,(lag_row_ind,lag_col_ind)),shape=(n,n))
        sparseMatrix = csc_matrix(C - L@M[t-1])
        if bPardiso:
            M[t] = slap.spsolve(sparseMatrix,F.todense())
            d[t] = -slap.spsolve(sparseMatrix,Fn + L@d[t-1])
        else:
            B = sla.splu(sparseMatrix,permc_spec='COLAMD')
            M[t] = B.solve(F.todense())
            d[t] = -B.solve(Fn + L@d[t-1])
                
    ### Terminal conditions
    dy = np.zeros((T+2,n))
    if terminal_values is None:    
        if BoundaryCondition.Condition.value ==  BoundaryConditions.ZeroDerivativeBoundaryCondition.value:
            dy[T+1] = la.solve(I + M[T-1],d[T-1])
    else:     
        ind2 = [var.index(k) for k in terminal_values if k in var]     
        ind1 = [k for k in range(len(var)) if not k in ind2]   
        dy[T+1][ind2] = 0   
        dy[T+1][ind1] = la.solve((I+M[T-1])[np.ix_(ind1,ind1)],d[T-1][ind1])
    
    # Backward substitution
    for t in range(T-1,-1,-1):
        dy[t+1] = d[t] - M[t]@dy[t+2]
        
    # New solution
    if bCupy:
       y = np.asarray(y)
       y = np.asnumpy(y + dy)
    else:
       y += dy
    
    del M,F,C,L,Fn,I,jacob,d,dy
    
    return y   


#@jit(parallel=True,fastmath=True,nopython=True,target="cpu")   
def LBJsolver(model,T,n,y,params,shocks):
    """
    Solve nonlinear model by LBJ method.
    
    Solver uses Newton's method and LBJ solver to update the solution.
    It applies backward substitution method.
    It differs from LBJsolver method by direction of endogenous variables substitution.
    
    Parameters:
        :param model: Model object.
        :type model: Model.
        :param T: Time span.
        :type T: int.
        :param n: Number of endogenous variables.
        :type n: int.
        :param y: array of endogenous variables for current iteration.
        :type y: numpy.ndarray.
        :param params: Values of parameters.
        :type params: numpy.ndarray.
        :param shocks: Array of shock values.
        :type shocks: numpy.ndarray.
        :returns: Numerical solution.
    """
    n_shk = len(model.symbols['shocks'])
    # Allocate memory for arrays
    L = np.empty((T,n,n))
    C = np.empty((T,n,n))
    F = np.empty((T,n,n))
    Fn = np.empty((T,n))
    M = np.empty((T,n,n))
    d = np.empty((T,n))
  
    if shocks is None:
        shocks = np.zeros(n_shk)
        
    R = np.empty((T,n,n_shk))
    nd = np.ndim(params)
    
    # Equation:  L(t)*dy(t-1) + C(t)*dy(t) + F(t)*dy(t+1) = - Fn(t)
    for t in range(1,T+1):    
        shk = shocks[t]
        par = params if nd == 1 else params[:,min(t,params.shape[1]-1)]
        
        func,jacob = get_function_and_jacobian(model=model,t=t,y=y[t:t+3,:],params=par,shock=shk)
        F[t-1] = jacob[:,0:n]
        C[t-1] = jacob[:,n:2*n]
        L[t-1] = jacob[:,2*n:3*n]
        R[t-1] = -jacob[:,3*n:] 
        Fn[t-1,:] =  func
      
    # Terminal values
    # M(T) = -C(T)^-1 * L(T)
    # d(T) = -C(T)^-1 * Fn(T)
    M[T-1] = -la.solve(C[T-1],L[T-1])
    d[T-1] = -la.solve(C[T-1],Fn[T-1])
    
    # Backward substitution
    # M(t) = -(C(t)+F(t)*M(t+1))^-1 * L(t)
    # d(t) = -(C(t)+F(t)*M(t+1))^-1 * (Fn(t)+F(t)*d(t+1))
    for t in range(T-2,-1,-1):
        temp = C[t]+F[t]@M[t+1]
        M[t] = -la.solve(temp,L[t])
        d[t] = -la.solve(temp,Fn[t]+F[t]@d[t+1])
        
    # Forward substitution
    dy = np.zeros((T+2,n))
    for t in range(1+T):
        tt = min(T-1,t)
        dy[t+1] = d[tt] + M[tt]@dy[t]
                             
    #if ZERO_DERIVATIVE_BOUNDARY_CONDITION:
    #    dy[T+1] = dy[T]
        
    # New solution
    y += dy 

    del M,F,C,L,Fn,jacob,d,dy
    
    return y

@njit
def solver_with_tunes(model:Model,start:int,T:int,n:int,y:np.array,params:list=None,shocks:list=None)->np.array:
    """
    Nonlinear solver with tunes.
    
    Solver employs Newton's method and LBJ solver to update the solution.
    It uses backward substitution method.
        
    Parameters:
        :param model: Model object.
        :type model: Model.
        :param T: Time span.
        :type T: int.
        :param n: Number of endogenous variables.
        :type n: int.
        :param y: Array of values of endogenous variables for current iteration.
        :type y: numpy.ndarray.
        :param params: Array of parameters.
        :type params: numpy.ndarray.
        :param shock: Array of shock values.
        :type shocks: numpy.ndarray.
        :returns: Numerical solution.
    """    
    global count
    if count == 1:
        count += 1
        cprint("LBJ solver","blue")
        cprint("Dense matrices algebra","blue")
        cprint("Using GPU cores\n" if bCupy else "Using CPU cores\n","blue")
        
    n_shk = len(model.symbols['shocks'])
    # Allocate memory for arrays
    I = np.eye(n,n)
    M = np.empty((T,n,n))
    d = np.empty((T,n))
    
    mapSwap = model.mapSwap
    rng  = sorted(model.mapSwap.keys())
    nd = np.ndim(params)
    
    if shocks is None:
        shocks = np.zeros(n_shk)
        
    # Equation:  L(t)*dy(t-1) + C(t)*dy(t) + F(t)*dy(t+1) = - Fn(t)
    # Starting values
    t = 0
    shk = shocks[min(t+start,T-1)]
    par = params if nd == 1 else params[:,t]
    y1 = y[t]; y2 = y[1+t]; y3 = y[2+t]
    if t+start in rng:
        m = mapSwap[t+start]
        for x in m:
            ind,_,val,_,_ = x
            y1[ind] = val
    if t+start+1 in rng:
        m = mapSwap[t+start+1]
        for x in m:
            ind,_,val,_,_ = x
            y2[ind] = val
    if t+start+2 in rng:
        m = mapSwap[t+start+2]
        for x in m:
            ind,_,val,_,_ = x
            y3[ind] = val
    yn = [y1,y2,y3]
        
    Fn,jacob = get_function_and_jacobian(model=model,t=t,y=yn,params=par,shock=shk)
    
    F = jacob[:,0:n]
    C = jacob[:,n:2*n]
    L = jacob[:,2*n:3*n]
    lu, piv = la.lu_factor(C)
    M[0] = la.lu_solve((lu, piv),F)
    d[0] = -la.lu_solve((lu, piv),Fn)
    # M[0] = la.solve(C,F)
    # d[0] = -la.solve(C,Fn)
    # Forward substitution
    # M(t) = (C(t)-L(t)*M(t-1))^-1 * F(t)
    # d(t) = -(C(t)-L(t)*M(t-1))^-1 * (Fn(t)+L(t)*d(t-1))
    for t in range(1,T):
        shk = shocks[t]
        par = params if nd == 1 else params[:,min(t,params.shape[1]-1)]
        y1 = y[t]; y2 = y[1+t]; y3 = y[2+t]
        if t+start in rng:
            m = mapSwap[t+start]
            for x in m:
                ind,_,val,_,_ = x
                y1[ind] = val
        if t+start+1 in rng:
            m = mapSwap[t+start+1]
            for x in m:
                ind,_,val,_,_ = x
                y2[ind] = val
        if t+start+2 in rng:
            m = mapSwap[t+start+2]
            for x in m:
                ind,_,val,_,_ = x
                y3[ind] = val
        yn = [y1,y2,y3]
            
        Fn,jacob = get_function_and_jacobian(model=model,t=t,y=yn,params=par,shock=shk)

        if bCupy:
            Fn = np.asarray(Fn)
            jacob = np.asarray(jacob)

        F = jacob[:,0:n]
        C = jacob[:,n:2*n]
        L = jacob[:,2*n:3*n]
        temp = C - L@M[t-1]
        lu, piv = la.lu_factor(temp)
        M[t] = la.lu_solve((lu, piv),F)
        d[t] = -la.lu_solve((lu, piv),Fn + L@d[t-1])
        # M[t] = la.solve(temp,F)
        # d[t] = -la.solve(temp,Fn + L@d[t-1])
                
    ### Terminal conditions
    dy = np.zeros((T+2,n))
    if BoundaryCondition.Condition.value ==  BoundaryConditions.ZeroDerivativeBoundaryCondition.value:
        dy[T+1] = la.solve(I + M[T-1],d[T-1])
       
    # Backward substitution
    for t in range(T-1,-1,-1):
        dy[t+1] = d[t] - M[t]@dy[t+2]
        
    # New solution
    y += dy
    
    # Correct again
    for t in range(T):
        if t+start in rng:
            m = mapSwap[t+start]
            for x in m:
                ind,_,val,_,_ = x
                y[t][ind] = val
    
    del M,F,C,L,Fn,I,jacob,d,dy
    
    return y
