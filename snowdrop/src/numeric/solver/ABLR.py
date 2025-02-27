# -*- coding: utf-8 -*-
"""
Created on Fri Nov 1, 2019

@author: agoumilevski
"""
from __future__ import division

from numba import njit
import numpy as np
import scipy.linalg as la
from snowdrop.src.model.settings import BoundaryConditions,BoundaryCondition
from snowdrop.src.preprocessor.function import get_function_and_jacobian
from snowdrop.src.misc.termcolor import cprint

try:
    import pypardiso as sla
    bPardiso = True
except:
    from scipy.sparse import linalg as sla
    bPardiso = False
    
count = 1

#@jit(parallel=True,fastmath=True,nopython=True,target="cpu")
@njit
def ABLRsolver(model,T,n,y,params=None,shocks=None):
    """
    Solve nonlinear model by direct method.
    
    Solver uses Newton's method to update the numerical solution.  
    It employs ABLR stacked matrices method to solve equations.  
    If number of equations greater is large, it uses sparse matrices spsolver algorithm.
    
    Parameters:
        :param model: Model object.
        :type model: Model.
        :param T: Time span.
        :type T: int.
        :param n: Number of endogenous variables.
        :type n: int.
        :param y: array of endogenous variables for current iteration.
        :type y: numpy.ndarray.
        :param params: array of parameters.
        :type params: numpy.ndarray.
        :param shocks: Array of shock values.
        :type shocks: numpy.ndarray.
        :returns: Numerical solution.
    """    
    global count
    bSparse = True  # Always use sparse algebra matrix calulations 
    if count == 1:
        count += 1
        cprint("ABLR solver","blue")
        if bSparse:
            cprint("Sparse matrices algebra","blue")
        else:
            cprint("Dense matrices algebra","blue")
        print()
    
    bDenseHasAttr = hasattr(dense_Fstack,"py_func")
    bSparseHasAttr = hasattr(sparse_Fstack,"py_func")
        
    n_shk = len(model.symbols['shocks'])
    n_shk *= 1+model.max_lead_shock-model.min_lag_shock
    if shocks is None:
        shocks = np.zeros(n_shk)
    
    if bSparse:
        if bSparseHasAttr:
            Fn,Jac = parallel_sparse_Fstack.py_func(model=model,T=T,y=y,n=n,params=params,shocks=shocks)
        else:
            Fn,Jac = parallel_sparse_Fstack(model=model,T=T,y=y,n=n,params=params,shocks=shocks)
    else:
        if bDenseHasAttr:
            Fn,Jac = dense_Fstack.py_func(model=model,T=T,y=y,n=n,params=params,shocks=shocks)
        else:
            Fn,Jac = dense_Fstack(model=model,T=T,y=y,n=n,params=params,shocks=shocks)
                                
    if bSparse:
        if bPardiso:
            dy = -sla.spsolve(Jac,Fn)
        else:
            #dy = -sla.spsolve(Jac,Fn)
            B = sla.splu(Jac,permc_spec='COLAMD')
            dy = -B.solve(Fn)
    else:
        dy = -la.solve(Jac,Fn)
              
    dy = dy.reshape((T+2,n))
    y += dy
    
    del Fn,Jac
    return y

@njit     
def dense_Fstack(model,T,y,n,params=None,shocks=None):
    """
    Compute values of functions and partial derivatives of model equations.
    
    Parameters:
        :param model: Model object.
        :type model: Model.
        :param T: Time span.
        :type T: int.
        :param y: array of endogenous variables.
        :type y: numpy.ndarray.
        :param n: Number of endogenous variables.
        :type n: int.
        :param params: array of parameters.
        :type params: numpy.ndarray.
        :param n_shocks: Number of shock variables.
        :type n_shocks: int.
        :returns: Stacked matrices of functions and jacobian values.
        
    """
    nd = np.ndim(params)
    I  = np.eye(n,n)
    
    Jacobian = np.zeros((n*(T+2),n*(T+2)))
    Fn = np.zeros((T+2,n))
    Jacobian[0:n,0:n] = I
    Jacobian[n*(T+1):n*(T+2),n*(T+1):n*(T+2)] =  I
        
    for t in range(1,T+1):
        shk = shocks[t-1]
        if nd == 1:
            par = params
        else:
            par = params[:,min(t,params.shape[1]-1)]
        func,jacob = get_function_and_jacobian(model=model,params=par,t=t,y=y[t-1:t+2,:],shock=shk)
        Fn[t,:] = func
        F = jacob[:,0:n]
        C = jacob[:,n:2*n]
        L = jacob[:,2*n:3*n]
        Jacobian[n*t:n*(t+1),n*(t-1):n*t] = L
        Jacobian[n*t:n*(t+1),n*t:n*(t+1)] = C
        Jacobian[n*t:n*(t+1),n*(t+1):n*(t+2)] = F

    if BoundaryCondition.Condition.value ==  BoundaryConditions.ZeroDerivativeBoundaryCondition.value:        
        Jacobian[n*(T+1):n*(T+2),:] = 0
        Jacobian[n*(T+1):n*(T+2),n*T:n*(T+1)] = -I
        Jacobian[n*(T+1):n*(T+2),n*(T+1):n*(T+2)] = I
        
    Fn = Fn.reshape(((T+2)*n,1))
 
    return Fn,Jacobian

@njit
def sparse_Fstack(model,T,y,n,params=None,shocks=None):
    """
    Compute values of functions and partial derivatives of model equations.
    
    Parameters:
        :param model: Model object.
        :type model: Model.
        :param T: Time span.
        :type T: int.
        :param y: array of endogenous variables.
        :type y: numpy.ndarray.
        :param n: Number of endogenous variables.
        :type n: int.
        :param params: array of parameters.
        :type params: numpy.ndarray.
        :param n_shocks: Number of shock variables.
        :type n_shocks: int.
        :returns: Stacked matrices of functions and jacobian.
        
    """
    from scipy.sparse import csc_matrix,lil_matrix,identity
    from snowdrop.src.numeric.solver.util import getIndicesAndData
    
    nd = np.ndim(params)
    I = identity(n)
    zero = csc_matrix(np.zeros(n*(T+2)))
    zero = zero.tolil()
    
    Jacobian = lil_matrix((n*(T+2),n*(T+2)))
    Fn = lil_matrix((T+2,n))
    Jacobian[0:n,0:n] = I
    Jacobian[n*(T+1):n*(T+2),n*(T+1):n*(T+2)] =  I
        
    for t in range(1,T+1):
        shk = shocks[t-1]
        if nd == 1:
            par = params
        else:
            par = params[:,min(t,params.shape[1]-1)]
            
        func,jacob,row_ind,col_ind = get_function_and_jacobian(model=model,t=t,y=y[t-1:t+2,:],params=par,shock=shk,bSparse=True)
        lead_row,lead_col,lead_data,current_row,current_col,current_data,lag_row,lag_col,lag_data = \
            getIndicesAndData(n,row_ind,col_ind,jacob) 
        
        Fn[t,:] = func
        F = csc_matrix((lead_data,(lead_row,lead_col)),shape=(n,n))
        C = csc_matrix((current_data,(current_row,current_col)),shape=(n,n))
        L = csc_matrix((lag_data,(lag_row,lag_col)),shape=(n,n))    
        Jacobian[n*t:n*(t+1),n*(t-1):n*t] = L
        Jacobian[n*t:n*(t+1),n*t:n*(t+1)] = C
        Jacobian[n*t:n*(t+1),n*(t+1):n*(t+2)] = F

    if BoundaryCondition.Condition.value ==  BoundaryConditions.ZeroDerivativeBoundaryCondition.value:        
        Jacobian[n*(T+1):n*(T+2),:] = zero
        Jacobian[n*(T+1):n*(T+2),n*T:n*(T+1)] = -I
        Jacobian[n*(T+1):n*(T+2),n*(T+1):n*(T+2)] = I
        
    Fn = Fn.todense().reshape(((T+2)*n,1))
    
    return Fn,Jacobian.tocsc()


@njit
def parallel_sparse_Fstack(model,T,y,n,params=None,shocks=None,use_thread_pool=True):
    """
    Compute values of functions and partial derivatives of model equations.
    
    Asynchronous run.
    
    Parameters:
        :param model: Model object.
        :type model: Model.
        :param T: Time span.
        :type T: int.
        :param y: array of endogenous variables.
        :type y: numpy.ndarray.
        :param n: Number of endogenous variables.
        :type n: int.
        :param params: array of parameters.
        :type params: numpy.ndarray.
        :param n_shocks: Number of shock variables.
        :type n_shocks: int.
        :returns: Stacked matrices of functions and jacobian.
        
    """    
    import multiprocessing as mp
    from concurrent.futures import as_completed
    from scipy.sparse import csc_matrix,lil_matrix,identity
    from snowdrop.src.numeric.solver.util import getIndicesAndData
    
    if use_thread_pool:
        max_workers = T+1
        from concurrent.futures import ThreadPoolExecutor as PoolExecutor 
    else:
        max_workers=mp.cpu_count()
        from concurrent.futures import ProcessPoolExecutor as PoolExecutor 

    nd = np.ndim(params)
    I = identity(n)
    zero = csc_matrix(np.zeros(n*(T+2)))
    zero = zero.tolil()
    
    Jacobian = lil_matrix((n*(T+2),n*(T+2)))
    Fn       = lil_matrix((T+2,n))
    Jacobian[0:n,0:n] = I
    Jacobian[n*(T+1):n*(T+2),n*(T+1):n*(T+2)] =  I
        
    lst = [(t,y[t-1:t+2,:],shocks[t-1],params if nd==1 else params[:,min(t,params.shape[1]-1)]) for t in range(1,T+1) ]
    
    def runJob(lst):
        t,x,shk,p = lst
        func,jacob,row_ind,col_ind = get_function_and_jacobian(model=model,t=t,y=x,params=p,shock=shk,bSparse=True)
        lead_row,lead_col,lead_data,current_row,current_col,current_data,lag_row,lag_col,lag_data = \
            getIndicesAndData(n,row_ind,col_ind,jacob) 
        
        Fn[t,:] = func
        Jacobian[n*t:n*(t+1),n*(t-1):n*t] = csc_matrix((lag_data,(lag_row,lag_col)),shape=(n,n)) 
        Jacobian[n*t:n*(t+1),n*t:n*(t+1)] = csc_matrix((current_data,(current_row,current_col)),shape=(n,n))
        Jacobian[n*t:n*(t+1),n*(t+1):n*(t+2)] = csc_matrix((lead_data,(lead_row,lead_col)),shape=(n,n))
        
        #return Fn,Jacobian
    
    # Run over time span
    with PoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(runJob,x): x for x in lst}
        for future in as_completed(futures):
            data = futures[future]
            try:
                results = future.result()
                #Fn,Jacobian = results
            except Exception as exc:
                print('%r Generated an exception: %s' % (data, exc))
            else:
                pass

    if BoundaryCondition.Condition.value ==  BoundaryConditions.ZeroDerivativeBoundaryCondition.value:        
        Jacobian[n*(T+1):n*(T+2),:] = zero
        Jacobian[n*(T+1):n*(T+2),n*T:n*(T+1)] = -I
        Jacobian[n*(T+1):n*(T+2),n*(T+1):n*(T+2)] = I
        
    Fn = Fn.todense().reshape(((T+2)*n,1))
    
    return Fn,Jacobian.tocsc()



            
   
