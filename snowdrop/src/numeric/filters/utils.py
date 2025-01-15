"""Compute initial error covariance matrices Pstar and Pinf.

The initial values of error covariance matrix P are assumed diffusive.
During the first d diffusive periods matrix P is decomposed into Pstar and Pinf
and separate equations for these matrices are applied.
After the d diffusive periods the standard Kalman filter is used and only one error covariance
matrix P is used.  The initial condition for P(d) is Pstar(d).
 
Translated from Dynare Matlab code to Python by A.G.
"""

import numpy as np
from scipy import linalg as la
#from warnings import warn 
itr = 0

def sorter(x):
    """Sort eigen values."""
    return abs(x)>=1
    
    
def compute_Pinf_Pstar(T,R,Q,N=0,n_shocks=-1,mf=None,qz_criterium=1.000001):
    """
    Compute of Pinf and Pstar for Durbin and Koopman diffuse filter.
    
    Kitagawa transformation of state space system with a quasi-triangular
    transition matrix with unit roots at the top, but excluding zero columns of the transition matrix.
    
    For references please see G. Kitagawa, "An algorithm for solving the matrix equation", 
    International Journal of Control, Volume 25, 1977 - Issue 5 
    
    The transformed state space is y = [ss z x], were:
        
        s: static variables (zero columns of T)
        
        z: unit roots
        
        x: stable roots
        
        ss: s - z = stationarized static variables

    Args:
      T:          double     
                  matrix of transition equations
      R:          double     
                  matrix of structural shock effects
      Q:          double     
                  matrix of covariance of structural shocks  
      N:          int        
                  shocks maximum lead number minus minimum lag number
      n_shocks:   int        
                  number pf shocks
      mf:         integer    
                  vector of indices of observed variables in state vector
      qz_criterium: double     
                  numerical criterium for unit roots

    Returns:
      Pinf:        double     
                   matrix of covariance initialization for nonstationary part
      Pstar:       double     
                   matrix of covariance of stationary part
      
    Algorithm:
      Real Schur transformation of transition equation
      
    .. note::
        Translated from Dynare version 4.5.1 to Python by AG
    
    """   
    fn,_ = R.shape 
    nplus,_ =  T.shape              
                       
    ST,QT,sdim = la.schur(T,output='real',sort = lambda x: abs(x) > 2-qz_criterium)
    #ST,QT = la.schur(T,output='complex')
        
    # Check correctness of Schur decomposition
    err1 = la.norm(QT@ST@QT.T - T) / la.norm(T)
    if err1 > 1.e-10:
        raise Exception("compute_Pinf_Pstar: Schur decomposition error.  \n Inconsistency of T matrix decomposition of {0}".format(round(err1,4)))
         
    # Re-arrange matrices to make unstable generalized eigenvalues appear in the upper left corner of matrices: s,t
    # from snowdrop.src.utils.sortSchur import sort_schur_decomposition
    # QT,ST,ap = sort_schur_decomposition(Q=QT,R=ST,z=2-qz_criterium,b=0)
    
    # Check correctness of Schur decomposition
    # err2 = la.norm(QT@ST@QT.T - T) / la.norm(T)
    # if err2 > 1.e-10:
    #     raise Exception("compute_Pinf_Pstar: Schur decomposition error.  \n Inconsistency of T matrix decomposition of {0}".format(round(err2,4)))
         
    nk = sum(abs(np.diag(ST))>2-qz_criterium)
    nk1 = nk+1
    
    Pstar = np.zeros((nplus,nplus))
    if n_shocks == -1:
        R1 = QT.T @ R
        B  = np.real(R1 @ Q @ R1.T)
    else:
        B = 0
        for i in range(1+N):
            R1 = QT.T @ R[:,i*n_shocks:(1+i)*n_shocks]
            B += np.real(R1 @ Q @ R1.T)
    i  = nplus-1
    
    while i >= nk1:
        if ST[i,i-1] == 0:
            if i == nplus-1:
                c = np.zeros(nplus-nk)
            else:
                c = ST[nk:i+1,:]@(Pstar[:,i:]@ST[i,i:].T) \
                  + ST[i,i]*ST[nk:i+1,i:]@Pstar[i:,i]
            
            qq = np.eye(i+1-nk)-ST[nk:i+1,nk:i+1]*ST[i,i]
            Pstar[nk:i+1,i] = la.solve(qq, B[nk:i+1,i]+c)
            Pstar[i,nk:i]   = Pstar[nk:i,i].T
            i -= 1
        else:
            if i == nplus-1:
                c = np.zeros(nplus-nk)
                c1 = np.zeros(nplus-nk)
            else:
                c = ST[nk:i+1,:]@(Pstar[:,i:]@ST[i,i:].T) \
                  + ST[i,i]*ST[nk:i+1,i:]@Pstar[i:,i] \
                  + ST[i,i-1]*ST[nk:i+1,i:]@Pstar[i:,i-1]
                c1 = ST[nk:i+1,:]@(Pstar[:,i:]@ST[i-1,i:].T) \
                   + ST[i-1,i-1]*ST[nk:i+1,i:]@Pstar[i:,i-1] \
                   + ST[i-1,i]*ST[nk:i+1,i:]@Pstar[i:,i]
            
            t11 = np.eye(i-nk+1)-ST[nk:i+1,nk:i+1]*ST[i,i]
            t12 = -ST[nk:i+1,nk:i+1]*ST[i,i-1]
            t1  = np.concatenate((t11,t12),axis=1)
            t21 = -ST[nk:i+1,nk:i+1]*ST[i-1,i]
            t22 = np.eye(i-nk+1)-ST[nk:i+1,nk:i+1]*ST[i-1,i-1]
            t2  = np.concatenate((t21,t22),axis=1)
            qq  = np.concatenate((t1,t2), axis=0)
            t   = np.concatenate((B[nk:i+1,i]+c, B[nk:i+1,i-1]+c1),axis=0)
            z   = la.solve(qq, t)
            Pstar[nk:i+1,i]   = z[:i-nk+1]
            Pstar[nk:i+1,i-1] = z[i-nk+1:]
            Pstar[i,nk:i]     = Pstar[nk:i,i].T
            Pstar[i-1,nk:i-1] = Pstar[nk:i-1,i-1].T
            i -= 2
        
    if i == nk:
        c = ST[nk,:]@(Pstar[:,nk1:]@ST[nk,nk1:].T)+ST[nk,nk]*ST[nk,nk1:]@Pstar[nk1:,nk]
        Pstar[nk,nk]=(B[nk,nk]+c)/(1-ST[nk,nk]*ST[nk,nk])
    
		# stochastic trends with no influence on observed variables are arbitrarily initialized to zero
    Pinf = np.zeros((nplus,nplus))
    Pinf[:nk,:nk] = np.eye(nk)
    if not mf is None:
        for k in range(nk):
            if la.norm(QT[mf,:]@ST[:,k]) < 1e-8:
                Pinf[k,k] = 0
                
    P1inf   = QT@Pinf@QT.T
    P1star  = QT@Pstar@QT.T
    
    return P1inf,P1star


def getTimeShift(model):
    """Get time shift between start of simulation and start of filtration."""   
    import pandas as pd
    from pandas.tseries.offsets import DateOffset
    import datetime as dt
    from snowdrop.src.utils.util import getDate

    n = -1
    simulation_range = model.options['range']
    start = getDate(simulation_range[0])
    filter_range = model.options['filter_range']
    end = getDate(filter_range[0])
    frequency = model.options['frequency']

    if start == end:
        # Starting simulation and filtration ranges are the same... Skip
        n = 0
    else:
        if frequency == 0:
            freq = DateOffset(months=12)
        elif frequency == 1:
            freq = DateOffset(months=3)
        elif frequency == 2:
            freq = DateOffset(months=1)
        elif frequency == 3:
            freq = DateOffset(weeks=1)
        elif frequency == 4:
            freq = DateOffset(days=1)
            
        rng = pd.date_range(start,end,freq=freq)
        n = len(rng)
        
    return n-1
        

def getSteadyStateCovarianceMatrix(T,R,Qm,Hm,Z,n,Nd,n_shocks):
    r"""Solve Lyapunov equation for stable part of error covariance matrix.
    
    Predict:
    
    .. math::
       P_{t|t-1} = T_{t} * P_{t-1|t-1} * F'_{t} + Q_{t}
   
    For details, see https://en.wikipedia.org/wiki/Kalman_filter

    Args:
        T: is the state-transition matrix,
        
        R: is the shock matrix,
        
        Qm: is the covariance matrix of state variables (endogenous variables),
        
        Hm: is the covariance matrix of space variables (measurement variables)
        
        Z: is the observation matrix,  
        
        n: is the number of endogenous variables,
        
        Nd: is the shocks maximum lead number minus minimum lag number,
        
        n_shocks: is the number of shocks, 
        
        iterate: is the boolean variable. If True then iterative method is used, otherwise Lyapunov equation solver is applied.
    
    For details, see https://en.wikipedia.org/wiki/Kalman_filter
    """
    from scipy import linalg as la 

    ITER = 1000
    EPS  = 1.e-4
    n = len(T)
    I = np.eye(n)
    Q = np.real(R @ Qm @ R.T)
    P = np.copy(Q)
    prev = np.copy(P)
    
    for i in range(ITER):
        P = T @ P @ T.T + Q                         # predicted error covariance 
        F = Hm + Z @ P @ Z.T                        # pre-fit residual covariance
        iF = la.pinv(F)                             # matrix inverse
        K = P @ Z.T @ iF                            # optimal Kalman gain
        P = (I-K@Z)@P                               # update (a posteriori) estimate covariance. This formula is valid for normal distribution.
        err = la.norm(P-prev)/la.norm(P)
        prev = np.copy(P)
        if err < EPS:
            print(f"Iter = {i}, error = {err:.3e}")
            break

    return P
            
def getMissingInitialVariables(model,ind,x0):
    """
    Return initial values of endogenous variables satisfying model equations.
    Parameters:
    :param model: Model object.
    :type model: `Model'.
    :param ind: Indices of variables with fixed values.
    :type ind: list.
    :param x0: Initial guess.
    :type x0: numpy ndarray.
    """
    from scipy.optimize import root
    from snowdrop.src.numeric.solver.util import getParameters
    from snowdrop.src.numeric.solver.util import getExogenousData
    from snowdrop.src.misc.termcolor import cprint
    
    #global TOLERANCE, NITERATIONS
    TOLERANCE = 1.e-6; NITERATIONS = 1000
    f_static = model.functions["f_steady"]
    bHasAttrStatic  = hasattr(f_static,"py_func")
    
    # Define objective function
    def fobj(x):
        global itr
        itr += 1
        x[ind] = z[ind]
        try:
            y = np.concatenate([x,e])
            if bHasAttrStatic:
                func = f_static.py_func(y,p,exog)
            else:
                func = f_static(y,p,exog)
        except ValueError:
            func = np.zeros(len(x)) + 1.e10 + itr
        return func
    
    p = getParameters(model=model)
    exog = getExogenousData(model,t=0)
    n_shk = len(model.symbols['shocks'])
    e = np.zeros(n_shk)
    z = np.copy(x0)
    
    sol = root(fobj,x0=z,method='lm',tol=TOLERANCE,options={"maxiter":NITERATIONS})
    x = sol.x    
    if not sol.success:
        ind_min = sol.fun.argmin()
        ind_max = sol.fun.argmax()
        ind = ind_min if abs(sol.fun[ind_min]) > abs(sol.fun[ind_max]) else ind_max
        err = la.norm(sol.fun)
        cprint(f"filters.itils.getInitialVariables:\n   Root solver failed: Number of iterations {itr}, Error {err:.3e}","yellow")
        cprint(f"   The following equation has the largest residual of {sol.fun[ind]:.2e},","yellow")
        cprint("      "+model.symbolic.equations[ind],"yellow")
        print()
    return x    
    
def getCovarianceMatrix(Qm,shocks,equations,n):
    """
    Return error covariance matrix of endogenous variables.

    Parameters
    ----------
    Qm : numpy 2D ndarray.
        Error covariance matrix.
    shocks : list.
        Shocks names.
    equations : list.
        Measuremen equations.
    n : int.
        Nimber of variables.

    Returns
    -------
    Q : numpy 2D array.
        Full error covariance matrix of endogenous variables.
    """
    import re
    
    delimiters = " ", ",", ";", "*", "/", ":", "+", "-", "^", "{", "}", "(", ")", "="
    regexPattern = '|'.join(map(re.escape, delimiters))
    
    Q = 1.e-10*np.eye(n)
    for i,eq in enumerate(equations):
        arr = re.split(regexPattern,eq)
        arr = list(filter(None,arr))
        for v in arr:
            if v in shocks:
                ind = shocks.index(v)
                Q[i,i] = Qm[ind,ind]
                
    return Q


def getEndogVarInMeasEqs(variables, measurement_equations):
    """
    Retrun list of endogenous variables that are present in measurement equations.

    Parameters
    ----------
    variables : list.
        Names of variables.
    measurement_equations : list.
        Measurement equations.

    Returns
    -------
    ind_var : list.
        Indices of endogenous variables.
    var_meas : list.
        Names of endogenous variables.
    """
    import re    

    # Get list of endogenous variables present in measurement equations
    delimiters = " ", ",", ";", "*", "/", ":", "+", "-", "^", "{", "}", "(", ")", "="
    regexPattern = '|'.join(map(re.escape, delimiters))
    var_meas = []
    for eq in measurement_equations:
        arr = re.split(regexPattern,eq)
        arr = list(filter(None,arr))
        for v in arr:
            if v in variables and not v in var_meas:
                var_meas.append(v)
    ind_var = [i for i,v in enumerate(variables) if v in var_meas]
    return ind_var,var_meas
    
    
if __name__ == "__main__":
    """Main entry point."""
    from mat4py import loadmat
    import os
    
    path = os.path.dirname(os.path.abspath(__file__))
    working_dir = os.path.abspath(os.path.join(os.path.abspath(path + "../../../..")))
    
    data = loadmat(working_dir + "/data/toy/data.mat")
    mf = np.array(data['mf'])
    mf = np.squeeze(mf) - 1
    T = np.array(data['T'])
    R = np.array(data['R'])
    Q = np.array(data['Q'])
    Pstar = np.array(data['Pstar'])
    Pinf = np.array(data['Pinf'])
    
    P1inf,P1star = compute_Pinf_Pstar(mf=mf,T=T,R=R,Q=Q)

    
    # Test of P1star, P1inf calculations
    #mf = [1, 2]
    # T = np.array( [[ 0, 0, 0, 0.75, 0.9, 1, 0],
    #                 [0, 0, 0, 0.75, 0.9, 1, 0],
    #                 [0, 0, 0, 0.333, 0, 0, 1],
    #                 [0, 0, 0, 0.75, 0, 0, 0],
    #                 [0, 0, 0, 0, 0.9, 0, 0],
    #                 [0, 0, 0, 0, 0.9, 1, 0],
    #                 [0, 0, 0, 0.333, 0, 0, 1]])
    # R = np.array( [[ 1, 1, 1, 0, 0, 0],
    #                 [1, 1, 1, 0, 1, 0],
    #                 [0, 0, 0.444, 1.333, 0, 1],
    #                 [0, 0, 1, 0, 0, 0],
    #                 [0, 1, 0, 0, 0, 0],
    #                 [1, 1, 0, 0, 0, 0],
    #                 [0, 0, 0.444, 1.333, 0, 0]])
    # Q = np.array([[ 1, 0, 0, 0, 0, 0],
    #                 [0, 1, 0, 0, 0, 0],
    #                 [0, 0, 1, 0, 0, 0],
    #                 [0, 0, 0, 1, 0, 0],
    #                 [0, 0, 0, 0, 0, 0],
    #                 [0, 0, 0, 0, 0, 0]])
    # static_rows = [0]
    
                           
    print('Pinf, Pstar difference:')
    print(np.max(abs(P1star-Pstar)))
    print(np.max(abs(P1inf-Pinf)))
    # print()
    # print("P1star:")
    # print(np.round(P1star,4))             
    # print()             
    # print("P1inf:")
    # print(np.round(P1inf,4))
 
