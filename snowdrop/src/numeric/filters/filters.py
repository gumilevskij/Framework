# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 18:49:21 2018

@author: agoumilevski
"""

import numpy as np
import scipy.linalg as la
import pandas as pd
import scipy as sp
from scipy.sparse import spdiags,lil_matrix,csr_matrix
from scipy.signal import fftconvolve
from scipy.sparse import linalg as sla
from scipy.signal import butter,lfilter
from snowdrop.src.misc.termcolor import cprint

oldK = 0

def LRXfilter(y, lmbd=1600, w1=1, w2=0, w3=0, dprior=None, prior=None):
    """
    Apply enchanced HP filter.
    
    This is LRX filter named after Laxton-Rose-Xie.
   
    A brief explanation of terms used in LRX filter
    
    Syntax: 
        y_eq = LRXfilter(y,lmbd[,w1[,w2[,w3[,DPRIOR[,PRIOR]]]]])
    
    Args:
        lmbd: float
            tightness constraint for HP filter,
        y:  array, dtype=float
            series to be detrended,
        w1: float
            weight on (y - y_eq)^2,
        w2: float
            weight on [(y_eq - y_eq(-1)) - dprior]^2,
        w3: float
            weight on (y_eq - prior)^2,
        dprior: array, dtype=float
            growth rate of y in steady-state,
        prior: array, dtype=float
            previously computed y_eq.
        
    Returns:
        y_eq: The trend component,
        
        gap: The cycle component
      
    .. note::
        the default value of lmbd is 1600,
        
        the default value of w1 is a vector of ones,
        
        the default value of w2 is a vector of zeros,
        
        the default value of w3 is a vector of zeros,
        
        the default value of dpr is a vector of zeros,
        
        the default value of pr is a vector of zeros,
        
        if any of w1, w2, w3, dpr, pr is of length 1, then it is extended to
        a vector of the appropriate length, in which all the entries are
        equal.
    """
    b = isinstance(y,pd.Series)
    if b:
        index = y.index
        y = y.values
    if isinstance(dprior,pd.Series):
        dprior = dprior.values
    if isinstance(prior,pd.Series):
        prior = prior.values
        
    n = len(y)
    w1 *= np.ones(n)
    
    if w2 == 0:
        w2 = np.zeros(n-1)
    else:
        w2 *= w2*np.ones(n-1)
    
    if w3 == 0:
        w3 = np.zeros(n)
    else:
        w3 *= np.ones(n)
    
    if dprior is None:
       dprior = np.zeros(n-1)
    elif np.isscalar(dprior):
       dprior *= np.ones(n-1)
    
    if  prior is None:
        prior = np.zeros(n)
    elif np.isscalar(prior):
        prior = np.ones(n)
        
    
    # This is the main part of the function
    I = np.eye(n)  
    B = np.diff(I,n=1,axis=0)
    A = np.diff(I,n=2,axis=0)
    # use sparse matrix to save memory and unnecessary computation
    MW1 = spdiags(w1, 0, n, n)
    MW2 = spdiags(w2, 0, n-1, n-1)
    MW3 = spdiags(w3, 0, n, n) 
    Y = y @ MW1 +  prior @ MW3 + dprior @ MW2 @ B
    if np.ndim(Y) > 1:
        Y = np.squeeze(Y)
    XX = lmbd * A.T @ A + B.T @ MW2 @ B + MW1 + MW3
    y_eq = la.solve(XX,Y)
    gap = y - y_eq
    
    if b:
        y = pd.Series(y_eq.T,index)
        gap = pd.Series(gap.T,index)
    else:
        y = y_eq.T
        gap = gap.T
    return y, gap
        

def HPF(data, lmbd=1600):
    """Hodrick-Prescott filter to a time series."""
    data = data.dropna()
    vals,_ = LRXfilter(data.values, lmbd=lmbd)
    filtered = pd.Series(vals,data.index)
    gap = data - filtered
    return filtered, gap
      

def HPfilter(data, lmbd=1600):
    """
    Apply a Hodrick-Prescott filter to a dataset.
    
    The return value is the filtered data-set found according to:
        
    .. math::
        
        \min_{T_{t}}[ \sum_{t=0}((X_{t} - T_{t})^2 + \lambda*((T_{t+1} - T_{t}) - (T_{t} - T_{t-1}))^2) ]


    Args:
        data: array, dtype=float
            The data set for which you want to apply the HP filter.
            This mustbe a numpy array.
        lmbd: array, dtype=float
            This is the value for lambda as used in the equation.

    Returns:
        T: array, dtype=float
            The solution to the minimization equation above (the trend).
        Cycle: array, dtype=float
            This is the 'stationary data' found by X - T.

    .. note::
        This function implements sparse methods to be efficient enough to handle
        very large data sets.
    """
    Y = np.asarray(data)

    if Y.ndim == 2:
        resp = [HPfilter(e) for e in data]
        T = np.row_stack( [e[0] for e in resp] )
        Cycle = np.row_stack( [e[1] for e in resp] )
        return [T,Cycle]

    elif Y.ndim > 2:
        raise Exception('HP filter is not defined for dimension >= 3.')

    lil_t = len(Y)
    big_Lambda = sp.sparse.eye(lil_t, lil_t)
    big_Lambda = lil_matrix(big_Lambda)

    # Use FOC's to build rows by group. The first and last rows are similar.
    # As are the second-second to last. Then all the ones in the middle...
    first_last = np.array([1 + lmbd, -2 * lmbd, lmbd])
    second = np.array([-2 * lmbd, (1 + 5 * lmbd), -4 * lmbd, lmbd])
    middle_stuff = np.array([lmbd, -4. * lmbd, 1 + 6 * lmbd, -4 * lmbd, lmbd])

    #--------------------------- Putting it together --------------------------#

    # First two rows
    big_Lambda[0, 0:3] = first_last
    big_Lambda[1, 0:4] = second

    # Last two rows. Second to last first : we have to reverse arrays
    big_Lambda[lil_t - 2, -4:] = second[::-1]
    big_Lambda[lil_t - 1, -3:] = first_last[::-1]

    # Middle rows
    for i in range(2, lil_t - 2):
        big_Lambda[i, i - 2:i + 3] = middle_stuff

    # spla.spsolve requires csr or csc matrix. I choose csr for fun.
    big_Lambda = csr_matrix(big_Lambda)

    T = sla.spsolve(big_Lambda, Y)

    Cycle = Y - T

    return T, Cycle
	

def bandpass_filter(data, w1, w2, k=None):
    """
    Apply a bandpass filter to data.
    
    This band-pass filter is of kth order.  It selects the band between w1 and w2.

    Args:
        data: array, dtype=float
            The data you wish to filter
        k: number, int
            The order of approximation for the filter. A max value for
            this is data.size/2
        w1: number, float
            This is the lower bound for which frequencies will pass
            through.
        w2: number, float
            This is the upper bound for which frequencies will pass
            through.

    Returns:
        y: array, dtype=float
            The filtered data.
    """
    if k is None:
        k = max(1,int(len(data)/2))
        
    data = np.asarray(data)
    low_w = np.pi * 2 / w2
    high_w = np.pi * 2 / w1
    bweights = np.zeros(2*k+1)
    bweights[k] = (high_w - low_w) / np.pi
    j = np.arange(1,k+1,1)
    weights = 1.0 / (np.pi * j) * (np.sin(high_w * j) - np.sin(low_w * j))
    bweights[k + j] = weights
    bweights[:k] = weights[::-1]

    bweights -= bweights.mean()
    filtered = fftconvolve(bweights, data, mode='same') #'full'
    
    return filtered[:-1]
 
    
def BPASS(data,W1,W2,drift=True,bChristianoFitzgerald=True,cutoff=False):
    """Bandpass filter to time series."""
    from scipy.fftpack import rfft, irfft, fftfreq
    from statsmodels.tsa.filters.cf_filter import cffilter

    data = data.dropna()
    time   = data.index
    signal = data.values
    
    if bChristianoFitzgerald:
        filtered_signal = cffilter(signal,low=W1,high=W2,drift=drift)[0]
    else:
        if cutoff:
            W = fftfreq(signal.size, 1./(time[1]-time[0]).days)
            f_signal = rfft(signal)    
            filtered_signal = np.copy(f_signal)
            filtered_signal[(np.abs(W)>1./W1)] = 0
            filtered_signal[(np.abs(W)<1./W2)] = 0
            filtered_signal = irfft(filtered_signal)
        else:
            delta = (time[1]-time[0]).days
            fs = 1/delta
            nyq = 365.0/2.0
            low = 365.0/W2/nyq
            high = 365.0/W1/nyq
            b,a = butter(N=6,Wn=[low,high]*fs,btype='bandpass')
            filtered_signal = lfilter(b,a,data)

    ts = pd.Series(filtered_signal,time)
    return ts
	

def Standard_Kalman_Filter(x, y, T, Z, P, Q, H, C=0, ind_non_missing=None, t = -1):
    r"""
    Implement Kalman filter algorithm for state-space model.

    This implementation assumes that measurement equation is specified at time t. 

    .. math:: 
            x_{t+1} = T_{t} * x_{t} + C_{t}  + w_{t},  w_{t} = Normal(0, Q),
            
            y_{t+1} = Z_{t} * x_{t+1} + v_{t},  v_{t} = Normal(0, H)
		
      
    KF uses two phase algorithm

    Predict
    
    .. math:: 
            x_{t|t} &= x_{t|t-1} + P_{t|t-1} * L_{t} * (y_{t} - Z_{t} * x_{t|t-1})
            
            P_{t|t} &= P_{t|t-1} - L * Z_{t} * P_{t|t-1}
            
            L{t} &= Z_{t} * [Z_{t}*P_{t|t-1}*Z'_{t} + H]^{-1}
		
    Update
    
    .. math:: 
    	   v_{t} &= y_{t} - Z_{t} * x_{t|t-1}
           
    	   x_{t+1|t} &= T_{t} * x_{t|t} + C_{t}
           
    	   P_{t+1|t} &= T_{t} * P_{t|t} * T'_{t} + Q_{t}


    Args:
        x : array, dtype=float
            is the state variables,
        y : array, dtype=float
            is the observation variables,
        T : 2D array, dtype=float
            is the state-transition matrix,
        Z : 2D array, dtype=float
            is the observation matrix,
        P : 2D array, dtype=float
            is the predicted error covariance matrix,
        Q : 2D array, dtype=float
            is the covariance matrix of state variables (endogenous variables),
        H : 2D array, dtype=float
            is the covariance matrix of space variables (measurement variables),
        C : array, dtype=float
            is the constant term matrix,
        ind_non_missing : array, dtype=int
            are the indices of non-missing observations.
  
    For details, please see https://stanford.edu/class/ee363/lectures/kf.pdf
    """
    n_meas = len(y)
    n_states = len(x)
    if ind_non_missing is None:
        cprint("Indeces of non-missing observations are not defined - resetting it.","red")
        ind_non_missing = np.arange(0,n_meas)

    iF = np.zeros((n_meas,n_meas))
    K  = np.zeros((n_states,n_meas))
    ZZ = Z[ind_non_missing]
        
    
    # Measurement pre-fit residual
    v = np.zeros(y.shape)
    v[ind_non_missing] = y[ind_non_missing] - ZZ @ (T @ x  + C)
       
    if len(ind_non_missing) == 0:
        
        log_likelihood = 0
        xn = x
        
    else:
            
        if True :
            F = H + Z @ P @ Z.T    # pre-fit residual covariance
            iF = la.pinv(F)        # matrix inverse 
            K[:,ind_non_missing]  = P @ ZZ.T @ iF[np.ix_(ind_non_missing,ind_non_missing)]  # Kalman gain
            
            # Prediction step
            x +=  K @ v            # predicted state estimate eq (4.26) DK(2012)
            P -= K @ ZZ @ P        # predicted error covariance
            
            
        else:   
            F = H + Z @ P @ Z.T    # pre-fit residual covariance
            iF = la.pinv(F)        # matrix inverse 
            M  = P @ Z.T
            PZ = M @ iF
            
            # Prediction step
            x +=  PZ @ v           # predicted state estimate eq (4.26) DK(2012)
            P -= M @ iF @ M.T      # predicted error covariance
            K  = T @ M             # Kalman gain
        
        # Update step 
        xn = T @ x  + C        # updated state estimate
        P  = T @ P @ T.T + Q   # updated error covariance
        P  = 0.5*(P+P.T)       # Force covariance matrix to be symmetric
            
        # Log-likelihood of the observed data
        log_likelihood = -0.5 * (n_meas*np.log(2*np.pi) + np.log(la.det(F)) + v.T @ iF @ v)
            
    return xn, v, P, K, iF, log_likelihood


def Kalman_Filter(x, y, T, Z, P, Q, H, C=0, ind_non_missing=None, bUnivariate=False, t=-1, tol=1.e-6):
    r"""
    Implement Kalman filter algorithm for state-space model.
 
    Args:
        x: array, dtype=float
           is the state variables,
        y: array, dtype=float
           is the observation variables,
        T: 2D array, dtype=float
           is the state-transition matrix,
        Z: 2D array, dtype=float
           is the observation matrix,
        P: 2D array, dtype=float
           is the predicted error covariance matrix,
        Q: 2D array, dtype=float
           is the covariance matrix of state variables (endogenous variables),
        H: 2D array, dtype=float
           is the covariance matrix of space variables (measurement variables),
        C: array, dtype=float
           is the constant term matrix,
        bUnivariate: bool
           if True univariate Kalman filter is used and if False - multivariate,
        ind_non_missing: array, dtype=int
           are the indices of non-missing observations,
        tol: number, dtype=float
           is the tolerance parameter (iteration over the Riccati equation)
      
    For details, see https://en.wikipedia.org/wiki/Kalman_filter
    """
    global oldK
    
    n_meas = len(y)
    n_states = len(x)
    I = np.eye(n_states)
    if ind_non_missing is None:
        cprint("Indeces of non-missing observations are not defined - resetting it.","red")
        ind_non_missing = np.arange(0,n_meas)
        
    log_likelihood = 0
        
    F  = np.zeros(shape=(n_meas))
    iF = np.zeros((n_meas,n_meas))
    K  = np.zeros((n_states,n_meas))
    ZZ = Z[ind_non_missing]
    
    # Prediction step
    P  = T @ P @ T.T + Q                        # predicted error covariance
    
    # Measurement pre-fit residual
    v = np.zeros(y.shape)
    v[ind_non_missing] = y[ind_non_missing] - ZZ @ x
    
    if len(ind_non_missing) > 0:
        # Update step
        F = H[np.ix_(ind_non_missing,ind_non_missing)] + ZZ @ P @ ZZ.T                    # pre-fit residual covariance
        iF[np.ix_(ind_non_missing,ind_non_missing)] = la.pinv(F)                          # matrix inverse
        K[:,ind_non_missing] = P @ ZZ.T @ iF[np.ix_(ind_non_missing,ind_non_missing)]     # optimal Kalman gain
        
        # Update (a posteriori) state estimate
        xn = T @ (x + K @ v)                                
        # Update (a posteriori) estimate covariance  
        P = (I-K@Z) @ P @ (I-K@Z).T + K @ H @ K.T   # This is a general case.
        # Force the covariance matrix to be symmetric
        P = 0.5*(P+P.T)
        
        steady = t > 0 and np.max(np.abs(K-oldK)) < tol
        oldK = np.copy(K)
    
        # Log-likelihood of the observed data
        log_likelihood = -0.5 * (n_meas*np.log(2*np.pi) + np.log(la.det(F)) + v.T @ iF @ v)
                        
    return xn, v, P, K, F, iF, log_likelihood, steady


def Kalman_Filter_SS(x, y, T, Z, K, F, iF, C=0, ind_non_missing=None):
    r"""
    Implement Kalman filter algorithm for stationary state-space model.

    Args:
        x: array, dtype=float
           is the state variables,
        y: array, dtype=float
           is the observation variables,
        T: 2D array, dtype=float
           is the state-transition matrix,
        Z: 2D array, dtype=float
           is the observation matrix,
        K: 2D array, dtype=float
           is the Kalman gain matrix,
        F: 2D array, dtype=float
           is the pre-fit residual covariance,
        iF: 2D array, dtype=float
           is the inverse of the pre-fit residual covariance,
        C: array, dtype=float
           is the constant term matrix,
        bUnivariate: bool
           if True univariate Kalman filter is used and if False - multivariate,
        ind_non_missing: array, dtype=int
           are the indices of non-missing observations,
        tol: number, dtype=float
           is the tolerance parameter
      
    For details, see https://en.wikipedia.org/wiki/Kalman_filter
    """
    n_meas = len(y)
    if ind_non_missing is None:
        cprint("Indeces of non-missing observations are not defined - resetting it.","red")
        ind_non_missing = np.arange(0,n_meas)
        
    log_likelihood = 0
        
    # Measurement pre-fit residual
    v = np.zeros(y.shape)
    v[ind_non_missing] = y[ind_non_missing] - Z @ x
    
    if len(ind_non_missing) > 0:
        # Update (a posteriori) state estimate
        xn =  T @ (x + K @ v)    
        # Log-likelihood of the observed data
        log_likelihood = -0.5 * (n_meas*np.log(2*np.pi) + np.log(la.det(F)) + v.T @ iF @ v)
                        
    return xn, v, log_likelihood
    

def fast_kalman_filter(Y,N,a,P,T,H,Z,kalman_tol=1.e-10):
    """Computes the likelihood of a stationary state space model, given initial condition for the states (mean and variance).
    
    References: 
        Edward P. Herbst, (2012). 'Using the ”Chandrasekhar Recursions” 
        for Likelihood Evaluation of DSGE Models''
    
    Args:
        Y:  array, dtype=float
            Measurement data,
        N:  scalar, int
            Last period,
        a:  array, dtype=float
            Initial mean of the state vector,
        P:  2D array, dtype=float 
            Initial covariance matrix of the state vector,
        T: 2D array, dtype=float
            Transition matrix of the state equation,
        H: 2D array, dtype=float
            Covariance matrix of the measurement errors (if no measurement errors set H as a zero scalar),
        Z: 2D array, dtype=float 
            Matrix relating the states to the observed variables or vector of indices,
        kalman_tol: scalar, float
            Tolerance parameter (rcond, inversibility of the covariance matrix of the prediction errors),
        
    """
    # Number of measurements
    n_meas = len(Y)

    # Initialize some variables.
    likk = np.zeros(N)      # Initialization of the vector gathering the densities.
    residuals = np.zeros(N)
    F_singular  = True


    K  = T @ P @ Z.T
    F  = Z @ P @ Z.T + H
    W  = np.copy(K)
    iF = la.inv(F)
    Kg = K @ iF
    M  = -iF

    for t in range(N) :
        v  = Y[t] - Z @ a
        residuals[t] = v
        dF = la.det(F)

        if np.linalg.cond(F,-2) < kalman_tol:
            if abs(dF)<kalman_tol:
                raise Exception("The univariate diffuse kalman filter should be used.")
            else:
                raise Exception("Pathological case, discard draw.")
        else:
            F_singular = False
            likk[t] = np.log(dF) + v.T @ iF @ v
            a       = T @ a + Kg @ v
            ZWM     = Z @ W @ M
            ZWMWp   = ZWM @ W.T
            M       = M + ZWM.T @ iF @ ZWM
            F       = F + ZWMWp @ Z.T
            iF      = la.inv(F)
            K       = K + T @ ZWMWp.T
            Kg      = K @ iF
            W       = (T - Kg @ Z) @ W

  
    if F_singular:
        raise Exception('The variance of the forecast error remains singular until the end of the sample')

    # Add observation's densities constants and divide by two.
    likk = -0.5 * (likk + n_meas*np.log(2*np.pi))

    # Compute the log-likelihood.
    if np.any(np.isnan(likk)):
        LIK = -1.e20
    else:
        LIK = np.sum(likk)

    return LIK,likk,residuals,a,P


def Rauch_Tung_Striebel_Smoother(Ytp,Xt,Xtp,Ft,Pt,Ptp,Ttp):
    r"""
    Implement Rauch-Tung-Striebel (RTS) smoother for state-space model.  
    
    RTS algorithm is
                
    .. math::
        
        L_{t} &= P_{t|t} * F'_{t} * P_{t+1|t}^{-1}
        
        X_{t|T} &= X_{t|t} + L_{t} * (X_{t+1|T} - X_{t+1|t})
        
        P_{t|T} &= P_{t+1|t} + L_{t} * (P_{t+1|T} - P_{t+1|t}) * L'_{t}

	
    Args:
        Ytp: array, dtype=float
             is the smoothed state variables at time t+1,
        Xt: array, dtype=float
            is the state variables at time t,
        Xtp: array, dtype=float
             is the state variables at time t+1,
        Ft: 2D array, dtype=float
            is the state-transition matrix at time t,
        Pt: 2D array, dtype=float
            is the predicted error covariance matrix at time t,
        Ptp: 2D array, dtype=float
             is the predicted error covariance matrix at time t+1,
        Ttp: 2D array, dtype=float
             is the smoothed covariance matrix of state variables.
      
    For details, see https://en.wikipedia.org/wiki/Kalman_filter
    """    
    L = Pt @ Ft.T @ la.pinv(Ptp)
    y =  Xt + L @ (Ytp - Xtp)
    P = Ptp + L @ (Ttp - Ptp) @ L.T
    
    return y, P


def Bryson_Frazier_Smoother(x,u,F,H,K,Sinv,P,L,l):
    r"""
    Implement modified Bryson-Frazier (BF) smoother for a state-space model.
    
    BF algorithm is:
                
    .. math::
        L_{t} &= H'_{t}*S_{t}^{-1}*H_{t} + (I-K_{t}*H_{t})'*L_{t}*(I-K_{t}*H_{t})
        
        L_{t-1} &= F'_{t}*Lt_{t}*F_{t}
        
        L_{T} &= 0
        
        l_{t} &= -H'_{t}*S_{t}^{-1}*u_{t} + (I-K_{t}*H_{t})'*l_{t}
        
        l_{t-1} &= F'_{t}*lt_{t}
        
        l_{T} &= 0                                                                   
        
    Smoothed variables and covariances
        
    .. math::
        xs_{t|T} &= x_{t} - P_{t}*l_{t}
        
        P_{t|T} &= P_{t} - P_{t}*L_{t}*P_{t}
        

    Args:
        x: array, dtype=float
           is the state variables,
        u: array, dtype=float
           is the vector of one-step-ahead prediction errors
        F: 2D array, dtype=float
           is the state-transition matrix,
        H: 2D array, dtype=float
           is the observation matrix,
        K: 2D array, dtype=float
           is the Kalman gain,
        Sinv: 2D array, dtype=float
           is the inverse of matrix S,
        P: 2D array, dtype=float
           is the predicted error covariance matrix,
        L: 2D array, dtype=float
           is the covariance matrix,
        l: 2D array, dtype=float
           is the temporary matrix
  
    For details, see https://en.wikipedia.org/wiki/Kalman_filter
    """   
    n  = len(x)
    C  = np.eye(n) - K @ H 
    Lt = H.T @ Sinv @ H + C.T @ L @ C               
    L  = F.T @ Lt @ F 
    lt = -H.T @ Sinv @ u + C.T @ l               
    l  = F.T @ lt 
    # Smoothed solution
    xs = x - P @ l
    # and covariance
    Ps = P - P.T @ L @ P
    
    return xs,Ps,L,l 


def DK_Filter(x, xtilde, y, v, T, Z, P, H, Q, K, C=0, ind_non_missing=None, t=-1):
    r"""
    Implement non-diffusive Durbin & Koopman version of Kalman filter algorithm for state-space model.
    
    Multivariate approach to Kalman filter.
    
    State-space model
    
    .. math::
        x_{t} = T_{t} * x_{t-1} + C_{t}  + w_{t},  w = Normal(0, Q)
        
        y_{t} = Z_{t} * x_{t} + v_{t},             v =  Normal(0, H)
		
 
    Durbin-Koopman algorithm

    .. math::			
		v_{t} &= y_{t} - Z_{t} * x_{t}
        
		x_{t} &= T_{t} * (x_{t-1} + K_{t} * v_{t}) + C_{t}
        
		F_{t} &= H_{t} + Z_{t} * P_{t-1} * Z'_{t}
        
		K_{t} &= P_{t} * Z'_{t} * F_{t}^{-1}
        
        L_{t} &= K_{t} * Z_{t} 
        
		P_{t+1} &= T_{t} * P_{t} *L'_{t-1} + Q_{t}		  
		
 
    Args:
        x: array, dtype=float
           is the variables at t+1|t+1 step,
        xtilde: array, dtype=float
           is the variables at t|t+1 forecast step,
        y: array, dtype=float
           is the observation variables,
        v: array, dtype=float
           is the residual,
        T: 2D array, dtype=float
           is the state-transition matrix,
        Z: 2D array, dtype=float
           is the observation matrix,
        P: 2D array, dtype=float
           is the predicted error covariance matrix,
        H: 2D array, dtype=float
           is the covariance matrix of measurement errors,
        Q: 2D array, dtype=float
           is the covariance matrix of structural errors,
        K: 2Darray, dtype=float
           is the Kalman gain,
        C: array, dtype=float
           is the constant term matrix,
        ind_non_missing: array, dtype=int
            are the indices of non-missing observations,
        tol1: number, float
            Kalman tolerance for reciprocal condition number,
        tol2: number, float
            diffuse Kalman tolerance for reciprocal condition number (for Finf) and the rank of Pinf
      
    For details, see Durbin & Koopman 2012, "Time series Analysis by State Space Methods" and
                 Koopman & Durbin 2000, "Fast Filtering and Smoothing For Multivariate State Space Models"
    """
    n_states = len(x)
    n_meas = len(y)
    iF = None  
            
    if ind_non_missing is None:
         cprint("Indeces of non-missing observations are not defined - resetting it.","red")
         ind_non_missing = np.arange(0,n_meas)
     
    ZZ = Z[ind_non_missing]
    v  = np.zeros(y.shape) 
    F  = np.zeros(shape=(n_meas))
    K  = np.zeros(shape=(n_states,n_meas))
    PZI = np.zeros(shape=(n_states,n_meas))
     
       
     ### --------------------------------------------------------- Standard Kalman Filter
    if t==-1 or len(ind_non_missing) == 0:
        xn  = T @ x  + C  
        P   = T @ P @ T.T + Q
        F   = H + Z @ P @ Z.T        # pre-fit residual covariance
        iF  = la.inv(F)
        K   = T @ P @ Z.T @ iF 
        v   = y - Z @ xn
        v[np.isnan(v)] = 0
        log_likelihood = 0
   
    else:
        ### Multivariate approach to Kalman filter
        iF = np.zeros(shape=(n_meas,n_meas))
        # measurement residual
        v[ind_non_missing] = y[ind_non_missing] - ZZ @ x
        ### Multivariate approach to Kalman Filter
        F = H[np.ix_(ind_non_missing,ind_non_missing)] + ZZ @ P @ ZZ.T      # pre-fit residual covariance
        iF[np.ix_(ind_non_missing,ind_non_missing)] = la.inv(F)             # matrix inverse  
        PZI = P @ ZZ.T @ iF[np.ix_(ind_non_missing,ind_non_missing)]
        K[:,ind_non_missing]  = T @ PZI                                     # Kalman gain
        xtilde = x + PZI @ v[ind_non_missing]                               # updated (a posteriori) state estimate eq (4.26) DK(2012)
        xn  = T @ xtilde  + C                                               # predicted state estimate
        L   = T - K[:,ind_non_missing] @ ZZ
        P   = T @ P @ L.T + Q
        # Log-likelihood of the observed data set
        log_likelihood = -0.5 * (n_meas*np.log(2*np.pi) + np.log(la.det(F)) + v @ iF @ v.T)
     
  
    return xn, xtilde, v, P, K, F, iF, log_likelihood
                                

def DK_Smoother(x,y,T,Z,F,iF,P,QRt,K,H,r,v,ind_non_missing=None,t=-1):
    r"""
    Implement non-diffusive Durbin-Koopman smoother for state-space model.
    
    Durbin-Koopman smoothing algorithm is
                
    .. math::
        r_{t-1} &= Z'_{t}*F_{t}^{-1}*v_{t} + L'_{t}**r_{t} 
        
        r_{T} &= 0 
        
        L_{t} &= I - K_{t}*Z_{t}                                                          
        
    Smoothed variables
        
    .. math::
        s_{t} = x_{t} + P_{t}*r_{t-1}
              
	
    Args:
        x: array, dtype=float
           is the updated state variables vector,
        y: array, dtype=float
           is the observation variables,
        T: 2D array, dtype=float
           is the state-transition matrix,
        Z: 2D array, dtype=float
           is the observation matrix,
        F: 2D array, dtype=float
           is the residual covariance matrix,
        iF: 2D array, dtype=float
           is the inverse of matrix F,
        P: 2D array, dtype=float
           is the updated error covariance matrix,
        QRt: 2D array, dtype=float
           is the product of the covariance matrix of structural errors and shock matrix
        K: 2D array, dtype=float
           is the Kalman gain matrix,
        H: 2D array, dtype=float
           is the covariance matrix of measurement errors,
        r: array, dtype=float
           is the vector of one-step-back errors,
        v: array, dtype=float
           is the vector of one-step-ahead prediction errors,
        ind_non_missing: array, dtype=int
           are the indices of non-missing observations.
        
    For details, see Durbin & Koopman (2012), "Time series Analysis by State Space Methods" and
    Koopman & Durbin (2000), "Fast Filtering and Smoothing For Multivariate State Space Models" and
    Koopman & Durbin (2003), in Journal of Time Series Analysis, vol. 24(1), pp. 85-98
    """ 
    n_meas = len(Z)
    res = np.zeros(n_meas)
    
    if ind_non_missing is None:
        cprint("Indeces of non-missing observations are not defined - resetting it.","red")
        ind_non_missing = np.arange(n_meas)
    
    etahat = [np.nan]*len(QRt)
    L = T - K @ Z
    # Smooth estimate
    if len(ind_non_missing) == 0:
        r = L.T @ r        
        etahat = QRt @ r  
        epsilonhat = H @ iF @ v
        xs = x + P @ r
    else:
        r = Z.T @ iF @ v + L.T @ r                          # eq (4.39) in DK(2012) 
        # Smooth estimation of state variables disturbance
        etahat = QRt @ r                                    # DK (2012)
        # Smooth estimation of observation disturbance
        epsilonhat =  H @ (iF @ v - K.T @ r) 
        # Smooth estimate
        xs = x + P @ r                                      # eq (4.39) in DK(2012)
        res = y - Z @ xs                                    # Smooth estimation of observation disturbance
 
    return xs,r,res,etahat,epsilonhat


def Durbin_Koopman_Non_Diffuse_Filter(x, xtilde, y, T, Z, P, H, Q, C=0, bUnivariate = False, ind_non_missing=None, tol1=1.e-6, tol2=1.e-6, t=-1):
    r"""
    Implement non-diffusive Durbin & Koopman version of Kalman filter algorithm for state-space model.
               
    State-space model:
    
    .. math::
        x_{t} = T_{t} * x_{t-1} + C_{t}  + w_{t},  w = Normal(0, Q)
        
        y_{t} = Z_{t} * x_{t} + v_{t},             v = Normal(0, H)
		
 
    Durbin-Koopman algorithm

    .. math::			
		v_{t} &= y_{t} - Z_{t} * x_{t}
        
		x_{t} &= T_{t} * (x_{t-1} + K_{t} * v_{t}) + C_{t}
        
		F_{t} &= H_{t} + Z_{t} * P_{t-1} * Z'_{t}
        
		K_{t} &= P_{t} * Z'_{t} * F_{t}^{-1}
        
        L_{t} &= K_{t} * Z_{t}
        
		P_{t+1} &= T_{t} * P_{t} *L'_{t-1} + Q_{t}		  
		

    Args:
        x: array, dtype=float
           is the variables at t+1|t+1 step,
        xtilde: array, dtype=float
           is the variables at t|t+1 forecast step,
        y: array, dtype=float
           is the observation variables,
        T: 2D array, dtype=float
           is the state-transition matrix,
        Z: 2D array, dtype=float
           is the observation matrix,
        P: 2D array, dtype=float
           is the predicted error covariance matrix,
        H: 2D array, dtype=float
           is the covariance matrix of measurement errors,
        Q: 2D array, dtype=float
           is the covariance matrix of structural errors,
        C: array, dtype=float
           is the constant term matrix,
        ind_non_missing: array, dtype=int
           are the indices of non-missing observations,
        tol1: number, float
           Kalman tolerance for reciprocal condition number,
        tol2: number, float
           diffuse Kalman tolerance for reciprocal condition number (for Finf) and the rank of Pinf
      
    For details, see Durbin & Koopman 2012, "Time series Analysis by State Space Methods" and
                 Koopman & Durbin 2000, "Fast Filtering and Smoothing For Multivariate State Space Models"
    """
    log_likelihood = 0
    n_meas         = len(y)
    n_states       = len(x)
    if ind_non_missing is None:
        cprint("Indeces of non-missing observations are not defined - resetting it.","red")
        ind_non_missing = np.arange(0,n_meas)
    
    ZZ = Z[ind_non_missing]
    v = np.zeros(y.shape) 
    
    F  = np.zeros(shape=(n_meas))
    K  = np.zeros(shape=(n_states,n_meas))
    iF,P1 = None,None
    
    #P = sp.linalg.tril(P) + np.transpose(sp.linalg.tril(P,-1))  
        
    # Check if matrices Finf, Fstar are singular
    bUnivariate |= np.linalg.cond(H+Z@P@Z.T,-2)<tol2  
                 
    ### --------------------------------------------------------- Standard Kalman Filter
    if len(ind_non_missing) == 0:
        xn = T @ xtilde  + C  
        P = T @ P @ T.T + Q
        
    else:
        if not bUnivariate:
            ### Multivariate approach to Kalman filter
            iF = np.zeros(shape=(n_meas,n_meas))
            # measurement residual
            v[ind_non_missing] = y[ind_non_missing] - ZZ @ x
            ### Multivariate approach to Kalman Filter
            F = H[np.ix_(ind_non_missing,ind_non_missing)] + ZZ @ P @ ZZ.T      # pre-fit residual covariance
            iF[np.ix_(ind_non_missing,ind_non_missing)] = la.inv(F)             # matrix inverse  
            PZI = P @ ZZ.T @ iF[np.ix_(ind_non_missing,ind_non_missing)]
            K[:,ind_non_missing]  = T @ PZI                                     # Kalman gain
            xtilde = x + PZI @ v[ind_non_missing]                               # updated (a posteriori) state estimate eq (4.26) DK(2012)
            xn  = T @ xtilde  + C                                               # predicted state estimate
            L   = T - K[:,ind_non_missing] @ ZZ
            P   = T @ P @ L.T + Q
            log_likelihood  += np.log(la.det(F)) + v @ iF @ v.T 
             
        else:
            
            ### Univariate approach to Kalman filter
            xtilde = np.copy(x)
            P1 = np.copy(P)
            for i in ind_non_missing:
                Z_i  = Z[i]
                v[i] = y[i] - Z_i @ xtilde                                      # measurement residual
                F_i  = F[i] = Z_i @ P @ Z_i.T + H[i,i]
                K_i  = K[:,i] = P @ Z_i.T
                if F_i > tol1:
                    xtilde += K_i * v[i] / F_i 
                    P -= np.outer(K_i,K_i) / F_i
                    log_likelihood  += np.log(F_i) + v[i]**2 /  F_i  
  
            # Transition equations
            xn = T @ xtilde + C # predicted state estimate
            P =  T @ P @ T.T  + Q                                      
                   
          
    # Log-likelihood of the observed data set
    log_likelihood = -0.5 * (n_meas*np.log(2*np.pi) + log_likelihood)
    if iF is None:
        iF = la.pinv(H+Z@P@Z.T )
        
    return xn, xtilde, v, P, P1, K, F,  iF, bUnivariate, log_likelihood
                

def Durbin_Koopman_Non_Diffuse_Smoother(x,y,T,Z,F,iF,P,P1,QRt,K,r,v,ind_non_missing=None,bUnivariate=False,tol=1.e-6,t=-1):
    r"""
    Implement non-diffusive Durbin-Koopman smoother for state-space model.
    
    Durbin-Koopman algorithm is
                
    .. math::
        r_{t-1} &= Z'_{t}*F_{t}^{-1}*v_{t} + L'_{t}**r_{t}
        
        r_{T} &= 0
        
        N_{t-1} &= Z'_{t}*F_{t}^{-1}*Z_{t} + L'_{t}*N_{t}*L{t}
        
        N_{T} &= 0
        
        L_{t} &= I - K_{t}*Z_{t}                                                          
        
    Smoothed variables
           
    .. math::
        s_{t} = x_{t} + P_{t}*r_{t-1}
        
    Args:
        x: array, dtype=float
           is the predicted variables vector,
        y: array, dtype=float
           is the observation variables,
        T: 2D array, dtype=float
           is the state-transition matrix,
        Z: 2D array, dtype=float
           is the observation matrix,
        F: 2D array, dtype=float
           is the residual covariance matrix,
        iF: 2D array, dtype=float
           is the inverse of matrix F,
        P: 2D array, dtype=float
           is the updated error covariance matrix,
        QRt: 2D array, dtype=float
           is the product of the covariance matrix of structural errors and shock matrix,
        K:2D array, dtype=float
           is the Kalman gain matrix,
        r: array, dtype=float
           is the vector of one-step-back errors,
        v: array, dtype=float
           is the vector of one-step-ahead prediction errors,
        ind_non_missing: array, dtype=int
           are the indices of non-missing observations.
        bUnivariate: bool
           True if univariate Kalman filter is used and False otherwise,
        tol: number, float
           is Kalman tolerance for reciprocal condition number.
        
    For details, see Durbin & Koopman (2012), "Time series Analysis by State Space Methods" and
    Koopman & Durbin (2000), "Fast Filtering and Smoothing For Multivariate State Space Models" and
    Koopman & Durbin (2003), in Journal of Time Series Analysis, vol. 24(1), pp. 85-98 
    """ 
    n_meas = len(Z)
    n = len(x)
    
    if ind_non_missing is None:
        cprint("Indeces of non-missing observations are not defined - resetting it.","red")
        ind_non_missing = np.arange(0,n_meas)
    rev_ind_non_missing = reversed(ind_non_missing)
    
    ZZ = Z[ind_non_missing]
    etahat = [np.nan]*len(QRt)
    L = T - K[:,ind_non_missing] @ ZZ
           
    if not bUnivariate:              #-----------------------------  Mutivariate smooting KF estimates
        # Smooth estimate
        r = ZZ.T @ iF[ind_non_missing,:] @ v[ind_non_missing] + L.T @ r               # eq (4.39) in DK(2012)  
        # Smooth estimate
        xhat = x + P @ r                                             # eq (4.39) in DK(2012)
        etahat = QRt @ r                                             # DK (2012)
    
    else:                            #-----------------------------  Univariate approach to smooting KF estimates

        for i in rev_ind_non_missing: 
                if F[i]  > tol:
                    L_i = np.eye(n) - np.outer(K[:,i],Z[i]) / F[i]
                    r = Z[i].T / F[i] * v[i] + L_i.T @ r             # DK (2012), 6.15, equation for r_{i-1}
                                                                     # DK (2012), below (6.15), r_{t-1}=r_{0}
        xhat  = x + P1 @ r                                           # DK (2012), eq (6.15)
        etahat = QRt @ r                                             # DK (2012), eq. 4.63
        r   = T.T @ r                                                # Smooth estimation of observation disturbance 
        
    # Smooth estimation of observation disturbance
    res = y - Z @ xhat           
                
    return xhat,r,res,etahat 


def Durbin_Koopman_Diffuse_Filter(x, xtilde, y, T, Z, P, H, Q, C=0, bUnivariate = False, ind_non_missing=None, Pstar=None, Pinf=None, tol1=1.e-6, tol2=1.e-6, t=-1):
    r"""
    Implement diffuse Durbin & Koopman version of Kalman filter algorithm for state-space model.
               
    State-space model:

    .. math::
        x_{t} = T_{t} * x_{t-1} + C_{t}  + w_{t},  w = Normal(0, Q)
        
        y_{t} = Z_{t} * x_{t} + v_{t}, v = Normal(0, H)
		
 
    Durbin-Koopman algorithm

    .. math::			
		v_{t} &= y_{t} - Z_{t} * x_{t}
        
		x_{t} &= T_{t} * (x_{t-1} + K_{t} * v_{t}) + C_{t}
        
		F_{t} &= H_{t} + Z_{t} * P_{t-1} * Z'_{t}
        
		K_{t} &= P_{t} * Z'_{t} * F_{t}^{-1}
        
        L_{t} &= K_{t} * Z_{t}
        
		P_{t+1} &= T_{t} * P_{t} *L'_{t-1} + Q_{t}		  
		
    .. note::
        Translated from Dynare Matlab code.
        
    Args:
        x: array, dtype=float
           is the state variables,
        xtilde: array, dtype=float
           is the variables at t|t+1 forecast step,
        y: array, dtype=float
           is the observation variables,
        T: 2D array, dtype=float
           is the state-transition matrix,
        Z: 2D array, dtype=float
           is the observation matrix,
        P: 2D array, dtype=float
           is the predicted error covariance matrix,
        H: 2D array, dtype=float
           is the covariance matrix of measurement errors,
        Q: 2D array, dtype=float
           is the covariance matrix of structural errors,
        C: array, dtype=float
           is the constant term matrix,
        ind_non_missing: array, dtype=int
           are the indices of non-missing observations,
        Pstar: 2D array, dtype=float
               is the finite part of the predicted error covariance matrix,
        Pinf: 2D array, dtype=float
              is the infinite part of the predicted error covariance matrix,
        tol1: number, float
            Kalman tolerance for reciprocal condition number,
        tol2: number, float
            diffuse Kalman tolerance for reciprocal condition number (for Finf) and the rank of Pinf
      
    For details, see Durbin & Koopman 2012, "Time series Analysis by State Space Methods" and
                 Koopman & Durbin 2000, "Fast Filtering and Smoothing For Multivariate State Space Models"
    """
    log_likelihood = 0
    n_meas         = len(y)
    n_states       = len(x)
    Finf_singular  = False
    if ind_non_missing is None:
        cprint("Indeces of non-missing observations are not defined - resetting it.","red")
        ind_non_missing = np.arange(0,n_meas)
    
    ZZ = Z[ind_non_missing]
    v = np.zeros(y.shape) 
    
    F      = np.zeros(shape=(n_meas))
    Finf   = np.zeros(shape=(n_meas,n_meas))
    Fstar  = np.zeros(shape=(n_meas,n_meas))
    K      = np.zeros(shape=(n_states,n_meas))
    Kstar  = np.zeros(shape=(n_states,n_meas))
    Kinf   = np.zeros(shape=(n_states,n_meas))
    iF,Linf,Lstar,P1,Pstar1,Pinf1 = None,None,None,None,None,None
    
    #P = sp.linalg.tril(P) + np.transpose(sp.linalg.tril(P,-1))  
    P1 = np.copy(P)
        
    # Check if matrices Finf, Fstar are singular
    diffuse = False
    if not Pinf is None and not Pstar is None:
        bUnivariate |= np.linalg.cond(Z@Pinf@ Z.T,-2)<tol2 or np.linalg.cond(H+Z@Pstar@ Z.T,-2)<tol2  
    if not Pinf is None:
        if bUnivariate:
            rank  = np.linalg.matrix_rank(Z@Pinf@Z.T,tol1)
        else:
            rank  = np.linalg.matrix_rank(Pinf,tol1)
        diffuse = not (rank == 0)
  
    ### --------------------------------------------------------- Standard Kalman Filter
    if len(ind_non_missing) == 0:
        xn = T @ xtilde  + C  
        L  = T
        Pstar = T @ P @ L.T + Q
        P = np.copy(Pstar)
        Pinf = T @ Pinf @ L.T
        
    else:
        if not diffuse:
            Kinf = Kstar = Linf = Lstar = Finf = Fstar = None
    
            if not bUnivariate:
                iF = np.zeros(shape=(n_meas,n_meas))
                ### Multivariate approach to Kalman filter
                # measurement residual
                v[ind_non_missing] = y[ind_non_missing] - ZZ @ x
                ### Multivariate approach to Kalman Filter
                F = H[np.ix_(ind_non_missing,ind_non_missing)] + ZZ @ P @ ZZ.T      # pre-fit residual covariance
                iF[np.ix_(ind_non_missing,ind_non_missing)] = la.inv(F)             # matrix inverse  
                PZI = P @ ZZ.T @ iF[np.ix_(ind_non_missing,ind_non_missing)]
                K[:,ind_non_missing]  = T @ PZI                                     # Kalman gain
                xtilde = x + PZI @ v[ind_non_missing]                               # updated (a posteriori) state estimate eq (4.26) DK(2012)
                xn     = T @ xtilde  + C                                            # predicted state estimate
                L      = T - K[:,ind_non_missing] @ ZZ
                Pstar  = T @ P @ L.T + Q
                P      = np.copy(Pstar)
                Pinf   = T @ Pinf @ L.T
                log_likelihood  += np.log(la.det(F)) + v @ iF @ v.T 
                 
            else:
                
                ### Univariate approach to Kalman filter
                xtilde = np.copy(x)
                for i in ind_non_missing:
                    Z_i  = Z[i]
                    v[i] = y[i] - Z_i @ xtilde                        # measurement residual
                    F_i  = F[i] = Z_i @ P @ Z_i.T + H[i,i]
                    K_i  = K[:,i] = P @ Z_i.T
                    if F_i > tol1:
                        xtilde += K_i * v[i] / F_i 
                        P -= np.outer(K_i,K_i) / F_i
                        log_likelihood  += np.log(F_i) + v[i]**2 /  F_i  
        
                # Transition equations
                xn = T @ xtilde + C # predicted state estimate
                Pstar =  T @ P @ T.T  + Q   
                P = np.copy(Pstar)  
                Pinf   = T @ Pinf @ L.T                                 
                   
                
        ### --------------------------------------------------------- Diffuse Kalman Filter
        else:
            
            if not bUnivariate: 
                ### Multivariate approach to KF
                # measurement residual
                v[ind_non_missing] = y[ind_non_missing] - ZZ @ x
                skip = False
                xtilde = np.copy(x)
                Finf[np.ix_(ind_non_missing,ind_non_missing)]  = ZZ @ Pinf  @ ZZ.T
                Fstar[np.ix_(ind_non_missing,ind_non_missing)] = ZZ @ Pstar @ ZZ.T + H[np.ix_(ind_non_missing,ind_non_missing)] 
                # Check if matrix Finf is singular
                if la.det(Finf) < tol1:
                    # Matrix Finf is singular
                    if not np.all(np.abs(Finf) < tol1): 
                        # The univariate diffuse Kalman filter should be used instead.
                        bUnivariate = True
                        skip = True
                    else: 
                        Finf_singular = True
                        if la.det(Fstar[np.ix_(ind_non_missing,ind_non_missing)]) < tol2:
                            if not np.all(np.abs(Fstar) < tol2): 
                                # The univariate diffuse Kalman filter should be used.
                                bUnivariate = True
                                skip = True
                            else:                                                 #rank 0
                                Pinf   = T @ Pinf @ T.T                           # eq (5.16)
                                Pstar  = T @ Pstar @ T.T + Q
                                xn     = T @ x + C  
                                log_likelihood  += np.log(la.det(Finf)) + v @ la.pinv(Finf) @ v.T 
                        else: # Matrix Fstar is positive definite
                            iFstar = la.inv(Fstar[np.ix_(ind_non_missing,ind_non_missing)]) 
                            Kstar[:,ind_non_missing] = Pstar @ ZZ.T @ iFstar      # K0 eq (5.15) in DK(2012)
                            Lstar  = T - T @ Kstar[:,ind_non_missing] @ ZZ        # L0 eq (5.15)
                            Pinf   = T @ Pinf @ T.T                               # eq (5.16)
                            Pstar  = T @ Pstar @ Lstar.T + Q
                            P      = np.copy(Pstar)
                            x     += Kstar @ v
                            xn     = T @ x + C  
                            log_likelihood  += np.log(la.det(Fstar)) + v @ iFstar @ v.T                          
                        
                elif not skip:  # Matrix Finf is positive definite
                    iFinf  = la.inv(Finf[np.ix_(ind_non_missing,ind_non_missing)]) 
                    Kinf[:,ind_non_missing]   = Pinf @ ZZ.T @ iFinf                 # K0 eq (5.12)
                    Linf   = T - T @Kinf[:,ind_non_missing] @ ZZ                  # L0 eq (5.12)
                    Kstar[:,ind_non_missing]  = (Pstar @ ZZ.T - Kinf[:,ind_non_missing] \
                        @ Fstar[np.ix_(ind_non_missing,ind_non_missing)] ) @ iFinf  # K1 in eq (5.12)  
                    Pstar  = T @ Pstar @ Linf.T \
                           - T @ Kinf[:,ind_non_missing] @ Finf @ Kstar[:,ind_non_missing].T @ T.T + Q  # eq (5.14)
                    Pinf   = T @ Pinf @ Linf.T                                    # eq (5.14) 
                    P      = np.copy(Pstar)
                    xtilde = x + Kinf @ v 
                    xn     = T @ xtilde  +  C  
                    log_likelihood  += np.log(la.det(Finf))  + v @ iFinf @ v
            
            if bUnivariate: 
                ### Univariate approach to KF
                Fstar  = np.zeros(n_meas)
                Finf   = np.zeros(n_meas)
                Pstar1 = np.copy(Pstar)
                Pinf1  = np.copy(Pinf)
                xtilde = np.copy(x)
                for i in ind_non_missing:
                    Z_i  = Z[i]
                    v[i] = y[i] - Z_i @ xtilde                                 # measurement residual
                    F[i] = Fstar[i] = Z_i @ Pstar @ Z_i.T + H[i,i]             # eq (15) in KD(2000)
                    Finf[i] = Z_i @ Pinf @ Z_i.T                        # eq (15)
                    K[:,i] = Kstar[:,i] = Kstar_i = Pstar @ Z_i.T               # eq (15)
                    Kinf[:,i] = Kinf_i  = Pinf  @ Z_i.T                             # eq (15)
                    if abs(Finf[i]) > tol1:
                        Kinf_Finf = Kinf_i / Finf[i]
                        xtilde += Kinf_Finf * v[i]                                    # eq (16)
                        Pstar  += np.outer(Kinf_i,Kinf_Finf) * Fstar[i]/Finf[i]  \
                               - np.outer(Kstar_i,Kinf_Finf) \
                               - np.outer(Kinf_Finf,Kstar_i)
                        Pinf  -= np.outer(Kinf_i,Kinf_i.T)/Finf[i]                    # eq (16)
                        log_likelihood   += np.log(Finf[i]) + v[i]**2 / Finf[i]  
                    elif abs(Fstar[i]) > tol2:
                        xtilde += Kstar_i * v[i] / Fstar[i]                            # eq (17)
                        Pstar -= np.outer(Kstar_i,Kstar_i) / Fstar[i]                  # eq (17)
                        log_likelihood += np.log(Fstar[i]) + v[i]**2 / Fstar[i]                          
                        
                      
                # Transition equations
                xn    = T @ xtilde + C  
                Pstar = T @ Pstar @ T.T + Q                                           # eq (18)  
                Pinf  = T @ Pinf @ T.T                                                # eq (18)
                P     = np.copy(Pstar)   
    
            
            if bUnivariate:
                rank  = np.linalg.matrix_rank(Z@Pinf@Z.T,tol1)
            else:
                rank  = np.linalg.matrix_rank(Pinf,tol1)
        
    # Log-likelihood of the observed data set
    log_likelihood = -0.5 * (n_meas*np.log(2*np.pi) + log_likelihood)
    
    if iF is None:
        iF = la.pinv(H+Z@P@Z.T )
        
    return xn, xtilde, v, P, P1, K, Kinf, Kstar, L, Linf, Lstar, F, Finf, Fstar, iF, log_likelihood, diffuse, Pstar, Pinf, Pstar1, Pinf1, Finf_singular, bUnivariate
                

def Durbin_Koopman_Diffuse_Smoother(x,y,T,Z,F,iF,QRt,K,Kstar,Kinf,H,
                                    P,P1,Pstar,Pinf,Pstar1,Pinf1,v,r,r0,r1,
                                    Linf=None,Lstar=None,ind_non_missing=None,
                                    diffuse=True,Finf_singular=False,bUnivariate=False,
                                    tol1=1.e-6,tol2=1.e-6,t=-1):
    r"""
    Implement diffuse Durbin-Koopman smoother for state-space model.
    
    Durbin-Koopman algorithm is
                
    .. math::
        r_{t-1} &= Z'_{t}*F_{t}^{-1}*v_{t} + L'_{t}**r_{t}
        
        r_{T} &= 0
        
        L_{t} &= I - K_{t}*Z_{t}                                                          
        
    Smoothed variables
          
    .. math:: 
        s_{t} = x_{t} + P_{t}*r_{t-1}
              
    .. note::
        Translated from Dynare Matlab code.
        
    Args:
        x: array, dtype=float
           is the predicted state variables vector,
        y: array, dtype=float
           is the observation variables,
        T: 2D array, dtype=float
           is the state-transition matrix,
        Z: 2D array, dtype=float
           is the observation matrix,
        F: 2D array, dtype=float
           is the residual covariance matrix,
        iF: 2D array, dtype=float
           is the inverse of matrix F,
        QRt: 2D array, dtype=float
           is the product of the covariance matrix of structural errors and shock matrix
        K: 2D array, dtype=float
           is the Kalman gain matrix,
        Kinf: 2D array, dtype=float
           is the infinite part of the Kalman gain matrix,
        Kstar: 2D array, dtype=float
           is the finite part of the Kalman gain matrix,
        H: 2D array, dtype=float
           is the observation errors covariance matrix,
        P: 2D array, dtype=float
           is the updated error covariance matrix P,
        P1:2D array, dtype=float
           is the updated error covariance matrix P1,
        Pstar: 2D array, dtype=float
           is the finite part of P matrix,
        Pinf: 2D array, dtype=float
           is the infinite part of P matrix,
        Pstar1: 2D array, dtype=float
           is the finite part of P1 matrix,
        Pinf1: 2D array, dtype=float
           is the infinite part of P1 matrix,
        ind_non_missing: array, dtype=int
           are the indices of non-missing observations.
        v: array, dtype=float
           is the vector of one-step-ahead prediction errors,
        r: array, dtype=float
           is  a smoothed state vector,
        r0: array, dtype=float
           is zero approximation in a Taylor expansion of a smoothed state vector,
        r1: array, dtype=float
           is the first approximation in a Taylor expansion of a smoothed state vector,
        Pstar: 2D array, dtype=float
           is the finite part of the predicted error covariance matrix,
        Pinf: 2D array, dtype=float
           is the infinite part of the predicted error covariance matrix,
        Finf_singular: bool
           True if Finf is a singular matrix and False otherwise,
        bUnivariate: bool
           True if univariate Kalman filter is used and False otherwise,
        tol1: number, float
           is Kalman tolerance for reciprocal condition number,
        tol2: number, float
           is diffuse Kalman tolerance for reciprocal condition number (for Finf) and the rank of Pinf
        
    For details, see Durbin & Koopman (2012), "Time series Analysis by State Space Methods" and
    Koopman & Durbin (2000), "Fast Filtering and Smoothing For Multivariate State Space Models" and
    Koopman & Durbin (2003), in Journal of Time Series Analysis, vol. 24(1), pp. 85-98)
    """ 
    n_meas = len(Z)
    n = len(x)
    
    if ind_non_missing is None:
        cprint("Indeces of non-missing observations are not defined - resetting it.","red")
        ind_non_missing = np.arange(0,n_meas)
    rev_ind_non_missing = reversed(ind_non_missing)
    
    ZZ = Z[ind_non_missing]
    Finf = ZZ @ Pinf @ ZZ.T
    Fstar = ZZ @ Pstar @ ZZ.T
    L = T - T@K[:,ind_non_missing] @ ZZ
    etahat = [np.nan]*len(QRt)
       
    if not diffuse:                      # Standard Kalman smoother
    
        if not bUnivariate:              #-----------------------------  Mutivariate smooting KF estimates
            # Smooth estimate
            r = ZZ.T @ iF[ind_non_missing,:] @ v + L.T @ r               # eq (4.39) in DK(2012)  
            # Smooth estimate
            xhat = x + P @ r                                             # eq (4.39) in DK(2012)
            etahat = QRt @ r                                             # DK (2012)
        
        else:                            #-----------------------------  Univariate approach to smooting KF estimates
    
            for i in rev_ind_non_missing: 
                    K_i = K[:,i]
                    if F[i]  > tol1:
                        L_i = np.eye(n) - np.outer(K_i,Z[i]) / F[i]
                        r = Z[i].T / F[i] * v[i] + L_i.T @ r             # DK (2012), 6.15, equation for r_{i-1}
                                                                         # DK (2012), below (6.15), r_{t-1}=r_{0}
            xhat  = x + P1 @ r                                           # DK (2012), eq (6.15)
            etahat = QRt @ r                                               # Smooth estimation of observation disturbance 
            
      
    else:                                #----------------------------------->  Diffuse Filter: time is within diffuse periods 
    
        if not bUnivariate:              #------------------------------------  Mutivariate smooting KF estimates
 
            if len(ind_non_missing) == 0:
                r1 = Linf.T @ r1
            else:
                if not Finf_singular:
                    iFinf = la.inv(Finf[np.ix_(ind_non_missing,ind_non_missing)])
                    r0   = Linf.T @ r0                                          # DK (2012), eq (5.21) where L^(0) is named Linf
                    r1   = ZZ.T @ (iFinf @ v[ind_non_missing]  \
                         - Kstar[:,ind_non_missing].T @ r0) \
                         + Linf.T @ r1 
                else:
                    iFstar = la.inv(Fstar[np.ix_(ind_non_missing,ind_non_missing)])
                    r0  = ZZ.T @ iFstar @ v[ind_non_missing] - Lstar.T @ r0     # DK (2003), eq (14)
                    r1  = T.T @ r1                                              # DK (2003), eq (14)
                   
                xhat  = x + Pstar @ r0 + Pinf @ r1                              # DK (2012), eq (5.23)                                            # Smooth estimation of observation disturbance
                etahat = QRt @ r0                                               # DK (2012), p. 135    
                
        else:                            #------------------------------------  Univariate approach to smooting KF estimates
                
            for i in rev_ind_non_missing:
                Z_i     = Z[i]
                Fstar_i = Z_i @ Pstar @ Z_i.T + H[i,i]                          # eq (15) in KD(2000)
                Finf_i  = Z_i @ Pinf @ Z_i.T                                    # eq (15)
                Kstar_i = Kstar[i]                                         # eq (15)
                Kinf_i  = Kinf[i]     
                if abs(Finf_i) > tol1:
                    Linf_i  = np.eye(n) - np.outer(Kinf_i,Z_i)/Finf_i
                    Lstar_i = np.outer(Kinf_i*Fstar_i/Finf_i-Kstar_i, Z_i) / Finf_i
                    r1 = Z_i.T * v[i] / Finf_i + Lstar_i @ r0    \
                       + Linf_i.T @ r1                                          # KD (2000), eq. (25) for r_1
                    r0 = Linf_i.T @ r0  
                elif abs(Fstar_i) > tol2:                                       # this step needed when Finf == 0
                    L_i = np.eye(n) - np.outer(Kstar_i,Z_i) / Fstar_i
                    r0  = Z_i.T / Fstar_i * v[i] + L_i.T @ r0                   # propagate r0 and keep r1 fixed
                    
            xhat  = x + Pstar1 @ r0 + Pinf1 @ r1                                # DK (2012), eq (5.23)
            etahat = QRt @ r0                                                   # DK (2012)
            r0  = T @ r0
            r1  = T @ r1
            
    # Smooth estimation of observation disturbance
    res = y - Z @ xhat           
        
    return xhat,r,r0,r1,res,etahat 



if __name__ == '__main__':
    """
    The test of implementation of Kalman filter for state-space model:
         
    .. math::
        a(t+1) = T * a(t) + eta(t),  eta(t) ~ N(0,Q)
        
        y(t) = Z * a(t) + eps(t),  eps(t) ~ N(0,H)
        
        a(0) ~ N(a0, P0)
    """
    import matplotlib.pyplot as plt

    # Parameters
    nobs = 1000
    sigma = 0.1
    
    # Example dataset
    u = np.geomspace(1,10,nobs)
    phi = 1+np.log(u[nobs-1]/u[0])/(nobs-1)
    
    np.random.seed(1234)
    eps = 0.5*np.random.normal(scale=sigma, size=nobs)
    x=np.zeros((nobs,1))
    x[:,0] = np.sin(10*u/np.pi)
    y=np.zeros((nobs,1))
    y[:,0] = np.sin(10*(u+eps)/np.pi)
    
    # State-Space
    Z = np.ones((1,1))
    T = phi*np.ones((1,1))
    Q = sigma**2*np.ones((1,1))
    H = np.ones((1,1))
    P = Q  
    
    ind_non_missing = np.arange(0,1)
    yk = []
    # Run the Kalman filter
    for t in range(nobs):
        xk,res,Q = Kalman_Filter(x=x[t],y=y[t],T=T,Z=Z,P=P,C=0,Q=Q,H=H,ind_non_missing=ind_non_missing)[:3]
        yk.append(xk)
        
    yk=np.array(yk)
    # Run the HP filter 
    yf = LRXfilter(y[:,0], 1600)[0]
    
    y = np.squeeze(y)
    x = np.squeeze(x)
    yk = np.squeeze(yk)
             
    plt.close("all")
    plt.figure(figsize=(10, 8))
    plt.plot(range(len(y)),y,'o',lw=1,label='Data')
    plt.plot(range(len(x)),x,lw=3,label='True Solution')
    plt.plot(range(len(yk)),yk,lw=2,label='Kalman Filter')
    plt.plot(range(len(yf)),yf,lw=1,label='HP Filter')
    plt.title('Filters',fontsize = 20)
    plt.legend(fontsize = 15)
    plt.grid(True)
    plt.show()
    