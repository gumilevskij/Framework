"""
Durbin-Koopman filter for non-stationary processes with missing and non-missing observations.
  
  It is a modification of Standard Kalman filter which accounts for growth of a covariance matrix P as time progresses.
  The initial values of P matrix is assumed diffusive.
  During the first d diffusive periods matrix P is decomposed into Pstar and Pinf
  and separate equations for these matrices are used.
  After number d of diffusive periods standard Kalman filter is applied.
 
  Translated from Dynare Matlab code to Python by A.Goumilevski.
"""

import numpy as np
from scipy import linalg as la
#from warnings import warn 
from snowdrop.src.misc.termcolor import cprint


def diffuse_filter(T,Z,R,Q,H,Y,C,a0,pp,mm,Ts,Nd,n_shocks,data_index,Pinf1,Pstar1,decomp_flag=False,state_uncertainty_flag=False):
    """
    Diffuse Kalman filter and smoother.
           
    References:
      * S.J. Koopman and J. Durbin (2003, "Filtering and Smoothing of State Vector for Diffuse State Space Models",  Journal of Time Series Analysis, vol. 24(1), pp. 85-98).                                                                 
      * Durbin/Koopman (2012): "Time Series Analysis by State Space Methods", Oxford University Press, Second Edition, Ch. 5
      
    
    Args:
       T:        mm*mm matrix    
                 is the state transition matrix
       Z:        pp*mm matrix    
                 is the selector matrix for observables in augmented state vector
       R:        mm*rr matrix    
                 is the second matrix of the state equation relating the structural innovations to the state variables
       Q:        rr*rr matrix    
                 is the covariance matrix of structural errors
       H:        pp*pp           
                 is the matrix of variance of measurement errors
       Y:        Ts*pp array           
                 is the vector of observations
       C:        mm array         
                 is the vector of constants
       a0:       mm array            
                 is the vector of initial values of endogenous variables
       pp:       int                
                 is the number of observed variables
       mm:       int                 
                 is the number of state variables
       Ts:       int                
                 is the sample size
       Nd:       int                
                 is the shocks maximum lead number minus minimum lag number
       n_shocks: int               
                 is the number of shocks
       data_index: 1*Tsarray        
                 cell of column vectors of indices
       nk:       int            
                 is the number of forecasting periods
       Pinf1:    mm*mm  2D array         
                 is the diagonal matrix with with q ones and m-q zeros
       Pstar1:   mm*mm 2D array        
                 is the variance-covariance matrix with stationary variables
       kalman_tol:  number, float             
                 is the tolerance for reciprocal condition number
       diffuse_kalman_tol: number, float     
                 is the tolerance for reciprocal condition number (for Finf) and the rank of Pinf
       decomp_flag:  bool
                 if True, compute filter decomposition
       state_uncertainty_flag: bool  
                 if True, compute uncertainty about smoothed state estimate
    
    Returns:
       alphahat: 
           smoothed variables (a_{t|T})
       epsilonhat:
           smoothed measurement errors
       etahat:   
           smoothed shocks
       atilde:   
           matrix of updated variables (a_{t|t})
       aK:       
           3D array of k step ahead filtered state variables (a_{t+k|t) (meaningless for periods 1:d)
       P:        
           3D array of one-step ahead forecast error variance matrices
       PK:       
           4D array of k-step ahead forecast error variance matrices (meaningless for periods 1:d)
       decomp:   
           decomposition of the effect of shocks on filtered values
       V:        
           3D array of state uncertainty matrices
       N:        
           3D array of state uncertainty matrices
       dLIK:
           1D array of likelihood
       log_likelihood:
           1D array of cumulative likelihood

    """
    try:
        from snowdrop.src.numeric.filters.diffuse import missing_obs_diffuse_multivariate_filter
        alphahat,epsilonhat,etahat,atilde,P1,aK,PK,decomp,V,N,dLIK,log_likelihood = \
            missing_obs_diffuse_multivariate_filter(T=T,Z=Z,R=R,Q=Q,H=H,Y=Y,C=C,a0=a0,pp=pp,mm=mm,Ts=Ts, \
                                                    Nd=Nd,n_shocks=n_shocks,data_index=data_index, \
                                                    Pinf1=Pinf1,Pstar1=Pstar1,decomp_flag=decomp_flag, \
                                                    state_uncertainty_flag=state_uncertainty_flag)
        if not alphahat is None: 
            P = P1
    except:
        alphahat = None
    if alphahat is None:
        from snowdrop.src.numeric.filters.diffuse import missing_obs_diffuse_univariate_filter 
        alphahat,epsilonhat,etahat,atilde,P,aK,PK,decomp,V,N,dLIK,log_likelihood = \
            missing_obs_diffuse_univariate_filter(T=T,Z=Z,R=R,Q=Q,H=np.diag(H),Y=Y,C=C,a0=a0,pp=pp,mm=mm,Ts=Ts, \
                                                  Nd=Nd,n_shocks=n_shocks,data_index=data_index,Pinf1=Pinf1,Pstar1=Pstar1, \
                                                  decomp_flag=decomp_flag,state_uncertainty_flag=state_uncertainty_flag)                  

    return alphahat,epsilonhat,etahat,atilde,P,aK,PK,decomp,V,N,dLIK,log_likelihood
            

def missing_obs_diffuse_multivariate_filter(T,Z,R,Q,H,Y,C,a0,pp,mm,Ts,Nd,n_shocks,data_index=None,nk=1,Pinf1=None,Pstar1=None,kalman_tol=1.e-10,diffuse_kalman_tol=1.e-6,decomp_flag=False,state_uncertainty_flag=False):
    """
    Compute diffuse Kalman smoother without measurement error, in case of a non-singular variance-covariance matrix of observation errors.
    
    Multivariate treatment of time series.
    
    .. note::
        Translated from Dynare Matlab code.
        
    References:
      See "Filtering and Smoothing of State Vector for Diffuse State Space
      Models", S.J. Koopman and J. Durbin (2003, in Journal of Time Series
      Analysis, vol. 24(1), pp. 85-98).
      Durbin/Koopman (2012): "Time Series Analysis by State Space Methods", Oxford University Press,
      Second Edition, Ch. 5
      
    Args:
       T:        mm*mm matrix    
                 is the state transition matrix
       Z:        pp*mm matrix        
                 is the selector matrix for observables in augmented state vector
       R:        mm*rr matrix        
                 is the second matrix of the state equation relating the structural innovations to the state variables
       Q:        rr*rr matrix        
                 is the covariance matrix of structural errors
       H:        pp*pp               
                 is the matrix of variance of measurement errors
       Y:        Ts*pp               
                 is the vector of observations
       C:        mm                  
                 is the vector of constants
       a0:       mm                  
                 is the vector of initial values of endogenous variables
       pp:       int                 
                 is the number of observed variables
       mm:       int                     
                 is the number of state variables
       Ts:       int                    
                 is the sample size
       Nd:       int                    
                 is the shocks maximum lead number minus minimum lag number
       n_shocks: int                    
                 is the number of shocks
       data_index: 1*Ts              
                 is the cell of column vectors of indices
       nk:       int                    
                 is the number of forecasting periods
       Pinf1:    mm*mm               
                 is the diagonal matrix with with q ones and m-q zeros
       Pstar1:   mm*mm               
                 is the variance-covariance matrix with stationary variables
       kalman_tol:                   
                 is the tolerance for reciprocal condition number
       diffuse_kalman_tol:           
                 is the tolerance for reciprocal condition number (for Finf) and the rank of Pinf
       decomp_flag:      
                 if true,  compute filter decomposition
       state_uncertainty_flag:   
                 if True, compute uncertainty about smoothed state estimate
    
    Returns:
       alphahat:     
                 is the smoothed variables (a_{t|T})
       epsilonhat:    
                 is the smoothed measurement errors
       etahat:       
                 is the smoothed shocks
       atilde:       
                 is the matrix of updated variables (a_{t|t})
       aK:           
                 is the 3D array of k step ahead filtered state variables (a_{t+k|t)
                 (meaningless for periods 1:d)
       P:            
                 is the 3D array of one-step ahead forecast error variance
                 matrices
       PK:           
                 is the 4D array of k-step ahead forecast error variance
                 matrices (meaningless for periods 1:d)
       decomp:       
                 is the decomposition of the effect of shocks on filtered values
       V:            
                 is the 3D array of state uncertainty matrices
       N:        
                 is the 3D array of state uncertainty matrices
       dLIK:
                 is 1D array of likelihood
       log_likelihood:
                 is 1D array of cumulative likelihood
       
      
    """
    d = 0           # number of diffuse periods
    n = len(T)
    decomp = None
    
    QQ = 0; QRt = 0
    for i in range(1+Nd):
        R1  = R[:,i*n_shocks:(1+i)*n_shocks]
        QQ  += R1 @ Q @ R1.T
        QRt += Q @ R1.T
            
    if Pinf1 is None:
        Pinf1 = 1.e6*np.eye(n)
    if Pstar1 is None:
        Pstar1 = np.copy(QQ)
    spinf           = Pinf1.shape
    spstar          = Pstar1.shape
    v               = np.zeros((pp,Ts))
    a               = np.zeros((mm,Ts+1))
    a[:,0]          = a0
    atilde          = np.zeros((mm,Ts))
    aK              = np.zeros((nk,mm,Ts+nk))
    PK              = np.zeros((nk,mm,mm,Ts+nk))
    iF              = np.zeros((pp,pp,Ts))
    Fstar           = np.zeros((pp,pp,Ts))
    iFstar          = np.zeros((pp,pp,Ts))
    iFinf           = np.zeros((pp,pp,Ts))
    K               = np.zeros((mm,pp,Ts))
    L               = np.zeros((mm,mm,Ts))
    Linf            = np.zeros((mm,mm,Ts))
    Lstar           = np.zeros((mm,mm,Ts))
    Kstar           = np.zeros((mm,pp,Ts))
    Kinf            = np.zeros((mm,pp,Ts))
    P               = np.zeros((mm,mm,Ts+1))
    Pstar           = np.zeros((spstar[0],spstar[1],Ts+1))
    Pstar[:,:,0]    = Pstar1
    Pinf            = np.zeros((spinf[0],spinf[1],Ts+1))
    Pinf[:,:,0]     = Pinf1
    rr              = len(Q)
    alphahat        = np.zeros((mm,Ts))
    etahat          = np.zeros((rr,Ts))
    epsilonhat      = np.zeros((rr,Ts))
    r               = np.zeros((mm,Ts+1))
    Finf_singular   = np.empty(Ts,dtype=bool)
    Finf_singular[:]= False
    dlik            = np.zeros(Ts)   # Initialization of the vector gathering the densities.
    dLIK            = np.Inf         # Default value of the log likelihood.
    if state_uncertainty_flag:
        V           = np.zeros((mm,mm,Ts))
        N           = np.zeros((mm,mm,Ts+1))
    else:
        V           = None
        N           = None
    if data_index is None:
        data_index = []
        for t in range(Ts):
            data_index.append(np.arange(0,pp))
            
    ## Forward pass of diffuse filter
    t = 0
    while np.linalg.matrix_rank(Pinf[:,:,t],diffuse_kalman_tol) and t < Ts:
        di = data_index[t]
        if len(di) == 0:
            #no observations, propagate forward without updating based on observations
            atilde[:,t]     = a[:,t]
            a[:,t+1]        = T@atilde[:,t] + C
            Linf[:,:,t]     = T
            Pstar[:,:,t+1]  = T@Pstar[:,:,t]@T.T + QQ
            Pinf[:,:,t+1]   = T@Pinf[:,:,t]@T.T
        else:
            ZZ = Z[di]  
            v[di,t] = Y[t,di] - ZZ@a[:,t]                                      #span selector matrix           
            Finf = ZZ@Pinf[:,:,t]@ZZ.T                                         # (5.7) in DK (2012)
            if np.linalg.cond(Finf,-2) < diffuse_kalman_tol:                   #F_{\infty,t} = 0
                if not np.all(abs(Finf) < diffuse_kalman_tol):                 #rank-deficient but not rank 0                                                             # The univariate diffuse kalman filter should be used.
                    return None,None,None,None,None,None,None,None,None,None,None
                else:                                                          #rank of F_{\infty,t} is 0
                    Finf_singular[t] = True
                    temp  = ZZ@Pstar[:,:,t]@ZZ.T + H[np.ix_(di,di)]            # (5.7) in DK (2012)
                    for k in di:
                        Fstar[k,di,t] = temp[k] 
                    if np.linalg.cond(Fstar[np.ix_(di,di)][:,:,t],-2) < kalman_tol: #F_{*} is singular
                        if not np.all(abs(Fstar[np.ix_(di,di)][:,:,t])<kalman_tol):
                            # The univariate diffuse kalman filter should be used.
                            return None,None,None,None,None,None,None,None,None,None,None
                        else: #rank 0
                            a[:,t+1] = T@a[:,t]+C
                            Pstar[:,:,t+1] = T@Pstar[:,:,t]@T.T+QQ
                            Pinf[:,:,t+1]  = T@Pinf[:,:,t]@T.T
                    else:
                        temp = la.inv(Fstar[np.ix_(di,di)][:,:,t])
                        for k in di:
                            iFstar[k,di,t] = temp[k]
                        Kstar[:,di,t]   = Pstar[:,:,t]@ZZ.T@iFstar[np.ix_(di,di)][:,:,t] #(5.15) of DK (2012) with Kstar=T^{-1}@K^(0)
                        Pinf[:,:,t+1]   = T@Pinf[:,:,t]*T.T                    # DK (2012), 5.16
                        Lstar[:,:,t]    = T - T@Kstar[:,di,t]@ZZ               # L^(0) in DK (2012), eq. 5.12
                        Pstar[:,:,t+1]  = T@Pstar[:,:,t]@Lstar[:,:,t].T+QQ     # (5.17) DK (2012)
                        a[:,t+1]        = T@(a[:,t]+Kstar[:,di,t]@v[di,t])+C   # (5.13) DK (2012)

            else:
                temp  = la.inv(Finf)
                for k in di:
                    iFinf[k,di,t] = temp[k]
                Kinf[:,di,t]    = Pinf[:,:,t]@ZZ.T@iFinf[np.ix_(di,di)][:,:,t] #define Kinf=T^{-1}@K_0 with M_{\infty}=Pinf@Z.T
                atilde[:,t]     = a[:,t] + Kinf[:,di,t]@v[di,t]
                Linf[:,:,t]     = T - T@Kinf[:,di,t]@ZZ                        # L^(0) in DK (2012), eq. 5.12
                temp  = ZZ@Pstar[:,:,t]@ZZ.T + H[di,di]                        #(5.7) DK(2012)
                for k in di:
                    Fstar[k,di,t] = temp[k]
                Kstar[:,di,t]   = (Pstar[:,:,t]@ZZ.T-Kinf[:,di,t]@temp)@iFinf[np.ix_(di,di)][:,:,t] #(5.12) DK(2012) with Kstar=T^{-1}@K^(1) note that there is a typo in DK (2003) with "+ Kinf" instead of "- Kinf", but it is correct in their appendix
                Pstar[:,:,t+1]  = T@Pstar[:,:,t]@Linf[:,:,t].T-T@Kinf[:,di,t]@Finf@Kstar[:,di,t].T@T.T + QQ #(5.14) DK(2012)
                Pinf[:,:,t+1]   = T@Pinf[:,:,t]@Linf[:,:,t].T                  #(5.14) DK(2012)
                a[:,t+1]  = T@atilde[:,t]+C
                
            aK[0,:,t] = a[:,t+1]
            # isn't a meaningless as long as we are in the diffuse part? MJ
            for j in range(1,nk):
                aK[j,:,t+j] = T@np.squeeze(aK[j-1,:,t+j-1])
                
        t += 1
       
    if t == Ts-1:
        return None,None,None,None,None,None,None,None,None,None,None
        
    # Forward pass of standard Kalman filter
    d = min(t-1,Ts-1)
    P = np.copy(Pstar)
    while t<Ts:
        P[:,:,t]=la.tril(P[:,:,t])+np.transpose(la.tril(P[:,:,t],-1))          # make sure P is symmetric
        di = data_index[t]
        if len(di) == 0:
            atilde[:,t]     = a[:,t]
            L[:,:,t]        = T
            P[:,:,t+1]      = T@P[:,:,t]@T.T + QQ                              #p. 111, DK(2012)
        else:
            ZZ = Z[di]
            v[di,t] = Y[t,di] - ZZ@a[:,t]
            F = ZZ@P[:,:,t]@ZZ.T + H[np.ix_(di,di)]
            diagF = np.copy(np.diag(F))
            diagF[diagF<0] = 0
            sig  = np.sqrt(diagF)
            sig2 = sig @ sig.T
    
            if np.any(np.diag(F)<kalman_tol) or np.linalg.cond(F/sig2,-2) < kalman_tol:
                return None,None,None,None,None,None,None,None,None,None,None
 
            temp = la.inv(F/sig2)/sig2
            k1 = -1
            for k in di:
                k1 += 1
                j1 = -1
                for j in di:
                    j1 += 1
                    iF[k,j,t] = temp[k1,j1]
            PZI         = P[:,:,t]@ZZ.T@iF[np.ix_(di,di)][:,:,t]
            atilde[:,t] = a[:,t] + PZI@v[di,t]
            K[:,di,t]   = T@PZI
            L[:,:,t]    = T-K[:,di,t]@ZZ
            P[:,:,t+1]  = T@P[:,:,t]@L[:,:,t].T + QQ
            dlik[t] += np.log(la.det(F)) + (v[di,t]@la.inv(F)@v[di,t].T) + np.log(2*np.pi)

        a[:,t+1]  = T@atilde[:,t] + C
        Pf        = P[:,:,t]
        aK[0,:,t] = a[:,t+1]
        for j in range(1,nk):
            Pf = T@Pf@T.T + QQ
            PK[j,:,:,t+j] = Pf
            aK[j,:,t+j] = T@np.squeeze(aK[j-1,:,t+j-1])

        t += 1
    
    ## Backward pass; r_T and N_T, stored in entry (N+1) were initialized at 0
    t = Ts-1
    while t>d:
        di = data_index[t]
        if len(di) == 0:
            # in this case, L is simply T due to Z=0, so that DK (2012), eq. 4.93 obtains
            r[:,t] = L[:,:,t].T@r[:,t+1]                                       #compute r_{t-1}, DK (2012), eq. 4.38 with Z=0
            if state_uncertainty_flag:
                N[:,:,t]=L[:,:,t].T@N[:,:,t+1]@L[:,:,t]                        #compute N_{t-1}, DK (2012), eq. 4.42 with Z=0

        else:
            ZZ = Z[di]
            r[:,t] = ZZ.T@iF[np.ix_(di,di)][:,:,t]@v[di,t] + L[:,:,t].T@r[:,t+1]            #compute r_{t-1}, DK (2012), eq. 4.38
            if state_uncertainty_flag:
                N[:,:,t]=ZZ.T@iF[np.ix_(di,di)][:,:,t]@ZZ+L[:,:,t].T@N[:,:,t+1]@L[:,:,t]    #compute N_{t-1}, DK (2012), eq. 4.42

        alphahat[:,t]       = a[:,t] + P[:,:,t]@r[:,t]                         #DK (2012), eq. 4.35
        etahat[:,t] = QRt@r[:,t]                                               #DK (2012), eq. 4.63
        if state_uncertainty_flag:
            V[:,:,t]    = P[:,:,t]-P[:,:,t]@N[:,:,t]@P[:,:,t]                  #DK (2012), eq. 4.43

        t -= 1
        
    if d >= 0: #diffuse periods
         # initialize r_d^(0) and r_d^(1) as below DK (2012), eq. 5.23
        r0 = np.zeros((mm,d+2))
        r0[:,d+1] = r[:,d+1]   #set r0_{d}, i.e. shifted by one period
        r1 = np.zeros((mm,d+2))     #set r1_{d}, i.e. shifted by one period
        if state_uncertainty_flag:
            #N_0 at (d+1) is N(d+1), so we can use N for continuing and storing N_0-recursion
            N_1=np.zeros((mm,mm,d+2))   #set N_1_{d}=0, i.e. shifted by one period, below  DK (2012), eq. 5.26
            N_2=np.zeros((mm,mm,d+2))   #set N_2_{d}=0, i.e. shifted by one period, below  DK (2012), eq. 5.26
            
        for t in range(d,-1,-1):
            di = data_index[t]
            if len(di) == 0:
                r1[:,t] = Linf[:,:,t].T@r1[:,t+1]
            else:
                if not Finf_singular[t]:
                    r0[:,t] = Linf[:,:,t].T@r0[:,t+1]                            # DK (2012), eq. 5.21 where L^(0) is named Linf
                    r1[:,t] = Z[di].T@(iFinf[np.ix_(di,di)][:,:,t]@v[di,t] \
                            - Kstar[:,di,t].T@T.T@r0[:,t+1]) \
                            + Linf[:,:,t].T@r1[:,t+1]                            # DK (2012), eq. 5.21, noting that i) F^(1)=(F^Inf)^(-1)(see 5.10), ii) where L^(0) is named Linf, and iii) Kstar=T^{-1}@K^(1)
                    if state_uncertainty_flag:
                        Lstar=-T@Kstar[:,di,t]@Z[di,:]                           # noting that Kstar=T^{-1}@K^(1)
                        N[:,:,t]=Linf[:,:,t].T@N[:,:,t+1]@Linf[:,:,t]            # DK (2012), eq. 5.19, noting that L^(0) is named Linf
                        N_1[:,:,t]=Z[di].T@iFinf[np.ix_(di,di)][:,:,t]@Z[di,:] \
                                  + Linf[:,:,t].T@N_1[:,:,t+1]@Linf[:,:,t] \
                                  + Lstar.T@N[:,:,t+1]@Linf[:,:,t]               # DK (2012), eq. 5.29; note that, compared to DK (2003) this drops the term (Lstar.T@N(:,:,t+1)@Linf(:,:,t)).T in the recursion due to it entering premultiplied by Pinf when computing V, and Pinf@Linf.T@N=0
                        N_2[:,:,t]=Z[di].T@(-iFinf[np.ix_(di,di)][:,:,t]@Fstar[np.ix_(di,di)][:,:,t]@iFinf[np.ix_(di,di)][:,:,t])@Z[di,:] \
                                  + Linf[:,:,t].T@N_2[:,:,t+1]@Linf[:,:,t] \
                                  + Linf[:,:,t].T@N_1[:,:,t+1]@Lstar \
                                  + Lstar.T@N_1[:,:,t+1].T@Linf[:,:,t] \
                                  + Lstar.T@N[:,:,t+1]@Lstar                     # DK (2012), eq. 5.29

                else:
                    r0[:,t] = Z[di].T@iFstar[np.ix_(di,di)][:,:,t]@v[di,t] \
                            - Lstar[:,:,t].T@r0[:,t+1] # DK (2003), eq. (14)
                            #ag -Lstar[:,di,t].T@r0[:,t+1] 
                    r1[:,t] = T.T@r1[:,t+1]                                      # DK (2003), eq. (14)
                    if state_uncertainty_flag:
                        N[:,:,t]=Z[di].T@iFstar[np.ix_(di,di)][:,:,t]@Z[di,:] \
                                + Lstar[:,:,t].T@N[:,:,t+1]@Lstar[:,:,t]         # DK (2003), eq. (14)
                        N_1[:,:,t]=T.T@N_1[:,:,t+1]@Lstar[:,:,t]                 # DK (2003), eq. (14)
                        N_2[:,:,t]=T.T@N_2[:,:,t+1]@T.T                          # DK (2003), eq. (14)

            alphahat[:,t]   = a[:,t] + Pstar[:,:,t]@r0[:,t] + Pinf[:,:,t]@r1[:,t]# DK (2012), eq. 5.23
            etahat[:,t]     = QRt@r0[:,t]                                        # DK (2012), p. 135
            if state_uncertainty_flag:
                V[:,:,t]=Pstar[:,:,t]-Pstar[:,:,t]@N[:,:,t]@Pstar[:,:,t] \
                        - (Pinf[:,:,t]@N_1[:,:,t]@Pstar[:,:,t]).T \
                        - Pinf[:,:,t]@N_1[:,:,t]@Pstar[:,:,t] \
                        - Pinf[:,:,t]@N_2[:,:,t]@Pinf[:,:,t]                     # DK (2012), eq. 5.30
    
    if decomp_flag:
        decomp = np.zeros((nk,mm,rr,Ts+nk))
        ZRQinv = la.inv(Z@QQ@Z.T)
        for t in range(d,Ts):
            di = data_index[t]
            # calculate eta_tm1t
            eta_tm1t = QRt@Z[di,:].T@iF[np.ix_(di,di)][:,:,t]@v[di,t]
            AAA = P[:,:,t]@Z[di,:].T@ZRQinv[np.ix_(di,di)]@((Z[di,:]@R)*eta_tm1t.T)
            # calculate decomposition
            decomp[0,:,:,t] = AAA
            for h in range(1,nk):
                AAA = T@AAA
                decomp[h,:,:,t+h] = AAA
    
    epsilonhat = Y.T-Z@alphahat
        
    # Divide by two
    dlik = -0.5*dlik
    dLIK = np.sum(dlik)
    
    return alphahat,epsilonhat,etahat,atilde,P,aK,PK,decomp,V,N,dLIK,dlik


def missing_obs_diffuse_univariate_filter(T,Z,R,Q,H,Y,C,a0,pp,mm,Ts,Nd,n_shocks,data_index=None,nk=1,Pinf1=None,Pstar1=None,kalman_tol=1.e-10,diffuse_kalman_tol=1.e-6,decomp_flag=False,state_uncertainty_flag=False):
    """
    Compute diffuse Kalman smoother in case of a singular var-cov matrix.
    
    Univariate treatment of multivariate time series. 
    It is applied for singular and non-singular  matrix of observation errors.

    References:
      * Durbin, Koopman (2012): "Time Series Analysis by State Space Methods", Oxford University Press, Second Edition, Ch. 6.4 + 7.2.5      
      * Koopman, Durbin (2000): "Fast Filtering and Smoothing for Multivariatze State Space Models", Journal of Time Series Analysis, vol. 21(3), pp. 281-296.      
      * S.J. Koopman and J. Durbin (2003): "Filtering and Smoothing of State Vector for Diffuse State Space Models", Journal of Time Series Analysis, vol. 24(1), pp. 85-98.
      
    .. note::
        Translated from Dynare Matlab code.
        
    Args:
       T:        mm*mm matrix         
                 is the state transition matrix
       Z:        pp*mm matrix         
                 is the selector matrix for observables in augmented state vector
       R:        mm*rr matrix         
                 is the second matrix of the state equation relating the structural innovations to the state variables
       Q:        rr*rr matrix         
                 is the covariance matrix of structural errors
       H:        pp                   
                 is the matrix of variance of measurement errors
       Y:        Ts*pp                
                 is the vector of observations
       C:        mm                   
                 is the vector of constants
       a0:       mm                   
                 is the vector of initial values of endogenous variables
       pp:       int                     
                 is the number of observed variables
       mm:       int                     
                 is the number of state variables
       Ts:       int                     
                 is the sample size
       Nd:       int                     
                 is the shocks maximum lead number minus minimum lag number
       n_shocks: int                     
                 is the number of shocks
       data_index: array, dtype=float                   
                 is the Ts cell of column vectors of indices
       nk:       int                     
                 is the number of forecasting periods
       Pinf1:    mm*mm                
                 is the diagonal matrix with with q ones and m-q zeros
       Pstar1:   mm*mm                
                 is the variance-covariance matrix with stationary variables
       kalman_tol:                    
                 is the tolerance for zero divider
       diffuse_kalman_tol:            
                 is the tolerance for zero divider
       decomp_flag:  if true,         
                 is the compute filter decomposition
       state_uncertainty_flag:    i    
                 is the f True, compute uncertainty about smoothed state estimate

    Returns:
       alphahat:     
                 is the smoothed state variables (a_{t|T})
       epsilonhat:     
                 is the measurement errors
       etahat:       
                 is the smoothed shocks
       a:            
                 is the matrix of updated variables (a_{t|t})
       aK:           
                 is the 3D array of k step ahead filtered state variables a_{t+k|t}
                 (meaningless for periods 1:d)
       P:            
                 is the 3D array of one-step ahead forecast error variance
                 matrices
       PK:           
                 is the 4D array of k-step ahead forecast error variance
                 matrices (meaningless for periods 1:d)
       decomp:       
                 is the decomposition of the effect of shocks on filtered values
       V:            
                 is the 3D array of state uncertainty matrices
       N:        
                 is the 3D array of state uncertainty matrices
       dLIK:
                 is 1D array of likelihood
       log_likelihood:
                 id 1D array of cumulative likelihood
        
    """
    assert np.ndim(H)==1,'missing_obs_diffuse_univariate_filter: H is not a vector.'
    
    n = len(T)
    d = 0
    decomp = None
    
    QQ = 0; QRt = 0
    for i in range(1+Nd):
        R1  = R[:,i*n_shocks:(1+i)*n_shocks]
        QQ  += R1 @ Q @ R1.T
        QRt += Q @ R1.T
        
    if Pinf1 is None:
        Pinf1 = 1.e6*np.eye(n)
    if Pstar1 is None:
        Pstar1 = np.copy(QQ)
    spinf           = Pinf1.shape
    spstar          = Pstar1.shape
    v               = np.zeros((pp,Ts))
    a               = np.zeros((mm,Ts))
    a1              = np.zeros((mm,Ts+1))
    a1[:,0]         = a0
    aK              = np.zeros((nk,mm,Ts+max(1,nk)))
    Fstar           = np.zeros((pp,Ts))
    Finf            = np.zeros((pp,Ts))
    Fi              = np.zeros((pp,Ts))
    Ki              = np.zeros((mm,pp,Ts))
    Kstar           = np.zeros((mm,pp,Ts))
    Kinf            = np.zeros((spstar[0],pp,Ts))
    P               = np.zeros((mm,mm,Ts+1))
    P1              = P
    PK              = np.zeros((nk,mm,mm,Ts+nk))
    Pstar           = np.zeros((spstar[0],spstar[1],Ts+1))
    Pstar[:,:,0]    = Pstar1
    Pinf            = np.zeros((spinf[0],spinf[1],Ts+1))
    Pinf[:,:,0]     = Pinf1
    Pstar1          = np.copy(Pstar)
    Pinf1           = np.copy(Pinf)
    rr              = len(Q) # number of structural shocks
    alphahat        = np.zeros((mm,Ts))
    etahat          = np.zeros((rr,Ts))
    epsilonhat      = np.zeros((rr,Ts))
    r               = np.zeros((mm,Ts))
    dlik = np.zeros(Ts)    # Initialization of the vector gathering the densities.
    dLIK = np.Inf          # Default value of the log likelihood.

    if state_uncertainty_flag:
        V           = np.zeros((mm,mm,Ts))
        N           = np.zeros((mm,mm,Ts))
    else:
        V           = None
        N           = None
    if data_index is None:
        data_index = []
        for t in range(Ts):
            data_index.append(np.arange(0,pp))
        
        
    ## Forward pass of diffuse filter 
    newRank = np.linalg.matrix_rank(Z@Pinf[:,:,0]@Z.T,diffuse_kalman_tol)
    t = 0
    while bool(newRank) and t < Ts:
        a[:,t] = a1[:,t]
        Pstar1[:,:,t] = Pstar[:,:,t]
        Pinf1[:,:,t] = Pinf[:,:,t]
        di = data_index[t]
        for i in di:
            Zi = Z[i]
            v[i,t]       = Y[t,i]-Zi@a[:,t]                                    # nu_{t,i} in 6.13 in DK (2012)
            Fstar[i,t]   = Zi@Pstar[:,:,t]@Zi.T+H[i]                           # F_{*,t} in 5.7 in DK (2012), relies on H being diagonal
            Finf[i,t]    = Zi@Pinf[:,:,t]@Zi.T                                 # F_{\infty,t} in 5.7 in DK (2012)
            Kstar[:,i,t] = Pstar[:,:,t]@Zi.T                                   # KD (2000), eq. (15)
            if abs(Finf[i,t]) > diffuse_kalman_tol and newRank > 0:            # F_{\infty,t,i} = 0, use upper part of bracket on p. 175 DK (2012) for w_{t,i}
                Kinf[:,i,t]       = Pinf[:,:,t]@Zi.T                           # KD (2000), eq. (15)
                Kinf_Finf         = Kinf[:,i,t]/Finf[i,t]
                a[:,t]           += Kinf_Finf*v[i,t]                           # KD (2000), eq. (16)
                Pstar[:,:,t]     += np.outer(Kinf[:,i,t],Kinf_Finf)*Fstar[i,t]/Finf[i,t] \
                                  - np.outer(Kstar[:,i,t],Kinf_Finf) \
                                  - np.outer(Kinf_Finf,Kstar[:,i,t])           # KD (2000), eq. (16)
                Pinf[:,:,t]      -= np.outer(Kinf[:,i,t],Kinf[:,i,t].T)/Finf[i,t]   # KD (2000), eq. (16)            
            elif abs(Fstar[i,t]) > kalman_tol:
                a[:,t]            += Kstar[:,i,t]*v[i,t]/Fstar[i,t]            # KD (2000), eq. (17)
                Pstar[:,:,t]      -= np.outer(Kstar[:,i,t],Kstar[:,i,t])/Fstar[i,t] # KD (2000), eq. (17)
                                                                               # Pinf is passed through unaltered, see eq. (17) of
                                                                               # Koopman/Durbin (2000)
            else:
                pass
                # do nothing as a_{t,i+1}=a_{t,i} and P_{t,i+1}=P_{t,i}, see
                # p. 157, DK (2012)
            
        
        a1[:,t+1] = T@a[:,t] + C
        aK[0,:,t] = a1[:,t+1]
        for j in range(1,nk):
            aK[j,:,t+j+1] = T@np.squeeze(aK[j-1,:,t+j])
        
        Pstar[:,:,t+1] = T@Pstar[:,:,t]@T.T + QQ
        Pinf[:,:,t+1] = T@Pinf[:,:,t]@T.T
        
        if bool(newRank):
            newRank = np.linalg.matrix_rank(Z@Pinf[:,:,t+1]@Z.T,diffuse_kalman_tol)
            oldRank = np.linalg.matrix_rank(Z@Pinf[:,:,t]@Z.T,diffuse_kalman_tol)
        else:
            oldRank = 0
        if not oldRank == newRank:
            cprint('missing_obs_diffuse_univariate_filter: T does influence the rank of Pinf!','red')
     
        t += 1
    
    d = min(t-1,Ts-1)
    P = Pstar
    Pstar1 = Pstar1[:,:,:d+1]
    Pinf1  = Pinf1[:,:,:d+1]
    
    #np.set_printoptions(precision=2)
    ## Forward pass of standard Kalman filter 
    while t<Ts:
        a[:,t] = a1[:,t]
        P1[:,:,t] = P[:,:,t]
        di = data_index[t]
        for i in di:
            Zi = Z[i]
            v[i,t]  = Y[t,i] - Zi@a[:,t]                                       # nu_{t,i} in 6.13 in DK (2012)
            Fi[i,t] = Zi@P[:,:,t]@Zi.T + H[i]                                  # F_{t,i} in 6.13 in DK (2012), relies on H being diagonal 
            Ki[:,i,t] = P[:,:,t]@Zi.T                                          # K_{t,i}*F_(i,t) in 6.13 in DK (2012)
            if Fi[i,t] > kalman_tol:
                a[:,t] += Ki[:,i,t]*v[i,t]/Fi[i,t]                             # filtering according to (6.13) in DK (2012)
                P[:,:,t] -= np.outer(Ki[:,i,t],Ki[:,i,t])/Fi[i,t]              # filtering according to (6.13) in DK (2012)
                dlik[t] +=  np.log(Fi[i,t]) + (v[i,t]**2/Fi[i,t]) + np.log(2*np.pi)
            else:
                pass
                # do nothing as a_{t,i+1}=a_{t,i} and P_{t,i+1}=P_{t,i}, see p. 157, DK (2012)
         
        a1[:,t+1] = T@a[:,t] + C                                               #transition according to (6.14) in DK (2012)
        Pf        = P[:,:,t]
        aK[0,:,t] = a1[:,t+1]
        for j in range(1,nk):
            Pf = T@Pf@T.T + QQ
            PK[j,:,:,t+j] = Pf
            aK[j,:,t+j] = T@np.squeeze(aK[j-1,:,t+j-1])
            
        P[:,:,t+1] = T@P[:,:,t]@T.T + QQ                                       #transition according to (6.14) in DK (2012)         
        t += 1
        
    ## Backward pass
    ri=np.zeros(mm)
    if state_uncertainty_flag:
        Ni=np.zeros((mm,mm))
    
    t = Ts-1
    while t > d:
        di = np.flipud(data_index[t])
        for i in di:
            if Fi[i,t] > kalman_tol:
                Li = np.eye(mm)-np.outer(Ki[:,i,t],Z[i])/Fi[i,t]
                ri = Z[i].T/Fi[i,t]*v[i,t]+Li.T@ri                             # DK (2012), 6.15, equation for r_{t,i-1}
                if state_uncertainty_flag:
                    Ni = Z[i].T/Fi[i,t]@Z[i]+Li.T@Ni@Li                        # KD (2000), eq. (23)
              
        r[:,t] = ri                                                            # DK (2012), below 6.15, r_{t-1}=r_{t,0}
        alphahat[:,t] = a1[:,t] + P1[:,:,t]@r[:,t]
        etahat[:,t] = QRt@r[:,t]
        ri = T.T@ri                                                            # KD (2003), eq. (23), equation for r_{t-1,p_{t-1}}
        if state_uncertainty_flag:
            N[:,:,t] = Ni                                                      # DK (2012), below 6.15, N_{t-1}=N_{t,0}
            V[:,:,t] = P1[:,:,t]-P1[:,:,t]@N[:,:,t]@P1[:,:,t]                  # KD (2000), eq. (7) with N_{t-1} stored in N(:,:,t)
            Ni = T.T@Ni@T
        
        t -= 1                                                                 # KD (2000), eq. (23), equation for N_{t-1,p_{t-1}}
        
    if d >= 0:
        r0 = np.zeros((mm,d+1))
        r0[:,d] = ri
        r1 = np.zeros((mm,d+1))
        if state_uncertainty_flag:
                                                                               # N_0 at (d+1) is N(d+1), so we can use N for continuing and storing N_0-recursion
            N_0=np.zeros((mm,mm,d+1))                                          # set N_1_{d}=0, below  KD (2000), eq. (24)
            N_0[:,:,d] = Ni
            N_1=np.zeros((mm,mm,d+1))                                          # set N_1_{d}=0, below  KD (2000), eq. (24)
            N_2=np.zeros((mm,mm,d+1))                                          # set N_2_{d}=0, below  KD (2000), eq. (24)
        
        for t in range(d,-1,-1):
            di = np.flipud(data_index[t])
            for i in di:
                if abs(Finf[i,t]) > diffuse_kalman_tol:
                    # recursions need to be from highest to lowest term in order to not
                    # overwrite lower terms still needed in this step
                    Linf    = np.eye(mm) - np.outer(Kinf[:,i,t],Z[i])/Finf[i,t]
                    Lstar   = np.outer((Kinf[:,i,t]*Fstar[i,t]/Finf[i,t]-Kstar[:,i,t]),Z[i])/Finf[i,t]
                    r1[:,t] = Z[i].T*v[i,t]/Finf[i,t] + \
                              Lstar.T@r0[:,t] + \
                              Linf.T@r1[:,t]   # KD (2000), eq. (25) for r_1
                    r0[:,t] = Linf.T@r0[:,t]   # KD (2000), eq. (25) for r_0
                    if state_uncertainty_flag:
                        N_2[:,:,t] = Z[i].T / Finf[i,t]**2 @ Z[i] * Fstar[i,t] \
                                   + Linf.T@N_2[:,:,t]@Linf \
                                   + Linf.T@N_1[:,:,t]@Lstar \
                                   + Lstar@N_1[:,:,t].T@Linf \
                                   + Lstar@N_0[:,:,t]@Lstar                    # DK (2012), eq. 5.29
                        N_1[:,:,t]=Z[i].T/Finf[i,t]@Z[i] \
                                   + Linf.T@N_1[:,:,t]@Linf \
                                   + Lstar@N_0[:,:,t]@Linf                     # DK (2012), eq. 5.29 note that, compared to DK (2003) this drops the term (Lstar.T*N(:,:,t+1)*Linf(:,:,t)).T in the recursion due to it entering premultiplied by Pinf when computing V, and Pinf*Linf.T*N=0
                        N_0[:,:,t]=Linf.T@N_0[:,:,t]@Linf                      # DK (2012), eq. 5.19, noting that L^(0) is named Linf
                    
                elif abs(Fstar[i,t]) > kalman_tol: # step needed when Finf == 0
                    L_i=np.eye(mm) - np.outer(Kstar[:,i,t],Z[i])/Fstar[i,t]
                    r0[:,t] = Z[i].T/Fstar[i,t]*v[i,t]+L_i.T@r0[:,t]         # propagate r0 and keep r1 fixed
                    if state_uncertainty_flag:
                        N_0[:,:,t]=np.outer(Z[i].T/Fstar[i,t],Z[i])+L_i.T@N_0[:,:,t]@L_i   # propagate N_0 and keep N_1 and N_2 fixed
                    
            alphahat[:,t] = a1[:,t] + Pstar1[:,:,t]@r0[:,t] + Pinf1[:,:,t]@r1[:,t] # KD (2000), eq. (26)
            r[:,t]        = r0[:,t]
            etahat[:,t]   = QRt@r[:,t]                                         # KD (2000), eq. (27)
            if state_uncertainty_flag:
                V[:,:,t]=Pstar[:,:,t]-Pstar[:,:,t]@N_0[:,:,t]@Pstar[:,:,t] \
                        - (Pinf[:,:,t]@N_1[:,:,t]@Pstar[:,:,t]).T \
                        - Pinf[:,:,t]@N_1[:,:,t]@Pstar[:,:,t] \
                        - Pinf[:,:,t]@N_2[:,:,t]@Pinf[:,:,t]                   # DK (2012), eq. 5.30
            
            if t >= 1:
                r0[:,t-1] = T.T@r0[:,t]                                        # KD (2000), below eq. (25) r_{t-1,p_{t-1}}=T.T*r_{t,0}
                r1[:,t-1] = T.T@r1[:,t]                                        # KD (2000), below eq. (25) r_{t-1,p_{t-1}}=T.T*r_{t,0}
                if state_uncertainty_flag:
                    N_0[:,:,t-1]= T.T@N_0[:,:,t]@T                             # KD (2000), below eq. (25) N_{t-1,p_{t-1}}=T.T*N_{t,0}*T
                    N_1[:,:,t-1]= T.T@N_1[:,:,t]@T                             # KD (2000), below eq. (25) N^1_{t-1,p_{t-1}}=T.T*N^1_{t,0}*T
                    N_2[:,:,t-1]= T.T@N_2[:,:,t]@T                             # KD (2000), below eq. (25) N^2_{t-1,p_{t-1}}=T.T*N^2_{t,0}*T
               
    if decomp_flag:
        decomp = np.zeros((nk,mm,rr,Ts+nk))
        ZRQinv = la.pinv(Z@QQ@Z.T)
        for t in range(d,Ts):
            ri_d = np.zeros(mm)
            di = np.flipud(data_index[t])
            for i in di:
                if abs(Fi[i,t]) > kalman_tol:
                    ri_d = Z[i].T/Fi[i,t]*v[i,t] + ri_d - Ki[:,i,t].T@ri_d/Fi[i,t]*Z[i].T
                
            # calculate eta_tm1t
            eta_tm1t = QRt@ri_d
            # calculate decomposition
            Ttok = np.eye(mm)
            AAA = P1[:,:,t]@Z.T@ZRQinv@Z@R
            for h in range(nk):
                BBB = Ttok@AAA
                for j in range(rr):
                    decomp[h,:,j,t+h] = eta_tm1t[j]*BBB[:,j]
                
                Ttok = T@Ttok
            
    epsilonhat = Y.T - Z@alphahat

    # Divide by two
    dlik = -0.5*dlik
    dLIK = np.sum(dlik)

    return alphahat,epsilonhat,etahat,a,P,aK,PK,decomp,V,N,dLIK,dlik


if __name__ == '__main__':
    """Test Python implementation of diffuse filters against Dynare implementation."""
    from mat4py import loadmat
    import os
    
    path = os.path.dirname(os.path.abspath(__file__))
    working_dir = os.path.abspath(os.path.join(os.path.abspath(path + "../../../..")))
    
    data = loadmat(working_dir + "/data/toy/filter.mat")

    T = np.array(data['T'])
    Z = np.array(data['Z'])
    R = np.array(data['R'])
    Q = np.array(data['Q'])
    H = np.array(data['H'])
    Pinf1 = np.array(data['Pinf'])
    Pstar1 = np.array(data['Pstar'])
    Y = np.array(data['Y']).T
    index = np.array(data['data_index'])
    index = np.squeeze(index)
    alphahat = np.array(data['alphahat'])
    epsilonhat = np.array(data['epsilonhat'])
    etahat = np.array(data['etahat'])
    ahat = np.array(data['ahat'])
    
    n = len(T)
    C = np.zeros(n)
    a0 = np.zeros(n)
    pp = int(data['pp'])
    mm = int(data['mm'])
    Ts = int(data['smpl'])
    nk = 1
    kalman_tol = float(data['kalman_tol'])
    diffuse_kalman_tol = float(data['diffuse_kalman_tol'])
    decomp_flag = bool(data['decomp_flag'])
    state_uncertainty_flag = bool(data['state_uncertainty_flag'])
    Nd = 0
    n_shocks = R.shape[1]
    # Matlab arrays start with index 1 and Python with 0 - subtract 1 from indices
    data_index = list()
    for ind in index:
        data_index.append(ind-1) 
       
    alphahat1,epsilonhat1,etahat1,ahat1,P,aK,PK,decomp,V,N,dLIK,dlik = \
        missing_obs_diffuse_multivariate_filter(T,Z,R,Q,H,Y,C,a0,pp,mm,Ts,Nd,n_shocks, \
        data_index,nk,Pinf1,Pstar1,kalman_tol,diffuse_kalman_tol,decomp_flag,state_uncertainty_flag)
    
    alphahat2,epsilonhat2,etahat2,ahat2,P,aK,PK,decomp,V,N,dLIK,dlik = \
        missing_obs_diffuse_univariate_filter(T,Z,R,Q,np.diag(H),Y,C,a0,pp,mm,Ts,Nd,n_shocks, \
        data_index,nk,Pinf1,Pstar1,kalman_tol,diffuse_kalman_tol,decomp_flag,state_uncertainty_flag)

    diff_alpha = la.norm(alphahat-alphahat1)+la.norm(alphahat-alphahat2)
    diff_ahat = la.norm(ahat-ahat1)+la.norm(ahat-ahat2)
    print(diff_alpha)
    print(diff_ahat)
