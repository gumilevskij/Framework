"""
Created on Mon Feb 26 18:49:21 2018.

@author: agoumilevski
"""

import numpy as np
from time import time
import numpy.linalg as la
from snowdrop.src.numeric.solver import linear_solver
#from snowdrop.src.numeric.filters.utils import getTimeShift
from snowdrop.src.model.settings import SolverMethod, SmootherAlgorithm
from snowdrop.src.numeric.solver.util import getStableUnstableRowsColumns 
from snowdrop.src.numeric.filters.filters import Bryson_Frazier_Smoother
from snowdrop.src.numeric.filters.filters import DK_Filter
from snowdrop.src.numeric.filters.filters import DK_Smoother
from snowdrop.src.numeric.filters.filters import Durbin_Koopman_Non_Diffuse_Filter as DK_Non_Diffuse_Filter 


def linear_filter(model,T,periods,y0,Qm=None,Hm=None,obs=None,MULT=1,skip=0,missing_obs=None,ind_non_missing=None):
    """
    Apply Kalman filter to linear model.
    
    Parameters:
        :param model: Model object.
        :type model: Model.
        :param T: Time span.
        :type T: int.
        :param periods: Array of endogenous variables.
        :type periods: numpy.array.
        :param y0: Starting values of endogenous variables.
        :type y0: numpy.array     
        :param Qm: Covariance matrix of errors of endogenous variables. 
        :type Qm: numpy.array.
        :param Hm: Covariance matrix of errors of measurement variables.
        :type Hm: numpy.array.
        :param obs: Measurements.
        :type obs: List.
        :param MULT: Multiplier of the terminal time range.  If set greater than one than 
                     the solution will be computed for this extended time range interval.
        :type MULT: int.
        :param missing_obs: Matrix of logical values with True or False value depending on observation missing or not.
        :type missing_obs: Matrix of bools. 
        :returns: Filtered solution.
    """
    from snowdrop.src.numeric.solver.linear_solver import simulate
    from snowdrop.src.model.settings import FilterAlgorithm, PriorAssumption
    
    T0 = T
    T = int(T*MULT)
    err = 0.0
    count = 1 
    # Get length of observations
    nt = len(obs)
    t0 = time()  
    
    steady_state = model.steady_state
    if steady_state is None:
       steady_state = y0
    
    if nt == 0 or Qm is None or Hm is None or obs is None:
        count,yy,yyIter,err,elapsed = simulate(model=model,T=T,periods=periods,y0=y0,steady_state=steady_state)
        return count,yy,None,None,err,elapsed
    
    variables = model.symbols['variables']
    n = len(variables)
    shock_var = model.symbols['shocks']
    n_shocks = len(shock_var)
   
    if 'distribution' in model.options:
        process = model.options['distribution']
        shock_values = process.simulate(1,1+T)
    else:
        if 'shock_values' in model.options:
            shock_values = model.options.get('shock_values')
            if isinstance(shock_values,list) and len(shock_values)==1:
                if isinstance(shock_values[0],list):
                   shock_values = shock_values[0] 
        else:
            shock_values = np.zeros(n_shocks)
    shock_values = np.array(shock_values) 
    
    if 'measurement_shocks' in model.symbols:
        n_meas_shocks = len(model.symbols['measurement_shocks'])
    else:
        n_meas_shocks = 0
        
    # Measurement function
    f_measurement = model.functions['f_measurement']
    bHasAttr  = hasattr(f_measurement,"py_func")
    meas_variables = model.symbols['measurement_variables']
    nm = len(meas_variables)
    K = np.zeros((n,n))
   
    Pstar=None;Pinf=None;P1=None;K=None;alphahat=None;loglk=None;Sinv=None;yl=None;epsilonhat=None
    residuals=[None];lK=[None];lP=[None];lP1=[None];lS=[None];lSinv=[None];S=None;Sinv=None;etahat=None;res=None
    if nt <= 0:
        # Solution that does not take into account measurements
        count,yy,yyIter,err,elapsed = simulate(model=model,T=T,periods=periods,y0=y0,steady_state=steady_state)
        return count,yy,None,None,err,elapsed
    
    else:
            
        # Leads and lags of shocks
        Nd = model.max_lead_shock - model.min_lag_shock
        
        meas_params = model.calibration['measurement_parameters']
        meas_var = np.zeros(n+nm+n_shocks+n_meas_shocks)
        if bHasAttr:
            meas_const,meas_jacob = f_measurement.py_func(meas_var,meas_params,order=1)
        else:
            meas_const,meas_jacob = f_measurement(meas_var,meas_params,order=1)
        
        obs -= meas_const
        Z = -meas_jacob[:,:n]
            
        # Solve linear model
        linear_solver.solve(model=model)
        if model.SOLVER.value == SolverMethod.Klein.value:
            # State transition matrix
            F1 = model.linear_model["Ta"]
            # Array of constants
            C = model.linear_model["Ka"]
            # Matrix of shocks
            R = model.linear_model["Ra"]
            # Matrix of state variables
            U = model.linear_model["U"]
            Z = Z @ U
            # Initial values
            y0 = la.solve(U,y0)
        else:
            # State transition matrix
            F = model.linear_model["A"]
            F1 = F[:n,:n]
            # Array of constants
            C = model.linear_model["C"]
            C = C[:n]
            U = np.eye(n)
            # Matrix of shocks
            R = model.linear_model["R"]
            R = R[:n]

        # mvar = dict(zip(variables,y0))
        # mobs = dict(zip(model.symbols['measurement_variables'],obs.T))

        mf = np.array(np.nonzero(Z))
        Q = 0
        for i in range(1+model.max_lead_shock-model.min_lag_shock):
            R1 = R[:n,i*n_shocks:(1+i)*n_shocks]
            Q += R1 @ Qm @ R1.T
        Q = 0.5*(Q+Q.T)
            
        P = np.copy(Q)
        Pstar = np.copy(P)
        
        rowStable,colStable,rowUnstable,colUnstable = getStableUnstableRowsColumns(model,T=F1,K=C)    

        y = np.zeros((T+2,n)); yt = np.zeros((T+2,n))
        filtered = np.copy(y0); ft = np.copy(y0)
        y[0] = yt[0] = y0
        
        # Get starting values of matrices Pstar, Pinf
        if model.PRIOR is None:
            Pinf = 1.e6*np.eye(n)
        elif model.PRIOR.value == PriorAssumption.Diffuse.value:
            # Diffuse intial condition for covariance matrix
            from snowdrop.src.numeric.filters.utils import compute_Pinf_Pstar
            Pinf,Pstar = compute_Pinf_Pstar(mf=mf,T=F1,R=R,Q=Qm,N=Nd,n_shocks=n_shocks)            
            P = np.copy(Pstar)
        elif model.PRIOR.value == PriorAssumption.StartingValues.value:
            Pinf  = 1.e6*np.eye(n) 
        elif model.PRIOR.value == PriorAssumption.Equilibrium.value:
            # Lyapunov equation for stable part of covariance matrix
            from snowdrop.src.numeric.solver.solvers import lyapunov_solver 
            Pinf = np.zeros((n,n))
            P = lyapunov_solver(T=F1[np.ix_(rowStable,colStable)],R=R[rowStable],Q=Qm,N=Nd,n_shocks=n_shocks,options="discrete")
            Pstar[np.ix_(rowStable,colStable)] = P
            P = np.copy(Pstar)
        elif model.PRIOR.value == PriorAssumption.Asymptotic.value:
            # Riccati discrete-time equation
            from scipy.linalg import solve_discrete_are 
            Pstar = solve_discrete_are(a=F1.T,b=Z.T,q=Q,r=Hm)
            P = np.copy(Pstar)
            Pinf = 1.e6*np.eye(n)
                
        # Make sure covariance matrix is symmetric  
        P = 0.5*(P+P.T)
        Pstar = 0.5*(Pstar+Pstar.T)  
        Pinf  = 0.5*(Pinf+Pinf.T)
       
###################################################### Kalman Filter
        
        if not model.FILTER is None and model.FILTER.value == FilterAlgorithm.Diffuse.value:
            from snowdrop.src.numeric.filters.diffuse import diffuse_filter
            alphahat,epsilonhat,etahat,atilde,P1,aK,PK,decomp,V,N,dLIK,log_likelihood = \
                diffuse_filter(T=F1,Z=Z,R=R[:n],Q=Qm,H=Hm,Y=obs,C=C,a0=filtered,pp=nm,mm=n,Ts=nt, \
                               Nd=Nd,n_shocks=n_shocks,data_index=ind_non_missing,Pinf1=Pinf,Pstar1=Pstar, \
                               decomp_flag=False,state_uncertainty_flag=False)                    
            epsilonhat = epsilonhat.T[:2+T0]
            etahat = etahat.T[:2+T0]
            y = np.array(atilde[:n]).T
            y = y[:2+T0]
            if model.SMOOTHER is None:
                yl = np.copy(y)
            else:
                yl = np.array(alphahat[:n]).T
                yl = yl[:2+T0]        
        
        elif not model.FILTER is None and model.FILTER.value == FilterAlgorithm.Particle.value:
            from snowdrop.src.numeric.filters import kalman
            ss_model = kalman.MVLinearGauss(F=F1,G=Z,mu0=y0,covX=Q,covY=Hm)
            kf = kalman.Kalman(ssm=ss_model,data=np.nan_to_num(obs))
            kf.filter()
            y = [kf.filt[i].mean for i in range(nt)]
            y = np.array(y)
            
            if not model.SMOOTHER is None:
                kf.smoother()
                yl = [kf.smth[i].mean for i in range(nt)]
                yl = np.array(yl)
                smoothed_residuals = obs - yl @ Z.T
                etahat = np.zeros((nt,n_shocks))
            
        else:
            
            # We apply Kalman filter for stationary and non-stationary models
            bUnivariate = False
            for t in range(T):
                if t < nt:
                    z = obs[t]
                    if model.FILTER.value == FilterAlgorithm.Durbin_Koopman.value:
                        ft,filtered,res,P,K,S,Sinv,loglk = \
                            DK_Filter(x=ft,xtilde=filtered,y=z,v=res,T=F1,Z=Z,P=P,Q=Q,H=Hm,K=K,C=C,ind_non_missing=ind_non_missing[t],t=t)          
                        y[t+1] = filtered
                        yt[t+1] = ft
                    elif model.FILTER.value == FilterAlgorithm.Non_Diffuse_Filter.value:                        
                        ft,filtered,res,P,P1,K,S,Sinv,bUnivariate,loglk = \
                            DK_Non_Diffuse_Filter(x=ft,xtilde=filtered,y=z,T=F1,Z=Z,P=P,Q=Q,H=Hm,C=C,bUnivariate=bUnivariate,ind_non_missing=ind_non_missing[t],t=t) 
                        y[t+1]  = filtered
                        yt[t+1] = ft
                    
                    # Save arrays
                    residuals.append(np.copy(res))
                    lP.append(np.copy(P))
                    lP1.append(np.copy(P1))
                    lK.append(np.copy(K))
                    lS.append(np.copy(S))
                    lSinv.append(np.copy(Sinv))
                        
                else:
                    y[t+1] = F1 @ y[t] + C
    
########################################################      Kalman smoothing:
    
            if model.SMOOTHER is None:
                y = y[1:2+T0]
                
            else:
                       
                smoothed_residuals = []; etahat = []; yl = []
                r=np.zeros(n)
                for t in range(nt,0,-1):
                    Sinv = lSinv[t]
                    K    = lK[t]
                    res  = residuals[t]
                    if model.SMOOTHER.value == SmootherAlgorithm.BrysonFrazier.value:
                        if t == nt:
                            L = np.zeros(F1.shape)
                            l = np.zeros(y[t].shape)
                        ys,Ps,L,l = Bryson_Frazier_Smoother(x=y[t],u=res,F=F1,H=Z,K=K,Sinv=Sinv,P=lP[t],L=L,l=l)
                        res = obs[t-1] - Z @ ys
                        smoothed_residuals.append(res)
                    elif model.SMOOTHER.value == SmootherAlgorithm.Durbin_Koopman.value:
                        ys,r,res,eta,eps = DK_Smoother(x=yt[t-1],y=obs[t-1],T=F1,Z=Z,F=lS[t],iF=lSinv[t],P=lP[t],QRt=Qm@R.T,K=K,H=Hm,r=r,v=res,ind_non_missing=ind_non_missing[t-1],t=t)
                        smoothed_residuals.append(res)
                        etahat.append(eta)
                        
                    yl.append(ys)
                        
                yl = yl[::-1]
                for t in range(nt,T0+2):
                    yl.append(F1@yl[t-1]+C)
                yl = np.array(yl[:1+T0])
                y = y[1:2+T0]
                y[0] = yl[0] = np.nan  # Starting values 
                
                residuals = smoothed_residuals[::-1]
                epsilonhat = np.array(residuals)[:1+T0]
                etahat = etahat[::-1]
                etahat = np.array(etahat)[:1+T0]
                
        yy = []
        y = y @ U.T
        if yl is None:
            yl = np.copy(y)
        else:
            yl = yl @ U.T
        yy.append(y)
        yy.append(yl)
        
        # Save shock values
        if not model.SMOOTHER is None:
            model.options['shock_values'] = etahat
        
        elapsed = time() - t0
            
        return count,yy,epsilonhat,etahat,err,elapsed
    
    
def nonlinear_filter(model,T,periods,y0,Qm=None,Hm=None,obs=None,MULT=1,skip=0,missing_obs=None,ind_non_missing=None):
    """
    Apply Kalman filter to non-linear model.
    
    Parameters:
        :param model: Model object.
        :type model: Model.
        :param T: Time span.
        :type T: int.
        :param periods: Array of endogenous variables.
        :type periods: numpy.array.
        :param y0: Starting values of endogenous variables.
        :type y0: numpy.array
        :param Qm: Covariance matrix of errors of endogenous variables. 
        :type Qm: numpy.array.
        :param Hm: Covariance matrix of errors of measurement variables.
        :type Hm: numpy.array.
        :param obs: Measurements.
        :type obs: list.
        :param MULT: Multiplier of the terminal time range.  If set greater than one than 
                     the solution will be computed for this extended time range interval.
        :type MULT: int.
        :param missing_obs: Matrix of logical values with True or False value depending on observation missing or not.
        :type missing_obs: Matrix of bools.
        :returns: Filtered solution.
    """
    from snowdrop.src.numeric.solver.nonlinear_solver import simulate
    from snowdrop.src.model.settings import FilterAlgorithm
    from snowdrop.src.model.settings import PriorAssumption
    from snowdrop.src.numeric.solver.BinderPesaran import getMatrices
 
    t0 = time()
    
    if Qm is None or Hm is None or obs is None:
        count,yy,yyIter,err,elapsed = simulate(model=model,T=T,periods=periods,y0=y0,order=1)
        return count,yy,None,None,err,elapsed
    
    steady_state = model.steady_state
    if steady_state is None:
       steady_state = y0
    
    TOLERANCE = 1.e-6
    NITERATIONS = 100
    err = 1.0; count = 0
    
    T0  = T
    T   = int(T*MULT)
    var = model.calibration['variables']
    var_names = model.symbols['variables']
    n,  = var.shape
    params = model.calibration['parameters']
    shock_var = model.symbols['shocks']
    n_shocks = len(shock_var)
    #variables = model.symbols['variables']
    meas_variables = model.symbols['measurement_variables']
    nm = len(meas_variables)
    K = np.zeros((n,n))
    
    Nd = model.max_lead_shock - model.min_lag_shock
       
    if 'distribution' in model.options:
        process = model.options['distribution']
        shock_values = process.simulate(1,1+T)
    else:
        if 'shock_values' in model.options:
            shock_values = model.options.get('shock_values')
            if isinstance(shock_values,list) and len(shock_values)==1:
                if isinstance(shock_values[0],list):
                   shock_values = shock_values[0] 
        else:
            shock_values = np.zeros(n_shocks)
    shock_values  = np.array(shock_values)     
    meas_shocks   = model.symbols['measurement_shocks'] 
    n_meas_shocks = len(meas_shocks)
        
    # Get reference to measurement function
    f_measurement = model.functions['f_measurement']
    bHasAttr_meas = hasattr(f_measurement,"py_func")
    
    # Get reference to RHS
    f_rhs = model.functions["f_rhs"]
    bHasAttr_rhs = hasattr(f_rhs,"py_func")
    
    # Get length of observations
    nt = len(obs)
    
    # initialize variables
    etahat=None;epsilonhat=None;res=None
    yy = []
        
    rowStable,colStable,rowUnstable,colUnstable = getStableUnstableRowsColumns(model)             
    
    # Iterate until solution converges
    y = np.zeros((T+2,n)); yprev = np.copy(y); yt = np.zeros((T+2,n))
    while (err > TOLERANCE and count < NITERATIONS):
        filtered = np.copy(y0); ft = np.copy(y0)
        y[:] = yt[:] = y0
        count += 1
            
        # initialize lists
        log_likelihood=[];residuals=[None];lK=[None];lP=[None];lS=[None];lSinv=[None]
        S=None;Sinv=None;loglk=0
        
        meas_params = model.calibration['measurement_parameters']
        meas_var = np.zeros(n+nm+n_shocks)
        if bHasAttr_meas:
            meas_const,meas_jacob = f_measurement.py_func(meas_var,meas_params,order=1)
        else:
            meas_const,meas_jacob = f_measurement(meas_var,meas_params,order=1)
        obs -= meas_const
        Z = -meas_jacob[:,:n]
        mf = np.array(np.nonzero(Z))
                
        F,C,R = getMatrices(model=model,n=n,t=0,y=y)
        
        Q = 0    
        for i in range(1+model.max_lead_shock-model.min_lag_shock):
            R1 = R[:n,i*n_shocks:(1+i)*n_shocks]
            Q += R1 @ Qm @ R1.T
        Q = 0.5*(Q+Q.T)
            
        P = np.copy(Q)
        Pstar = np.copy(P)
        
        # Get starting values of matrices Pstar, Pinf
        if model.PRIOR is None:
            Pinf = 1.e6*np.eye(n)
        elif model.PRIOR.value == PriorAssumption.Diffuse.value:
            # Diffuse intial condition for covariance matrix
            from snowdrop.src.numeric.filters.utils import compute_Pinf_Pstar
            Pinf,Pstar = compute_Pinf_Pstar(mf=mf,T=F,R=R,Q=Qm,N=Nd,n_shocks=n_shocks)            
            P = np.copy(Pstar)
        elif model.PRIOR.value == PriorAssumption.StartingValues.value:
            Pinf  = 1.e6*np.eye(n) 
        elif model.PRIOR.value == PriorAssumption.Equilibrium.value:
            # Lyapunov equation for stable part of covariance matrix
            from snowdrop.src.numeric.solver.solvers import lyapunov_solver 
            Pinf = np.zeros((n,n))
            P = lyapunov_solver(T=F[np.ix_(rowStable,colStable)],R=R[rowStable],Q=Qm,N=Nd,n_shocks=n_shocks,options="discrete")
            Pstar[np.ix_(rowStable,colStable)] = P
            P = np.copy(Pstar)
        elif model.PRIOR.value == PriorAssumption.Asymptotic.value:
            # Riccati discrete-time equation
            from scipy.linalg import solve_discrete_are 
            Pstar = solve_discrete_are(a=F.T,b=Z.T,q=Q,r=Hm)
            P = np.copy(Pstar)
            Pinf = 1.e6*np.eye(n)
                
        # Make sure covariance matrix is symmetric  
        P = 0.5*(P+P.T)
        Pstar = 0.5*(Pstar+Pstar.T)  
        Pinf  = 0.5*(Pinf+Pinf.T)
    
        ### Kalmnan filtering
        if model.FILTER.value == FilterAlgorithm.Diffuse.value:
            F,C,R = getMatrices(model=model,n=n,y=steady_state)
                    
            from snowdrop.src.numeric.filters.diffuse import diffuse_filter
            alphahat,epsilonhat,etahat,atilde,P1,aK,PK,decomp,V,N,dLIK,log_likelihood = \
                diffuse_filter(T=F,Z=Z,R=R[:n],Q=Qm,H=Hm,Y=obs,C=C,a0=filtered,pp=nm,mm=n,Ts=nt, \
                               Nd=Nd,n_shocks=n_shocks,data_index=ind_non_missing,Pinf1=Pinf,Pstar1=Pstar, \
                               decomp_flag=False,state_uncertainty_flag=False)                    
            epsilonhat = epsilonhat.T[1:2+T0]
            etahat = etahat.T[1:2+T0]
            y = np.array(atilde[:n]).T
            y = y[:2+T0]
            y[0] *= np.nan
            if model.SMOOTHER is None:
                yl = np.copy(y)
            else:
                yl = np.array(alphahat[:n]).T
                yl = yl[:2+T0] 
                yl[0] *= np.nan
                
        elif model.FILTER.value == FilterAlgorithm.Particle.value:
            F,C,R = getMatrices(model=model,n=n,y=steady_state)
                    
            from snowdrop.src.numeric.filters import kalman
            ss_model = kalman.MVLinearGauss(F=F,G=Z,mu0=y0,covX=Q,covY=Hm)
            kf = kalman.Kalman(ssm=ss_model,data=np.nan_to_num(obs))
            kf.filter()
            y = [kf.filt[i].mean for i in range(nt)]
            y[0] *= np.nan
            
            if model.SMOOTHER is None:
                yl = np.copy(y)
                
            if not model.SMOOTHER is None:
                kf.smoother()
                yl = [kf.smth[i].mean for i in range(nt)]
                yl = np.array(yl)
                smoothed_residuals = obs - yl @ Z.T
                etahat = np.zeros((nt,n_shocks)) 
                smoothed_residuals = smoothed_residuals[1:]
                etahat = etahat[1:]
                yl[0] *= np.nan
                
        elif model.FILTER.value == FilterAlgorithm.Unscented.value:
            from filterpy import kalman as kf
  
            def fx(x,dt=0.1):
                 """State transition function predicts next state."""
                 z = np.concatenate([x,x,x,np.zeros(n_shocks)])
                 if bHasAttr_rhs:
                     xn = f_rhs.py_func(z,p=params,order=0)
                 else:
                     xn = f_rhs(z,p=params,order=0)
                 return xn
            
            def hx(x):
                """Measurement function maps state variables into a measurement variables."""
                z = [x[i] if i in ind_var else 0 for i in range(n)] + [0]*(nm+n_meas_shocks)
                if bHasAttr_meas:
                    f = -f_measurement.py_func(z,meas_params,order=0)
                else:
                    f = -f_measurement(z,meas_params,order=0)
                return f
            
            if count == 1:
                from snowdrop.src.numeric.filters.utils import getCovarianceMatrix,getEndogVarInMeasEqs
                
                Pm = getCovarianceMatrix(Qm,shock_var,model.symbolic.equations,n)
                ind_var,_ = getEndogVarInMeasEqs(var_names, model.symbolic.measurement_equations)
                
            # Create sigma points to use in the filter. This is standard for Gaussian processes
            points = kf.MerweScaledSigmaPoints(n=n,alpha=2,beta=3,kappa=0)
            #sigmas = points.sigma_points(y0, Pm)
            kf = kf.UnscentedKalmanFilter(dim_x=n,dim_z=nm,dt=0.1,fx=fx,hx=hx,points=points)
            kf.x = y0 # initial state
            kf.P = Pm # initial uncertainty
            kf.R = 0.5*Hm + 0.5*np.diag([np.random.normal(loc=0,scale=np.sqrt(Hm[i]))**2 for i in range(nm)])
            kf.Q = Q
            xs = []
            for z in obs:
                 kf.predict()
                 kf.update(z)
                 xs.append(kf.x)
            y = np.array(xs)
            
            if model.SMOOTHER is None:
                yl = y
            else:
                mu,cov = kf.batch_filter(obs)
                xs,Ps,Ks = kf.rts_smoother(mu,cov)
                yl = xs
            err = 0
                
                
        else: 
            
            for t in range(T):
                if t < nt:
                    # F,C,R = getMatrices(model=model,n=n,t=t,y=y)
                    # Q = 0
                    # for i in range(1+model.max_lead_shock-model.min_lag_shock):
                    #     R1 = R[:n,i*n_shocks:(1+i)*n_shocks]
                    #     Q += R1 @ Qm @ R1.T
                    # Q = 0.5*(Q+Q.T)
                                        
                    # Filter endogenous variables
                    z = obs[t]
                    if model.FILTER.value == FilterAlgorithm.Durbin_Koopman.value:
                        ft,filtered,res,P,K,S,Sinv,loglk = \
                            DK_Filter(x=ft,xtilde=filtered,y=z,v=res,T=F,Z=Z,P=P,Q=Q,H=Hm,K=K,C=C,ind_non_missing=ind_non_missing[t],t=t)          
                        y[t+1] = filtered
                        yt[t+1] = ft
                    elif model.FILTER.value == FilterAlgorithm.Non_Diffuse_Filter.value:                        
                        ft,filtered,res,P,P1,K,S,Sinv,bUnivariate,loglk = \
                            DK_Non_Diffuse_Filter(x=ft,xtilde=filtered,y=z,T=F,Z=Z,P=P,Q=Q,H=Hm,C=C,bUnivariate=False,ind_non_missing=ind_non_missing[t],t=t) 
                        y[t+1]  = filtered
                        yt[t+1] = ft
                    
                    
                    # Save arrays
                    log_likelihood.append(loglk)
                    residuals.append(np.copy(res))
                    lP.append(np.copy(P))
                    lK.append(np.copy(K))
                    lS.append(np.copy(S))
                    lSinv.append(np.copy(Sinv))
                else:
                    y[t+1] = F @ y[t] + C
                
            y = y[1:2+T0]
            y[0] *= np.nan
            ### Kalman smoothing:
            if model.SMOOTHER is None:
                yl = np.copy(y)
                
            else:
                
                smoothed_residuals = []; etahat = []; yl = []
                r=np.zeros(n)
                for t in range(nt,0,-1):
                    Sinv = lSinv[t]
                    K    = lK[t]
                    res  = residuals[t]
                    if model.SMOOTHER.value == SmootherAlgorithm.BrysonFrazier.value:
                        if t == nt:
                            L = np.zeros(F.shape)
                            l = np.zeros(y[t].shape)
                        ys,Ps,L,l = Bryson_Frazier_Smoother(x=y[t-1],u=res,F=F,H=Z,K=K,Sinv=Sinv,P=lP[t],L=L,l=l)
                        res = obs[t-1] - Z @ ys
                        smoothed_residuals.append(res)
                    elif model.SMOOTHER.value == SmootherAlgorithm.Durbin_Koopman.value:
                        ys,r,res,eta,eps = DK_Smoother(x=yt[t-1],y=obs[t-1],T=F,Z=Z,F=lS[t],iF=lSinv[t],P=lP[t],QRt=Qm@R.T,K=K,H=Hm,r=r,v=res,ind_non_missing=ind_non_missing[t-1],t=t)
                        smoothed_residuals.append(res)
                        etahat.append(eta)
                    yl.append(ys)
                     
                yl = yl[::-1]
                for t in range(nt,T0+2):
                    yl.append(F@yl[t-1]+C)
                yl = np.array(yl[:1+T0])
                yl[0] *= np.nan
                
                residuals = smoothed_residuals[::-1]
                epsilonhat = np.array(residuals)[:1+T0]
                etahat = etahat[::-1]
                etahat = np.array(etahat)[:1+T0]
            
        err = la.norm(yprev[:len(y)]-y)/max(1.e-10,la.norm(y))
        yprev = np.copy(y)
        #print(count,err,norm(y))
        
    yy.append(y)
    yy.append(yl)
    
    # Save shock values
    if not model.SMOOTHER is None:
        model.options['shock_values'] = etahat
    
    elapsed = time() - t0
    
    return count,yy,epsilonhat,etahat,err,elapsed
    
    
    