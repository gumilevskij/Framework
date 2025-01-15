# -*- coding: utf-8 -*- 
""" 
Created on Wed Jan 23 12:52:44 2019 
 
@author: AGoumilevski 
""" 
import os,sys 
import numpy as np 
from time import time 
from scipy.optimize import minimize,Bounds 
import warnings 

path = os.path.dirname(os.path.abspath(__file__))
working_dir = os.path.abspath(path + "/../../../..")
sys.path.append(working_dir)
os.chdir(working_dir)

from snowdrop.src.misc.termcolor import cprint 
from snowdrop.src.numeric.solver.util import getParameters  
from snowdrop.src.numeric.solver.util import getCovarianceMatrix 
from snowdrop.src.preprocessor.function import get_function_and_jacobian 
from snowdrop.src.utils.distributions import getHyperParameters 
from snowdrop.src.utils.distributions import pdf 
from snowdrop.src.model.settings import PriorAssumption,FilterAlgorithm
from snowdrop.src.numeric.filters.filters import DK_Filter as dk_filter
from snowdrop.src.numeric.filters.filters import Durbin_Koopman_Non_Diffuse_Filter as dk_non_diffuse_filter 
import snowdrop.src.numeric.solver.linear_solver as ls 
from snowdrop.src.utils.prettyTable import PrettyTable 
 
it = 0; itr = 0 
F,B,R,Q,Qm_,Hm_ = None,None,None,None,None,None 
LARGE_NUMBER = 1e6
est_shocks_names = [] 
 
ESTIMATE_PARAMETERS_STDS = True 
 
 
def run(y0,model,T,Qm=None,Hm=None,obs=None,steady_state=None,
        ind_non_missing=None,fit_data_only=False,
        estimate_Posterior=True,estimate_ML=False,
        algorithm="SLSQP",linearized=True): 
    """ 
    Estimates linear/nonlinear model parameters given measurement data. 
     
    Parameters: 
        :param y0: Starting values of endogenous variables. 
        :type y0: numpy.array 
        :param model: Model object. 
        :type model: Model. 
        :param T: Time span. 
        :type T: int. 
        :param Qm: Covariance matrix of errors of endogenous variables.  
        :type Qm: numpy.array. 
        :param Hm: Covariance matrix of errors of measurement variables. 
        :type Hm: numpy.array. 
        :param obs: List of measurement variables. 
        :type obs: list. 
        :param ss: List of steady states. 
        :type s: list. 
        :param ind_non_missing: Indices of non-missing observations. 
        :type ind_non_missing: list. 
        :param fit_data_only: If True calibrate model parameters by minimizing sum of squared errors of model fit to data.   
                         Otherwise, calibrate model by maximizing sum of the prior likelihood of model parameters and the likelihood of model fit to data. 
        :type fit_data_only: bool. 
        :param estimate_ML: If True estimate maximum likelihood only. 
        :type estimate_ML: bool. 
        :param linearized: If True model transition matrices are computed at steady state.  Use this option to speed up calculations. 
        :type linearized: bool. 
        :returns: Numerical solution. 
    """ 
    
     
    global est_shocks_names 
     
    t0 = time()  
    var = model.calibration['variables'] 
    n, = var.shape 
    shocks = model.symbols['shocks'] 
    n_shocks = len(shocks) 
    Qm_ = np.copy(Qm) 
    Hm_ = np.copy(Hm) 
    est_stds,params_std = None,None  
  
    meas_variables = model.symbols['measurement_variables'] 
    nm = len(meas_variables) 
         
    if 'measurement_shocks' in model.symbols: 
        meas_shocks = model.symbols['measurement_shocks'] 
        n_meas_shocks = len(meas_shocks) 
    else: 
        meas_shocks = [] 
        n_meas_shocks = 0 
         
    # Get reference to measurement function 
    f_measurement = model.functions['f_measurement'] 
    bHasAttr  = hasattr(f_measurement,"py_func")
    nt = len(obs) 
    param_names = model.symbols['parameters'] 
    params = getParameters(model=model) 
     
    if len(model.priors) == 0: 
        elapsed = time() - t0 
        return elapsed,params 
        
    # Find measurement equations jacobian 
    meas_params = model.calibration['measurement_parameters'] 
    meas_var = np.zeros(n+nm+n_shocks+n_meas_shocks) 
    if bHasAttr:
        meas_const,meas_jacob = f_measurement.py_func(meas_var,meas_params,order=1) 
    else:
        meas_const,meas_jacob = f_measurement(meas_var,meas_params,order=1)    
    obs -= meas_const 
    Z = -meas_jacob[:,:n]                
         
    # Find prior parameters distribution   
    param_index = [] 
    if not model.priors is None:        
        for i,k in enumerate(param_names): 
            if k in model.priors: 
                param_index.append(i) 
                 
    # Find standard deviations of calibrated shocks 
    est_shocks_std = [] 
    cal = {k:v for k,v in model.priors.items() if k.startswith("std_")} 
    for i,v in enumerate(shocks+meas_shocks): 
        std_v = "std_"+v 
        if std_v in cal: 
            est_shocks_names.append(std_v) 
            est_shocks_std.append(cal[std_v]) 
         
    def func(ytm,yt,ytp,params): 
        z = np.vstack((ytm,yt,ytp)) 
        D,jacob = get_function_and_jacobian(model,params=params,y=z,order=1)   
        F = jacob[:,0:n] 
        C = jacob[:,n:2*n] 
        L = jacob[:,2*n:3*n] 
        jac = jacob[:,3*n:] 
        return D,F,C,L,jac 
                 
     
    ################################################### LINEAR MODEL 
    if model.isLinear: 
 
        # Define objective function 
        def fobj(obj_parameters): 
            global it, est_shocks_names 
            it += 1 
            K=None; S=None; Sinv=None 
            # Initialize log-likelihood 
            log_prior_likelihood = 0; log_posterior_likelihood = 0; res = 0; residual = None 
            parameters = np.copy(params) 
             
            # Find prior parameters distribution 
            for i,index in enumerate(param_index): 
                par = obj_parameters[i] 
                parameters[index] = par 
                prior = model.priors[param_names[index]] 
                distr = prior['distribution'] 
                pars  = np.copy(prior['parameters']) 
                lower_bound = float(pars[1]) 
                upper_bound = float(pars[2]) 
                #print(par,lower_bound,upper_bound)
                if par < lower_bound or par > upper_bound: 
                    return  (LARGE_NUMBER + it) 
                else: 
                    # Get distribution hyperparameters 
                    if not distr.endswith("_hp"): 
                        pars[3],pars[4] = getHyperParameters(distr=distr,mean=pars[3],std=pars[4]) 
                    x,b   = pdf(distr,par,pars) 
                    if not b:
                        return  (LARGE_NUMBER + it)
                    log_prior_likelihood += np.log(max(1.e-10,x))
                 
            # Find standard deviations of shocks  
            calib = {} 
            npar = len(param_index) 
            for i,name in enumerate(est_shocks_names): 
                par = obj_parameters[i+npar] 
                calib[name] = obj_parameters[i+npar] 
                prior = model.priors[name]
                distr = prior['distribution'] 
                pars  = np.copy(prior['parameters'])
                lower_bound = float(pars[1]) 
                upper_bound = float(pars[2]) 
                if par < lower_bound or par > upper_bound: 
                    return  (LARGE_NUMBER + it) 
                else: 
                    # Get distribution hyperparameters 
                    if not distr.endswith("_hp"): 
                        pars[3],pars[4] = getHyperParameters(distr=distr,mean=pars[3],std=pars[4]) 
                    x,b   = pdf(distr,par,pars)
                    if not b:
                        return  (LARGE_NUMBER + it)
                    log_prior_likelihood += np.log(max(1.e-10,x))
              
            # Set values of covariance matrix 
            Qm,Hm = getCovarianceMatrix(Qm_,Hm_,calib,shocks,meas_shocks)   
                 
            if np.isnan(log_prior_likelihood) or np.isinf(log_prior_likelihood): 
                return (LARGE_NUMBER + it) 
                                      
            # Solve linear model 
            try: 
                model.solved=False 
                ls.solve(model=model,p=parameters,steady_state=np.zeros(n),suppress_warnings=True) 
                # State transition matrix 
                F = np.copy(model.linear_model["A"]) 
                F1 = F[:n,:n] 
                # Array of constants 
                C = np.copy(model.linear_model["C"][:n]) 
                # Matrix of coefficients of shocks 
                R = model.linear_model["R"][:n] 
                # Initialize covariance matrix 
                P = np.copy(Pstar) 
                Q = 0 
                for i in range(1+model.max_lead_shock-model.min_lag_shock): 
                    R1 = R[:n,i*n_shocks:(1+i)*n_shocks] 
                    Q += np.real(R1 @ Qm @ R1.T) 
                     
                bUnivariate = False 
                ft = filtered = np.copy(y0) 
             
                for t in range(nt): 
                    if not model.FILTER is None and model.FILTER.value == FilterAlgorithm.Non_Diffuse_Filter.value:
                        ft,filtered,residual,P,_,K,S,Sinv,bUnivariate,loglk  = \
                            dk_non_diffuse_filter(x=ft,xtilde=filtered,y=obs[t],T=F1,Z=Z,P=P,Q=Q,H=Hm,C=C,bUnivariate=bUnivariate,ind_non_missing=ind_non_missing[t]) 
                    elif model.FILTER.value == FilterAlgorithm.Durbin_Koopman.value: 
                        ft,filtered,residual,P,K,S,Sinv,loglk = \
                            dk_filter(x=ft,xtilde=filtered,y=obs[t],v=residual,T=F1,Z=Z,P=P,H=Hm,Q=Q,K=K,C=C,ind_non_missing=ind_non_missing[t],t=t)
                     
                    res += np.sum(residual**2) 
                    if np.isnan(loglk): 
                        return (LARGE_NUMBER + it) 
                    else: 
                        log_posterior_likelihood += loglk 
                         
                # update progress bar 
                if it%10 == 0: 
                    sys.stdout.write("\b") 
                    sys.stdout.write("..") 
                    sys.stdout.flush() 
                    
                    
                if fit_data_only: 
                    likelihood = -res  
                elif estimate_ML: 
                    likelihood = log_posterior_likelihood 
                elif estimate_Posterior: 
                    likelihood = log_prior_likelihood + log_posterior_likelihood 
                    
                if it%400 == 0:  
                    cprint(f"\nIteration: {it}, Likelihood: {likelihood:.2f}","blue")
                 
                #print('prior likelihood: ',-log_prior_likelihood, 'posterior likelihood: ',-log_posterior_likelihood) 
                return -likelihood 
             
            except: 
                #print(log_posterior,params) 
                return (LARGE_NUMBER + it) 
         
    ################################################### NON-LINEAR MODEL 
    else: 
         
        from numpy.linalg import norm 
        from snowdrop.src.numeric.solver.BinderPesaran import getMatrices 
         
        # Define objective function 
        def fobj(obj_parameters): 
            global it,F,C,R,Q 
            global it, est_shocks_names 
            it += 1 
            K=None; S=None; Sinv=None 
            # Initialize log-likelihood 
            log_prior_likelihood = 0; log_posterior_likelihood = 0; res = 0; residual = None 
            parameters = np.copy(params) 
             
            # Find prior parameters distribution 
            for i,index in enumerate(param_index): 
                par = obj_parameters[i] 
                parameters[index] = par 
                prior = model.priors[param_names[index]] 
                distr = prior['distribution'] 
                pars  = np.copy(prior['parameters']) 
                lower_bound = float(pars[1]) 
                upper_bound = float(pars[2]) 
                #print(par,lower_bound,upper_bound)
                if par < lower_bound or par > upper_bound: 
                    return  (LARGE_NUMBER + it) 
                else: 
                    # Get distribution hyperparameters 
                    if not distr.endswith("_hp"): 
                        pars[3],pars[4] = getHyperParameters(distr=distr,mean=pars[3],std=pars[4]) 
                    x,b   = pdf(distr,par,pars) 
                    if not b:
                        return  (LARGE_NUMBER + it)
                    log_prior_likelihood += np.log(max(1.e-10,x))
                 
            # Find standard deviations of shocks  
            calib = {} 
            npar = len(param_index) 
            for i,name in enumerate(est_shocks_names): 
                par = obj_parameters[i+npar] 
                calib[name] = obj_parameters[i+npar] 
                prior = model.priors[name]
                distr = prior['distribution'] 
                pars  = np.copy(prior['parameters'])
                lower_bound = float(pars[1]) 
                upper_bound = float(pars[2]) 
                if par < lower_bound or par > upper_bound: 
                    return  (LARGE_NUMBER + it) 
                else: 
                    # Get distribution hyperparameters 
                    if not distr.endswith("_hp"): 
                        pars[3],pars[4] = getHyperParameters(distr=distr,mean=pars[3],std=pars[4]) 
                    x,b   = pdf(distr,par,pars)
                    if not b:
                        return  (LARGE_NUMBER + it)
                    log_prior_likelihood += np.log(max(1.e-10,x))
              
            # Set values of covariance matrix 
            Qm,Hm = getCovarianceMatrix(Qm_,Hm_,calib,shocks,meas_shocks)   
                 
            if np.isnan(log_prior_likelihood) or np.isinf(log_prior_likelihood): 
                return (LARGE_NUMBER + it) 
             
            count = 0; max_f = 1.0; bUnivariate = False 
            TOLERANCE = 1.e-6; NITERATIONS = 1 if linearized else 100 
            y = np.zeros((T+2,n))
            y[:] = y0; yprev = np.copy(y) 
             
            try: 
                while (max_f > TOLERANCE and count < NITERATIONS): 
                    count += 1 
                    filtered = yprev[0]; ft = np.copy(filtered); res = 0 
                    # Initialize covariance matrix 
                    P = np.copy(Pstar) 
                    for t in range(nt): 
                        if t==0 or not linearized: 
                            F,C,R = getMatrices(model=model,n=n,y=y0) 
                            Q = 0 
                            for i in range(1+model.max_lead_shock-model.min_lag_shock): 
                                R1 = R[:n,i*n_shocks:(1+i)*n_shocks] 
                                Q += R1 @ Qm @ R1.T 
                            # Compute constant matrix 
                            #C = yprev[t+1] - F @ yprev[t] + C 
                        
                        # Apply Kalman filter to find log-likelyhood 
                        if not model.FILTER is None and model.FILTER.value == FilterAlgorithm.Non_Diffuse_Filter.value:
                            ft,filtered,residual,P,_,K,S,Sinv,bUnivariate,loglk  = \
                                dk_non_diffuse_filter(x=ft,xtilde=filtered,y=obs[t],T=F1,Z=Z,P=P,Q=Q,H=Hm,C=C,bUnivariate=bUnivariate,ind_non_missing=ind_non_missing[t]) 
                        elif model.FILTER.value == FilterAlgorithm.Durbin_Koopman.value: 
                            ft,filtered,residual,P,K,S,Sinv,loglk = \
                                dk_filter(x=ft,xtilde=filtered,y=obs[t],v=residual,T=F1,Z=Z,P=P,H=Hm,Q=Q,K=K,C=C,ind_non_missing=ind_non_missing[t],t=t)
                         
                        res += np.sum(residual**2) 
                        if np.isnan(loglk): 
                            return (LARGE_NUMBER + it) #np.inf 
                        else: 
                            log_posterior_likelihood += loglk 
                        y[t+1] = filtered 
                         
                    max_f = norm(yprev-y)/max(1.e-10,norm(y)) 
                    yprev = np.copy(y) 
                         
                # update progress bar 
                if it%10 == 0: 
                    sys.stdout.write("\b") 
                    sys.stdout.write("..") 
                    sys.stdout.flush() 
                         
                if fit_data_only: 
                    likelihood = -res  
                elif estimate_ML: 
                    likelihood = log_posterior_likelihood 
                elif estimate_Posterior: 
                    likelihood = log_prior_likelihood + log_posterior_likelihood 
                 
                if it%400 == 0:  
                    cprint(f"\nIteration: {it}, Likelihood: {likelihood:.2f}","blue")
                 
                #print('prior likelihood: ',-log_prior_likelihood, 'posterior likelihood: ',-log_posterior_likelihood) 
                return -likelihood 
             
            except: 
                #print(log_prior_likelihood,log_likelihood,params) 
                return (LARGE_NUMBER + it) 
          
    print("\nEstimating model parameters...") 
     
    if linearized and not model.isLinear: 
        from snowdrop.src.numeric.solver.BinderPesaran import getMatrices
        global F,C,R,Q 
        ss = model.steady_state if bool(model.steady_state) else model.calibration["variables"]
        F,C,R = getMatrices(model=model,n=n,y=ss) 
        Q = 0 
        for i in range(1+model.max_lead_shock-model.min_lag_shock): 
            R1 = R[:n,i*n_shocks:(1+i)*n_shocks] 
            Q += R1 @ Qm @ R1.T 
    all_parameters = np.copy(params) 
     
    # Find parameters bounds 
    lower = []; upper = []; initial_values = []; mp = {}
    for i,index in enumerate(param_index): 
        mp[param_names[index]] = i
        prior = model.priors[param_names[index]] 
        pars  = np.copy(prior['parameters'])
        initial_value = float(pars[0]) 
        lb = float(pars[1]) 
        ub = float(pars[2]) 
        initial_values.append(initial_value) 
        lower.append(lb) 
        upper.append(ub) 
         
    for shock_name in est_shocks_names: 
        prior = model.priors[shock_name] 
        pars  = np.copy(prior['parameters']) 
        initial_value = float(pars[0]) 
        lb = float(pars[1]) 
        ub = float(pars[2]) 
        initial_values.append(initial_value) 
        lower.append(lb) 
        upper.append(ub) 
         
    lower = np.array(lower) 
    upper = np.array(upper) 
    bounds = Bounds(lower,upper) 
     
    ### Get covariance matrices 
    model.solved=False 
    ls.solve(model=model,p=params,steady_state=np.zeros(n),suppress_warnings=True) 
    # State transition matrix 
    F = np.copy(model.linear_model["A"]) 
    F1 = F[:n,:n] 
    # Array of constants 
    C = np.copy(model.linear_model["C"][:n]) 
    # Matrix of coefficients of shocks 
    R = model.linear_model["R"][:n] 
    Q = 0 
    for i in range(1+model.max_lead_shock-model.min_lag_shock): 
        R1 = R[:n,i*n_shocks:(1+i)*n_shocks] 
        Q += np.real(R1 @ Qm @ R1.T) 
         
    Nd = model.max_lead_shock - model.min_lag_shock 
    # Get starting values of matrix P 
    if model.PRIOR is None: 
        Pstar = np.copy(Q) 
    elif model.PRIOR.value == PriorAssumption.StartingValues.value: 
        Pstar = np.copy(Q) 
    elif model.PRIOR.value == PriorAssumption.Diffuse.value: 
        from snowdrop.src.numeric.filters.utils import compute_Pinf_Pstar 
        mf = np.array(np.nonzero(Z)) 
        Pinf,Pstar = compute_Pinf_Pstar(mf=mf,T=F1,R=R,Q=Qm,N=Nd,n_shocks=n_shocks)  
    elif model.PRIOR.value == PriorAssumption.Equilibrium.value: 
        from snowdrop.src.numeric.solver.solvers import lyapunov_solver                      
        from snowdrop.src.numeric.solver.util import getStableUnstableRowsColumns 
        rowStable,colStable,rowUnstable,colUnstable = getStableUnstableRowsColumns(model,T=F1,K=C)                        
        Pstar = lyapunov_solver(T=F1[np.ix_(rowStable,colStable)],R=R[rowStable],Q=Qm,N=Nd,n_shocks=n_shocks,options="discrete")                
    elif model.PRIOR.value == PriorAssumption.Asymptotic.value: 
        # Riccati equation 
        from scipy.linalg import solve_discrete_are  
        Pstar = solve_discrete_are(a=F1.T,b=Z.T,q=Q,r=Hm) 
 
    with warnings.catch_warnings(): 
        warnings.simplefilter("ignore") 
         
        # METHODS :'SLSQP','Powell','CG','BFGS','Newton-CG' ,'L-BFGS-B','TNC','COBYLA','trust-constr','dogleg','trust-ncg','trust-exact','trust-krylov' 
        # Bounds on variables are available for: Nelder-Mead, L-BFGS-B, TNC, SLSQP, Powell, and trust-constr methods 
        if ESTIMATE_PARAMETERS_STDS: 
            calibration_results = minimize(fun=fobj,x0=initial_values,method="L-BFGS-B",bounds=bounds,tol=1.e-7,options={'disp':False,'maxiter':100000}) 
            # if hasattr(calibration_results,"hess_inv"):
            #     hess_inv = calibration_results.hess_inv.todense() 
            #     if not np.all(np.linalg.eigvals(hess_inv) > 0):
            #         cprint("The hessian matrix not positive semi-definite: \ntry to change the initial values of the parameters!","red")
            #     params_std = [np.sqrt(abs(hess_inv[i,i])) if not hess_inv[i,i] == 0 else np.inf for i in range(hess_inv.shape[0])]     
            # 
            # Brute force calculations  
            nc = len(calibration_results.x)
            hess = np.zeros(nc)
            f = np.zeros(3)
            delta = 1.e-4
            for i in range(nc):
                x = np.copy(calibration_results.x)
                for m in range(3):
                    x[i] += delta*(m-1)
                    f[m] = fobj(x)
                hess[i] = (f[2]-2*f[1]+f[0])/delta**2
            params_std = [1./np.sqrt(abs(x)) if not x == 0 else np.nan for x in hess]     
                  
        else: 
            # L-BFGS-B, TNC, SLSQP, Powevll, and trust-constr  
            calibration_results = minimize(fun=fobj,x0=initial_values,method=algorithm,bounds=bounds,tol=1.e-6,options={'disp':False,'maxiter':10000})  
     
        if not calibration_results.success:
            it = 0
            cprint(f"\nConstrained minimization failed: {calibration_results.message}","red") 
            print("Running un-constrained minimization...") 
            calibration_results = minimize(fun=fobj,x0=initial_values,method='SLSQP',bounds=None,tol=1.e-6,options={'disp':False,'maxiter':10000}) 
         
    if calibration_results.success: 
        cprint(f"\nNumber of iterations performed by the optimizer: {calibration_results.nit}, function value: {-calibration_results.fun:.2f}","blue") 
    else: 
        print(calibration_results) 
        raise Exception("\nModel calibration failed.") 
         
    #sys.stdout.write("\n") # this ends the progress bar 
    elapsed = time() - t0 
    elapsed = np.round(elapsed,2) 
    new_params = calibration_results.x 
    all_parameters[param_index] = new_params[:len(param_index)] 
     
    if params_std is None: 
        header = ['Name','Starting Value','Estimate'] 
    else: 
        header = ['Name','Starting Value','Estimate','Std.']     
    pTable = PrettyTable(header) 
    pTable.float_format = '7.5' 
     
    # Estimated parameters 
    def_params = dict(zip([x for i,x in enumerate(param_names) if i in param_index],initial_values[:len(param_index)]))     
    for i in param_index: 
        name = param_names[i] 
        if params_std is None or not name in mp: 
            row = [name,def_params[name],all_parameters[i]] 
        else: 
            row = [name,def_params[name],all_parameters[i],params_std[mp[name]]]
        pTable.add_row(row) 
     
    # Estmated standard deviations of shocks 
    if bool(est_shocks_names): 
        def_stds = dict(zip(est_shocks_names,initial_values[len(param_index):])) 
        est_shocks_std = new_params[len(param_index):] 
        est_stds = dict(zip(est_shocks_names,est_shocks_std)) 
        if not params_std is None: 
            est_stds_stds = dict(zip(est_shocks_names,params_std[len(param_index):])) 
        for i,name in enumerate(est_shocks_names):
            if params_std is None: 
                row = [name,def_stds[name],est_stds[name]] 
            else: 
                row = [name,def_stds[name],est_stds[name],est_stds_stds[name]] 
            pTable.add_row(row) 
            # Get shocks standard deviations 
        Qest,Hest = getCovarianceMatrix(Qm_,Hm_,est_stds,shocks,meas_shocks)   
    else: 
        Qest,Hest = None,None 
         
    if fit_data_only: 
        cprint("\nEstimation Based on Data Fit","green")  
    elif estimate_ML: 
        cprint("\nMaximum Likelihood Estimation","green")  
    elif estimate_Posterior: 
        cprint("\nPosterior Maximum Likelihood Estimation","green") 
    else: 
        cprint("\nEstimated Parameters","green") 
    print(pTable) 
     
    # Update priors initial values 
    priors = model.priors 
    for index in param_index: 
        name = param_names[index] 
        pars = priors[name]['parameters'] 
        pars[0] = all_parameters[index] 
        priors[name]['parameters'] = pars 
                     
    print("\nEstimation time: {0} (seconds)".format(elapsed)) 
     
             
    return elapsed,all_parameters,est_stds,Qest,Hest,priors 
