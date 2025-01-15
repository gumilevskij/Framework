"""
Created on Fri Apr  5 13:09:34 2019

Curently implemented for linear models only.

@author: A.Goumilevski
"""

import os,sys
import numpy as np
import multiprocessing
from time import time
#from numeric.bayes.logp import logp
from snowdrop.src.misc.termcolor import cprint
from snowdrop.src.utils.distributions import pdf
from snowdrop.src.utils.distributions import getHyperParameters
from snowdrop.src.numeric.solver import linear_solver
from snowdrop.src.numeric.filters.filters import DK_Filter as dk_filter
from snowdrop.src.numeric.filters.filters import Durbin_Koopman_Non_Diffuse_Filter as dk_non_diffuse_filter 
from snowdrop.src.model.settings import SamplingAlgorithm
from snowdrop.src.model.settings import PriorAssumption,FilterAlgorithm
import snowdrop.src.numeric.solver.linear_solver as ls
from snowdrop.src.utils.prettyTable import PrettyTable

num_cores = multiprocessing.cpu_count()
num_chains = 3
PARALLEL = False # if True runs parallel sampling

count=0; nt=-1; n_=-1; n_shocks_=-1; ind_non_missing_=[]
model_=None; params=None; param_names=None; y0_=None; param_index=None; meas_shocks_=[]
obs_=None; Z_=None; Hm_=None; Qm_=None; pm_model=None; Pstar=None
pm_names=None; pm_params=None; priors=None; data_=None; est_shocks_names=[]; shocks_=[]
LARGE_NUMBER = 1.e10

path = os.path.dirname(os.path.abspath(__file__))
path_to_dir = os.path.join(path,'../../../graphs')

def prior_logp(p):
    """Function defining log of prior probability."""
    global model_,params,param_names,param_index,count
    log_likelihood = 0     
    try:
        b = True
        for i,index in enumerate(param_index):
            name  = param_names[index]
            prior = model_.priors[name]
            distr = prior['distribution']
            pars  = np.copy(prior['parameters'])
            x     = p[i]
            lower_bound = float(pars[1])
            upper_bound = float(pars[2])
            if x < lower_bound or x > upper_bound:
                return -LARGE_NUMBER - count
            # Get distribution hyperparameters
            if not distr.endswith("_hp"):
                pars[3],pars[4] = getHyperParameters(distr=distr,mean=pars[3],std=pars[4])
            likelihood,b = pdf(distr,x,pars)
            if not b:
                return -LARGE_NUMBER - count
            if np.isnan(likelihood) or np.isinf(likelihood):
                log_likelihood = -LARGE_NUMBER - count
            elif likelihood > 0:
                log_likelihood += np.log(max(1.e-10,likelihood))
                                     
        # Find standard deviations of shocks 
        npar = len(param_index)
        for i,name in enumerate(est_shocks_names):
            par = p[i+npar]
            prior = model_.priors[name]
            distr = prior['distribution']
            pars  = prior['parameters']
            lower_bound = float(pars[1])
            upper_bound = float(pars[2])
            if par < lower_bound or par > upper_bound:
                return -LARGE_NUMBER - count
            else:
                # Get distribution hyperparameters
                if not distr.endswith("_hp"):
                    pars[3],pars[4] = getHyperParameters(distr=distr,mean=pars[3],std=pars[4])
                x,b   = pdf(distr,par,pars)
                if not b: 
                    return -LARGE_NUMBER - count
                log_likelihood += np.log(max(1.e-10,x))

    except:
        log_likelihood = -LARGE_NUMBER - count
            
    return log_likelihood


def likelihood_logp(parameters,stds):
    """Function defining log of likelihood."""
    global model_,count,n_,nt,n_shocks_,Pstar,shocks_,meas_shocks_
    global y0_,Qm_,obs_,Z_,Hm_,ind_non_missing_
      
    from snowdrop.src.numeric.solver.util import getCovarianceMatrix
    
    # Expected value of outcome
    log_likelihood = 0
    K = None; S = None; Sinv = None
    
    calib = {}
    for i,name in enumerate(est_shocks_names):
        calib[name] = stds[i]
             
    # Set values of covariance matrix
    Qm,Hm = getCovarianceMatrix(Qm_,Hm_,calib,shocks_,meas_shocks_)  
            
    # Solve linear model
    model_.solved=False
    linear_solver.solve(model=model_,p=parameters,steady_state=np.zeros(n_),suppress_warnings=True)
    # State transition matrix
    F = model_.linear_model["A"]
    F1 = F[:n_,:n_]
    # Array of constants
    C = model_.linear_model["C"]
    C = C[:n_]
    # Matrix of coefficients of shocks
    R = model_.linear_model["R"]
    R = R[:n_]

    Q = 0
    for i in range(1+model_.max_lead_shock-model_.min_lag_shock):
        R1 = R[:,i*n_shocks_:(1+i)*n_shocks_]
        Q += np.real(R1 @ Qm_ @ R1.T)
            
    bUnivariate = True
    y = yt = np.copy(y0_)
    
    # Get model likelihood by applying Kalman filter
    for t in range (nt):
        # Initialize covariance matrices
        if t == 0:
            P = np.copy(Pstar)
        # Apply Kalman filter to find log-likelyhood
        if not model_.FILTER is None and model_.FILTER.value == FilterAlgorithm.Non_Diffuse_Filter.value:
            yt,y,residual,P,_,K,S,Sinv,bUnivariate,loglk  = \
                dk_non_diffuse_filter(x=yt,xtilde=y,y=obs_[t],T=F1,Z=Z_,P=P,Q=Q,H=Hm,C=C,bUnivariate=bUnivariate,ind_non_missing=ind_non_missing_[t]) 
        elif model_.FILTER.value == FilterAlgorithm.Durbin_Koopman.value: 
            yt,y,residual,P,K,S,Sinv,loglk = \
                dk_filter(x=yt,xtilde=y,y=obs_[t],v=residual,T=F1,Z=Z_,P=P,H=Hm,Q=Q,K=K,C=C,ind_non_missing=ind_non_missing_[t],t=t)
 
        log_likelihood += loglk
        
    return log_likelihood

            
def logp(p,*args):
    """Custom function defining log of posterioir probability."""
    global count,params,param_index,model_
    
    count += 1
    #print(p)
    # Expected value of outcome
    posterior_log_likelihood = 0 
    try:
        # Get prior likelihood of model parameters 
        prior_log_likelihood = prior_logp(p)
        if abs(prior_log_likelihood) > LARGE_NUMBER:
            return prior_log_likelihood
        
        # Update parameters
        parameters = np.copy(params)
        parameters[param_index] = p[:len(param_index)] 
        stds = p[len(param_index):]
            
        # Get poserior likelihood of model fit to data
        log_likelihood = likelihood_logp(parameters,stds)
            
        # update the progress bar
        if PARALLEL and count%50 == 0:
            sys.stdout.write("\b")
            sys.stdout.write("=>")
            sys.stdout.flush()
        
        #print('parameters: ',p)
        #print('prior: ',prior_log_likelihood,'log_likelihood: ',log_likelihood)
        posterior_log_likelihood = prior_log_likelihood + log_likelihood
        return posterior_log_likelihood
    
    except:
        return -np.inf
   
    
def func_likelihood(p,*args):
    """Custom function defining log of likelihood."""
    global count,params,param_index
    
    count += 1
    # Expected value of outcome
    log_likelihood = 0 
    try:
        # Update parameters
        parameters = np.copy(params)
        parameters[param_index] = p[:len(param_index)]
        stds = p[len(param_index):]
        
        # Get poserior likelihood of model fit to data
        log_likelihood = likelihood_logp(parameters,stds)
            
        # update the progress bar
        if PARALLEL and count%50 == 0:
            sys.stdout.write("\b")
            sys.stdout.write("=>")
            sys.stdout.flush()
        
        return log_likelihood
    
    except:
        return -LARGE_NUMBER-count
    
    
def func_prior(p,*args):
    """Custom function defining log of prior probability."""
    global count,params,param_index
    
    count += 1
    #print(p)
    # Expected value of outcome
    log_likelihood = 0 
    try:
        # Get prior likelihood of model parameters 
        log_likelihood = prior_logp(p)
            
        # update the progress bar
        if PARALLEL and count%50 == 0:
            sys.stdout.write("\b")
            sys.stdout.write("=>")
            sys.stdout.flush()
        
        return log_likelihood
    
    except:
        return -LARGE_NUMBER-count
         

def gelman_rubin(chain):
    """The Gelman-Rubin Test."""
    ssq = np.var(chain,axis=1,ddof=1)
    W = np.mean(ssq,axis=0)
    θb = np.mean(chain,axis=1)
    θbb = np.mean(θb,axis=0)
    m = chain.shape[0]
    n = chain.shape[1]
    B = n/(m-1)*np.sum((θbb-θb)**2,axis=0)
    var_θ = (n-1)/n*W+1/n*B
    R = np.sqrt(var_θ/W)
    
    return R

        
def sample(model,n,obs,Qm,Hm,y0,method="emcee",parameters=None,steady_state=None,
           burn=10,Ndraws=310,Niter=200,ind_non_missing=None,Parallel=False,
           resetParameters=True,debug=True,save=False):
    """
    Draw model parameters samples by using MCMC sampler.
    
    The variants of MC2 sampler are:
        
    * Affine Invariant Markov Chain Monte Carlo (MCMC) Ensemble sampler (emcee package). 
        * Emcee is a Python ensemble sampling toolkit for affine-invariant MCMC.
        * For references on emcee package please see http://dfm.io/emcee/current/
        * The algorithm is based on 2010 paper of Jonathan Goodman and Jonathan Weare’s paper, "Ensemble Samplers with Affine Invariance", https://projecteuclid.org/download/pdf_1/euclid.camcos/1513731992
    * Markov Chain Monte Carlo (MCMC) sampler includes different Metropolis based sampling techniques:
        * Metropolis-Hastings (MH): Primary sampling method.
        * Adaptive-Metropolis (AM): Adapts covariance matrix at specified intervals.
        * Delayed-Rejection (DR): Delays rejection by sampling from a narrower distribution. Capable of n-stage delayed rejection.
        * Delayed Rejection Adaptive Metropolis (DRAM): DR + AM
        Please see https://pymcmcstat.readthedocs.io/_/downloads/en/latest/pdf/
    * Markov Chain Monte Carlo (MCMC) sampler with "particles" package
        Please see https://pypi.org/project/particles ,
        https://github.com/nchopin/particles ,
        
        Nicolas, Chopin and Omiros, Papaspiliopoulos, 2020, "An Introduction to Sequential
        Monte Carlo", Springer Series in Statistics.
      
    Parameters:
        :param model: Model object.
        :type model: Model.
        :param n: Number of endogenous variables.
        :type n: int.
        :param obs: Measurement data.
        :type obs: numpy.array.
        :param Qm: Covariance matrix of errors of endogenous variables. 
        :type Qm: numpy.array.
        :param Hm: Covariance matrix of errors of measurement variables.
        :type Hm: numpy.array.
        :param y0: Starting values of endogenous variables.
        :type y0: list.
        :param method: Sampler algorithm.
        :type method: str.
        :param parameters: List of model parameters.
        :type parameters: list.
        :param steady_state: List of steady states.
        :type steady_state: list.
        :param burn: Number of samples to discard.
        :type burn: int.
        :param Ndraws: Number of draws.
        :type Ndraws: int.
        :param Niter: Number of iterations.
        :type Niter: int.
        :param ind_non_missing: indices of non-missing observations.
        :type ind_non_missing: list.
        :param Parallel: If True runs parallel parameters sampling.
        :type Parallel: bool.
        :param resetParameters: If True resets parameters to the samples mean values.
        :type resetParameters: bool
        :param debug: If True print chain statistics information for pymcmcstat method.
        :type debug: bool.
        
    """
    import warnings
    warnings.filterwarnings("ignore")
        
    global count,n_,nt,params,param_names,param_index,est_shocks_names
    global Z_,y0_,Qm_,obs_,data_,Hm_,ind_non_missing_,n_shocks_,shocks_
    global model_,priors,pm_model,pm_names,pm_params,meas_shocks_,Pstar,PARALLEL
    
    t0 = time() 
    n_  = n
    PARALLEL = Parallel
    model_ = model
    y0_ = y0; obs_= obs
    Qm_ = Qm; Hm_ = Hm
    ind_non_missing_ = ind_non_missing
    samples = None
    
    variables = model.calibration['variables']
    n, = variables.shape
    shocks = meas_shocks_ = model.symbols['shocks']
    n_shocks = len(shocks)
    n_shocks_ = n_shocks
                
    param_names = model.symbols['parameters']
    params = model.calibration['parameters']
    #m_params = dict(zip(param_names,params))
    priors = model_.priors
    
    if 'measurement_shocks' in model.symbols:
        meas_shocks = meas_shocks_ = model.symbols['measurement_shocks']
    else:
        meas_shocks = meas_shocks_ = []
                            
    # Find standard deviations of calibrated shocks
    est_shocks_std = []
    cal = {k:v for k,v in model.priors.items() if k.startswith("std_")}
    for i,v in enumerate(shocks+meas_shocks):
        std_v = "std_"+v
        if std_v in cal:
            est_shocks_names.append(std_v)
            p = cal[std_v]['parameters']
            est_shocks_std.append(float(p[0]))

    param_index = []; names = []      
    for i,k in enumerate(param_names):
        if k in priors:
            param_index.append(i)
            names.append(k)
            
    for i in range(len(est_shocks_names)):
        name = est_shocks_names[i]
        names.append(name)

    # Get reference to measurement function
    nt = len(obs)
       
    s = model.symbols['variables']
    meas_variables = model.symbols['measurement_variables']
    meas_vars = [x.lower() for x in meas_variables]
    ind = []
    for mvr,x in zip(meas_variables,meas_vars):
        if "_meas" in x:
            v = mvr[:-5]
        elif "obs_" in x:
            v = mvr[4:]
        ind.append(s.index(v))
        
    if 'measurement_shocks' in model.symbols:
        n_meas_shocks = len(model.symbols['measurement_shocks'])
    else:
        n_meas_shocks = 0
        
    # Get reference to measurement function
    f_measurement = model.functions['f_measurement']
    bHasAttr  = hasattr(f_measurement,"py_func")
    mv = model.calibration['measurement_variables']
    nm = len(mv)
    
    # Find measurement equations jacobian
    meas_params = model.calibration['measurement_parameters']
    meas_var = np.zeros(n+nm+n_shocks+n_meas_shocks)
    if bHasAttr:
        meas_const,meas_jacob = f_measurement.py_func(meas_var,meas_params,order=1)
    else:
        meas_const,meas_jacob = f_measurement(meas_var,meas_params,order=1)
    obs -= meas_const
    Z = -meas_jacob[:,:n]  
    Z_ = Z
    
    
    ### Get model matrices
    model.solved=False
    ls.solve(model=model,p=params,steady_state=steady_state,suppress_warnings=True)
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
        
    # Set covariance matrix
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
 
    ###------------------------------------------------   Sampling around optimized solution
    print("\nRunning Markov Chain Monte Carlo Sampling...")
    
    print(model.SAMPLING_ALGORITHM)
    if model.SAMPLING_ALGORITHM.value == SamplingAlgorithm.Emcee.value:
        # Bayesian parameter estimation with "emcee" package
        import emcee
        
        p = np.array(list(params[param_index]) + est_shocks_std)
        ndim = len(p)
        nwalkers = 2*ndim
        pos = p * (1+0.5*np.random.rand(nwalkers,ndim))
        
        if PARALLEL:
            n_threads = num_cores
        else:
            n_threads = 1
        # Set sampler
        if emcee.__version__ < "3":
            sampler = emcee.EnsembleSampler(nwalkers=nwalkers,dim=ndim,lnpostfn=logp,threads=n_threads)
        else:
            sampler = emcee.EnsembleSampler(nwalkers=nwalkers,ndim=ndim,log_prob_fn=logp,threads=n_threads)
     
        # Run the MCMC for N steps
        sampler.run_mcmc(pos, Ndraws, progress=True)
        chain = sampler.chain
        
        # Discard the initial "burn" steps
        samples = chain[:, burn:, :]
        samples = samples.reshape((-1, ndim))
        panel   = np.mean(chain,axis=0)
        panel2  = chain.reshape((-1, ndim))
        params_last = chain[-1,-1,:]
        
        if debug:
            print("\n\nAcceptance fraction: {:.1f}%".format(100*np.mean(sampler.acceptance_fraction)))
            
            print("\n\tμ\t\t\tσ\t\t\tGelman-Rubin")
            print("========================================")
            print("\t{:.2f}\t\t{:.2f}\t\t{:.2f}".format(
                chain.reshape(-1, chain.shape[-1]).mean(axis=0)[0],
                chain.reshape(-1, chain.shape[-1]).std(axis=0)[0],
                gelman_rubin(chain)[0]))
            
            from snowdrop.src.graphs.util import plot_chain,plot_pairwise_correlation
            plot_chain(path_to_dir=path_to_dir,chains=panel,names=names,title="Convergence",save=save)
            plot_pairwise_correlation(path_to_dir=path_to_dir,chains=panel2,names=names,title="Pairwise Correlation",save=save)
                    

    elif model.SAMPLING_ALGORITHM.value in [SamplingAlgorithm.Particle_pmmh.value,
                                            SamplingAlgorithm.Particle_smc.value,
                                            SamplingAlgorithm.Particle_gibbs.value]:
        # Bayesian parameter estimation with "particles" package
        from snowdrop.src.utils.distributions import StructDist
        #import particles.distributions as dists  # Distributions
        from particles import mcmc  # MCMC algorithms (PMMH, Particle Gibbs, etc.)
        from particles import smc_samplers as ssp
        #from particles import collectors as col
        from collections import OrderedDict
        from snowdrop.src.utils.distributions import getParticleParameter
        from snowdrop.src.numeric.filters.particle import StateSpaceModel
        from snowdrop.src.numeric.filters.particle import ParticleGibbs
        
        #from statsmodels.stats.correlation_tools import cov_nearest as nearestPositiveDefinite
    
        non_empty = [bool(x) for x in ind_non_missing_]
        data = data_ = obs[non_empty]
        ### Get matrices
        # State transition matrix.
        #T_ = model.linear_model["A"][:n,:n]
        # Array of constants.
        #C_ = model.linear_model["C"][:n]
        # Matrix of coefficients of shocks
        R  = model.linear_model["R"][:n]
        Q_ = 0
        for i in range(1+model.max_lead_shock-model.min_lag_shock):
            R1 = R[:,i*n_shocks:(1+i)*n_shocks]
            Q_ += np.real(R1 @ Qm @ R1.T)
        Q_ = 0.5*(Q_+Q_.T)
            
        # Initialize parameter's array
        prior_names = []; prior_dict = OrderedDict(); val = []
        for i,index in enumerate(param_index):
            name  = param_names[index]
            prior = model_.priors[name]
            pars  = prior['parameters']
            distr = prior['distribution']
            p     = getParticleParameter(distr,pars)
            prior_names.append(name)
            prior_dict[name] = p
            val.append(float(pars[0]))
            
        for i in range(len(est_shocks_names)):
            name = est_shocks_names[i]
            prior = model_.priors[name]
            pars  = prior['parameters']
            distr = prior['distribution']
            p     = getParticleParameter(distr,pars)
            prior_dict[name] = p
        
        my_prior = StructDist(prior_dict)
        
        # Initial values of parameters
        theta0 = np.zeros((1), dtype=my_prior.dtype)
        for v,name in zip(val,names):
            theta0[name] = v
        for i in range(len(est_shocks_names)):
            theta0[est_shocks_names[i]] = est_shocks_std[i]
        
        # Initial number of particles
        Nparticles = 100
        
        # Initialize state-space model           
        # ss_model = StateSpaceModel(x=y0,y=obs,theta0=theta0,data=data,T=T_,R=R,Z=Z,P=Q_,Q=Q_,Qm=Qm,H=Hm,C=C_,n_shocks=n_shocks,verbose=True)
        # Simulate from the model 'nt' data points
        # states, y_data = ss_model.simulate(nt)
        
        if model.SAMPLING_ALGORITHM.value == SamplingAlgorithm.Particle_gibbs.value:
            # Particle Gibbs sampler for a state-space model
            pf = ParticleGibbs(niter=Niter,Nx=Nparticles,ssm_cls=StateSpaceModel,
                               theta0=theta0,prior=my_prior,data=data)
            pf.run()
            theta = pf.chain.theta
        elif model.SAMPLING_ALGORITHM.value == SamplingAlgorithm.Particle_pmmh.value:
            # Particle Marginal Metropolis Hastings algorithm
            pf = mcmc.PMMH(niter=Niter,Nx=Ndraws,ssm_cls=StateSpaceModel,
                           prior=my_prior,data=data,theta0=theta0)
            pf.run()
            theta = pf.chain.theta
        elif model.SAMPLING_ALGORITHM.value == SamplingAlgorithm.Particle_smc.value:
            # Sequential Quasi-Monte Carlo algorithm
            import particles
            fk_smc = ssp.SMC2(ssm_cls=StateSpaceModel,prior=my_prior,data=data,
                              init_Nx=Nparticles,wastefree=True)
            collect = None #[col.Moments()]
            pf = particles.SMC(fk=fk_smc,qmc=False,N=Nparticles,store_history=False,
                               collect=collect,verbose=False)
            pf.run()
            # for i in range(len(data)-1):
            #     next(pf)
            theta = pf.X.theta
            
        samples = []
        for name in names:
            z = theta[name]
            if len(z) > burn:
                y = z[burn:]  # discard the first "burn" iterations
            else:
                y = z
            samples.append(y)
        samples = np.array(samples).T
        params_last = samples[-1]
        
        if debug:
            from snowdrop.src.graphs.util import plot_chain,plot_pairwise_correlation
            plot_chain(path_to_dir=path_to_dir,chains=samples,names=names,title="Convergence",save=save)
            plot_pairwise_correlation(path_to_dir=path_to_dir,chains=samples,names=names,title="Pairwise Correlation",save=save)
 
                
    elif model.SAMPLING_ALGORITHM.value in [SamplingAlgorithm.Pymcmcstat_mh.value,
                                            SamplingAlgorithm.Pymcmcstat_am.value,
                                            SamplingAlgorithm.Pymcmcstat_dr.value,
                                            SamplingAlgorithm.Pymcmcstat_dram.value]:       
        # Bayesian parameter estimation with "pymcmcstat" package
        import tempfile 
        from datetime import datetime
        import pymcmcstat as mcstat
        from pymcmcstat.MCMC import MCMC
        from pymcmcstat.ParallelMCMC import ParallelMCMC
        from pymcmcstat.chain import ChainProcessing as CP
        from pymcmcstat.chain import ChainStatistics as CS

        # Define a directory name based on the date/time to ensure we do not overwrite any results.
        datestr = datetime.now().strftime('%Y%m%d_%H%M%S')
        savedir = tempfile.gettempdir() + os.sep +  'resources' + os.sep + str('{}_{}'.format(datestr, 'parallel_chains')) 
        
        if model.SAMPLING_ALGORITHM.value == SamplingAlgorithm.Pymcmcstat_mh.value:
            sampling_method = "mh"
        elif model.SAMPLING_ALGORITHM.value == SamplingAlgorithm.Pymcmcstat_am.value:
            sampling_method = "am"
        elif model.SAMPLING_ALGORITHM.value == SamplingAlgorithm.Pymcmcstat_dr.value:
            sampling_method = "dr"
        elif model.SAMPLING_ALGORITHM.value == SamplingAlgorithm.Pymcmcstat_dram.value:
            sampling_method = "dram"
            
        # Initialize MCMC object
        mc2 = MCMC()
        # initialize data structure 
        mc2.data.add_data_set(x=[], y=[])
        mc2.simulation_options.define_simulation_options(
            nsimu=Ndraws,updatesigma=True,method=sampling_method,
            savedir=savedir,savesize=1000,save_to_json=True,
            verbosity=0,waitbar=not PARALLEL,save_lightly=True,save_to_bin=True)
        mc2.model_settings.define_model_settings(sos_function=func_likelihood,prior_function=func_prior,sigma2=[1])
        
        # Initialize parameter's array
        iv = []
        for i,index in enumerate(param_index):
            name  = param_names[index]
            prior = model_.priors[name]
            pars  = prior['parameters']
            lower_bound = float(pars[1])
            upper_bound = float(pars[2])
            x = pars[0]
            #x = m_params[name]
            iv.append(x)
            #print(name,x,lower_bound,upper_bound)
            mc2.parameters.add_model_parameter(name=name,theta0=x,minimum=lower_bound,maximum=upper_bound,sample=1)

        # Initialize standard deviation's array
        for name in est_shocks_names:            
            prior = model_.priors[name]
            pars  = prior['parameters']
            lower_bound = float(pars[1])
            upper_bound = float(pars[2])
            x = pars[0]
            iv.append(x)
            #print(name,x,lower_bound,upper_bound)
            mc2.parameters.add_model_parameter(name=name,theta0=x,minimum=lower_bound,maximum=upper_bound,sample=1)
           
        if PARALLEL:
            # # Setup Parallel Simulation and Define Initial Values
            # - You can specify the number of chains to be generated (`num_chain`) 
            #   and the number of cores to use (`num_cores`).
            # setup parallel MCMC
            parMC = ParallelMCMC()
            parMC.setup_parallel_simulation(mcset=mc2,initial_values=np.array([iv]*num_chains),
                                            num_chain=num_chains,num_cores=num_cores)
            
            try:
                # Run Parallel Simulations.
                parMC.run_parallel_simulation()
                # Display Chain Statistics.
                #parMC.display_individual_chain_statistics()
                
                # Processing Chains
                # To load the results we must specify the name of the top level directory 
                # where the parallel results are stored.  
                pres = CP.load_parallel_simulation_results(savedir)
                combined_chains,index = CP.generate_combined_chain_with_index(pres)
                
                chains = CP.generate_chain_list(pres,burnin_percentage=int(100.*burn/Ndraws))
                # Aggregate chains
                samples = np.concatenate(chains)
                params_last = samples[-1]
                
                if debug:
                    # Gelman-Rubin Diagnostic
                    # We generated the list of chains which are required for the Gelman-Rubin diagnostic.  
                    # The output will be a dictionary where the keys are the parameter names.  
                    # Each parameter references another dictionary which contains
                    print(" - `R`, Potential Scale Reduction Factor (PSRF)")
                    print(" - `B`, Between Sequence Variance")
                    print(" - `W`, Within Sequence Variance")
                    print(" - `V`, Mixture-of-Sequences Variance")
                    print(" - `neff`, Effective Number of Samples")
                    CS.gelman_rubin(chains=chains, display=True)
            
                    from snowdrop.src.graphs.util import plot_chain,plot_pairwise_correlation
                    plot_chain(path_to_dir=path_to_dir,chains=chains[-1],names=names,title="Convergence",save=save)
                    plot_pairwise_correlation(path_to_dir=path_to_dir,chains=chains[-1],names=names,title="Pairwise Correlation",save=save)
                            
            except:
                samples = []
            finally:
                # Remove temporary directory
                if os.path.exists(savedir):
                    import shutil
                    shutil.rmtree(savedir)
                
        else:
            
            try:
                # Run Sequential Simulations.
                mc2.run_simulation()
                # Extract results.
                results = mc2.simulation_results.results
                chain   = results['chain']
                samples = chain[burn:]
                params_last = chain[-1]
                
                if debug:
                    # Display chain statistics
                    mc2.chainstats(samples,results)
                    
                    # Generate mcmc plots
                    mcpl = mcstat.mcmcplot # initialize plotting methods
                    mcpl.plot_chain_panel(samples,names)
                    mcpl.plot_density_panel(samples,names)
                    mcpl.plot_pairwise_correlation_panel(samples,names)
                
            except Exception as ex:
                print(ex)
                
            finally:
                # Remove temporary directory
                if os.path.exists(savedir):
                    import shutil
                    shutil.rmtree(savedir)

    else:
        raise Exception(f"Undefined sampling algorithm: {model.SAMPLING_ALGORITHM.name}")
    
    elapsed = time() - t0
    elapsed = np.round(elapsed,2)
    
    # Mean, median and standard deviation of sample
    params_mean = np.mean(samples, axis=0)
    params_median = np.median(samples, axis=0)
    params_std  = np.std(samples, axis=0)
    
    header = ['Name','Start Val.','Median','Std.','Last','Shape','L.B.','U.B.']
    pTable = PrettyTable(header)
    pTable.float_format = '5.3'
    
    # Estimated parameters
    p = params[param_index]
    for i,name in enumerate(names):
        prior = model.priors[name]
        distr = prior['distribution']
        pars  = prior['parameters']
        par   = p[i] if i < len(p) else pars[0]
        row   = [name,par,params_median[i],params_std[i],params_last[i],distr.replace("_pdf","").replace("_hp",""),pars[1],pars[2]]
        pTable.add_row(row)
    
    cprint("\nEnsemble Estimation","green")
    print(pTable)
    
    # Save parameters
    all_parameters = np.copy(parameters)
    if resetParameters:
        all_parameters[param_index] = params_mean
        #all_parameters[param_index] = params_last
    
    sys.stdout.write("\n\n") # this ends the progress bar
    model.solved = False
    
    print("\nSampling elapsed time: {0} (seconds)".format(elapsed))
    

    return all_parameters,names,samples
    
    
