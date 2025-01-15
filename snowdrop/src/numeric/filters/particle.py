#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 26 09:23:51 2021

@author: Alexei Goumilevski
"""
import sys
import numpy as np
from scipy.linalg import solve
from particles import state_space_models as ssm
from particles import mcmc, core
from particles import distributions as dists
from snowdrop.src.numeric.solver import linear_solver
from snowdrop.src.model.settings import SamplingAlgorithm
from snowdrop.src.numeric.solver.util import getCovarianceMatrix
from statsmodels.stats.correlation_tools import cov_nearest as nearestPositiveDefinite
from snowdrop.src.numeric.bayes.mcmc import y0_,obs_,data_,model_,params,Z_,Qm_,Hm_,param_names,param_index,est_shocks_names

count = 0; inst = 0
        
class StateSpaceModel(ssm.StateSpaceModel):
    """State space model class."""

    def __init__(self,x=None,y=None,theta0=None,data=None,T=None,R=None,Z=None,
                 Qm=None,H=None,C=None,parameters=None,model=None,
                 ind_non_missing=None,*args,**kwargs):
        """
        Constructor for State Space Model class.
        
        Args:
            x : array, dtype=float
                is the state variables,
            y : array, dtype=float
                is the observation variables,
            theta0 : structured array, dtype=float
                is the initial values of parameters,
            T : 2D array, dtype=float
                is the state-transition matrix,
            R : 2D array, dtype=float
                is the matrix of shocks,
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
            parameters : array, dtype=float
                is the parameters array,               
            model : object, dtype=Model
                is the Model object,
            ind_non_missing : array, dtype=int
                are the indices of non-missing observations.
        """
        global inst, est_shocks_names
        inst += 1
        #print("\n",inst)
        
        pars  = {}
        self.scale  = 1
        self.ind_non_missing = ind_non_missing
        self.param_names = param_names
        self.param_index = param_index
        self.mult = 1.0
        self.lower_bound = {}
        self.upper_bound = {}
        if model is None:
            self.model = model_
        else:
            self.model = model
        self.b = hasattr(self.model,"SAMPLING_ALGORITHM") and \
                 self.model.SAMPLING_ALGORITHM.value == SamplingAlgorithm.Particle_smc.value
        if parameters is None:
            self.parameters = np.copy(params)
        else:
            self.parameters = np.copy(parameters)
        if not x is None:  
            self.data = data
            self.x0 = x
            self.y  = y
            self.C  = C
            self.T  = T
            self.R  = R
            self.Z  = Z
            self.Qm = Qm
            self.sigmaY = H
            self.nx = len(x)
            self.ny = y.shape
        else:
            self.data = data_
            self.x0 = y0_
            self.y  = obs_
            self.nx = len(self.x0)
            self.ny = obs_.shape
            self.Z  = Z_
            self.Qm = Qm_
            self.sigmaY = Hm_
            ### Get matrices
            # State transition matrix.
            self.T = self.model.linear_model["A"][:self.nx,:self.nx]
            # Array of constants.
            self.C = self.model.linear_model["C"][:self.nx]
            # Matrix of coefficients of shocks.
            self.R = self.model.linear_model["R"][:self.nx]
        
        for k,v in kwargs.items():
            setattr(self,k,v)
            
        self.shocks = self.model.symbols['shocks']
        if 'measurement_shocks' in self.model.symbols:
            self.meas_shocks = self.model.symbols['measurement_shocks']
        else:
            self.meas_shocks = []
                
        # Set parameters
        if not self.param_index is None:
            for i,index in enumerate(self.param_index):
                name  = self.param_names[index]
                prior = self.model.priors[name]
                p     = prior['parameters']
                self.lower_bound[name] = lb =float(p[1])
                self.upper_bound[name] = ub =float(p[2])
                # Get or set parameter value.
                if hasattr(self,name):
                    attr = float(getattr(self,name))
                    if attr < lb or attr > ub:
                        self.mult = 1.e6
                    pars[name] = attr
                
        # Find standard deviations of shocks 
        if hasattr(self,"est_shocks_names") :
            for name in est_shocks_names:
                prior = self.model.priors[name]
                par  = prior['parameters']
                self.lower_bound[name] = lb = float(par[1])
                self.upper_bound[name] = ub = float(par[2])
                # Get or set std of shocks values.
                if hasattr(self,name):
                    attr = float(getattr(self,name))
                    if attr < lb or attr > ub:
                        self.mult = 1.e6
                    pars[name] = attr

        # Set values of covariance matrix
        self.Qm,self.sigmaY = getCovarianceMatrix(self.Qm,self.sigmaY,pars,self.shocks,self.meas_shocks)
                    
        # Get reduced form matrices
        if len(pars) > 0:
            b,T_,C_,R_ = getMatrices(self,pars)
            if b:
                self.T = T_
                self.C = C_
                self.R = R_
            else:
                pass
        
        self.sigmaX = self.R @ self.Qm @ self.R.T       
        self.sigmaX = 0.5*nearestPositiveDefinite(self.sigmaX+self.sigmaX.T) 
        self.sigmaX *= self.mult
        self.sigmaY = 0.5*nearestPositiveDefinite(self.sigmaY+self.sigmaY.T) 
        self.sigmaY *= self.mult        
        #self.sigmaY *= 1.e6
        #print(b,pars)

        # Call parent class constructor
        super().__init__(*args,**kwargs) 
        
    def PX0(self):  
        """Distribution of X_0."""
        if self.b:
            pars  = {}
            for i,index in enumerate(self.param_index):
                name  = self.param_names[index]
                if hasattr(self,name):
                    attr = float(getattr(self,name))
                    pars[name] = attr
                    #print(name,attr)
            
            # Find standard deviations of shocks 
            for name in est_shocks_names:
                if hasattr(self,name):
                    attr = float(getattr(self,name))
                    pars[name] = attr
                    
            # Set values of covariance matrix
            self.Qm,self.sigmaY = getCovarianceMatrix(self.Qm,self.sigmaY,pars,self.shocks,self.meas_shocks)
          
            # Get reduced form matrices
            b,t,c,r = getMatrices(self,pars)
            if b:
                self.T = t
                self.C = c
                self.R = r
            else:
                pass
                    
        self.sigmaX = self.R @ self.Qm @ self.R.T       
        self.sigmaX = 0.5*nearestPositiveDefinite(self.sigmaX+self.sigmaX.T) 
                    
        if np.ndim(np.squeeze(self.sigmaX)) > 1:
            px0 = dists.MvNormal(loc=self.x0,scale=self.scale,cov=self.sigmaX)
        else:
            px0 = dists.Normal(loc=self.x0,scale=np.squeeze(self.sigmaX))

        return px0

    def PX(self, t, xp):  
        """Distribution of X_t given X_{t-1} = xp (p=past)."""      
        if self.b:
            pars  = {}
            for i,index in enumerate(self.param_index):
                name  = self.param_names[index]
                if hasattr(self,name):
                    attr = float(getattr(self,name))
                    pars[name] = attr
                    #print(name,attr)
                        
            # Find standard deviations of shocks 
            for name in est_shocks_names:
                if hasattr(self,name):
                    attr = float(getattr(self,name))
                    pars[name] = attr
                    
            # Set values of covariance matrix
            self.Qm,self.sigmaY = getCovarianceMatrix(self.Qm,self.sigmaY,pars,self.shocks,self.meas_shocks)
                                               
            # Get reduced form matrices
            b,t,c,r = getMatrices(self,pars)
            if b:
                self.T = t
                self.C = c
                self.R = r
            else:
                pass
        
        self.x = xp @ self.T + self.C # + np.sqrt(np.diag(self.sigmaX)) *  (2*np.random.random(self.nx)-1)
        self.sigmaX = self.R @ self.Qm @ self.R.T       
        self.sigmaX = 0.5*nearestPositiveDefinite(self.sigmaX+self.sigmaX.T) 
                             
        if np.ndim(np.squeeze(self.sigmaX)) > 1:
            px = dists.MvNormal(loc=self.x,scale=self.scale,cov=self.mult*self.sigmaX)
        else:
            px = dists.Normal(loc=self.x,scale=np.squeeze(self.sigmaX))

        return px

    def PY(self, t, xp, x):  
        """Distribution of Y_t given X_t=x, and X_{t-1}=xp."""
        global count
        count += 1
        
        if np.isnan(x).all():
            yn = self.Z @ xp.T
        else:
            yn = self.Z @ x.T
       
        if np.ndim(np.squeeze(self.sigmaY)) > 1:
            py = dists.MvNormal(loc=yn.T,scale=self.scale,cov=self.mult*self.sigmaY)
        else:
            py = dists.Normal(loc=yn.T,scale=np.squeeze(self.sigmaY))
            
        if count%500 == 0:
            sys.stdout.write("\b")
            sys.stdout.write("=>")
            sys.stdout.flush()
        return py
    
    # def simulate_given_x(self, x):
    #     lag_x = [None] + x[:-1]
    #     return [self.PY(t, xp, x).rvs(size=1)
    #             for t, (xp, x) in enumerate(zip(lag_x, x))]

    
class NonlinearStateSpaceModel(ssm.StateSpaceModel):
    """State space model class."""

    def __init__(self,x=None,y=None,parameters=None,Z=None,sigmaX=None,sigmaY=None,model=None,ind_non_missing=None,n_shocks=None,*args,**kwargs):
        """
        Constructor for State Space Model class.
        
        Args:
            x : array, dtype=float
                is the state variables,
            y : array, dtype=float
                is the observation variables,
            parameters : array, dtype=float
                is the parameters array,               
            model : object, dtype=Model
                is the Model object,
            ind_non_missing : array, dtype=int
                are the indices of non-missing observations.
        
        """
        self.scale = 1
        self.model = model
        self.x0 = x
        self.y = y
        self.Z = Z
        self.nx = len(x)
        self.sigmaX = sigmaX
        self.sigmaY = sigmaY
        self.param_names = param_names
        self.param_index = param_index
        self.parameters = np.copy(parameters)
        self.ind_non_missing = ind_non_missing
        self.n_shocks = n_shocks
        
        # Set parameters
        self.pars  = np.copy(self.parameters)
        if not self.model is None:           
            for i,index in enumerate(self.param_index):
                name  = self.param_names[index]
                prior = self.model.priors[name]
                p     = np.copy(prior['parameters'])
                lb    = float(p[1])
                ub    = float(p[2])
                # Set parameter value.
                if hasattr(self, name):
                    par = float(getattr(self, name))
                    self.pars[i] = min(ub,max(lb,par))
                
        for k,v in kwargs.items():
            setattr(self,k,v)
            
        # Call parent class constructor
        super().__init__(*args,**kwargs) 
        
    def PX0(self):  
        """Distribution of X_0."""
        
        # This assumes that equations are written in such a way that states variables
        # at time t+1 are expressed as non-linear functions of these variables at time t      
        f_rhs = self.model.functions["f_rhs"]
        z = np.concatenate([self.x0,self.x0,self.x0,np.zeros(self.n_shocks)])
        xn = f_rhs(z,self.pars,order=0)
        self.x = xn
        px0 = dists.MvNormal(loc=xn,scale=self.scale,cov=nearestPositiveDefinite(self.sigmaX))
        return px0

    def PX(self, t, xp):  
        """Distribution of X_t given X_{t-1} = xp (p=past)."""
                        
        x = np.squeeze(xp)
        ndim = np.ndim(x)
        if ndim == 2:
            x = xp[t]
        # This assumes that equations are written in such a way that states variables
        # at time t+1 are expressed as non-linear functions of these variables at time t      
        f_rhs = self.model.functions["f_rhs"]
        z = np.concatenate([x,x,x,np.zeros(self.n_shocks)])
        xn = f_rhs(z,self.pars,order=0)
        self.x = xn
        px = dists.MvNormal(loc=xn,scale=self.scale,cov=nearestPositiveDefinite(self.sigmaX))
        return px
    
    def PY(self, t, xp, x):  
        """Distribution of Y_t given X_t=x, and X_{t-1}=xp."""
        xs = np.squeeze(x)
        ndim = np.ndim(xs)
        if ndim == 1:
            yn = self.Z @ xs.T
        else:
            yn = self.Z @ x.T
        py = dists.MvNormal(loc=yn.T,scale=self.scale,cov=self.sigmaY)
        return py
    
    
class ParticleGibbs(mcmc.ParticleGibbs):
    
    def update_theta(self, theta, x):
        """ Update model parameters. """
        orig = self.theta0.view(np.float)
        new = np.array(theta.tolist())
        delta = new - orig
        if np.all(delta==0):
            eps = 0.1 * max(abs(new)) *  (2*np.random.random(len(orig))-1)
        else:
            eps = 0.1 * delta *  (2*np.random.random(len(orig))-1)
        delta = new + eps
            
        new_theta = theta.copy()
        for i,k in enumerate(theta):
            k += delta[i]
            
        # sigma, rho = 0.2, 0.95  # fixed values
        # xlag = np.array(x[1:] + [0.,])
        # dx = (x - rho * xlag) / (1. - rho)
        # s = sigma / (1. - rho)**2
        # new_theta['mu'] = self.prior.laws['mu'].posterior(dx, sigma=s).rvs()
        
        return new_theta

    
def getMatrices(obj,par):
     """ Compute matrices."""
     b = False; T = None; C = None; R = None
     if not obj.model is None:
         # Set parameters
         pars  = np.copy(obj.parameters)
         for index in param_index:
             name  = param_names[index]
             # Set parameter value.
             if name in pars:
                 pars[index] = par[name]
                     
         ### Solve linear system
         try:
             obj.model.solved=False
             mdl = linear_solver.solve(model=obj.model,p=pars,steady_state=np.zeros(obj.nx),suppress_warnings=True)

             ### Get matrices
             # State transition matrix.
             T = mdl.linear_model["A"][:obj.nx,:obj.nx]
             # Array of constants.
             C = mdl.linear_model["C"][:obj.nx]
             # Matrix of coefficients of shocks.
             R = mdl.linear_model["R"][:obj.nx]
             b = True
         except:
             b = False
         
     return b,T,C,R
    