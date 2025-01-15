"""
Dynamic Programming.

Created on Mon Feb 15 16:48:37 2021
@author: A.Goumilevski
"""
import os
import numpy as np
import pandas as pd
from time import time
import scipy.sparse as sparse
from scipy.linalg import pinv,norm

from scipy.sparse.linalg import spsolve
from scipy.optimize import fmin
from numpy.matlib import repmat
#from numpy import log, exp, sin, cos, tan, sqrt, pi, inf

# Grids    
#from snowdrop.src.numeric.dp.grids import UnstructuredGrid,CartesianGrid,NonUniformCartesianGrid,SmolyakGrid

NITERATIONS = 100 # Maximum nymber of iterations
EPSILON = 1.e-5   # Convergence tolerance


class DynProg:

    def __init__(self,functions,m_var,m_par,options,state_variables,control_variables,utilities,value_functions,eqs,be_eqs,
                 lower_boundary,upper_boundary,ns=2,rho=0.8,shk_stddedv=0.3,Ngrid=1000,ngrid=100,n_order=20,R=None,Q=None):
        """
        Set up the elements that define an instance of the DP class.

        Parameters:
            functions : dict
                Map of model functions.
            m_var : dict
                Map of variables names and its intitial values.
            m_par : dict
                Map of parameters names and values.
            options : dict
                Model options.
            state_variables : list
                Names of state variables. These variables are functions of decision rules that depend on 
                control variables,i.e., state_variables = function(decision_rules(control_variables)).
            control_variables : list
                Names of control variables. Value function is maximized with respect to these variables.
            eqs : list
                List of transient equations.
            be_eqs : list
                List of equations used to calculate value function.
            utilities : list
                Utility functions.
            value_functions : list
                Value functions.
            lower_boundary : dict
                Lower boundary of state variables.
            upper_boundary : dict
                Upper boundary of state variables.
            ns : int, optional
                Number of admissable states of shocks. The default is 2.
                Shocks are stockastic and are descried by VAR(1) process with 'ns' states.
            rho : float, optional
                Persistancy of shocks in VAR(1) process. The default is 0.8.
            shk_stddedv : float, optional
                Standard deviation of shpcks. The default is 0.1.
            Ngrid : int, optional
                Number of grid points. The default is 1000.
            ngrid : int, optional
                Number of polynomial nodes. The default is 100.
            n_order : int, optional
                Order of Chebyshev polynomials. The default is 10.
            R : function object, optional
                Utility (aka Reward) function..
            Q : function object, optional
                Transition probabilities 𝑄(s_{t},a_{t},s_{t+1}) for the next period state s_{t+1}.
            
        """
        self.functions         = functions
        self.beta              = m_par["beta"]           # discount factor
        self.m_var             = m_var
        self.m_par             = m_par
        self.options           = options
        self.variables         = list(m_var.keys())
        self.state_variables   = state_variables
        self.control_variables = control_variables
        self.equations         = eqs
        self.bellman_equations = be_eqs
        self.utilities         = utilities
        self.value_functions   = value_functions
        self.lower_bound       = lower_boundary
        self.upper_bound       = upper_boundary
        self.Ngrid             = Ngrid
        self.ngrid             = ngrid
        self.n                 = len(m_var)               # bunber of variables
        self.nc                = len(control_variables)   # number of control variables.
        self.ns                = len(state_variables)     # number of state variables.
        self.n_order           = n_order                  # Order of polinomials
        self.n_shocks_states   = ns                       # Number of admissable shocks states
        self.rho               = rho
        self.shock_stddedv     = shk_stddedv
        self.R                 = R
        self.Q                 = Q
        
        util_compiled = {}
        for u in utilities:
            arr = u.split("=")
            key = arr[0].strip()
            rhs = arr[1].strip()
            util_compiled[key] = compile(rhs, "",mode="eval") # Convert string to code object)
        self.util_compiled = util_compiled
        
        
    def utility_function(self,m):
        """
        Evaluate utility function.

        Parameters:
            m : dict
                Dictionary of variables and parameters names and values.

        Returns:
            numpy array
                Utility values.

        """
        arr = []
        
        # for u in self.utilities:
        #     c = m['c']
        #     sigma = m["sigma"]
        #     arr = u.split("=")
        #     eq = arr[1].strip()
        #     v = eval(eq,globals(),m)
        #     arr.append(v.copy())
        
        for key in self.util_compiled:
            compiled = self.util_compiled[key]
            v = eval(compiled,globals(),m)
            arr.append(v.copy())
        
        utility = np.squeeze(np.array(arr)) 
        ind     = np.isnan(utility)
        utility[ind] = -1.e30
        ind     = np.isinf(utility)
        utility[ind] = -1.e20
                        
        return utility
    

    def value(self,expr,m):
        arr = expr.split("=")
        lhs = arr[0].strip()
        rhs = arr[1].strip()
        v = eval(rhs,globals(),m)
        #print(np.min(v),np.max(v))
        return lhs,v
        
        
    def updateVariables(self,d,j=None):
        """
        Updates control variables.

        Parameters:
            d : dict
                Dictionary of variables and parameters names and values.
            j : int
                Index of grid.

        Returns:
            numpy.array
                Utility function.

        """
        def EXPECTATION(k):
            kp = k+"__(1)"
            if kp in m:
                return m[kp]
            elif k in m:
                return m[k]
            else:
                return np.nan
                
        mp = self.m_par.copy()
        mp["EXPECTATION"] = EXPECTATION
        if j is None:
            for v in self.variables:
                mp[v] = d[v]
        else:
            z = np.array([d[x] for x in self.variables])
            for i,v in enumerate(self.variables):
                mp[v] = z[i][j]
            
        keys = []; m = d.copy()  
        for eq in self.bellman_equations:
            key,val = self.value(eq,mp)
            keys.append(key)
            m[key] = val
            
        return m,keys
            

    def transient_func(self,x,params):
        """
        Compute function values.

        Parameters:
            x : list
                Variables values.
            params : list
                Parameters values.

        Returns:
            Function values.

        """
        f_dynamic = self.functions["f_dynamic"]
        f,der = f_dynamic(x,params,order=1)
        
        return f,der
        
      
    def getBounds(self,m):
        """
        Returns bound on variables.

        Parameters:
            m : dict
                Locals mapping.

        Returns:
            Array of 2-tuples.
            
        """
        def f(v):
            if isinstance(v,str):
                fun = compile(v, "",mode="eval")
                x = eval(fun,globals(),m)
                return float(x)
            else:
                return v
            
        var = self.variables
        lb  = self.lower_bound
        if lb is None: 
            lb = []
        ub  = self.upper_bound 
        if ub is None: 
            ub = []
        mp = dict()
        for v in var:
            if v in lb:
                lower = f(lb[v])
            else:
                lower = -1.e30
            if v in ub:
                upper = f(ub[v])
            else:
                upper = 1.e30
            mp[v] = (lower,upper)
                
        return mp 
        
        
    def value_iteration(self,debug=False):
        """
        Solve the OGM by value function iteration.
    
        Parameters:
            Instance of tis class.
        
        Returns:
            iterations : int
                Number of iterations.
            crit : float
                Convergence value.
            y : numpy array
                Solution of Bellman equation
            
        """
        y = []
        crit = 1.e6                                   # convergence value
        iterations = 0                                # number of iterations
        dr = np.zeros((self.nc,self.Ngrid),dtype=int)      # decision rule
        
        # Initial guess for value function
        Tv = np.zeros((self.nc,self.Ngrid))
        v = np.zeros((self.nc,self.Ngrid))
        
        # Build control variables grid.
        for i in range(self.nc):
            var_name = self.control_variables[i]
            if var_name in self.options:
                kmin,kmax = self.options[var_name]
            else:
                kmin, kmax = 0, 3*self.m_var[var_name]
                
        
        # Initialize dictionary with values of variables
        m = {}
        for k in self.variables:
            if k in self.options:
                kmin,kmax = self.options[k]
            else:
                kmin,kmax = -3*self.m_var[k],3*self.m_var[k]
            m[k] = np.linspace(kmin,kmax,self.Ngrid)
        for k in self.m_par:
            m[k] = self.m_par[k]
        
        # Get variables bounds
        bounds = self.getBounds(m)
      
        while crit > EPSILON and iterations < NITERATIONS:
               
            for j in range(self.Ngrid):
                # Update control variables.
                mu,keys = self.updateVariables(m,j)
                for i in range(self.nc):
                    # Get utility
                    u = self.utility_function(mu)
                    x = u + self.beta * v[i]
                    Tv[i,j] = np.max(x)
                    dr[i,j] = np.argmax(x)
                            
            # Update expected control variables by applying the decision rule
            mu = m.copy()
            for i in range(self.nc):
                var = self.control_variables[i]
                val	 = m[var][dr[i]]
                if var in bounds:
                    lower,upper = bounds[var]
                    val = np.maximum(lower,np.minimum(upper,val))
                mu[var+"__(1)"] = val
                
            # Update state variables of Bellman equations
            mu,keys = self.updateVariables(mu)
            for key in keys:
                val = mu[key]
                if var in bounds:
                    lower,upper = bounds[var]
                    val = np.maximum(lower,np.minimum(upper,val))
                m[key] = val
            
            # Update state variables of transient equations 
            for eq in self.equations:
                #print(eq)
                # N = mu["N"]; S = mu["S"]
                # print("N",N)
                # print("S:",S)
                var,val = self.value(eq,mu)
                #print(var,val)
                if var in self.state_variables:
                    if var in bounds:
                        lower,upper = bounds[var]
                        val = np.maximum(lower,np.minimum(upper,val))
                    m[var] = val
        
            crit = np.max(abs(Tv-v))/np.linalg.norm(Tv)
            v    = Tv.copy()
            
            iterations += 1
            if debug:
                print('Iteration # {:03d} \tCriterion: {:.2e}'.format(iterations,crit))
               
        
        # Update state variables of transient equations 
        for i in range(2):
            for eq in self.equations:
                var,val = self.value(eq,m)
                if var in self.variables:
                    if var in bounds:
                        lower,upper = bounds[var]
                        val = np.maximum(lower,np.minimum(upper,val))
                    m[var] = val
                    
        # Get solution
        for var in self.variables:
            y.append(m[var])
            
        return iterations,crit,np.array(y)
            
    
    def policy_iteration(self,debug=False):
        """
        Solve the OGM by policy function iteration (Howard method).
    
        Parameters:
            Instance of tis class.
        
        Returns:
            iterations : int
                Number of iterations.
            crit : float
                Convergence value.
            y : numpy array
                Solution of Bellman equation.
            
        """
        y = []
        crit = 1.e6                                   # convergence value
        iterations = 0                                # number of iterations
        dr = np.zeros((self.nc,self.Ngrid),dtype=int)      # decision rule
        
        # Initial guess for value function
        Tv = np.zeros((self.nc,self.Ngrid))
        v = np.zeros((self.nc,self.Ngrid))
        
        # Build control variables grid.
        control_nodes = []; state_nodes = []
        for i in range(self.nc):
            var_name = self.control_variables[i]
            if var_name in self.options:
                kmin,kmax = self.options[var_name]
            else:
                kmin, kmax = 0, 3*self.m_var[var_name]
        
        # Initialize dictionary with values of variables
        m = {}
        for k in self.variables:
            if k in self.options:
                kmin,kmax = self.options[k]
            else:
                kmin,kmax = -3*self.m_var[k],3*self.m_var[k]
            m[k] = np.linspace(kmin,kmax,self.Ngrid)
        for k in self.m_par:
            m[k] = self.m_par[k]
        
        # Get variables bounds
        bounds = self.getBounds(m)
        
        while crit > EPSILON and iterations < NITERATIONS:
               
            for j in range(self.Ngrid):
                # Update control variables.
                mu,keys = self.updateVariables(m,j)
                for i in range(self.nc):
                    # Get utility
                    u = self.utility_function(mu)
                    x = u + self.beta * v[i]
                    Tv[i,j] = np.max(x)
                    dr[i,j] = np.argmax(x)
                            
            # Update expected control variables by applying the decision rule
            mu = m.copy()
            for i in range(self.nc):
                var = self.control_variables[i]
                val	 = m[var][dr[i]]
                if var in bounds:
                    lower,upper = bounds[var]
                    val = np.maximum(lower,np.minimum(upper,val))
                mu[var+"__(1)"] = val
                
            # Update state variables of Bellman equations
            mu,keys = self.updateVariables(mu)
            for key in keys:
                val = mu[key]
                if var in bounds:
                    lower,upper = bounds[var]
                    val = np.maximum(lower,np.minimum(upper,val))
                m[key] = val
            
            # Update state variables of transient equations 
            for eq in self.equations:
                var,val = self.value(eq,mu)
                if var in self.state_variables:
                    if var in bounds:
                        lower,upper = bounds[var]
                        val = np.maximum(lower,np.minimum(upper,val))
                    m[var] = val
        
            Q = sparse.lil_matrix((self.Ngrid*self.nc,self.Ngrid*self.nc))
            for i in range(self.nc):
                Q0  = sparse.lil_matrix((self.Ngrid,self.Ngrid))
                for j in range(self.Ngrid):
                    Q0[j,dr[i,j]] = 1
                i1 = i*self.Ngrid
                i2 = (i+1)*self.Ngrid
                Q[i1:i2,i1:i2] = Q0
            q = Q.todense()
                   
            u   = self.utility_function(m)
            M   = sparse.eye(self.Ngrid*self.nc) - self.beta *Q
            Tv  = spsolve(M, u)
            Tv  = np.reshape(Tv,(self.nc,self.Ngrid))
            
            crit = np.max(abs(Tv-v))/norm(Tv)
            v    = Tv.copy()
            
            iterations += 1
            if debug:
                print('Iteration # {:03d} \tCriterion: {:.2e}'.format(iterations,crit))
        
        # Update state variables of transient equations 
        for i in range(2):
            for eq in self.equations:
                var,val = self.value(eq,m)
                if var in self.variables:
                    if var in bounds:
                        lower,upper = bounds[var]
                        val = np.maximum(lower,np.minimum(upper,val))
                    m[var] = val
                    
        # Get solution
        for var in self.variables:
            y.append(m[var])
            
        return iterations,crit,np.array(y)
            
        
    def parametric_value_iteration(self,n=None,debug=False):
        """
        Solve the OGM by parametric value iteration.
    
        Parameters:
            n : int, optional
                Number of data points in the grid. The default is 20.
        
        Returns:
            iterations : int
                Number of iterations.
            crit : float
                Convergence value.
            y : numpy array
                Solution of Bellman equation.
           
        """
        from snowdrop.src.numeric.dp.util import Chebychev_Polinomial
        
        y = []
        crit = 1.e6                                   # convergence value
        iterations = 0                                # number of iterations
        ngrid  = self.ngrid if n is None else n
        
        # Initial guess for value function
        Tv = np.zeros((self.nc,ngrid))
        v  = np.zeros((self.nc,ngrid))
        
        # Interpolating nodes
        rk = -np.cos((2*np.arange(1,ngrid+1)-1)*np.pi/(2.*ngrid)) 
        # Chebyshev polynomials
        ch = Chebychev_Polinomial(rk,ngrid)    
        Cheb  = repmat(ch,1,self.nc)  
        iCheb = pinv(Cheb)   
        theta = iCheb @ v.T   # Initial guess of parameters                      
                
        # Initialize dictionary with values of variables
        m = {}
        for k in self.variables:
            if k in self.options:
                kmin,kmax = self.options[k]
            else:
                kmin,kmax = -3*self.m_var[k],3*self.m_var[k]
            if k in self.control_variables:    
                m[k] = kmin+(rk+1)*(kmax-kmin)/2             # Mapping
            else:
                m[k] = np.linspace(kmin,kmax,ngrid)
        for k in self.m_par:
            m[k] = self.m_par[k]
            
        # Get variables bounds
        bounds = self.getBounds(m)
            
        # Value function
        def tv(theta):
            v = Cheb @ theta
            res = []
            # Update control variables.
            mu,keys = self.updateVariables(m)
            for i in range(self.nc):
                # Get utility
                u = self.utility_function(mu)
                value = u + self.beta * v[i]
                res.append(value.copy())
                    
            res = np.array(res)
            return res
        
        # Minus norm of tv function    
        tv_norm = lambda x: -norm(tv(x))
            
        while crit > EPSILON and iterations < NITERATIONS:
            x = fmin(tv_norm,theta,xtol=1.e-3,ftol=1.e-3,maxiter=100,disp=False)
            Tv = tv(x)
            theta = iCheb @ Tv.T
            
            # Update state variables of Bellman equations
            mu,keys = self.updateVariables(m)
            for key in keys:
                val = mu[key]
                if key in bounds:
                    lower,upper = bounds[key]
                    val = np.maximum(lower,np.minimum(upper,val))
                m[key] = val
                
            # Update state variables of transient equations 
            for eq in self.equations:
                var,val = self.value(eq,mu)
                if var in self.state_variables:
                    if var in bounds:
                        lower,upper = bounds[var]
                        val = np.maximum(lower,np.minimum(upper,val))
                    m[var] = val
                
            crit= np.max(abs(Tv-v))/norm(Tv)
            v   = Tv.copy()
            
            iterations += 1
            if debug:
                print('Iteration # {:03d} \tCriterion: {:.2e}'.format(iterations,crit))
            
        # Update state variables of transient equations 
        for i in range(2):
            for eq in self.equations:
                var,val = self.value(eq,m)
                if var in self.variables:
                    if var in bounds:
                        lower,upper = bounds[var]
                        val = np.maximum(lower,np.minimum(upper,val))
                    m[var] = val
                    
        # Get solution
        for var in self.variables:
            y.append(m[var])
            
        return iterations,crit,np.array(y)
    
        
    def stochastic_value_iteration(self,p=0.9,debug=False):
        """
        Solve the stochastic OGM by value iteration.
    
        Parameters:
            p : float, optional
                Probability value. The default is 0.9.
                It is used for discrete approximation of VAR(1) stochastic process.
        
        Returns:
            iterations : int
                Number of iterations.
            crit : float
                Convergence value.
            y : numpy array
                Solution of Bellman equation
            
        """
        y = []
        crit = 1.e6                                                     # convergence value
        iterations = 0                                                  # number of iterations
        dr = np.zeros((self.nc,self.Ngrid,self.n_shocks_states),dtype=int)   # decision rule
        
        p  = (1+self.rho)/2 if p is None else p
        PI = np.array([[p,1-p],[1-p,p]])
        A  = [np.exp(-self.shock_stddedv**2/(1-self.rho**2)), 
              np.exp(+self.shock_stddedv**2/(1-self.rho**2))]
        A  = np.array(A)
        
        self.m_par["A"] = A
        
        # Initial guess for value function
        Tv = np.zeros((self.nc,self.Ngrid,self.n_shocks_states))
        v = np.zeros((self.nc,self.Ngrid,self.n_shocks_states))
        
        # Build control variables grid.
        control_nodes = []; state_nodes = []
        for i in range(self.nc):
            var_name = self.control_variables[i]
            if var_name in self.options:
                kmin,kmax = self.options[var_name]
            else:
                kmin, kmax = 0, 3*self.m_var[var_name]
        
        # Initialize dictionary with variables values
        m = {}
        for k in self.variables:
            if k in self.options:
                kmin,kmax = self.options[k]
            else:
                kmin,kmax = -3*self.m_var[k],3*self.m_var[k]
            grid = np.linspace(kmin,kmax,self.Ngrid)
            m[k] = repmat(grid,self.n_shocks_states,1).T
        for k in self.m_par:
            m[k] = self.m_par[k]

        # Get variables bounds
        bounds = self.getBounds(m)

        while crit > EPSILON and iterations < NITERATIONS:
                   
            for j in range(self.Ngrid):
                # Update control variables.
                mu,keys = self.updateVariables(m,j)
                for i in range(self.nc):
                    # Get utility
                    u = self.utility_function(mu)
                    x = u + self.beta * (v[i] @ PI)
                    for s in range(self.n_shocks_states):
                        Tv[i,j,s] = np.max(x[:,s])
                        dr[i,j,s] = np.argmax(x[:,s])
                            
            # Update expected control variables by applying the decision rule
            mu = m.copy()
            for i in range(self.nc):
                var = self.control_variables[i]
                arr = []
                for s in range(self.n_shocks_states):
                    q = m[var][:,s]
                    dri = dr[i,:,s]
                    val	= q[dri]
                    arr.append(val.copy())
                value = np.array(arr)
                if var in bounds:
                    lower,upper = bounds[var]
                    val = np.maximum(lower,np.minimum(upper,val))
                mu[var+"__(1)"] = value.T
                
            # Update state variables of Bellman equations
            mu,keys = self.updateVariables(mu)
            for key in keys:
                m[key] = mu[key]
            
            # Update state variables of transient equations 
            for eq in self.equations:
                var,val = self.value(eq,mu)
                if var in self.state_variables:
                    if var in bounds:
                        lower,upper = bounds[var]
                        val = np.maximum(lower,np.minimum(upper,val))
                    m[var] = val
        
            crit = np.max(abs(Tv-v))/norm(Tv)
            v    = Tv.copy()
            
            iterations += 1
            if debug:
                print('Iteration # {:03d} \tCriterion: {:.2e}'.format(iterations,crit))
        
        # Update state variables of transient equations 
        for i in range(2):
            for eq in self.equations:
                var,val = self.value(eq,m)
                if var in self.variables:
                    if var in bounds:
                        lower,upper = bounds[var]
                        val = np.maximum(lower,np.minimum(upper,val))
                    m[var] = val
                    
        # Get solution
        for var in self.variables:
            y.append(m[var])
            
        return iterations,crit,np.array(y)
    
    
    def stochastic_policy_iteration(self,p=0.9,kmin=0.2,kmax=6,debug=False):
        """
        Solve the stochastic OGM by policy iteration.
        
        Parameters:
            p : float, optional
                Probability value. The default is 0.9.
                It is used for discrete approximation of VAR(1) stochastic process.
            kmin : float, optional
                Lower bound on the grid. The default is 0.2
            kmax : float, optional
                Upper bound on the grid. The default is 6.
                
        Returns:
            iterations : int
                Number of iterations.
            crit : float
                Convergence value.
            y : numpy array
                Solution of Bellman equation
            
        """
        y = []
        crit = 1.e6                                                         # convergence value
        iterations = 0                                                      # number of iterations
        Tv0 = 0
        dr = np.zeros((self.nc,self.Ngrid,self.n_shocks_states),dtype=int)  # decision rule
        
        p  = (1+self.rho)/2 if p is None else p
        p=1
        PI = np.array([[p,1-p],[1-p,p]])
        A  = [np.exp(-self.shock_stddedv**2/(1-self.rho**2)), 
              np.exp(+self.shock_stddedv**2/(1-self.rho**2))]
        A  = np.array(A)
        
        self.m_par["A"] = A
        
        # Initial guess for value function
        v = np.zeros((self.nc,self.Ngrid,self.n_shocks_states))
        
        # Initialize dictionary with values of variables
        m = {}
        for k in self.variables:
            if k in self.options:
                kmin,kmax = self.options[k]
            else:
                kmin,kmax = -3*self.m_var[k],3*self.m_var[k]
            grid = np.linspace(kmin,kmax,self.Ngrid)
            m[k] = repmat(grid,self.n_shocks_states,1).T
        for k in self.m_par:
            m[k] = self.m_par[k]
            
        # Get variables bounds
        bounds = self.getBounds(m)
            
        while crit > EPSILON and iterations < NITERATIONS:
                
            for j in range(self.Ngrid):
                # Update control variables.
                mu,keys = self.updateVariables(m,j)
                for i in range(self.nc):
                    # Get utility
                    u = self.utility_function(mu)
                    x = u + self.beta * (v[i] @ PI)
                    for s in range(self.n_shocks_states):
                        dr[i,j,s] = np.argmax(x[:,s])
          
            # Update expected control variables by applying decision rule
            mu = m.copy()
            for i in range(self.nc):
                var = self.control_variables[i]
                arr = []
                for s in range(self.n_shocks_states):
                    q = m[var][:,s]
                    val	 = q[dr[i,:,s]]
                    arr.append(val.copy())
                val = np.array(arr)
                if var in bounds:
                    lower,upper = bounds[var]
                    val = np.maximum(lower,np.minimum(upper,val))
                mu[var+"__(1)"] = val.T
                
            # Update state variables of Bellman equations
            mu,keys = self.updateVariables(mu)
            for key in keys:
                m[key] = mu[key]
                
            # Update state variables of transient equations 
            for eq in self.equations:
                var,val = self.value(eq,mu)
                if var in self.state_variables:
                    if var in bounds:
                        lower,upper = bounds[var]
                        val = np.maximum(lower,np.minimum(upper,val))
                    m[var] = val
                
            N1 = self.Ngrid*self.n_shocks_states
            N  = N1*self.nc
            Q = sparse.lil_matrix((N,N))
            
            for i in range(self.nc):
                Q1 = sparse.lil_matrix((N1,N1))
                for s in range(self.n_shocks_states):
                    Q2  = sparse.lil_matrix((self.Ngrid,self.Ngrid))
                    for j in range(self.Ngrid):
                       Q2[j,dr[i,j,s]] = 1  
                       #q2 = Q2.todense()
                   
                    z = sparse.kron(PI[s],Q2)
                    Q1[s*self.Ngrid:(s+1)*self.Ngrid] = z
                    #q1 = Q1.todense()
                
                Q[i*N1:(i+1)*N1] = Q1
            #q = Q.todense()
                
            u   = self.utility_function(m)
            M   = sparse.eye(N) - self.beta *Q
            Tv  = spsolve(M, np.ravel(u.T))   
            v   = np.reshape(Tv,(self.n_shocks_states,self.Ngrid,self.nc)).T
            
            crit= np.max(abs(Tv-Tv0))/norm(Tv)
            Tv0 = Tv.copy()
            
            iterations += 1
            if debug:
                print('Iteration # {:03d} \tCriterion: {:.2e}'.format(iterations,crit))
            
        # Update state variables of transient equations 
        for i in range(2):
            for eq in self.equations:
                var,val = self.value(eq,m)
                if var in self.variables:
                    if var in bounds:
                        lower,upper = bounds[var]
                        val = np.maximum(lower,np.minimum(upper,val))
                    m[var] = val
                    
        # Get solution
        for var in self.variables:
            y.append(m[var])
            
        return iterations,crit,np.array(y)
    
        
    def discrete(self):
        """Solve Bellman equation by discrete dynamic programming method."""
        import quantecon as qe
        
        ddp = qe.markov.DiscreteDP(self.R, self.Q, self.beta)
        results = ddp.solve(method='policy_iteration',epsilon=EPSILON)
        y = results.v
        crit = results.sigma
        iterations = results.num_iter
        #self.distr = results.mc.stationary_distributions
        
        return iterations,crit,y
    

def Plot(path_to_dir,data,variable_names,var_labels={}):
    
    from snowdrop.src.graphs.util import plotTimeSeries
    
    header = 'Solution of Bellman Equation'
    series = []; labels = []; titles = []
    
    if np.ndim(data) == 3:
        ns,n,nvar = data.shape
        for i in range(nvar):
            ser = []; lbls = []
            for j in range(ns):
                ser.append(pd.Series(data[j,:,i]))
                lbls.append("Markov State " + str(1+j))
            series.append(ser)
            labels.append(lbls)
            var = variable_names[i]
            name = var_labels[var] if var in var_labels else var
            titles.append(name)
    elif np.ndim(data) == 2:
        n,nvar = data.shape
        for i in range(nvar):
            ser = pd.Series(data[:,i])
            series.append([ser])
            labels.append([])
            var = variable_names[i]
            name = var_labels[var] if var in var_labels else var
            titles.append(name)    
    else:
        nvar = len(y)
        for i in range(nvar):
            ser = []; lbls = []
            data = y[i]
            ns = len(data)
            for j in range(ns):
                ser.append(pd.Series(data[j]))
                if ns>1:
                    lbls.append("Markov State " + str(1+j))
                else:
                    lbls.append(" ")
            series.append(ser)
            labels.append(lbls)
            var = variable_names[i]
            name = var_labels[var] if var in var_labels else var
            titles.append(name)

    if nvar <= 4:
        sizes = [2,2]
    else:
        sizes = [int(np.ceil(nvar/3)),3]
        
    plotTimeSeries(path_to_dir=path_to_dir,header=header,titles=titles,labels=labels,series=series,sizes=sizes)
    
        
def simulate(model):
    """
    Solve Bellman equation.

    Parameters:
        model : Model object
            Instance of model class.

    Returns:
        numpy.array.

    """
    from collections import OrderedDict
    
    t0 = time()
    # Get list of control variables equations
    eqs = model.symbolic.equations
    options = model.options
    
    # Get variables and parameters
    var_names  = model.symbols["variables"]
    var_values = model.calibration["variables"]
    var        = OrderedDict(zip(var_names,var_values))
    par_names  = model.symbols["parameters"]
    par_values = model.calibration["parameters"]
    par        = OrderedDict(zip(par_names,par_values))
    
    # Retrieve section related to Bellman equation.
    arr = model.symbolic.bellman 
    assert bool(arr), "Information on Bellman equation is missing in a model file!"
    be = arr[0]
    # Get Bellman value function
    utility_func      = be.get("utilities", [])
    value_func        = be.get("value_functions", [])
    be_eqs            = be.get("equations", [])
    control_variables = be.get("control_variables", [])
    lower_boundary    = be.get("lower_boundary", None)
    upper_boundary    = be.get("upper_boundary", None)
    
    # Variables include control and state variables...
    # So, get control variables as the subset of variables.
    state_variables = [ x for x in var_names if not x in control_variables]
    
    # Instantiate class   . 
    dp = DynProg(functions=model.functions,m_var=var,m_par=par,options=options,state_variables=state_variables,
            control_variables=control_variables,utilities=utility_func,value_functions=value_func,
            eqs=eqs,be_eqs=be_eqs,lower_boundary=lower_boundary,upper_boundary=upper_boundary)
    
    # Solve Bellman equation
    iterations,crit,y = dp.value_iteration()
    # iterations,crit,y = dp.policy_iteration()
    # iterations,crit,y = dp.parametric_value_iteration()
    # iterations,crit,y = dp.stochastic_value_iteration()
    # iterations,crit,y = dp.stochastic_policy_iteration()
    
    elapsed = time() - t0
        
    return iterations,y.T,crit,elapsed 


if __name__ == '__main__':
    """The main entry point."""
    from snowdrop.src.driver import importModel

    fname = 'DP/ff.yaml'     # Bellman equation example
    fpath = os.path.dirname(os.path.abspath(__file__))
    path_to_dir = os.path.abspath(os.path.join(fpath,'../../../graphs'))
    file_path = os.path.abspath(os.path.join(fpath, '../../../models', fname))

    ## Create model object
    model = importModel(file_path)
    
    ## Run simulations
    iterations, y, crit, elapsed = simulate(model=model)

    # Plot Evolution of Endogenous Variables
    variable_names = model.symbols["variables"]
    var_labels = model.symbols.get("variables_labels",{})
    
    Plot(path_to_dir=path_to_dir,data=y,variable_names=variable_names,var_labels=var_labels)
    

    
