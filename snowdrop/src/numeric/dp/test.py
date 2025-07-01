#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 17:50:30 2021

Originally developed by Fabrice Collard.

Please see Dynamic Programming Notes #7 at:
http://fabcol.free.fr/pdf/lectnotes7.pdf

Translated from Matlab code to Python by A.Goumilevski
"""

import os, math
from time import time
import numpy as np
from numpy.matlib import repmat
from scipy.optimize import fmin
from scipy.interpolate import CubicSpline
import scipy.sparse as sparse
from scipy.linalg import pinv
from scipy.sparse.linalg import spsolve
from snowdrop.src.numeric.dp.util import Chebychev_Polinomial, tv, scalar_tv, markov
from snowdrop.src.graphs.util import plot

sigma   = 1.50                     # utility parameter
delta   = 0.10                     # depreciation rate
beta    = 0.95                     # discount factor
alpha   = 0.30                     # capital elasticity of output

nbk     = 100                      # number of data points in the grid
nbc     = 10
nba     = 2
crit    = 1                        # convergence criterion
itr     = 0                        # iteration
epsi    = 1e-6                     # convergence tolerance

ks      = ((1-beta*(1-delta))/(alpha*beta))**(1/(alpha-1))
dev     = 0.9                      # maximal deviation from steady state
kmin    = (1-dev)*ks               # lower bound on the grid
kmax    = (1+dev)*ks               # upper bound on the grid
devk    = (kmax-kmin)/(nbk-1)      # implied increment
kgrid   = np.linspace(kmin,kmax,nbk)  # build the grid
v       = np.zeros(nbk)            # value function
kp0     = 0                        # initial guess on k(t+1)
dr      = np.zeros(nbk,dtype=int)  # decision rule (will contain indices)

cmin    = 0.01                     # lower bound on the grid
cmax    = kmax**alpha              # upper bound on the grid
c       = np.linspace(cmin,cmax,nbc)  # build the grid
Tv      = np.zeros(nbk)            # value function

u       = (c**(1-sigma)-1)/(1-sigma)
k		= np.linspace(kmin,kmax,nbk)


def value_iteration(interpolate=False):
    """
    Solve the OGM by value function iteration.

    Parameters:
        interpolate : bool, optional
            If True performs interpolation of value function to grid nodes. 
            The default is False.

    """
    global itr, crit, nbk, kp0, kgrid, v, dr
    t0 = time()
    
    if interpolate:
        
        nbc = 1000; nbk = 100
        c     = np.linspace(cmin,cmax,nbc)  # builds the grid
        kgrid = np.linspace(kmin,kmax,nbk)  # builds the grid
        Tv    = np.zeros(nbk)               # value function
        csy	  = 1-alpha*beta*delta/(1-beta*(1-delta))
        v     = ((csy*kgrid**alpha)**(1-sigma)-1)/(1-sigma) # value function
        u     = (c**(1-sigma)-1)/(1-sigma)
        dr    = np.zeros(nbk,dtype=int) 
        
        ### Main loop
        while crit>epsi:
            for i in range(nbk):
                 kp  = kgrid[i]**alpha+(1-delta)*kgrid[i]-c
                 cs  = CubicSpline(kgrid,v)
                 vi  = cs(kp)
                 x   = u + beta*vi
                 Tv[i] = np.max(x)
                 dr[i] = np.argmax(x)
               
            crit = np.max(abs(Tv-v))   # Compute convergence criterion
            v    = Tv.copy()            # Update the value function
            itr += 1
            print('Iteration # {0} \tCriterion: {1}'.format(itr,crit))
              
        c	= c[dr]
        kp = kgrid**alpha+(1-delta)*kgrid-c
        u	= (c**(1-sigma)-1)/(1-sigma)
        v   = u/(1-beta)
        
    else:
        
        ### Main loop
        while crit>epsi:
            for i in range(nbk):
                # compute indexes for which consumption is positive
                imin = max(math.ceil(((1-delta)*kgrid[i]-kmin)/devk)+1,0)
                imax = min(math.floor((kgrid[i]**alpha+(1-delta)*kgrid[i]-kmin)/devk)+1,nbk)
                
                # consumption and utility
                c  = kgrid[i]**alpha+(1-delta)*kgrid[i]-kgrid[imin:imax]
                u  = (c**(1-sigma)-1)/(1-sigma)
                # find new policy rule
                x = u + beta*v[imin:imax]
                idr  = np.argmax(x)
                dr[i] = imin+idr
          
            # decision rules
            kp  = kgrid[dr]
            c   = kgrid**alpha+(1-delta)*kgrid-kp
            # update the value
            u = (c**(1-sigma)-1)/(1-sigma)
            Q   = sparse.lil_matrix((nbk,nbk))
            for i in range(nbk):
                Q[i,dr[i]] = 1
            
            Tv  = spsolve(sparse.eye(nbk)-beta*Q, u)
            crit= np.max(abs(kp-kp0))
            v   = Tv.copy()
            kp0 = kp
            itr += 1
            print('Iteration # {0} \tCriterion: {1}'.format(itr,crit))
    
    elapsed = time() - t0
    print("Elapsed time: %.2f (seconds)" % elapsed)
    
    Plot(header="Value Function Iteration",titles=["Consumption","Capital","Utility","Value function"],data=(c,kp,u,v))
    
    return (c,kp,u,v)
    
    
def policy_iteration():
    """
    Solve the OGM by policy iteration.

    Returns:
        c : numpy array
            Consumption.
        kp : numpy array
            Capital.
        u : TYPE
            Utility.
        v : numpy array
            Value function.
            
    """
    global itr, crit, c, u, v, Tv, kp0
    
    t0 = time()
    ### Main loop
    while crit>epsi:
      for i in range(nbk):
            # compute indexes for which consumption is positive
            imin  = max(math.ceil(((1-delta)*kgrid[i]-kmin)/devk)+1,0)
            imax  = min(math.floor((kgrid[i]**alpha+(1-delta)*kgrid[i]-kmin)/devk)+1,nbk)
            # consumption and utility
            c     = kgrid[i]**alpha+(1-delta)*kgrid[i]-kgrid[imin:imax]
            u     = (c**(1-sigma)-1)/(1-sigma)
            # find new policy rule
            x = u+beta*v[imin:imax]
            #vi = np.max(x)
            idr = np.argmax(x)
            dr[i] = imin+idr
         
      
      ### Decision rules
      kp  = kgrid[dr]
      c   = kgrid**alpha+(1-delta)*kgrid-kp
      #Update the value
      u = (c**(1-sigma)-1)/(1-sigma)
      Q = sparse.lil_matrix((nbk,nbk))
      for i in range(nbk):
            Q[i,dr[i]] = 1
        
      Tv  = spsolve(sparse.eye(nbk)-beta*Q, u)
      crit= max(abs(kp-kp0))
      v   = Tv.copy()
      kp0 = kp
        
    itr += 1
    
    elapsed = time() - t0
    print("Elapsed time: %.2f (seconds)" % elapsed)
    
    Plot(header="Policy Iteration",titles=["Consumption","Capital","Utility","Value function"],data=(c,kp,u,v))
    
    return (c,kp,u,v)


def parametric_value_iteration(n=10, nbk = 20):
    """
    Solve the OGM by parametric value iteration.

    Parameters:
        n : int, optional
            Order of polynomials. The default is 10.
        nbk : int, optional
            Number of data points in the grid. The default is 20.

    Returns:
        c : numpy array
            Consumption.
        v : numpy array
            Value function.
            
    """
    global itr, crit, c, u, v, Tv, kp0
    nbk = 20                                                         # Number of data points in the grid
    
    t0 = time() 
    rk		= -np.cos((2*np.arange(1,nbk+1)-1)*np.pi/(2.*nbk))       # Interpolating nodes
    kgrid	= kmin+(rk+1)*(kmax-kmin)/2                              # Mapping
    v		= (((kgrid**alpha)**(1-sigma)-1)/((1-sigma)*(1-beta)))   # Initial guess for the approximation
    X		= Chebychev_Polinomial(rk,nbk)                                      # Chebyshev polynomials
    iX      = pinv(X)
    theta0	= pinv(X) @ v                                            # Polynomial decomposition
    Tv		= np.zeros(nbk)
    kp		= np.zeros(nbk)
    
    ### Main loop
    while crit>epsi:
        k0 = kgrid.copy()    
        param = (alpha,beta,delta,sigma,kmin,kmax,nbk,kgrid,theta0)
        Tv = -tv(k0,*param)
        k0 = fmin(scalar_tv,k0,args=param,xtol=1.e-3,ftol=1.e-3,maxiter=20,disp=False)
        Tv = -tv(k0,*param)
        
        theta0 = iX @ Tv
        crit = np.max(abs(Tv-v))
        print('Iteration # {0} \tCriterion: {1}'.format(itr,crit))
        v = Tv.copy()
        itr += 1
    
    kp = k0
    c = kgrid**alpha+(1-delta)*kgrid-kp
    
    elapsed = time() - t0
    print("Elapsed time: %.2f (seconds)" % elapsed)
        
    Plot(header="Parametric Iteration",titles=["Consumption","Capital","Value function"],data=(c,kp,v))
  
    return c,kp,v


def stochastic_value_iteration(p=0.9,interpolate=False):
    """
    Solve the stochastic OGM by value iteration.

    Parameters:
        p : float, optional
            Probability value. The default is 0.9.
            It is used for discrete approximation of VAR(1) stochastic process.
        interpolate : bool, optional
            If True interpolate grid. The default is False.

    Returns:
        c : numpy array
            Consumption.
        u : numpy array
            Utility.
        v : numpy array
            Value function.
            
    """
    global crit, itr
    
    t0 = time()
    PI		= np.array([[p,1-p],[1-p,p]])
    kmin	= 0.2 #(1-dev)*ks;
    kmax	= 6
    se		= 0.2
    ab		= 0
    A       = [np.exp(ab-se), np.exp(ab+se)]
    k		= np.linspace(kmin,kmax,nbk)
    u		= np.zeros((nbk,nba))
    v		= repmat(k**(alpha*(1-sigma))/(1-sigma),nba,1).T
    Tv		= np.zeros((nbk,nba))
    dr		= np.zeros((nbk,nba),dtype=int)
 
    if interpolate:
        c = np.linspace(0.01,A[1]*kmax**alpha,nbk)
        u = (c**(1-sigma)-1)/(1-sigma)
        
        while crit>epsi:
           for i in range(nbk):
              for j in range(nba):
                 kp	= A[j]*k[i]**alpha + (1-delta)*k[i] - c
                 cs  = CubicSpline(k,v)
                 vi  = cs(kp)
                 x = u + beta*(vi @ PI[j])
                 Tv[i] = np.max(x)
                 dr[i] = np.argmax(x)
      
           crit	= np.max(np.max(abs(Tv-v)))
           print(crit)
           v = Tv.copy()
           itr += 1
        
        kp = np.zeros((nbk,nba))
        ct = np.zeros((nbk,nba))
        for j in range(nba):
            ct[:,j] = c[dr[:,j]]
            kp[:,j]	= A[j]*k**alpha + (1-delta)*k - ct[:,j]
            
        c = ct

    else:
        
        c = np.zeros((nbk,nba))
        while crit>epsi:
           for i in range(nbk):
              for j in range(nba):
                 x	= A[j]*k[i]**alpha + (1-delta)*k[i] - k
                 neg	= (x<=0)    
                 x[neg]	= np.nan
                 u[:,j]	= (x**(1-sigma)-1)/(1-sigma)
                 u[neg,j] = -1.e12
              
              x = u + beta*(v @ PI)
              Tv[i] = np.max(x,axis=0)
              dr[i]	= np.argmax(x,axis=0)
        
           crit	= np.max(np.max(abs(Tv-v)))
           print(crit)
           v 	= Tv.copy()
           itr	+= 1
        
        kp = k[dr]
        for j in range(nba):
           c[:,j] = A[j]*k**alpha + (1-delta)*k - kp[:,j]
      
        u	= (c**(1-sigma)-1)/(1-sigma)
        v	= u/(1-beta)

    elapsed = time() - t0
    print("Elapsed time: %.2f (seconds)" % elapsed)
        
    Plot(header="Stochastic Value Iteration",titles=["Consumption","Capital Stock","Value function"],data=(c,kp,v))
  
    return (c,kp,v)  
    

def stochastic_policy_iteration(p=0.9,kmin=0.2,kmax=6):
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
        c : numpy array
            Consumption.
        kp : numpy array
            Capital.
        v : numpy array
            Value function.
            
    """
    global crit, itr, kgrid
    
    t0 = time()
    PI	 = np.array([[p,1-p],[1-p,p]])
    se	 = 0.2
    ab   = 0
    A    = np.array([np.exp(ab-se), np.exp(ab+se)])
    v    = np.zeros((nbk,nba))             # value function
    u    = np.zeros((nbk,nba))             # utility function
    c    = np.zeros((nbk,nba))             # consumption
    kgrid= np.linspace(kmin,kmax,nbk)      # builds the grid
    dr   = np.zeros((nbk,nba),dtype=int)   # decision rule (will contain indices)
    kp0  = 0
    Tv0  = 1
    
    # Main loop
    while crit > epsi:
        for i in range(nbk):
            for j in range(nba):
               x  = A[j]*kgrid[i]**alpha + (1-delta)*kgrid[i] - kgrid
               neg	= (x<=0 )   
               x[neg] = np.nan
               u[:,j]   = (x**(1-sigma)-1)/(1-sigma)
               u[neg,j] = -np.inf
           
            x = u + beta * v @ PI
            # vi = np.max(x,axis=0)
            dr[i] = np.argmax(x,axis=0)
       
        # decision rules
        kp  = kgrid[dr]
      
        # update the value
        Q  = sparse.lil_matrix((nbk*nba,nbk*nba))
        for j in range(nba):
            z	= A[j]*kgrid**alpha + (1-delta)*kgrid - kp[:,j]
            # Update the value
            u[:,j] = (z**(1-sigma)-1)/(1-sigma)
            
            Q0  = sparse.lil_matrix((nbk,nbk))
            for i in range(nbk):
               Q0[i,dr[i,j]] = 1   
               
            x = sparse.kron(PI[j],Q0)
            Q[j*nbk:(j+1)*nbk,:] = x
   
          
        M = sparse.eye(nbk*nba) - beta*Q
        Tv = spsolve(M,np.ravel(u.T))
        v  = np.reshape(Tv,(nba,nbk)).T
        crit= np.max(abs(Tv-Tv0))
        crit= np.max(np.max(abs(kp-kp0)))
        print(crit)
        Tv0 = Tv.copy()
        kp0 = kp.copy()
        itr += 1
    
    for j in range(nba):
        c[:,j]	= A[j]*kgrid**alpha + (1-delta)*kgrid - kp[:,j]
        
    elapsed = time() - t0
    print("Elapsed time: %.2f (seconds)" % elapsed)
     
    Plot(header="Stochastic Policy Iteration",titles=["Consumption","Capital","Utility","Value function"],data=(c,kp,u,v))
  
    return (c,kp,u,v)
    

def stochastic_constarined_policy_iteration(p=0.8,r=0.02,beta=0.95,gam=0.5,amin=0,amax=10):
    """
    Solve and simulate the borrowing constraint problem (policy iteration).

    Parameters:
        p : float, optional
            Probability value. The default is 0.8.
            It is used for discrete approximation of VAR(1) stochastic process.
        r : float, optional
            Interest rate. The default is 0.02.
        beta : float, optional
            Discount factor. The default is 0.95.
        gam : float, optional
            Replacement ratio. The default is 0.5.
        amin : float, optional
            Lower bound on the grid. The default is 0.
        amax : float, optional
            Upper bound on the grid. The default is 10.

    Returns:
        c : numpy array
            Consumption.
        k : numpy array
            Capital.
        ast : numpy array
            Asset values.
        v : numpy array
            Value function.
            
    """
    global crit, itr
    
    t0 = time()
    w		= 1                             # employment benefits
    PI		= np.array([[p,1-p],[1-p,p]])   # matrix of probablities
    Om		= np.array([gam*w, w])
    agrid   = np.linspace(amin,amax,nbk)    # builds the grid
    v       = np.zeros((nbk,nba))           # value function
    u       = np.zeros((nbk,nba))           # value function
    ap0     = 1                             # initial guess on k(t+1)
    c       = np.zeros((nbk,nba))           # consumption
    util    = np.zeros((nbk,nba))           # value function
    u       = []
    dr      = np.zeros((nbk,nba),dtype=int) # decision rule (will contain indices)
    
    for j in range(nba):
        x1 = repmat((1+r)*agrid + Om[j], nbk,1)
        x2 = repmat(agrid, nbk,1)
        x  = x1 - x2.T
        neg	= (x<=0)
        x[neg]  = np.nan
        uj = (x**(1-sigma)-1)/(1-sigma)
        uj[neg] = -np.inf
        u.append(uj.copy())

    # Main loop
    while crit > epsi:
       for j in range(nba):
           x = u[j] + beta*repmat(v@PI[j],nbk,1).T
           dr[:,j] = np.argmax(x,axis=0)
       
       # Decision rules
       ap  = agrid[dr]
       # Update the value
       Q  = sparse.lil_matrix((nbk*nba,nbk*nba))
       for j in range(nba):
          for i in range(nbk):
              util[i,j] = u[j][dr[i,j],i]
              
          Q0  = sparse.lil_matrix((nbk,nbk))
          for i in range(nbk):
             Q0[i,dr[i,j]] = 1
         
          Q[j*nbk:(j+1)*nbk] = sparse.kron(PI[j],Q0)
     
       Tv  = spsolve(sparse.eye(nbk*nba)-beta*Q, np.ravel(util.T))
       crit= np.max(abs(ap-ap0))
       print('Iteration # {0} \tCriterion: {1}'.format(itr,crit))
       v   = np.reshape(Tv,(nba,nbk)).T
       ap0 = ap.copy()
       itr += 1
       
    for j in range(nba):
       c[:,j] = (1+r)*agrid + Om[j] - ap[:,j]

    n  = 7
    x  = 2*(agrid-amin)/(amax-amin)-1
    Ta = Chebychev_Polinomial(x,n)
    ba = pinv(Ta) @ ap
    
    ws,ids = markov(PI,Om,nbk,1)
    ast = np.ones(nbk+1)
    cs = np.zeros(nbk)
    x = 2*(ast-amin)/(amax-amin)-1
    Ta = Chebychev_Polinomial(x,n)
    
    for t in range(nbk):
        ast[t+1]  = Ta[t] @ ba[:,ids[t]]
        cs[t] = (1+r)*ast[t] + ws[t] - ast[t+1]
   
    elapsed = time() - t0
    print("Elapsed time: %.2f (seconds)" % elapsed)
         
    Plot(header="Borrowing Constraint Problem (Policy Iteration)",titles=["Stochastic Consumption","Stochastic Assets"],data=(cs,ast[:-1]))
    Plot(header="Borrowing Constraint Problem (Policy Iteration)",titles=["Consumption","Value Function"],data=(c,v))
   
    return (cs,k,ast,v)


def Plot(header,titles,data,labels=[],save=True):
    """
    Plot graphs.

    Parameters:
        header : str.
            Graphs header.
        titles : str,
            Titles.
        data : numpy array
            Data.
        labels : list, optional
            Labels. The default is [].
        save : TYPE, optional
            If True saves plots in a directory. The default is True.
        
    """
    path = os.path.dirname(os.path.abspath(__file__))
    path_to_dir = os.path.abspath(os.path.join(path,"../../../graphs"))
    
    data = np.array(data)
    ndim = np.ndim(data)
    if ndim == 3:
        n,nx,na = data.shape
        y = []
        for i in range(n):
            for k in range(na):
                y.append(data[i,:,k])
        data = np.array(y)
        var_names = []
        for x in titles:
            for k in range(na):
                var_names.append(x + " (state " + str(k+1) + ")")
        titles = var_names
                    
    print("Plotting graphs for " + header)
    plot(path_to_dir=path_to_dir,data=[data.T],variable_names=titles,save=save)
    
        
if __name__ == '__main__':
    """The main entry point."""
    #c,kp,u,v = value_iteration(interpolate=False)
    #c,kp,u,v = policy_iteration()
    #c,kp,v = parametric_value_iteration()
    c,kp,v = stochastic_value_iteration(interpolate=True)
    #c,kp,u,v = stochastic_policy_iteration()
    #c,k,ast,v = stochastic_constarined_policy_iteration()
      
    print("Done!")
