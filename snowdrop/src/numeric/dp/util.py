"""
Created on Mon Feb  8 15:30:10 2021

Originally developed by Fabrice Collard.

Please see Dynamic Programming Notes at:
http://fabcol.free.fr/pdf/lectnotes7.pdf

Translated from Matlab code to Python by A. Goumilevski
"""
import numpy as np
from scipy.linalg import solve


def Chebychev_Polinomial(x,n):
    """
    Compute Chebychev polinomials.

    Parameters:
        x : numpy array.
            Nodes.
        n : int.
            Degree of polinomial.

    Returns:
        Tx : numpy array.
             Chebyshev Polinomial of order n.

    """
    assert n>0,'Degree of polinomial should be a positive integer.'
    
    # x=x[:]
    lx = 1 if np.isscalar(x) else len(x)
    if n == 0:
        Tx = np.ones(lx)
    elif n == 1:
        Tx = np.vstack((np.ones(lx),x))
    else:
        Tx = np.vstack((np.ones(lx),x))
        for i in range(2,n+1):
            temp = 2*x*Tx[i-1] - Tx[i-2]
            Tx = np.vstack((Tx,temp))

    return Tx.T


def gauss_herm(n):
    """
    Compute the coefficients of Hermite polynomials using the recursion:
    
    .. math::
        H_{n+1} = 2*H_{n} - 2*n*H_{n-1}

    Parameters:
        n : int
            Number of nodes.

    Returns:
        x : numpy array
            Gauss Hermite nodes
        w : numpy array
            Gauss Hermite weights

    """
    p0 = 1
    p1 = [2,0]
    for i in range(n-1):
       p = 2*(p1+[0])-2*i*([0]+[0]+p0)
       p0,p1 = p1,p
    p = np.array(p)

    # Compute the roots
    x = sorted(np.roots(p))
    # Compute the weights imposing that integration is exact for lower order polynomials
    A = np.zeros((n,n))
    A[0] = np.ones(n)
    A[1] = 2*x.T
    for i in range(n-2):
       A[i+2] = 2*x.T*A[i+1] - 2*i*A[i]

    w = solve(A, np.array([np.sqrt(np.pi),np.zeros(n-1)]))

    return (x,w)


def tausc_hussey(n=4,mx=0,rx=0.9,sigma=0.01):
    """
    Implement Tauchen-Hussey algorithm.
    
    It asymptotically reproduces AR(1) process

        .. math:: 
	    Z_{t+1} = (1-rx)*mx + rx*Z_{t} + eps_{t+1}
                    
	Here eps is a random process with normal distribution with standard deviation, sigma.
    
    Parameters:
        n : int, optional
            Number of nodes. The default is 4.
        mx : int, optional
            Unconditional mean of process. The default is 0.
        rx : float, optional
            Persistence coefficient. The default is 0.9.
        sigma : float, optional
            Standard deviation. The default is 0.01.

    Returns:
        p : numpy array
            Transition probabilities.

    """
    xx,wx = gauss_herm(n) # nodes and weights for x
    st = np.sqrt(2)*sigma*xx+mx
    x  = xx[:,:n,]
    y  = x.T
    w  = wx[:,:n].T
    p  = (np.exp(y*y-(y-rx*x)*(y-rx*x))*w)/np.sqrt(np.pi)
    sm = np.sum(p.T)
    p /= sm[:,:n]

    return p


def markov(PI,s,n,s0=1,seed=1):
    """
    Simulate a Markov chain.

    Parameters:
        PI: numpy array
            Transition matrix of Markov states.
        s : list
            State vector.
        n : int
            Length of simulation.
        s0: float, optional
            Initial state (index)
        seed : int, optional
            Random seed. The default is 1.

    Returns:
        chain : list
            Values for the simulated Markov chain
        state : numpy array
            Index of the state

    """
    from numpy.random import rand
    
    rpi,cpi = PI.shape
    #s=s[:]
    assert rpi==cpi,'Transition matrix must be square.'
    assert len(s)==cpi,'Number of states does not match size of Transition matrix.'
  
    cum_PI = np.vstack((np.zeros(rpi),np.cumsum(PI,axis=1).T)).T
    sim    = rand(n)
    state  = np.zeros(n,dtype=int)
    state[0] = s0
    
    for k in range(1,n):
        ind1 = sim[k] <= cum_PI[state[k-1],1:cpi+1]
        ind2 = sim[k] >  cum_PI[state[k-1],:cpi]
        ind = ind1 * ind2
        i = np.argwhere(ind==True)
        state[k] = i[0]
    
    chain = s[state]

    return (chain,state)


def scalar_tv(kp,alpha,beta,delta,sigma,kmin,kmax,n,k,theta):
    """Compute norm of a function."""
    v = tv(kp,alpha,beta,delta,sigma,kmin,kmax,n,k,theta)
    x = np.linalg.norm(v)
    return x

    
def tv(kp,alpha,beta,delta,sigma,kmin,kmax,n,k,theta):
    """
    Compute value function.

    Parameters:
        kp : float
            Grid nodes.

    Returns:
        res : float
            Minus value function.
            
    """
    kp = np.abs(kp)
    kp  = 2*(k-kmin)/(kmax-kmin) - 1
    ch = Chebychev_Polinomial(kp,n)
    v = ch @ theta
    c = k**alpha+(1-delta)*k-kp
    if np.isscalar(c):
        if c<=0:
            c = util = np.nan
            res = -1.e20
        else:
           util = (c**(1-sigma)-1)/(1-sigma) 
           res	= util + beta*v 
    else:
        ind	= (c<=0)
        util = (c**(1-sigma)-1)/(1-sigma) 
        util[ind]=-1.e20
        res	= util + beta*v
        
    res = -res
    #print(kp,np.max(theta),res)
    
    return res