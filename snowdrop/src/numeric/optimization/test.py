#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 00:09:58 2022

@author: A.Goumilevski
"""
import numpy as np
from scipy.optimize import root
from compecon import MCP

it = 0
# Function
def func(x):
    f = (a*x-xc)**2
    return f

# Function
def jacob(x):
    j = np.zeros((n,n))
    for i in range(n):
        j[i,i] = 2*a[i]*(x[i]-xc[i])
    return j

def fun(x):
    f = func(x)
    jac = jacob(x)
    # f = path_func(x)
    # jac = path_jacob(x)
    return f,jac

# Mapping function
def map_func(a,b):
    return np.sqrt(a*a+b*b)-a-b
 
 # Mapping function derivatives
def map_func_der(a,b):
    return a/np.sqrt(a*a+b*b)-1
 
 # MCP Function
def path_func(x):
    global it
    if np.any(x<lower) or np.any(x>upper):
        it += 1
        #return 1.e5 + it + np.zeros(n)
    f = func(x)
    if bool(Il) and bool(Iu):
        a1 = np.array([upper[i]-x[i] if not np.isinf(upper[i]) else -x[i] for i in range(n)])
        b1 = -f
        b = map_func(a1,b1)
        a = np.array([x[i]-lower[i] if not np.isinf(lower[i]) else x[i] for i in range(n)])
        y = map_func(a,b)
    elif bool(Il):
        a = np.array([x[i]-lower[i] if not np.isinf(lower[i]) else x[i] for i in range(n)])
        b = f
        y = map_func(a,b)
    elif bool(Iu):
        a = np.array([upper[i]-x[i] if not np.isinf(upper[i]) else -x[i] for i in range(n)])
        b = -f
        y = -map_func(a,b)
    else:
        y = -f
    return y
    
# MCP Function Jacobian.
def path_jacob(x):
    I = np.eye(n)
    f = func(x)
    jac = jacob(x)
    if bool(Il) and bool(Iu):
        a1 = np.array([upper[i]-x[i] if not np.isinf(upper[i]) else -x[i] for i in range(n)])
        b1 = -f
        b = map_func(a1,b1)
        a = np.array([x[i]-lower[i] if not np.isinf(lower[i]) else x[i] for i in range(n)])
        z = map_func_der(a1,b1)*I + map_func_der(b1,a1)*jac
        y = map_func_der(a,b)*I - map_func_der(b,a)*z
    elif bool(Il):
        a = np.array([x[i]-lower[i] if not np.isinf(lower[i]) else x[i] for i in range(n)])
        b = f
        y = map_func_der(a,b)*I + map_func_der(b,a)*jac
    elif bool(Iu):
        a = np.array([upper[i]-x[i] if not np.isinf(upper[i]) else -x[i] for i in range(n)])
        b = -f
        y = map_func_der(a,b)*I + map_func_der(b,a)*jac
    else:
        y = -jac
    return y

def jacobian(x):
    z = np.empty((n,n))
    f1 = path_func(x)
    for i in range(n):
        for j in range(n):
            x2 = x
            x2[j] += delta
            f2 = path_func(x2)
            z[i,j] = (f2[j]-f1[j])/delta
    return z

if __name__ == '__main__':
    """
    The main test program.
    """
    n = 3
    a = np.array([1,1,1])
    xc = np.array([1,15,2])
    x0 = np.array([-1,1,70])

    lower = np.array([-1,-10,-10])
    upper = np.array([10,10,10]) 
    Il = np.any(x0<lower)
    Iu = np.any(x0>upper)
    z1 = np.empty((n,n))
    delta = 1.e-6

    results = root(fun=path_func,x0=x0,jac=None,method="lm",tol=1.e-8)
    x = results.x;
    print('status:',results.success,x,"\n")
    print(results,"\n")
    print(jacobian(x),"\n")
    print(path_jacob(x),"\n")
    #'hybr','lm','broyden1','broyden2','anderson','linearmixing','diagbroyden','excitingmixing','krylov','df-sane'

    F = MCP(f=fun,a=lower,b=upper,x0=x0,maxit=10000)
    # Using minmax formulation
    x1 = F.zero(transform='minmax')
    print(x1,fun(x1)[0])
    # Using semismooth formulation
    x2 = F.zero(transform='ssmooth')
    print(x2,fun(x2)[0])
