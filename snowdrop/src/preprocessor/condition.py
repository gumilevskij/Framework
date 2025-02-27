#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 08:47:07 2021

@author: alexei
"""
import numpy as np
from scipy.special import erf

def log(x):
    """Returns negative number if argment is negative."""
    if x <= 0:
        return -100
    else:
        return np.log(x)
    
def Subs(x,*args):
    """Nothing to substitute."""
    return x

def Derivative(f,x):
    """
    Returns derivative of a constant, i.e. zero

    Parameters:
        f : function
            Function
        x : float
            variable.

    Returns:
        None.

    """
    return 0

def Positive(x):
    """
    Returns True if argument is non-negative

    Parameters:
        x : float
            variable.

    Returns:
        None.

    """
    return (x>=0)

def Negative(x):
    """
    Returns True if argument is non-negative

    Parameters:
        x : float
            variable.

    Returns:
        None.

    """
    return (x<0)

def IfThen(condition,x):
    """
    Checks condition and returns x or 0.

    Parameters:
        condition : bool
            Condition.
        x : float
            Variable value.

    Returns:
        Variable x value if condition is satisfied and 0 if not.

    """
    if np.isnan(condition):
        return np.nan
    if condition >= 0:
        return x
    else:
        return 0   
    
    
def IfThenElse(condition,a,b):
    """
    Checks condition and returns a or b.

    Parameters:
        condition : bool
            Condition.
        a : float
            Variable value.
        b : float
            Variable value.

    Returns:
        Variable 'a' value if condition is satisfied and variable 'b' value if not.

    """
    if np.isnan(condition):
        return np.nan
    if condition >= 0:
        return a
    else:
        return b
    
def myzif(x):
    """
    Return user defined function value.

    Parameters:
        x : float
            Variable value.
    """
    y = x*(1+erf(3.*x))/2.
    return y
    
