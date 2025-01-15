#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 17:47:40 2021

@author: A.Goumilevski
"""
import numpy as np

def Heaviside(x):
    return np.heaviside(x,1)

def Min(*args):
    return min(*args)

def Max(*args):
    return max(*args)

def Abs(x):
    return abs(x)

def DiracDelta(x):
    if x==0:
        return np.inf
    else:
        return 0
    
def PNORM(x,mean=0.0,std=1.0):
    """Troll 'PNORM' normal distribution function."""
    return np.exp(-0.5*(x-mean)*(x-mean)/std)
    
def log(x):
    if x <= 0:
        return -1.e10
    else:
        return np.log(x)
    
