# -*- coding: utf-8 -*-
"""
Created on Fri Nov 1, 2019

@author: agoumilevski
"""
import numpy as np
import scipy.linalg as la
from snowdrop.src.misc.termcolor import cprint
from snowdrop.src.numeric.solver.util import getParameters
from snowdrop.src.preprocessor.function import get_function_and_jacobian

count = 1       

def solve_am(model,steady_state,p=None,suppress_warnings=False):
    """
    Find first-order accuracy approximation solution.
    
    It implements an algoritm of Anderson and Moore.
    
    Please see "Anderson-Moore Algorithm (AMA)", https://www.federalreserve.gov/econres/ama-index.htm
    
    Parameters:
        :param model: The Model object.
        :type model: instance of class Model.
        :param steady_state: Steady state.
        :type steady_state: list.
        :param p: List of parameters.
        :type p: list.
        :param suppress_warnings: Do not show warnings if True
        :type suppress_warnings: bool.
    """
    from snowdrop.src.numeric.solver.AIM.AIMsolver import AIMsolver

    global count
    if count == 1:
        count += 1
        cprint("Anderson-Moore solver","blue")
        print()
    
    A, C, R, Z, W = None, None, None, None, None
    p = getParameters(parameters=p,model=model)
    
    try:
        #Find jacobian
        z = np.vstack((steady_state,steady_state,steady_state))
        c,jacob = get_function_and_jacobian(model,params=p,y=z,order=1)
        n = len(jacob)
        if  model.max_lead == 0 and model.min_lag < 0:
            C = jacob[:,n:2*n]
            L = jacob[:,2*n:3*n] 
            R = jacob[:,3*n:]
            C_inv = la.inv(C)
            A = -C_inv @ L
            R = -C_inv @ R
            C = -C_inv @ c
            phi = -C_inv
        elif  model.max_lead > 0 and model.min_lag < 0:
            A,C,R,phi,Z,aimcode,rts = AIMsolver(jacobian=jacob,c=c,model=model,suppress_warnings=suppress_warnings)
        else:
            cprint('AIM algorithm requires system equations to have leads and lags','red')
            raise ValueError('Model must have at least one lag and one lead.')
            
    except :
        if not suppress_warnings:
            import sys
            sys.exit('Anderson-Moore solver: unable to find solution of linear model.')
                                
    model.linear_model["A"]  = A
    model.linear_model["R"]  = R
    model.linear_model["C"]  = C
    model.linear_model["Z"]  = Z
    model.linear_model["W"]  = phi
    
    return model