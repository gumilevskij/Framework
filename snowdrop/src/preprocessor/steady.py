#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 09:21:50 2021

@author: alexei
"""
from numpy import nan

instance = None

class SteadyStates:
    
    ss = None
    
    def __init__(self,dict_steady_states=None):
        """
        Model constructor.

        Parameters:
            dict_steady_states : dict
                Map of variables names and steady state values.

        """
        self.ss = dict_steady_states
        
        
    def steady_state(self,x):
        """
        Returns steady state value of a variable.
    
        Parameters:
            x : str.
                Variable name.
    
        Returns:
            Steady state value.
    
        """
        if not self.ss is None:
            if x in self.ss:
                return self.ss[x]
            elif hasattr(self.ss,x):
                return self.ss.x
                
        return x
    
    
def STEADY_STATE(x,dict_steady_states=None):
    """
    Wrapper for a "steady_state" function of a "steady_states" class.

    Parameters:
        x : str.
            Variable name.
        dict_steady_states : dict
            Map with variables names and steady state values.

    Returns:
        Steady state value.

    """
    global instance
    
    if instance is None:
        instance = SteadyStates(dict_steady_states)
        
    return instance.steady_state(x)


def Derivative(f,x):
    """
    By definition, derivative of a steady state function is zero.

    Parameters:
        f : str.
            Function name, f = f(x).
        x : str.
            Variable name.

    Returns:
        Zero value.

    """
    return 0
    
    
