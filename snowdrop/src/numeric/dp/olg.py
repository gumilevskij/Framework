#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 08:40:21 2021

@author: A.Goumilevski
"""

import os
import numpy as np
import pandas as pd
from time import time
import scipy.sparse as sparse
from scipy.optimize import fmin
from scipy.linalg import pinv,norm
from snowdrop.src.utils.prettyTable import PrettyTable

NITERATIONS = 100 # Maximum nymber of iterations
EPSILON = 1.e-5   # Convergence tolerance
Nt = 10
Ngrid = 100
ngrid = 10

# Parameters
period_duration = 60/2       # We are assuming that average life expectancy is 60 years
alpha = 0.5                  # Cobb-Douglas productivity function parameter, exponent
beta  = 0.5                  # Impatience parameter
pi = 10                      # Cobb-Douglas productivity function parameter, normalization
phi = 0.9                    # Selfishness parameter
labor = 1                    # Unskilled labor multiplier
U = 1.03**period_duration    # Final machine productivity
u = np.ones(Nt); u[3:] = U   # Machine productivity
R = alpha*pi*((1-alpha)/alpha/U)**(alpha-1)

# Initialize variables
L = 1
N=np.zeros(Nt); Q=np.ones(Nt); QL=np.zeros(Nt); QM=np.zeros(Nt); QS=np.zeros(Nt); WL=np.zeros(Nt)
I1=np.zeros(Nt); I2=np.zeros(Nt); C1=np.zeros(Nt); C2=np.zeros(Nt); Sv=np.zeros(Nt)
B=np.zeros(Nt); U1=np.zeros(Nt); U2=np.zeros(Nt);  M=np.ones(Nt); S=np.ones(Nt)

    
def value_iteration(Output=False):
    """
    Solve Overlapping Generations Model by value function iteration.

    Parameters:
        Output: If True save results in excel file.
    
    Returns:
        iterations : int
            Number of iterations.
        crit : float
            Convergence value.
        y : numpy array
            Solution of Bellman equation
            
    """
    global u, M, S
    
    crit = 1.e6                              # convergence value
    iterations = 0                           # number of iterations
    dr = np.zeros((Ngrid,Ngrid),dtype=int)   # decision rule
    
    # Initial guess for value function
    Tv      = np.zeros((Ngrid,Ngrid))
    utility = np.zeros((Ngrid,Ngrid))
    x       = np.zeros((Ngrid,Ngrid))
    v       = 0
    sv      = 0
    
    # Need to seed M, typically with 'equilibrium' values.
    Mt,St = MSequilibrium(L,u[0]); 
    M *= Mt; S *= St
    
    # Define grid
    cy = np.linspace(1.e-6,10,Ngrid)
    co = np.linspace(1.e-6,100,Ngrid)
    
    for t in range(1,Nt):
        
        # The marginal products of current M,S determine the income/consumption
        # of the current old generation.  
        n  = L + u[t]*Mt
        s  = (1-alpha)*(sv+L/u[t])
        QM = dQdM(n,s,u[t])
        QS = dQdS(n,s)
        
        ### Solve Bellamn equation
        while crit > EPSILON:
            
            m = Mt; s = St; sv = 0
            # Utility
            for i in range(Ngrid):
                cy_g = cy[i]
                for j in range(Ngrid):
                    co_g = co[j]
                    utility  = phi*beta*np.log(cy_g) + phi*(1-beta)*np.log(co_g)
                    x[i,j]   = utility + (1-phi)*v 
                    # find new policy rule
                    Tv[i,j]  = np.max(x)
                    dr[i,j]  = np.argmax(x)
          
            # Decision rules
            for i in range(Ngrid):
                for j in range(Ngrid):
                    cy  = cy[dr[i,j]]
                    co  = co[dr[i,j]]
            
            # Update variables
            n  = L + u*m
            # Skills
            s  = (1-alpha)*(sv+L/u[t])
            # Wage rate
            w  = alpha * pi * (n/s)**(alpha-1)
            # Income of old
            io = QM*(m+s)
            # Bequest from old to young
            b  = np.max(0,io - co)
            # Income of young
            iy = w*L + b
            # Savings of youn
            sv = iy -  cy
            # sv = m + s
            # Perfect foresigh assumption
            m  = alpha*sv + (1-alpha)*L/u
            s  = (1-alpha)/alpha*n
  
            crit= np.max(abs(Tv-v))/norm(Tv)
            v   = Tv.copy()
            iterations += 1
            
        print('Iteration # {} \tCriterion: {:.2e}'.format(iterations,crit))
            
         # Compute intermediate product N:
        N[t] = L + u[t]*M[t]
        # Compute productivity and marginal products:
        Q[t]  = Qtot(N[t],S[t])
        QL[t] = dQdL(N[t],S[t])
        WL[t] = QL[t]*L
        
        # Compute interest factor. Use market clearing condition:
        R = QM[t]
        # Compute constant m:
        m = WL[t]*R/(R-1)
        
        # Old generation income is a return on machines and skilled labor 
        I2[t] = QM[t]*M[t] + QS[t]*S[t]
        # Compute bequest of old generation. 
        bequest = (1-phi)/(1-phi*beta)*I2[t] - m*phi*(1-beta)/(1-phi*beta)
        # It can not become negative.
        b[t] = min(I2[t],max(0,bequest))  
        # Compute consumption of old generation
        C2[t] = I2[t] - b[t]
        
        # Consumption of young generation:
        # Income of young generation is labor plus bequest
        I1[t] = WL[t] + b[t]
        C1[t] = beta*phi*(I1[t] + m/R)
        # Consumption can not be greater than income
        C1[t] = min(I1[t],C1[t])
        # Savings of young generation:
        Sv[t] = I1[t] - C1[t]
    
        # Perfect foresight assumption means the Young now guess the next M and
        # S values, using their unanticipated guess for u.  These become the
        # values of M,S for next period.  
        Mt,St = MSnext(L[t],Sv[t],u[t])
        M[t+1] = max(0,Mt)
        S[t+1] = max(0,St)
    
        # Utility definition
        U1[t] = beta*phi*np.log(C1[t])  # This is the current young utility
        U2[t] = (1-beta)*phi*np.log(C2[t])   # This is the current old utility
    
       
    # Compute Lifetime Utility as a sum of utilities for each generation:
    LU = list(U1[:-1] + U2[1:])
    LU.append(0)
    U = np.exp(LU)
    
    share_of_labor = (QL*L+QS*S)/Q
    
    # Display results:
    table = [m, S, QL, QS, C1, C2, Sv, Q, I1, I2, share_of_labor, LU, U, b]
    table = np.around(table, decimals=2)
    table = np.row_stack((u, table))
    table = np.row_stack((np.arange(Nt), table)) 
    
    header = ['T','u','m','Mg','Mp','TP','S','Wl','Ws','C','D','Sv','Q','Iy','Io','Share of Labor','LU','U','Bequest']
    pTable = PrettyTable(header)
    pTable.float_format = '6.2'
    for i in range(1,Nt-2):
        pTable.add_row(table[:,i])
 
    if Output:
        print()
        print('Bequest model:')
        print(pTable)
        print()
        
        table = table.T
        path = os.path.dirname(os.path.abspath(__file__))
        fname = path + '\\..\\data\\bequest.csv'
        rows,columns = table.shape
        with open(fname, 'w') as f:
            f.writelines(",".join(header) + "\n")
            for r in range(1,rows-2):
                for c in range(columns):
                    if c == columns:
                        f.writelines(table[r,c])
                    else:
                        f.write(str(table[r,c]) + ",")
                f.write("\n")
    
    var_names
    var_labels
    
    return iterations,crit,y,var_names,var_labels 
    
def MSequilibrium(Labor,u):
    # Find equilibrium M,S values.
    A = np.array([[1, 1], [-1, alpha/(1-alpha)]])
    WL = pi*alpha*(alpha*u/(1-alpha))**(alpha-1)
    B = np.array([(1-beta)*WL, Labor/u])
    M,S = np.linalg.solve(A,B)
    return M,S

def MSnext(Labor,Saving,u):
    # Find non-equilibrium M,S values.
    A = np.array([[1, 1], [-1, alpha/(1-alpha)]])
    B = np.array([ Saving, Labor/u ])
    M,S = np.linalg.solve(A,B)
    return M,S

def Qtot(N,S):
    # Compute total productivity Q.
    Q = pi * N**alpha * S**(1-alpha)
    return Q

def dQdL(N,S):
    # Compute marginal product w.r.t. L.
    QL = alpha * pi * (N/S)**(alpha-1)
    return QL

def dQdM(N,S,u):
    # Compute marginal product w.r.t. M.
    QM = alpha * pi * u * (N/S)**(alpha -1)
    return QM

def dQdS(N,S):
    # Compute marginal product w.r.t. S. 
    QS = (1-alpha) * pi * (N/S)**(alpha )
    return QS

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
    

if __name__ == '__main__':
    """The main entry point."""
    # Solve Bellman equation
    t0 = time()
    iterations,crit,y,var_names,var_labels = value_iteration()
    elapsed = time() - t0
    print("\nElapsed time: %.2f (seconds)" % elapsed)
    
    Plot(path_to_dir=path_to_dir,data=y,variable_names=variable_names,var_labels=var_labels)