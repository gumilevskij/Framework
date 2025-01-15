# -*- coding: utf-8 -*-
"""
Created on Fri Nov 1, 2019

@author: agoumilevski
"""
from __future__ import division

import numpy as np
import scipy.linalg as la
from sys import exit
from snowdrop.src.misc.termcolor import cprint
from snowdrop.src.numeric.solver.util import getParameters
from snowdrop.src.numeric.solver.util import sorter
from snowdrop.src.preprocessor.function import get_function_and_jacobian
from snowdrop.src.utils.equations import getLeadLagIncidence

count = 1

def solve_villemot(model,steady_state,p=None,suppress_warnings=False):
    """
    Finds first-order accuracy approximation solution.
    
    It implements an algoritm described by Sebastien Villemot in his paper:
    "Solving rational expectations model at first order: what Dynare does."
    For references please see https://www.dynare.org/wp-repo/dynarewp002.pdf
    
    Parameters:
        :param model: The Model object.
        :type model: Instance of a class Model.
        :param steady_state: Steady state.
        :type steady_state: list.
        :param p: Model parameters.
        :type p: list.
        :param suppress_warnings: Do not show warnings if True
        :type suppress_warnings: bool.
    """ 
    global count
    if count == 1:
        count += 1
        cprint("Sebastien Villemot solver","blue")
        print()
    
    A,C,R,Z,Xa,J,Ru,U = None,None,None,None,None,None,None,None
    p = getParameters(parameters=p,model=model)
    
    try:
        #Find jacobian
        z = np.vstack((steady_state,steady_state,steady_state))
        K,jacob = get_function_and_jacobian(model,params=p,y=z,order=1)
        n = len(jacob)
        if  model.max_lead == 0 and model.min_lag < 0:
            C = jacob[:,n:2*n]
            L = jacob[:,2*n:3*n]
            R = jacob[:,3*n:]
            C_inv = la.inv(C)
            A = -C_inv @ L
            R = -C_inv @ R
            C = -C_inv @ K
            model.linear_model["A"]   = A
            model.linear_model["C"]   = C
            model.linear_model["R"]   = R
            model.linear_model["Xa"]  = 0*C
            model.linear_model["Ru"]  = 0*R
            model.linear_model["J"]   = 0*A
            model.linear_model["U"]   = 0*A
            model.linear_model["Z"] = -C_inv 
            
        else:
            jac = jacob[:,:3*n]
            r = jacob[:,3*n:]
            U,A,C,R,Z,Xa,J,Ru = solution(model=model,Jacob=jac,Psi=-r,K=K,suppress_warnings=suppress_warnings)
            model.linear_model["A"]   = A
            model.linear_model["C"]   = C
            model.linear_model["R"]   = R
            model.linear_model["Ru"]  = Ru
            model.linear_model["Z"]   = np.real(Z)
            model.linear_model["Xa"]  = Xa
            model.linear_model["J"]   = J
            model.linear_model["U"]   = U
            
    except :
        if not suppress_warnings:
            exit('Villemot solver: unable to find solution of a linear model.\n   Please use other solvers: BinderPesaran, AndersonMoore, ...')
                
    return model



def solution(model,Jacob,Psi,K=None,qz_factor=1+1.e-6,debug=False,suppress_warnings=False):
    """
    Find first-order accuracy approximation solution.
    
    It implements an algoritm described by Sebastien Villemot in the paper:
    "Solving rational expectations model at first order: what Dynare does."
    For references please see https://www.dynare.org/wp-repo/dynarewp002.pdf
    
    Parameters:
        :param model: The Model object.
        :type model: instance of class Model.
        :param jacob: Jacobian matrix.
        :type jacob: numpy.ndarray.
        :param Psi: Matrix of coefficients of shocks.
        :type Psi: numpy.ndarray.
        :param C: Array of constants.
        :type C: numpy.ndarray.
    """
    debug &= suppress_warnings
    
    variables = model.symbols["variables"]
    n = len(Jacob)
    n_shock = Psi.shape[1]
    
#    n_shocks = Psi.shape[1]
    A,B,R = None,None,None
    
    if K is None:
        K = np.zeros(n)
    
    # Retrieve lead-lag incidence matrix
    lli = model.lead_lag_incidence 
    if lli is None:
        lli = getLeadLagIncidence(model)
    
    forward_columns = np.array([int(f) for f in lli[2] if not np.isnan(f)]) 
    forward_variables = [x for x in model.topology["forward"].keys() if not "_p_" in x]
    forward_variables = [variables[i] for i in forward_columns]
    N_forward = len(forward_columns)
    N = n + N_forward
    
    ### ------------------- Make structural state-space repesentation of equations
        
    Psi1 = np.concatenate((Psi,np.zeros((N_forward,n_shock))))
    K1 = np.concatenate((K,np.zeros(N_forward)))
    
    # Build matrix A
    A = np.zeros((N,N))
    # Current values
    A[:n,:n] = Jacob[:,n:2*n]
    # Lead variables values
    A[:n,n:] = Jacob[:,forward_columns]
    # Identity equations
    for i in range(N_forward):
        col = forward_columns[i]
        A[n+i,col] = 1
        
    # Build matrix B
    B = np.zeros((N,N))
    # Lag variables values
    B[:n,:n] = -Jacob[:,2*n:3*n]
    # Identity equations
    B[n:,n:] = np.eye(N_forward)
    
    if debug:
        np.set_printoptions(precision=1)
        print("Matrix A:")
        print(A)
        print()
        print("Matrix B:")
        print(B)
        print()
        
    ### ------------------------ Generalized Schur decomposition
        
    # Apply generalized Schur decomposition (a.k.a. QZ decomposition) to the pencil (A,B)
    
    a,b,alpha,beta,q,z = la.ordqz(A,B,output='complex',sort=sorter) 

    # Transpose matrices since python QZ algorithm returns a transposed matrix versus Matlab
    z = z.T.conj()  
    
    # Check correctness of QZ decomposition
    err1 = la.norm(q @ a @ z - A) / la.norm(A)
    err2 = la.norm(q @ b @ z - B) / la.norm(B)
    if err1 > 1.e-10 or err2 > 1.e-10:
        cprint("Villemot Solver: QZ decomposition error.  \n Inconsistency of A and B matrix decomposition of {0} and {1}".format(round(err1,4),round(err2,4)),"red")
        raise
               
    # Find number of stable and unstable roots
    roots = np.array([abs(y/x) if abs(x)>0 else np.inf for x,y in zip(np.diag(a),np.diag(b)) ])
    n_unstable = sum(roots>=qz_factor)
    n_stable = N - n_unstable 

    # Check Blanchard-Kahn condition
    if not suppress_warnings:
        print()
        if N_forward == n_unstable: 
            cprint('Blanchard-Kahn condition of existance and unique solution is satisfied','green')
        elif N_forward < n_unstable:
            raise Exception('Blanchard-Kahn condition is not satisfied: no stable solution!')
        else:
            raise Exception('Blanchard-Kahn condition is not satisfied: multiple stable solutions!') 
        print()
    
    if debug:
        print('n_unstable=',n_unstable,',   n_forward_looking=',N_forward,'   forward variables=',forward_variables)
        print('roots:',roots)
        
        
    unstable_roots = roots[n_stable:]
    if np.any(unstable_roots<1) and not suppress_warnings:
        cprint("Some of unstable roots are less than one: \n {}".format(np.round(unstable_roots,2)),"red")


    ### --------------------------Find solution for dynamic variables
    ### Partition matrices
    a11 = a[:n_stable, :n_stable]
    a12 = a[:n_stable, n_stable:]
    a22 = a[n_stable:, n_stable:]
    
    b11 = b[:n_stable, :n_stable]
    b12 = b[:n_stable, n_stable:]
    b22 = b[n_stable:, n_stable:]
    
    z11 = z[:n_stable,:n_stable]
    z12 = z[:n_stable,n_stable:]
    z21 = z[n_stable:,:n_stable]
    z22 = z[n_stable:,n_stable:]
    
    
    U = z11
    D = q @ Psi1
    #psi1 = D[:n_stable]
    psi2 = D[n_stable:]
    #Const = q @ K1
    #k1 = Const[:n_stable]
    #k2 = Const[n_stable:]
    
    # Non-predetermined (or unstable) variables solution
    try:
        g_plus = -la.solve(z22, z21)
    except la.LinAlgError as err:
        if not suppress_warnings:
            cprint(err,"red")
            cprint("n={0}, n_unstable={1}".format(n,n_unstable),"red")
            print('det(z22)=',la.det(z22))
        raise
    
    # Predetermined (state) variables solution
    X = z11 + z12 @ g_plus
    
    try:
        tmp = la.solve(a11,b11 @ X)
    except la.LinAlgError as err:
        if not suppress_warnings:
            cprint(err,"red")
            cprint("n={0}, n_unstable={1}".format(n,n_unstable),"red")
            print('det(b11)=',la.det(b11))
        raise
    
    try:
        g_minus = la.solve(X,tmp)
    except la.LinAlgError as err:
        if not suppress_warnings:
            cprint(err,"red")
            cprint("n={0}, n_unstable={1}".format(n,n_unstable),"red")
            print('det(X)=',la.det(X))
        raise

    ### -----------------------------  Build matrices 
        
    # Build transition matrix
    A = np.real(g_minus)
    
    # Build matrix R of shocks
    Fy0 = Jacob[:,n:2*n]
    Fyp = Jacob[:,forward_columns]
    temp = Fyp @ g_plus + Fy0
    try:
        Z = la.inv(temp)
    except la.LinAlgError as err:
        if not suppress_warnings:
            cprint(err.message,"red")
            print('Z=',temp)
        raise
    
    # Shock matrix
    R = Z @ Psi
    # Constant matrix 
    C = -np.real(Z @ K)
    
    # Get solution decomposition matrices for future shocks
    # Unstable block
    G  = -la.solve(z11,z12)
    Ru = -la.solve(b22,psi2)
    Xa0 = la.solve(a11,b11 @ G + b12)
    Xa1 = G + la.solve(a11,a12)
    J  = -la.solve(b22,a22)
    Xa = Xa1 + Xa0 @ J
    #Xa = G @ J - la.inv(a11) @ b12
    
    return U,A,C,R,Z,Xa,J,Ru

