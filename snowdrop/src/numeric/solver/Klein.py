# -*- coding: utf-8 -*-
"""
Created on Wed Jul 8, 2020

@author: agoumilevski
"""
from __future__ import division

import numpy as np
import scipy.linalg as la
from sys import exit
from snowdrop.src.misc.termcolor import cprint, colored
from snowdrop.src.numeric.solver.util import qzdiv
from snowdrop.src.numeric.solver.util import getParameters
from snowdrop.src.numeric.solver.util import sorter
#from snowdrop.src.numeric.solver.util import clustered_sorter
from snowdrop.src.preprocessor.function import get_function_and_jacobian
from snowdrop.src.utils.equations import getLeadLagIncidence

count = 1


def solve_klein(model,steady_state,p=None,suppress_warnings=False):
    """
    Find first-order accuracy approximation solution.
    
    It implements an algorithm described by Michal Andrle in the paper:
    "Linear Approximation to Policy Function in IRIS Toolbox".
    
    For references please see: 
    Paul Klein (2000), "Using the generalized Schur form to solve a multivariate 
    linear rational expectations models", Journal of Economic Dynamics & Control,
    Vol.24, pp. 1405-1423
    
    and
    
    https://michalandrle.weebly.com/uploads/1/3/9/2/13921270/michal_iris_solve.pdf
    
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
        cprint("Paul Klein solver","blue")
        
    U,T,Ta,Tf,R,Ra,Rf,K,Ka,Kf,Xa,Xf,J,Z = None,None,None,None,None,None,None,None,None,None,None,None,None,None
    p = getParameters(parameters=p,model=model)
    
    try:
        #Find jacobian
        z = np.vstack((steady_state,steady_state,steady_state))
        K,Jacob = get_function_and_jacobian(model,params=p,y=z,order=1)
        n = len(Jacob)
        if  model.max_lead == 0 and model.min_lag < 0:
            C = Jacob[:,n:2*n]
            L = Jacob[:,2*n:3*n]
            R = Jacob[:,3*n:]
            C_inv = la.inv(C)
            Ta = -C_inv @ L
            Ra = -C_inv @ R
            Ka = -C_inv @ K
            model.linear_model["A"]  = Ta
            model.linear_model["R"]  = Ra
            model.linear_model["C"]  = Ka
            model.linear_model["Xa"] = 0*Ka
            model.linear_model["Ru"] = 0*Ra
            model.linear_model["J"]  = 0*Ta
            model.linear_model["U"]  = 0*Ta
            model.linear_model["Z"] = -C_inv 
            
        else:
            U,T,Ta,Tf,R,Ra,Rf,K,Ka,Kf,C,Xa,Xf,J,Ru,Z = solution1(model=model,Jacob=-Jacob,C=-K,suppress_warnings=suppress_warnings)
            model.linear_model["A"]  = U @ Ta @ la.inv(U)
            model.linear_model["R"]  = U @ Ra
            #model.linear_model["C"]  = C
            model.linear_model["C"]  = U @ Ka
            model.linear_model["U"]  = U
            model.linear_model["J"]  = J
            model.linear_model["Ta"] = Ta
            model.linear_model["Tf"] = Tf
            model.linear_model["Ra"] = Ra
            model.linear_model["Rf"] = Rf
            model.linear_model["Re"] = R
            model.linear_model["Ka"] = Ka
            #model.linear_model["Ka"] = la.inv(U) @ C 
            model.linear_model["Kf"] = Kf
            model.linear_model["Xa"] = Xa
            model.linear_model["Xf"] = Xf
            model.linear_model["Ru"] = Ru
            model.linear_model["Z"]  = np.real(Z)
    
    except :
        if not suppress_warnings:
            exit('Paul Klein solver: unable to find solution of a linear model.\n   Please use other solvers: BinderPesaran, AndersonMoore, ...')

    return model


def solution1(model,Jacob,C=None,qz_factor=1+1.e-6,debug=False,suppress_warnings=False):
    """
    Find first-order accuracy approximation solution.
    
    Implements an algoritm described by Michal Andrle in the paper:
    "Linear Approximation to Policy Function in IRIS Toolbox".
    
    For references please see:  
    https://michalandrle.weebly.com/uploads/1/3/9/2/13921270/michal_iris_solve.pdf
    
    Parameters:
        :param model: The Model object.
        :type model: instance of class `Model'.
        :param Jacob: Jacobian matrix.
        :type Jacob: numpy.ndarray.
        :param C: Array of constants.
        :type C: numpy.ndarray.
    """   
    U,T,K,R,Ta,Ra,Ka,Tf,Rf,Kf = None,None,None,None,None,None,None,None,None,None
    debug &= suppress_warnings
    
    variables = model.symbols["variables"]
    n = len(variables)
    
    if C is None:
        C = np.zeros(n)
        
    Psi = Jacob[:,3*n:]
    n_shock = Psi.shape[1]
    
    # Retrieve lead-lag incidence matrix
    lli = model.lead_lag_incidence 
    if lli is None:
        lli = getLeadLagIncidence(model)
        
    
    forward_columns = np.array([int(f) for f in lli[2] if not np.isnan(f)]) 
    #forward_variables = [x for x in model.topology["forward"].keys() if not "_p_" in x]
    forward_variables = [variables[i] for i in forward_columns]
    N_forward = len(forward_columns)
    N = n + N_forward
    
    ### ------------------- Make structural state-space representation of equations
    # Build matrix A
    A = np.zeros(shape=(N,N))
    # Current values
    A[:n,:n] = Jacob[:,n:2*n]
    # Lead variables values
    A[:n,n:] = Jacob[:,forward_columns]
    # Identity equations
    for i in range(N_forward):
        col = forward_columns[i]
        A[n+i,col] = 1
        
    # Build matrix A
    B = np.zeros(shape=(N,N))
    # Lag variables values
    B[:n,:n] = Jacob[:,2*n:3*n]
    # Identity equations
    B[n:,n:] = -np.eye(N_forward)
    
    # Matrix of shocks
    Psi1 = np.concatenate((Psi,np.zeros((N_forward,n_shock))))
    
    # Matrix of constants
    C1 = np.concatenate((C,np.zeros(N_forward)))
    
    if debug:
        np.set_printoptions(precision=2)
        print("Matrix A:")
        print(A)
        print()
        print("Matrix B:")
        print(B)
        print()
        print("Matrix Psi:")
        print(Psi1)
        print()
        print("Vector C:")
        print(C1)
        print()   
    ### ------------------------ Generalized Schur decomposition (a.k.a. QZ decomposition) to the pencil (A,B)
    
    
    # If the QZ re-ordering fails, change the order of equations .
    # Place the first equation last, and repeat.
    eqOrd = list(np.arange(N))
    for shuffle in range(N):
        
        AA = A[eqOrd]
        BB = B[eqOrd]
        
        a,b,alpha,beta,q,z = la.ordqz(AA,BB,output='real',sort=sorter) 
        
        # Transpose matrix
        q = q.T

        # Reorder decomposition matrices so that eigen values are ordered.
        a,b,q,z,col_order = qzdiv(a,b,q,z)
            
        # Find number of stable and unstable roots
        roots = np.array([abs(y/x) if abs(x)>0 else np.inf for x,y in zip(np.diag(a),np.diag(b))])
        #print(roots)
        n_unstable = sum(roots>=qz_factor)
        n_stable = N - n_unstable  
            
        # Reshuffle equations order if QZ algorithm fails
        unstable_roots = roots[n_stable:]
        if np.any(unstable_roots<2-qz_factor) or N_forward != n_unstable:
            eqOrd = eqOrd[1:] + [eqOrd[0]]
        else:
            break


    # Check correctness of QZ decomposition
    err1 = la.norm(q.T @ a @ z.T - AA) / la.norm(AA)
    err2 = la.norm(q.T @ b @ z.T - BB) / la.norm(BB)
    if err1 > 1.e-10 or err2 > 1.e-10:
        raise Exception("Paul Klein Solver: QZ decomposition error.  \n Inconsistency of A and B matrices decomposition of {0} and {1}".format(round(err1,4),round(err2,4)))

    if debug:
        print('n_unstable=',n_unstable,',   n_forward_looking=',N_forward,'   forward variables=',forward_variables)
        print('roots:',roots)
        
    # Check Blanchard-Kahn condition
    if not suppress_warnings:
        print()
        if N_forward == n_unstable: 
            cprint('Blanchard-Kahn condition of existance and unique solution is satisfied','green',attrs=['bold','underline'])
        elif N_forward < n_unstable:
            cprint(f"Number of unstable roots {n_unstable} is larger than number of forward looking variables {N_forward}!","red")
            raise ValueError(colored('Blanchard-Kahn condition is not satisfied: no stable solution!','red',attrs=['underline']))  
        else:
            cprint(f"Number of unstable roots {n_unstable} is smaller than number of forward looking variables {N_forward}!","red")
            raise ValueError(colored('Blanchard-Kahn condition is not satisfied: multiple stable solutions!','red',attrs=['underline'])) 
        print()
         
    if np.any(unstable_roots<1) and not suppress_warnings:
        cprint("Some of unstable roots are less than one: \n {}".format(np.round(unstable_roots,2)),"red")

    
    ### Partition matrices
    a11 = a[:n_stable, :n_stable]
    a12 = a[:n_stable, n_stable:]
    a22 = a[n_stable:, n_stable:]
    
    b11 = b[:n_stable, :n_stable]
    b12 = b[:n_stable, n_stable:]
    b22 = b[n_stable:, n_stable:]
    
    z11 = z[:n_stable, :n_stable]
    z12 = z[:n_stable, n_stable:]
    z21 = z[n_stable:, :n_stable]
    z22 = z[n_stable:, n_stable:]
    
   
    # Equations have been re-ordered while computing QZ
    D = q @ Psi1[eqOrd]
    Const = q @ C1[eqOrd] 

    c1 = Const[:n_stable]
    c2 = Const[n_stable:]
    d1 = D[:n_stable]
    d2 = D[n_stable:]
    
    # Quasi-triangular state-space form
    U = z11
    
    ### Unstable block.
    G  = -la.solve(z11,z12)
    Ru = -la.solve(b22,d2)
    Ku = -la.solve(a22+b22,c2)
    
    # Transform stable block, i.e., transform backward-looking variables:
    # a(t) = s(t) + G u(t+1).
    Ta = -la.solve(a11,b11)
    Xa0 = la.solve(a11,b11 @ G + b12)
    Ra = -Xa0 @ Ru - la.solve(a11,d1)
    Xa1 = G + la.solve(a11,a12)
    Ka = -(Xa0 + Xa1) @ Ku - la.solve(a11,c1)
    J  = -la.solve(b22,a22)
    Xa = Xa1 + Xa0 @ J 

    # Forward expansion for time t+k.
    # a(t) <- -Xa J^(k-1) Ru e(t+k)
    # xf(t) <- Xf J^k Ru e(t+k)
    Tf = z21
    Xf = z21 @ G + z22
    Rf = Xf  @ Ru
    Kf = Xf  @ Ku
    
    # Compute constants
    g_plus = -la.solve(z.T[n_stable:,n_stable:], z.T[n_stable:,:n_stable])
    Fy0 = Jacob[:,n:2*n]
    Fyp = Jacob[:,forward_columns]
    C = -la.solve(Fyp @ g_plus + Fy0, C)
    Z = -la.inv(Fyp @ g_plus + Fy0)
   
    
    #####
    # State-space form:  [xf(t),a(t)] = [Tf,Ta] a(t-1) + [Kf,Ka] + [Rf,Ra] e(t)
    # where U a(t) = xb(t).
    T = np.concatenate((Tf,Ta),axis=0)
    K = np.concatenate((Kf,Ka),axis=0)
    R = np.concatenate((Rf,Ra),axis=0)
    
    if debug: 
        print("\nTransition matrix:")
        Trans = U @ Ta @ la.inv(U)
        print(Trans)
        print("\nMatrix of shocks:")
        Tshk = U @ Ra
        print(Tshk)
        print("\nMatrix of constants:")
        Tconst = U @ Ka
        print(Tconst)
       
    return U,T,Ta,Tf,R,Ra,Rf,K,Ka,Kf,C,Xa,Xf,J,Ru,Z


def solution2(model,Jacob,C=None,qz_factor=1+1e-10,debug=False,suppress_warnings=False):
    """
    Find first-order accuracy approximation solution.
    
    Implements an algoritm described by Michal Andrle in the paper:
    "Linear Approximation to Policy Function in IRIS Toolbox".
    
    For references please see:  
    https://michalandrle.weebly.com/uploads/1/3/9/2/13921270/michal_iris_solve.pdf
    
    Parameters:
        :param model: The Model object.
        :type model: instance of class Model.
        :param Jacob: Jacobian matrix.
        :type Jacob: numpy.ndarray.
        :param C: Array of constants.
        :type C: numpy.ndarray.
    """   
    from snowdrop.src.utils.d2s import myd2s
    
    U,T,K,R,Ta,Ra,Ka,Tf,Rf,Kf = None,None,None,None,None,None,None,None,None,None
    debug &= suppress_warnings
    
    variables = model.symbols["variables"]
    n = len(variables)
                
    if C is None:
        C = np.zeros(n)
        
    # # Normalize derivatives by the largest number 
    # for i in range(n):
    #     norm = max(np.max(abs(Jacob[i])),abs(C[i]))
    #     Jacob[i] /= norm
    #     C[i] /= norm
    
    Psi = Jacob[:,3*n:]
    n_shock = Psi.shape[1]
    
    # Retrieve variables position
    d2s = myd2s(model)
    
    # Retrieve lead-lag incidence matrix
    lli = model.lead_lag_incidence 
    if lli is None:
        lli = getLeadLagIncidence(model)
        
    forward_columns = np.array([int(f) for f in lli[2] if not np.isnan(f)]) 
    forward_variables = [x for x in model.topology["forward"].keys() if not "_p_" in x]
    forward_variables = [variables[i] for i in forward_columns]
    N_forward = len(forward_columns)
    N = n + N_forward
    
    ### ------------------- Make structural state-space representation of equations 
    # Initialize matrices
    A = np.zeros(shape=(N,N))
    B = np.zeros(shape=(N,N))
    E = np.zeros(shape=(N,n_shock))
    D = np.zeros(N)
    
    # Transition equations: A2*[xf+;xb+] + B2*[xf;xb] + E2*e + K2 = 0
    A[:n,d2s.xu1] = Jacob[:,d2s.xu1_]
    A[:n,d2s.xp1] = Jacob[:,d2s.xp1_]
    B[:n,d2s.xp]  = Jacob[:,d2s.xp_]
    E[:n,d2s.e]   = Jacob[:,d2s.e_]
    D[:n]         = C
       
    # Dynamic identity matrices
    A[n:,d2s.ident1] = d2s.ident1_
    B[n:,d2s.ident] = d2s.ident_
    
    # Measurement equations: A1*y + B1*xb+  + E1*e + K1 = 0
    
    
    if debug:
        np.set_printoptions(precision=2)
        print("Matrix A:")
        print(A)
        print()
        print("Matrix B:")
        print(B)
        print()
        print("Matrix Psi:")
        print(E)
        print()
        print("Vector C:")
        print(D)
        print()
        
    
    ### ------------------------ Generalized Schur decomposition (a.k.a. QZ decomposition) to the pencil (A,B)
    
    # If the QZ re-ordering fails, change the order of equations...
    # Place the first equation last, and repeat.
    method = 2
    eqOrd = list(np.arange(N))
    for shuffle in range(N):
        
        AA = A[eqOrd]
        BB = B[eqOrd]
        
        if method == 1:
            a,b,alpha,beta,q,z = la.ordqz(AA,BB,output='real',sort=sorter) 
            # Transpose matrix
            q = q.T
            # Reorder decomposition matrices
            cluster = [2-qz_factor,qz_factor]
            a,b,q,z,col_order = qzdiv(a,b,q,z,cluster)
            
        elif method == 2:
            a,b,alpha,beta,q,z = la.ordqz(AA,BB,output='real',sort=sorter) 
            # Transpose matrix
            q = q.T
            # Reorder decomposition matrices so that eigen values are listed in the order
            # of less than one, then one, and then greater than one
            a,b,q,z,col_order = qzdiv(a,b,q,z)
            
        if np.any(np.isclose(alpha, beta)):
            cprint('Warning: unit root detected!','red')
        
        # Find number of stable and unstable roots
        roots = np.array([abs(y/x) if abs(x)>0 else np.inf for x,y in zip(np.diag(a),np.diag(b))])
        n_unstable = sum(roots>=qz_factor)
        n_stable = N - n_unstable  
            
        # Reshuffle equations order if QZ algorithm fails
        unstable_roots = roots[n_stable:]
        if np.any(unstable_roots<=1) or N_forward != n_unstable:
            eqOrd = eqOrd[1:] + [eqOrd[0]]
            #print("\nUnstable {}, Forward {} : Iteration {} of {}, Roots:".format(n_unstable,N_forward,1+shuffle,N))
            #print(unstable_roots)
        else:
            break
        
    # Check correctness of QZ decomposition
    err1 = la.norm(q.T @ a @ z.T - AA) / la.norm(AA)
    err2 = la.norm(q.T @ b @ z.T - BB) / la.norm(BB)
    if err1 > 1.e-10 or err2 > 1.e-10:
        raise Exception("Paul Klein Solver: QZ decomposition error.  \n Inconsistency of A and B matrices decomposition of {0} and {1}".format(round(err1,4),round(err2,4)))

    if debug:
        print('n_unstable=',n_unstable,',   n_forward_looking=',N_forward,'   forward variables=',forward_variables)
        print('roots:',roots)
        
    # Check Blanchard-Kahn condition
    if not suppress_warnings:
        print()
        if N_forward == n_unstable: 
            cprint('Blanchard-Kahn condition of existance and unique solution is satisfied','green',attrs=['bold','underline'])
        elif N_forward < n_unstable:
            cprint(f"Warning: number of unstable roots {n_unstable} is larger than number of forward looking variables {N_forward}!","red")
            raise ValueError(colored('Blanchard-Kahn condition is not satisfied: no stable solution!','red',attrs=['underline']))  
        else:
            cprint(f"Warning: number of unstable roots {n_unstable} is smaller than number of forward looking variables {N_forward}!","red")
            raise ValueError(colored('Blanchard-Kahn condition is not satisfied: multiple stable solutions!','red',attrs=['underline'])) 
        print()
         
    if np.any(abs(unstable_roots)<1) and not suppress_warnings:
        cprint("Some of unstable roots are less than one: \n {}".format(np.round(unstable_roots,2)),"red")


    ### Partition matrices
    a11 = a[:n_stable, :n_stable]
    a12 = a[:n_stable, n_stable:]
    a22 = a[n_stable:, n_stable:]
    
    b11 = b[:n_stable, :n_stable]
    b12 = b[:n_stable, n_stable:]
    b22 = b[n_stable:, n_stable:]
    
    z11 = z[:n_stable, :n_stable]
    z12 = z[:n_stable, n_stable:]
    z21 = z[n_stable:, :n_stable]
    z22 = z[n_stable:, n_stable:]
    
   
    # Equations have been re-ordered while computing QZ
    D = q @ D[eqOrd] 
    E = q @ E[eqOrd]

    c1 = D[:n_stable]
    c2 = D[n_stable:]
    d1 = E[:n_stable]
    d2 = E[n_stable:]
    
    # Quasi-triangular state-space form.
    U = z11
    
    ### Unstable block.
    G  = -la.solve(z11,z12)
    Ru = -la.solve(b22,d2)
    Ku = -la.solve(a22+b22,c2)
    
    # Transform stable block, i.e., transform backward-looking variables:
    # a(t) = s(t) + G u(t+1).
    Ta = -la.solve(a11,b11)
    Xa0 = la.solve(a11,b11 @ G + b12)
    Ra = -Xa0 @ Ru - la.solve(a11,d1)
    Xa1 = G + la.solve(a11,a12)
    Ka = -(Xa0 + Xa1) @ Ku - la.solve(a11,c1)
    J  = -la.solve(b22,a22)
    Xa = Xa1 + Xa0 @ J 

    # Forward expansion for time t+k.
    # a(t) <- -Xa J^(k-1) Ru e(t+k)
    # xf(t) <- Xf J^k Ru e(t+k)
    Tf = z21
    Xf = z21 @ G + z22
    Rf = Xf  @ Ru
    Kf = Xf  @ Ku
    
    # Compute constants
    g_plus = -la.solve(z.T[n_stable:,n_stable:], z.T[n_stable:,:n_stable])
    Fy0 = Jacob[:,n:2*n]
    Fyp = Jacob[:,forward_columns]
    C = -la.solve(Fyp @ g_plus + Fy0, C)
    Z = -la.inv(Fyp @ g_plus + Fy0)
   
    ##### State-space form:  [xf(t),a(t)] = [Tf,Ta] a(t-1) + [Kf,Ka] + [Rf,Ra] e(t)
    # where U a(t) = xb(t).
    T = np.concatenate((Tf,Ta),axis=0)
    K = np.concatenate((Kf,Ka),axis=0)
    R = np.concatenate((Rf,Ra),axis=0)
    
    if debug: 
        print("\nTransition matrix:")
        Trans = U @ Ta @ la.inv(U)
        print(Trans)
        print("\nMatrix of shocks:")
        Tshk = U @ Ra
        print(Tshk)
        print("\nMatrix of constants:")
        Tconst = U @ Ka
        print(Tconst)
        
    return U,T,Ta,Tf,R,Ra,Rf,K,Ka,Kf,C,Xa,Xf,J,Ru,Z

  
if __name__ == '__main__':
    """
    Test QZ decomposition.
    """
    from snowdrop.src.numeric.solver.util import qzdiv2
    
    #Psi1,C1,c1,c2,d1,d2,q,z,a11,a12,a22,b11,b12,b22,z11,z12,z21,z22 = getMatrices()
    
    A =  [[0,-1, 1,1,0,0],
          [0,0,-1,0,1,0],
          [0,0,0,0,-1,0],
          [0,0,0,-1,0,0],
          [.25,0,0,.25,0,-1],
          [0,0,0,0,0,1]]
    A = np.array(A)
    B = [[0,0,0,0,0,0],
         [0,0,1,0,0,0],
         [0,0,0,0,0.9,0],
         [0,0,0,.75,0,0],
         [0,0,0,0,0,.75],
         [-1,0,0,0,0,0]]
    B = np.array(B)
    
    n = len(A)
    qz_factor = 1+1.e-10
    
    if True:
        a,b,alpha,beta,q,z = la.ordqz(A,B,output='real',sort=sorter) #sort='ouc') 
        # Transpose matrix
        q = q.T
    else:
        a,b,q,z = la.qz(A,B,output='real') 
        # Transpose matrix since python QZ algorithm returns a transposed matrix versus Matlab
        q = q.T
        # Re-arrange matrices in such a way that stable generalized eigenvalues are in the upper left corner of matrices: a,b
        a,b,q,z,col_order  = qzdiv2(qz_factor,a,b,q,z)


    # Check correctness of QZ decomposition
    err1 = la.norm() / la.norm(A)
    err2 = la.norm(q.T.conj() @ b @ z.T.conj() - B) / la.norm(B)
    if err1 > 1.e-10 or err2 > 1.e-10:
        raise Exception("Paul Klein Solver: QZ decomposition error.  \n Inconsistency of A and B matrix decomposition of {0} and {1}".format(round(err1,4),round(err2,4)))
            
    
    # Find number of stable and unstable roots
    roots = np.array([abs(y/x) if abs(x)>0 else np.inf for x,y in zip(np.diag(a),np.diag(b))])
    n_unstable = sum(roots>=qz_factor)
    n_stable = n - n_unstable  
    
    unstable_roots = roots[n_stable:]
    if np.any(unstable_roots<1):
        cprint("Some of unstable roots are less than one: \n {}".format(np.round(unstable_roots,2)),"red")
        
    
    np.set_printoptions(precision=4)

    print("\na:")
    print(np.real(a))     
    print("\nb:")
    print(np.real(b))    
    print("\nq:")
    print(np.real(q))   
    print("\nz:")
    print(np.real(z))
    
    """
    Matlab results:
        
    a:
   -0.8839         0         0         0   -0.3974    1.0324
         0   -0.7071   -0.7071    1.0000         0         0
         0         0    1.4142         0    0.9733    0.2294
         0         0         0   -1.0000         0         0
         0         0         0         0   -1.0073   -0.1626
         0         0         0         0         0   -0.2808

    b:
    0.8839         0         0         0   -0.0568    0.2409
         0    0.7071    0.7071         0         0         0
         0         0         0         0         0         0
         0         0         0    0.9000         0         0
         0         0         0         0    0.7555   -0.0466
         0         0         0         0         0    0.8423

    q:
         0         0         0         0    0.6000   -0.8000
         0   -1.0000         0         0         0         0
   -1.0000         0         0         0         0         0
         0         0   -1.0000         0         0         0
         0         0         0   -0.9662    0.2061    0.1546
         0         0         0   -0.2577   -0.7730   -0.5797

    z:
    0.7071         0         0         0   -0.1622    0.6882
         0   -0.7071    0.7071         0         0         0
         0   -0.7071   -0.7071         0         0         0
         0         0         0         0   -0.9733   -0.2294
         0         0         0   -1.0000         0         0
    0.7071         0         0         0    0.1622   -0.6882

    """

    

def getMatrices():
    """Load matlab file."""
    from mat4py import loadmat
    
    data = loadmat("C:/Temp/data.mat")
    Psi = np.array(data["E2"])
    C   = np.array(data["K2"])
    q   = np.array(data["QQ"])
    z   = np.array(data["ZZ"])
    c1  = np.array(data["C1"])
    c2  = np.array(data["C2"])
    d1  = np.array(data["D1"])
    d2  = np.array(data["D2"])
    a11 = np.array(data["S11"])
    a12 = np.array(data["S12"])
    a22 = np.array(data["S22"])
    b11 = np.array(data["T11"])
    b12 = np.array(data["T12"])
    b22 = np.array(data["T22"])
    z21 = np.array(data["Z11"])
    z22 = np.array(data["Z12"])
    z11 = np.array(data["Z21"])
    z12 = np.array(data["Z22"])
    
    return Psi,C,c1,c2,d1,d2,q,z,a11,a12,a22,b11,b12,b22,z11,z12,z21,z22
    