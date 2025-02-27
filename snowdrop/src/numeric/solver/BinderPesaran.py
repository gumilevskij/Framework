# -*- coding: utf-8 -*-
"""
Created on Fri Nov 1, 2019

@author: agoumilevski
"""
from __future__ import division

import numpy as np
import scipy.linalg as la
from snowdrop.src.misc.termcolor import cprint
from snowdrop.src.numeric.solver.util import getParameters
from snowdrop.src.preprocessor.function import get_function_and_jacobian

count = 1

def solve_quadratic_determinantal_equation(model,steady_state,p=None,suppress_warnings=False):
    r"""Find first-order accuracy approximation.
    
    It applies substitution algorithm similar to Binder and Pesaran (1995).
	For general system of the form
    
    .. math:: F * x_{t+1} + C * x_{t} + L * x_{t-1} + C = G * w_{t+1},  w = Normal(0, Q) 
    		
    We seek solution of the form
        
	.. math:: x_{t+1} = A * x_{t} + B + E * w_{t+1}
    
    
    For details please see https://callumjones.github.io/files/dynare_str.pdf
    
    Parameters:
        :param model: The Model object.
        :type model: instance of class Model.
        :param steady_state: Steady state.
        :type steady_state: list.
        :param p: Parameters values.
        :type p: list.
        :param suppress_warnings: Do not show warnings if True
        :type suppress_warnings: bool.
    """
    global count
    if count == 1:
        count += 1
        cprint("Binder-Pesaran solver","blue")
        print()
    
    A, B, R, Z, W = None, None, None, None, None
    p = getParameters(parameters=p,model=model)
    
    try:
        #Find jacobian
        z = np.vstack((steady_state,steady_state,steady_state))
        c,jacob = get_function_and_jacobian(model,params=p,y=z,order=1)
        n = len(jacob)
        F = jacob[:,:n]
        C = jacob[:,n:2*n]
        L = jacob[:,2*n:3*n]
        R = jacob[:,3*n:]
        if  model.max_lead == 0 and model.min_lag < 0:
            C_inv = la.inv(C)
            A = -C_inv @ L
            R = -C_inv @ R
            B = -C_inv @ c
        elif model.max_lead > 0 and model.min_lag == 0:
            A,B,R,Z,W = solve_quadratic_equation(F=F,C=C,L=L,c=c,psi=R,suppress_warnings=suppress_warnings)
            #A,B,R,Z,W = solve_recursive(F=F,C=C,L=L,c=c,psi=R)
        else:
            A,B,R,Z,W = solve_quadratic_equation(F=F,C=C,L=L,c=c,psi=R,suppress_warnings=suppress_warnings)
            #A,B,R,Z,W = solve_recursive(F=F,C=C,L=L,c=c,psi=R)
    except :
        if not suppress_warnings:
            import sys
            sys.exit('Binder-Pesaran solver: unable to find solution of model.')
                
    model.linear_model["A"]  = A
    model.linear_model["C"]  = B
    model.linear_model["R"]  = R
    model.linear_model["Z"]  = Z
    model.linear_model["W"]  = W
    
    return model

	
def solve_recursive(F,C,L,c,psi,N=300,TOLERANCE = 1.e-8,suppress_warnings=False):
    r"""
    Find first-order accuracy approximation solution.
    
    It applies recursive algorithm of Binder and Pesaran (1995).
    Please see paper 'Multivariate Linear Rational Expectations Models: Characterization of the Nature of the Solutions and their Fully Recursive Computation'
    
    For general system of equations with leads and lags
    
        .. math:: F * x_{t+1} + C * x_{t} + L * x_{t-1} + c + psi * w_{t+1} = 0,  w = Normal(0, Q)
    
    We seek solution of the form
    
    	.. math:: x_{t+1} = A * x_{t} + B + R * w_{t+1}
	
    Matlab program can be downloaded from: http://www.inform.umd.edu/econ/mbinder
    
    Parameters:
        :param F: The Jacobian matrix of lead variables.
        :type F:  numpy.ndarray.
        :param C: The Jacobian matrix of current variables.
        :type C:  numpy.ndarray.
        :param L: The Jacobian matrix of lag variables.
        :type L:  numpy.ndarray.
        :param jacob: Jacobian matrix.
        :type jacob: numpy.ndarray.
        :param c: Array of constants.
        :type c: numpy.ndarray.
        :param Psi: Matrix of coefficients of shocks.
        :type Psi: numpy.ndarray.
        :param N: Initial Forecasting Horizon (See Binder and Pesaran, 1995)
        :type N: int.
    """
    # N - Initial Forecasting Horizon (See Binder and Pesaran, 1995)
    
    (dim1,dim2) = F.shape
    I = np.eye(dim1)
    
    Ci = la.inv(C) 
    F = Ci @ F
    L = Ci @ L
    c = Ci @ c
    psi = Ci @ psi

    # Carry Out Recursions
    Q = np.eye(dim1)
    r = psi
    j = 0
    while j <= N:
        Qi = la.inv(Q)
        r = F @ Qi @ r + psi
        Q = I - F @ Qi @ L
        j += 1
	
    epsilon = 1
    QN = Q
    RN = r

    while epsilon > TOLERANCE:
        Q  = QN
        r  = RN
        QN = I
        RN = psi
        j  = 0
        while j <= (N+1):
            QNi = la.inv(QN)
            RN  = F @ QNi @ RN + psi
            QN  = I - F @ QNi @ L
            j  += 1
        Qi  = la.inv(Q)
        epsilon = np.max(abs(QNi @ RN - Qi @ r))

    W = la.inv(QN)
    A = -W @ L
    R = -W @ psi
    B = -W @ c
    Z = -W @ F

#    printResults(A,B,R)
     
    return A, B, R, Z, W
	
	
def solve_quadratic_equation(F,C,L,c,psi,N=1000,TOLERANCE = 1.e-12,suppress_warnings=False):
    r"""
    Find first-order accuracy approximation solution.
    It applies substitution algorithm similar to Binder and Pesaran (1995).
    For general system of equations with leads and lags
    
    .. math:: F*x_{t+1} + C*x_{t} + L*x_{t-1} + c + psi*w_{t+1} = 0,  w = Normal(0, Q)
    
    Rewriting equations
    
    .. math:: x_{t} = -C^{-1}*[L*x_{t-1} + F*x_{t+1} + c + psi*w_{t+1}]
    
    Representing
    
    .. math:: x_{t} = A*x_{t} + z_{t}
   
    and substituting, we can decouple x and z by finding matrix A satisfying quadratic equation
    
    .. math:: C^{-1} * L + A + C^{-1} * F * A^{2} = 0
    
    Forward eqution for z is
    
    .. math:: z_{t} = [I+C^{-1}*F*A]*C^{-1} * [F*z_{t+1} + c + psi*w_{t+1}]^{-1}
    
    By induction we get
    
    .. math:: z_{t} = [I+C^{-1}*F*A] * C^{-1} * \sum_{k} [F^{k}*(c+psi*w_{t+k})]^{-1}
	
    
    Parameters:
        :param F: The Jacobian matrix of lead variables.
        :type F:  numpy.ndarray.
        :param C: The Jacobian matrix of current variables.
        :type C:  numpy.ndarray.
        :param L: The Jacobian matrix of lag variables.
        :type L:  numpy.ndarray.
        :param jacob: Jacobian matrix.
        :type jacob: numpy.ndarray.
        :param c: Array of constants.
        :type c: numpy.ndarray.
        :param Psi: Matrix of coefficients of shocks.
        :type Psi: numpy.ndarray.
        
        
    .. note::
        Please see paper 'Multivariate Linear Rational Expectations Models:Characterization of the Nature of the Solutions and their Fully Recursive Computation' describing the theory underlying this approach.
    
	Matlab program is available at: http://www.inform.umd.edu/econ/mbinder
    
    """
    # TOLERANCE - Convergence Criterion 
    # N - Initial Forecasting Horizon (See Binder and Pesaran, 1995)
    
    (dim1,dim2) = F.shape
    (dim3,dim4) = psi.shape
    I           = np.eye(dim1)
    
    Ci  = la.inv(C) 
    F   = -Ci @ F
    L   = -Ci @ L
    c   = -Ci @ c
    psi = -Ci @ psi

    # Compute Matrix C Using Brute-Force Iterative Procedure
    A = np.copy(I)   # Initial Conditions
    eps = 1                
    it = 0
    while eps >= TOLERANCE:
        temp = la.pinv(I-F@A) @ L 
        eps  = np.max(abs(temp-A))
        A    = temp
        it  += 1
        if it > N:
            if not suppress_warnings:
                print(' The brute-force iterative procedure did not converge after '  + str(it) + ' iterations. ')
                #print('See Binder and Pesaran (1995, 1997) for alternative algorithms to compute the matrix A. ')
            break

    # Use Recursive Method of Binder and Pesaran (1995) to compute the forward part of the solution
    W = la.pinv(I-F@A)
    R = W @ psi
    B = W @ c
    Z = W @ F	
     
    return A, B, R, Z, W


def getMatrices(model,n,y,t=None):
    """Return state-space model representation matrices."""
    global count
    if count == 1:
        count += 1
        cprint("Binder and Pesaran solver","blue")
        
    # Find matrices of state-space model.
    if t is None:
        y1 = y2 = y3 = y
    else:
        y1 = y[max(0,t-1)]; y2 = y[t]; y3 = y[t+1]
        
    c,jacob= get_function_and_jacobian(model=model,t=t,y=[y1,y2,y3])
    F = jacob[:,:n]
    C = jacob[:,n:2*n]
    L = jacob[:,2*n:3*n]
    Psi = jacob[:,3*n:]
    
    A,B,R,Z,W = solve_quadratic_equation(F=F,C=C,L=L,c=c,psi=Psi,N=100,TOLERANCE=1e-6,suppress_warnings=True)
    # Vector of constants is found by equating endogenous variables to zero.
    c,jacob= get_function_and_jacobian(model=model,t=t,y=[1e-10*y1,1e-10*y2,1e-10*y3])
    F = jacob[:,:n]
    C = jacob[:,n:2*n]
    L = jacob[:,2*n:3*n]
    Psi = jacob[:,3*n:]
    A1,B,R1,Z1,W1 = solve_quadratic_equation(F=F,C=C,L=L,c=c,psi=Psi,N=100,TOLERANCE=1e-6,suppress_warnings=True)
    
    return A,B,R
    

def printResults(A,C,R,Z,W):
	"""Display results."""
	print('The decision rule is: ')
	print('x[t] = A*x[t-1] + R*w[t] + C ')
	print('where: A')
	print(A)
	print('where: R')
	print(R)
	print('where: C')
	print(C)
	print('Matrix Z')
	print(Z)
	print('Matrix W')
	print(W)
	print()
    
    
    
if __name__ == "__main__":
    """
    Main entry point
    """
    n = 5
    jacob = np.array([[0, 0 ,0 ,0 ,0 ,1 ,-1 ,0 ,-1 ,0 ,0 ,0 ,0 ,0 ,0],
                      [0 ,0 ,0 ,0 ,0 ,0 ,1 ,-1 ,0 ,0 ,0 ,-1 ,0 ,0 ,0],
                      [0 ,0 ,0 ,0 ,0 ,0 ,0 ,1 ,0 ,0 ,0 ,0 ,-0.9 ,0 ,0],
                      [0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,1 ,0 ,0 ,0 ,0 ,-0.75 ,0],
                      [0 ,0 ,0 ,0 ,-0.25 ,0 ,0 ,0 ,-0.25 ,1 ,0 ,0 ,0 ,0 ,-0.75]])
    F = jacob[:,:n]
    C = jacob[:,n:2*n]
    L = jacob[:,2*n:3*n]
    
    R = np.array([[ 0, 0, 0, 0],
                  [ 1, 0, 0, 0],
                  [ 0, 1, 0, 0],
                  [ 0, 0, 1, 0],
                  [ 0, 0, 0, 1]])
     
    c = np.zeros(n)
    
    A1,B1,r1,Z1,W1 = solve_quadratic_equation(F=F,C=C,L=L,c=c,psi=R)
    print('Quadratic matrix equation solver:')
    printResults(A1,B1,r1,Z1,W1)
    
    A2,B2,r2,Z2,W2 = solve_recursive(F=F,C=C,L=L,c=c,psi=R)
    print('Recursive solver:')
    printResults(A2,B2,r2,Z2,W2)
    
    print('Quadratic and Recursive solvers difference:')
    print('Matrix A:')
    print(A1-A2)
    print('Matrix B:')
    print(B1-B2)
    print('Matrix R:')
    print(r1-r2)
    print('Matrix Z:')
    print(Z1-Z2)
    print('Matrix W:')
    print(W1-W2)
    print()



