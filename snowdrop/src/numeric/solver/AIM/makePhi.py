"""This function calculates the phi ndarray used in the AMA algorithm."""

# Import the numpy package
import numpy as np

def makePhi(q,cof,nlag,nlead,neq):
    
    # Fix the size of the Phi ndarray, which is neq x neq
    phi = np.zeros(shape=(neq,neq))

    # Calculate the H_0 ndarray, which is neq x neq
    H_0 = cof[:,neq*nlag:neq*(nlag+1)]

    # Calculate the H_+ ndarray, which is nlead*neq x neq
    H_plus = cof[:,neq*(nlag+1):neq*(nlag+nlead+1)]

    # Calculate the Q_L ndarray, which is neq*nlead x neq*nlag
    Q_L = q[:,0:neq*nlag]

    # Calculate the Q_R ndarray, which is neq*nlead x neq*nlead
    Q_R = q[:,neq*nlag:neq*(nlag+nlead)]

    # Calculate the B ndarray, B = Q_R^-1 * Q_L, which is neq*nlead x neq*nlag
    B = -np.linalg.solve(Q_R, Q_L)

    # Calculate the B_R ndarray, which is neq*nlead x neq
    B_R = B[:,neq*(nlag-1):neq*nlag]

    # Calculate the phi ndarray, phi = (H_0 + H_+*B_R)^-1, which is neq x neq 
    temp2 = H_0 + (H_plus @ B_R)
    phi = np.linalg.inv(temp2)

    return phi
