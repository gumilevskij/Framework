"""This function calculates the F matrix used in the AMA algorithm."""

# Import the numpy package
import numpy as np

def makeF(phi,cof,q,nlag,nlead,neq):

    # Fix the size of the F Matrix, which is nlead*neq x nlead*neq
    F = np.matrix(np.zeros(shape=((nlead*neq,nlead*neq))))

    # Calculate the H_+ Matrix, which is nlead*neq x neq
    H_plus = cof[:,neq*(nlag+1):neq*(nlag+nlead+1)]

    # Calculate the Q_L Matrix, which is neq*nlead x neq*nlag
    Q_L = q[:,0:neq*nlag]

    # Calculate the Q_R Matrix, which is neq*nlead x neq*nlead
    Q_R = q[:,neq*nlag:neq*(nlag+nlead)]

    # Calculate the B Matrix, B = Q_R^-1 * Q_L, which is neq*nlead x neq*nlag
    B = -Q_R.I * Q_L

    # Calculate the B_R Matrix, which is neq*nlead x neq
    B_R = B[:,neq*(nlag-1):neq*nlag]

    # Fill in the identity matrices in F 
    # Establish the veritcal index for inserting 1s into F

    for i in range(0,neq*(nlead-1)):
        # Establish the horizontal index for inserting 1s into F
        j = i + neq
        # Insert Identity values into F
        F[i,j] = 1
        
    # Construct the B_R Matrix used in the last column of F 
    newB_R = np.matrix(np.zeros(shape=((neq*nlead,neq))))

    # Insert Identity matrix into first element of newB_R
    newB_R[0:neq,0:neq] = np.eye(neq)

    # Insert B_R^theta into newB_R for theta from 1 to theta_1
    for a in range(1,nlead):
        temp = B_R[neq*(a-1):neq*a,:]
        newB_R[neq*a:neq*(a+1),:] = temp
        
    # Now, newB_R should be fully constructed
        
    # Calculate the final row of matrices in F by looping over the values
    # from 1 to nlead.
    for k in range(nlead,0,-1):
        # Calculate the matrix to be entered into F and enter it
        newEntry = -phi * H_plus * newB_R
        F[neq*(nlead-1):neq*nlead,neq*(k-1):neq*k] = newEntry
        if k > 1:
            # Update newB_R for the next matrix to be inserted by shifting down 
            # each element L rows
            for alpha in range(neq*(nlead-1)-1,-1,-1):
                for beta in range(0,neq):
                    newB_R[alpha+neq,beta] = newB_R[alpha,beta]
                
            # Then, insert 0's into the newly emptied cells.
            newB_R[0:neq,0:neq] = np.matrix(np.zeros(shape=((neq,neq))))
                
    return F

