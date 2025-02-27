# Import the numpy and scipy packages
import numpy as np
from scipy.sparse import csr_matrix
from snowdrop.src.numeric.solver.AIM.Shifts import Shiftright

def Obstruct(cof,cofb,neq,nlag,nlead):
    """
    Construct the coefficients in the observable structure.
    
    Input arguments:
               cof    structural coefficients
               cofb   reduced form
               neq    number of equations
               nlag   number of lags
               nlead  number of leads
    
      Output arguments:
               scof  observable structure coefficients
    
    Original author: Gary Anderson
    Original file downloaded from:
    http://www.federalreserve.gov/Pubs/oss/oss4/code.html
    
    This code is in the public domain and may be used freely.
    However the authors would appreciate acknowledgement of the source by
    citation of any of the following papers:
    
    Anderson, G. and Moore, G.
    "A Linear Algebraic Procedure for Solving Linear Perfect Foresight
    Models.", Economics Letters, 17, 1985.
    
    Anderson, G.
    "Solving Linear Rational Expectations Models: A Horse Race"
    Computational Economics, 2008, vol. 31, issue 2, pages 95-113
    
    Anderson, G.
    "A Reliable and Computationally Efficient Algorithm for Imposing the
    Saddle Point Property in Dynamic Models"
    Journal of Economic Dynamics and Control, 2010, vol. 34, issue 3,
    pages 472-489
    """
    

    # Append the negative identity to cofb

    cofb = np.concatenate((cofb.T,-np.eye(neq))).T
    scof = np.matrix(np.zeros(shape=((neq,neq*(nlag+1)))))
    qq = np.matrix(np.zeros(shape=((neq*nlead,neq*(nlag+nlead)))))
    rc, cc = cofb.shape
    qs = csr_matrix(qq)
    qs[0:rc,0:cc] = csr_matrix(cofb)
    #qcols = neq*(nlag+nlead)
    
    if nlead > 1: 
        for i in range(1,nlead):
            rows = range(0,neq)
            shiftRows = range(0,neq)
            for j in range(0,len(rows)):
                rows[j] = rows[j] + i*neq
                shiftRows[j] = rows[j] - neq
            qs[rows,:] = Shiftright( qs[shiftRows,:], neq )

    l = range(0,neq*nlag)
    r = range(neq*nlag,neq*(nlag+nlead))

    qs[:,l] = -qs[:,r].I * qs[:,l]

    minus = np.arange(0,neq*(nlag+1))
    plus  = np.arange(neq*(nlag+1),neq*(nlag+1+nlead))
    
    cofs = csr_matrix(cof)
    scof[:,neq:neq*(nlag+1)] = cofs[:,plus] * qs[:,l]
    scof = scof + cofs[:,minus]
    
    return scof
