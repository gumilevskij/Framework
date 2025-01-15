# import numpy and scipy packages
import numpy as np 
from scipy import *
from scipy.sparse import lil_matrix
import scipy.linalg as la

def buildA(h,qcols,neq):

    """ Build the companion matrix, deleting inessential lags.
    Solve for x_{t+nlead} in terms of x_{t+nlag},...,x_{t+nlead-1}.
    
    Original author: Gary Anderson
    Original file downloaded from:
    http://www.federalreserve.gov/Pubs/oss/oss4/code.html
    
    This code in the public domain and may be used freely.
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
    
    
    left  = list(range(0,qcols))
    right = list(range(qcols,qcols+neq))
    hs = lil_matrix(h)
    #a0 = hs[:,right]
    lu, piv = la.lu_factor(hs[:,right].toarray())
    hs[:,left] = -la.lu_solve((lu,piv),hs[:,left].toarray(),trans=0)
    
    #  Build the big transition matrix.
    
    a = np.matrix(np.zeros(shape=((qcols,qcols))))
    
    if qcols > neq:
        eyerows = range(0,qcols-neq)
        eyecols = range(neq,qcols)
        a[np.ix_(eyerows,eyecols)] = np.eye(qcols-neq)
    
    hrows = list(range(qcols-neq,qcols))
    a[hrows,:] = hs[:,left].toarray()

    #  Delete inessential lags and build index array js.  js indexes the
    #  columns in the big transition matrix that correspond to the
    #  essential lags in the model.  They are the columns of q that will
    #  get the unstable left eigenvectors. 
    
    js = list(range(0,qcols))
    zerocols = list()
    sumVector = abs(a).sum(axis=0)
    sumVectorRows, sumVectorCols = sumVector.shape
    for i in range(0,sumVectorCols):
        if sumVector[0,i] == 0:
            zerocols.append(i)

    while len(zerocols) > 0:
        a = np.delete(a,zerocols,1)
        a = np.delete(a,zerocols,0)
        js = np.delete(js,zerocols)
        while len(zerocols) > 0:
            zerocols.pop()
        sumVector2 = abs(a).sum(axis=0)
        sumVector2Rows, sumVector2Cols = sumVector2.shape
        for i in range(0,sumVector2Cols):
            if sumVector2[0,i] == 0:
                zerocols.append(i)

    ia = len(js)
    return a, ia, js
