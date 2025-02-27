# import numpy and scipy packages
import numpy as np
from scipy import linalg as la
from scipy.sparse import lil_matrix

def Eigensystem(a,uprbnd,rowsLeft):
    """
    Compute the roots and the left eigenvectors of the companion
    matrix, sort the roots from large-to-small, and sort the
    eigenvectors conformably.  Map the eigenvectors into the real
    domain. Count the roots bigger than uprbnd.
    
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

    rts, w = la.eig(a.T)
    #mag = sorted(-abs(rts))
    k = np.argsort(-abs(rts))
    rts = rts[k]

    ws = lil_matrix(w)
    ws = ws[:,k]
    
    #  Given a complex conjugate pair of vectors W = [w1,w2], there is a
    #  nonsingular matrix D such that W*D = real(W) + imag(W).  That is to
    #  say, W and real(W)+imag(W) span the same subspace, which is all
    #  that aim cares about. 
    
    ws = ws.real + ws.imag
    
    lgroots = sum(abs(np.asarray(rts)) > uprbnd)
    
    w = lil_matrix(ws).todense()
    
    return w, rts, lgroots
