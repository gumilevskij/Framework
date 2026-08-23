# -*- coding: utf-8 -*-
"""
Created on Fri Nov 1, 2019

.. note::  THIS CODE DEVELOPMENT IS IN A PROGRESS!

@author: A.Goumilevski
"""
from __future__ import division

import numpy as np
import scipy.linalg as la
from snowdrop.src.misc.termcolor import cprint
#from numeric.solver.util import qzdiv2
from snowdrop.src.numeric.solver.util import getParameters
from snowdrop.src.numeric.solver.util import sorter
from snowdrop.src.preprocessor.function import get_function_and_jacobian
     
count = 1

def solve_sims(model,steady_state,p=None,suppress_warnings=False):
    """
    Find first-order accuracy approximation solution.
    
    It implements algoritm described by Christopher Sims in paper:
    "Solving Rational Expectations Models."
    Please see http://sims.princeton.edu/yftp/gensys/LINRE3A.pdf
    
    Parameters:
        :param model: The Model object.
        :type model: instance of class `Model'.
        :param steady_state: Steady states solution.
        :type steady_state: list.
        :param p: Parameters values.
        :type p: list.
        :param suppress_warnings: Do not show warnings if set to True
        :type suppress_warnings: bool.
    """   
    global count
    if count == 1:
        count += 1
        cprint("Christopher Sims solver","blue")
        print()
    
    A, C, Cm, R, Z = None, None, None, None, None
    p = getParameters(parameters=p,model=model)
        
    forward_variables = [x for x in model.topology["forward"].keys() if not "_p_" in x]
    Nforward = len(forward_variables)
    
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
            C = -C_inv @ c
            Z = None
        elif  model.max_lead > 0 and model.min_lag == 0:
            A,C,R,Cm,G0I,theta_f,theta_z,fmat,fwt,ywt,gev,eu,loose = solution(g0=F,g1=-C,psi=-R,N=n,c=-c,N_nstable=Nforward)
            Z = (G0I,theta_f,theta_z,fmat,fwt,ywt,gev,eu,loose)
        else:
            c = np.concatenate((c,np.zeros(c.shape)),axis=0)
            # Build A,B matrices: A*[y(t+1),y(t)] = B*[y(t),y(t-1)] + R*[shock(t+1),0]
            a = np.zeros((2*n,2*n))
            a[:n,:n] = F
            a[n:,n:] = np.eye(n)
            b = np.zeros((2*n,2*n))
            b[:n,:n] = C
            b[:n,n:] = L
            b[n:,:n] = -np.eye(n)
            r = np.concatenate((R,np.zeros(R.shape)),axis=0)
            A,C,R,Cm,G0I,theta_f,theta_z,fmat,fwt,ywt,gev,eu,loose = solution(g0=a,g1=-b,psi=-r,N=n,c=-c,N_nstable=Nforward)
            Z = (G0I,theta_f,theta_z,fmat,fwt,ywt,gev,eu,loose)
    except :
        if not suppress_warnings:
            import sys
            sys.exit('Sims solver: unable to find solution of linear model.')
                
    model.linear_model["A"]  = A
    model.linear_model["C"]  = C
    model.linear_model["Cm"] = Cm
    model.linear_model["R"]  = R
    model.linear_model["Z"]  = Z
    
    return model


def solution(g0,g1,psi,N,c=0,pi=None,div=1+1.e-6,N_nstable=-1):
    """
    Solve linear system by Sims algorithm.
    
    System given as:
        
    .. math:: 
        g_{0}*y_{t} = g_{1}*y_{t-1} + c + psi * z_{t} + pi * eta_{t},
    
    with z an exogenous variable process and eta being endogenously determined one-step-ahead expectational errors.  
    
    Returned system is:
        
    .. math:: 
        y_{t} = G_{1}*y_{t-1} + C + impact * z_{t} + ywt * [I-fmat*(L)^{-1}]^{-1} * fwt * z_{t+1}.
    
    If z(t) is iid, the last term drops out.
    If div is omitted from argument list, a div>1 is calculated.
    eu[0]=1 for existence, 1 for existence only with not-s.c. z 
    eu[1]=1 for uniqueness.  
    eu=[-2,-2] for coincident zeros.

    By Christopher A. Sims 
    Matlab code is located at http://sims.princeton.edu/yftp/gensys/
    Corrected 10/28/96 by CAS
    
    For references pease see http://sims.princeton.edu/yftp/gensys/LINRE3A.pdf
     
    Translated from Matlab  to Python code by Alexei Goumilevski.
    """
    eu=[0, 0] 
    realsmall=1e-7
    fixdiv = not div is None
    n = len(g0)  
    a,b,alpha,beta,q,z = la.ordqz(g0,g1,output='complex',sort=sorter)
    
    # Python QZ algorithm returns transposed matrix versus Matlab
    q = np.conjugate(q.T)
        
    nunstab = 0 
    zxz = False
    for i in range(n):
     # ------------------div calc------------
       if not fixdiv:
          if abs(a[i,i]) > 0:
             divhat=abs(b[i,i])/abs(a[i,i]) 
    	     # bug detected by Vasco Curdia and Daria Finocchiaro, 2/25/2004  A root of
    	     # exactly 1.01 and no root between 1 and 1.02, led to div being stuck at 1.01
    	     # and the 1.01 root being misclassified as stable.  Changing < to <= below fixes this.
             if 1+realsmall<divhat and divhat<=div:
                div=0.5*(1+divhat) 
     # ----------------------------------------
       if abs(b[i,i])>div*abs(a[i,i]):
           nunstab+=1
       if abs(a[i,i])<realsmall and abs(b[i,i])<realsmall:
          zxz=True 

    nstab = n - nunstab
    #n_shocks = psi.shape[1]
    #print('nunstab:', nunstab, ' nstab:', nstab)
    # if not zxz:
    #     a,b,q,z,col_order = qzdiv2(div,a,b,q,z)
        
        
    # Check correctness of QZ decomposition
    err1 = la.norm(g0-q.T.conj()@a@z.T.conj()) / la.norm(g0)
    err2 = la.norm(g1-q.T.conj()@b@z.T.conj() )/ la.norm(g1)
    if err1 > 1.e-10 or err2 > 1.e-10:
        raise Exception("Sims Solver: QZ decomposition error.  \n Inconsistency of g0 and g1 matrix decomposition of {0} and {1}".format(round(err1,4),round(err2,4)))
      
    
    gev=[np.diag(a), np.diag(b)] 
    if zxz:
        cprint('Coincident zeros.  Indeterminacy and/or nonexistence.','red')
        eu=[-2, -2] 
        # correction added 7/29/2003.  Otherwise the failure to set output
        # arguments leads to an error message and no output (including eu).
        G1=[]; C=[]; impact=[]; fmat=[]; fwt=[]; ywt=[]; gev=[]; loose=[] 
        return G1,C,impact,fmat,fwt,ywt,gev,eu,loose
    
    q1=q[:nstab,:]
    q2=q[nstab:n,:]
#    z1=np.conjugate(z[:,:nstab].T)
#    z2=np.conjugate(z[:,nstab:n].T)
    a2=a[nstab:n,nstab:n]
    b2=b[nstab:n,nstab:n]
    if pi is None:
        pi = np.zeros((n,n))
    neta = len(pi[0]) 
    #neta1 = min(neta,nstab)
    etawt=q2 @ pi 
    etawt1=q1 @ pi 
     # zwt=q2*psi 
     # branch below is to handle case of no stable roots, which previously (5/9/09)
     # quit with an error in that case.
    ndeta = min(nunstab,neta)
    if ndeta == 0:
      ueta = np.zeros((nunstab,1)) 
      deta = 1 #ag np.zeros(0)
      veta = np.zeros((neta,1)) 
      bigev = []
    else:
      ueta,deta,veta=np.linalg.svd(etawt)
      deta=np.diag(deta)
      md=min(deta.shape)  
      bigev=np.where(np.diag(deta[:md,:md])>realsmall)
      bigev=bigev[0]
      ueta=ueta[:,bigev]
      veta=veta[:,bigev] 
      deta=deta[bigev,bigev]
    
     # ------ corrected code, 3/10/04
    eu[0] = len(bigev)>=nunstab 
     # ------ Code below allowed "existence" in cases where the initial lagged state was free to take on values
     # ------ inconsistent with existence, so long as the state could w.p.1 remain consistent with a stable solution
     # ------ if its initial lagged value was consistent with a stable solution.  This is a mistake, though perhaps there
     # ------ are situations where we would like to know that this "existence for restricted initial state" situation holds.
    ## [uz,dz,vz]=svd(zwt) 
    ## dz = np.diag(dz)
    ## md=min(dz.shape) 
    ## bigev=find(np.diag(dz[:md,:md])>realsmall) 
    ## bigev=np.where(np.diag(dz[:md,:md])>realsmall)
    ## bigev=bigev[0]
    ## uz=uz[:,bigev] 
    ## vz=vz[:,bigev] 
    ## dz=dz[bigev,bigev]
    ## if not bool(bigev):
    ## 	  exist=1 
    ## else:
    ## 	exist=norm(uz-ueta@np.conjugate(ueta.T)@uz) < realsmall*n 
    ## 
    ## if bool(bigev):
    ## 	  zwtx0=la.inv(zwt) @ b2 
    ## 	  zwtx=zwtx0 
    ## 	  M=la.inv(a2)@b2 
    ## 	  for i in range(1,nunstab):
    ## 		zwtx=np.concatenate((M@zwtx, zwtx0), axis=1) 
    #
    ## 	  zwtx=b2@zwtx 
    ## 	  ux,dx,vx=np.linalg.svd(zwtx) 
    ##    dx = np.diag(dx) 
    ## 	  md=min(dx.shape) 
    ## 	  bigev=find(np.diag(dx(1:md,1:md))>realsmall) 
    ##    bigev=np.where(np.diag(dx[:md,:md])>realsmall)
    ##    bigev=bigev[0]
    ## 	  ux=ux[:,bigev]
    ## 	  vx=vx[:,bigev]
    ## 	  dx=dx[bigev,bigev] 
    ## 	  existx=norm(ux-ueta@np.conjugate(ueta.T)@ux) < realsmall*n 
    ## else:
    ## 	  existx=1 
    ##
     # ----------------------------------------------------
     # Note that existence and uniqueness are not just matters of comparing
     # numbers of roots and numbers of endogenous errors.  These counts are
     # reported below because usually they point to the source of the problem.
     # ------------------------------------------------------
     # branch below to handle case of no stable roots
    ndeta1 = min(nstab,neta)
    if ndeta1 == 0:
      ueta1 = np.ones((nstab,1)) 
      veta1 = np.ones((neta,1)) 
      deta1 = np.zeros((1,nstab))
      bigev = [] 
    else:
      [ueta1,deta1,veta1]=np.linalg.svd(etawt1) 
      deta1=np.diag(deta1)
      md=min(deta1.shape) 
      bigev=np.where(np.diag(deta1[:md,:md])>realsmall)
      bigev=bigev[0]
      ueta1=ueta1[:,bigev]
      veta1=veta1[:,bigev]
      deta1=deta1[bigev,bigev]
    
    ## if existx | nunstab==0:
    ##    print('solution exists') 
    ##    eu[0]=1 
    ## else:
    ##     if exist:
    ##         print('solution exists for unforecastable z only') 
    ##         eu[0]=-1 
    ##     else:
    ##         printf('No solution.  %{0} unstable roots. %{1} endog errors.\n'.format(nunstab,len(ueta1[0]))) 
    ## 
    ##     print('Generalized eigenvalues')
    ##     print(gev) 
    ##     md=abs(np.diag(a))>realsmall 
    ##     ev=np.diag(md@np.diag(a)+(1-md)@np.diag(b))\ev 
    ##     print(ev)
    ##  #   return 
    
    if len(veta1)==0:
    	unique=True
    else:
    	loose = veta1-veta@np.conjugate(veta.T)@veta1 
    	[ul,dl,vl] = np.linalg.svd(loose) 
    	nloose = sum(abs(np.diag(dl)) > realsmall*n) 
    	unique = (nloose == 0) 
    
    if unique:
       #print('solution unique') 
       eu[1]=1 
    else:
       cprint('Indeterminacy.  {} Loose endog errors.\n'.format(nloose),'red') 
       print('Generalized eigenvalues:')
       print(gev) 
#       md=abs(np.diag(a))>realsmall 
#       ev=la.inv(np.diag(md@np.diag(a)+(1-md)@np.diag(b))) @ ev 
#       print(ev)
     #   return 
    
    if len(deta) == 0:
        temp = ueta @ np.conjugate(veta.T) @ veta1
    else:
        temp = ueta @ (np.conjugate(veta.T)/deta) @ veta1
    if len(deta1) == 0:
        temp = temp @ np.conjugate(ueta1.T)
    else:
        temp = temp @ (deta1 @ np.conjugate(ueta1.T))
    tmat = np.concatenate((np.eye(nstab),-np.conjugate(temp.T)), axis=1)
    temp = np.concatenate((np.zeros((nunstab,nstab)), np.eye(nunstab)), axis=1)
    G0 = np.concatenate((tmat @ a, temp), axis=0) 
    G1 = np.concatenate((tmat @ b, np.zeros((nunstab,n))), axis=0)
    # ----------------------
    # G0 is always non-singular because by construction there are no zeros on
    # the diagonal of a(1:nstab,1:nstab), which forms G0's ul corner.
    # -----------------------
    G0I = la.inv(G0) 
    G1 = G0I @ G1 
    au = a[nstab:n,nstab:n]
    bu = b[nstab:n,nstab:n]
    temp = np.linalg.solve(au-bu, q2)
    Cm = G0I @ np.concatenate((tmat @ q, temp), axis=0)
    Cm = z @ Cm
    C = np.dot(Cm, c) 
    temp = np.zeros((nunstab,len(psi[0])))
    impact = G0I @ np.concatenate((tmat @ q @ psi, temp), axis=0) 
    fmat = np.linalg.solve(bu, au)
    fwt = -np.linalg.solve(bu, q2) @ psi 
    ywt = G0I[:,nstab:n]
    
    # -------------------- above are output for system in terms of z'y -------
    G1 = z @ G1 @ np.conjugate(z.T)
    impact = z @ impact 
    
    # Correction 5/07/2009:  formerly had forgotten to premultiply by G0I
    temp = np.eye(neta) - veta @ np.conjugate(veta.T)
    loose = G0I @ np.concatenate((etawt1 @ temp, np.zeros((nunstab, neta))), axis=0)
    loose = np.real(z @ loose) 
    
    # Correction 10/28/96:  formerly line below had real(z@ywt) on rhs, an error.
    ywt = z @ ywt 
    
    # Added calculation of terms related to exogenous variables z(t+1), z(t+2), ... 
    bI = la.inv(b2)
    theta_f = bI @ a2
    theta_z = bI @ q2 @ psi
    H = z @ G0I
                    
    return G1,C,impact,Cm,H,theta_f,theta_z,fmat,fwt,ywt,gev,eu,loose


if __name__ == '__main__':
    
    g0 = np.array( [[-0.5,0,0,0,0,0], 
                    [-0.5,-1,0,0,0,0],
                    [0,0,0,0,0,0],
                    [0,0,0,1,0,0],
                    [0,0,0,0,1,0],
                    [0,0,0,0,0,1]] 
                 )
    
    g1 = np.array( [[-1,0.5,0,0.5,0,0],
                    [0,-1,-0.5,0,0,0],
                    [1.5,0,-1,0,0,0],
                    [1,0,0,0,0,0],
                    [0,1,0,0,0,0],
                    [0,0,1,0,0,0]]
                )
    
    n = len(g0)
    
    psi = np.array([0,1,0,0,0,0]).reshape(n,1)
    c = np.array([0,0.005,0.015,0,0,0]).reshape(n,1)
    
    G1,C,impact,Cm,H,theta_f,theta_z,fmat,fwt,ywt,gev,eu,loose = solution(g0=g0,g1=g1,c=c,psi=psi,N=n)
    
    gev = np.array(gev).T
    
    np.set_printoptions(precision=3)
    print('\nG1:')
    print(np.real(G1))
    print('\nC:')
    print(np.real(C))
    print('\nimpact:')
    print(np.real(impact))
    print('\nfmat:')
    print(fmat)
    print('\nfwt:')
    print(fwt)
    print('\nywt:')
    print(np.real(ywt))
    print('\ngev:')
    print(gev)
    print('\neu:')
    print(np.real(eu))
    print('\nloose:')
    print(np.real(loose))
    
    
    # # Matlab Results    
    # G1 =[[0.4464   -0.0983   -0.0307   -0.0370        0         0],
    #     [-0.4083    0.0900    0.0281    0.0338        0         0],
    #     [0.6696   -0.1475   -0.0460   -0.0555         0         0],
    #     [1.1238   -0.2476   -0.0772   -0.0931         0         0],
    #     [-0.0000    1.0000    0.0000   -0.0000        0         0],
    #     [0.0000    0.0000    1.0000   -0.0000         0         0]] 
    
        
    # C = [
    #    -0.0059
    #    -0.0037
    #     0.0061
    #     0.0003
    #          0
    #          0
    # ]
    
    # impact =[
   #        0.0614
   #       -0.0561
   #        0.0920
   #        0.1545
   #        0
   #        0
   #  ]
    
    # fmat =[
    #    0.0000 - 0.0000i  -0.7182 + 0.1926i  -0.4327 + 0.2215i
    #    0.0000 + 0.0000i   0.6162 - 0.1322i   0.3581 + 0.0417i
    #    0.0000 + 0.0000i   0.0000 + 0.0000i   0.6162 + 0.1322i2i]
    
    # fwt = [
    #    0.5577 + 0.0000i
    #   -0.4518 - 0.2098i
    #   -0.5420 - 0.1303i]
    
    # ywt = [
    #    0.0000 - 0.0000i  -1.0012 + 0.0291i   0.1256 + 0.3330i
    #   -0.0000 + 0.0000i  -0.0518 + 0.3159i  -0.9509 - 0.0147i
    #    1.1293 - 0.0000i  -0.1711 - 0.0239i   0.1093 + 0.0598i
    #   -0.0000 + 0.0000i   0.0084 + 0.0516i  -0.1588 - 0.0081i
    #    0.0000 + 0.0000i   0.0000 + 0.0000i   0.0000 + 0.0000i
    #    0.0000 + 0.0000i   0.0000 + 0.0000i   0.0000 + 0.0000i]
    
    # gev = [ 
    #    1.0000 + 0.0000i   0.0000 + 0.0000i
    #    1.0000 + 0.0000i   0.0000 + 0.0000i
    #   -0.8053 - 0.0000i  -0.3199 + 0.0000i
    #   -0.0000 + 0.0000i  -1.2593 + 0.0000i
    #    0.7476 - 0.1603i   1.2132 + 0.0000i
    #    0.6305 + 0.1352i   1.0231 + 0.0000i]
       
    #    eu = [ 
    #      0
    #      1]
         
    #   loose = [
    #      0     0     0     0     0     0
    #      0     0     0     0     0     0
    #      0     0     0     0     0     0
    #      0     0     0     0     0     0
    #      0     0     0     0     0     0
    #      0     0     0     0     0     0]
