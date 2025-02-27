from __future__ import print_function
import numpy as np
from snowdrop.src.misc.termcolor import cprint
from numba import guvectorize
from numpy.linalg import solve as linalg_solve
from scipy import linalg as la 

   
def solve(m, sol):
    """
    Seems to segfault on windows.
    
    Args:
        m : TYPE
            DESCRIPTION.
        sol : TYPE
            DESCRIPTION.

    Returns:
        None.

    """
    h,w = m.shape

    for y in range(0,h):
        maxrow = y
        for y2 in range(y+1, h):    # Find max pivot
            if abs(m[y2,y]) > abs(m[maxrow,y]):
                maxrow = y2
        for y2 in range(0,w):
            t = m[y,y2]
            m[y,y2] = m[maxrow,y2]
            m[maxrow,y2] = t

        for y2 in range(y+1, h):    # Eliminate column y
            c = m[y2,y] / m[y,y]
            for x in range(y, w):
                m[y2,x] -= m[y,x] * c

    for y in range(h-1, 0-1, -1): # Back substitute
        c  = m[y,y]
        for y2 in range(0,y):
            for x in range(w-1, y-1, -1):
                m[y2,x] -=  m[y,x] * m[y2,y] / c
        m[y,y] /= c
        for x in range(h, w):       # Normalize row y
          m[y,x] /= c

    for y in range(h):
        sol[y] = m[y,w-1]


def serial_solve(A, B, diagnose=True):
    """Generalized universal functions solve."""
    if diagnose:
        sol = np.zeros_like(B)
        for i in range(sol.shape[0]):
            try:
                sol[i,:] = linalg_solve( A[i,:,:], B[i,:])
            except:
                # Should be a special type of exception
                a = Exception("Error solving point {}".format(i))
                a.x = B[i,:]
                a.J = A[i,:,:]
                a.i = i
                raise a

    else:
        serial_solve_numba = guvectorize('void(f8[:,:], f8[:])', '(m,n)->(m)')(solve)
        M = np.concatenate([A,B[:,:,None]],axis=2)
        sol = np.zeros_like(B)
        serial_solve_numba(M,sol)

    return sol


def newton(f, x, verbose=False, tol=1e-6, maxit=5, jactype='serial'):
    """
    Solve nonlinear system using safeguarded Newton iterations
    Args:
    Returns:
    """
    if verbose:
        print = lambda txt: print(txt)
    else:
        print = lambda txt: None

    it = 0
    converged = False
    maxbacksteps = 30

    if jactype == 'sparse':
        from scipy.sparse.linalg import spsolve as solve
    elif jactype == 'full':
        from numpy.linalg import solve
    else:
        solve = serial_solve

    while it<maxit and not converged:

        [v,dv] = f(x)
        # TODO: rewrite starting here
        error_0 = abs(v).max()
        if error_0 < tol:

            if verbose:
                print("> System was solved after iteration {}. Residual={}".format(it,error_0))
            converged = True

        else:

            it += 1
            dx = solve(dv, v)
            # norm_dx = abs(dx).max()

            for bck in range(maxbacksteps):
                xx = x - dx*(2**(-bck))
                vm = f(xx)[0]
                err = abs(vm).max()
                if err < error_0:
                    break
            x = xx

            if verbose:
                print("\t> {} | {} | {}".format(it, err, bck))

    if not converged:
        import warnings
        warnings.warn("Did not converge")
        
    return [x, it]


def SerialDifferentiableFunction(f, epsilon=1e-8):

    def df(x):
        v0 = f(x)
        N = v0.shape[0]
        n_v = v0.shape[1]
        assert(x.shape[0] == N)
        n_x = x.shape[1]
        dv = np.zeros( (N, n_v, n_x) )

        for i in range(n_x):
            xi = x.copy()
            xi[:,i] += epsilon
            vi = f(xi)
            dv[:,:,i] = (vi - v0)/epsilon

        return [v0, dv]

    return df


def test_serial_solve():

    N = 10
    A = np.random.random( (N,2,2) )
    B = np.random.random( (N,2) )

    print(A)
    print(B)
    out = serial_solve(A,B)
    print("A")
    print(A)
    print("B")
    print(B)
    print("out")
    print(out)
    
    res = np.linalg.solve(A[0,:,:], B[0,:])
    print(res)


def lyapunov_solver(T,R,Q,N,n_shocks,method=1,options=None):
    """
    Solve the Lyapunov equation: P = T*P*T' + R*Q*R'.
    
    This equation arising in a state-space
    system, where P is the variance of the states.
 
    Args:
      T:  double      
          n*n matrix
      R:  double      
          n*n matrix
      Q:  double      
          nsh*nsh matrix
      method:  int         
               method number
      N:  int         
          shocks maximum lead number minus minimum lag number
      n_shocks: int         
                number of shocks
      options:  dictionary  
                options
    Returns:
      P:   double      
           nsh*nsh matrix.
 
    Algorithms:
      options:
          steady: The transition matrix is decomposed into stable and unstable parts. For the stable part we solve the Lyapinov equation with the help of the scipy package Lyapunov solver.
          
          fp:     True iteration-based fixed point algorithm is used.
          
          db:     Then doubling algorithm is used.
          
          If none of these algorithms is selected then the reordered Schur decomposition, a.k.a. 
          Bartels-Stewart algorithm is used.
          
    """
    Q2 = 0
    for i in range(1+N):
        R1 = R[:,i*n_shocks:(1+i)*n_shocks]
        Q2 += np.real(R1 @ Q @ R1.T)
    
    if options is None:
        P = la.solve_discrete_lyapunov(T,Q2)
    elif options == "stochastic":
        n   = len(T)
        # Find unstable eigen values and eigen vectors
        # T is the Hermitian (aka conjugate symmetric) matrix
        eig,S = np.linalg.eigh(T)
        ind = [i for i in range(n) if abs(eig[i]) < 1-1.e-8]
        if len(ind) == 0:
            return None
        # Decompose transition matrix into stable T1 and unstable T2 parts
        S  = S[:,ind]
        T1 = S.T @ T @ S
        Q1 = S.T @ Q2 @ S
        X  = la.solve_discrete_lyapunov(T1, Q1)  
        P  = S @ X @ S.T
    elif options == "discrete":
        P    = la.solve_discrete_lyapunov(T, Q2)  
    elif options == "steady_state":
        P    = la.solve_lyapunov(T, Q2) 
    elif options == "symmetric":
        P,u = lyapunov_symm(a = T,b = Q2, method = method)
    elif options == "doubling_algorithm":
        P, errorflag = disclyap_fast(G = T, V = Q2)
        if errorflag:  #use Schur-based method
            P,u = lyapunov_symm(a = T, b = Q2)
    else:
        P,u = lyapunov_symm(a = T, b = Q2, method = 1)
   
    return np.real(P)


def disclyap_fast(G,V,tol=1.e-16,check_flag=None):
    """
    Args:
      G:             [double]    first input matrix
      V:             [double]    second input matrix
      tol:           [scalar]    tolerance criterion
      check_flag:    if non-empty - check positive-definiteness
    Returns:
      X:             [double]    solution matrix
      exitflag:      [scalar]    0 if solution is found, 1 otherwise
    
    Solve the discrete Lyapunov equation by using the Doubling Algorithm
    
    .. math::
        X = G * X * G' + V
    
    
    If check_flag is defined then the code will check if the resulting X
    is positive definite and generate an error message if it is not
    
    Joe Pearlman and Alejandro Justiniano
    3/5/2005
    """
    
    if check_flag is None:
        flag_ch = False
    else:
        flag_ch = True
    
    exitflag = False
    
    P0 = V
    A0 = G
    
    matd = 1
    iter = 0
    while matd > tol and iter< 2000:
        P1 = P0 + A0 @ P0 @ A0.T
        A1 = A0 @ A0
        matd = np.max(abs(P1-P0)) 
        P0 = P1
        A0 = A1
        iter += 1
  
    if iter==5000:
        X = np.NaN(P0)
        exitflag = True
        return X,exitflag

    del A0, A1, P1
    
    X=(P0+P0.T)/2
    
    # Check that X is positive definite
    if flag_ch:
        [C,p] = la.cholesky(X)
        if not p == 0:
            exitflag = True
            raise ValueError('X is not positive definite')

    
    return X,exitflag
    

def lyapunov_symm(a,b,method=0,X=None,lyapunov_fixed_point_tol=1.e-10,qz_criterium=1+1.e-6,lyapunov_complex_threshold=1.e-15,debug=False):
    """
    Solves the Lyapunov equation x-a*x*a' = b, for b and x symmetric matrices.
    If a has some unit roots, the function computes only the solution of the stable subsystem.
    
    Args:
      a:                           [double]    n*n matrix.
      
      b:                           [double]    n*n matrix.
      
      qz_criterium:                [double]    unit root threshold for eigenvalues
      
      lyapunov_fixed_point_tol:    [double]    convergence criteria for fixed_point algorithm.
      
      lyapunov_complex_threshold:  [double]    scalar, complex block threshold for the upper triangular matrix T.
      
      method:                      [integer]   Scalar, if method=0 [default] then U, T, n and k are not persistent; method=1 then U, T, n and k are declared as persistent; method=3 fixed point method
    Returns:
      x:      [double]    m*m solution matrix of the lyapunov equation, where m is the dimension of the stable subsystem.
      u:      [double]    Schur vectors associated with unit roots
    
    Algorithm:
      Uses reordered Schur decomposition (Bartels-Stewart algorithm)
      [method<3] or a fixed point algorithm (method==3)
    
    """
    u = []
    if debug:
        print('lyapunov_symm:: [method={}] \n'.format(method))
    
    if method == 0:
        
        #tol = 1e-10
        it_fp = 0
        evol = 100
        if X is None or not len(X)==len(b):
            X = b
            max_it_fp = 2000
        else:
            max_it_fp = 300
        
        at = a.T
        #fixed point iterations
        while evol >  lyapunov_fixed_point_tol and it_fp < max_it_fp:
            X_old = X
            X = a @ X @ at + b
            evol = np.max(np.sum(abs(X - X_old))) #norm_1
                                             #evol = max(sum(abs(X - X_old)')) #norm_inf
            it_fp += 1
       
        if debug:
            print('lyapunov_symm: lyapunov fixed_point iterations={0} norm={1}\n'.format(it_fp,evol))
        
        if it_fp >= max_it_fp:
            cprint('lyapunov_symm: convergence not achieved in solution of Lyapunov equation after {} '.format(it_fp) + ' iterations, switching method from 3 to 0','red')
            method = 0
        else:
            x = X
            return x,u

    else:
        
        if len(a) == 1:
            x=b/(1-a*a)
            return x,u
    
        T,U,sdim = la.schur(a,sort = lambda x: abs(x) > 2-qz_criterium)
        e1 = abs(np.diag(T)) > 2-qz_criterium
        k = sum(e1)       # Number of unit roots.
        n = len(e1)-k     # Number of stable variables.
        if k > 0:
            # # Re-arrange matrices in such a way that unstable generalized eigenvalues are in the upper left corner of matrices: s,t
            from snowdrop.src.utils.sortSchur import sort_schur_decomposition
            U,T,ap = sort_schur_decomposition(Q=U,R=T,z=2-qz_criterium,b=0)
            k = sum(abs(np.diag(T))>2-qz_criterium)
            
        # Selects stable roots
        T = T[k:,k:]
        B = U[:,k:].T @ b @ U[:,k:]
        x = np.zeros((n,n))
        i = n-1
        
        while i >= 1:
            if abs(T[i,i-1])<lyapunov_complex_threshold:
                if i == n-1:
                    c = np.zeros(n)
                else:
                    c = T[:1+i,:] @ (x[:,i+1:]@T[i,i+1:].T) + \
                        T[i,i]*T[:1+i,i+1:] @ x[i+1:,i]
    
                q = np.eye(1+i)-T[:1+i,:1+i]*T[i,i]
                x[:1+i,i] = la.solve(q,B[:1+i,i]+c)
                x[i,:i] = x[:i,i].T
                i -= 1
    
            else:
                if i == n-1:
                    c  = np.zeros(n)
                    c1 = np.zeros(n)
                else:
                    c = T[:1+i,:]  @ (x[:,i+1:] @ T[i,i+1:].T) + \
                        T[i,i]     * T[:1+i,i+1:] @ x[i+1:,i] + \
                        T[i,i-1]   * T[:1+i,i+1:] @ x[i+1:,i-1]
                    c1= T[:1+i,:]  @ (x[:,i+1:] @ T[i-1,i+1:].T) + \
                        T[i-1,i-1] * T[:1+i,i+1:] @ x[i+1:,i-1] + \
                        T[i-1,i]   * T[:1+i,i+1:] @ x[i+1:,i]
    
                
                tmp1 = np.concatenate((np.eye(1+i)-T[:1+i,:1+i]*T[i,i],-T[:1+i,:1+i]*T[i,i-1]),axis=1)
                tmp2 = np.concatenate((-T[:1+i,:1+i]*T[i-1,i],np.eye(1+i)-T[:1+i,:1+i]*T[i-1,i-1]),axis=1)
                q = np.concatenate((tmp1,tmp2),axis=0)
                tmp = np.concatenate((B[:1+i,i]+c,B[:1+i,i-1]+c1),axis=0)
                z =  la.solve(q,tmp)
                x[:1+i,i] = z[:1+i]
                x[:1+i,i-1] = z[1+i:]
                x[i,:i] = x[:i,i].T
                x[i-1,:i-1] = x[:i-1,i-1].T
                i -= 2
        
        if i == 0:
            c = T[0] @ (x[:,1:] @ T[0,1:].T) + T[0,0] * T[0,1:] @ x[1:,0]
            x[0,0] = (B[0,0]+c)/(1-T[0,0]**2)
    
        x = U[:,k:] @ x @ U[:,k:].T
        u = U[:,:k]
        
        return x,u


def sylvester_solver(A,C,D,B=None):
    """
    Solves Silvester equation:
        
    .. math:: 
        
        A*x + B*x*C = D
    """
    if B is None:
        x = la.solve_sylvester(A,C,D)
    else:
        if np.ndim(B) == 2 and B.shape[0] == B.shape[1] and la.det(B) > 0:
            b_inv = la.inv(B)
            x  = la.solve_sylvester(b_inv @ A,b_inv @ C,b_inv @ D)
        else:
            x = Sylvester_solver(A=A,B=B,C=C,D=D)
    
#    err = np.max(A@x+B@x@C-D)
#    if err > 1.e-10:
#        cprint("Sylvester equation solver error: large residual - {}".format(np.round(err,2)),"red")
        
    return x


def Sylvester_iterative_solver(x0,A,B,C,D):
    """
    Iterative solution of Sylvester equation.
        
    .. math:: 
        
        A*x + B*x*C = D
    """
    N = 500
    a_inv = la.inv(A)
    b = a_inv @ B
    flag = 0
    p = D.shape[2]
    for j in range(p):
        d = a_inv @ D[:,:,j]
        e = 1.0
        it = 0
        while e > 1e-8 and it < N:
            x = d - b @ x0[:,:,j] @ C
            e = np.max(abs(x-x0[:,:,j]))
            x0[:,:,j] = x
            it += 1
   
        if it == N:
            cprint('Sylvester iterative solver: Only accuracy of {} is achieved after {} iterations'.format(e,N),'red')
            flag = 1
        
    return x0, flag, it


def Sylvester_solver(A,B,C,D):
    """"
    Solves Silvester equation.
        
    .. math:: 
        
        A*x + B*x*C = D
        
    where D is n*m*p matrix
    """
    
    n = A.shape[0]
    m = C.shape[0]
    p = D.shape[2]
    d = np.zeros(D.shape)
    CC= np.squeeze(C)
    x = np.zeros((n,m,p))
    
    if n == 1:
        tmp = np.repeat(A,m)+B@C
        for j in range(p):
            z = np.divide(D[:,:,j],tmp)
            x[:,:,j] = z
        return x
    if isinstance(CC,int) or isinstance(CC,float):
        for j in range(p):
            x[:,:,j] = la.solve(A+CC*B,D[:,:,j])
        return x
 
    t,u = la.schur(C)
    aa,bb,qq,zz = la.qz(A,B,'real') 
    qq = qq.T
    for j in range(p):
        d[:,:,j]=qq @ D[:,:,j] @ u

    i  = 0
    c  = np.zeros((n,p))
    c1 = np.zeros((n,p))
    while i < m-1:
        if t[i+1,i] == 0:
            if i == 0:
                c = np.zeros((n,p))
            else:
                for j in range(p):
                    c[:,j] = bb @ (x[:,:i-1,j] @ t[:i-1,i])
                    
            x[:,i,:] = la.solve(aa+bb*t[i,i],np.squeeze(d[:,i,:]-c))
            i += 1
        else:
            if i == n-1:
                c = np.zeros((n,p))
                c1 = np.zeros((n,p))
            else:
                for j in range(p):
                    c[:,j]  = bb @ (x[:,:i-1,j] @ t[:i-1,i])
                    c1[:,j] = bb @ (x[:,:i-1,j] @ t[:i-1,i+1])

            tmp1 = np.concatenate((aa+bb*t[i,i], bb*t[i+1,i]),axis=1)
            tmp2 = np.concatenate((bb*t[i,i+1],  aa+bb*t[i+1,i+1]),axis=1)
            bigmat = np.concatenate((tmp1,tmp2),axis=0)
            tmp1 = np.squeeze(d[:,i,:]-c)
            tmp2 = np.squeeze(d[:,i+1,:]-c1)
            tmp = np.concatenate((tmp1,tmp2),axis=0)
            z = la.solve(bigmat,np.squeeze(tmp))
            x[:,i] = z[:n]
            x[:,i+1] = z[n:]
            i += 2

    if i == m-1:
        for j in range(p):
            c[:,j] = (bb @ (x[:,:i,j] @ t[:i,i])).T

        aabbt = aa + bb * t[i,i]
        x[:,i,:] = la.solve(aabbt,np.squeeze(d[:,i,:]-c))
  
    for j in range(p):
        x[:,:,j] = zz @ x[:,:,j] @ u.T
    
    return x


def test1():
    """Test Lypunov equations solvers."""
    # Solves the Lyapunov equation x-a*x*a' = b, for b and x symmetric matrices.
    # n = 5
    # a = np.random.rand(n,n); a += a.T
    # b = np.random.rand(n,n); b += b.T
    # a = np.round(a,2)
    # b = np.round(b,2)
    # print(a)
    # print(b)
    a = np.array([[0.29,1.47,0.62,0.77,1.71],
                 [1.47,1.27,1.24,1.68,0.3 ],
                 [0.62,1.24,1.58,1.04,0.51],
                 [0.77,1.68,1.04,0.87,1.31],
                 [1.71,0.3,0.51,1.31,1.56 ]])
    b =  np.array([[0.87,1.35,0.8,1.12,0.8 ],
                 [1.35,0.52,1.21,1.18,0.57],
                 [0.8,1.21,1.34,1.87,1.17 ],
                 [1.12,1.18,1.87,1.23,1.07],
                 [0.8,0.57,1.17,1.07,0.77 ]])
    print()
    print("Lyapunov solution:")
    x,u = lyapunov_symm(a,b,method=1)
    
   #  # Matlab results:
   #  0.0478    0.1680   -0.2258    0.0579   -0.0837
   #  0.1680    0.1168   -0.0519   -0.1805   -0.0446
   # -0.2258   -0.0519   -0.0948    0.3278    0.0047
   #  0.0579   -0.1805    0.3278   -0.2412    0.1008
   # -0.0837   -0.0446    0.0047    0.1008    0.0151
        
    print(x)
    print()
    print("Error:")
    err = x + a@x@a.T - b
    print(err)
       
    # Test Silvester equation solver: a*x + b*x*c = d
    a = np.array([[-3, -.2, 0], [-1, -.1, 0], [0, -.5, -1]])
#    c = np.ones((1,1)) #np.array([[ 2,  4,-1], [-1, -1, 0], [0, -5, -1]])
    b = np.array([[ .3,  .1,-.1], [-.1, +.1, .3], [0, +.5, +1]])
#    d = np.ones((3,1,3))
#    x =  sylvester_solver(A=a,B=b,C=c,D=np.copy(d))
#    print()
#    print("Sylvester solution:")
#    print(x)
    
    #print()
    #x = la.solve_discrete_lyapunov(a,b,method='direct')  
    #print(x)
   
    if False:
        # Test serial solve
        test_serial_solve()
        
        # Test Lyapunov equation  solver
        t = np.zeros(7)
        n_small=8
        m_small=10
        T_small=np.random.randn(n_small,n_small)
        eig = la.eig(T_small)[0]
        t1 = abs(np.amax(eig))
        t2 = abs(np.amin(eig))
        T_small=0.99*T_small/max(t1,t2)
        tmp2=np.random.randn(m_small,m_small)
        Q_small=tmp2@tmp2.T
        R_small=np.random.randn(n_small,m_small)
        
        n_large=9
        m_large=11
        T_large=np.random.randn(n_large,n_large)
        T_large=0.99*T_large/max(abs(la.eig(T_large)[0]))
        tmp2=np.random.randn(m_large,m_large)
        Q_large=tmp2@tmp2.T
        R_large=np.random.randn(n_large,m_large)
     
    
        Pstar1_small = lyapunov_solver(T_small,R_small,Q_small,method=1,options="fp")
        Pstar1_large = lyapunov_solver(T_large,R_large,Q_large,method=1,options="fp")
        t[0] = 1
    
        Pstar2_small = lyapunov_solver(T_small,R_small,Q_small,options="db")
        Pstar2_large = lyapunov_solver(T_large,R_large,Q_large,options="db")
        t[1] = 1
              
        # Standard
        Pstar3_small = lyapunov_solver(T_small,R_small,Q_small,options="standard")
        Pstar3_large = lyapunov_solver(T_large,R_large,Q_large,options="standard")
        t[2] = 1
    
        # Test the results.
        t[3] = np.max(abs(Pstar1_small-Pstar2_small)) < 1e-8
        t[4] = np.max(abs(Pstar2_small-Pstar3_small)) < 1e-8
        t[5] = np.max(abs(Pstar1_large-Pstar2_large)) < 1e-8
        t[6] = np.max(abs(Pstar2_large-Pstar3_large)) < 1e-8
        
        print(t)
        
        
def test2():
    """Test Lypunov equation solver."""
    T = np.array([
                  [0,0,-0.516185401208764],
                  [0,0.9,0],
                  [0,0,0.75]
                  ])
    R = np.array([
                 [0,0,0.707106781186548,0],
                 [0,-1,0,0],
                 [0,0,-1.02740233382816,0]
                 ])
    Q = np.eye(4)
    Pstable = lyapunov_solver(T,R,Q,N=0,n_shocks=R.shape[1],options="steady_state")#"equilibrium","steady_state","doubling_algorithm"
    print(Pstable)
    # Matlab results
   #  Pa0stable =
   #  1.1429         0   -1.6605
   #       0    5.2632         0
   # -1.6605         0    2.4127
    
    
    
if __name__ == "__main__":
    """Main entry point."""
    test2()
