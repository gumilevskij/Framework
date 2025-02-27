from __future__ import print_function
import numpy as np
from numpy import zeros_like, zeros
from numpy.linalg import solve
from numba import guvectorize
from numpy.linalg import solve as linalg_solve


def my_solve(m, sol):
    """
    Solve system of algebraic equations by elimination method.
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

    for y in range(h-1, 0-1, -1):   # Backsubstitute
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
    """Solve matrix equation."""
    if diagnose:
        sol = zeros_like(B)
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
        M = np.concatenate([A,B[:,:,None]],axis=2)
        sol = np.zeros_like(B)
        serial_solve_numba = guvectorize('void(f8[:,:], f8[:])', '(m,n)->(m)')(solve)
        serial_solve_numba(M,sol)

    return sol


def newton(f, x, verbose=False, tol=1e-6, maxit=5, jactype='serial'):
    """
    Solve nonlinear system using safeguarded Newton iterations

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
    """Finite difference of function."""
    def df(x):
        v0 = f(x)
        N = v0.shape[0]
        n_v = v0.shape[1]
        assert(x.shape[0] == N)
        n_x = x.shape[1]
        dv = zeros( (N, n_v, n_x) )

        for i in range(n_x):
            xi = x.copy()
            xi[:,i] += epsilon
            vi = f(xi)
            dv[:,:,i] = (vi - v0)/epsilon

        return [v0, dv]

    return df


def test_serial_solve():
    """Test serial solve metod."""
    N = 10
    A = np.random.random( (N,2,2) )
    B = np.random.random( (N,2) )

    print("A")
    print(A)
    print("B")
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


if __name__ == "__main__":
    test_serial_solve()

