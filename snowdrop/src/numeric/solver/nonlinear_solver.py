""" 
The non-linear solver. It is either the ABLR solver that uses stacked matrices algotithm or the
LBJ solver that solves equations with maximum of one lead and one lag variables.
""" 
import numpy as np
from time import time
import scipy.linalg as la
from scipy.optimize import root
#from scipy.optimize import fsolve
from snowdrop.src.model.model import Model
from snowdrop.src.model.settings import SolverMethod
from snowdrop.src.numeric.solver.util import getAllShocks,getParameters
from snowdrop.src.numeric.solver.LBJ import dense_system_solver as LBJ_dense_system_solver
from snowdrop.src.numeric.solver.LBJ import sparse_system_solver as LBJ_sparse_system_solver
from snowdrop.src.numeric.solver.ABLR import ABLRsolver
from snowdrop.src.numeric.solver.linear_solver import solve
from snowdrop.src.preprocessor.function import get_function_and_jacobian
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 


"Solution convergence tolerance"
TOLERANCE = 1.e-8
"Maximum number of iterations"
NITERATIONS = 100

NSTEPS = 1; itr = 0

def simulate(model,T,periods,y0,steady_state=None,params=None,cov=None,order=1,Npaths=1,MULT=1):
    """
    Find solution by iterations by applying Newton method.
    
    It iterates until either the change in subsequent iterations of this solution is less 
    than TOLEARANCE level or the number of iterations exceeds NITERATIONS.
    
    Parameters:
        :param model: Model object.
        :type model: `Model'.
        :param T: Time span.
        :type T: int.
        :param y0: Starting values or guessed values of endogenous variables.
        :type y0: numpy.array
        :param steady_state: Steady state solution.
        :type steady_state: numpy.ndarray
        :param periods: Array of endogenous variables.
        :type periods: numpy.ndarray.
        :param params: Values of parameters.
        :type params: numpy.ndarray.
        :param cov: Covariance matrix of shocks.
        :type cov: numpy.ndarray.
        :param order: Order of partial derivatives of Jacobian.
        :type order: int.
        :param Npaths: Number of simulation paths. This is the number of paths of stochastic shocks.
        :type Npaths: int.
        :param MULT: Multiplier defining terminal time.  If set greater than one than 
                     the solution will be computed for this extended time range interval.
        :type MULT: int.
        :returns: Numerical solution.
    """
    if bool(model.mapSwap) or bool(model.condShocks):  
        from snowdrop.src.numeric.solver.tunes import forecast_with_tunes
        
        shock_var = model.symbols['shocks']
        n_shocks = len(shock_var)
        all_shocks = getAllShocks(model,periods,n_shocks,Npaths,T)
        shocks = all_shocks[0]
        
        y,adjusted_shocks,model,count,err,elapsed = forecast_with_tunes(model=model,Nperiods=T,shocks=shocks) 
        
        yy = [y]; yyIter = [yy]
        
        model.calibration["adjusted_shocks"] = adjusted_shocks

    else:
        if model.order == 1:
            count,yy,yyIter,err,elapsed = first_order_solution(model=model,T=T,periods=periods,y0=y0,steady_state=steady_state,params=params,Npaths=Npaths,MULT=MULT)
        elif model.order == 2:
            if steady_state is None:
                steady_state,_ = find_steady_state(model)
            count,yy,yyIter,err,elapsed = second_order_solution(model=model,T=T,periods=periods,y0=y0,steady_state=steady_state,params=params,cov=cov,order=model.order,Npaths=Npaths,MULT=MULT)
        
    return (count,yy,yyIter,err,elapsed)
    
 
def second_order_derivatives(model,jacobian,hessian,g_y,g_u,cov=None):
    """
    Compute the second order reduced form solution of the DSGE model.
    
    It implements the algoritm described by Michel Juillard in
    "Computing first and second order approximation of DSGE models with DYNARE", CEPREMAP
    
    For references please see: 
    https://archives.dynare.org/events/paris-1005/first-second-order.pdf%3Fmonth:int=12&year:int=2018&orig_query=
    https://stephane-adjemian.fr/dynare/slides/dsge-perturbation-method.pdf
    
    .. note::
        THIS CODE CURRENTLY IS UNDER  DEVELOPMENT...
        It is assumed that model is linear to shock variables (there is no stochastic volatility).
        
    Parameters:
        :param model: The Model object.
        :type model: instance of class `Model'.
        :param jacobia: Matrix containing the Jacobian of the model.
        :type jacobian: numpy.ndarray.
        :param hessian: Hessian matrix (second order derivative with respect to endogenous and shock variables).
        :type hessian: numpy.ndarray.
        :param g_y: Transition matrix.
        :type g_y: numpy.ndarray.
        :param g_u: Matrix of shocks.
        :type g_u: numpy.ndarray.
        :param g_xx: Second derivative of of policy function.
        :type g_xx: numpy.ndarray.
        :param sigma: Matrix of error covariances.
        :type sigma: numpy.ndarray.
        
    """
    from snowdrop.src.numeric.solver.solvers import sylvester_solver
    
    var    = model.symbols["variables"]
    shocks = model.symbols["shocks"]
    n      = len(var)
    n_shk  = len(shocks)
    
    # First order derivatives
    F_yp = jacobian[:n,:n]
    F_yc = jacobian[:n,n:2*n]
    
    # Second order derivatives
    F_yc_yc = hessian[:n,n:2*n,n:2*n] 
    F_yp_yp = hessian[:n,:n,:n] 
    F_ym_ym = hessian[:n,2*n:3*n,2*n:3*n] 
    F_ym_u  = hessian[:n,2*n:3*n,3*n:,]
    F_yp_yc = hessian[:n,:n,n:2*n] 
    F_yp_ym = hessian[:n,:n,2*n:3*n] 
    F_yc_ym = hessian[:n,n:2*n,2*n:3*n]
    F_y_yp  =  hessian[:n,n:2*n,:n]
    F_u_u   = hessian[:n,3*n:,3*n:]
    F_ym_yc = hessian[:n,2*n:3*n,n:2*n]
    F_yc_u  = hessian[:n,n:2*n,3*n:]
    F_yp_u  = hessian[:n,:n,3*n:]
            
    # Compute matrix g_y_y
    A  = F_yp @ g_y + F_yc
    B  = F_yp
    C  = la.kron(g_y, g_y)
    D  = F_ym_ym 
    D += 3*F_yc_yc @ g_y @ g_y @ g_y @ g_y 
    D += 3*F_y_yp @ g_y @ g_y @ g_y 
    D += F_yc_yc @ g_y 
    D += F_yc_ym @ g_y
    D += F_yc_yc @ g_y @ g_y
    D += 3*F_yp_ym @ g_y @ g_y
    D += F_yp_yc @ g_y @ g_y @ g_y 
    
    # Solve Sylvester equation: A*x + B*x*C = D
    g_y_y = sylvester_solver(A=A,B=B,C=C,D=-D)

    # Compute matrix g_y_u
    rhs  = F_ym_u
    rhs += 3*F_yc_yc @ g_y @ g_u @ g_y @ g_u
    rhs += 3*F_yp_yc @ g_y @ g_u @ g_y
    rhs += F_yp_yc @ g_y @ g_u @ g_y
    rhs += F_yc_yc @ g_y @ g_u 
    rhs += 2*F_yp_ym @ g_y @ g_u 
    rhs += F_ym_yc @ g_u 
    rhs += F_yc_u @ g_y @ g_u
    rhs += F_yp_u @ g_y @ g_u
    rhs += F_ym_yc @ g_u
    g_y_u  = la.solve(A,-rhs)
    
    # Compute matrix g_u_u
    rhs  = F_u_u
    rhs += 3*F_yc_yc @ g_y @ g_u @ g_y @ g_u  
    rhs += 3*F_yp_yc @ g_y @ g_u @ g_u
    rhs += F_yp_yc @ g_y @ g_u @ g_y
    rhs += F_yc_yc @ g_u @ g_u 
    rhs += F_yc_u @ g_y @ g_u
    rhs += 3*F_yc_ym @ g_y @ g_u
    rhs += F_yc_yc @ g_u
    rhs += F_yc_u @ g_u
    g_u_u = la.solve(A,-rhs)
    
    # Compute matrix g_s_s
    if cov is None:
        g_s_s = None
    else:
        rhs = F_yp @ g_u_u
        rhs += np.reshape(np.reshape(F_yp_yp,(n,n*n)) @ la.kron(g_u, g_u),(n,n,n))
        E = F_yp @ (np.eye(len(g_y)) + g_y) + F_yc
        E = np.dot(E,cov)
        g_s_s = la.solve(E,-rhs) 
    
    return g_y_y,g_y_u,g_u_u,g_s_s


def second_order_solution(model,T,periods,y0,steady_state=None,params=None,cov=None,order=2,Npaths=1,MULT=1):
    """
    Find the second order approximation solution.
    
    This algorithm uses the Jacobian (first order derivatives) and the Hessian (second oredr derivatives).
    
    Parameters:
        :param model: Model object.
        :type model: `Model'.
        :param T: Time span.
        :type T: int.
        :param y0:  Starting values or guessed values of endogenous variables.
        :type y0: numpy.ndarray
        :param steady_state: Steady state solution.
        :type steady_state: numpy.ndarray.
        :param periods: Array of endogenous variables.
        :type periods: numpy.ndarray.
        :param params: Values of parameters.
        :type cov: Covariance matrix of stochastic shock.
        :type cov: numpy.ndarray.
        :param order: Order of partial derivatives of Jacobian.
        :type order: int.
        :param Npaths: Number of simulation paths. This is the number of paths of stochastic shocks.
        :type Npaths: int.
        :param MULT: Multiplier defining terminal time.  If set greater than one than 
                     the solution will be computed for this extended time range interval.
        :type MULT: int.
        :returns: Second order approximation to numerical solution.
        
    """
    count = 0; err = 1;  yIter = []
    t0 = time()
    var = model.symbols['variables']
    n = len(var)
    if params is None:
        params = model.calibration['parameters']
        
    shock_var = model.symbols['shocks']
    n_shocks = len(shock_var)
       
    all_shocks = getAllShocks(model,periods,n_shocks,Npaths,T)
    
    # Find deviation of solution from the steady state
    y = np.empty((T+1, n))
    y[:] = y0 - steady_state
 
    for path in range(Npaths):
        # Get first order approximation solution
        solve(model,p=params,steady_state=steady_state)
        yy = []
        shocks = np.array(all_shocks[path])
   
        while (err > TOLERANCE and count < NITERATIONS):
            count += 1
            y_prev = np.copy(y)
            
            # State transition matrix
            F = model.linear_model["A"]
            # Matrix of coefficients of shocks
            R = model.linear_model["R"]
            # Array of constants
            C = model.linear_model["C"]
            
            # Compute Jacobian and Hessian matrices.
            # We calculate system derivatives only one time at the steady state.
            fn,jacobian,hessian = get_function_and_jacobian(model=model,y=np.vstack((y,y,y)),params=params,order=order)
            
            # Get second order derivatives
            g_xx,g_xu,g_uu,g_ss = second_order_derivatives(model=model,jacobian=np.real(jacobian),hessian=np.real(hessian),g_y=np.real(F[:n,:n]),g_u=np.real(R[:n]),cov=cov)
    
            # Simulate,
            for t in range(T):
                # First order approximation
                u      = shocks[t]
                y[t+1] = F @ y[t] + C + R @ u
                v      = y[t,:n]
                # Correct solution with the second order approximation
                second_order_approx = np.zeros(n)
                for i in range(n):
                    for j in range(n):
                        for k in range(n):
                            second_order_approx[i] += g_xx[i,j,k] * v[j] * v[k]
                        for m in range(n_shocks):
                            second_order_approx[i] += g_xu[i,j,m] * v[j] * u[m] + g_uu[i,j,m] * u[m] * u[m]
                            
                    if not cov is None:
                        second_order_approx[i] += g_ss
                y[t+1,:n] += 0.5*second_order_approx
            
                err = la.norm(y_prev-y)/max(1.e-10,la.norm(y))
        
        # Add steady state to deviations to arrive to solution
        sol = y[:,:n] + steady_state
        err = max(err,la.norm(y[:,:n]))
        yy.append(sol)
        
    yIter.append(np.copy(yy))
    elapsed = time() - t0
    return (count,yy,yIter,err,elapsed)


def first_order_solution(model,T,periods,y0,steady_state=None,params=None,Npaths=1,MULT=1):
    """
    Find solution by iterations.  This is agent's perfect foresight forecast.
    
    This algorithm iterates until either the change in subsequent
    iterations of this solution is less than TOLEARANCE level
    or the number of iterations exceeds NITERATIONS.
    
    Parameters:
        :param model: Model object.
        :type model: `Model'.
        :param T: Time span.
        :type T: int.
        :param periods: Array of endogenous variables.
        :type periods: numpy.ndarray.
        :param y0: Starting values or guessed values of endogenous variables.
        :type y0: numpy.ndarray
        :param steady_state: Steady state solution.
        :type steady_state: numpy.ndarray
        :param params: Values of parameters.
        :type params: numpy.ndarray.
        :param Npaths: Number of simulation paths. This is the number of paths of stochastic shocks.
        :type Npaths: int.
        :param MULT: Multiplier defining terminal time.  If set greater than one than 
                     the solution will be computed for this extended time range interval.
        :type MULT: int.
        :returns: First order approximation numerical solution.
    """
    global TOLERANCE, NITERATIONS, NSTEPS
    global it

    t0 = time()
    T0 = T
    T  = int(T*MULT)
    #var_names = model.symbols['variables']
    var = model.calibration['variables']
    n = len(var)
    if params is None:
        params = model.calibration['parameters']
    shock_var = model.symbols['shocks']
    n_shocks = len(shock_var)
    if periods is None:
        n_periods = T 
        periods = np.arange(1,T+1)
    else:
        n_periods =  len(periods)
       
    all_shocks = getAllShocks(model,periods,n_shocks,Npaths,T)
    #print(all_shocks)
    
    # If y0 is 1d vector convert it to 2d array
    if np.ndim(y0) == 1:
        y_0 = np.empty(shape=(T+2,n))
        for i in range(T+2):
            y_0[i] = y0
        y0 = y_0
    # Append the last values of array to match this array first dimension
    elif len(y0) < T+2:
        y_0 = np.empty(shape=(T+2,n))
        y_0[:len(y0)] = y0
        y_0[len(y0):] = y0[-1]
        y0 = y_0
        
    # Assign terminal conditions
    terminal_values = model.terminal_values
    if not model.isLinear and not terminal_values is None:
        for i in range(n): 
            if var[i] in terminal_values.keys():
                y0[-1,i] = terminal_values[var[i]]
    
    
    yy = []; yyIter = []
    
    bHasAttrLBJSolver       = hasattr(LBJ_dense_system_solver,"py_func")
    bHasAttrLBJSparseSolver = hasattr(LBJ_sparse_system_solver,"py_func")
    bHasAttrABLSolver       = hasattr(ABLRsolver,"py_func")
    
    for path in range(Npaths):
        err = 1.0; count = 0
        y = np.copy(y0); y_prev = np.copy(y)
        yprev = np.copy(y)
        shocks = all_shocks[path]
        yIter = []
        # Iterate until solution converges
        if model.anticipate is None or model.anticipate :
            # We assume that all shocks are imposed
            while (err > TOLERANCE and count < NITERATIONS):
                count += 1
                for i in range(NSTEPS):
                    yp = np.copy(y)
                    if model.SOLVER.value == SolverMethod.ABLR.value:
                        if bHasAttrABLSolver:                            
                            y = ABLRsolver.py_func(model=model,T=T,n=n,y=y,params=params,shocks=shocks)                                                   
                        else:
                            y = ABLRsolver(model=model,T=T,n=n,y=y,params=params,shocks=shocks)
                            
                    elif model.SOLVER.value == SolverMethod.LBJ.value:
                        if bHasAttrLBJSolver:
                            if model.bSparse:
                                if bHasAttrLBJSparseSolver:
                                    y = LBJ_sparse_system_solver.py_func(model=model,T=T,n=n,y=y,params=params,shocks=shocks)
                                else:
                                   y = LBJ_sparse_system_solver(model=model,T=T,n=n,y=y,params=params,shocks=shocks) 
                            else:    
                                y = LBJ_dense_system_solver.py_func(model=model,T=T,n=n,y=y,params=params,shocks=shocks)
                        else:
                            if model.bSparse:
                                y = LBJ_sparse_system_solver(model=model,T=T,n=n,y=y,params=params,shocks=shocks)
                            else:  
                                y = LBJ_dense_system_solver(model=model,T=T,n=n,y=y,params=params,shocks=shocks)
                    
                    else:
                        if bHasAttrLBJSolver:
                            y = LBJ_dense_system_solver.py_func(model=model,T=T,n=n,y=y,params=params,shocks=shocks)
                        else:
                            y = LBJ_dense_system_solver(model=model,T=T,n=n,y=y,params=params,shocks=shocks)
                    y = yp + (y-yp)/NSTEPS  
                    
                yIter.append(np.copy(y[:2+T0]))
                err = la.norm(yprev-y)/max(1.e-10,la.norm(y))
                yprev = np.copy(y)
                model.y = y;
                
        else:
            # We assume that only one shock at a time is imposed
            shock_periods = [1+i for i,x in enumerate(shocks) if np.any(shocks[i])]
            n_periods = len(shock_periods)
            while (err > TOLERANCE and count < NITERATIONS):
                count += 1
                yv = []
                for i in range(1+n_periods):
                    shockValues = np.zeros(shocks.shape)
                    if len(shocks) > 0:
                        if i == 0:
                            period = 0
                            next_period = shock_periods[i]-1
                        elif i == n_periods:
                            period = shock_periods[i-1]
                            next_period = len(y)+1
                            shockValues[0] = shocks[period-1]
                        else:
                            period = shock_periods[i-1]
                            next_period = shock_periods[i]
                            shockValues[0] = shocks[period-1]
                    #print(period,next_period)
                    for i in range(NSTEPS):
                        yp = np.copy(y)
                        if model.SOLVER.value == SolverMethod.ABLR.value:
                            if bHasAttrABLSolver: 
                                y = ABLRsolver.py_func(model=model,T=T,n=n,y=y,params=params,shocks=shockValues)
                            else:
                                y = ABLRsolver(model=model,T=T,n=n,y=y,params=params,shocks=shockValues)
                                
                        elif model.SOLVER.value == SolverMethod.LBJ.value:
                            if bHasAttrLBJSolver:
                                if model.bSparse:
                                    y = LBJ_sparse_system_solver.py_func(model=model,T=T,n=n,y=y,params=params,shocks=shockValues)
                                else:
                                    y = LBJ_dense_system_solver.py_func(model=model,T=T,n=n,y=y,params=params,shocks=shockValues)
                            else:
                                if model.bSparse:
                                    y = LBJ_sparse_system_solver(model=model,T=T,n=n,y=y,params=params,shocks=shockValues)
                                else:
                                    y = LBJ_dense_system_solver(model=model,T=T,n=n,y=y,params=params,shocks=shockValues)
                                 
                        else:
                            y = LBJ_dense_system_solver(model=model,T=T,n=n,y=y,params=params,shocks=shockValues)
                            
                        y = yp + (y-yp)/NSTEPS 
                        
                    yv.append(np.copy(y[:next_period-period]))
                    y[:] = y[min(len(y)-1,next_period-period)]
                
                y = np.concatenate((yv),axis=0)
                err = la.norm(y_prev-y)/max(1.e-10,la.norm(y))
                y_prev = np.copy(y)
                yIter.append(np.copy(y[:2+T0]))
                model.y = y;
        
        # If equations are log-linearized, then compute original variables
        if model.LOG_LINEARIZE_EQS:
            variables = model.symbols["variables"]
            log_variables = model.symbols["log_variables"]
            ind = [i for i,v in enumerate(variables) if v in log_variables]
            for j in ind:
                y[:,j] = np.log(y[:,j])
            for i in range(len(yIter)):
                y = yIter[i]
                for j in ind:
                    y[:,j] = np.log(y[:,j])
                yIter[i] = y
            
        yy.append(np.copy(y[:2+T0]))
        yyIter.append(np.copy(yIter))
        
    elapsed = time() - t0
    return (count,yy,yyIter,err,elapsed)


def homotopy_solver(model,y0,par_name,par_steps,periods,T=101,tol=1.e-4,debug=False,MULT=1.0):
    """
    Solve model by adjusting parameters step-by-step.
    
    Solves this model iteratively since it could blow up for final parameters.  

    Parameters:
        :param model: Model object.
        :type model: `Model'.
        :param y0: Starting values of endogenous variables.
        :type y0: numpy ndarray.
        :param par_name: Parameter name.
        :type par_name: str.
        :param start: Starting value of parameter.
        :type start: float.
        :param stop: Final value of parameter.
        :type stop: float.
        :param par_steps: Steps values.
        :type par_steps: numpy ndarray.
        :param tol: Tolerance of solution convergence.
        :type tol: float, optional. The default is 1.e-4.
        :param periods: Periods.
        :type periods: list.
        :param T: Time span of simulations.
        :type T: int, optional. The default is 101.
        :param debug: Debug flag.
        :type debug: bool, optional. The default is False.
        :param MULT: Multiplier defining terminal time.  If set greater than 1 than 
                     solution will be computed for this extended time range interval.
        :type MULT: float, optional. The default is 1.
        :returns: The solution.

    """
    # from snowdrop.src.model.util import setCalibration
    
    global TOLERANCE 
    TOLERANCE = tol
    
    yy = None; prev_yy = None
    
    for par in par_steps:  
        model.setCalibration(par_name,par)
        # setCalibration(model,par_name,par)
        # Simulate using previous solution as an initial guess
        count,yy,yyIter,err,elapsed = simulate(model=model,T=T,periods=periods,y0=y0,MULT=MULT)
        if debug:
            print(f"Paramater {par_name} = {par:.2f}")
            print(f"Number of iterations: {count}; error: {err:.1e}")
            print(f"Elapsed time: {elapsed:.2f} (seconds)\n")
        if np.isnan(yy[-1]).any():
            yy = prev_yy
            return
            #break
        prev_yy = yy
        y0 = yy[-1]

    return yy,prev_yy,y0
   
    
def find_steady_state(model):
    """
    Find the steady state.
    
    Parameters:
        :param model: Model object.
        :type model: `Model'.
        :returns: Array of endogenous variables steady states and their growth.
    """
    from snowdrop.src.misc.termcolor import cprint
    from snowdrop.src.numeric.solver.util import getExogenousData
    
    #global TOLERANCE, NITERATIONS
    TOLERANCE = 1.e-6; NITERATIONS = 1000
    
    t0 = time()
    f_static = model.functions["f_steady"]
    f_jacob = model.functions["f_jacob"]
    bHasAttrStatic  = hasattr(f_static,"py_func")
    bHasAttrJacob  = hasattr(f_jacob,"py_func")
    
    # Define objective function
    def fobj(x):
        global itr
        itr += 1
        try:
            if bFlip:
                z[ind_rest] = x[ind_rest]
                p[model.indexOfFlippedExogVariables] = x[ind_flipped]
                y = np.concatenate([z,e])
            else:
                y = np.concatenate([x,e])
            if bHasAttrStatic:
                func = f_static.py_func(y,p,exog)
            else:
                func = f_static(y,p,exog)
        except ValueError:
            func = np.zeros(len(x)) + 1.e10 + itr
        return func
    
    # Define jacobian
    def fjacob(x):
        try:
            if bFlip:
                z[ind_rest] = x[ind_rest]
                p[model.indexOfFlippedExogVariables] = x[ind_flipped]
                y = np.concatenate([z,e])
            else:
                y = np.concatenate([x,e])
            if bHasAttrJacob:
                jacob = f_jacob.py_func(y,p,exog)
            else:
                jacob = f_jacob(y,p,exog)
            J = jacob[:nss,:nss]
        except ValueError:
            J = np.zeros((len(x),len(x))) + np.inf
        return J
	
    variables = model.symbols['variables']
    ss_variables = model.symbols['endogenous']
    param_names = model.symbols['parameters']
    index = [variables.index(x) for x in ss_variables]
    n = len(variables)
    nss = len(ss_variables)
    z = np.copy(model.calibration['variables'])
    p = getParameters(model=model).copy()
    n_shk = len(model.symbols['shocks'])
    e = model.calibration['shocks']
    #e = model.options["shock_values"]
    # Get exogenous time series
    exog = getExogenousData(model,t=-1)
    
    # Take last period shock for steady state solution.
    # Alternatively, you can take a first period shock if it is permanent.
    if np.ndim(e) == 2:
        e = e[-2]
    #print(e)
    growth = np.zeros(n)
    
    # Filter out added auxillary variables (those that represent variables with max/min lags greater than 1)
    bFlip = not model.flippedEndogVariables is None and not model.indexOfFlippedExogVariables is None
    z = z[index]
    if bFlip:
        # Indices of the flipped variables
        ind_flipped = [i for i,x in enumerate(ss_variables) if x in model.flippedEndogVariables]
        z[ind_flipped] = p[model.indexOfFlippedExogVariables]
        # Indices of the rest of variables
        ind_rest = [i for i in range(len(ss_variables)) if not i in ind_flipped]


    if True:
        # y = fsolve(func=fobj,x0=x,fprime=fjacob,xtol=TOLERANCE,maxfev=NITERATIONS)
        # err = la.norm(fobj(y)) 
        # Methods: 'hybr','lm','broyden1','broyden2','anderson','linearmixing','diagbroyden','excitingmixing','krylov','df-sane'
        if not model.indexOfFlippedEndogVariables is None and not model.indexOfFlippedExogVariables is None:
            sol = root(fobj,x0=z,method='broyden1',tol=TOLERANCE)
        else:
            try:
                sol = root(fobj,x0=z,jac=fjacob,method='lm',tol=TOLERANCE,options={"maxiter":NITERATIONS})    
            except:    
                sol = root(fobj,x0=z,jac=None,method='lm',tol=TOLERANCE,options={"maxiter":NITERATIONS})    
        y = sol.x
        success = sol.success
        err = la.norm(sol.fun)
        bConverged = err < TOLERANCE
        b = True
        if not success:
            ind_min = sol.fun.argmin()
            ind_max = sol.fun.argmax()
            ind = ind_min if abs(sol.fun[ind_min]) > abs(sol.fun[ind_max]) else ind_max
            cprint(f"numeric.solver.find_steady_state:\n   Root solver failed: Number of iterations {itr}, Error {err:.3e}","yellow")
            cprint(f"   The following equation has the largest residual of {sol.fun[ind]:.2e},","yellow")
            cprint("      "+model.symbolic.equations[ind],"yellow")
            print()
    else: #model.bSparse:
        y,err = get_steady_state(model,x0=z,ss_variables=ss_variables,p=p,e=e)
        b = success = False
        bConverged = err < TOLERANCE
    
    # Add auxillary variables
    m = dict(zip(ss_variables,y))
    steady_state = np.empty(n)
    for i,v in enumerate(variables):
        if "_plus_" in v:
            ind = v.index("_plus_")
            vv = v[:ind]
            steady_state[i] = m[vv]
        elif "_minus_" in v:
            ind = v.index("_minus_")
            vv = v[:ind]
            steady_state[i] = m[vv]
        else:
            steady_state[i] = m[v]
    #y0 = np.copy(steady_state)
    
    # Solution does not converge if model is unsteady...
    # Get steady state as an asymptotic approximation of numeric solution at large time.
    steady_model = model.options.get("steady_model",0)
    if not success and not steady_model:
        T0 = model.options.get("ss_interval",50)
        # cprint(f"Nonlinear_solver.find_steady_state:\n Root solver failed: Number of iterations {itr}, Error {err:.3e}","blue")
        # cprint(f"Finding steady state by running model forecast for horizon: {T0}","blue")
        params = model.calibration['parameters']
        T = int(1.2*T0)
        err = 1; count = 0
        yprev = np.empty(shape=(T+2,n))
        for t in range(T+2):
            yprev[t] = steady_state
            
        bHasAttr = hasattr(LBJ_dense_system_solver,"py_func")
        n_shocks = len(model.symbols['shocks'])
        n_shocks *= 1 + model.max_lead - model.min_lag
        shocks = np.zeros(shape=(T,n_shocks))
        
        while (err > TOLERANCE and count < NITERATIONS):
            count += 1
            if bHasAttr:
                y = LBJ_dense_system_solver.py_func(model=model,T=T,n=n,y=yprev,params=params,shocks=shocks)
            else:
                y = LBJ_dense_system_solver(model=model,T=T,n=n,y=yprev,params=params,shocks=shocks)
            err = la.norm(yprev-y)/max(1.e-10,la.norm(y))
            yprev = np.copy(y)
            
            
        growth = y[T0] - y[T0-1]
        ind = abs(growth)>1.e-3
        steady_state = y[T0]
        # if success: steady_state[~ind] = y0[~ind]
        steady_state[ind] = 0
        growth[~ind] = 0
        steady_state = np.round(steady_state,6)
        growth = np.round(growth,6)
        
    else:
        elapsed = time()-t0
        #cprint(f"Nonlinear_solver.find_steady_state:\n Elapsed time {elapsed:.2f} (sec.), Number of terations {itr}, Error {err:.3e}","blue")
            
    return steady_state, growth

            
def get_steady_state(model,x0,ss_variables=None,p=None,e=None):
    """
    Find the steady state solution.  Uses Euler iterative algorithm.
    
    Parameters:
        :param model: Model object.
        :type model: `Model'.
        :param model: Initial guess.
        :type x0: numpy.ndarray.
        :param nss: Number of model equations.
        :type nss: int.
        :param p: Parameters.
        :type p: numpy.ndarray.
        :param e: Shocks.
        :type e: numpy.ndarray.
        :returns: Array of endogenous variables steady states and this solution relative error.
    """
    #from time import time
    from scipy.sparse import csc_matrix
    #from misc.termcolor import cprint
    from snowdrop.src.numeric.solver.util import getExogenousData
    try:
        import pypardiso as sla
    except:
        from scipy.sparse import linalg as sla
        
    global itr,TOLERANCE,NITERATIONS
    
    err = 1; relax = 1.0
    #t0 = time()
    
    f_static = model.functions["f_steady"]
    f_jacob = model.functions["f_jacob"]
    bHasAttrStatic  = hasattr(f_static,"py_func")
    bHasAttrJacob  = hasattr(f_jacob,"py_func")
    
    x = np.copy(x0); x_prev = np.copy(x0); dx = 0
    if ss_variables is None:
        ss_variables = model.symbols['endogenous']
    nss = len(ss_variables)
    if p is None:
        p = getParameters(model=model)
    if e is None:
        #e = model.calibration['shocks']  
        n_shk = len(model.symbols['shocks'])
        e = [0]*(n_shk*(1+model.max_lead_shock-model.min_lag_shock))
    if np.ndim(e) == 2:
        e = list(e[-2])*(1+model.max_lead_shock-model.min_lag_shock)
    # Get exogenous data
    exog = getExogenousData(model)
    
    while itr < NITERATIONS and err > TOLERANCE and relax > 1.e-8:
        itr += 1
        # Get function values
        y = np.concatenate([x,e])
        if bHasAttrStatic:
            func = f_static.py_func(y,p,exog)
        else:
            func = f_static(y,p,exog)
        # Get Jacobian
        if bHasAttrJacob:
            jacob = f_jacob.py_func(y,p,exog)
        else:
            jacob = f_jacob(y,p,exog)
        J = jacob[:nss,:nss]
        # Use Euler algorithm to compute next iteration
        A = csc_matrix(J)
        dx = -sla.spsolve(A,func)
        # Update solution
        y = x + relax*dx
        if any(np.isnan(y)):
            relax *= 0.5
            # Update solution
            x = np.copy(x_prev)
        else:
            x_prev = np.copy(x)
            # Relative error
            err = la.norm(dx)/max(1.e-10,la.norm(x))
        x += relax*dx
        
        
    if any(np.isnan(x)):
        x = x_prev
        
    #elapsed = time()-t0
    #cprint(f"<--- Elapsed time {elapsed:.2f} (sec.), Iterations {itr}, Error {err:.4e}\n","blue")                    
    return x,err

def predict(model:Model,T:int,y:np.array,params:list=None,shocks:list=None,debug=False) -> np.array:
    """
    Solve backward looking nonlinear model equations.

    .. note::
        This code was developed to solve IMFE aplication.  It doesn't
        have lead variables nor shocks.
    
    Parameters:
        :param model: Model object.
        :type model: Model.
        :param T: Time span.
        :type T: int.
        :param n: Number of endogenous variables.
        :type n: int.
        :param y: Array of values of endogenous variables for current iteration.
        :type y: numpy.ndarray.
        :param params: Array of parameters.
        :type params: numpy.ndarray.
        :param shock: Array of shock values.
        :type shocks: numpy.ndarray.
        :returns: Numerical solution.
        
    
    """ 
    from snowdrop.src.numeric.solver.util import getExogenousData
    global itr

    t0 = time()
    n = y.shape[1]
    n_shk = len(model.symbols['shocks'])
    nd = np.ndim(params) 
    if debug:    
        from snowdrop.src.preprocessor.f_dynamic import f_dynamic as f
    else:
        f = model.functions["f_dynamic"]
    bHasAttr  = hasattr(f,"py_func")
    itr0 = itr
    
    # Define objective function
    def f_jac(x):
        global itr
        itr += 1
        try:
            z = np.concatenate([y[tp1],x,y[tm1],shk])
            if bHasAttr:
                func,jacob = f.py_func(z,par,exog)
            else:
                func,jacob = f(z,par,exog)
        except ValueError:
            func = np.zeros(len(x)) + 1.e10 + itr
        return func,jacob
    
    def fobj(x):
        func,_ = f_jac(x)
        return func
    
    def fjac(x):
        _,jacob = f_jac(x)
        return jacob[:,n:2*n]
    
    err = np.zeros(T)

    for t in range(-2,T+1):
        tp = max(0,t); tp1 = max(0,t+1); tm1 = max(0,t-1)
        shk = shocks[tm1] if not shocks is None and tp <= len(shocks) else np.zeros(n_shk)
        par = params if nd == 1 else params[:,min(t,params.shape[1]-1)]
        exog = getExogenousData(model,tp)
        # Find root
        try:
            sol = root(fobj,x0=y[tm1],jac=fjac,method='lm',tol=TOLERANCE,options={"maxiter":10*NITERATIONS})    
            y[tp] = sol.x
            fun = sol.fun
            fun[np.isnan(fun)] = 0
            err[t] = la.norm(fun)/la.norm(sol.x)
        except:
            pass
    
    # Impose continuity of slope or value.
    if len(y) > 1:
        y[-1] = y[-2]
    # if len(y) > 2:
    #     y[-1] = 2*y[-2]-y[-3]
        
    if debug:
        var = model.symbols["variables"]
        mv = dict(zip(var,y.T))
        
    err[np.isnan(err)] = 0
    elapsed = time()-t0
    print(f"Elapsed time: {elapsed:.2f} (sec.), total # of iterations: {1+itr-itr0}, Error: {np.sum(err)/T:.1e}")
    
    return y


if __name__ == '__main__':
    """ Formulas for second derivatives."""
    from sympy import symbols,Function,simplify,latex
    from IPython.display import display, Math
    
    x,u,s,up = symbols('x u s up')
    f = Function('f')
    g = Function('g')
    
    # Functional form:
    func = f(g(g(x,u,s),up,s),g(x,u,s),x,u,s)
    
    print('\nF_x_x:')
    h = func.diff(x).diff(x)
    h = simplify(h)
    fxx = latex(h)
    display(Math(fxx))
    
    """
    func = f(h,k,x,u,s); h = y+, k = y
    d²f/dx² = (f_hh * (g_x')² + f_h * g_xx' ) + (f_hk * g_x' * g_x * 2) + (f_kk * (g_x)² + f_k * g_xx)
    g' = g(g(x,u,s),up,s)
    g1 = g(x,u,s)
    f_h = df/dh
    f_k = df/dk
    f_x = df/dx
    g_x = dg/dx
    g_x' = dg'/dx
    g_xx = d²g/dx²
    g_xx' = d²g'/dx²
    f_hh = d²f/dh²
    f_hk = d²f/(dhdk)
    f_kk = d²f/dk
    """
                                                 
    print('\nF_x_u:')
    h = func.diff(x).diff(u)
    h = simplify(h)
    fxu = latex(h)
    display(Math(fxu))
            
    print('\nF_u_u:')
    h = func.diff(u).diff(u)
    h = simplify(h)
    fuu = latex(h)
    display(Math(fuu))
        
    print('\nF_s_s:')
    h = func.diff(s).diff(s)
    h = simplify(h)
    fss = latex(h)
    display(Math(fss))