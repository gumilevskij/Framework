"""Utility module functions."""

import numpy as np
from scipy import linalg as la
from scipy.optimize import root
from snowdrop.src.preprocessor.function import get_function
from snowdrop.src.preprocessor.function import get_function_and_jacobian
from numpy.polynomial.hermite import hermgauss as  hermgauss_numpy

TOL = 1e-10; itr = 0

count = 0
shockLeadsLags = None
paramLeadsLags = None


def is_pd(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = np.linalg.cholesky(B)
        return True
    except np.linalg.LinAlgError:
        return False
    
def nearest_pd(A):
    """Find the nearest positive-definite matrix to input A."""
    from scipy.linalg import eigh 

    B = (A + A.T) / 2
    s, V = eigh(B)

    H = np.dot(V, np.dot(np.diag(np.maximum(s, 0)), V.T))

    A2 = (B + H) / 2
    A3 = (A2 + A2.T) / 2

    if is_pd(A3):
        return A3

    spacing = np.spacing(np.linalg.norm(A))
    I = np.eye(A.shape[0])
    k = 1
    while not is_pd(A3):
        mineig = np.min(np.real(np.linalg.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3

def orthogonalize_shocks(model):
    """
    Uses Cholesky decomposition to orthogonalize shocks.
    
    Parameters:
        :param model: Model object
        :type model: `Model'.
    """
    shocks_names = model.symbols["shocks"]
    n_shk = len(shocks_names)
    shocks_values = model.calibration['shocks']
    if 'shock_values' in model.options:
        shocks_values = model.options['shock_values'] 

    # Matrix of coefficients of shocks
    R = model.linear_model["R"]
    
    # Get shocks covariance matrix
    cal = {k:v for k,v in model.symbolic.calibration_dict.items() if k.startswith("std_")}
    
    if len(cal) > 0:
        Q = np.zeros((n_shk,n_shk))  # covariance matrices of shocks
        for i,v in enumerate(shocks_names):
            std_v = "std_"+v
            if std_v in cal:
                Q[i,i] = cal[std_v]**2
        Q = R.T @ Q @ R
    else:
        Q = R.T @ R
        
    # Cholesky decomposition
    L = la.cholesky(Q,lower=True)
    # try:
    #     L = la.cholesky(Q,lower=True)
    # except np.linalg.LinAlgError:
    #     L = la.cholesky(nearest_pd(Q),lower=True)
        
    orthogonal_shocks = shocks_values @ L
    
    model.options['shock_values'] = orthogonal_shocks
    model.calibration['shocks'] = orthogonal_shocks
    
def getCovarianceMatrix(Q,H,calib,shocks,meas_shocks):
    """
    Build covariance matrices of errors of endogenous and measurement variables.
    
    It is assumed that standard deviations of variables are prefixed with 'std_RES_*' or 'std_SHK_*' name.
    
    Parameters:
        :param Q: Covariance matrix of shocks errors.
        :type Q: numpy array.
        :param calib: Map of estimated shock names and values of standard deviations.
        :type calib: dict.
        :param shocks: List of shock variables.
        :type shocks: list.
        :param meas_shocks: List of shock of measurement variables.
        :type meas_shocks: list.
        :returns: Diagonal covariance matrices of endogenous variables.
    
    """
    for i,v in enumerate(shocks):
        std_v = "std_"+v
        if std_v in calib:
            Q[i,i] = calib[std_v]**2
            
    for i,v in enumerate(meas_shocks):
        std_v = "std_"+v
        if std_v in calib:
            H[i,i] = calib[std_v]**2
            
    return Q,H
            
def getCovarianceMatricies(measurement_variables,measurement_shocks,measurement_equations,shocks,model):
    """
    Build covariance matrices of errors of endogenous and measurement variables.
    
    It is assumed that standard deviations of variables are prefixed with 'std_RES_*' or 'std_SHK_*' name.
    
    Parameters:
        :param measurement_variables: List of measurement variables.
        :type measurement_variables: list.
        :param measurement_shocks: Measurement shock variables.
        :type measurement_shocks: list.
        :param shocks: List of shock variables.
        :type shocks: list.
        :param model: Model object.
        :type model: Model.
        :returns: Covariance matrices of endogenous `Q' and measurement `H' variables.
                  These covariance matrices are diagonal.
    
    """
    cal = {k:v for k,v in model.symbolic.calibration_dict.items() if k.startswith("std_")}
    mp = {**model.options,**cal}
    n = len(shocks)
    Q = np.zeros((n,n))  # covariance matrices of endogenous variables shocks
    for i,v in enumerate(shocks):
        std_v = "std_"+v
        if std_v in mp:
            Q[i,i] = mp[std_v]**2
       
    if not measurement_shocks is None:
        import re
        delimiters = " ", ",", ";", "*", "/", ":", "+", "-", "^", "{", "}", "(", ")", "="
        regexPattern = '|'.join(map(re.escape, delimiters))
    
        n = len(measurement_variables)        
        H = np.zeros((n,n)) # covariance matrices of measurement variables shocks
       
        for shk in measurement_shocks:
            std_v = "std_"+shk
            if std_v in mp:
                for i,eq in enumerate(measurement_equations):
                    if shk in eq:
                        arr = re.split(regexPattern,eq)
                        arr = list(filter(None,arr))
                        for v in arr:
                            if v in measurement_variables:
                                ind = measurement_variables.index(v)
                                H[ind,ind] = mp[std_v]**2
    else:
        n = len(measurement_variables)        
        variables = list(mp.keys())
        options_vars = [x.lower() for x in variables]
        H = np.zeros((n,n))
        
        for i,v in enumerate(measurement_variables):
            for prefix in ["std_","std_res_","std_shk_"]:
                std_v = prefix+v.lower()
                if std_v in options_vars:
                    ind = options_vars.index(std_v)
                    std_v = variables[ind]
                    H[i,i] = mp[std_v]**2
            
    return (Q,H)

def getAllShocks(model,periods,n_shocks,Npaths,T):
    """
    Return list of shocks for all paths.

    Args:
        model : Model.
            Model object.
        periods : 
            Simulation periods.
        n_shocks : int
            Number of shocks.
        Npaths : int
            Number of paths.
        T : int
            Time range.

    Returns:
        all_shocks : List
            List of shocks for all paths.
    """
    max_lead_shock = model.max_lead_shock
    min_lag_shock = model.min_lag_shock
    bStochastic = 'distribution' in model.options
    if 'shock_values' in model.options:
        shock_values = model.options.get('shock_values')
        if isinstance(shock_values,list) and len(shock_values)==1:
            if isinstance(shock_values[0],list):
               shock_values = shock_values[0] 
    elif "shocks" in model.calibration:
        shock_values = model.calibration["shocks"]
    else:
        shock_values = np.zeros(n_shocks)
    shock_values = np.array(shock_values)
        
    bPermanent= not "periods" in model.options
    all_shocks=[]
    for path in range(Npaths):
        if bStochastic:
            process = model.options['distribution']
            shocks = np.squeeze(process.simulate(n_shocks,T+2))
        else:
            shocks = []
            i = 0
            for t in range(T+2):
                shock,i = getShocks(i=i,t=t,periods=periods,shock_values=shock_values,n_shk=n_shocks,bStochastic=bStochastic,permanent_shocks=bPermanent)
                shocks.append(np.copy(shock))
            shocks = np.array(shocks)
            if max_lead_shock > min_lag_shock:
                shocks = getCombinedShocks(shocks,n_shocks,T+2,min_lag_shock,max_lead_shock)
  
        all_shocks.append(np.copy(shocks))
        
    return all_shocks
        
def getShocks(i=0,t=0,periods=None,shock_values=None,model=None,n_shk=0,bStochastic=False,permanent_shocks=False):
    """
    Return shock values.
    It combines lead, lag and current values of shocks.

    Parameters:
        :param i: Current period.
        :type i: int.
        :param t: Time.
        :type t: int.
        :param periods: list of periods at which shocks are applied.
        :type periods: List.
        :param shock_values: Array of shock values.
        :type shock_values: List.
        :param n_shk: Number of shocks
        :type n_shk: int.
        :param bStochastic: True means that shocks are stochastic and are described by an exogenous process. 
        :type bStochastic: bool.
        :returns: Shocks values.
    """
    if shock_values is None:
        shock_values = model.calibration['shocks']
        
    nd = np.ndim(shock_values)
                
    if periods is None:
        if nd == 2:
            shock = shock_values[t] if t < len(shock_values) else np.zeros(n_shk)
        elif nd == 1:
            shock = shock_values
        return shock,i
            
           
    if nd == 2:
        dim1,dim2 = shock_values.shape
        size = dim2
    else:
        if np.isscalar(shock_values):
            shock_values = [shock_values]
            size = 1
            nd = 1
        else:
            size = shock_values.size
        if size < n_shk:
            temp = np.zeros(n_shk)
            temp[:size] = shock_values
            shock_values = temp
        
    if bStochastic:
        shock = [shock_values[t]]
    elif permanent_shocks:
        if nd > 1 and len(shock_values) > 0:            
            shock = shock_values[0]
        else:
            shock = shock_values
    elif nd == 2:  
        shock = np.zeros(size)
        if i<len(periods) and t==periods[i]-1:   
            shock = shock_values[i] if i < len(shock_values) else np.zeros(n_shk)
            i += 1
    elif nd == 1:  
        shock  = [0] if n_shk==1 else np.zeros(n_shk)
        if i<len(periods) and t==periods[i]-1:
            shock = [shock_values[i]] if n_shk==1 else shock_values                   
            i += 1
        
    #print(i,t,nd,PERMANENT_SHOCK,len(shock_values),shock)
    return shock,i

def getCombinedShocks(shocks,n_shocks,T,min_lag_shock,max_lead_shock):
    """
    Combine lagged and lead shocks.
        
    Parameters:
        :param shocks: Array of shocks.
        :type shocks: np.array.
        :param n_shocks: Number of shock variables.
        :type n_shocks: int.
        :param T: Maximum time.
        :type t: int.
        :param min_lag_shock: Minimum lag of shocks
        :type min_lag_shock: int
        :param max_lead_shock: Maximum lead of shocks
        :type max_lead_shock: int
        
        :returns: Arrays of shocks arranged from laged shocks to lead shocks.

    """
    nd = np.ndim(shocks)   
    shks = list()
          
    for t in range(T):
        shk = np.zeros((1+max_lead_shock-min_lag_shock,n_shocks)) 
        for k in range(n_shocks):
            leads_lags = shockLeadsLags[k]
            for ll in leads_lags:
                j = ll - min_lag_shock
                if t+ll >= 0 and t+ll < T:
                    if nd == 1:
                        shk[j,k] = shocks[t+ll]
                    else:
                        shk[j,k] = shocks[t+ll,k]
        
        shk = shk.reshape((1+max_lead_shock-min_lag_shock)*n_shocks)
        shks.append(shk)                 
                    
    combined_shocks = np.squeeze(np.array(shks))
    if np.ndim(combined_shocks)==0:
        combined_shocks.shape = (1)
        
    return combined_shocks

def getExogenousData(model,t=0):
    """
    Return combined lag and lead time series.

    Args:
        model : Model.
            Model object.
        t : Time period.
            int.
       
    Returns:
        exog : List
            Exogenous data at time t.
    """
    from snowdrop.src.utils.util import getExogenousSeries
    
    max_lead_exog = model.max_lead_exog
    min_lag_exog = model.min_lag_exog
    exogenous = model.symbols.get("exogenous",[])
    exog_data = model.calibration.get("exogenous",{})
    if not bool(exog_data):
        exog_data = getExogenousSeries(model)
    
    exog = []
    if len(exogenous) > 0:
        for i in range(min_lag_exog,1+max_lead_exog):
            tt = t + i
            for exo in exogenous:
                if exo in exog_data:
                    ser = exog_data[exo]
                    j = max(0,min(len(ser)-1,tt))
                    val = ser[j]
                else:
                    val = 0
                exog.append(val)
            
    return exog
        
def getStationaryVariables(model,T=None,K=None,debug=False):
    """
    Return stationary and non-stationary endogenous variables.
    
    Parameters:
        :param model: Model object.
        :type model: Model.
        :param T: Transition matrix.
        :type T: 2-D ndarray.
        :param K: Vector of constants.
        :type K: 1-D ndarray.
        
    """  
    from snowdrop.src.utils.equations import getTopology
    mp = getTopology(model) 
    
    var = model.symbols["variables"]
    n = len(var)
    
    if T is None or K is None:
        # State transition matrix
        T = model.linear_model["A"][:n,:n]
        # Array of constants
        K = model.linear_model["C"][:n]
        
    eig,w  = la.eig(T)

   # Get non-stationary indices
    unstable = [i for i in len(eig) if np.abs(eig[i]) > 1+1.e-7]
    non_stationary = np.unique(np.sum(np.abs(w[:,unstable]),2) > 1.e-7)
    
    stationary = [i for i in range(n) if i not in non_stationary]
    var_non_stationary = [var[i] for i in non_stationary]
    var_stationary = [var[i] for i in stationary]
    n_stationary = len(stationary)
    n_non_stationary = len(non_stationary)
    
    stationaryRows = []
    for k in var_stationary:
        v = mp[k]
        if not "+" in " ".join(v):
            for e in v:
                ind = e.index(':')
                lag = int(e[:ind])
                if lag < 0:
                    e = e[1+ind:]
                    ind = e.index(',')
                    row = int(e[:ind])
                    stationaryRows.append(row)
    
    stationaryColumns = stationary
    stationaryRows = np.unique(stationaryRows)
    nonStationaryColumns = non_stationary
    nonStationaryRows = [i for i in range(n) if not i in stationaryRows]  
        
    if debug:
        print()
        print("Number of stationary variables: {0}".format(n_stationary))
        print("Stationary variables: ",sorted(var_stationary))
        print()
        print("Number of non-stationary variables: {0}".format(n_non_stationary))
        print("Non-stationary variables: ",sorted(var_non_stationary))
        print()
    
    return var_stationary,var_non_stationary,stationaryColumns,stationaryRows,nonStationaryColumns,nonStationaryRows
    
def getStableUnstableVariables(model,T=None,K=None,debug=False):
    """
    Get indices of stable and unstable endogenous variables.
    
    Args:
        model : Model
            Instance of model object.
        T: 2-D ndarray.
            State transition matrix.
        K: 1-D ndarray
            Vector of constants.
        debug : bool, optional
            If True prints names of stable and unstable variables..

    Returns:
        Indices of stable and unstable endogenous variables.
    """
    var = model.symbols["variables"]
    n = len(var)
    
    if T is None or K is None:
        if not "A" in model.linear_model:
            from snowdrop.src.numeric.solver.linear_solver import solve
            isLinear = model.isLinear
            model.isLinear = True
            solve(model)
            model.isLinear = isLinear
        # State transition matrix
        T = model.linear_model["A"][:n,:n]
        # Array of constants
        K = model.linear_model["C"][:n]
        
    eig,w  = la.eig(T)
    #TODO order the un-ordered eigen decomposition.
    #Q = la.pinv(T-np.eye(n)) @ K @ la.inv(w)
    
    eig_stable = [i for i in range(n) if abs(eig[i]) <= 1-1.e-8]
    eig_unstable = [i for i in range(n) if i not in eig_stable]
    rowUnstable = colUnstable = [i for i in range(n) if sum(abs(w[i,eig_unstable])) > 1e-8]
    rowStable = colStable  = [i for i in range(n) if not i in colUnstable]
    n_stable     = len(colStable)
    n_unstable   = len(colUnstable)
    var_stable   = [var[i] for i in colStable]
    var_unstable = [var[i] for i in colUnstable]
      
    if debug:
        print()
        print("Number of stable variables: {0}".format(n_stable))
        print("Stable variables: ",sorted(var_stable))
        print()
        print("Number of unstable variables: {0}".format(n_unstable))
        print("Unstable variables: ",sorted(var_unstable))
        print()
        
    return var_stable,var_unstable,colStable,rowStable,colUnstable,rowUnstable
 
def getForwardLookingVariables(model,T=None,K=None,debug=False):
    """
    Get indices of forward/backward looking endogenous variables.
    
    Args:
        model : Model
            Instance of model object.
        T: 2-D ndarray.
            State transition matrix.
        K: 1-D ndarray
            Vector of constants.
        debug : bool, optional
            If True prints names of backward and forward variables.

    Returns:
        List of backward and forward variables and their column and row numbers.
    """
    from snowdrop.src.utils.equations import getTopology
    
    var = model.symbols["variables"]
    n = len(var)
      
    rowForward = list(); colForward = list(); Map = dict()
    mp = getTopology(model)

    for k in mp:
        v = mp[k]
        for e in v:
            ind = e.index(':')
            lead_lag = int(e[:ind])
            e = e[1+ind:]
            ind = e.index(',')
            row = int(e[:ind])
            col = int(e[1+ind:]) % n
            if lead_lag > 0 :
               if not col in Map:
                   Map[col] = []
               rowForward.append(row)
               colForward.append(col)
               rows = Map[col]
               if row in rows:
                   for r in rows:
                       if not r == row:
                           del rowForward[r]
                       rows += [row]
                    
    colForward = np.unique(colForward)
    rowForward = np.unique(rowForward)
    
    colBackward  = [i for i in range(n) if not i in colForward]  
    rowBackward  = [i for i in range(n) if not i in rowForward] 
    n_backward   = len(colBackward)
    n_forward = n - n_backward
    var_backward  = [var[i] for i in colBackward]
    var_forward = [var[i] for i in colForward]
    
    if debug:
        print()
        print("Number of forward looking variables: {0}".format(n_forward))
        print("Forward looking variables: ",sorted(var_forward))
        print()
        print("Number of static and backward looking variables: {0}".format(n_backward))
        print("Static and backward looking  variables: ",sorted(var_backward))
        print()
        
    return var_backward,var_forward,colBackward,rowBackward,colForward,rowForward

def getStableUnstableRowsColumns(model,T=None,K=None,debug=True):
    """
    Return row and column numbers of stable and unstable endogenous variables.
    
    Parameters:
       :param model: Model object.
       :type model: Model.
       :param T: State transition matrix.
       :type T: 2-D ndarray.
       :param K: Vector of constants.
       :type K: 1-D ndarray.
       :returns: Array of starting values
    """  
    #from snowdrop.src.utils.equations import getStableUnstableVariables
    
    global count
    count += 1
    debug &= count==1
    
    var = model.symbols["variables"]
    n = len(var)
        
    # Get stable variables
    var_stable,var_unstable,colStable,rowStable,colUnstable,rowUnstable = \
        getStableUnstableVariables(model,T,K,debug=False)  
    model.stable = var_stable
    model.unstable = var_unstable
    
    n_cols_stable = len(colStable)
    n_rows_stable = len(rowStable)
    n_stable = min(n_cols_stable,n_rows_stable)
    #n_unstable = n - n_stable
    
    if n_rows_stable > n_stable:
        rowStable = rowStable[:n_stable]
    elif n_cols_stable > n_stable:
        colStable = colStable[:n_stable]
        
    colUnstable = [i for i in range(n) if not i in colStable]
    rowUnstable = [i for i in range(n) if not i in rowStable]
    n_cols_stable = len(colStable)
    
    varstable = [x for x in var_stable if not "_minus_" in x and not "_plus_" in x]
    varunstable = [x for x in var_unstable if not "_minus_" in x and not "_plus_" in x]
        
    if count == -1 and debug:
        print()
        print("Number of stable variables: {0}".format(len(varstable)))
        print("Stable variables: ",sorted(varstable))
        print()
        print("Number of unstable variables: {0}".format(len(varunstable)))
        print("Unstable variables: ",sorted(varunstable))
        print()
        
    return rowStable,colStable,rowUnstable,colUnstable 

def find_starting_lag_values(model):
    """
    Find starting values of lead/lag variables.
     
    These are starting values of variables with lead or lag greater than one.
     
    Parameters:
        :param model: Model object.
        :type model: Model.
        :returns: Array of starting values.
    """
    TOLERANCE = 1.e-8
    NITERATIONS = 100
    
    # Define objective function
    def fobj(x):
        y = np.copy(v)
        for i,j in enumerate(indices):
            y[j] = x[i]
        z = np.vstack((y,y,y))
        func = get_function_and_jacobian(model,params=p,y=z,order=0)
        return func
	
    n_shk = len(model.symbols['shocks'])
    v = model.calibration['variables']
    p = getParameters(model=model)
    e = getShocks(model=model,n_shk=n_shk)[0]
    v = np.array(v)
    n, = v.shape   
    if e.ndim == 2:
        e = e[0]
    
    s = model.symbols['variables']
    indices = [i for i, x in enumerate(s) if "_minus_" in x or "__m_" in x]
    x = [v[i] for i in indices]
                                
    if len(indices) > 0:
        sol = root(fobj,x,method='lm',tol=TOLERANCE,options={"maxiter":NITERATIONS})
        historic_values = sol.x
    else:
        historic_values = v
       
    return historic_values,indices
    
def find_eigen_values(model,steady_state=None):
    """
    Find eigen values of Jacobian.
     
    Parameters:
        :param model: Model object.
        :type model: Model.
        :param steady_state: Endogenous variables steady states.
        :type steady_state: list.
        :returns: Eigen values.
    """
    var = model.symbols['variables']
    n = len(var)
    eigen_values = []
    p = getParameters(model=model)
    if steady_state is None:
        steady_state = np.ones(n)
    
    z = np.vstack((steady_state,steady_state,steady_state))
    func,jacob =  get_function_and_jacobian(model,params=p,y=z,order=1)
    F = jacob[:,0:n]
    C = jacob[:,n:2*n]
    L = jacob[:,2*n:3*n]
    
    # Build A,B matrices: A*[dt(t+1), dy(t)] = B*[dt(t), dy(t-1)] 
    A = np.zeros((2*n,2*n))
    A[:n,:n] = F
    A[n:2*n,n:2*n] = np.identity(n)
    B = np.zeros((2*n,2*n))
    B[:n,:n] = -C
    B[:n,n:2*n] = -L
    B[n:2*n,:n] = np.identity(n)
    
    # Get QZ matrix decomposition (or generalized Schur decomposition)
    AA, BB, Q, Z = la.qz(A,B,output='complex')
    #Compute eigen values as ratio of diagonal elements of BD and AD
    BD = np.diagonal(BB)
    AD = np.diagonal(AA)
    
    ## Compute eigen values as ratio of beta and alpha
    eigen_values = np.array([BD[i]/AD[i] for i in range(2*n) if not AD[i]==0])
    
    tol = 1.e-10
    model.nUnit = sum(abs(abs(eigen_values)-1)<tol)
    model.nStable = sum(abs(eigen_values)<=1-tol)
    model.nUnstable = sum(abs(eigen_values)>=1+tol)
    return eigen_values
	
def checkSteadyState(model,endog):
    """
    Find values of equations residuals for a given steady state.
     
    Parameters:
        :param model: Model object.
        :type model: Model.
        :param endog: List of endogenous variables values.
        :type endog: list.
        :returns: Array of equations residuals.
    """  
    n_shk = len(model.symbols['shocks'])
    z = model.calibration['variables']
    p = getParameters(model=model)
    e = getShocks(model=model,n_shk=n_shk)[0]
        
    n = len(endog)
    x = np.copy(z)
    for i in range(n):
        v = endog[i]
        if v in model.steady_state:
            x[i] = model.steady_state[v]
        else:
            print(f"checkSteadyState: {v} is not present in model steady-state map!")
            
    errors = model.functions["f_static"](x,p,0*e)
    
    return errors
	
def checkSolution(model,periods,y):
    """
    Return errors of equations given solution.
     
    Parameters:
        :param model: Model object.
        :type model: Model.
        :param periods: Array of periods.
        :type periods: numpy.array.
        :param y: Array of endogenous variables values.
        :type y: np.array.
        :returns: Array of equations residuals.
    """
    shock_var = model.symbols['shocks']
    n_shocks = len(shock_var)
    if 'shock_values' in model.options:
        shock_values = model.options.get('shock_values')
    else:
        shock_values = np.zeros(n_shocks)
    shock_values = np.array(shock_values) 
        
    params = getParameters(model=model)
    
    T = len(y)
    n_shk = len(model.symbols['shocks'])
    residuals = []; shocks = []
    i = 0
    
    for t in range(T-2):
        shock,i = getShocks(i=i,t=t,periods=periods,shock_values=shock_values,n_shk=n_shk)
        shocks.append(shock)
        z = np.vstack((y[t],y[t+1],y[t+2]))
        res =  get_function_and_jacobian(model,params=params,y=z,shock=shock,t=t,order=0)
        residuals.append(res)
           
    return np.array(residuals)
      
def find_residuals(model,y):
    """
    Find values of shocks that minimize errors of equations.
     
    Parameters:
        :param model: Model object.
        :type model: Model.
        :param y: Array of endogenous varuables values.
        :type y: np.array.
        :returns: Array of endogenous variables residuals.
    """
    import re
    

    # Define objective function
    def fobj(x):
        z = [ytm,yt,ytp]
        func =  get_function_and_jacobian(model,params=params,y=z,shock=x,order=0)
        return la.norm(func[ind])
	
    shock_names = model.symbols['shocks']
    n_shk = len(shock_names)
    params = getParameters(model=model)
    eqs = model.symbolic.equations
    n = len(eqs)
    
    delimiters = " ", ",", ";", "*", "/", ":", "+", "-", "^", "{", "}", "(", ")", "="
    regexPattern = '|'.join(map(re.escape, delimiters))
    ind = list(); lShocks = list()
    # Skip measurement equations
    for i in range(n):
        eq = eqs[i]
        arr = re.split(regexPattern,eq)
        arr = set(filter(None,arr))
        for v in arr:
            if v in shock_names:
                ind.append(i)
                k = shock_names.index(v)
                lShocks.append(k)
    ind = np.unique(ind)
    lShocks = np.unique(lShocks)
    n_ind = len(ind)
        
    T,n = y.shape
    residuals = []
    for t in range(T):
        tm = max(0,t-1)
        tp = min(t+1,T-1)
        ytm,yt,ytp = y[tm],y[t],y[tp]
        e = np.zeros(n_shk)
        f,jacob = get_function_and_jacobian(model,params=params,y=[ytm,yt,ytp])
        R = jacob[ind,3*n:]
        # Skip measurement shocks
        rhs = f[ind]
        try:
            if n_shk == n_ind:
                # Finding residuals using the direct method
                e[lShocks] = -la.solve(R, rhs)
            else:    
                # Finding residuals using minimization method
                e[lShocks] = -la.pinv(R) @ rhs
        except:
            e[lShocks] = np.zeros(len(lShocks))
            
        residuals.append(e)
           
    return np.array(residuals)

def decompose(model,y,endog,n,T,periods,isKF):
    """
    Decomposes the right-hand-side of equations into components that are caused by contributions of endogenous variables one at a time.
     
    Parameters:
        :param model: Model object.
        :type model: Model.
        :param y: Values of endogenous variables.
        :type y: numpy.array
        :param endog: Valiables names.
        :type endog: List.
        :param n: Endogenous variable name.
        :type n: str.
        :param T: Time span.
        :type T: int.
        :param periods: Vector of periods that define shocks timing.
        :type periods: numpy.array.
        :param isKF: True if shocks are obtained by Kalman Filter.
        :type isKF: bool.
        :returns: Array of decompositions.
    """
    from re import split,escape
    import warnings, sys
    if not sys.warnoptions:
        warnings.simplefilter("ignore")
    
    
    data = {}; var = []
    var_labels = model.symbols.get("variables_labels",{})
     
    delimiters = "+", "-", "*", "/", "**", "^", "=", " ", "(", ")" 
    regexPattern = '|'.join(map(escape, delimiters))
        
    shock_names = model.symbols['shocks']
    n_shocks = len(shock_names)
    exog_names = model.symbols.get('exogenous',[])
    ne = len(exog_names)
    if 'adjusted_shocks' in model.calibration:
        shock_values = model.calibration.get('adjusted_shocks')
    elif 'shocks' in model.calibration:
        shock_values = np.squeeze(getAllShocks(model,periods=periods,n_shocks=n_shocks,Npaths=1,T=T))
    else:
        shock_values = [0]*n_shocks
    shock_values = np.array(shock_values)
   
    
    # Remove periods if it is not set in model
    if not "periods" in model.options:
        periods = None
        
    nd_shock = 1 + model.max_lead_shock - model.min_lag_shock 
        
    eqs  = model.symbolic.equations
    func = model.functions["f_rhs"]
    ne = 0
       
    for i,eq in enumerate(eqs):
        # Exclude exogenous variablesx
        
        arr = eq.split("=")
        if len(arr) == 2:
            eq1 = arr[0]
            eq2 = arr[1]
        else:
            ind1 = eq.rfind("(")
            ind2 = eq[ind1:].find(")") + ind1
            eq1 = eq[1+ind1:ind2]
            eq2 = eq[:ind1]
            
        eq1 = eq1.replace(" ","")
        eq2 = eq2.replace(" ","")
        leftEqArr = split(regexPattern,eq1)
        leftEqArr = list(filter(None,leftEqArr))
        rightEqArr = split(regexPattern,eq2)
        rightEqArr = list(filter(None,rightEqArr))
        variables = [x.strip() for x in rightEqArr if x in endog and bool(x.strip())]
        variables2 = []
        for v in rightEqArr:
            ind1 = eq2.find(v)
            ind = ind1+len(v)
            x = eq2[ind1:ind]
            if x.isdigit():
                continue
            if x in variables:
                if ind < len(eq2) and eq2[ind] == "(":
                    ind2 = eq2.find(")",ind)
                    variables2.append(eq2[ind1:1+ind2])
                    eq2 = eq2[1+ind2:]
                else:
                    variables2.append(v)
                    eq2 = eq2[1+ind:]
            else:
                eq2 = eq2[1+ind:]
                
        var.extend(variables2)
                
        if n in leftEqArr:
            ne += 1
            if ne > 1: # Ensure that we treat only one equation at a time
                continue
            
            model.calibration["exogenous"] = {}
            ### Endogenous variables
            for v,v2 in zip(variables,variables2):
                z = np.copy(y)
                lst = [j for j,x in enumerate(endog) if v==x]
                for j in range(len(endog)):
                    if not j in lst:
                        z[:,j] = 1.e-10
                vv = []
                for t in range(T):
                    ym = np.copy(z[max(0,t-1)])
                    yc = np.copy(z[t])
                    yp = np.copy(z[min(T-1,t+1)])
                    if t==0:
                        # Don't do decomposition for starting conditions.
                        yc *= 1.e-10; ym *= 1.e-10; yp *= 1.e-10
                    else:
                        if "(-" in v2:
                            # For lag variables set current amd lead variables to zero.
                            yc *= 1.e-10; yp *= 1.e-10;
                        elif "(" in v2 or "(+" in v2:
                            # For lead variables set current amd lag variables to zero.
                            yc *= 1.e-10; ym *= 1.e-10;
                        else:
                            # For current variables set lead amd lag variables to zero.
                            ym *= 1.e-10; yp *= 1.e-10;
                    yt = np.array([ym,yc,yp])
                    try:
                        f = get_function(model,yt,func=func,t=t,shock=np.zeros(nd_shock*n_shocks),exog=np.zeros(ne))
                        fi = f[i]
                    except ZeroDivisionError:
                        fi = float('Inf')
                    except Exception:
                        fi = float('nan')
                    vv.append(fi)
                if len(vv)>0:
                    var_label = var_labels[v2] if v2 in var_labels else v2
                    data[var_label] = vv
            
            ### Shocks
            shock_variables = [x for x in shock_names if x in rightEqArr]
            if bool(shock_variables):
                    
                shock_variables = [var_labels[x] if x in var_labels else x for x in shock_variables]
    
                z = y[0] * 1.e-10
                yt = np.array([z,z,z])
                nd = np.ndim(shock_values)
                shocks = []
        
                if isKF:
                    shocks = shock_values
                else:
                    for t in range(T):
                        if nd == 2:
                            if t <= len(shock_values):
                                shock = shock_values[t] 
                                shocks.append(shock)
                        else:
                            shock = np.copy(shock_values)
                            if periods is None: 
                                shock = np.copy(shock_values)
                            elif not t in periods:
                                shock = np.zeros(n_shocks)
                            if t > 0: 
                                shocks.append(list(shock))
                vv = []
                for t in range(T):
                    try:
                        f = get_function(model,yt,func=func,t=t,shock=shocks[t-1],exog=np.zeros(ne))
                        fi = f[i]
                    except ZeroDivisionError:
                        fi = float('Inf')
                    except:
                        fi = float('nan')
                    vv.append(fi)
                if len(vv)>0:
                    data[",".join(shock_variables)] = vv
              
            
            ### Exogenous variables
            shock = np.zeros(nd_shock*n_shocks)
            exog_variables = [x for x in exog_names if x in rightEqArr]
            exog_labels = [var_labels[x] if x in var_labels else x for x in exog_variables]
            
            if bool(exog_variables):
                vv = []
                z = y[0] * 1.e-10
                yt = np.array([z,z,z])
                for t in range(T):
                    try:
                        exog = getExogenousData(model,t)
                        f = get_function(model,yt,func=func,exog=exog,t=t,shock=np.zeros(nd_shock*n_shocks))
                        fi = f[i]
                    except ZeroDivisionError:
                        fi = float('Inf')
                    except:
                        fi = float('nan')
                    vv.append(fi)
                if len(vv)>0:
                    data[",".join(exog_labels)] = vv
        
    var = list(set(var))
    return data,var 

def getParameters(parameters=None,model=None,t=0):
    """
    Return model parameters.
    
    Parameters:
        :param parameters: Values of parameters.
        :type parameters: numpy.array.
        :param model: Model object.
        :type model: model.
        :param t: Current period.
        :type t: int.
    """
    if parameters is None:
        parameters = model.calibration['parameters']
    if np.ndim(parameters) == 2:
        dim1,dim2 = parameters.shape
        k = min(t,dim2-1)
        params = np.copy(parameters[:,k])
        if not paramLeadsLags is None:
            for i in range(min(dim1,len(paramLeadsLags))):
                j1,j2 = paramLeadsLags[i]
                if not j2 == 0:
                    j = min(max(0,k+j2),dim2)
                    params[i] = parameters[j1,j]
    else:
        params = np.array(parameters)
        
    return params

def qzdiv(A,B,Q,Z,cluster=None,v=None):
    """
    Reorder QZ decomposition matrices.
    
    Takes U.T. matrices A, B, orthogonal matrices Q,Z, rearranges them
    according to the order of matrix roots: abs(B(i,i)/A(i,i)) 
    so that lower values are in lower right corner, while 
    preserving U.T. and orthonormal properties and Q'AZ' and Q'BZ'.  
    The columns of v are sorted correspondingly.
    
    """
    def order(root):
        n  = len(cluster)
        m  = len(root)
        cl = {}
        b  = root<=cluster[0]
        for i in range(m):
            if b[i]:
                cl[i] = 0
        for j in range(1,n):
            b1 = root>cluster[j-1]
            b2 = root<=cluster[j]
            b  = b1*b2
            for i in range(m):
                if b[i]:
                    cl[i] = j
        b = root>cluster[-1]
        for i in range(m):
            if b[i]:
                cl[i] = n
        return cl
        
    n = len(A)
    if v is None: 
        v = np.arange(n)
        
        
    diagA = abs(np.diag(A))
    diagB = abs(np.diag(B))
    diagA = [x if not x==0 else -y for x,y in zip(diagA,diagB)]
    root  = diagB / diagA
    
    
    if cluster is None:  # Reorder matrices by the diagonal values ascending order
        
        for i in range(n):
           for j in range(i-1):
              if root[j]>root[j+1]:
                  A,B,Q,Z = qzswitch2(j,A,B,Q,Z)
                  root = np.array([abs(y/x) if not x==0 else np.inf for x,y in zip(np.diag(A),np.diag(B))])
                  v[j],v[j+1] = v[j+1],v[j]
                  
    else:  # Reorder matrices by cluster order
        
        cl = order(root)
        for i in range(n):
           for j in range(i-1):
              if cl[j]>cl[j+1]:
                  A,B,Q,Z = qzswitch2(j,A,B,Q,Z)
                  root = np.array([abs(y/x) if not x==0 else np.inf for x,y in zip(np.diag(A),np.diag(B))])
                  cl = order(root)
                  v[j],v[j+1] = v[j+1],v[j]
        

    return A,B,Q,Z,v

def qzdiv2(stake,A,B,Q,Z,v=None):
    """
    Reorder QZ decomposition matrices.
    
    Takes U.T. matrices A, B, orthonormal matrices Q,Z, rearranges them
    so that all cases of abs(B(i,i)/A(i,i))>stake are in lower right 
    corner, while preserving U.T. and orthonormal properties and Q'AZ' and
    Q'BZ'.  The columns of v are sorted correspondingly.
    
    by Christopher A. Sims
    modified (to add v to input and output) 7/27/2000
    """
    n = len(A)
    if v is None: 
        v = np.arange(n)
        
    diagA = np.diag(A)
    diagB = np.diag(B)
    root  = abs(np.vstack((diagA,diagB))).T
    ind   = root[:,0] < 1.e-13
    temp  = root[:,0] + root[:,1]
    root[:,0] = root[:,0] - ind * temp
    root[:,1] = root[:,1] / root[:,0]
    for i in range(n-1,-1,-1):
        m = 0
        for j in range(i,-1,-1):
          #print(i,j,func(root[j,1]),root[j,1])
          if root[j,1] > stake or root[j,1] < -0.1:
              m = j
              break
        if m == 0: 
          return A,B,Q,Z,v
       
        for k in range(m,i):
          A,B,Q,Z = qzswitch2(k,A,B,Q,Z)
          root[k],root[k+1] = root[k+1],root[k]
          v[k],v[k+1] = v[k+1],v[k]

    return A,B,Q,Z,v      

def qzswitch(i,A,B,Q,Z):
    """
    Swap QZ decomposition matrices.  This is a simplified version of Sims' code.
    
    Takes U.T. matrices A, B, orthonormal matrices Q,Z, interchanges
    diagonal elements i and i+1 of both A and B, while maintaining
    Q'AZ' and Q'BZ' unchanged.  If diagonal elements of A and B
    are zero at matching positions, the returned A will have zeros at both
    positions on the diagonal.  This is natural behavior if this routine is used
    to drive all zeros on the diagonal of A to the lower right, but in this case
    the qz transformation is not unique and it is not possible simply to switch
    the positions of the diagonal elements of both A and B.
    """
    a = A[i,i];   d = B[i,i];     b = A[i,i+1]
    e = B[i,i+1]; c = A[i+1,i+1]; f = B[i+1,i+1]
    
    wz = np.array([c*e-f*b, (c*d-f*a).conjugate()])
    xy = np.array([(b*d-e*a).conjugate(), (c*d-f*a).conjugate()])
    n = np.sqrt(wz @ wz.T.conjugate())
    m = np.sqrt(xy @ xy.T.conjugate())
    
    if n*m == 0: 
        return A,B,Q,Z
    
    wz = wz/n; xy = xy/m
    temp = np.array([-wz[1].conjugate(), wz[0].conjugate()])
    wz = np.array([wz,temp])
    temp = np.array([-xy[1].conjugate(), xy[0].conjugate()])
    xy = np.array([xy,temp])
       
    A[i:i+2,:] = xy @ A[i:i+2,:]
    B[i:i+2,:] = xy @ B[i:i+2,:]
    A[:,i:i+2] = A[:,i:i+2] @ wz
    B[:,i:i+2] = B[:,i:i+2] @ wz
    Z[:,i:i+2] = Z[:,i:i+2] @ wz
    Q[i:i+2,:] = xy @ Q[i:i+2,:]
    
    return A,B,Q,Z

def qzswitch2(i,A,B,Q,Z):
    """
    Swap QZ decomposition matrices.  
    This is an original version of Sims' code.
    
    Takes U.T. matrices A, B, orthonormal matrices Q,Z, interchanges
    diagonal elements i and i+1 of both A and B, while maintaining
    Q'AZ' and Q'BZ' unchanged.  If diagonal elements of A and B
    are zero at matching positions, the returned A will have zeros at both
    positions on the diagonal.  This is natural behavior if this routine is used
    to drive all zeros on the diagonal of A to the lower right, but in this case
    the qz transformation is not unique and it is not possible simply to switch
    the positions of the diagonal elements of both A and B.
    
    Translated from the original Sims' Matlab code.
    """
    eps = 1.e-15
    realsmall = 1.e-7
    
    a = A[i,i];   d = B[i,i];     b = A[i,i+1]
    e = B[i,i+1]; c = A[i+1,i+1]; f = B[i+1,i+1]
    
    if (abs(c)<realsmall and abs(f)<realsmall):
        if abs(a)<realsmall:
            # l.r. coincident 0's with u.l. of A=0; do nothing
            return A,B,Q,Z
        else:
    		# l.r. coincident zeros; put 0 in u.l. of a
            wz = np.array([b,-a]).T
            wz = wz / np.sqrt(wz.T.conjugate() @ wz)
            wz = np.vstack((wz, np.array([wz[1].conjugate(), -wz[0].conjugate()])))
            xy = np.eye(2)
    elif (abs(a)<realsmall and abs(d)<realsmall):
        if abs(c)<realsmall:
            # u.l. coincident zeros with l.r. of A=0; do nothing
            return A,B,Q,Z
        else:
            # u.l. coincident zeros; put 0 in l.r. of A
            wz = np.eye(2)
            xy = np.array([c, -b])
            xy = xy / np.sqrt(xy @ xy.T.conjugate())
            xy = np.vstack((np.array([xy[1].conjugate(), -xy[0].conjugate()]), xy))
    else:
        # usual case
        wz = np.array([c*e-f*b, (c*d-f*a).conjugate()])
        xy = np.array([(b*d-e*a).conjugate(), (c*d-f*a).conjugate()])
        n = np.sqrt(wz @ wz.T.conjugate())
        m = np.sqrt(xy @ xy.T.conjugate())
        if m < eps * 100:
            # all elements of A and B proportional
            return A,B,Q,Z

        wz = wz / n
        xy = xy / m
        wz = np.vstack((wz, np.array([-wz[1].conjugate(), wz[0].conjugate()])))
        xy = np.vstack((xy, np.array([-xy[1].conjugate(), xy[0].conjugate()])))

    A[i:i+2,:] = xy @ A[i:i+2,:]
    B[i:i+2,:] = xy @ B[i:i+2,:]
    A[:,i:i+2] = A[:,i:i+2] @ wz
    B[:,i:i+2] = B[:,i:i+2] @ wz
    Z[:,i:i+2] = Z[:,i:i+2] @ wz
    Q[i:i+2,:] = xy @ Q[i:i+2,:]
    
    return A,B,Q,Z

def cartesian(arrays, out=None):
    """
    Generate a cartesian product of input arrays.

    Args:
        arrays: list of array-like
            1-D arrays to form the cartesian product of.
        out: ndarray
            Array to place the cartesian product in.

    Returns:
        out : ndarray
            2-D array of shape (M, len(arrays)) containing cartesian products
            formed of input arrays.

    Examples:
        
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    >>> array([[1, 4, 6],
               [1, 4, 7],
               [1, 5, 6],
               [1, 5, 7],
               [2, 4, 6],
               [2, 4, 7],
               [2, 5, 6],
               [2, 5, 7],
               [3, 4, 6],
               [3, 4, 7],
               [3, 5, 6],
               [3, 5, 7]])

    """
    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = n // arrays[0].size
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m,1:])
        for j in range(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]
    return out
	
def hermgauss(n):
    """
    Compute weights and nodes of Gauss-Hermite quadrature.

    Credits : both routines below are ported from the Compecon Toolbox
    by Paul L Fackler and Mario J. Miranda.
    Original code is available at http://www4.ncsu.edu/~pfackler/compecon/toolbox.html
   
    Translated from Matlab code to Python by A.G.
    """
    maxit = 100
    pim4 = 1/np.pi**0.25
    m = int( np.fix( (n+1)/2 ) )

    x = np.zeros(n)
    w = np.zeros(n)
    # reasonable starting values
    for i in range(m):
        if i==0:
            z = np.sqrt(2*n+1)-1.85575*((2*n+1)**(-1/6))
        elif i==1:
            z = z-1.14*(n**0.426)/z
        elif i==2:
            z = 1.86*z+0.86*x[0]
        elif i==3:
            z = 1.91*z+0.91*x[1]
        else:
            z = 2*z+x[i-2]
        # root finding iterations
        its = 0
        while its < maxit:
            its += 1
            p1 = pim4
            p2 = 0
            for j in range(n):
                p3 = p2
                p2 = p1
                p1 = z*np.sqrt(2/(j+1))*p2-np.sqrt(j/(j+1))*p3;
            pp = np.sqrt(2*n)*p2
            z1 = z
            z = z1-p1/pp
            if abs(z-z1)<1e-14:
                break
        if its >= maxit:
            raise Exception('Failure to converge')
        x[n-i-1] = z
        x[i] = -z
        w[i] = 2/pp**2
        w[n-i-1] = w[i]

    return [x,w]

def gauss_hermite_nodes(orders, sigma, mu=None):
    """
    Compute the weights and nodes for Gauss Hermite quadrature.

    Args:
        orders : int, list, array
            The order of integration used in the quadrature routine
        sigma : array-like
            If one dimensional, the variance of the normal distribution being
            approximated. If multidimensional, the variance-covariance matrix of
            the multivariate normal process being approximated.

    Returns:
        x : array
            Quadrature nodes
        w : array
            Quadrature weights
    """
    if isinstance(orders, int):
        orders = [orders]

    if mu is None:
        mu = np.array( [0]*sigma.shape[0] )

    herms = [hermgauss(i) for i in orders]

    points = [ h[0]*np.sqrt(2) for h in herms]
    weights = [ h[1]/np.sqrt( np.pi) for h in herms]

    if len(orders) == 1:
        # Note: if sigma is 2D, x will always be 2D, even if sigma is only 1x1.
        # print(points.shape)
        x = np.array(points[0])*np.sqrt(float(sigma))
        if sigma.ndim==2:
            x = x[:,None]
        w = weights[0]
        return [x,w]

    else:
        x = cartesian(points).T

        from functools import reduce
        w = reduce( np.kron, weights)

        zero_columns = np.where(sigma.sum(axis=0)==0)[0]
        for i in zero_columns:
            sigma[i,i] = 1.0

        C = np.linalg.cholesky(sigma)
        x = np.dot(C, x) + mu[:,np.newaxis]
        x = np.ascontiguousarray(x.T)

        for i in zero_columns:
            x[:,i] =0

        return [x,w]

def test_hermite():
    """Test hermgauss."""
    [xg,wg] = hermgauss_numpy(10)
    [x,w] = hermgauss(10)
    print(w-wg)
    print(x-xg)
	

def getTrend(data,N,deg=1):
    """
    Get trend component.
    
    Parameters:
        :param data: Observation data.
        :type data: pandas DataFrame.
        :param deg: Fitted polynomial degree.
        :type deg: int.
    """
    if np.ndim(data)==1:
        T = data.shape[0]
        n = 1
    elif np.ndim(data)==2:
        T,n = data.shape
    trend = []
    X = np.arange(N)
    
    for i in range(n):
        if np.ndim(data)==1:
            y = data
        elif np.ndim(data)==2:
            y = data[:,i]
        if sum(np.isnan(y)) > len(y)-2:
            trend.append(np.zeros(N))
        else:
            xi = [X[j] for j,x in enumerate(y) if not np.isnan(x)]
            yi = [x for x in y if not np.isnan(x)]
            if deg == 0:
                c = np.polyfit(xi,yi,deg=deg)
                fit = c *np.ones(N)
            elif deg == 1:
                slope,c = np.polyfit(xi,yi,deg=deg)
                fit = c + slope*X
            elif deg == 2:
                curv,slope,c = np.polyfit(xi,yi,deg=deg)
                fit = c + slope*X + curv*X*X
            data[:,i] -= fit[:len(data[:,i])]
            trend.append(fit)
        
    trend = np.array(trend)
    
    return trend,data
    
def getTrendCofficients(model,T):
    """
    Create an array with steady state paths for all variables.
    
    Parameters:
        :param model: The Model object.
        :type model: instance of class Model.
        :param T: Time range.
        :type T: int.
    """
    trend_coeff,unstable = None,None
    
    if "A" in model.linear_model and "C" in model.linear_model:
        A = model.linear_model["A"]
        C = model.linear_model["C"]
        
        w,U  = (A)
        unst = np.abs(w) > 1+1.e-7
        unstable = np.unique(np.sum(U[:,unst],2) > 1.e-7)   
        
        v  = np.array(model.calibration['variables'])
        x0 = v
        x  = v
        for t in range(T):
            x =  A @ x + C
            
        trend_coeff = (x-x0)/T
        trend_coeff[abs(trend_coeff)<1.e-10] = 0
        
    return trend_coeff[unstable],unstable

def sorter(a, b):
    """
    Sort eigen values.
    
    The sorting algorithm of ordered Schur decomposition function.

    Args:
        a : Diagonal element of Schur decomposition.
            complex or real array.
        b : Diagonal element of Schur decomposition.
            complex or real array.

    Returns:
        clusters : np.array
            Clusters of eigen values.

    """
    TOLERANCE = 1e-10
    eigval = np.array([abs(x/y) if abs(y)>0 else np.inf for x,y in zip(a,b)])
    stable = eigval <= 1 - TOLERANCE
    unit = abs(eigval-1) < TOLERANCE
    unstable = eigval >= 1 + TOLERANCE
    
    clusters = np.empty_like(a,dtype=bool)
    clusters[stable] = False
    clusters[unit] = True
    clusters[unstable] = True
    
    return clusters

def clustered_sorter(a,b,div):
    """
    Sort eigen values.
    
    The sorting algorithm of General Schur decomposition function.

    Args:
        a : Diagonal element of Schur decomposition.
            complex or real array.
        b : Diagonal element of Schur decomposition.
            complex or real array.

    Returns:
        clusters : np.array
            Clusters of eigen values.

    """
    eigval = np.array([abs(y/x) if abs(x)>0 else np.sign(y/x)*np.inf for x,y in zip(a,b)])
    stable = eigval <= div
    unstable = eigval > div
    
    n = len(eigval)
    clusters = np.zeros(n,dtype=bool)
    clusters[unstable] = False
    clusters[stable] = True
    
    return clusters

def arnoldi_iteration(A, b, n: int):
    """Computes a basis of the (n + 1)-Krylov subspace of A: the space spanned by {b, Ab, ..., A^n b}.

    Arguments:
      A: m × m array
      
      b: initial vector (length m)
      
      n: dimension of Krylov subspace, must be >= 1
    
    Returns:
      Q: m x (n + 1) array, the columns are an orthonormal basis of the Krylov subspace.
      
      h: (n + 1) x n array, A on basis Q. It is upper Hessenberg.  
      
    Please see: 
        
    * https://en.wikipedia.org/wiki/Arnoldi_iteration 
    
    * https://en.wikipedia.org/wiki/Generalized_minimal_residual_method
    
    """
    m = A.shape[0]
    h = np.zeros((n + 1, n))
    Q = np.zeros((m, n + 1))
    q = b / np.linalg.norm(b)  # Normalize the input vector
    Q[:, 0] = q  # Use it as the first Krylov vector

    for k in range(n):
        v = A.dot(q)  # Generate a new candidate vector
        for j in range(k + 1):  # Subtract the projections on previous vectors
            h[j, k] = np.dot(Q[:, j].conj(), v)
            v = v - h[j, k] * Q[:, j]

        h[k + 1, k] = np.linalg.norm(v)
        eps = 1e-12  # If v is shorter than this threshold it is the zero vector
        if h[k + 1, k] > eps:  # Add the produced vector to the list, unless
            q = v / h[k + 1, k]  # the zero vector is produced.
            Q[:, k + 1] = q
        else:  # If that happens, stop iterating.
            return Q, h
    return Q, h

def getIndicesAndData(n,row_ind,col_ind,jacob):
    """
    Build row and column indices and partition jacobian matrix into lead, current, and lag parts.

    Parameters
    ----------
    n : int
        Number of rows.
    row_ind : list
        Row indices.
    col_ind : list
        Column indices.
    jacob : 2d numpy.array
        Jacobian matrix.

    Returns
    -------
    lead_row_ind : list
        Row indices of lead variables.
    lead_col_ind : list
        Column indices of lead variables.
    lead_data : list
        Jacobian matrix of lead variables.
    current_row_ind : list
        Row indices of current variables
    current_col_ind : list
        Column indices of current variables.
    current_data : list
        Jacobian matrix of current variables.
    lag_row_ind : list
        Row indices of lag variables
    lag_col_ind : list
        Column indices of lag variables.
    lag_data : list
        Jacobian matrix of lag variables.

    """
    lead_row_ind = []; lead_col_ind = []; lead_data = [] 
    current_row_ind = []; current_col_ind = []; current_data = []
    lag_row_ind = []; lag_col_ind = []; lag_data = []
    
    if isinstance(jacob,list):
        for r,c,d in zip(row_ind,col_ind,jacob):
            if c < n:
                lead_row_ind.append(r)
                lead_col_ind.append(c)
                lead_data.append(d)
            elif c < 2*n:
                current_row_ind.append(r)
                current_col_ind.append(c-n)
                current_data.append(d)
            elif c < 3*n:
                lag_row_ind.append(r)
                lag_col_ind.append(c-2*n)
                lag_data.append(d)
    else:
        for r,c in zip(row_ind,col_ind):
            if c < n:
                lead_row_ind.append(r)
                lead_col_ind.append(c)
                lead_data.append(jacob[r,c])
            elif c < 2*n:
                current_row_ind.append(r)
                current_col_ind.append(c-n)
                current_data.append(jacob[r,c])
            elif c < 3*n:
                lag_row_ind.append(r)
                lag_col_ind.append(c-2*n)
                lag_data.append(jacob[r,c])
        
    return lead_row_ind,lead_col_ind,lead_data,current_row_ind,current_col_ind,current_data,lag_row_ind,lag_col_ind,lag_data 
        
def getMatrix(n,row_ind,col_ind,jacob):
    """
    Build matrix given row, column and data indices.

    Parameters
    ----------
    n : int
        Number of rows.
    row_ind : list
        Row indices.
    col_ind : list
        Column indices.
    jacob : 2d numpy.array
        Jacobian matrix.

    Returns
    -------
    lead_row_ind : list
        Row indices of lead variables.
    lead_col_ind : list
        Column indices of lead variables.
    lead_data : list
        Jacobian matrix of lead variables.
    current_row_ind : list
        Row indices of current variables
    current_col_ind : list
        Column indices of current variables.
    current_data : list
        Jacobian matrix of current variables.
    lag_row_ind : list
        Row indices of lag variables
    lag_col_ind : list
        Column indices of lag variables.
    lag_data : list
        Jacobian matrix of lag variables.

    """
    data = np.zeros((n,n))
    
    for r,c,d in zip(row_ind,col_ind,jacob):
        data[r,c] = d
        
    return data 

def checkBK(model,x,qz_factor=1+1.e-6,debug=False):
    """Check Blachard-Kahn Condition."""
    from snowdrop.src.misc.termcolor import cprint
    from snowdrop.src.utils.equations import getTopology
    from snowdrop.src.preprocessor.function import get_function_and_jacobian

    variables = model.symbols['variables']
    n = len(variables)
    p = model.calibration["parameters"]  
    e = model.calibration['shocks']
    if np.ndim(e) == 2:
        e = list(e[0])*(1+model.max_lead_shock-model.min_lag_shock)
    # n_shock = len(model.symbols['shocks'])
    # e = [0]*n_shock
    forward_columns = []
    
    C,jacob = get_function_and_jacobian(model=model,y=[x,x,x],params=p,shock=e)
    Jacob = jacob[:n,:3*n]
    
    mp = getTopology(model) 
    for k in mp:
        col = variables.index(k)
        v = mp[k]
        for e in v:
            ind = e.index(':')
            lead_lag = int(e[:ind])
            if lead_lag == 1:
                e = e[1+ind:]
                ind = e.index(',')
                var_col = int(e[1+ind:])
                forward_columns.append(var_col)
    
    forward_variables = [variables[i] for i in forward_columns]
    N_forward = len(forward_columns)
    N = n + N_forward
    
    ### ------------------- Make structural state-space representation of equations
    # Build matrix A
    A = np.zeros(shape=(N,N))
    # Current values
    A[:n,:n] = Jacob[:,n:2*n]
    # Lead variables values
    A[:n,n:] = Jacob[:,forward_columns]
    # Identity equations
    for i in range(N_forward):
        col = forward_columns[i]
        A[n+i,col] = 1
        
    # Build matrix A
    B = np.zeros(shape=(N,N))
    # Lag variables values
    B[:n,:n] = Jacob[:,2*n:3*n]
    # Identity equations
    B[n:,n:] = -np.eye(N_forward)
    
    # Matrix of constants
    C1 = np.concatenate((C,np.zeros(N_forward)))
    
    if debug:
        np.set_printoptions(precision=2)
        print("Matrix A:")
        print(A)
        print()
        print("Matrix B:")
        print(B)
        print()
        print("Vector C:")
        print(C1)
        print()   
    ### ------------------------ Generalized Schur decomposition (a.k.a. QZ decomposition) to the pencil (A,B)
    
    
    # If the QZ re-ordering fails, change the order of equations .
    # Place the first equation last, and repeat.
    eqOrd = list(np.arange(N))
    for shuffle in range(N):
        AA = A[eqOrd]
        BB = B[eqOrd]
        a,b,alpha,beta,q,z = la.ordqz(AA,BB,output='real',sort=sorter) 
        
        # Transpose matrix
        q = q.T

        # Reorder decomposition matrices so that eigen values are ordered.
        a,b,q,z,col_order = qzdiv(a,b,q,z)
            
        # Find number of stable and unstable roots
        roots = np.array([abs(y/x) if abs(x)>0 else np.inf for x,y in zip(np.diag(a),np.diag(b))])
        #print(roots)
        n_unstable = sum(roots>=qz_factor)
        n_stable = N - n_unstable  
            
        # Reshuffle equations order if QZ algorithm fails
        unstable_roots = roots[n_stable:]
        if np.any(unstable_roots<2-qz_factor) or N_forward != n_unstable:
            eqOrd = eqOrd[1:] + [eqOrd[0]]
        else:
            break

    # Check correctness of QZ decomposition
    err1 = la.norm(q.T @ a @ z.T - AA) / la.norm(AA)
    err2 = la.norm(q.T @ b @ z.T - BB) / la.norm(BB)
    if err1 > 1.e-10 or err2 > 1.e-10:
        raise Exception("Klein Solver: QZ decomposition error.  \n Inconsistency of A and B matrices decomposition of {0} and {1}.".format(round(err1,4),round(err2,4)))

    if debug:
        print('n_unstable=',n_unstable,',   n_forward_looking=',N_forward,'   forward variables=',forward_variables)
        print('roots:',roots)
        
    # Check Blanchard-Kahn condition
    print()
    if N_forward == n_unstable: 
        cprint('Blanchard-Kahn condition of existence and unique solution is satisfied.','green',attrs=['bold','underline'])
        status = 0
    elif N_forward < n_unstable:
        cprint(f"Number of unstable roots {n_unstable} is larger than number of forward looking variables {N_forward}!","red")
        cprint('Blanchard-Kahn condition is not satisfied: no stable solution!','red',attrs=['underline']) 
        status = 1
    else:
        cprint(f"Number of unstable roots {n_unstable} is smaller than number of forward looking variables {N_forward}!","red")
        cprint('Blanchard-Kahn condition is not satisfied: multiple stable solutions!','red',attrs=['underline'])
        status = 2
    print()
         
    if np.any(unstable_roots<1):
        cprint("Some of unstable roots are less than one: \n {}".format(np.round(unstable_roots,2)),"red")
        status = 3

    return status
    
def get_adjusted_shocks(model,x,shocks,N,debug=False):
    """
    Get shocks that satisfy model equations.
        
    Parameters:
        :param model: Model object.
        :type model: `Model'.
        :param x: Solution.
        :type x: numpy 2D array.
        :param shocks: Shocks at different time periods..
        :type shocks: list.
        :param N: Number of time periods.
        :type N: int.
        
        :returns: List of shocks that satisfy original equations.
    """
    import re
    from scipy.optimize import root
    from snowdrop.src.misc.termcolor import cprint
    
    TOLERANCE = 1.e-8; NITERATIONS = 100
    delimiters = " ", ",", ";", "*", "/", ":", "+", "-", "^", "{", "}", "(", ")", "="
    regexPattern = '|'.join(map(re.escape, delimiters))
      
    adjusted_shocks = np.zeros_like(shocks)
    shock_names = model.symbols["shocks"]
    n_shocks = len(shock_names)
    p = getParameters(model=model)
    
    f_dynamic = model.functions["f_dynamic"]
    bHasAttr  = hasattr(f_dynamic,"py_func")
    
    # Define objective function
    def fobj(z,exog):
        global itr
        itr += 1
        try:
            y = np.concatenate([x[t+2],x[t+1],x[t],z])
            if bHasAttr:
                func,jacob = f_dynamic.py_func(y,p,exog)
            else:
                func,jacob = f_dynamic(y,p,exog)
            f = func[ind]
        except ValueError:
            f = 1.e10 + itr + np.zeros(len(ind))
        return f

    lst_eqs = []; ind = []; b = True
    eqs = model.symbolic.equations
    
    for i,eq in enumerate(eqs):
        arr = re.split(regexPattern,eq)
        arr = list(filter(None,arr))
        for shock in shock_names:
            if shock in arr:
                if not eq in lst_eqs:
                    lst_eqs.append(eq)
                if not i in ind:
                    ind.append(i)
                    
    for t in range(N):
        exog = getExogenousData(model,t)
        sol = root(fobj, x0=np.zeros(n_shocks),args=(exog),method='lm',tol=TOLERANCE,options={"maxiter": NITERATIONS, "disp":True})      
        adjusted_shocks[t+1] = sol.x
        b &= sol.success
        
    if not b:
        cprint("numeric.solver.util: Cannot find adjusted shocks","red")
        
    adjusted_shocks = np.round(adjusted_shocks,6)
    if debug:
        print(ind)
        print("\n".join(lst_eqs))
        print(adjusted_shocks)   
        
    return adjusted_shocks
    
def getMatlabMatrices(n_meas_shocks):
    """Read Matlab matrices."""
    from mat4py import loadmat
    import os
    
    path = os.path.dirname(os.path.abspath(__file__))
    working_dir = os.path.abspath(os.path.join(os.path.abspath(path + "../../../..")))
    
    data = loadmat(working_dir + "/data/toy/iris.mat")
    Omg = np.array(data["Omg"])
    Pa0 = np.array(data["Pa0"])
    Painit = np.array(data["Painit"])
    Ta = np.array(data["Ta"])
    U = np.array(data["U"])
    Z = np.array(data["Z"])
    Ka = np.array(data["ka"])
    Ka = np.squeeze(Ka)
    Ra = np.array(data["Ra"])
    Ra = Ra[:,n_meas_shocks:]
    
    return U,Ta,Ra,Ka,Painit,Z,Pa0,Omg

def getMatrices(model,n,y,t=None):
    from snowdrop.src.numeric.solver.linear_solver import solve
    solve(model=model)
    # State transition matrix
    A = model.linear_model["A"][:n,:n]
    # Array of constants
    C = model.linear_model["C"][:n]
    # Matrix of shocks
    R = model.linear_model["R"][:n]
    return A,C,R

def getOneVarEqsSolution(lvars,lfunc,largs,leqs,mv,mp,exog_data,T):
    """
    Solve one variable equations.
    
    Parameters:
        :param lvars: List of equations variables.
        :type lvars: list.
        :param lfunc: Compiled functions.
        :type lfunc: list.
        :param largs: Functions arguments.
        :type largs: list.
        :param leqs: Equations.
        :type leqs: list.
        :param mv: Values of variables.
        :type mv: dictionary.
        :param mp: Parameters.
        :type mp: dictionary.
        :param exog_data: Exogenous data.
        :type exog_data: dictionary.
        :param T: Time span.
        :type T: int.
        :returns: Dictionary of time series.

    """
    from time import time
    from snowdrop.src.misc.termcolor import cprint
    from snowdrop.src.utils.util import findVariableLag
    from snowdrop.src.utils.util import findVariableLead

    m = {}
    t0 = time()
    for x,func,args,eq in zip(lvars,lfunc,largs,leqs):
        arr = []; bSuccess = True
        try:
            for t in range(1,T):
                lst = list()
                for j in range(len(args)):
                    a = args[j]
                    lag = findVariableLag(a)
                    lead = findVariableLead(a)
                    if "__" in a:
                        a = a[:a.index("__")] 
                    if a in exog_data:
                        v = exog_data[a]
                        tt = min(max(0,t+lag+lead),len(v)-1)
                        z = v.iloc[tt]
                    else:
                        z = eval(a,mv,mp)
                    lst.append(z)
                if len(lst) > 0:
                    u = func(*lst)
                else:
                    u = func()
                arr.append(u)
        except Exception as ex:
            bSuccess = False
            u = np.nan
            cprint(f"Error computing equation: {eq}","yellow")
            print(f"   {ex}; function arguments: {args}")
        if bSuccess: 
            m[x] = arr
            
    for k in m:
        if k in mv:
            m[k] = [mv[k]] + m[k]
               
    elapsed = time() - t0
    cprint(f"\nOne Variable Equation Solver: elapsed time {elapsed:.2f} (sec.)","blue")
    return m
    
# Original code from  https://yuminlee2.medium.com/union-find-algorithm-ffa9cd7d2dba               
class UnionFind:
    def __init__(self, numOfElements):
        self.parent = list(range(numOfElements))
        self.rank = [1]*numOfElements
        self.count = numOfElements

    # Time: O(logn) | Space: O(1)
    def find_set(self, node):
        while node != self.parent[node]:
            # path compression
            self.parent[node] = self.parent[self.parent[node]]
            node = self.parent[node]
        return node
    
    # Time: O(1) | Space: O(1)
    def union_set(self, node1, node2):
        root1 = self.find_set(node1)
        root2 = self.find_set(node2)

        # already in the same set
        if root1 == root2:
            return

        if self.rank[root1] > self.rank[root2]:
            self.parent[root2] = root1
            self.rank[root1] += 1
        else:
            self.parent[root1] = root2
            self.rank[root2] += 1
        
        self.count -= 1
        
def check():
    uf = UnionFind(10)
    uf.union_set(0, 1)
    uf.union_set(2, 3)
    uf.union_set(4, 5)
    uf.union_set(6, 7)
    uf.union_set(8, 9)
    
    uf.union_set(0, 2)
    uf.union_set(4, 6)
    uf.union_set(8, 0)
    
    assert uf.find_set(0) == uf.find_set(1)
    assert uf.find_set(0) == uf.find_set(2)
    assert uf.find_set(0) == uf.find_set(3)
    assert uf.find_set(7) == uf.find_set(4)
    assert uf.find_set(7) == uf.find_set(5)
    assert uf.find_set(7) == uf.find_set(6)
    assert uf.find_set(0) == uf.find_set(8)
    assert uf.find_set(0) == uf.find_set(9)
    
    uf.union_set(7, 1)
    
    assert uf.find_set(0) == uf.find_set(1)
    assert uf.find_set(0) == uf.find_set(2)
    assert uf.find_set(0) == uf.find_set(3)
    assert uf.find_set(0) == uf.find_set(4)
    assert uf.find_set(0) == uf.find_set(5)
    assert uf.find_set(0) == uf.find_set(6)
    assert uf.find_set(0) == uf.find_set(7)
    assert uf.find_set(0) == uf.find_set(8)
    assert uf.find_set(0) == uf.find_set(9)
    
    uf.union_set(5, 2)
    uf.union_set(9, 4)
    uf.union_set(3, 7)
    uf.union_set(5, 8)
    uf.union_set(9, 1)
    
    assert uf.find_set(3) == uf.find_set(8)

    print(uf.parent)
    print(uf.rank)
    print("Number of connected components:", uf.count)

if __name__ == '__main__':
    """Main program."""
    test_hermite()
    check()
    
