"""Model class."""

import os
import numpy as np
import pandas as pd
import datetime as dt
from snowdrop.src.model.settings import SolverMethod
from snowdrop.src.model.util import setValues
from snowdrop.src.utils.equations import getMaxLeadsLags
from snowdrop.src.misc.termcolor import cprint, colored
from snowdrop.src.preprocessor.function import get_function_and_jacobian

path = os.path.dirname(os.path.abspath(__file__))
lst1 = []; lst2 = []; lst3 = []

class Model:

    def __init__(self,interface,anticipate=None,options=None,infos=None,bCompileAll=True):
        """
        Construct Model class.
        
        Parameters:
            :param self: Model object.
            :type self: `Model'.
            :param interface: Interface object.
            :type interface: `Interface'.
            :param anticipate: If True future shocks are anticipated.
            :type anticipate: bool.
            :param options: Model options.
            :type options: dict.
            :param infos: Model Info.
            :type infos: dict.
            :param bCompileAll: If True compiles all functions source code.
            :type bCompileAll:  bool, optional.
            :returns:  Instance of a `Model' class.
            
        """
        self.symbolic             = interface
        self.symbols              = interface.symbols
        self.calibration          = interface.calibration_dict
        self.eq_vars              = interface.eq_vars
        self.steady_state         = interface.steady_state
        self.steady_growth        = None
        self.ss                   = interface.ss
        self.eqLabels             = interface.eqLabels
        self.numberOfNewEqs       = interface.numberOfNewEqs
        self.terminal_values      = interface.terminal_values
        self.autodiff             = interface.autodiff
        self.jaxdiff              = interface.jaxdiff
        self.SOLVER               = interface.SOLVER
        self.order                = 1 if interface.order is None else interface.order
        
        #self.variables = sum( [tuple(e) for k,e in  self.symbols.items() if k not in ('parameters','shocks','exogenous', 'values','equations_labels')], ())
        self.options              = options if options is not None else {}
        self.data_sources         = interface.data_sources if interface.data_sources is not None else {}
        self.infos                = infos
        self.name                 = "model" if not bool(self.infos) else self.infos['name'] 
        self.priors               = interface.priors   
        self.isLinear             = False
        self.solved               = False
        self.ev                   = None
        self.nUnit                = None
        self.anticipate           = None
        self.mapSwap              = None
        self.condShocks           = None
        self.date_range           = None
        self.count                = 0
        self.T                    = 100
        self.FILTER               = None
        self.SMOOTHER             = None
        self.INITIAL_CONDITION    = None
        self.INIT_COND_CORRECTION = None
        self.PRIOR                = None
        self.SAMPLING_ALGORITHM   = None
        self.total_nmbr_shocks    = 0
        self.estimate             = False
        self.Topology             = None
        self.stable               = None
        self.unstable             = None
        self.max_lead_shock       = 0
        self.min_lag_shock        = 0
        self.max_lead_exog        = 0
        self.min_lag_exog         = 0
        self.LOG_LINEARIZE_EQS    = bool(self.symbols["log_variables"]) if "log_variables" in self.symbols else False
        self.COMPLEMENTARITY_CONDITIONS   = interface.COMPLEMENTARITY_CONDITIONS
        self.flippedEndogVariables = None
        self.flippedExogVariables  = None
        self.indexOfFlippedEndogVariables = None
        self.indexOfFlippedExogVariables  = None
        self.y                     = None
        self.functions             = {}
        self.bCompileAll           = bCompileAll
        
        # Use sparse algebra matrix calculations
        self.bSparse = interface.bSparse or interface.SOLVER == SolverMethod.ABLR

        self.anticipate           = None
        self.lead_lag_incidence   = None
        self.var_rows_incidence   = None
        self.linear_model         = {}
        self.FILTER               = None
        self.SMOOTHER             = None
        self.INITIAL_CONDITION    = None
        self.INIT_COND_CORRECTION = None
        self.PRIOR                = None
        self.state_vars           = interface.state_vars
        self.eqs_number           = interface.eqs_number 
        
        # Update calibration parameters
        self.__update_from_symbolic__()
        
        # Save exogenous series
        from snowdrop.src.utils.util import getExogenousSeries
        self.calibration["exogenous"] = getExogenousSeries(self)
            
        if bCompileAll:
            self.__compile_functions__()
                
            # Check if model is linear
            var,p,e = self.calibration['variables','parameters','shocks']
            if p.ndim == 2:
                p = np.copy(p[:,0])
                
            z = np.vstack((var,var,var))
            jacob1 = get_function_and_jacobian(self,params=p,y=z,order=1)[1]
            jacob2 = get_function_and_jacobian(self,params=p,y=0.001+z,order=1)[1]
            isLinear = np.all(jacob1 == jacob2)
            self.isLinear = isLinear
            
            if self.SOLVER is None:
                if isLinear:
                    self.SOLVER = SolverMethod.AndersonMoore
                else:
                    self.SOLVER = SolverMethod.LBJ
            if self.isLinear:
                from snowdrop.src.utils.equations import getLeadLagIncidence
                from snowdrop.src.utils.equations import topology
                # from snowdrop.src.utils.equations import getVarRowsIncidence
                # Get Lead Lag Incidence matrix
                lli = getLeadLagIncidence(self)
                # getVarRowsIncidence(self)
                # Get variables topology. Find variables position in equations.     
                if not (interface.bellman) and not (interface.objective_function):
                    topology(self) 
            else:
                lli = None
    
         
    def __update_from_symbolic__(self):
        """
        Update calibration dictionary according to the symbols definitions.
        
        Parameters:
            :param self: Model object.
            :type self: `Model'.
            
        """
        from snowdrop.src.preprocessor.misc import CalibrationDict, calibration_to_vector
        
        
        # Check if calibration dictionary or options contain symbolic expressions
        b = False
        distribution = self.symbolic.options.get('distribution')
        if not distribution is None:
            b = True
        else:
            for k in self.calibration:
                val = self.calibration[k]
                if type(val) is str:
                    b = True
                    break
            shock_values = self.symbolic.options.get('shock_values')
            if not shock_values is None:
                for val in shock_values:
                    if type(val) is str:
                        b = True
                        break
                    elif type(val) is list:
                        for v in val:
                            if type(v) is str:
                                b = True
                                break
                    
        if b:
            from snowdrop.src.preprocessor.eval_solver import evaluate
            from snowdrop.src.preprocessor.symbolic_eval import NumericEval
            
            # x=dict(self.symbolic.calibration_dict)
            self.calibration_dict = evaluate(self.symbolic.calibration_dict)
    
            calib = calibration_to_vector(self.symbols,self.calibration_dict)
            self.calibration = CalibrationDict(self.symbols, calib)
    
            # Read symbolic structure
            evaluator = NumericEval(self.calibration_dict)
            self.options = evaluator.eval(self.symbolic.options)
            self.distribution = evaluator.eval(distribution)
    
            if self.distribution is None:
                self.covariances = None
            else:
                self.covariances = np.atleast_2d(np.array(self.distribution.cov))
                
        else:   
            calib = calibration_to_vector(self.symbols, self.calibration)
            self.calibration = CalibrationDict(self.symbols, calib)
            self.options = self.symbolic.options


    def get_calibration(self, pname, *args):
        """
        Return calibration value.
        
        Parameters:
            :param self: Model object.
            :type self: `Model'.
            :param pname: name of the variable.
            :type pname: str
            :param args: Variable number of arguments.
            :type args: Tuple
            :returns:  Calibration value.
            
        """
        if isinstance(pname, list):
            return [self.get_calibration(p) for p in pname]
        elif isinstance(pname, tuple):
            return tuple([self.get_calibration(p) for p in pname])
        elif len(args)>0:
            pnames = (pname,) + args
            return self.get_calibration(pnames)

        group = [g for g in self.symbols.keys() if pname in self.symbols[g]]
        try:
            group = group[0]
        except Exception:
            raise Exception('Unknown symbol {}.'.format(pname))
        i = self.symbols[group].index(pname)
        v = self.calibration[group][i]

        return v


    def set_calibration(self, *args, **kwargs):
        """
        Set the calibration value.
        
        .. note::
            PLEASE USE THIS FUNCTION ONLY WHEN VALUE IS A SCALAR!!!
            IN OTHER WORDS THIS VALUE SHOULD NOT BE TIME DEPENDENT.
            
        Parameters:
            :param self: Model object.
            :type self: `Model'
            :param args: Variable number of arguments.
            :type args: Tuple
            :param kwargs: Arbitrary number of keyword arguments 
            :type kwargs: dict
            
        """
        # raise exception if unknown symbol ?
        if len(args)==2:
            pname, pvalue = args
            if isinstance(pname, str):
                self.set_calibration(**{pname:pvalue})
        else:
            # else ignore pname and pvalue
            calib =  self.symbolic.calibration_dict
            calib.update(kwargs)
            self.__update_from_symbolic__()


    def __compile_functions__(self,order=1):
        """
        Compile model equations function and compute Jacobian matrix up to the third order.
            
        Parameters:
            :param self: Parameter.
            :type self: Model.
            :param order: Maximum order of partial equations derivatives.
            :type order: int.
            :returns:  List of functions.
        """ 
        from snowdrop.src.preprocessor.function_compiler_sympy import compile_higher_order_function
        from snowdrop.src.preprocessor.function_compiler_sympy import compile_function
        from snowdrop.src.preprocessor.function_compiler_sympy import compile_jacobian
        from snowdrop.src.utils.equations import getRHS
        import snowdrop.src.preprocessor.function_compiler_sympy as fc
        
        func = f_jacob = f_hessian = f_tensor = src_dynamic = src_sparse = None
        
        shocks = self.symbols.get('shocks',[])
        measurement_shocks = self.symbols.get('measurement_shocks',[])
        vv = self.symbols['variables']
        logv = self.symbols.get('log_variables',[])
        ss_vv = self.symbols['endogenous']
            
        # Gets maximum of leads and minimum of lags of equations variables
        max_lead,min_lag,n_fwd_looking_var,n_bkwd_looking_var,var_lead,var_lag = \
            getMaxLeadsLags(eqs=self.symbolic.equations,variables=vv)
        
        self.max_lead = max_lead
        self.min_lag = min_lag
        self.n_fwd_looking_var = n_fwd_looking_var
        self.n_bkwd_looking_var = n_bkwd_looking_var
        self.var_lead = var_lead
        self.var_lag = var_lag
        
        # Gets maximum of leads and minimum of lags of equations shock variables
        max_lead_shock,min_lag_shock,n_fwd_looking_shocks,n_bkwd_looking_shocks,shock_lead,shock_lag = \
            getMaxLeadsLags(eqs=self.symbolic.equations,variables=shocks)
         
        self.max_lead_shock = max_lead_shock
        self.min_lag_shock = min_lag_shock
        self.n_fwd_looking_shocks = n_fwd_looking_shocks
        self.n_bkwd_looking_shocks = n_bkwd_looking_shocks
        self.var_lead = shock_lead
        self.var_lag = shock_lag
           
        # Constructs arguments of function f([y(1),y,y(-1),e],p)
        syms = [(v,1) for v in vv] + [(v,0) for v in vv] + [(v,-1) for v in vv]
        syms_shock = []
        for i in range(min_lag_shock,1+max_lead_shock):
            syms_shock += [(s,i) for s in shocks]
        
        syms += syms_shock
        syms_ss = [(v,0) for v in ss_vv] + [(v,0) for v in shocks]
        
        exog = self.symbols.get('exogenous',None)
        
        if exog is None:
            syms_exog = []
            syms_exog_ss = []
            
        else:
            # Get maximum of leads and minimum of lags of equations exogenous variables
            max_lead_exog,min_lag_exog,n_fwd_looking_exog,n_bkwd_looking_exog,exog_lead,exoglag = \
                getMaxLeadsLags(eqs=self.symbolic.equations,variables=exog)
            
            self.max_lead_exog = max_lead_exog
            self.min_lag_exog = min_lag_exog
            
            syms_exog = []
            for i in range(min_lag_exog,1+max_lead_exog):
                syms_exog += [(s,i) for s in exog]
                
            syms_exog_ss = [(v,0) for v in exog]
            
         # Measurement variables
        if 'measurement_variables' in self.symbols:
            vm = self.symbols['measurement_variables']
        else:   
            vm = []
        
         # Constructs arguments of function f(y,e,p)
        meas_syms = [(v,0) for v in vv]
        meas_syms += [(v,0) for v in vm]
        if bool(measurement_shocks):
            meas_syms += [(s,0) for s in measurement_shocks]
            
        params = self.symbols['parameters']

        ### Construct list of equations to differentiate
        # Transient equations
        eqs = []
        for eq in self.symbolic.equations:
            arr = str.split(eq,'=')
            if len(arr) == 2:
                s = '{} - ({})'.format(arr[0].strip(),arr[1].strip())
                s = str.strip(s)
            else:
                s = eq
            eqs.append(s)
            
        steady_state_eqs = []
        for eq in self.symbolic.ss_equations:
            arr = str.split(eq,'=')
            if len(arr) == 2:
                s = '{} - ({})'.format(arr[0].strip(),arr[1].strip())
                s = str.strip(s)
            else:
                s = eq
            steady_state_eqs.append(s)

        if bool(self.symbolic.measurement_equations):
            # Measurement equations   
            meas_eqs = []
            for eq in self.symbolic.measurement_equations:
                arr = str.split(eq,'=')
                if len(arr) == 2:
                    s = '{} - ({})'.format(arr[0].strip(),arr[1].strip())
                    s = str.strip(s)
                else:
                    s = eq
                meas_eqs.append(s)
                    
            # Compile measurement function and compute its partial derivatives 
            fc.skip = True
            meas_params = self.symbols['measurement_parameters']
            f_measurement = compile_higher_order_function(equations=meas_eqs,syms=meas_syms,
                                                          params=meas_params,syms_exog=syms_exog,order=order,
                                                          function_name='f_measurement',out='f_measurement',
                                                          model_name=self.infos["name"],log_variables=logv)
            f_measurement = f_measurement[0]
            fc.skip = False
                
        else:
            f_measurement = None
                
     
        # Right-hand-side of equations
        b = None
        # Compile right-hand-side equations 
        rhs_eqs,_ = getRHS(eqs=self.symbolic.equations,variables=vv,eqLabels=self.eqLabels,b=b)
        f_rhs,txt_rhs,_,_,_,src_rhs = compile_higher_order_function(equations=rhs_eqs,syms=syms,params=params,syms_exog=syms_exog,
                                                                    eq_vars=self.eq_vars,order=0,function_name='f_rhs',
                                                                    out='f_rhs',b=b,model_name=self.infos["name"],log_variables=[])
            
            
        if self.autodiff or self.jaxdiff:
            
            # Constructs arguments of function f(y(1),y,y(-1),p)
            endog = [(v,1) for v in vv] + [(v,0) for v in vv] + [(v,-1) for v in vv]
            exog = self.symbols['parameters']
            for i in range(min_lag_shock,1+max_lead_shock):
                endog += [(s,i) for s in shocks]
                
            func,txt_dynamic = compile_function(eqs,syms=endog,params=exog,syms_exog=syms_exog,log_variables=logv,b=True)
            f_dynamic = f_steady = f_sparse = txt_sparse = src_ss = src_ss_jacob = None
            
            if self.bSparse:
                from snowdrop.src.preprocessor.function_compiler_sympy import get_indices
                from snowdrop.src.numeric.solver import LBJax
                #ind = get_indices(eqs,syms=endog)
                LBJax.row_ind, LBJax.col_ind = get_indices(eqs,syms=endog)
            
            if self.autodiff:
                from autograd import jacobian,hessian  
                if order <= 1:
                    f_jacob = jacobian(func)
                elif order <= 2:
                    f_hessian = hessian(func)
                elif order <= 3:
                    f_tensor = jacobian(hessian)
            elif self.jaxdiff:
                from jax import jacobian,hessian,jit
                if order <= 1:
                    f_jacob = jit(jacobian(func),backend='cpu')
                elif order <= 2:
                    f_hessian = jit(hessian(func),backend='cpu')
                elif order <= 3:
                    f_tensor = jit(jacobian(hessian),backend='cpu')
                func = jit(func,backend='cpu')
            
        else:
            
            if True or self.bSparse:
                # Compile dynamic function and compute its partial derivatives  
                f_sparse,txt_sparse,txt_der,txt_der2,txt_der3,src_sparse = compile_higher_order_function(
                        equations=eqs,syms=syms,params=params,syms_exog=syms_exog,eq_vars=self.eq_vars,order=order,
                        function_name='f_sparse',out='f_sparse',bSparse=True,b=b,model_name=self.infos["name"],log_variables=logv)
            else:
                f_sparse = txt_sparse = src_sparse = None

            # Compile dynamic function and compute its partial derivatives  
            f_dynamic,txt_dynamic,txt_der,txt_der2,txt_der3,src_dynamic = compile_higher_order_function(
                    equations=eqs,syms=syms,params=params,syms_exog=syms_exog,eq_vars=self.eq_vars,
                    order=order,function_name='f_dynamic',out='f_dynamic',b=b,
                    model_name=self.infos["name"],log_variables=logv)

            # Compile steady-state function
            f_steady,src_ss = compile_function(
                equations=steady_state_eqs,syms=syms_ss,params=params,syms_exog=syms_exog_ss,
                eq_vars=self.eq_vars,function_name='f_steady',out='f_steady',
                b=b,model_name=self.infos["name"],log_variables=logv)
            
            # Compile partial derivatives of steady-state function            
            f_jacob,src_ss_jacob = compile_jacobian(
                equations=steady_state_eqs,syms=syms_ss,params=params,syms_exog=syms_exog,
                eq_vars=self.eq_vars,function_name='f_jacob',out='f_jacob',
                b=b,model_name=self.infos["name"],log_variables=logv)

        def f_static(y,p,e):
            from snowdrop.src.numeric.solver.util import getExogenousData
            
            if np.ndim(p) == 2:
                p = p[:,-1]
            x = list(e)          
            for i in range(min_lag_shock,max_lead_shock):
                x += x
            z = np.concatenate([y,y,y,x]) 
            
            # Get exogenous time series
            exog = getExogenousData(self)
    
            bHasAttr  = hasattr(f_dynamic,"py_func")   
            if bHasAttr:
                f = f_dynamic.py_func(z,p,exog=exog,order=0)  
            else:
                f = f_dynamic(z,p,exog=exog,order=0)  
            return f
        
        functions = {
            'f_static':       f_static,
            'f_dynamic':      f_dynamic,
            'f_sparse':       f_sparse,
            'f_steady':       f_steady,
            'f_jacob':        f_jacob,
            'f_hessian':      f_hessian,
            'f_tensor':       f_tensor,
            'f_rhs':          f_rhs,
            'f_measurement':  f_measurement,
            'func':           func
        }
        
        functions_src = {
            'f_dynamic_txt': txt_dynamic,
            'f_dynamic_src': src_dynamic,
            'f_sparse_txt':  txt_sparse,
            'f_sparse_src':  src_sparse,
            'f_steady_src':  src_ss,
            'f_jacob_src':   src_ss_jacob,
            'f_rhs_src':     src_rhs
        }
                
        self.functions = functions
        self.functions_src = functions_src
        return


    def condition(self,mapCond,reset=False):
        """
        Set conditional shocks.
        
        Parameters:
            :param self: Parameter.
            :type self: `Model'.
            :param mapCond: Dictionary of conditional shocks.
            :type mapCond: dict.
            :param reset: Flag to reset model swap dictionary.
            :type reset: bool.
        """
        if reset or self.condShocks is None:
            self.condShocks = {}
            
        m = self.condShocks
        n_range = len(self.date_range)
        var_names = self.symbols['variables']
        for k in mapCond:
            if k in var_names:
                ind = var_names.index(k)
                ts = mapCond[k]
                for x in ts.items():
                    t,v = x
                    b = False
                    for i in range(n_range-1):
                        x1 = self.date_range[i]
                        x2 = self.date_range[1+i]
                        if t >= x1 and t < x2:
                            b = True
                            period = i
                            break
                    if b:
                        if t in m:
                            values = m[t] + [(ind,v)]
                        else:
                            values = [(ind,v)]
                        m[period] = values
         
        self.condShocks = m
        

    def swap(self,var1,var2,val1=None,val2=None,delta1=None,delta2=None,rng=None,reset=False):
        """
        Swap variables, i.e. exogenize variable var1 and endogenize variable var2.
        
        Parameters:
            :param self: Parameter.
            :type self: `Model'.
            :param var1: Endogenous variable.
            :type var1: str or list of str or dict.
            :param var2: Exogenous variable.
            :type var2: str or list of str.
            :param val1: New value of exogenized variable.
            :type val1: float or pandas.Series.
            :param val2: New value of endogenized variable.
            :type val2: float.
            :param delta1: New value of exogenized variable is the value of edogenized variable plus delta1.
            :type delta1: float.
            :param delta2: New value of endogenized variable is the value of exogenized variable plus delta2.
            :type delta2: float.
            :param rng: Simulation range or a date.
            :type rng: list of datetime.
            :param reset: Flag to reset model swap dictionary.
            :type reset: bool.
        """
        from snowdrop.src.utils.util import simulationRange
        
        simulationRange(self)
        
        if reset:
            self.mapSwap = None
            
        m = self.__swap__(var1,var2,val1,val2,delta1,delta2,rng)
        
        size = 0
        for k in m:
            size += len(m[k])
                
        self.total_nmbr_shocks = size
        
        return
                
        
    def __swap__(self,var1,var2,val1=None,val2=None,delta1=None,delta2=None,rng=None):
        """
        Swap variables with their shocks (a.k.a. exogenizes variable var1 and endogenizes variable var2).
        
        Parameters:
            :param self: Parameter.
            :type self: `Model'.
            :param var1: Endogenous variable.
            :type var1: str or list of str or dict.
            :param var2: Exogenous variable.
            :type var2: str or list of str.
            :param val1: New value of exogenized variable.
            :type val1: float or pandas.Series.
            :param val2: New value of endogenized variable.
            :type val2: float.
            :param delta1: New value of exogenized variable is the value of edogenized variable plus delta1.
            :type delta1: float.
            :param delta2: New value of endogenized variable is the value of exogenized variable plus delta2.
            :type delta2: float.
            :param rng: Simulation range or a date.
            :type rng: list of datetime.
        """
        
        if isinstance(var1,dict) and isinstance(var2,list):
            sz1 = len(var1)
            sz2 = len(var2)
            if len(var1) != len(var2):
                cprint("Length of exogenized/endogenized variables {0} - {1} and {2} - {3} is different...  Skipping swap!".format(list(var1),sz1,list(var2),sz2),"red")
                return  
            i = 0 
            for v1 in var1:
                val1 = var1[v1]
                v2 = var2[i]
                self.__swap__(var1=v1,var2=v2,val1=val1,val2=v2)
                i += 1
        elif isinstance(val1,pd.Series) and isinstance(var2,str):
            self.__swap__(var1=var1,var2=var2,val1=val1.values,rng=val1.index)
        elif isinstance(var1,list) and isinstance(var2,list):
            sz1 = len(var1)
            sz2 = len(var2)
            if len(var1) != len(var2) and len(var2) > 1:
                cprint("Length of exogenized/endogenized variables {0}:{1} and {2}:{3} is different...  Skipping swap!".format(var1,sz1,var2,sz2),"red")
                return
            for i,x1 in enumerate(var1):
                x2 = var2[min(len(var2)-1,i)]
                if not val1 is None and isinstance(val1,list) and i<len(val1):
                    v1 = val1[i]
                else:
                    v1 = None
                if not val2 is None and isinstance(val2,list) and i<len(val2):
                    v2 = val2[i]
                else:
                    v2 = None
                if not delta1 is None and isinstance(delta1,list) and i<len(delta1):
                    d1 = delta1[i]
                else:
                    d1 = None
                if not delta2 is None and isinstance(delta2,list) and i<len(delta2):
                    d2 = delta2[i]
                else:
                    d2 = None
                self.__swap__(var1=x1,var2=x2,val1=v1,val2=v2,delta1=d1,delta2=d2,rng=rng)
        elif isinstance(var1,str) and isinstance(var2,str) and isinstance(val1,np.ndarray):
            for i,v1 in enumerate(val1):
                self.__swap__(var1=var1,var2=var2,val1=v1,rng=rng[i])
        else:
            variables = self.symbols["variables"]
            if var1 in variables:
                ind0 = variables.index(var1)
            else:
                if not var1 in lst1:
                    cprint("Swap: Variable '{}' is not in the list of variables".format(var1),"red")
                lst1.append(var1)
            labels = self.eqLabels
            if var1 in labels:
                ind1 = labels.index(var1)
            else:
                ind1 = None
                if not var1 in lst2:
                    cprint("Swap: Variable '{}' is not in the list of equations labels".format(var1),"red")
                lst2.append(var1)
            shocks = self.symbols["shocks"]
            if var2 in shocks:
                ind2 = shocks.index(var2)
            else:
                ind2 = None
                if not var1 in lst3:
                    cprint("Swap: Variable '{}' is not in the list of shocks".format(var2),"red")
                lst3.append(var1)
            if not delta1 is None:
                val = self.calibration[var2]
                val1 = val+delta1
            if np.isnan(val1):
                return
            if not delta2 is None:
                val = self.calibration[var1]
                val2 = val+delta2
            if self.mapSwap is None:
                self.mapSwap = {}
            if isinstance(rng,dt.datetime):
                period = 0; b = False
                n_range = len(self.date_range)
                if rng == self.date_range[-1]:
                    b = True
                    period = n_range-1
                else:
                    for i in range(n_range-1):
                        x1 = self.date_range[i]
                        x2 = self.date_range[1+i]
                        if rng >= x1 and rng < x2:
                            b = True
                            period = i
                            break
                if not b:
                    return
            elif isinstance(rng,int):
                period = rng
            else:
                period = rng
            if period in self.mapSwap:
                if self.mapSwap[period] != [(ind0,ind1,val1,ind2,val2)]:
                    self.mapSwap[period] += [(ind0,ind1,val1,ind2,val2)]
            else:
                self.mapSwap[period] = [(ind0,ind1,val1,ind2,val2)]

        return self.mapSwap
       
        
    def flip(self,endogVariables,exogVariables,bParams=True):
        """
        Flip endogenous variables with their exogenous counterparts.
        
        .. note:: This method is in DEVELOPMENT stage and needs testing.
        
        Parameters:
            :param self: instance of Model.
            :type self: `Model'.
            :param endogVariables: Endogenous variables to be flipped.
            :type endogVariables: list.
            :param exogVariables: Exogenous (shock) variables or parameters to be flipped.  
            :type exogVariables: list.
            
        """
        if isinstance(endogVariables,str):
            endogVariables = [endogVariables]
        if isinstance(exogVariables,str):
            exogVariables = [exogVariables]
            
        assert len(endogVariables) == len(exogVariables), f"Number of endogenous variables {len(endogVariables)} and exogenous variables {len(exogVariables)} is different!"
        
        if bParams:
            variables_names = self.symbols['endogenous'].copy()
            exog_names = self.symbols['parameters'].copy()
        else:
            variables_names = self.symbols['variables'].copy()
            exog_names = self.symbols['shocks'].copy()
        nv = len(variables_names)
        ns = len(exog_names)
        
        # Treat wildcards - replace astericks with the names of endogenous variables
        arr_variables = []; arr1 = []; swapped_endog = []
        for x in endogVariables:
            if x.endswith("*"):
                arr_variables += [v for v in variables_names if v.startswith(x[:-1]) and not v in arr_variables]
                swapped_endog.extend(arr_variables)
                arr1.extend(arr_variables)
            else:
                swapped_endog.append(x)
                arr1.append(x)
            
        swapped_endog = set(swapped_endog)
           
        # Treat wildcards - replace astericks with the names of exoogenous variables or parameters names
        arr_shocks = []; arr2 = []; swapped_exog = []
        for x in exogVariables:
            if x.endswith("*"):
                arr_shocks += [v for v in exog_names if v.startswith(x[:-1])  and not v in arr_shocks]
                swapped_exog.extend(arr_shocks)
                arr2.extend(arr_shocks)
            else:
                swapped_exog.append(x)
                arr2.append(x)
            
        swapped_exog = set(swapped_exog)
                
        assert len(swapped_endog) == len(swapped_exog), f"Number of swapped endogenous {swapped_endog} and exogenous variables {swapped_exog} have different number of elements and cannot be swapped!"
        
        self.flippedEndogVariables = swapped_endog
        self.flippedExogVariables = swapped_exog
        
        # Order of variables in partial derivatives matrix: lead, current, lag variables (i+nv,i,i-nv)
        # order of shocks in partial derivatives matrix: lag, current, lead shocks (i-ns,i,i+ns)
        # Mapping: i-nv -> i+ns; i -> i; i+nv -> i-ns
        if bParams:
            self.indexOfFlippedEndogVariables =  [i for i,x in enumerate(variables_names) if x in swapped_endog] 
            self.indexOfFlippedExogVariables  =  [i for i,x in enumerate(exog_names) if x in swapped_exog]
        else:
            if self.max_lead_shock == 0 and self.min_lag_shock == 0:
                self.indexOfFlippedEndogVariables =  [i+nv for i,x in enumerate(variables_names) if x in swapped_endog] 
                self.indexOfFlippedExogVariables  =  [i for i,x in enumerate(exog_names) if x in swapped_exog]
                
            elif self.max_lead_shock < 0 and self.min_lag_shock == 0:        
                self.indexOfFlippedEndogVariables =  [i+2*nv for i,x in enumerate(variables_names) if x in swapped_endog]   \
                                                  +  [i+nv for i,x in enumerate(variables_names) if x in swapped_endog] 
                self.indexOfFlippedExogVariables  =  [i for i,x in enumerate(exog_names) if x in swapped_exog]    \
                                                  +  [i+ns for i,x in enumerate(exog_names) if x in swapped_exog]
                
            elif self.max_lead_shock == 0 and self.min_lag_shock > 0:   
                self.indexOfFlippedEndogVariables =  [i for i,x in enumerate(variables_names) if x in swapped_endog]     \
                                                  +  [i+nv for i,x in enumerate(variables_names) if x in swapped_endog]        
                self.indexOfFlippedExogVariables  =  [i+ns for i,x in enumerate(exog_names) if x in swapped_exog]      \
                                                  +  [i for i,x in enumerate(exog_names) if x in swapped_exog]
                                              
            else:  
                self.indexOfFlippedEndogVariables =  [i for i,x in enumerate(variables_names) if x in swapped_endog]     \
                                                  +  [i+nv for i,x in enumerate(variables_names) if x in swapped_endog]   \
                                                  +  [i+2*nv for i,x in enumerate(variables_names) if x in swapped_endog]  
                self.indexOfFlippedExogVariables  =  [i+2*ns for i,x in enumerate(exog_names) if x in swapped_exog]    \
                                                  +  [i+ns for i,x in enumerate(exog_names) if x in swapped_exog]      \
                                                  +  [i for i,x in enumerate(exog_names) if x in swapped_exog]
        
        #cprint(f"Endogenus variable(s)\n{','.join(swapped_endog)} \nwere flipped with exogenous variable(s)\n{','.join(swapped_exog)}\n","blue")
            
        return   
    

    def __str__(self):
        """
        Represent Model class object as a string.
        
        Parameters:
            :param self: Model object.
            :type self: `Model'.
            
        """
        from snowdrop.src.utils.util import findVariableLead,findVariableLag
        from numpy import zeros
        import re
        
        delimiters = " ", ",", ";", "*", "/", ":", "^", "=", "+", "-", "(", ")"
        regexPattern = '|'.join(map(re.escape, delimiters))
        endog = self.symbols["variables"]
        
        s = u'''
Model:
------
name: "{name}"
file: "{filename}\n'''.format(**self.infos)


        b = [True for x in self.calibration['variables'] if np.isnan(x)]
        if not any(b):
            if self.isLinear:
                ss = colored('\nLinear Model\n','blue',attrs=['underline'])
            else:        	
                ss = colored('\nNon-Linear Model\n','blue',attrs=['underline'])
        else:
            ss = ""
            
        meas_eqs = self.symbolic.measurement_equations
        if meas_eqs is None:
            meas_eqs = []
         
        numberOfNewEqs = self.numberOfNewEqs
        ss += '\nTransition Equations:\n---------------------\n\n'
        res = self.residuals()
        res.update({'definitions': zeros(1)})

        equations = {'equations': self.symbolic.equations.copy()}
        definitions = self.symbolic.definitions
        tmp = []
        for deftype in definitions:
            tmp.append(deftype + ' = ' + definitions[deftype])
        definitions = {'definitions': tmp}
        equations.update(definitions)
        # for eqgroup, eqlist in self.symbolic.equations.items():
        for eqgroup in res.keys():
            if eqgroup == 'equations':
                eqlist = equations
            if eqgroup == 'definitions':
                eqlist = equations[eqgroup]
                # Update the residuals section with the right number of empty
                # values. Note: adding 'zeros' was easiest (rather than empty
                # cells), since other variable types have  arrays of zeros.
                res.update({'definitions': [None for i in range(len(eqlist))]})
            else:
                eqlist = equations[eqgroup]
            #ss += u"{}\n".format(eqgroup)
            eqLabels = self.eqLabels
            neq = len(eqlist) - numberOfNewEqs
            for i, eq in enumerate(eqlist):
                if i >= neq:
                    continue
                if not eqLabels is None and i < len(eqLabels):
                    label = eqLabels[i]
                else:
                    label = str(i+1)
                if not bool(label):
                    label = str(i+1)
                
                val = res[eqgroup][i]
                arr = re.split(regexPattern,eq)
                arr = list(filter(None,arr))
                e = eq
                for v in arr:
                    if v in endog:
                        ind = e.find(v)
                        e = e[ind+len(v):].strip()
                        if e.startswith("(") and ")" in e:
                            ind = e.find(")")
                            vv = v + e[:1+ind]
                            if "_plus_" in vv:
                                lead = findVariableLead(vv.strip())
                                ind = vv.index("_plus")
                                name = vv[:ind] + "(" + str(lead) + ")"
                                eq = eq.replace(vv,name)
                            elif "_minus_"  in vv:
                                lag  = findVariableLag(vv.strip())
                                ind = vv.index("_minus")
                                name = vv[:ind] + "(" + str(lag) + ")"
                                eq = eq.replace(vv,name)
                if self.symbolic.objective_function is not None or val is None or np.isnan(val):
                    ss += u" {label:4} :  {eqn}\n".format(label=label, eqn=eq)
                else:
                    if abs(val) < 1e-8:
                        val = 0
                    vals = '{:6.3f}'.format(val)
                    if abs(val) > 1e-8:
                        vals = colored(vals, 'red')
                    else:
                        vals = colored(vals, 'green')
                    if not str(label).isdigit():
                        lbl = str(1+i)
                        ss += u" {lbl:2}\t{vals:6} :  {eqn}\n".format(lbl=lbl,vals=vals,eqn=eq)
                    else:
                        ss += u" {label:2}\t{vals:6} :  {eqn}\n".format(label=label,vals=vals,eqn=eq)
                        
            ss += "\n"
            
        if bool(self.symbolic.measurement_equations):
            ss += '\nMeasurement Equations:\n----------------------\n\n' 
            for i,eq in enumerate(self.symbolic.measurement_equations):
                ss += u" {eqn:2}  :  {eqs}\n".format(eqn=i+1, eqs=eq)
            ss += "\n"
            
        if not self.FILTER is None and not self.symbolic.measurement_equations is None:
            ss += "Kalman Filter: {Filter}\n".format(Filter=colored(self.FILTER.name,"blue"))
            if not self.SMOOTHER is None:
                ss += "Kalman Smoother: {Smoother}\n".format(Smoother=colored(self.SMOOTHER.name,"blue"))
            else:
                ss += "No Kalman Smoother.\n"
            if not self.INIT_COND_CORRECTION is None:
                ss += "Method to initialize the error covariance matrix: {Prior}.\n".format(Prior=colored(self.INIT_COND_CORRECTION.name,"blue"))
            if not self.PRIOR is None:
                ss += "Method to initialize the error covariance matrix: {Prior}.\n".format(Prior=colored(self.PRIOR.name,"blue"))
            else:
                ss += "No transformation of the error covariance matrix.\n"
            if not self.INITIAL_CONDITION is None:
                ss += "Initial condition of endogenous variables: {initCond}\n".format(initCond=colored(self.INITIAL_CONDITION.name,"blue"))
            else:
                ss += "Initial values of endogenous variables are set in a model file.\n\n"
            print ()
        
        if self.estimate and not self.priors is None:
            ss += colored("Prior distribution of parameters:\n","blue")
            for k in self.priors:
                distr = self.priors[k]['distribution']
                params = self.priors[k]['parameters']
                init_val = params[0]
                lb = params[1]
                ub = params[2]
                prior_params = str(params[3:])
                ss += f"  {k} - {distr}, val={init_val}, lb={lb}, ub={ub}, prior parameters={prior_params} \n"
        
        if not self.SAMPLING_ALGORITHM is None:
            ss += colored(f"\nSampling algorithm: {self.SAMPLING_ALGORITHM.name}\n","blue")
            
        if bool(self.symbolic.objective_function):
            ss += colored("\nObjective Function: \n","blue")
            ss += f"func = {self.symbolic.objective_function}\n"
          
        if not self.symbolic.constraints is None:
            ss += colored("\nConstraints:\n","blue")
            #ss += "-"*12 + "\n"
            for c in self.symbolic.constraints:
                c = c.replace(".lt.", " < ")
                c = c.replace(".le.", " <= ")
                c = c.replace(".gt.", " > ")
                c = c.replace(".ge.", " >= ")
                ss += f"   {c}\n"
            
        s += ss

        return s


    def __repr__(self):
        """
        Official representation of Model class object.
        
        Parameters:
            :param self: Model object.
            :type self: `Model'.
            
        """
        return self.__str__()


    def _repr_html_(self):
        """
        HTML representation of Model class object.
        
        Parameters:
            :param self: Model object.
            :type self: `Model'.
            
        """
        from snowdrop.src.misc.latex import eq2tex

        # general informations
        infos = self.infos
        table_infos = """
     <table>
         <td><b>Model</b></td>
      <tr>
        <td>name</td>
        <td>{name}</td>
      </tr>
      <tr>
        <td>filename</td>
        <td>{filename}</td>
      </tr>
     </table>""".format(name=infos['name'],filename=infos['filename'].replace("<","&lt").replace(">","&gt"))


        # Equations and residuals
        resids = self.residuals()
        equations = {"equations": self.symbolic.equations.copy()}
        # Create definitions equations and append to equations dictionary
        definitions = self.symbolic.definitions
        tmp = []
        for deftype in definitions:
            tmp.append(deftype + ' = ' + definitions[deftype])

        definitions = {'definitions': tmp}
        equations.update(definitions)

        variables = self.symbols["variables"]
        table = "<tr><td><b>Type</b></td><td><b>Equation</b></td><td><b>Residual</b></td></tr>\n"

        for eq_type in equations:

            eq_lines = []
            for i in range(len(equations[eq_type])):
                eq = equations[eq_type][i]
                val = resids[eq_type][i]
                if abs(val) > 1e-8:
                    vals = '<span style="color: red;">{:.4f}</span>'.format(val)
                else:
                    vals = '{:.3f}'.format(val)
                if '|' in eq:
                    # keep only lhs for now
                    eq, comp = str.split(eq,'|')
                lat = eq2tex(variables, eq)
                lat = '${}$'.format(lat)
                line = [lat, vals]
                h = eq_type if i==0 else ''
                fmt_line = '<tr><td>{}</td><td>{}</td><td>{}</td></tr>'.format(h, *line)
        #         print(fmt_line)
                eq_lines.append(fmt_line)
            table += str.join("\n", eq_lines)
        table = "<table>{}</table>".format(table)

        return table_infos + table


    def residuals(self):
        """
        Compute residuals of model equations.
        
        Parameters:
            :param self: Model object.
            :type self: `Model'.
        """
        from snowdrop.src.numeric.solver.util import getParameters
     
        y = self.calibration['variables']
        p = getParameters(model=self)
        e = self.calibration['shocks']  
        if np.ndim(e) == 2:
            e = e[-2]
        if self.autodiff or self.jaxdiff:
            func = self.functions['func']
            n = 1+self.max_lead_shock-self.min_lag_shock  
            if n > 1:
                e = np.repeat(e,n)
            x = np.concatenate([y,y,y,e])
            bHasAttr  = hasattr(func,'py_func')
            if bHasAttr:
                res = func.py_func(x,p)
            else:
                res = func(x,p)
        else:
            func = self.functions['f_static']
            res = func(y,p,e)
         
        return {'equations': res}
    
            
    def setStartingValues(self,hist,skip_rows=0,bTreatMissingObs=True,debug=False):
        """
        Set starting values for current and lagged endogenous variables.
            
        Parameters:
            :param self: Model object.
            :type self: `Model'.
            :param hist: Path to historical data file or dictionary.
            :type hist: str or dict.
            :param skip_rows: The number of rows to skip.
            :type skip_rows: int.
            :param bTreatMissingObs: If True find missing variables values by minimizing residuals of steady state equations given the observations.
            :type bTreatMissingObs: bool.
            :param debug: If True print missing variables information.
            :type debug: bool.
        """
        global itr
        from snowdrop.src.model.util import getStartingValues
    
        var_values = self.calibration["variables"]
        var_names = self.symbols["variables"]
        var_values,calib,missing = getStartingValues(hist=hist,var_names=var_names,orig_var_values=var_values,options=self.options,skip_rows=skip_rows,debug=debug)
        
        if bTreatMissingObs and len(missing)>0:
            from snowdrop.src.numeric.filters.utils import getMissingInitialVariables
            var = [x for i,x in enumerate(var_names) if not "_plus_" in x and not "_minus_" in x]
            ind1 = [i for i,x in enumerate(var_names) if not "_plus_" in x and not "_minus_" in x]
            x0 = [var_values[i] for i,x in enumerate(var_names) if not "_plus_" in x and not "_minus_" in x]
            ind2 = [i for i,x in enumerate(var) if not x in missing]         
            values = getMissingInitialVariables(self,ind=ind2,x0=x0)
            var_values[ind1] = values
            
        self.calibration["variables"] = var_values    
        #x = dict(zip(self.symbols["variables"],var_values ))
        return 
    
    def setShocks(self,d,start=None,reset=False):
        """
        Set shocks values given the time of their appearance.
    
        Args:
            model : self
                self object.
            d : dict
                Map of shock name and shock values.
            start : datetime.
                Start date of simulations.  Default is None.
        Returns:
            None.
        """
        shock_names = self.symbols["shocks"]
        if reset:
            shock_values = np.zeros(len(shock_names))
        else:    
            if "shock_values" in self.options:
                shock_values = self.options["shock_values"]
            else:
                shock_values = self.calibration["shocks"]
           
        calib,startingDate,interval = setValues(model=self,d=d,names=shock_names,values=shock_values,start=start,isShock=True)           
        self.calibration["shocks"] = calib.T
        self.options["shock_values"] = calib.T
        
        n_shk,n_t = calib.shape
        
        return
    
        
    def setParameters(self,d,start=None):
        """
        Set parameters values given the time of their appearance.
    
        Args:
            self : Model
                self object.
            d : dict
                Map of parameters name and parameters values.
            start : datetime.
                Start date of simulations.  Default is None.
    
        Returns:
            None.
    
        """
        param_names = self.symbols["parameters"]
        param_values = self.calibration["parameters"]
        
        calib,_,_ = setValues(model=self,d=d,names=param_names,values=param_values,start=start,isShock=False)
        
        self.calibration["parameters"] = calib
        
            
    def setCalibration(self,param_name,param_value):
        """
        Set calibration dictionary values given the time of their appearance.
    
        Args:
            self : Model
                self object.
            param_name : str
                Parameter name.
            param_value : numeric.
                Parameter value.
    
        Returns:
            None.
    
        """
        param_names = self.symbols["parameters"]
        param_values = self.calibration["parameters"].copy()
        ind = param_names.index(param_name)
        value = param_values[ind]
        if np.isscalar(value):
            param_values[ind] = param_value
        else:
            param_values[ind] = np.zeros(len(value)) + param_value
            
        self.calibration["parameters"] = param_values

### End of Model definition


if __name__ == '__main__':
    """Main entry point."""
    
    import os
    from snowdrop.src.model.factory import import_model
    
    path = os.path.dirname(os.path.abspath(__file__))
    fname = "../../models/Toy/JLMP98.yaml"
    fpath = os.path.abspath(os.path.join(path,fname))

    interface = import_model(fpath, return_interface=True)
    infos = {
        'name' : 'JLMP98',
        'filename' : fpath
    }
    model = Model(interface, infos=infos)
    print(model)
    
    model.set_calibration(g=0.5)
    # model.setCalibration('g', 0.5)
    # model.set_calibration({'g': 0.5})
    # model.set_calibration({'g': '0.5'})
    # model.set_calibration(g=0.5)
    # model.setParameters(dict) - set time changing parameters
    p = model.calibration['parameters']
    print('Parameters:')
    print(p)
    y = model.calibration['variables']
    print('Variables values:')
    print(y)
    
