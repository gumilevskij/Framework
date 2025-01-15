import numpy as np
from snowdrop.src.misc.termcolor import colored

class Interface:

    def __init__(self,model_name,symbols,equations,ss_equations,calibration,exog_data=None,ss=None,order=1,
                 constraints=None,objective_function=None,eq_vars=[],steady_state=None, 
                 measurement_equations=None,bellman=None,terminal_values=None,domain=None,
                 exogenous=None,options=None,data_sources=None,measurement_file_path=None,
                 definitions=None,priors=None,bSparse=False):

        """
        Constructor that initializes Iterface object.
        
        Parameters:
            :param self: Parameter.
            :type self: Model.
            :param model_name: Name of model.
            :type model_name: str.
            :param symbols: List of symbols.
            :type symbols: list.
            :param equations: List of equations.
            :type equations: list.
            :param ss_equations: List of steady state equations. Those are obtained by dropping time index from original equations.
            :type ss_equations: list.
            :param calibration: Calibration dict.
            :type calibration: dict.
            :param exog_data: Exogenous time series.
            :type exog_data: dict.            
            :param ss: Map with variables names as a key and steady states as values.
            :type ss: dict.
            :param order: Approximation order of solution of the non-linear system of equations.
            :type order: int.
            :param constraints: Constraints of variables.
            :type constraints: list.
            :param objective_function: Objective function.
            :type objective_function: str.
            :param eq_vars: List of variables in each equation.
            :type eq_vars: list.
            :param measurement_equations:List of measurement equations.
            :type measurement_equations: list.
            :param bellman: Bellman equation.
            :type bellman: list.
            :param terminal_values: List of terminal values.
            :type terminal_values: list.
            :param domain: The domain of endogenous variables.
            :type domain: dict.
            :param exogenous: List of exogenous variables.
            :type exogenous: list.
            :param options: Model options.
            :type options: dict.
            :param measurement_file_path: Path to a file with measurement data.
            :type measurement_file_path: str.
            :param definitions: Model definitions.
            :type definitions: dict.
            :param priors: Prior distribution and parameters values.
            :type priors: dict.
            :param bSparse: Use sparse matrix algebra.
            :type bSparse: bool.
            :returns:  Instance of SymbolicModel class.
        """
        self.name = model_name
        self.steady_state = steady_state
        self.ss = ss

        # reorder symbols
        from collections import OrderedDict
        canonical_order = ['variables', 'original_variables', 'new_variables', 'measurement_variables', 'exogenous', 'states', 'controls', 'values', 'steady-state', 'shocks', 'measurement_shocks', 'parameters']
        osyms = OrderedDict()
        for vg in canonical_order:
            if vg in symbols:
                 osyms[vg] = symbols[vg]
        for vg in symbols:
            if vg not in canonical_order:
                 osyms[vg] = symbols[vg]

        self.order                  = order
        self.symbols                = osyms
        self.eq_vars                = eq_vars
        self.equations              = equations
        self.ss_equations           = ss_equations
        self.measurement_equations  = measurement_equations
        self.bellman                = bellman
        self.calibration_dict       = calibration
        self.bSparse                = bSparse
    
        self.domain                 = domain
        self.exogenous              = exogenous
        self.options                = options
        self.data_sources           = data_sources
        self.definitions            = definitions
        self.terminal_values        = terminal_values
        self.priors                 = priors
        self.measurement_file_path  = measurement_file_path
        self.exog_data              = exog_data
        
        self.constraints            = constraints
        self.objective_function     = objective_function
        
        self.SOLVER                 = None
        self.labels                 = None
        self.eqLabels               = None
        self.state_vars             = None
        self.eqs_number             = None
        self.numberOfNewEqs         = 0
        self.COMPLEMENTARITY_CONDITIONS = None
        
        # Use auto-differentiation autograd/jax package instead of symbolic differentiation with sympy package
        self.autodiff               = False
        self.jaxdiff                = False
        
        # Correct state equations if variable was assigned to fixed value:
        if not ss is None:
            for i,eq in enumerate(equations):
                if eq in ss:
                    k,v = ss[eq]
                    self.ss_equations[i] = f"{k}={v}"
		

    # def eval_formula(self, expr, dataframe=None, calib=None):
    #      """Evaluate formula."""
    #      from snowdrop.src.snowdrop.src.preprocessor.eval_formula import eval_formula  as evaluate
    #      if calib is None:
    #          calib = self.calibration_dict
    #      return evaluate(expr,dataframe=dataframe,context=calib)
    
      
    # def get_exogenous(self, **opts):
    #      """
    #      Return exogenous process object.
         
    #      Parameters:
    #          :param self: Model object.
    #          :type self: 'SymbolicModel'
    #          :param opts: Keyword arguments .
    #          :type opts: Dictionary.
    #          :returns:  Exogenous process object.
             
    #      """
    #      from snowdrop.src.preprocessor.processes import IIDProcess
    #      import copy
    #      gg = self.exogenous
    #      if gg is None:
    #          raise Exception("Model has no exogenous process.")
    #      d = copy.deepcopy(gg)
    #      d.update(opts)
    #      if 'type' in d: d.pop('type')
    #      obj =  d.eval(self.calibration_dict.flat)
    #      if not isinstance(obj, IIDProcess):
    #          raise Exception("Exogenous shocks don't follow an IID process.")
    #      else:
    #          return obj
    
    
    def get_exogenous(self):
        """Exogenous process assumption."""
        from snowdrop.src.preprocessor.language import Normal,MvNormal,LogNormal,Beta,Binomial,Gamma,Logistic,Uniform
        
        exo = self.data.get("exogenous", {})
        calibration = self.get_calibration()
        type = get_type(exo)
        if type == "Normal":
            exog = Normal(**exo)
        elif type == "MvNormal":
            exog = MvNormal(**exo)
        elif type == "LogNormal":
            exog = LogNormal(**exo)
        elif type == "Beta":
            exog = Beta(**exo)
        elif type == "Binomial":
            exog = Binomial(**exo)
        elif type == "Gamma":
            exog = Gamma(**exo)
        elif type == "Logistic":
            exog = Logistic(**exo)
        elif type == "Uniform":
            exog = Uniform(**exo)
        else:
            raise Exception("Unknown exogenous type {}.".format(type))
            
        d = exog.eval(d=calibration)
        
        return d
   
    
    def get_distribution(self, **opts):
         """
         Return random process distribution.
         
         Parameters:
             :param self: Model object.
             :type self: 'SymbolicModel'
             :param opts: Keyword arguments.
             :type opts: Dictionary.
             :returns:  Exogenous process object.
         
         """
         import copy
         gg = self.options.get('distribution')
         if gg is None:
             raise Exception("Model has no distribution.")
         d = copy.deepcopy(gg)
         d.update(opts)
         if 'type' in d: 
             d.pop('type')
         return d.eval(self.calibration_dict.flat)
     
     
    def get_cov(self, **opts):
         """
         Return covariance of distribution.
         
         Parameters:
             :param self: Model object.
             :type self: 'SymbolicModel'
             :param opts: Keyword arguments .
             :type opts: dict.
             :returns:  Covariance matrix.
             
         """
         gg = self.options.get('distribution')
         if gg is None:
             raise Exception("Model has no distribution.")
         return gg.get("cov")
     
    
    def get_domain(self):
         """
         Return domain of endogenous variables.
         
         Parameters:
             :param self: Model object.
             :type self: 'SymbolicModel'
             :returns:  Domain of endogenous variables.
             
         """
         sdomain = self.domain
         states = self.symbols['states']
         from snowdrop.src.preprocessor.language import Domain
         d = Domain(**sdomain)
         domain = d.eval(d=self.calibration_dict.flat)
         # a bit of a hack...
         for k in domain.keys():
             if k not in states:
                 domain.pop(k)
         return domain
     
        
    # def get_grid(self):
    #     """Grid definitions."""
    #     # determine bounds:
    #     domain = self.get_domain()
    #     min = domain.min
    #     max = domain.max

    #     options = self.data.get("options", {})

    #     # determine grid_type
    #     grid_type = get_type(options.get("grid"))
    #     if grid_type is None:
    #         grid_type = get_address(self.data,
    #                                 ["options:grid:type", "options:grid_type"])
    #     if grid_type is None:
    #         raise Exception('Missing grid geometry ("options:grid:type")')

    #     if grid_type.lower() in ('cartesian', 'cartesiangrid'):
    #         from snowdrop.src.utils.grids import CartesianGrid
    #         orders = get_address(self.data,
    #                              ["options:grid:n", "options:grid:orders"])
    #         if orders is None:
    #             orders = [20] * len(min)
    #         grid = CartesianGrid(min=min, max=max, n=orders)
            
    #     elif grid_type.lower() in ('smolyak', 'smolyakgrid'):
    #         from snowdrop.src.utils.grids import SmolyakGrid
    #         mu = get_address(self.data, ["options:grid:mu"])
    #         if mu is None:
    #             mu = 2
    #         grid = SmolyakGrid(min=min, max=max, mu=mu)
            
    #     else:
    #         raise Exception("Unknown grid type.")

    #     return grid
     
    def get_grid(self, **dis_opts):
         """
         Return model grid object.
         
         Parameters:
             :param self: Model object.
             :type self: 'SymbolicModel'
             :param dis_opts: Distribution options.
             :type dis_opts: dict
             :returns:  Domain of endogenous variables.
         """
         import copy
         domain = self.get_domain()
         a = np.array([e[0] for e in domain.values()])
         b = np.array([e[1] for e in domain.values()])
    
         gg = self.options.get('grid',{})
    
         d = copy.deepcopy(gg)
         gtype = dis_opts.get('type')
         if gtype:
             from snowdrop.src.preprocessor.language import minilang
             try:
                 cls = [e for e in minilang if e.__name__.lower()==gtype.lower()][0]
             except:
                 raise Exception("Unknown grid type {}.".format(gtype))
             d = cls(**d)
    
         d.update(**dis_opts)
         if 'a' not in d.keys():
             d['min'] = a
         if 'b' not in d.keys():
             d['max'] = b
         if 'type' in d: d.pop('type')
         grid = d.eval(d=self.calibration_dict.flat)
         from snowdrop.src.numeric.grids import CartesianGrid, SmolyakGrid
         if 'Cartesian' in str(grid.__class__):
             return CartesianGrid(grid.a, grid.b, grid.orders)
         if 'Smolyak' in str(grid.__class__):
             return SmolyakGrid(grid.a, grid.b, grid.mu)


    def __str__(self):
        """
        Represent Model class object as a string.
        
        Parameters:
            :param self: Model object.
            :type self: Model.
            
        """
        equations = self.equations.copy()
        variables = self.symbols["variables"].copy()
        parameters = self.symbols["parameters"].copy()
        eqLabels = self.eqLabels
        s = '\nTransition Equations:\n---------------------\n\n' 
        for i, eq in enumerate(equations):
            if not eqLabels is None and i < len(eqLabels):
                label = eqLabels[i]
            else:
                label = str(i+1)
            if not bool(label):
                label = str(i+1)
            
            s += u" {label:4} :  {eqn}\n".format(label=label, eqn=eq)

        s += "\n"
            
        if bool(self.measurement_equations):
            s += '\nMeasurement Equations:\n---------------------\n\n' 
            for i,eq in enumerate(self.measurement_equations):
                s += u" {eqn:2}  :  {eqs}\n".format(eqn=i+1, eqs=eq)
            s += "\n"

        if bool(variables):
            s += '\nVariables:\n----------\n' 
            for v in variables:
                if v in self.calibration_dict:
                    s += f" {v}  :  {self.calibration_dict[v]}\n"
                else:
                    s += f" {v}\n"
            s += "\n"
            
        if bool(self.exogenous):
            s += '\nExogenous variables:\n-----------\n'
            for v in self.exogenous:
                s += f" {v}\n"
            s += "\n"
            
        if bool(parameters):
            s += '\nParameters:\n-----------\n'
            for p in parameters:
                if p in self.calibration_dict:
                    s += f" {p}  :  {self.calibration_dict[p]}\n"
                else:
                    s += f" {p}\n"
            s += "\n"
            
        if bool(self.calibration_dict):
            s += '\nCalibration:\n-----------\n'
            for k in self.calibration_dict:  
                s += f" {k}  :  {self.calibration_dict[k]}\n"
        s += "\n"
        
        if bool(self.objective_function):
            s += colored("\nObjective Function: \n","blue")
            s += f"func = {self.objective_function}\n"
          
        if not self.constraints is None:
            s += colored("\nConstraints:\n","blue")
            #ss += "-"*12 + "\n"
            for c in self.constraints:
                c = c.replace(".lt.", " < ")
                c = c.replace(".le.", " <= ")
                c = c.replace(".gt.", " > ")
                c = c.replace(".ge.", " >= ")
                s += f"   {c}\n"
            

        return s


    def __repr__(self):
        """
        Official representation of Model class object.
        
        Parameters:
            :param self: Model object.
            :type self: Model.
            
        """
        return self.__str__()
    
    
def get_type(d):
    """Type of a tag."""
    try:
        s = d.tag.value
        return s.strip("!")
    except:
        v = d.get("type")
        return v
		
		
def get_address(data, address, default=None):
    """Type of a grid."""
    if isinstance(address, list):
        found = [get_address(data, e, None) for e in address]
        found = [f for f in found if f is not None]
        if len(found) > 0:
            return found[0]
        else:
            return default
        
    fields = str.split(address, ':')
    while len(fields) > 0:
        data = data.get(fields[0])
        fields = fields[1:]
        if data is None:
            return default
        
    return data

