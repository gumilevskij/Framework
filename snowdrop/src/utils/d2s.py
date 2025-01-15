### Class D2S retrieves variables position

class D2S:
    """Declare member variables."""
    
    N_backward = -1
    N_forward = -1
    N_static = -1
    systemid = None
    variables = None
    meas_variables = None
    shock_variables = None
    
    def __init__(self):
        """Pre-allocate member lists."""
        # Pre-allocate vectors of positions in derivative matrices.
        self.y_   = list()
        self.xu1_ = list()
        self.xp1_ = list()
        self.xp_  = list()
        self.e_   = list()
        # Pre-allocate vectors of positions in unsolved system matrices
        #---------------------------------------------------------------
        self.y   = list()
        self.xu1 = list()
        self.xp1 = list()
        self.xp  = list()
        self.e   = list()
        # Pre-allocate identity and duplicate vectors
        self.ident_  = list()
        self.ident1_ = list()
        self.ident  = list()
        self.ident1 = list()
        # Pre-allocate identity and duplicate vectors
        self.c = list()
    
    def order(self):
        """ Re-order variables."""
        un_predetermined = self.xu1_
        self.xu1 = reorder(un_predetermined,self.xu1)
        self.xp = reorder(un_predetermined,self.xp)
        self.xp1 = reorder(un_predetermined,self.xp1)
        self.ident = reorder(un_predetermined,self.ident)
        self.ident1 = reorder(un_predetermined,self.ident1)
        
    def __str__(self):
        """
        Representation of D2S class object as a string.
        
        :param self: D2S object.
        :type self: D2S.
        """
        s = u'''
D2S:
------
N_forward = {0}
N_backward = {1}
N_static = {2}
systemid = {3}\n

transition equations: A * [xb+;xf+] + B * [xb;xf] + E * e + K = 0

A[:,(xp+,xu+)] = Jacob[:,(xp+_,xu+_)]\n'''.format(self.N_forward,self.N_backward,self.N_static,self.systemid,self.variables)

        s += 'xu+ = {0}\n'.format(self.xu1)
        s += 'xu+_ = {0}\n'.format(self.xu1_)
        s += 'xp+ = {0}\n'''.format(self.xp1)
        s += 'xp+_ = {0}\n\n'''.format(self.xp1_)
        
        s += 'B[:,xp] = Jacob[:,xp_]\n'
        s += 'xp = {0}\n'''.format(self.xp)
        s += 'xp_ = {0}\n\n'''.format(self.xp_)
        
        s += 'E[:,e] = Jacob[:,e_]\n'
        s += 'e = {0}\n'.format(self.e)
        s += 'e_ = {0}\n\n'.format(self.e_)
        
        s += 'Measurement equations: A * y + B * xb+  + E * e + K1 = 0\n'
        s += 'y = {0}\n'.format(self.y)
        s += 'y_ = {0}\n\n'.format(self.y_)
        
        s += 'Identity equations:\n'
        s += 'ident1 = {0}\n'.format(self.ident1)
        s += 'ident = {0}\n\n'.format(self.ident)
        
        s += 'Constants:\n'
        s += 'c = {0}\n'.format(self.c)
        return s
    
    
    def __repr__(self):
        """
        Official representation of D2S class object.
        
        Parameters:
            :param self: D2S object.
            :type self: D2S.
        """
        return self.__str__()
        
        
def reorder(un_predetermined,lst):
    """Reorder list."""
    un = [] 
    for x in un_predetermined:
        if x in lst:
            un.append(x)
            
    order = un + [i for i in lst if not i in un]
    return order
      

def myd2s(model,debug=False) -> D2S:
    """
    Create derivative-to-system convertor.
    
    Args:
        model : Model
            Instance of Model class.
    
    Returns:
        None.
    
    """
    import numpy as np
    #from snowdrop.src.utils.equations import getTopology
    from snowdrop.src.utils.equations import getLeadLagIncidence
    
    # Create 'structure' object
    d2s = D2S()
    
    # Get lead Lag incidence matrix
    lli = getLeadLagIncidence(model)
    
    var = model.symbols['variables']
    if 'measurement_variables' in model.symbols:
        meas_var = model.symbols['measurement_variables']
    else:
        meas_var = []
    measVars = [x.lower() for x in meas_var]
    shock_var = model.symbols['shocks']
    
    d2s.variables = var
    d2s.meas_variables = meas_var
    d2s.shock_variables = shock_var
    
    # Find indices of variables in measurement equations
    meas_ind = list()
    for mv,x in zip(meas_var,measVars):
        if "_meas" in x:
            v = mv[:-5]
        elif "obs_" in x:
            v = mv[4:]
        lst = [i for i,x in enumerate(var) if x==v]
        if bool(lst):
            meas_ind.append(lst[0])
    
    predetermined = [int(b) for b in lli[0] if not np.isnan(b)] 
    un_predetermined = [int(f) for f in lli[2] if not np.isnan(f)]
    static = [int(c) for b,c,f in zip(lli[0],lli[1],lli[2]) if not np.isnan(c) and np.isnan(b) and np.isnan(f)]
    nb = len(predetermined)
    nu = len(un_predetermined)
    
    ny  = len(meas_var)
    nxx = len(var)
    ne  = len(shock_var)
    nx  = nxx + nu
    
    
    # Find min lag `minSh`, and max lead, `maxSh`, for each tratransition equations: A [xf+;xb+] + B [xf;xb] + E e + K = 0nsition variable.
    minSh = np.array([0 if np.isnan(b) else -1 for b in lli[0]])
    # If `x(t-k)` occurs in measurement equations then add k-1 lag
    minSh = np.array([-1 if i in meas_ind else minSh[i] for i in range(len(minSh))])
    
    maxSh = np.array([0 if np.isnan(f) else 1 for f in lli[2]])
    # If `minSh(i)` == `maxSh(i)` == 0, add an artificial lead to treat the  variable as forward-looking (to reduce state space), 
    # and to guarantee that all variables will have `maxShift > minShift`.
    maxSh = np.array([1 if (b==0 and f==0) else f for b,f in zip(minSh,maxSh)])
        
    # Minimum lag and maximum lead
    min_lag = min(minSh)
    max_lead = max(maxSh)
    
    systemid = list()
    for k in range(max_lead,min_lag-1,-1):
        # Add transition variables with this shift.
        b = (k >= minSh) * (k < maxSh)
        ind = [(k,i,var[i]) for i in range(nxx) if b[i]]
        systemid += list(ind)
        
    d2s.systemid = systemid
    
    ### Measurement equations: A1*y + B1*xb+  + E1*e + K1 = 0
    #--------------------------------------------------------
    
    # Transition equations: A [xb+;xf+] + B [xb;xf] + E e + K = 0
    #------------------------------------------------------------
    d2s.xu1  = np.arange(nxx,nx)
    d2s.xu1_ = np.array(un_predetermined,dtype=int)
    d2s.xp1  = np.arange(nxx)
    d2s.xp1_ = np.arange(nxx,2*nxx)
    
    d2s.xp  = np.arange(nxx)
    d2s.xp_ = np.arange(2*nxx,3*nxx)
    
    lst = []; lst1 = []
    # Identity equations
    for i in range(nu):
        v1 = np.zeros(nx,dtype=int)
        col = un_predetermined[i]
        v1[col] = 1
        
        lst1.append(v1)
        v = np.zeros(nx,dtype=int)
        v[nxx+i] = -1
        lst.append(v)
          
    d2s.ident = np.arange(nx)
    d2s.ident1 = np.arange(nx)
    d2s.ident_ = np.array(lst)
    d2s.ident1_ = np.array(lst1)
        
    # Shocks
    #--------
    d2s.e_ = np.arange(3*nxx,3*nxx+ne)
    d2s.e  = np.arange(ne)
    
    # Constants
    # --------
    d2s.c = np.arange(nxx+nu)
    
    d2s.N_backward = nb
    d2s.N_forward = nu
    d2s.N_static = len(static)
    
    # Reoder variables so that un-predetermined variables come first.
    #TODO: need additional work.
    #d2s.order()
    
    if debug:
        print(d2s)
    
    return d2s



