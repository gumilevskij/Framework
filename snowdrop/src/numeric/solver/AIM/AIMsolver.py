def AIMsolver(jacobian,c,model,suppress_warnings=False):
    """
      Maps jacobian to AIM 1st order model solver designed and developed by Gary Anderson
      and derives the solution for ghx and ghu from the AIM outputs.

      AIM System is given as a sum:
      i.e. for i=-$...+&   SUM(Hi*xt+i)= £*zt, t = 0, . . . ,?
      and its input as single array of matrices: [H-$...  Hi ... H+&]
      and its solution as xt=SUM( Bi*xt+i) + @*£*zt for i=-$...-1
      with the output in form bb=[B-$...  Bi ... B-1] and @=inv(Ho+H1*B-1)
      jacobian = [fy'-$...  fy'i ... fy'+&  fu']
      where [fy'-$...  fy'i ... fy'+&]=[H-$...  Hi ... H+&] and fu'= £
     
      INPUTS
        jacobian   [matrix] 1st order derivatives of the model's equations
        model      [object] Definition of the model.
     
      OUTPUTS
        aimcode    [integer]          1: the model defines variables uniquely
        aimcode is resolved in AIMerr as
    		   1:  unique solution.
    		   2:  roots not correctly computed by real_schur.
    		   3:  too many big roots. 
    		   35: too many big roots, and q(:,right) is singular.
    		   4:  too few big roots. 
    		   45: too few big roots, and q(:,right) is singular.
    		   5:  q(:,right) is singular.
    		   61: too many exact shiftrights.
    		   62: too many numeric shiftrights.
    		   63: A is NAN or INF. 
    		   64: problem in SPEIG.
               else:  return code not properly specified.
     
    """
    import numpy as np
    from .Amalg import Amalg
    from .Aimerr import Aimerr

    aimcode = -1 
    neq = len(jacobian)  # no of equations
    lags = 1  # no of lags
    leads = 1 # no of leads
    klen = leads+lags+1   # total lenght
    H = np.zeros((neq, neq*klen))  # allocate space
    # "sparse" the compact jacobia into AIM H aray of matrices
    # without exogenous shocks
    #H[:,np.argwhere(lli[:])] = jacobian[:,np.nonzero(lli[:])]
    # Rearrange matrices so that lag derivatives come first
    H[:,:neq] = jacobian[:,2*neq:3*neq]
    H[:,neq:2*neq] = jacobian[:,neq:2*neq]
    H[:,2*neq:3*neq] = jacobian[:,:neq]
    condn  = 1.e-10 #Amalg uses this in zero tests
    uprbnd = 1 + 1.e-6 #allow unit roots
    				   # forward only models - AIM must have at least 1 lead and 1 lag.
    if lags == 0:
        H = np.concatenate((np.zeros(neq),H),axis=0) 
        lags = 1 

    # backward looking only models
    if leads ==0:
        H = np.concatenate((H,np.zeros(neq)),axis=0) 
        leads = 1 

    try: # try to run AIM
        (bb,phi,F,rts,ia,nexact,nnumeric,lgroots,aimcode) = Amalg(H,neq,lags,leads,condn,uprbnd) 
    except ValueError as err:
        if not suppress_warnings:
            print('AIM Solver error: '+err)
            raise

    if aimcode==1: #if OK
        # Matrix A
        A   = np.asarray(bb)	
        F   = np.asarray(F)	
        phi = np.asarray(phi)		
		# Build matrix of shocks			 
        Psi = - jacobian[:,3*neq:] 
        R   = phi @ Psi
        # Find constants
        C   = -phi @ c
    else:
        err = Aimerr(aimcode) 
        #warning('Error in AIM: aimcode=#d, erro=#s', aimcode, err)  
        if not suppress_warnings:
            print(f'Error in AIM: aimcode= {aimcode}: {err}') 
        if aimcode < 1 or aimcode > 5:  # too big exception, use mjdgges
            msg = f'Error in AIM: aimcode={aimcode}  # {err}'
            if not suppress_warnings:
                raise ValueError(msg) 
        #    if aimcode > 5
        #        print(f'Error in AIM: aimcode= {aimcode} : {aimcode}') 
        #        aimcode=5 

    return A,C,R,phi,F,aimcode,rts
