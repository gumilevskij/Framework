#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The Dynamic Integrated Climate-Economy (DICE) model by William Nordhaus. 

This is the beta version of DICE-2016R. The major changes are outlined in Nordhaus,
"Revisiting the social cost of carbon: Estimates from the DICE-2016R model."

Translation from Python version: https://github.com/hazem2410/PyDICE

Created on July 3, 2023.
@author: A.Goumilevski
"""


import os,sys
import numpy as np
from scipy.optimize import minimize
#from scipy.optimize import Bounds 

path = os.path.dirname(os.path.abspath(__file__))
working_dir = os.path.abspath(path + "/../../../..")
fdir = os.path.join(working_dir,"graphs")

sys.path.append(working_dir)
os.chdir(working_dir)

from snowdrop.src.graphs.util import plot
from snowdrop.src.driver import run,importModel

it = 0; model = None; y = None

def runDICEmodel(file_path,data_path):
    global model
    output_variables = ["E","FORC","C","DAMAGES","DAMFRAC","ABATECOST","Y","TATM","TOCEAN"]       # List of variables for which decomposition plots are produced
    decomp_variables = ["Y","DAMFRAC","ABATECOST","E"]

    # Create model object
    model = importModel(fname=file_path,use_cache=False)

    T = model.options["T"]
    shocks = model.symbols["shocks"]
    n_shk = len(shocks)
    variables_names  = model.symbols["variables"]
    ind_CEMUTOTPER = variables_names.index("CEMUTOTPER")
    variables_values = model.calibration["variables"].copy()
    variables = dict(zip(variables_names,variables_values))
    param_names = model.symbols["parameters"]
    param_values = model.calibration["parameters"].copy()
    params = dict(zip(param_names,param_values))
    var_labels = model.symbols.get("variables_labels",{})
    
    # Parameters
    tstep = params["tstep"]
    scale1 = params["scale1"]
    scale2 = params["scale2"]
    fex0 = params["fex0"]
    fex1 = params["fex1"]
    pback = params["pback"] 
    gback = params["gback"]
    prstp = params["prstp"]
    dela = params["dela"]
    gsig = params["gsig"]
    dsig = params["dsig"]
    Al = params["Al"]
    sigma = params["sigma"]
    cost1 = params["cost1"]
    eland0 = params["eland0"]
    deland = params["deland"]
    expcost2 = params["expcost2"]
    cumetree = params["cumetree"]
    popasym = params["popasym"] 
    popadj = params["popadj"]
    ga0 = params["ga0"] 
    etree = params["etree"]
    L = params["L"]
    optlrsav = params["optlrsav"]
    dk = params["dk"]
    gama = params["gama"]
    a1 = params["a1"]
    a2 = params["a2"]
    a3 = params["a3"]
    c1 = params["c1"]
    c3 = params["c3"]
    c4 = params["c4"]
    b11 = params["b11"]
    b12 = params["b12"]
    b22 = params["b22"] 
    b21 = params["b21"]
    b23 = params["b23"]
    b33 = params["b33"]
    b32 = params["b32"]  
    fco22x = params["fco22x"]
    forcoth = params["forcoth"]
    fco22x = params["fco22x"]
    t2xco2 = params["t2xco2"]
    elasmu = params["elasmu"]
    
    # Set parameters
    TT = T + 20
    L = np.full(TT,L)
    ga = np.full(TT,ga0)
    Al = np.full(TT,Al)
    gsig = np.full(TT,gsig)
    sigma = np.full(TT,sigma)
    cost1 = np.full(TT,cost1)
    etree = np.full(TT,etree)
    pbacktime = np.full(TT,pback)
    cumetree = np.full(TT,cumetree)
    
    # Time
    time = np.arange(1,TT)
    ind = param_names.index("t")
    # Discount factor 
    rr = 1/((1+prstp)**(tstep*(time-1)))
   # Radiative forcing
    forcoth = np.full(TT,fex0)
    forcoth[:18] = forcoth[:18] + (1/17)*(fex1-fex0)*np.arange(18)
    forcoth[18:] = forcoth[18:] + (fex1-fex0)
    
    for t in time:
        # Initialize Labor
        L[t] = L[t-1]*(popasym / L[t-1])**popadj
        # TFP growth rate dynamics, Eq. 7
        ga[t] = ga0 * np.exp(-dela*5*(t-1)) 
        # Initialize TFP
        Al[t] = Al[t-1]/(1-ga[t-1])
        # Initialize Growth Sigma
        gsig[t] = gsig[t-1]*((1+dsig)**tstep)
        # Backstop price
        pbacktime[t] = pback * (1-gback)**(t-1)
        # Initialize Sigma
        sigma[t] = sigma[t-1] * np.exp(gsig[t-1] * tstep)
        cost1[t] = pback * (1-gback)**(t-1) * sigma[t]  / expcost2 /1000
        #Emissions from de-forestation
        etree[t] = eland0*(1-deland)**(t-1) 
        # Initialize Carbon Tree
        cumetree[t] = cumetree[t-1] + etree[t-1]*(5/3.666)

    p = {"t": time[:T], "rr": rr[:T], "pbacktime": pbacktime[:T], "forcoth": forcoth[:T],
         "ga": ga[:T], "Al": Al[:T], "gsig": gsig[:T], "sigma": sigma[:T], "cost1": cost1[:T],
         "L": L[:T], "etree": etree[:T], "cumetree": cumetree[:T]}
    model.setParameters(p)
    
    # Allocate space and initialize arrays
    K = np.full(TT,variables["K"])
    YGROSS = np.full(TT,variables["YGROSS"])
    EIND = np.full(TT,variables["EIND"])
    E = np.full(TT,variables["E"])
    CCA = np.full(TT,variables["CCA"])
    CCATOT = np.full(TT,variables["CCATOT"])
    MAT = np.full(TT,variables["MAT"])
    ML = np.full(TT,variables["ML"])
    MU = np.full(TT,variables["MU"])
    FORC = np.full(TT,variables["FORC"])
    TATM = np.full(TT,variables["TATM"])
    TOCEAN = np.full(TT,variables["TOCEAN"])
    DAMFRAC = np.full(TT,variables["DAMFRAC"])
    DAMAGES = np.full(TT,variables["DAMAGES"])
    ABATECOST = np.full(TT,variables["ABATECOST"])
    MCABATE = np.full(TT,variables["MCABATE"])
    COSTS = np.full(TT,variables["COSTS"])
    YNET = np.full(TT,variables["YNET"])
    Y = np.full(TT,variables["Y"])
    I = np.full(TT,variables["I"])
    C = np.full(TT,variables["C"])
    CPC = np.full(1+TT,variables["CPC"])
    CEMUTOTPER = np.full(TT,variables["CEMUTOTPER"])
    RI = np.full(TT,variables["RI"])
    
    def func(MIU,S):
        # Direct forecast
        util = 0
        for t in range(1,TT-1):
            K[t] = (1-dk)**tstep * K[t-1] + tstep * I[t-1]
            YGROSS[t] = Al[t] * ((L[t]/1000)**(1-gama)) * K[t]**gama
            EIND[t] = sigma[t] * YGROSS[t] * (1 - MIU[t])
            E[t] = EIND[t] + etree[t]
            CCA[t] = CCA[t-1] + EIND[t-1] * 5 / 3.666
            CCATOT[t] = CCA[t] + cumetree[t]
            MAT[t] = MAT[t-1]*b11 + MU[t-1]*b21 + E[t-1] * 5/3.666
            ML[t] = ML[t-1] * b33  + MU[t-1] * b23
            MU[t] = MAT[t-1]*b12 + MU[t-1]*b22 + ML[t-1]*b32
            FORC[t] = fco22x * np.log(MAT[t]/588.0)/np.log(2) + forcoth[t]
            TATM[t] = TATM[t-1] + c1 * (FORC[t] - (fco22x/t2xco2) * TATM[t-1] - c3 * (TATM[t-1] - TOCEAN[t-1]))
            TOCEAN[t] = TOCEAN[t-1] + c4 * (TATM[t-1] - TOCEAN[t-1])
            DAMFRAC[t] = a1*TATM[t] + a2*TATM[t]**a3
            DAMAGES[t] = YGROSS[t] * DAMFRAC[t]
            ABATECOST[t] = YGROSS[t] * cost1[t] * MIU[t]**expcost2
            MCABATE[t] = pbacktime[t] * MIU[t]**(expcost2-1)
            COSTS[t] = YGROSS[t] - Y[t]
            YNET[t] = YGROSS[t] * (1 - DAMFRAC[t])
            Y[t] = YNET[t] - ABATECOST[t]
            I[t] = S[t] * Y[t] 
            C[t] = Y[t] - I[t]
            CPC[t] = 1000 * C[t] / L[t]
            PERIODU = ((C[t]*1000/L[t])**(1-elasmu) - 1) / (1 - elasmu) - 1
            CEMUTOTPER[t] = PERIODU * L[t] * rr[t]
            RI[t] = (1 + prstp) * (CPC[t+1]/CPC[t])**(elasmu/tstep) - 1
            util += CEMUTOTPER[t]
            
        mp = {"I":I[:T],"K":K[:T],"L":L[:T],"YGROSS":YGROSS[:T],"EIND":EIND[:T],"E":E[:T],"CCA":CCA[:T],"CCATOT":CCATOT[:T],"COSTS":COSTS[:T], 
              "MAT":MAT[:T],"MU":MU[:T],"ML":ML[:T],"FORC":FORC[:T],"TATM":TATM[:T],"TOCEAN":TOCEAN[:T],"DAMFRAC":DAMFRAC[:T],"DAMAGES":DAMAGES[:T], 
              "ABATECOST":ABATECOST[:T],"MCABATE":MCABATE[:T],"YNET":YNET[:T],"Y":Y[:T],"C":C[:T],"CPC":CPC[:T],"CEMUTOTPER":CEMUTOTPER[:T],"RI":RI[:T]}
        
        y = []
        for i,v in enumerate(variables_names):
            y = np.hstack((y,mp[v]))
        y = np.reshape(y,(len(y)//T,T)).T
            
        return y,util
        
    def fobj(obj_parameters): 
        global it, model, y
        MIU = obj_parameters[:TT]
        S = obj_parameters[TT:2*TT]
        it += 1 
        try:
            # p = {"MIU": MIU, "S": S}
            # model.setParameters(p)
            # y,rng_date = run(model=model,y0=y)
            # util = np.sum(y[ind_CEMUTOTPER])
            y,util = func(MIU,S)
            # Utility function
            util = -tstep * scale1 * util - scale2
        except:
            util = 1.e6 + it
        if it%1000 == 0:
            print(f"iter={it}, utility={util:.2f}")
        return util
    
    # Control rate limits
    MIU_lo = np.full(TT,0.01)
    MIU_up = np.full(TT,params["limmiu"])
    MIU_up[0:30] = 1
    MIU_lo[0] = params["miu0"]
    MIU_up[0] = params["miu0"]
    MIU_lo[MIU_lo==MIU_up] = 0.99999*MIU_lo[MIU_lo==MIU_up]
    bnds1 = []
    for i in range(TT):
        bnds1.append((MIU_lo[i],MIU_up[i]))
        
    # Control variables
    lag10 = np.arange(TT) > TT #ag- 10
    S_lo = np.full(TT,0.1)
    S_lo[lag10] = optlrsav
    S_up = np.full(TT,0.9)
    S_up[lag10] = optlrsav
    S_lo[S_lo==S_up] = 0.99999*S_lo[S_lo==S_up]
    bnds2 = []
    for i in range(TT):
        bnds2.append((S_lo[i],S_up[i]))
    bnds = bnds1 + bnds2
    
    # Arbitrary starting values for the control variables
    S_start = np.full(TT,0.2)
    S_start[S_start < S_lo] = S_lo[S_start < S_lo]
    S_start[S_start > S_up] = S_lo[S_start > S_up]
    MIU_start = 0.1*MIU_lo + 0.9*MIU_up
    MIU_start[MIU_start < MIU_lo] = MIU_lo[MIU_start < MIU_lo]
    MIU_start[MIU_start > MIU_up] = MIU_up[MIU_start > MIU_up]
    x_start = np.concatenate([MIU_start,S_start])
    MIU = MIU_start; S = S_start
    
    # METHODS :'SLSQP','Powell','CG','BFGS','Newton-CG' ,'L-BFGS-B','TNC','COBYLA','trust-constr','dogleg','trust-ncg','trust-exact','trust-krylov' 
    # Bounds on variables are available for: Nelder-Mead, L-BFGS-B, TNC, SLSQP, Powell, and trust-constr methods   
    print("Calibrating model parameters...")
    calibration = minimize(fun=fobj,x0=x_start,method='SLSQP',bounds=tuple(bnds),tol=1.e-5,options={'disp':True,'maxiter':100}) 
    MIU = calibration.x[:T]
    S = calibration.x[TT:TT+T]
    p = {"MIU": MIU, "S": S}
    #print(p)
    model.setParameters(p)
     

    print("\nRunning simulations...")
    rng_date,yy = run(model=model,fout=fout,
                      output_variables=output_variables,
                      decomp_variables=decomp_variables,
                      Output=True,Plot=True)
    
    data=[np.array([MIU[1:],S[1:]]).T]
    plot(path_to_dir=fdir,data=data,variable_names=["MIU","S"],rng=np.arange(T),sizes=[2,1],figsize=(8,5),var_labels=var_labels)
   
    print("\nDone!")
    
    
if __name__ == '__main__':
    """ The main test program. """
     
    fname = 'supplements/models/CLIMATE/DICE/dice.yaml' # Model file name
    fout = 'data/CLIMATE/results.csv'     # Results are saved in this file

    # Path to model file
    file_path = os.path.abspath(os.path.join(working_dir, fname))
    data_path = os.path.abspath(os.path.join(working_dir, fout))
    runDICEmodel(file_path,data_path)
    
