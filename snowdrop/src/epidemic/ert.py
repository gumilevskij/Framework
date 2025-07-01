# -*- coding: utf-8 -*-
"""
Created on Monday December 28, 2020
@author: A.Goumilevski

This code solves and simulates the model developed by 
Eichenbaum M.S., Rebelo S., Trabandt M. (2020) in
“Epidemics in the Neoclassical and New Keynesian Models”.
NBER Working Paper 27430, http://www.nber.org/papers/w27430

"""

import os,sys
import pandas as pd
import numpy as np
import math
import warnings
#from scipy.optimize import root
#from scipy import signal
from scipy.optimize import minimize,Bounds
from timeit import default_timer as timer 

fdir = os.path.dirname(__file__)
working_dir = os.path.abspath(fdir + "/../../..")
os.chdir(working_dir)

from snowdrop.src.driver import run as simulate
from snowdrop.src.driver import importModel
from snowdrop.src.graphs.util import plotTimeSeries
#from snowdrop.src.model.util import setCalibration
#from snowdrop.src.numeric.filters.filters import LRXfilter as lrx
from snowdrop.src.numeric.solver import nonlinear_solver as ns
from snowdrop.src.numeric.solver.nonlinear_solver import homotopy_solver
from snowdrop.src.utils.util import simulationRange
from snowdrop.src.utils.util import getPeriods
from snowdrop.src.misc.termcolor import cprint


warnings.filterwarnings('ignore') 

graphs_path = os.path.abspath(os.path.join(working_dir,"graphs"))
dir_path = os.path.abspath(os.path.join(working_dir,'supplements/data/COVID19'))

CALIBRATE = True; HOMOTOPY = True; LOCKDOWN = False; VACCINATION = False 

I_ini=None;D_ini=None;pir=None;pid=None;mult=0;mult2=1.e10;virus_variant_start=0
Pi1_shr_target=None;Pi2_shr_target=None;Pi3_shr_target=None;sigma=1
RplusD_target=None;C_ss=None;N_ss=None;Model_R=None;vaccination_rate=0
I=None;S=None;D=None;R=None;T=None;I_OBS=None;D_OBS=None;R_OBS=None;S_OBS=None
scale1=1.e6; scale2=1.e3 # scale pi for numerical solver

Tmax = int(3*52); Nmax = 1000; MULT = 2

# Model file
fname = 'ert1_model.yaml'

    
def main(Plot=False,save=True):
    """Forecast based on ERT model."""
    from snowdrop.src.utils.load import getCalibration
    global Tmax,T,I_OBS,D_OBS,R_OBS,S_OBS,C_ss,N_ss,Pir,Pid,I_ini,D_ini
    global Pi1_shr_target,Pi2_shr_target,Pi3_shr_target,RplusD_target
    global vaccination_rate,virus_variant_start,mult,mult2,sigma
    
    file_path = os.path.abspath(os.path.join(working_dir,'supplements/models/COVID19/'+fname))

    # Create model
    model = importModel(file_path,model_info=False,graph_info=False,Solver="LBJ",use_cache=False)
    model.anticipate = True
    var_names  = model.symbols["variables"]
    #var_values = model.calibration["variables"]
    #var = dict(zip(var_names,var_values))
    par_names  = model.symbols["parameters"]
    par_values = model.calibration["parameters"]
    par = dict(zip(par_names,par_values))
    if "mult" in par:
        mult = par["mult"]
    if "mult2" in par:
        mult2 = par["mult2"]
    if "sigma" in par:
        sigma = par["sigma"]
    if "vaccination_rate" in par:
        vaccination_rate = par["vaccination_rate"]
    if "virus_variant_start" in par:
        virus_variant_start = par["virus_variant_start"]
    
    # Get calibration values
    var_names = model.symbols["variables"]
    #shock_names = model.symbols["shocks"]
    #n_shocks = len(shock_names)
    
    SHIFT = model.options["periods"][0]  # Simulations start two quarters before the start of epidemimc
    
    
    calib = getCalibration(fpath=file_path,names=var_names)

    I_ini = calib["i_ini"]; D_ini = calib["d_ini"]; share = calib['share']
    Pir = pir = calib["pir"]; Pid = pid = calib["pid"]
    pi1 = calib["pi1"]; pi2 = calib["pi2"]; pi3 = pi3_final = calib["pi3"]
    C_ss = c_ss = calib["c_ss"];  N_ss = n_ss = calib["n_ss"]
    y_ss = calib["y_ss"];  x_ss  = calib["x_ss"]; k_ss = calib["k_ss"]
    # cs_ss = calib["cs_ss"]; ci_ss = calib["ci_ss"]
    # ns_ss = calib["ns_ss"]; ni_ss  = calib["ni_ss"]
    xi = calib["xi"]; xi_final = calib["xi_final"]
    #pie_ss = calib["pie_ss"]
    Pi1_shr_target = pi1_shr_target = calib["pi1_shr_target"]
    Pi2_shr_target = pi2_shr_target = calib["pi2_shr_target"]
    Pi3_shr_target = pi3_shr_target = calib["pi3_shr_target"]
    RplusD_target  = calib["RplusD_target"]
    
    # Read SIR model parameters data
    data_path  = os.path.abspath(os.path.join(working_dir,"supplements/data/COVID19/epidemic.csv"))
    if os.path.exists(data_path):
        df = pd.read_csv(data_path,index_col=0,parse_dates=True)
        # Convert to weekly frequency
        df = df.resample("W").mean()
        shift = [0]*(1+SHIFT)
        i_obs=100*df["OBS_i"].dropna(); i_total_obs=100*df["OBS_itotal"].dropna()
        r_obs=100*df["OBS_r"].dropna(); d_obs=100*df["OBS_d"].dropna() 
        s_obs=100-i_obs-d_obs-r_obs
        I_OBS = 0.01*i_obs.values; D_OBS = 0.01*d_obs.values; R_OBS = 0.01*r_obs.values; S_OBS = 0.01*s_obs.values
  
    # Calibrate pi values
    if CALIBRATE:
        Model_R,pi1_final,pi2_final,pi3_final = calibrate(pi1,pi2,pi3,pi1_shr_target,pi2_shr_target)
        pi1 = pi1_final; pi2 = pi2_final; pi3 = pi3_final;
        SIR_R = pi3/(pir+pid)
        cprint("Solution: pi1 = {:.2e}, pi2 = {:.2e}, pi3 = {:.2f}, R+D = {:.2f}".format(pi1,pi2,pi3,RplusD_target),"green")
        model.setCalibration('pi1',pi1)
        model.setCalibration('pi2',pi2)
        model.setCalibration('pi3',pi3)
    else:
        SIR_R = Model_R = pi3/(pir+pid) 
        
    #cprint("pir={:.2e}, pid={:.2e}, pi3={:.2f}\n".format(pir,pid,pi3),"green")    
    cprint('Share={:.2f}, pi1_shr_target: {:.2f}, pi2_shr_target: {:.2f}, pi3_shr_target: {:.2f}'.format(share,Pi1_shr_target,Pi2_shr_target,Pi3_shr_target),"green")
    cprint("SIR and ERT models basic reproduction numbers: {:.2f} and {:.2f}\n".format(SIR_R, Model_R),"green")
    
    
    # SIR model forecast
    lockdown_rate = par["theta_lockdown"] if LOCKDOWN else 0
    #SIR_R = 1.3; pi3=SIR_R*(pir+pid)
    tau,s,i,r,dd,pop,vaccinated = SIR(pi3,pir,pid,s0=1,i0=I_ini,r0=0,dd0=D_ini,pop0=1,
                                      lockdown_rate=lockdown_rate,vaccination_rate=vaccination_rate)
    
    dates        = np.arange(Tmax)/13.0
    i            = 100*pd.Series(i[:Tmax],dates)
    tau          = 100*pd.Series(tau[:Tmax],dates)
    #i_tot        = np.cumsum(tau)
    dd           = 100*pd.Series(dd[:Tmax],dates)
    pop          = 100*pd.Series(pop[:Tmax],dates)
    s            = 100*pd.Series(s[:Tmax],dates)
    r            = 100*pd.Series(r[:Tmax],dates)
    vaccinated   = 100*pd.Series(vaccinated[:Tmax],dates)
    
    if Plot:
        
        # SIR model forecast
        if VACCINATION:
            series = [[i,dd],[pop,s,r,vaccinated]]
            labels=[['Infected','Deaths'],['Population','Sucseptible','Recovered','Vaccinated']]
        
        else:    
            series = [[i,dd],[pop,s,r]]
            labels=[['Infected','Deaths'],['Population','Sucseptible','Recovered']]
        
        header = 'SIR Model Forecast'
        titles = ['Basic Reproduction Number, Rb = {:.1f}'.format(SIR_R),
                  'Rb = {:.1f}'.format(SIR_R)]
 
        #plotTimeSeries(path_to_dir=graphs_path,header=header,titles=titles,labels=labels,series=series,sizes=[2,2],stacked=False,save=save)
       
    
        # Comparison of SIR model forecast with Data
        I,S,D,R,T = func([scale1*pi1,scale2*pi2,pi3])
        
        n = len(i_obs);
        dates = np.arange(n)/13.
        i  = pd.Series(100*I[:n],dates)
        d  = pd.Series(100*D[:n],dates)
        r  = pd.Series(100*R[:n],dates)
        s  = pd.Series(100*S[:n],dates)
        series = [
                  [i,pd.Series(i_obs.values,dates),pd.Series(i_total_obs.values,dates)],
                  [d,pd.Series(d_obs.values,dates)]
                 ]
        header = 'Calibrated SIR Model Forecast'
        titles = ['Infected','Deaths']
        labels = [
                  ['Infected','Active Cases','Total Cases'],
                  ['Deaths','Data']
                 ]

        #plotTimeSeries(path_to_dir=graphs_path,header=header,titles=titles,labels=labels,series=series,sizes=[2,2],stacked=False,save=save)
        

    # Append zeros at the beginning of arrays
    i_obs=np.array(shift+list(i_obs.values)); d_obs=np.array(shift+list(d_obs.values))
    r_obs=np.array(shift+list(r_obs.values)); i_total_obs=np.array(shift+list(i_total_obs.values))
    s_obs=100-i_obs-d_obs-r_obs
        
    # Add starting time of virus strain
    VIRUS_STRAIN_START = SHIFT + int(par["virus_variant_start"])  # vaccination strain starts one year after epidemics begins
    arr = [0]*VIRUS_STRAIN_START + [1]*500
    p = {"virus_resistant_strain": arr}
    model.setParameters(p)
                
    if LOCKDOWN:
        n_lockdown = 5 if VACCINATION else 1
    else:
        n_lockdown = 1
    if VACCINATION:
        n_vaccination = 5 if LOCKDOWN else 1
    else:
        n_vaccination = 1
    polycies = []; Z = np.zeros((n_lockdown,n_vaccination))
    
    
    for ii in range(n_lockdown):
        fl = ii/(n_lockdown-1.) if n_lockdown>1 else 1
        for jj in range(n_vaccination):
            fv = jj/(n_vaccination-1.) if n_vaccination>1 else 1
            y0 = None; yy = None; prev_yy = None
            if LOCKDOWN:
                # Set lockdown duration of eight weeks
                LOCKDOWN_DURATION = 3*13+2      # lockdown duration of three quarters
                LOCKDOWN_START = SHIFT + 13   # lockdown starts at the second quarter of epidemics
                arr = []
                for i in range(LOCKDOWN_DURATION):
                    arr.append((LOCKDOWN_START+i,fl*np.sin(math.pi*i/(LOCKDOWN_DURATION-1.0))))   
                p = {"lockdown_policy": arr}
                model.setParameters(p)
                ind1 = par_names.index("lockdown_policy")
                lockdown_policy = model.calibration["parameters"][ind1]
            else:
                lockdown_policy = None
                
            if VACCINATION:
                # Set vaccination duration of eight weeks
                VACCINATION_DURATION = 3*13        # vaccination duration of three quarters
                VACCINATION_START = SHIFT + 13 #5 + 8 + 9 # 5 + 8 + 9# 13  # vaccination starts at the second quarter of epidemics
                arr = []
                for i in range(VACCINATION_DURATION):
                    arr.append((VACCINATION_START+i,fv))
                arr.append((VACCINATION_START+VACCINATION_DURATION,0)) 
                p = {"vaccination_policy": arr}
                model.setParameters(p)
                ind2 = par_names.index("vaccination_policy")
                vaccination_policy = model.calibration["parameters"][ind2]
            else:
                vaccination_policy = None
           
            t0 = timer()
            ### Framework requires user to solve model by adjusting parameters step-by-step 
            ### and solving this model since it can not directly solve it for final parameters.  
            # Homotopy for pi3 and xi parameters (otherwise no solution if you set them to final values right away)
            _,_,rng,_,_,_,_,_,T = simulationRange(model=model)   
            periods = getPeriods(model,T,rng)
            if T is None:
                T = MULT*len(rng)
            
            if HOMOTOPY:
                try: 
                    # Homotopy for pi3 
                    pi3_final_steps = np.arange(start=pi3_final/3,stop=pi3_final,step=max(0.005,pi3_final/30))
                    yy,prev_yy,y0 = homotopy_solver(model=model,y0=model.calibration['variables'],par_name='pi3',par_steps=pi3_final_steps,periods=periods,T=T,tol=1.e-4,debug=True)
                    # Homotopy for xi 
                    xi_final_steps = np.arange(start=xi,stop=xi_final,step=max(0.01,(xi_final-xi)/30))
                    yy,prev_yy,y0 = homotopy_solver(model=model,y0=y0,par_name='xi',par_steps=xi_final_steps,periods=periods,T=T,tol=1.e-4,debug=True)

                except Exception as ex:
                    cprint(ex,"red")
            else:
                y0 = model.calibration['variables']
                    
            # Get the final solution
            pi3 = pi3_final
            model.setCalibration('pi3',pi3)
            cprint('\npi3 = {:.2f}'.format(pi3),"green")
            xi = xi_final
            model.setCalibration('xi',xi)
            cprint('xi = {:.2f}'.format(xi),"green")
            ns.TOLERANCE = 1.e-7
            results,dates = simulate(model=model,y0=y0,MULT=MULT,graph_info=False,model_info=False)
            
            cprint("Total CPU time: {:.1f} seconds.".format(timer()-t0),"red",attrs=["bold","underline"])
            
            if results is None:
                return
            
            rows,columns = results.shape
            if "range" in model.options:
                start_date = dates[SHIFT+1]
                dates = dates.to_pydatetime()
            else:
                dates = [(x-SHIFT+1)*4./52 for x in dates]
                start_date = None
            
            if LOCKDOWN and VACCINATION:
                highlight=[min((LOCKDOWN_START-SHIFT)/13.,(VACCINATION_START-SHIFT)/13.),
                           max((LOCKDOWN_START+LOCKDOWN_DURATION-SHIFT-2)/13.,(VACCINATION_START+VACCINATION_DURATION-SHIFT-2)/13.)]   
            elif LOCKDOWN:
                highlight=[(LOCKDOWN_START-SHIFT)/13.,(LOCKDOWN_START+LOCKDOWN_DURATION-SHIFT-2)/13.]
            elif VACCINATION:
                highlight=[(VACCINATION_START-SHIFT)/13.,(VACCINATION_START+VACCINATION_DURATION-SHIFT-2)/13.]
            else:
                highlight = None
            d = {}
            for j in range(columns):
                n = var_names[j]
                data = results[:,j] 
                ts = pd.Series(data[1:-1][:Tmax],dates[:Tmax])
                d[n] = ts 
            if "iw" in df:
                df = pd.concat([d["iw"],d["iv"],d["ivr"]])
                df.plot()
            
            
            # Retrieve shock to infected.   
            Tincome = 20 # years of income
            utility,cost_of_lives = integral(par["betta"],par["theta"],
                                            d["s"].values,d["i"].values,d["r"].values,
                                            d["cs"].values,d["ci"].values,d["cr"].values,
                                            d["ns"].values,d["ni"].values,d["nr"].values,
                                            Tincome,par["inc_target"])
            
            #cost_of_lives = Tincome*par["inc_target"]
            cost_of_lives *= -d["dd"].values[-1]
            cprint("Household lifetime utility {:.2f}".format(utility),"blue")
            cprint("Cost of lives lost {:.2f}".format(cost_of_lives),"blue")
            cprint("Total utility {:.2f}\n".format(utility+cost_of_lives),"blue")
            Z[ii,jj] = utility+cost_of_lives
            
            polycies.append([str(fl),str(fv),str(utility-cost_of_lives),str(utility),str(cost_of_lives)])
           
            if LOCKDOWN or VACCINATION:
                cprint("\n<----  Completed {:.0f}% of total tasks.\n".format(100.*(1+ii*n_vaccination+jj)/n_lockdown/n_vaccination),"blue")
           
            
    
    if LOCKDOWN: 
        lockdown_policy = pd.Series(100*lockdown_policy[:min(Tmax,len(lockdown_policy))],dates[:min(Tmax,len(lockdown_policy))])
    if VACCINATION: 
        vaccination_policy = pd.Series(100*vaccination_policy[:min(Tmax,len(vaccination_policy))],dates[:min(Tmax,len(vaccination_policy))])

    #Normalize macro variables
    d["y_n"]     = 100*(d["y"]/y_ss-1);          d["yF_n"]      = 100*(d["yF"]/y_ss-1)
    d["c_n"]     = 100*(d["c"]/c_ss-1);          d["cF_n"]      = 100*(d["cF"]/c_ss-1)
    d["w_n"]     = 100*(d["w"]/calib["w_ss"]-1); d["wF_n"]      = 100*(d["wF"]/calib["w_ss"]-1)
    d["n_n"]     = 100*(d["n"]/n_ss-1);          d["nF_n"]      = 100*(d["nF"]/n_ss-1)
    d["x_n"]     = 100*(d["x"]/x_ss-1);          d["xF_n"]      = 100*(d["xF"]/x_ss-1)
    d["rr_n"]    = 100*(d["rr"]**52-1);          d["rrF_n"]     = 100*(d["rrF"]**52-1)
    d["Rb_n"]    = 100*(d["Rb"]**52-1);          d["RbF_n"]     = 100*(d["RbF"]**52-1)
    d["k_n"]     = 100*(d["k"]/k_ss-1);          d["kF_n"]      = 100*(d["kF"]/k_ss-1)
    d["pie"]     = 100*(d["pie"]**52-1);         d["pieF"]      = 100*(d["pieF"]**52-1)
    d["cs_n"]    = 100*(d["cs"]/c_ss-1);         d["csF_n"]     = 100*(d["csF"]/c_ss-1)
    d["ci_n"]    = 100*(d["ci"]/c_ss-1);         d["ciF_n"]     = 100*(d["ciF"]/c_ss-1)
    d["cr_n"]    = 100*(d["cr"]/c_ss-1);         d["crF_n"]     = 100*(d["crF"]/c_ss-1)
    d["ns_n"]    = 100*(d["ns"]/n_ss-1);         d["nsF_n"]     = 100*(d["nsF"]/n_ss-1)
    d["ni_n"]    = 100*(d["ni"]/n_ss-1);         d["niF_n"]     = 100*(d["niF"]/n_ss-1)
    d["nr_n"]    = 100*(d["nr"]/n_ss-1);         d["nrF_n"]     = 100*(d["nrF"]/n_ss-1)
    d["r"]       = 100*d["r"];                   d["rF"]        = 100*d["rF"]
    d["i_total"] = 100*np.cumsum(d["tau"]);      d["iF_total"]  = 100*np.cumsum(d["tauF"])
    d["v_total"] = 100*np.cumsum(d["v"]);        d["vF_total"]  = 100*np.cumsum(d["vF"])
      
    y_min = np.min(d["y_n"]);                    #yF_min = np.min(d["y_n"]);
    c_min = np.min(d["c_n"]);                    #cF_min = np.min(d["cF_n"]);
    cs_min = np.min(d["cs_n"]);                  #csF_min = np.min(d["csF_n"]);
    ci_min = np.min(d["ci_n"]);                  #ciF_min = np.min(d["ciF_n"]);
    cr_min = np.min(d["cr_n"]);                  #crF_min = np.min(d["crF_n"]);
    n_min = np.min(d["n_n"]);                    #nF_min = np.min(d["nF_n"]);    
    ns_min = np.min(d["ns_n"]);                  #nsF_min = np.min(d["nsF_n"]);    
    ni_min = np.min(d["ni_n"]);                  #niF_min = np.min(d["niF_n"]); 
    nr_min = np.min(d["nr_n"]);                  #nrF_min = np.min(d["nrF_n"]); 
    i_max = 100*np.max(d["i"]);                  #iF_max = 100*np.max(d["iF"]);
    d_max = 100*np.max(d["dd"]);                 #dF_max = 100*np.max(d["ddF"]);
    
    if VACCINATION and LOCKDOWN:
        cprint("Vaccination and lockdown programs:","red",attrs=["bold","underline"])
    elif VACCINATION:
        cprint("Vaccination program:","red",attrs=["bold","underline"])
    elif LOCKDOWN:
        cprint("Lockdown program:","red",attrs=["bold","underline"])
    else:
        cprint("Baseline scenario:","red",attrs=["bold","underline"])
        
    cprint(f"Maximum number of infected {i_max:.1f}% and death {d_max:.2f}%","blue")
    cprint(f"Minimum output {y_min:.1f}%, aggregated consumption {c_min:.1f}% and work hours {n_min:.1f}%","blue")
    cprint(f"Minimum consumption of susceptible{cs_min:.1f}%, infectious {ci_min:.2f}% and recovered {cr_min:.2f}%","blue")
    cprint(f"Minimum working hours of susceptible{ns_min:.1f}%, infectious {ni_min:.2f}% and recovered {nr_min:.2f}%\n","blue")
    
    # Save aggregate working hours
    fpath = os.path.abspath(os.path.join(working_dir,"data/Dignar/inputs.xlsm"))
    getYearlySeries(d["n_n"],fpath,"Exogenous series","AP",SHIFT)

    # # Compute output loss
    # loss = 0.01*np.sum(d["y_n"])/52.
    # drop = 0.01*min(d["y_n"])
    # print(f'{par["theta_lockdown"]},{loss},{drop}\n')
    # fout = os.path.abspath(os.path.join(dir_path,'output_loss.csv'))
    # with open(fout, mode='a') as file:
    #     file.write(f'{par["theta_lockdown"]},{loss},{drop}\n')
           
            
    if Plot:
        
        list_headers = []
        
        # Compare SIR model forecast against data
        index = d["i"].index; T = min(Tmax,len(i_obs),len(index))
        i_obs = pd.Series(i_obs[:T],index[:T]); d_obs = pd.Series(d_obs[:T],index[:T])
        r_obs = pd.Series(r_obs[:T],index[:T]); s_obs = pd.Series(s_obs[:T],index[:T])
        i_total_obs = pd.Series(i_total_obs[:T],index[:T])
        header = 'Virus'
        list_headers.append(header)
        titles = ['Infected','Deaths']
        if "i1" in d and "i2" in d:
            if "range" in model.options:
                series = [[100*(d["i1"]+d["i2"])[start_date:],100*df["OBS_i"].dropna(),100*d["i1"][start_date:],100*d["i2"][start_date:]],[100*d["dd"][start_date:],100*df["OBS_d"].dropna()]]
            else:
                series = [[100*(d["i1"]+d["i2"])[1.e-6:T],i_obs[1.e-6:],100*d["i1"][1.e-6:T],100*d["i2"][1.e-6:T]],
                          [100*d["dd"][1.e-6:T],d_obs[1.e-6:]]]
            labels=[['Forecast','Active Cases','Strain 1','Strain 2'],['Forecast','Data']]
        else:
            if "range" in model.options:
                series = [[100*(d["i1"]+d["i2"]),100*df["OBS_i"].dropna()],[100*d["dd"],100*df["OBS_d"].dropna()]]
            else:
                series = [[100*(d["i1"]+d["i2"])[1.e-6:T],i_obs[1.e-6:]],
                          [100*d["dd"][1.e-6:T],d_obs[1.e-6:]]]
            labels=[['Forecast','Current Cases'],['Forecast','Data']]

        plotTimeSeries(path_to_dir=graphs_path,header=header,titles=titles,labels=labels,series=series,sizes=[1,2],stacked=False,fig_sizes=(10,5),save=save)


        # ERT model economy forecast
        if "range" in model.options:
            series = [[d["y_n"][start_date:],d["yF_n"][start_date:]],
                      [d["c_n"][start_date:],d["cF_n"][start_date:]],
                      [d["w_n"][start_date:],d["wF_n"][start_date:]],
                      [d["n_n"][start_date:],d["nF_n"][start_date:]],
                      [d["x_n"][start_date:],d["xF_n"][start_date:]],
                      [d["rr_n"][start_date:],d["rrF_n"][start_date:]],
                      [d["Rb_n"][start_date:],d["RbF_n"][start_date:]],
                      [d["k_n"][start_date:],d["kF_n"][start_date:]],
                      [d["pie"][start_date:],d["pieF"][start_date:]]
                 ]
        else:
            series = [[d["y_n"][1.e-6:],d["yF_n"][1.e-6:]],
                      [d["c_n"][1.e-6:],d["cF_n"][1.e-6:]],
                      [d["w_n"][1.e-6:],d["wF_n"][1.e-6:]],
                      [d["n_n"][1.e-6:],d["nF_n"][1.e-6:]],
                      [d["x_n"][1.e-6:],d["xF_n"][1.e-6:]],
                      [d["rr_n"][1.e-6:],d["rrF_n"][1.e-6:]],
                      [d["Rb_n"][1.e-6:],d["RbF_n"][1.e-6:]],
                      [d["k_n"][1.e-6:],d["kF_n"][1.e-6:]],
                      [d["pie"][1.e-6:],d["pieF"][1.e-6:]]
                 ]
        header = 'Economy'
        list_headers.append(header)
        titles = ['Output','Aggregate Consumption','Wage Rate', 'Aggregate Hours',
                  'Investment','Real Interest Rate','Policy Rate','Capital','Inflation'
                 ]
        
        labels=[['Sticky Price Economy','Flexible Price Economy'],[],[],[],[],[],[],[],[]]
    
        plotTimeSeries(path_to_dir=graphs_path,header=header,titles=titles,labels=labels,series=series,sizes=[3,3],highlight=highlight,stacked=False,fig_sizes=(10,10),save=save)
    
        
        # ERT model economy forecast (continued)
        if "range" in model.options:
            series = [[d["cs_n"][start_date:],d["csF_n"][start_date:]],[d["ci_n"][start_date:],d["ciF_n"][start_date:]],
                     [d["cr_n"][start_date:],d["crF_n"][start_date:]],[d["ns_n"][start_date:],d["nsF_n"][start_date:]],
                     [d["ni_n"][start_date:],d["niF_n"][start_date:]],[d["nr_n"][start_date:],d["nrF_n"][start_date:]],
                    ]
        else:
            series = [[d["cs_n"][1.e-6:],d["csF_n"][1.e-6:]],[d["ci_n"][1.e-6:],d["ciF_n"][1.e-6:]],
                     [d["cr_n"][1.e-6:],d["crF_n"][1.e-6:]],[d["ns_n"][1.e-6:],d["nsF_n"][1.e-6:]],
                     [d["ni_n"][1.e-6:],d["niF_n"][1.e-6:]],[d["nr_n"][1.e-6:],d["nrF_n"][1.e-6:]],
                    ]
        header = 'Economy (continued)'
        list_headers.append(header)
        titles = ['Consumption of Susceptibles','Consumption of Infected','Consumption of Recovered',
                  'Work Hours of Susceptibles ','Work Hours of Infected ','Work Hours of Recovered'
                 ]
        labels=[['Sticky Price Economy','Flexible Price Economy'],[],[],[],[],[]]
        plotTimeSeries(path_to_dir=graphs_path,header=header,titles=titles,labels=labels,series=series,sizes=[2,3],highlight=highlight,stacked=False,fig_sizes=(10,6),save=save)

    else:
        list_headers = ['SIR Model Forecast','Sticky and Flexible Prices Economies','Economies']
        
    if save:
        import matplotlib.pyplot as plt
        from matplotlib import cm
        from matplotlib.ticker import LinearLocator
        from mpl_toolkits.mplot3d import Axes3D
        from snowdrop.src.utils.merge import merge
        from snowdrop.src.utils.util import saveTimeSeries
        
        
        if Plot:
            outputFile = os.path.abspath(os.path.join(working_dir,"results/ERT.pdf"))
            files = []
            for f in list_headers:
                files.append(os.path.abspath((os.path.join(working_dir,"graphs/"+f+".pdf"))))
            merge(outputFile,files)
                      
            if n_lockdown > 1 and n_vaccination > 1:
                # Make mesh grid.
                X = np.arange(0.,1.,1./n_lockdown) # * theta_lockdown
                Y = np.arange(0.,1.,1./n_vaccination) # * vaccination_rate
                X, Y = np.meshgrid(X, Y)
                # Plot the surface.
                
                fig = plt.figure(figsize=(10,8))
                ax  = fig.add_subplot(111,projection='3d')
                surf = ax.plot_surface(X,Y,Z,cmap=cm.coolwarm,linewidth=0,antialiased=False)
                # Customize the z axis.
                ax.set_zlim(np.min(Z), np.max(Z))
                ax.zaxis.set_major_locator(LinearLocator(5))
                # A StrMethodFormatter is used automatically
                #ax.zaxis.set_major_formatter('{x:.0f}')
                # Add a color bar which maps values to colors.
                ax.set_ylabel('Lockdown')
                ax.set_xlabel('Vaccination')
                #ax.set_zlabel('Utility')
                plt.title('Utility Function')
                fig.colorbar(surf, shrink=0.5, aspect=5)
                plt.savefig(os.path.abspath(os.path.join(graphs_path,'Polycies.png')))
                plt.show(block=False)
                plt.close(fig)
        
        
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)     
        if LOCKDOWN and VACCINATION:
            fout = os.path.abspath(os.path.join(dir_path,list_headers[-1]+'_lockdown_vaccination.csv'))
        elif LOCKDOWN:
            fout = os.path.abspath(os.path.join(dir_path,list_headers[-1]+'_lockdown.csv'))
        elif VACCINATION:
            fout = os.path.abspath(os.path.join(dir_path,list_headers[-1]+'_vaccination.csv'))
        else:
            fout = os.path.abspath(os.path.join(dir_path,list_headers[-1]+'.csv'))
        d["s"] *= 100; d["r"] *= 100; d["i"] *= 100; d["dd"] *= 100; d["pop"] *= 100
        for k in d:
            d[k] = d[k].iloc[SHIFT-1:]
        saveTimeSeries(fname=fout,data=d)
        
        fout = os.path.abspath(os.path.join(dir_path,'polycies.csv'))
        with open(fout, mode='w') as file:
            file.write("Lockdown,Vaccination,Total Utility,Utility,Cost of Loss of Lives\n")
            for x in polycies:
                file.write(",".join(x)+"\n")
  
    print('Done with ERT model simulations!!!')


def SIR(pii,pir,pid,s0,i0,r0,dd0,pop0,lockdown_rate=None,vaccination_rate=None):
    """SIR model forecast."""
    global Nmax,mult,mult2
    T = Nmax
    tau = np.zeros(T+1); tau1 = np.zeros(T+1); tau2 = np.zeros(T+1)
    vaccinated = np.zeros(T+1)
    s  = np.zeros(T+1);   s[0] = s0
    i  = np.zeros(T+1);   i[0] = i0
    i1 = np.zeros(T+1);   i[0] = i0
    i2 = np.zeros(T+1);   i[0] = 0.1*i0
    r  = np.zeros(T+1);   r[0] = r0
    dd = np.zeros(T+1);   dd[0] = dd0
    pop = np.zeros(T+1);  pop[0] = pop0
    
    if fname == "ert1_model.yaml":
        for t in range(T):
            # Vaccinated
            vaccinated[t+1] = max(0,vaccinated[t] + vaccination_rate*s[t] - (pii*(1-sigma)*i1[t]+mult*pii*i2[t])*vaccinated[t])
            # Wildtype virus infection
            tau1[t+1] = pii*s[t]*i1[t]
            i1[t+1] = max(0,i1[t] + tau1[t+1] + pii*(1-sigma)*i1[t]*vaccinated[t]- (pir+pid)*i2[t])
            # Virus variant infection
            if t >= 7*virus_variant_start:
                tau2[t+1] = mult*pii*s[t]*i2[t] 
                i2[t+1]  = max(0,i2[t] + tau2[t+1] + mult*pii*i2[t]*vaccinated[t] - (pir+pid/mult2)*i2[t])
            else:
                i2[t+1] = i2[t]
            # Susceptible
            s[t+1] = max(0,s[t] - tau1[t+1] - tau2[t+1] - vaccination_rate*s[t])
            # Total infected
            i[t+1] = i1[t+1] + i2[t+1]
            # Recovered
            r[t+1] = r[t] + pir*i[t]
            # Deceased 
            dd[t+1] = dd[t] + pid*i1[t] + pid/mult2*i2[t]
    else:
            
        for t in range(T):
            if not vaccination_rate is None:
                newly_vaccinated = s[t]*vaccination_rate
            else:
                newly_vaccinated = 0
            if not lockdown_rate is None:
                lockdown = 1 - lockdown_rate
            else:
                lockdown = 1
                
            vaccinated[t+1] += vaccinated[t] + newly_vaccinated
            # New infections
            tau[t+1] = pii*s[t]*i[t] * lockdown**2
            # Susceptibles
            s[t+1] = s[t]-tau[t+1] - newly_vaccinated
            # Infected
            i[t+1] = i[t] + tau[t+1] - (pir+pid)*i[t]
            # Recovered
            r[t+1] = r[t] + pir*i[t] + newly_vaccinated
            # Total deaths 
            dd[t+1] = dd[t] + pid*i[t] 
            # Population
            pop[t+1] = pop[t] - pid*i[t]
        
    return tau[1:],s[1:],i[1:],r[1:],dd[1:],pop[1:],vaccinated[1:]


def func(x):
    """Calculate infection spread."""
    global Nmax,I_ini,D_ini,Pir,Pid,C_ss,N_ss
    global I,S,D,R,T
    
    # Preallocate arrays
    I=np.zeros(Nmax+1);S=np.zeros(Nmax+1);D=np.zeros(Nmax+1);R=np.zeros(Nmax+1);T=np.zeros(Nmax+1)

    # Initial guess
    pi1 = x[0]/scale1
    pi2 = x[1]/scale2
    pi3 = x[2]

    # Initial conditions
    I[0] = I_ini
    S[0] = 1-I[0]
    D[0] = D_ini
    R[0] = 0
    
    # Iterate SIR model equations
    for j in range(Nmax):
        T[j]   = pi1*S[j]*C_ss**2*I[j] + pi2*S[j]*N_ss**2*I[j] + pi3*S[j]*I[j]
        S[j+1] = min(1,max(0,S[j] - T[j]))
        I[j+1] = min(1,max(0,I[j] + T[j] - (Pir+Pid)*I[j]))
        R[j+1] = R[j] + Pir*I[j]
        D[j+1] = D[j] + Pid*I[j]
    
    return I,S,D,R,T


def func1(x):
    """Calibration function."""
    global Pi1_shr_target,Pi2_shr_target,Pi3_shr_target,RplusD_target,C_ss,N_ss,scale1,scale2,Model_R
    global I,S,D,R,T
    
    # Initial guess
    pi1 = x[0]/scale1
    pi2 = x[1]/scale2
    pi3 = x[2]
    if len(x)==5:
        RplusD_target = x[3]
        pi3_shr_target = x[4]
        share = 0.5
        Pi1_shr_target = (1-pi3_shr_target)*share
        Pi2_shr_target = (1-pi3_shr_target)*(1-share)
        Pi3_shr_target = pi3_shr_target
    
    # Calculate infection desease spread.
    func(x)
    
    err = np.zeros(3)
    # Calculate residuals for target equations
    err[0] = Pi1_shr_target-(pi1*C_ss**2)/(pi1*C_ss**2+pi2*N_ss**2+pi3)
    err[1] = Pi2_shr_target-(pi2*N_ss**2)/(pi1*C_ss**2+pi2*N_ss**2+pi3)
    err[2] = RplusD_target-(R[-1]+D[-1])
    
    Model_R = T[0]/I[0]/(Pir+Pid)
    return err


def func_squared(x):
    """Error function for constrained calibration."""
    y = func1(x)
    return np.sum(y*y)


def func2_squared(x):
    """Error function for constrained calibration."""
    global RplusD_target
    y = func1(x)
    err1 = np.sum(y*y)
    if I_OBS is None or D_OBS is None:
        err2 = 0
    else:    
        err2 = ( D_OBS-D[:len(D_OBS)])**2+(I_OBS-I[:len(I_OBS)])**2 
        #err2 += (R_OBS-R[:len(R_OBS)])**2+(S_OBS-S[:len(S_OBS)])**2
        #print(100*np.sqrt(err2))
    err2 = np.sum(err2)
    err  = err1 + err2
    return err


def calibrate(pi1,pi2,pi3,pi1_shr_target,pi2_shr_target):
    """Calibrate pi1, pi2 and pi3 using SIR model."""
    global Nmax,Pi1_shr_target,Pi2_shr_target,RplusD_target,scale1,scale2,Model_R 
    
    Pi1_shr_target = pi1_shr_target; Pi2_shr_target = pi2_shr_target

    x0 = np.array([pi1*scale1,pi2*scale2,pi3])
    #Unconstrained optimization
    #sol = root(func,x0,method='lm',tol=1e-7,options={"maxiter":1000})
    
    # Constrained optimization.
    # We assume that weekly infaction rate can not be smaller than the sum of
    # weekly recovery and death rates.
    bounds = Bounds([0,0,Pir+Pid],[1.,1.,np.inf])
    # ERT paper uses unconstrained optimization... 
    # So, uncomment this line to match ERT results.
    # bounds = Bounds([0,0,0],[1.,1.,np.inf])
    sol = minimize(func_squared,x0,method='trust-constr',bounds=bounds)
    
    pi  = sol.x
    pi1 = pi[0] / scale1
    pi2 = pi[1] / scale2
    pi3 = pi[2]
    err = abs(sol.fun)
    status = sol.status
            
    if not sol.success:
        cprint(f"\nCould not calibrate SIR model: error={err:.2e}, status={status}\n","red")
        sys.exit(-1)
    else:
        cprint(f"\nCalibrated SIR model: error={err:.2e}, status={status}","green")
        
    return (Model_R,pi1,pi2,pi3)


def integral(betta,theta,S,I,R,cs,ci,cr,ns,ni,nr,Ti,income):
    """Compute household lifetime utility.
    
     Cost of loss of lives is a discounted value of 'Ti' years of individual income.
     """
    utility = 0
    beta = 1
    T = min(len(I),len(ns))
    for t in range(T):
        utility += beta*(S[t]*(np.log(max(1e-10,cs[t]-0.5*theta*ns[t]**2))))
        utility += beta*(I[t]*(np.log(max(1e-10,ci[t]-0.5*theta*ni[t]**2))))
        utility += beta*(R[t]*(np.log(max(1e-10,cr[t]-0.5*theta*nr[t]**2))))
        beta *= betta
    utility /= 52
        
    cost_of_lives = 0
    beta = 1
    for t in range(52*Ti):
        cost_of_lives += beta * income
        beta *= betta
    cost_of_lives /= 52
        
    return utility,cost_of_lives
    

def getYearlySeries(series,fpath,sheet_name,column,startrow,save=False):
    """Convert weekly frequency to yearly frequency.
    
       This approximates shortage of labor supply due to COVID-19 effects.
       It is used in DIGNAR-19 toolkitf to calibrate model.
    """
    if os.path.exists(fpath):
        # Convert weekly frequency to daily
        index = series.index
        if isinstance(index,pd.DatetimeIndex):
            df = series[startrow:]
            weeks = list(range(len(df)))
        else:
            mask = index >= 0
            df = series[mask]
            index = list(df.index)
            weeks = [1+int(13*x) for x in df.index]
        
        values = df.values
        
        # Aggregate by year
        arr = []; lst = []; yprev = 0
        for i,w in enumerate(weeks):
            v = values[i]
            y = w//52
            if y-yprev >= 1:
                arr.append(np.array(lst))
                yprev = y
                lst = []
            lst.append(v)
            
        # Append the last batch
        if weeks[-1] % 52 > 0:
            arr.append(np.array(lst))
            
            
        agg = [np.min(x) for x in arr]
        print('\nMinimum yearly working hours drop:')
        print(pd.DataFrame(agg))
        
        agg = [np.mean(x) for x in arr]
        print('\nAverage yearly working hours drop:')
        print(pd.DataFrame(agg))
        
        if len(agg) < 10:
            agg += [0]*(10-len(agg))
        
        if save:
            # Save data
            from openpyxl import load_workbook
            
            try:
                wb = load_workbook(fpath)
                sh = wb[sheet_name]
                if not sh is None:
                    for row_num, data in enumerate(agg):
                        rng = column+str(startrow+row_num)
                        sh[rng] = data
                wb.save(fpath)
            finally:
                if not wb is None:
                    wb.close()
                
                
if __name__ == '__main__':
    """The main program."""
    main(Plot=True,save=True)
