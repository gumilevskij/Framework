#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 17:01:29 2021

@author: alexei
"""
import os, sys
import pandas 
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from scipy.optimize import root
from scipy.optimize import minimize,Bounds

working_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),"../../.."))
os.chdir(working_dir)

from snowdrop.src.misc.termcolor import cprint

I=None; R=None; S=None; DD=None
v=None;iw=None;iv=None;ivr=None;rw=None;rv=None

# Initial conditions
Standard = False
N=800; i0=1.e-5; s0=1; r0=0; d0=0; pop0=1

#pi=0.05; pr=0.04; pd=0.0003
pi=0.39/7; pr=0.3/7; pd=0.0017/7 # first strain
pi2=2.6*pi; pd2=pd/3; sigma=0.e-2; theta=0.9 # second strain
pv = 0.0014/7 # Vaccination rate per day.
pmu = 0. # Probability of getting virus strain once vaccinated.
start = 400  # Time when delta variant virus kicks in.

def Virus(x):
    if Standard:
        return SIR(x)   # SIR model
    else:
        #return SIRS(x)  # SIR model with wildtype and resistant virus strain
        return SIRV(x)  # SIR model with vaccination and wildtype and resistant virus strain
        # return SIRVS(x) # SIR model with vaccination and wildtype and resistant virus strain

    
def SIR(x):
    """SIR model."""
    pi,pr,pd = x
    tau = np.zeros(N)
    s = np.zeros(N);   s[0] = s0
    i = np.zeros(N);   i[0] = i0
    r = np.zeros(N);   r[0] = r0
    d = np.zeros(N);   d[0] = d0
    pop = np.zeros(N); pop[0] = pop0
            
    for t in range(N-1):
        # New infections
        tau[t+1] = pi*s[t]*i[t]
        # Total susceptibles
        s[t+1] = max(0,s[t]-tau[t+1]-pv*s[t])
        # Total infected
        i[t+1] = max(0,i[t] + tau[t+1] - (pr+pd)*i[t])
        # Total recovered
        r[t+1] = r[t] + pr*i[t] + pv*s[t]
        # Total deaths 
        d[t+1] = d[t] + pd*i[t] 
        # Total population
        pop[t+1] = max(0,pop[t] - pd*i[t])
        
    return tau,s,i,r,d,pop


def SIRS(x):
    """SIR model with wildtype and resistant virus strain."""
    pi,pr,pd = x
    tau_v = np.zeros(N); tau_s = np.zeros(N)
    s = np.zeros(N);   s[0] = s0
    i = np.zeros(N)
    i_v = np.zeros(N); i_v[0] = i0
    i_s = np.zeros(N); i_s[start] = 0.1*i0        
    r_v = np.zeros(N); r_v[0] = r0     
    r_s = np.zeros(N); r_s[0] = r0
    d = np.zeros(N);   d[0] = d0
    pop = np.zeros(N); pop[0] = pop0
            
    for t in range(N-1):
        # New wildtype infections
        tau_v[t+1] = pi*s[t]*i_v[t]
        i_v[t+1] = max(0,i_v[t] + tau_v[t+1] - (pr+pd)*i_v[t])
        r_v[t+1] = r_v[t] + pr*i_v[t]
        # Virus strain infected
        if t >= start:
            tau_s[t+1] = pi2*s[t]*i_s[t]
            i_s[t+1]  = max(0,i_s[t] + tau_s[t+1] - (pr+pd2)*i_s[t])
            r_s[t+1] = r_s[t] + pr*i_s[t]
        # Total susceptibles
        s[t+1] = max(0,s[t]-tau_v[t+1]-tau_s[t+1])
        # Total infected
        i[t+1] = i_v[t+1] + i_s[t+1]
        # Total deaths 
        d[t+1] = d[t] + pd*i_v[t] + pd2*i_s[t]
        # Total population
        pop[t+1] = max(0,pop[t] - pd*i_v[t] - pd2*i_s[t])
        
    return tau_v+tau_s,s,i,r_v+r_s,d,pop


def SIRV(x):
    """SIR model with two virus strains and vaccination."""
    pi,pr,pd = x
    v = np.zeros(N)
    tau_v = np.zeros(N); tau_s = np.zeros(N)
    s = np.zeros(N);   s[0] = s0
    i = np.zeros(N)
    i_v = np.zeros(N); i_v[0] = i0
    i_s = np.zeros(N); i_s[start] = 0.1*i0        
    r = np.zeros(N); r[0] = r0     
    d = np.zeros(N);   d[0] = d0
    pop = np.zeros(N); pop[0] = pop0
            
    for t in range(N-1):
        # Newly vaccinated
        v[t] = pv*s[t]
        # Wildtype virus infected population
        tau_v[t+1] = pi*s[t]*i_v[t]
        i_v[t+1] = max(0,i_v[t] + tau_v[t+1] - (pr+pd)*i_v[t])
        # Virus variant infected population
        if t >= start:
            tau_s[t+1] = pi2*s[t]*i_s[t]
            i_s[t+1]  = max(0,i_s[t] + tau_s[t+1] - (pr+pd2)*i_s[t])
        # Susceptible
        s[t+1] = max(0,s[t] + sigma*r[t] - tau_v[t+1] - tau_s[t+1] - v[t])
        # Total infected
        i[t+1] = i_v[t+1] + i_s[t+1]
        # Recovered
        r[t+1] = r[t] + pr*i[t] - sigma*r[t] + v[t]
        # Deceased 
        d[t+1] = d[t] + pd*i_v[t] + pd2*i_s[t]
        # Total population
        pop[t+1] = max(0,pop[t] - pd*i_v[t] - pd2*i_s[t])
    #total = s+i+v+r+d
        
    return tau_v+tau_s,s,i,r,d,pop

def SIRVS(x):
    """SIR model with wildtype and resistant virus strain."""
    global v,iw,iv,ivr,rw,rv
    pi,pr,pd = x
    pmu = 0
    tau = np.zeros(N)
    s  = s0 + np.zeros(N)
    iw = i0 + np.zeros(N)   # wildtype virus infected
    iv = np.zeros(N)        # resitant virus strain infected
    ivr = np.zeros(N)       # vaccinated that become infected again by resitant virus strain 
    i  = iw + iv + ivr
    rw = r0 + np.zeros(N)
    rv = np.zeros(N)
    r  = rw + rv
    v  = np.zeros(N)
    d  = d0 + np.zeros(N)
    pop = pop0 + np.zeros(N)
            
    for t in range(N-1):
        # New infections
        tau[t+1] = pi*s[t]*iw[t] + pi2*s[t]*(iv[t]+ivr[t])
        # Total susceptibles
        s[t+1] = s[t] + pmu*rw[t] - pv*s[t] - (pi*iw[t]+pi2*(iv[t]+ivr[t]))*s[t]
        # Wildtype infected
        iw[t+1] = max(0,iw[t] - (pr+pd)*iw[t] + pi*s[t]*iw[t])
        # Virus strain infected
        if t == start:
            iv[t]  = 0.1*i0
            ivr[t] = 0.1*i0 
            iv[t+1]  = max(0,iv[t]  - (pr+pd)*iv[t]  + pi2*s[t]*(iv[t]+ivr[t]))
            ivr[t+1] = max(0,ivr[t] - (pr+pd)*ivr[t] + pi2*v[t]*(iv[t]+ivr[t]))
        elif t > start:
            iv[t+1]  = max(0,iv[t]  - (pr+pd)*iv[t]  + pi2*s[t]*(iv[t]+ivr[t]))
            ivr[t+1] = max(0,ivr[t] - (pr+pd)*ivr[t] + pi2*v[t]*(iv[t]+ivr[t]))
        else:
            iv[t+1]  = 0
            ivr[t+1] = 0            
        # Total infected
        i[t+1] = iw[t+1] + iv[t+1] + ivr[t+1]
        # Wildtype recovered
        rw[t+1] = max(0,rw[t] - pmu*rw[t] + pr*(iw[t]+iv[t]))
        # Virus strain recovered
        rv[t+1] = max(0,rv[t] - pmu*rv[t] + pr*ivr[t])
        # Total recovered
        r[t+1] = rw[t] + rv[t]
        # Total deaths 
        d[t+1] = d[t] + pd*i[t] 
        # Vaccinated
        v[t+1] = max(0,v[t] + pmu*rv[t] + pv*s[t] - pi2*v[t]*(iv[t]+ivr[t]))
        # Total population
        pop[t+1] = max(0,pop[t] - pd*i[t])
        
    return tau,s,i,r,d,pop
    

def func(x):
    """Calibration function."""
    tau,s,i,r,d,pop = Virus(x)
    
    err1 = np.sum(100*i[:T]-I.values)
    err2 = 0
    err3 = np.sum(100*d[:T]-DD.values)
    
    # Calculate residuals
    err = np.array([err1,err2,err3])
    return err


def func_squared(x):
    """Error function for constrained calibration."""
    y = func(x)
    return np.sum(y*y)


def getData(start,end,country=None,save=True):
    """Read data files."""
    db = {}
    pop={"Germany":83.8*10**6,"US":331*10**6,"France":67*10**6,"Italy":60.5*10**6, "Spain":46.8}
    population = pop[country]

    file_path  = os.path.abspath(os.path.join(working_dir,'supplements/data/COVID19/US_data.xlsx'))
    df = pandas.read_excel(file_path,index_col=0,parse_dates=True)
    index = pandas.to_datetime(df.columns)
    data = df.iloc[1].values.astype('float64') / population
    Itotal = pandas.Series(data,index)
    data = df.iloc[0].values.astype('float64') / population
    I = pandas.Series(data,index)
    data = df.iloc[3].values.astype('float64') / population
    D = pandas.Series(data,index)
    data = df.iloc[4].values.astype('float64') / population
    R = pandas.Series(data,index)
    #R = Itotal - I - D
      
    ind1 = I.index >= dt.datetime.strptime(start,"%Y-%m-%d")
    ind2 = I.index < dt.datetime.strptime(end,"%Y-%m-%d")
    I = I[ind1*ind2]
    Itotal = Itotal[ind1*ind2]
    
    ind1 = D.index >= dt.datetime.strptime(start,"%Y-%m-%d")
    ind2 = D.index < dt.datetime.strptime(end,"%Y-%m-%d")
    D = D[ind1*ind2]
    
    ind1 = R.index >= dt.datetime.strptime(start,"%Y-%m-%d")
    ind2 = R.index <= dt.datetime.strptime(end,"%Y-%m-%d")
    R = R[ind1*ind2]
    
    S  = 1 - Itotal - D
    #I -= R + D
    
    # Get daily infection and death rates.
    # beta = np.diff(I)/I.iloc[1:]/S.iloc[1:]
    # gamma = np.diff(D)/I.iloc[1:]
    # delta = np.diff(R)/I.iloc[1:]
    beta = -(Itotal-Itotal.shift(-7))/Itotal.shift(-3)/S.shift(-3)
    beta[beta<0] = 0
    gamma = (D.shift(-7)-D)/I.shift(-3)
    delta = (R.shift(-7)-R)/I.shift(-3)
    delta[delta<0] = 0
    delta2 = (R.shift(-7)-R)/(I.shift(-7)-I)
    # delta2 = np.diff(R)/np.diff(I); 
    # delta2 = pandas.Series(delta2,I.index[1:])
    Ro = beta/(gamma+delta)
    Ro[Ro<0] = np.nan
    Ro[Ro>3] = np.nan
    beta = beta.dropna()
    gamma = gamma.dropna()
    delta = delta.dropna()
    delta2 = delta2.dropna()
    Ro = Ro.dropna()
    
    cycle_Ro, trend_Ro = filter_solution(Ro.values)
    series_Ro_trend = pandas.Series(trend_Ro,Ro.index)
    #series_Ro_cycle = pandas.Series(cycle_Ro,Ro.index)
    
    cycle_infected_total, trend_infected_total = filter_solution(I.values)
    series_infected_total_trend = pandas.Series(trend_infected_total,I.index)
    #series_infected_total_cycle = pandas.Series(cycle_infected_total,I.index)
    
    cycle_infected, trend_infected = filter_solution(I.values)
    series_infected_trend = pandas.Series(trend_infected,I.index)
    #series_infected_cycle = pandas.Series(cycle_infected,I.index)
    
    cycle_recovered, trend_recovered = filter_solution(R.values)
    series_recovered_trend = pandas.Series(trend_recovered,R.index)
    #series_recovered_cycle = pandas.Series(cycle_recovered,R.index)
    
    cycle_suspected, trend_suspected = filter_solution(S.values)
    series_suspected_trend = pandas.Series(trend_suspected,S.index)
    #series_suspected_cycle = pandas.Series(cycle_suspected,S.index)
    
    cycle_deceased, trend_deceased = filter_solution(D.values)
    series_deceased_trend = pandas.Series(trend_deceased,D.index)
    #series_deceased_cycle = pandas.Series(cycle_deceased,D.index)
    
    cycle_delta, trend_delta = filter_solution(delta.values)
    series_delta_trend = pandas.Series(trend_delta,delta.index)
    #series_delta_cycle = pandas.Series(cycle_delta,delta.index)
    
    #cycle_delta2, trend_delta2 = filter_solution(delta2.values)
    #series_delta2_trend = pandas.Series(trend_delta2,delta2.index)
    #series_delta2_cycle = pandas.Series(cycle_delta2,delta2.index)
    
    cycle_beta, trend_beta = filter_solution(beta.values)
    series_beta_trend = pandas.Series(trend_beta,beta.index)
    #series_beta_cycle = pandas.Series(cycle_beta,beta.index)
    
    cycle_gamma, trend_gamma = filter_solution(gamma.values)
    series_gamma_trend = pandas.Series(trend_gamma,gamma.index)
    #series_gamma_cycle = pandas.Series(cycle_gamma,gamma.index)
    
    db[r"$\beta$"] = [100*beta,100*series_beta_trend]
    db[r"$\gamma$"] = [100*gamma,100*series_gamma_trend]
    db[r"$\mu$"] = [100*delta,100*series_delta_trend]
    #db[r"$\mu_2$"] = [100*delta2,100*series_delta2_trend]
    db["Basic Reproduction Number"] = [Ro,series_Ro_trend]
    db["Infected"] = [100*I,100*series_infected_trend]
    db["Total Infected"] = [100*Itotal,100*series_infected_total_trend]
    db["Recovered"] = [100*R,100*series_recovered_trend]
    db["Susceptibles"] = [100*S,100*series_suspected_trend]
    db["Deaths"] = [100*D,100*series_deceased_trend]
    
    m = {"Beta":series_beta_trend,"Gamma":series_gamma_trend,"Delta":series_delta_trend,"OBS_i":I,"OBS_itotal":Itotal,"OBS_d":D,"OBS_r":R}
    df = pandas.DataFrame(m)
    # Convert daily frequency to weekly 
    df = df.resample("W").mean()
    df.to_csv(os.path.join(working_dir,"supplements/data/COVID19/epidemic.csv"))
    #print(df)
    
    return db
      
        
def plot_data(db,country,fname=None,save=True,show=True,legend=None,sizes=[1,1]):
    """Plots data."""
    m = 0
    r,c = sizes
    fig, axes = plt.subplots(r,c,figsize=(8,8))
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    for k in db:
        m += 1
        if m <= r*c:
            ax = plt.subplot(r,c,m)
            series = db[k]
            for i,ser in enumerate(series):
                lw = 2 if i < 2 else 1
                ser.plot(ax=ax,lw=lw)
            plt.box(True)
            plt.grid(True)
            plt.title(k,fontsize=20)
            plt.rc('xtick', labelsize=15) 
            plt.rc('ytick', labelsize=15) 
            plt.ylabel("Percent",fontsize=15)
            if not legend is None:
                plt.legend(legend,fontsize=15)
            plt.tight_layout()
            plt.subplots_adjust(top=0.85)
    #plt.suptitle(country,fontsize=30)
            
    if save:
        path_to_dir = os.path.join(working_dir,"graphs")
        plt.savefig(os.path.join(path_to_dir,fname))
    if show: 
        plt.show(block=False)


def filter_solution(data,lmbd=1600):
    """Filter data."""
    from statsmodels.tsa.filters.hp_filter import hpfilter
    y = hpfilter(data,lmbd)
      
    return y


def calibrate(x0):
    """Calibrate pi1, pi2 and pi3 using SIR model."""
   
    #Unconstrained optimization
    sol = root(func,x0,method='lm',tol=1e-7,options={"maxiter":1000})
    x0 = sol.x
    
    # Constrained optimization.
    # We assume that weekly infaction rate can not be smaller than the sum of
    # weekly recovery and death rates.
    #bounds = Bounds([0,0,0],[1,1,1]) if len(x0)==3  else Bounds([0,0,0,-1],[1,1,1,1])
    #sol = minimize(func_squared,x0,method='trust-constr',bounds=bounds)
    
    err = abs(sol.fun)
    status = sol.status
            
    if not sol.success:
        cprint(f"\nCould not calibrate SIR model: error={err}, status={status}\n","red")
    else:
        cprint(f"\nCalibrated SIR model: error={np.max(err):.2e}, solution={sol.x}","green")
        
    return sol.x

    
if __name__ == '__main__':
    """The main entry point."""
    
    country = "US"
    db = getData("2020-8-1","2023-1-1",country)  
    plot_data(db,country,fname=country+"_data1.png",legend=["Data","HP Filter"],sizes=[2,1])
    
    beta = db[r"$\beta$"][0]
    gamma = db[r"$\gamma$"][0]
    delta = db[r"$\mu$"][0]
    #delta2 = db[r"$\mu_2$"][0]
    I = db["Infected"][0]
    R = db["Recovered"][0]
    S = db["Susceptibles"][0]
    DD = db["Deaths"][0]
    i0 = 0.01*I.values[0]
    s0 = 0.01*S.values[0]
    r0 = 0.01*R.values[0]
    d0 = 0.01*DD.values[0]
    T = len(I)
    
    # Calibrate model
    x = x0 = [pi,pr,pd]
    #x = calibrate(x0)
    
    tau,s,i,r,dd,pop = Virus(x)
    index = pandas.date_range(start=beta.index[0],periods=N,freq="D")
    tau = 100*pandas.Series(tau,index,dtype=float)
    s = 100*pandas.Series(s,index,dtype=float)
    i = 100*pandas.Series(i,index,dtype=float)
    r = 100*pandas.Series(r,index,dtype=float)
    dd = 100*pandas.Series(dd,index,dtype=float)
    pop = 100*pandas.Series(pop,index,dtype=float)
    if Standard:     
        m = {"Infected":[i,I],"Deaths":[dd,DD]} 
        legend=["SIR Model","Data"]
    else:     
        v = 100*pandas.Series(v,index,dtype=float)
        iw = 100*pandas.Series(iw,index,dtype=float)
        iv = 100*pandas.Series(iv,index,dtype=float)
        ivr = 100*pandas.Series(ivr,index,dtype=float)
        m = {"Infected":[i,I,iw,iv],"Deaths":[dd,DD]}
        legend=["Forecast","Data"]
        
    plot_data(m,country,fname=country+".png",legend=legend,sizes=[2,1])
    
    print(f"Strain #1, Ro = {pi/(pr+pd):.2f}")
    print(f"Strain #2, Ro = {pi2/(pr+pd2):.2f}")
    
    