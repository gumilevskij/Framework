#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 23:39:29 2021

@author: alexei
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.filters.hp_filter import hpfilter
    
numb = 0
path = os.path.dirname(os.path.abspath(__file__))
working_dir = os.path.abspath(os.path.join(os.path.abspath(path + "../../../..")))
os.chdir(working_dir)


def getData(start,end,country=None,save=True):
    """Read excel file."""
    file_path  = os.path.abspath(os.path.join(working_dir,'supplements/data/COVID19/Countries.xlsx'))
    xl = pd.ExcelFile(file_path)
    sheet_names = xl.sheet_names  # see all sheet names
    dbs = {}; index = None
    
    fpath = os.path.abspath(os.path.join(working_dir,"supplements/data/COVID19/country_data.xlsx"))
    writer = pd.ExcelWriter(fpath, engine='xlsxwriter')
    
    for sht in sheet_names:
        if not country is None and not sht == country:
            continue
        print(sht)
        df = xl.parse(sheet_name=sht,header=0,parse_dates=True)
        df = df.iloc[7:]
        columns = list(df.columns)
        ih = columns.index("Haver")/4
        var_names = [x for i,x in enumerate(columns) if i%4==0]
        var_names = var_names[:9] # Take only the first nine data series
        db = {}; ls = []; lc = []; ii = 0
        for i,name in enumerate(var_names):
            print("   " + name)
            if i < ih:
                ser = df.iloc[:,[ii+1,ii+2]]
            else:
                ser = df.iloc[:,[ii+1,ii+3]]
            ser.set_index(columns[ii+1],inplace=True)
            ser = ser.dropna()
            if name == "Consumption Expenditure":
                filtered = hpfilter(x=ser.values,lamb=1600)
                values = np.squeeze(ser.values)/filtered[1] -1
                ser = pd.Series(100*values,ser.index)
                ser.rename(name,inplace=True)
            try:
                if i < ih:
                    ser.index = pd.to_datetime(ser.index)
                else:
                    indx = [x[3:]+x[:2] for x in ser.index]
                    ser.index = pd.to_datetime(indx)
                index = ser.index
                ser.columns = [name]
            except:
                if name.endswith(".M") or name == "Consumer Prices Change":
                    # Convert monthly data to quarterly
                    ser.index = pd.to_datetime(ser.index,format="%YM%m")   
                    ser = ser.resample('Q',convention='start').first()
                    ser.index = pd.to_datetime(ser.index)
                    ser.columns = [name]
                else:    
                    # Convert yearly data to quarterly
                    ser.index = pd.to_datetime(ser.index.astype(int),format="%Y")
                    values = []
                    for x in ser.values:
                        values.append(x[0])
                        values.append(np.nan)
                        values.append(np.nan)
                        values.append(np.nan)
                        sz = min(len(values),len(index))
                    ser = pd.DataFrame(values[:sz],index[:sz])
                    ser.interpolate(method="slinear",limit_direction="forward",inplace=True)
                    ser.columns = [name]
            
            if name=="Inflation":
                val = ser.values
                val = annualize(val)
                values = filter_solution(val,0.5)
                ser = pd.DataFrame(values,index)
                ser.columns = [name]
            
            ser = ser[start:end]
                
            db[name] = ser
            ls.append(ser)
            lc.append(name)
            
            ii += 4
                
        if save:
            df_all = pd.concat(ls,axis=1)
            df_all = df_all.rename(columns={"Output Gap":"OBS_y","Unemployment":"OBS_unempl","Inflation":"OBS_pinf","Interest Rate":"OBS_r","Investment":"OBS_inve","Consumption Expenditure":"OBS_c","Long Term Bond Yield":"OBS_yield","Consumption Growth":"OBS_cgr","GDP":"OBS_gdp"})
            df_all.to_excel(writer,sheet_name=sht)

        dbs[sht] = db
        
    # Close the Pandas Excel writer and output the Excel file.
    writer.close()
    
    return dbs


def getData2(start,end,sht,save=True):
    """Read excel file."""
    db = {}
    file_path  = os.path.abspath(os.path.join(working_dir,'supplements/data/COVID19/Countries.xlsx'))
    df = pd.read_excel(file_path,sht,parse_dates=True)
    dt = df["Date"][6:]
    dt = pd.to_datetime(dt)
    v = df["Output Gap.1"][6:]
    output_gap = pd.Series(v.values,dt).dropna()
    db["Output Gap"] = output_gap[start:end]
    
    #v = df["ID:WEO_PUBLISHED.6"][6:]
    #consumption_growth = pd.Series(v.values,dt).dropna()
    #db["Consumption Growth"] = consumption_growth[start:end]
    
    v = df["Consumer Prices Change"][6:]
    consumption_gap = pd.Series(v.values,dt).dropna()
    db["Consumption Gap"] = 100*consumption_gap[start:end]
    
    v = df["ID:WEO_PUBLISHED.1"][6:]
    unemployment = pd.Series(v.values,dt).dropna()
    db["Unemployment"] = unemployment[start:end]
    
    v = df["ID:WEO_PUBLISHED.2"][6:]
    v = annualize(v.values)
    v = filter_solution(v,0.5)
    inflation = pd.Series(v,dt[:len(v)]).dropna()
    db["Inflation"] = inflation[start:end]
    
    v = df["Average Hours at Work"][2:]
    hours = pd.Series(v.values[:len(dt)],dt).dropna()
    db["Work Hours"] = hours[start:end]
    
    # Convert yearly data to quarterly
    dt = df["EcDatabase.3"][6:]
    dt = [x if np.isnan(x) else "12/31/"+str(x) for x in dt]
    dt = pd.to_datetime(dt)
    v = df["ID:WEO_PUBLISHED.3"][6:]
    ir = pd.Series(v.values,dt).dropna()
    #ir.index = pd.to_datetime(ir.index.astype(int),format="%Y")
    db["Interest Rate"] = ir[start:end]
    
    dbs = {sht:db}
    return dbs
          
            
def make_space_above(fig,topmargin=1):
    """ increase figure size to make topmargin (in inches) space for 
        titles, without changing the axes sizes"""
    
    s = fig.subplotpars
    w, h = fig.get_size_inches()

    figh = h - (1-s.top)*h  + topmargin
    fig.subplots_adjust(bottom=s.bottom*h/figh, top=1-topmargin/figh)
    fig.set_figheight(figh)
           
        
def plot_data(db,labels,save=True,size=[3,3],show=True):
    global numb
    numb += len(db)
    """Plots data."""
    ms,ns = size
    for k in db:
        fig = plt.figure(figsize=(8,8))
        fig.suptitle(k,fontsize='30')
        d = db[k]
        m = 0
        for n in d:
            series = d[n]
            if len(series) == 0:
                continue
            m += 1
            if m > 9: 
                continue
            ax = plt.subplot(ms,ns,m)
            series.plot(ax=ax,lw=3,color='b',legend=None)
            plt.box(True)
            plt.grid(True)
            plt.xlabel('')
            plt.ylabel(labels[m-1])
            plt.title(n,fontsize = 'x-large')
            plt.tight_layout()
            
        make_space_above(fig)
        
        if save:
            path_to_dir = os.path.join(working_dir,"graphs")
            plt.savefig(os.path.join(path_to_dir,k+'_'+str(numb)+'.png'))
        if show: 
            plt.show(block=False)
            plt.close(fig)


def annualize(data):
    """Annualizes growth rate."""
    xp = 1
    x = []
    for v in data:
        if np.isscalar(v):
            xp *= 1+v/100
        else:
            xp *= 1+v[0]/100
        if not np.isnan(xp):
            x.append(xp)
    n = len(x)
    arr = np.zeros(n)
    s = 0
    for i in range(4,n):
        arr[i] = 100*x[i]/x[i-4] - 100
        s += arr[i]
    arr[:4] = s/(n-4)
        
    return arr
    

def filter_solution(data,lmbd=1600):
    """Filter data."""
    from snowdrop.src.numeric.filters.filters import LRXfilter as lrx
    # from scipy import signal
    # import statsmodels.api as sm
    
    y = lrx(y=data,lmbd=lmbd)[0]
    # b, a = signal.butter(N=5,Wn=0.1)
    # zi = signal.lfilter_zi(b, a)
    # f,zo = signal.lfilter(b, a, data, zi=zi*data[0])
      
    return y

    
if __name__ == '__main__':
    """The main entry point."""
    
    db = getData2("2006-01-01","2023-12-01","US")
    labels = ['Percent']*4 + [''] + ['Percent']
    plot_data(db,labels,size=[3,2])
    
    db = getData("2000","2023","US")  # US France Germany Italy France
    labels = ['Percent'] + [''] + ['Percent']*4+['']+['Percent']*2
    plot_data(db,labels,size=[3,3])
    