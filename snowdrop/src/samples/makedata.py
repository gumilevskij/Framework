# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 14:12:20 2019

@author: Alexei Goumilevski

 Reproduces results of  https://www.edx.org Monetary Policy Analysis and Forecasting course.
"""
import os
import pandas as pd
import numpy as np
from math import ceil
import matplotlib.pyplot as plt
from datetime import datetime as dt
import ruamel.yaml as yaml

working_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),"../../.."))
os.chdir(working_dir)

from snowdrop.src.numeric.filters.filters import HPF as hpf
from snowdrop.src.numeric.filters.filters import BPASS as bpass
from snowdrop.src.numeric.filters.filters import LRXfilter as lrx
from snowdrop.src.numeric.sa.x13 import x13
from snowdrop.src.graphs.util import plotTimeSeries
from snowdrop.src.utils.load import read_file_or_url
 

bin_dir = os.path.abspath(os.path.join(working_dir,"bin"))
if os.path.exists(bin_dir):
    sys_path = os.environ['PATH'] 
    if not bin_dir in sys_path:
        os.environ['PATH'] += ":" + bin_dir
    os.environ['X13PATH'] = bin_dir

    
def makedata(Plot=False,save=False):
    """
    Prepares data.
    """
    plt.close('all')
    
    path_to_dir = os.path.abspath(os.path.join(working_dir,"graphs"))
    
    ## Load quarterly data
    file_path = os.path.abspath(os.path.join(working_dir, "supplements/data/MPAF/data.csv"))
    df = pd.read_csv(filepath_or_buffer=file_path,sep=',',header=0,index_col=0,parse_dates=True,infer_datetime_format=True)
    df.index = pd.to_datetime(df.index,format="#Q")  # Quartely data frequency
    
    # Load quarterly data
    d = {} 
    for c in df.columns:
        d[c] = df[c]
          
    ## Seasonal adjustment
    count = 0
    lst = []
    for name in df.columns:
        if len(name)>1:
            if '_U' == name[-2:]:
                count += 1
                series = df[name].dropna()
                df_adj = x13(series=series,freq='Q')[0]
                d[name[:-2]] = df_adj["Adjusted"]
                lst.append((series,df_adj))
                
    # GDP,_ = hpf(data=d["GDP"], lmbd=10)
    # d["GDP"] = pd.Series(GDP,d["GDP"].index)
    
    ## Make log of variables
    exceptions = ['RS','RS_RW','D4L_CPI_TAR']
    m = {}
    for n in d:
        if not n in exceptions:
            m['L_'+n] = 100*np.log(d[n])
    d = {**d, **m}

    ##  Define the real exchange rate
    d["L_Z"] = d["L_S"] + d["L_CPI_RW"] - d["L_CPI"]

    ## Growth rate qoq, yoy
    m = {}
    for n in d:
        if n.startswith('L_'):
            m['DLA_'+n[2:]] = 4*d[n].diff(periods=1)
            m['D4L_'+n[2:]] = d[n].diff(periods=4)
    d = {**d, **m}

    ##  Real variables
    ##  Domestic real interest rate
    d["RR"] = d["RS"] - d["D4L_CPI"]

    ##Foreign real interest rate
    d["RR_RW"] = d["RS_RW"] - d["D4L_CPI_RW"]

    ## Trends and Gaps - Hodrick-Prescott filter
    ## Even seasonally adjusted HDP is very volatile
    var = ['RR','L_Z','RR_RW'];
    m = {}
    for n in var:
        m[n+'_BAR'],d[n+'_GAP'] = hpf(data=d[n], lmbd=1600)
    d = {**d, **m}
        
    ## Trend and Gap for Output - Band-pass filterP_GAP_HP'] = d['L_GDP_GAP']      
    #  Band-pass
    d['L_GDP_GAP'] = bpass(d['L_GDP'],6.0,32.0) 
    d['L_GDP_BAR'] = d['L_GDP'] - d['L_GDP_GAP']
    d['DLA_GDP_BAR'] = 4*d['L_GDP_BAR'].diff()
    
    ## Foreign Output gap - HP filter with judgements
    d["L_GDP_RW_BAR_PURE"], d["L_GDP_RW_GAP_PURE"] = hpf(d["L_GDP_RW"],lmbd=1600)
    d["DLA_Z_BAR"], _ = hpf(d["DLA_Z"],lmbd=1000)
    
    ## Expert judgement on the foreign output gap
    # Make sure that the last 5-6 observations by the HP filter correspond  
    # to World Economic Outlook (WEO) etc. "Bad" values will compromise the kalman filter results.
    # Override if necessary using WEO, and so on:
    
    start = dt.strptime("2011-1-1","%Y-%m-%d")
    end = dt.strptime("2013-12-1","%Y-%m-%d")
    rng = pd.date_range(start=start,end=end,freq='QS')
    data = [-1,-0.9,-1.3,-1.6,-2,-2.1,-2.3,-2.7,-3 ,-3.2,-3.4,-3.6]
    JUDGEMENT = pd.Series(data, rng)
    prior = d["L_GDP_RW"].dropna()
    prior[rng] -= JUDGEMENT
    d["L_GDP_RW_BAR"],d["L_GDP_RW_GAP"] = lrx(y=d["L_GDP_RW"].dropna(),lmbd=1600,w3=1,prior=prior)
    
    
    ### Read model file
    fmodel = os.path.abspath(os.path.join(working_dir,"supplements/models/MPAF/model.yaml")) 
     
    txt,_ = read_file_or_url(fmodel)
    data = yaml.load(txt, yaml.RoundTripLoader)
    mv = data["symbols"]["measurement_variables"]
    var = [x[4:] for x in mv] 
    
    ## Save the database
    from snowdrop.src.utils.util import saveTimeSeries as dbsave
    file_path = os.path.abspath(os.path.join(working_dir,"supplements/data/MPAF/history_new.csv"))
    dbsave(fname=file_path,data=d,variables=var,prefix="OBS_") 

    # # Compare
    # df1 = pd.read_csv(file_path.replace("_new.csv", ".csv"))
    # df2 = pd.read_csv(file_path)
    # for col in df1.columns:
    #     if col in df2.columns:
    #         series = [[df1[col], df2[col]]]
    #         labels = [["original", "new"]]
    #         plotTimeSeries(path_to_dir=path_to_dir, header=None, titles=[
    #                        col], labels=labels, series=series, sizes=[1, 1], fig_sizes=(8, 6), save=False)
    #     else:
    #         print(col)
            
    ## Take time slice of data
    for n in d:
        d[n] = d[n]['1997-1-1':'2013-1-1'].dropna()


    if Plot:
        
        fig = plt.figure(figsize=(10, 10))
        i = 0
        columns = 1
        rows = ceil(1+count/columns)
        for name in df.columns:
            if len(name)>1:
                if '_U' == name[-2:]:
                    (series,df_adj) = lst[i]
                    i += 1
                    ax = plt.subplot(rows,columns,i)
                    ax.set_title(name[:-2])
                    ax.plot(df_adj.index,df_adj["Adjusted"],linewidth=6,color='blue',marker='',markersize=3,zorder=1,label="Seasonally Adjusted")
                    ax.plot(series.index,series.values,linewidth=3,color='yellow',marker='',markersize=3,zorder=1,label="Actual")
                    ax.tick_params(labelsize=10)
                    plt.legend(fontsize = 10)
                    plt.grid(True)
        fig.suptitle('Seasonal Adjustment',fontsize=10)
        plt.show()
        
        if save:
            fig.savefig(os.path.join(path_to_dir,"SeasonallyAdjusted.pdf"))
    
    
        ## Plot results
        header = 'Nominal Variables'
        titles = ['Headline Inflation (%)',
                  'Foreign Inflation (%)',
                  'Nominal Exchange Rate: LCY per 1 FCY',
                  'Nominal Exchange Rate (%)',
                  'Nom. Interest Rate (% p.a.)',
                  'Foreign Nom. Interest Rate (% p.a.)'
                 ]
        series = [[d["DLA_CPI"], d["D4L_CPI"]],
                  [d["DLA_CPI_RW"], d["D4L_CPI_RW"]],
                  [d["S"]],
                  [d["DLA_S"],d["D4L_S"]],
                  [d["RS"]],
                  [d["RS_RW"]]
                 ]
        labels=[['q-o-q','y-o-y'],['q-o-q','y-o-y'],[],['q-o-q','y-o-y']]
        plotTimeSeries(path_to_dir=path_to_dir,header=header,titles=titles,labels=labels,series=series,sizes=[2,3],save=save)


        header = 'Real Variables'
        titles = ['GDP Growth (%)',
                  'GDP (100*log)',
                  'Real Interest Rate (% p.a.)',
                  'Real Exchange Rate (100*log)',
                  'Foreign GDP (100*log)',
                  'Foreign Real Interest Rate (% p.a.)']
        series = [[d["DLA_GDP"], d["D4L_GDP"]],
                  [d["L_GDP"], d["L_GDP_BAR"]],
                  [d["RR"], d["RR_BAR"]],
                  [d["L_Z"],d["L_Z_BAR"]],
                  [d["L_GDP_RW"], d["L_GDP_RW_BAR"]],
                  [d["RR_RW"], d["RR_RW_BAR"]]
                 ]
        labels=[['q-o-q','y-o-y'],['level','trend']]
        plotTimeSeries(path_to_dir=path_to_dir,header=header,titles=titles,labels=labels,series=series,sizes=[2,3],save=save)

        header = 'Gaps'
        titles = ['GDP Gap (%)',
                  'Foreign GDP Gap (%)',
                  'Real Interest Rate Gap (p.p. p.a.)',
                  'RER Gap (%)']
        series = [[d["L_GDP_GAP"]],
                  [d["L_GDP_RW_GAP"]],
                  [d["RR_GAP"]],
                  [d["L_Z_GAP"]]
                 ]
        labels=[[]]
        plotTimeSeries(path_to_dir=path_to_dir,header=header,titles=titles,labels=labels,series=series,sizes=[2,2],save=save)
    
        if save:
            from snowdrop.src.utils.merge import merge
            # imagelist is the list with all image filenames
            lst = ["SeasonallyAdjusted","Nominal Variables","Real Variables","Gaps"]
            outputFile = os.path.abspath(os.path.join(working_dir,"results/MPAF_Stylized_facts.pdf"))
            files = []
            for f in lst:
                files.append(working_dir+"/graphs/"+f+".pdf")
            merge(outputFile,files)
    
    print('Done with makedata script !!!')
    return d


if __name__ == '__main__':
    """
    The main program
    """
    makedata(Plot=True,save = True)
