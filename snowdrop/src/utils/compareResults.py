# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 18:50:05 2018
Compares results of a baseline and an alternative scenarios runs.
@author: agoumilevski
"""

import os
from math import ceil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style

path = os.path.dirname(os.path.abspath(__file__))
working_dir = os.path.join(path,"../")

STYLE = "seaborn-v0_8-whitegrid"
style.use(STYLE)

def readData(file_path):
    """
    Read csv file and returns pandas dataframe.
    
    Parameters:
        :param file_path: Path to excel file.
        :type file_path: str.
        :returns:  pandas dataframe.
    """
    df = pd.read_csv(file_path,sep=',',header=0,index_col=0,parse_dates=True,infer_datetime_format=True)
    df = df.fillna(0)
    return df


def plot(dfs,header="",output_variables=None,var_names=None,b=False,legend=None,sizes=[2,3]):
    """
    Plot 1D graphs.
    
    Parameters:
        :param dfs: List of Dataframe objects.
        :type dfs: list.
        :param header: Y axis label.
        :type header: str.
        :param output_variables: Output variables.
        :type output_variables: List.
        :param output_variables: Variables names.
        :type output_variables: List.
        :returns:
    """
    path = os.path.dirname(os.path.abspath(__file__))
    path_to_dir = os.path.join(path,'../../graphs')
    style.use(STYLE)
        
    #years = dates.YearLocator()   # every year
    #yearsFmt = dates.DateFormatter('%Y')
    
    print()
    print(" "*50+header.upper())
    
    df = dfs[0]
    column_names = list(df)
    if output_variables is None:
        plot_vars = [x for x in column_names if "_minus_" not in x and "_plus_" not in x]
    else:
        plot_vars = [x for x in output_variables if x in column_names]
        
    k = 0; m = 0
    nr,nc = sizes
    Nsubplots = nr*nc
    
    # Uncomment this line if you want to arrange column names in alhabetic order
    #plot_vars.sort()
      
    for i,n in enumerate(plot_vars):
        if not n in plot_vars:
            continue
        if not var_names is None and i < len(var_names):
            name = var_names[i]
        else:
            name = n
        if k%Nsubplots == 0:
            m = 0
        k += 1
        m += 1
        ax = plt.subplot(nr,nc,m)
        for ii,df in enumerate(dfs):
            if ii == 0:
                ax = df.plot(kind="line",figsize=(10,8),x=0,y=n,title=name,grid=True,lw=3,legend=False,fontsize=None,ax=ax)
            else:
                df.plot(kind="line",x=0,y=n,title=name,lw=3,legend=False,ax=ax)
            if legend is None:
                ax.legend(['Baseline','Alternative'])
            else:
                if m==1: ax.legend(legend)
        ax.set_xlabel("")
        if b: 
            ax.set_ylabel(header)
            ax.set_xlabel("Date")
        plt.grid(True)
  

        if k%Nsubplots == 0:
            plt.savefig(os.path.join(path_to_dir,header+'_Variables_'+str(ceil(k/6))+'.png'))
            plt.tight_layout()
            plt.show() 
            
    if k%Nsubplots > 0:    
        plt.savefig(os.path.join(path_to_dir,header+'_Variables_'+str(ceil((1+k)/6))+'.png'))
        plt.tight_layout()
        plt.show() 
        
    
def compare(scenarios,output_variables=None,var_names=None,legend=None,sizes=[2,3],b=False,Tmax=1.e10):
    """Compare results."""
    
    dfs = []; #dfs1 = []; dfs2 = []
    for i,scenario in enumerate(scenarios):
        file_path = os.path.abspath(os.path.join(working_dir,'../data/' + scenario))
        df = readData(file_path)
        df = df.reset_index().rename(columns={'index': 'datetime'})
        row = np.where(df.index <= Tmax)[0][-1]
        start = 0
        end = max(start,row) 
        df = df.iloc[start:end]
        if "i" in df.columns:
            df["i"] *= 100
        if "dd" in df.columns:
            df["dd"] *= 100
        dfs.append(df)
        # if i > 0:
        #     # Compute difference and percent difference
        #     df_diff = 100*df.subtract(dfs[0])
        #     df_diff["Date"] = df["Date"]
        #     dfs1.append(df_diff)
        #     df_pct_diff = (df.divide(dfs[0])-1).fillna(0)
        #     df_pct_diff["Date"] = df["Date"]
        #     dfs2.append(df_pct_diff)
        
    # Plot the results
    plot(dfs,header="Level",output_variables=output_variables,var_names=var_names,legend=legend,sizes=sizes,b=b)
    # plot(dfs1,header="Difference",output_variables=output_variables,var_names=var_names,legend=legend,Nsubplots=Nsubplots,b=b)
    # plot(dfs2,header="Percentage Difference",output_variables=output_variables,var_names=var_names,legend=legend,Nsubplots=Nsubplots,b=b)
    
        
if __name__ == '__main__':
    """The main entry point."""
    # base,alt = 'ESM_Lockdown_1Q.csv','ESM_Lockdown_2Q.csv'    
    # output_variables = ['y','ygap','r','lab','c','labf','inve','pinf','labstar']
    # var_names = ['Output','Output Gap','Nominal Policy Rate','Employment Rate','Consumption',
    #              'Labor Force Participation Rate','Investment','Inflation Rate','Labor Supply']
    # legend = ['Lockdown 1Q','Lockdown 2Q']
    
    #base,alt = 'Covid19/Simulation Results(share=1.0).csv','Covid19/Simulation Results(share=0.0).csv'
    base,alt,alt2 = 'Covid19/Sticky and Flexible Prices Economies (continued).csv','Covid19/Sticky and Flexible Prices Economies (continued)_lockdown.csv','Covid19/Sticky and Flexible Prices Economies (continued)_vaccination.csv'
    base,alt = 'Covid19/Sticky and Flexible Prices Economies (continued)_lockdown 0_05.csv','Covid19/Sticky and Flexible Prices Economies (continued)_lockdown 0_1.csv'
    output_variables = ["i","dd","y_n","c_n","n_n","x_n","w_n","rr_n","Rb_n"]
    var_names = ['Infected','Deaths',
                 'Output','Aggregate Consumption','Aggregate Hours',
                 'Investment','Wage Rate','Real Interest Rate','Policy Rate'
                  ]
    legend = ['Baseline','Lockdown','Vaccination']
    legend = [r'$\theta$ = 5%',r'$\theta$ = 10%']
    
    base,alt,alt2 = 'Covid19/Sticky and Flexible Prices Economies (continued).csv','Covid19/Sticky and Flexible Prices Economies (continued)_lockdown.csv','Covid19/Sticky and Flexible Prices Economies (continued)_vaccination.csv'
    output_variables = ["i","dd","y_n","c_n","n_n","x_n","w_n","rr_n","Rb_n"]
    var_names = ['Infected','Deaths',
                  'Output','Aggregate Consumption','Aggregate Hours',
                  'Investment','Wage Rate','Real Interest Rate','Policy Rate'
                  ]
    legend = ['Baseline','Lockdown','Vaccination']
    
    # base,alt,alt2 = 'Covid19/Sticky and Flexible Prices Economies (continued)_vaccination_2M.csv','Covid19/Sticky and Flexible Prices Economies (continued)_vaccination_4M.csv','Covid19/Sticky and Flexible Prices Economies (continued)_vaccination_6M.csv'
    # legend = ['2M','4M','6M']
    
    # base,alt = ['Covid19/US_Lockdown_1Q.csv','Covid19/US_Lockdown_2Q.csv']
    # output_variables = ['labstar','unempl','y','pinf']
    # var_names = ['Labor Supply','Unemployment','Output','Inflation']
    # legend = ['1Q','2Q']
    
    
    compare([base,alt,alt2],output_variables,var_names,legend,sizes=[3,3],b=False,Tmax=1000)
    print("Done!")
