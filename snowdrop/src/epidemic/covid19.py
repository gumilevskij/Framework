# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 18:50:05 2018
Compares results of a baseline and an alternative scenarios runs.
@author: agoumilevski
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import matplotlib.style as style

working_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),"../../.."))
path_to_dir = os.path.abspath(os.path.join(working_dir,"graphs"))


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


def plot(df1,df2,countries,titles,labels,fname):
    """Plot 1D graphs."""
    style.use("seaborn-v0_8-whitegrid")

    fig = plt.gcf()
    ax = plt.subplot(2,1,1)
    df1.plot(title=titles[0],legend=True,grid=True,figsize=(6,8),ax=ax)
    #ax.set_xlabel("Date")
    ax.set_ylabel(labels[0])
    plt.ylim(ymin=0)
    plt.tight_layout()
    
    ax = plt.subplot(2,1,2)
    df2.plot(title=titles[1],legend=True,grid=True,figsize=(6,8),ax=ax)
    #ax.set_xlabel("Date")
    ax.set_ylabel(labels[1])
    plt.tight_layout()
    plt.ylim(ymin=0)
    plt.show() 
    
    fig.savefig(os.path.join(path_to_dir,fname))
             
    
def div(df,d):
    for k in d:
        df[k] /= d[k]
    return df


def hp(x):
    cycle,trend =  sm.tsa.filters.hpfilter(x,lamb=1600)
    return trend
    

def run(f1,f2,countries,population):
    """Compare results."""
    file_path1 = os.path.abspath(working_dir+'/supplements/data/' + f1)
    file_path2 = os.path.abspath(working_dir + '/supplements/data/' + f2)
    
    df1 = readData(file_path1)
    #df1.set_index("Country/Region",inplace=True)
    df1 = df1.loc[countries]
    #df1 = df1.groupby("Country/Region").sum()
    #df1.drop(columns=['Lat','Long'],inplace=True)
    df1.index.name = "Country"
    df1 = df1.T
    df1.index = pd.to_datetime(df1.index)
    df1 = 100*div(df1,population)
    
    df2 = readData(file_path2)
    #df2.set_index("Country/Region", inplace=True)
    df2 = df2.loc[countries]
    #df2 = df2.groupby("Country/Region").sum()
    #df2.drop(columns=['Lat','Long'],inplace=True)
    df2.index.name = "Country"
    df2 = df2.T
    df2.index = pd.to_datetime(df2.index)
    df2 = 100*div(df2,population)
    
    titles = ['Infected','Deaths']
    labels = ['Percent of Population','Percent of Population']
    plot(df1,df2,countries,titles,labels,fname='Covid.png')
    
    df3 = 0.01*df1.diff()/(0.01*df1)/(1-0.01*df1)
    df3 = df3['6-1-2020':]
    df3 = 100*df3.apply(hp,axis=0)
    df4 = df2.diff()/df1
    df4 = df4['6-1-2020':]
    df4 = 100*df4.apply(hp,axis=0) 
    
    titles = ['Daily Infection Rate','Daily Death Rate']
    labels = [r'$\beta$',r'$\gamma$']
    plot(df3,df4,countries,titles,labels,fname='CovidNewCases.png')

        
if __name__ == '__main__':
    """The main entry point."""
    fi,fd = 'COVID19/infected.csv','COVID19/deaths.csv'
    countries = ["US","Germany","France","Italy","Spain"] #,"Russia","Brazil","United Kingdom"]
    population = {"US":310.e6, "Germany":83.e6,"France":67.e6,"Italy":60.e6,"Spain":47.e6}
    run(fi,fd,countries,population)
    print("Done!")
