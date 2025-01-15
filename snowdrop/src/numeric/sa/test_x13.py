import os, sys
import pandas as pd
import statsmodels.api as api
import matplotlib.pyplot as plt

path = os.path.dirname(os.path.abspath(__file__))
working_dir = os.path.abspath(path+"../../../..") 
os.chdir(working_dir)

from snowdrop.src.numeric.sa.x13 import x13


def test1():
    file_path = os.path.abspath(working_dir + "/data/temp/Rail_Road_Data.xlsx")   
    df_railroad_adj,df_railroad = x13(file_path=file_path,freq="M")
    df_railroad.columns=["Railroad"]
    
    fig, ax = plt.subplots(figsize=(13,6))
    ax.set_title("Actual and Seasonally Adjusted Series: US Railroad Traffic \n Monthly (Jan 2000 - Feb 2017)")
    ax.plot(df_railroad_adj.index,df_railroad_adj["Adjusted"],linewidth=2, marker='', markersize=3, zorder=1,label="Seasonally Adjusted")
    ax.plot(df_railroad_adj.index,df_railroad["Railroad"],linewidth=2, marker='', markersize=3, zorder=1,label="Actual")
    plt.show()

def test2():
    df = pd.DataFrame.from_records(api.datasets.co2.load().data)
    df['date'] = df.date.apply(lambda x: x.decode('utf-8'))
    df['date'] = pd.to_datetime(df.date, format='%Y%m%d')
    df.set_index('date')
    dta = df.resample("M")
    dta = dta.fillna(method="ffill")
    res = api.tsa.seasonal_decompose(dta)
    
    fig = res.plot()
    fig.set_size_inches(10, 5)
    plt.tight_layout()
    
    

if __name__ == '__main__':
    """
    The main program
    """
    test1()
    #test2()
    