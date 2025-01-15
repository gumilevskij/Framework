# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 11:25:02 2020

@author: AGoumilevski
"""
import os
import numpy as np
import pandas as pd

path = os.path.dirname(os.path.abspath(__file__))


def getHaverData(series,db=None,country=None,freq=None,transform=None):
    """
    Retrieves Haver database time series.

    Args:
        series : str
            Series name.
        freq : str, optional
            Frequency. The default is None.

    Returns:
        df : Pandas DataFrame
            Time series.
    """
    from  imf_datatools import haver_utilities 

    # Read Haver data
    df = haver_utilities.get_haver_data(series, eop=False)

    return transformation(df,transform)


def getEcosData(series,db='WEO_WEO_PUBLISHED',country=None,freq=None,transform=None):
    """
    Retrieves ECOS time series.

    Args:
        series : str
            Series name.
        db : str, optional
            Database name. The default is 'WEO_WEO_PUBLISHED'.
        country : str, optional
            Country name. The default is 'WEO_WEO_PUBLISHED'.
        freq : str, optional
            Frequency. The default is None.

    Returns:
        df : Pandas DataFrame
            Time series.
    """
    from  imf_datatools import ecos_sdmx_utilities

    if "@" in series:
        ind = series.index("@")
        db = series[:ind]
        series = series[ind+1:]
    elif not country is None:
        db = country
        
    df = ecos_sdmx_utilities.get_ecos_sdmx_metadata(dbname=db, seriesname=series)

    return transformation(df,transform)


def getEdiData(series,db=None,country='111',freq=None,transform=None):
    """
    Retrieves World Bank time series.

    Args:
        series : str
            Series name.
        db : str, optional
            Database name. The default is 'WEO_WEO_PUBLISHED'.
        country : str, optional
            Country name. The default country name is USA.

    Returns:
        df : Pandas DataFrame
            Time series.
    """
    from  imf_datatools import  edi_utilities

    if "_" in series:
        ind = series.index("_")
        country = series[:ind]
        variable = series[ind+1:]
    else:
        variable = series
    
    df =  edi_utilities.get_edi_weo_data(indicator=variable,country=country)

    return transformation(df,transform)


def getWorldBankData(series,db=None,country='111',freq=None,transform=None):
    """
    Retrieves World Bank time series.

    Args:
        series : str
            Series name.
        db : str, optional
            Database name. The default is 'WEO_WEO_PUBLISHED'.
        country : str, optional
            Country name. The default country name is USA.

    Returns:
        df : Pandas DataFrame
            Time series.
    """
    from  imf_datatools import worldbank_utilities

    if "_" in series:
        ind = series.index("_")
        country = series[:ind]
        variable = series[ind+1:]
    else:
        variable = series
    
    df = worldbank_utilities.get_worldbank_data(seriesname=variable,country=country)
    
    return transformation(df,transform)


def transformation(df,transform=None):
    """
    Return transformed time series.

    Args:
        df : Pandas DataFrame
            Time series data.
        transform : str, optional
            Transformation of time series data. The default is None.

    Returns:
        Transformed time series.

    """
    if transform is None:
        return df
    
    if transform.lower() == 'log':
        return 100*np.log(df)
    
    if transform.lower() == 'difflog':
        ldf = 100*np.log(df)
        return ldf.diff()
    
    return df

def fetchData(data_sources,meas_df=None,fpath='',freq='AS'):
    """
    Fetch data from database.
    
    Args:
        data_sources : str
            Name of the data source.
        meas_df : Pandas data Frame
            Measurement data to be overwritten.
        fpath : str
            Path of the original data file.
        freq : str
            Data frequency.

    Returns:
        Updated data frame.

    """
    from snowdrop.src.misc.termcolor import cprint
    
    meas_df2 = None
    if 'HAVER' in data_sources:
        get_data = getHaverData
        obs_data = data_sources['HAVER']
        datasource_name = 'HAVER'
    elif 'ECOS' in data_sources:
        get_data = getEcosData
        obs_data = data_sources['ECOS']
        datasource_name = 'ECOS'
    elif 'EDI' in data_sources:
        get_data = getEdiData
        obs_data = data_sources['EDI']
        datasource_name = 'EDI'
    elif 'WORLD_BANK' in data_sources:
        get_data = getWorldBankData
        obs_data = data_sources['WORLD_BANK']
        datasource_name = 'WORLD_BANK'
    else:
        obs_data = []
        datasource_name = None
    lseries = []    
    for k in obs_data:
        txt = obs_data[k].strip()
        if ',' in txt:
            ind = txt.index(',')
            series_name = txt[:ind]
            transform = txt[1+ind:]
        else:  
            series_name = txt
            transform = None
        if bool(series_name):
            series = get_data(series=series_name,freq=freq,transform=transform)
            column_name = series.columns[0]
            series.rename(columns = {column_name:k},inplace=True) 
            series = series.resample(freq,convention='start').mean()
            lseries.append(series)
    if len(lseries) > 0:
        meas_df2 = pd.concat(lseries,axis=1)
    if meas_df is None:
        meas_df = meas_df2
    elif not meas_df2 is None:
        nrows = len(meas_df)
        for col in meas_df2.columns:
            if col in meas_df.columns:
                series = meas_df[col]
                dates = series.index
                series2 = meas_df2[col]
                dates2 = series2.index
                mask = (dates2>=dates[0]) * (dates2 <= dates[-1])
                series2 = series2[mask]
                if len(series2) == nrows:
                    meas_df[col] = series2.values
                    cprint("Time series '{0}' has been updated from data source {1}".format(col,datasource_name),"blue")
                else:
                    cprint("Time series '{0}' has not been updated from data source {1}... Using {2} data".format(col,datasource_name,fpath),"red")

    return meas_df