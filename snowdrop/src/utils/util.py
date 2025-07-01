import os, re
import json
import pandas as pd
import numpy as np
import datetime as dt
from scipy import linalg as la
from tkinter import messagebox
from snowdrop.src.misc.termcolor import cprint
from dateutil.relativedelta import relativedelta
from datetime import datetime
from snowdrop.src.utils.db import create_sqlitefile,insert_values,get_data,print_data
   
path = os.path.dirname(os.path.abspath(__file__))
working_dir = os.path.abspath(os.path.join(path,"../.."))


def caseInsensitiveDict(d):
    """Convert keys of a dictionary to upper case."""
    m = dict()
    if not d is None:
        for k in d:
            m[k.upper()] = d[k]
    return m


class MyDict(dict):
    """Dictionary class with keys diagnostics."""
    
    def __init__(self,*args,**kwargs):
        self.update(*args,**kwargs)
        
    def __getitem__(self,key):
        if key in self.keys():
            return self.get(key)
        else:
            print("Key: " + key + " not found")
            return 0
        

def isPositiveDefinite(B):
    """Return true when input is positive-definite, via Cholesky."""
    try:
        _ = la.cholesky(B,lower=True)
        return True
    except la.LinAlgError:
        return False
  
    
def nearestPositiveDefinite(A):
    """Find the nearest positive-definite matrix to input.

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    Referemces:
        
        [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd
    
        [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
        matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
        
    """
    B = (A + A.T) / 2
    _, s, V = la.svd(B)
    H = np.dot(V.T, np.dot(np.diag(s), V))
    A2 = (B + H) / 2
    A3 = (A2 + A2.T) / 2

    if isPositiveDefinite(A3):
        return A3

    spacing = np.spacing(la.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = np.eye(A.shape[0])
    k = 1
    while not isPositiveDefinite(A3):
        mineig = np.min(np.real(la.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3

    
def save(fname,data,variable_names,dates=None):
    """
    Save results either to a database or to an excel file based on file name extension.
       
    Parameters:
        :param fname: The path to output file.
        :type fname: str.
        :param data: The 2D array of data.
        :type data: array.
        :param variable_names: List of output variables names.
        :type variable_names: list.
        :param dates: List of dates.
        :type dates: list.  
        
    """
    name, ext = os.path.splitext(fname)
    if ext.lower() == ".sqlite":
        saveToDatabase(fname=fname,data=data,columns=variable_names,dates=dates)
    elif ext.lower() == ".csv":
        saveToExcel(fname=fname,data=data,variable_names=variable_names,rng=dates)
    else:
        messagebox.showwarning("Input File Extension","Only saving of data to files with extensions: sqlite or csv is supported! \nYou are trying to save data to a file with extension: {}".format(ext))
 
    
def saveToDatabase(dbfilename,data,columns,dates=None):
    """
    Save data to Python sqlite database.
    
    Parameters:
        :param dbfilename: Path to output sqlite database file.
        :type fname: str.
        :param data: The 2D array of data.
        :type data: array.
        :param columns: List of output variables names.
        :type columns: list.
        :param dates: List of dates.
        :type dates: list.
        
    """
    conn = create_sqlitefile(dbfilename,columns)
    insert_values(conn,data,dates)
    
    
def saveToExcel(fname,data,par_values,variable_names,var_labels={},output_variables=None,par_names=None,rng=None,Npaths=1):
    """
    Save results to excel file.
        
    Parameters:
        :param fname: Path to output excel file.
        :type fname: str.
        :type data: Data array.
        :param data: numpy.array.
        :type par_values: Parameters values.
        :param par_values: list.
        :param variable_names: Variables names.
        :type variable_names: list.
        :param var_labels: Variables labels.
        :type var_labels: list.
        :param variable_names: Variables names.
        :type variable_names: list.
        :param output_variables: List of output variables names.
        :type output_variables: list.
        :param par_names: List of parameters names.
        :type par_names: list.
        :param rng: List of dates.
        :type rng: list.
        :param Npaths: Number of simulation paths.
        :type Npaths: int.
        
    """
    path = os.path.dirname(os.path.abspath(__file__))
    if fname is None:
        fname = os.path.abspath(os.path.join(path,'../../data/Data.csv'))
              
    # Sort variables by name
    ind = sorted(range(len(variable_names)), key=lambda k: variable_names[k])
    names = [variable_names[i] for i in ind]
    # Filter out lead and lag variables
    indices = []; var_names = []
    for i,n in enumerate(names):
        if not '_plus_' in n and not '_minus_' in n and not '__' in n:
            if bool(output_variables):
                if n in output_variables:
                    output_variables.append(n)
                    indices.append(ind[i])
            else:
                var_names.append(n)
                indices.append(ind[i])
    
    names = var_names + par_names
    var_names = ['Dates'] + names
    columns = ','.join(var_names)
    
    with open(fname, 'w') as f:
        for p in range(Npaths,0,-Npaths):
            f.writelines(columns + '\n')
            if bool(var_labels):
                labels = [str(var_labels[x]).replace(",",";") if x in var_labels else "" for x in names]
                labels = "Comments," + ",".join(labels)
                f.writelines(labels + '\n')
            if np.ndim(data) <= 2:
                yIter = data
            else:
                yIter = np.array(data[p-1])
            dim1,dim2 = yIter.shape
            if rng is None:
                T = dim1
            else:
                T = min(dim1,len(rng))
            for t in range(T):
                y = yIter[t,indices] 
                if rng is None:
                    d = str(t)
                elif isinstance(rng[t],dt.date):
                    d = str(rng[t])
                else:
                    year = dt.datetime.now().year
                    d = str(year-1+rng[t])
                f.writelines(d + ',' + ",".join(map(str, y)) + ',' + (",".join(map(str, par_values)) if t==0 else ",".join([""]*len(par_values))) + '\n')
            f.writelines('')

   
def saveScenariosToExcel(fname,scenarios,par_values,variables_names,variables_labels,output_variables=None,par_names=None,rng=None):
    """
    Save results to excel file.
        
    Parameters:
        :param fname: Path to output excel file.
        :type fname: str.
        :type scenarios: Scenarios.
        :param scenarios: list.
        :type par_values: Data array.
        :param par_values: list.
        :param variables_names: Variables names.
        :type variables_names: list.
        :param output_variables: List of output variables names.
        :type output_variables: list.
        :param par_names: List of parameters names.
        :type par_names: list.
        :param rng: List of dates.
        :type rng: list.
        
    """
    import xlsxwriter as writer
    
    path = os.path.dirname(os.path.abspath(__file__))
    if fname is None:
        fname = os.path.abspath(os.path.join(path,'../../data/Data.csv'))
    fdir = os.path.dirname(fname)
    if not os.path.exists(fdir):
        os.makedirs(fdir)
    
    # Sort variables by name
    names = [variables_labels[x] if x in variables_labels else x for x in  variables_names]
    names = ['Dates'] + names + par_names
    nparams = len(par_names)
    
    wb = writer.Workbook(fname)
    for i,scenario in enumerate(scenarios):
        wsht_name = "Baseline" if i==0 else "Scenario #" + str(i)
        wsht = wb.add_worksheet(wsht_name)
        row = 0
        # Write variables names
        for i,c in enumerate(names):
            wsht.write(row,i,c)
        # Write dates
        for row in range(len(rng)-1):
            v = rng[row] 
            wsht.write(1+row,0,v.strftime("%m-%d-%Y"))
        # Write series
        nrows,ncols = len(scenario[0]),len(scenario)
        for col,scen in enumerate(scenario):
            data = scen.values
            for row in range(nrows):
                v = data[row] 
                if np.isnan(v):
                    wsht.write(1+row,1+col,"NaN")
                else:    
                    wsht.write(1+row,1+col,v)
        # Write parameters
        for col in range(nparams):
            v = par_values[col] 
            wsht.write(1,1+ncols+col,v)
            
    wb.close()  
    wb.handles = None
    del wb
          

def saveTimeSeries(fname,data,sheetName='Sheet1',variables=None,prefix="",postfix=""):
    """
    Save time series to excel file.
        
    Parameters:
        :param fname: Path to output excel file.
        :type fname: str.
        :param data: Data.
        :type data: pd.Series, list, dict.
        
    """
    fdir = os.path.dirname(fname)
    if not os.path.exists(fdir):
        os.makedirs(fdir)
        
    if isinstance(data,pd.Series):
        data.columns = [prefix+x+postfix for x in data.columns]
        data.to_excel(excel_writer=fname,sheet_name=sheetName,header=True)
    
    elif isinstance(data,list):
        for i,d in enumerate(data):
            d.columns = [prefix+x+postfix for x in d.columns]
            d.to_excel(excel_writer=fname,sheet_name=sheetName+str(1+i),header=True)
        
    elif isinstance(data,dict):
        columns = ['Date']
        rng = set()
        for k in data:
            ts = data[k]
            if isinstance(ts,pd.Series):
                if variables is None:
                    columns.append(prefix+k+postfix)
                elif k in variables:
                    columns.append(prefix+k+postfix)
                else:
                    columns.append(k)
                if hasattr(ts.index,'to_pydatetime'):
                    time = ts.index.to_pydatetime()
                    for t in time:
                        rng.add(t)
                else:                    
                    rng = ts.index
                
        rng = np.sort(list(rng))
        with open(fname, 'w') as f:
            f.writelines(','.join(columns) + '\n')
            for r in rng:
                if isinstance(r,str) or isinstance(r,int) or isinstance(r,float):
                    txt = str(r) + ','
                else:
                    txt = r.strftime("%Y-%m-%d") + ','
                for k in data:
                    ts = data[k]
                    if isinstance(ts,pd.Series):
                        v = ts.get(r)
                        if v is None:
                            txt += ','
                        elif np.isnan(v):
                            txt += ','
                        else:
                            txt += str(v) + ','

                f.writelines(txt[:-1] + '\n')


def read(fname=None,Output=True):
    """
    Read data either from a database or an excel file based on file name extension.
        
    Parameters:
        :param fname: Path to input file.
        :type fname: str.
        :param Output: Boolean variable.
        :type Output: bool.
        
    """
    if fname is None:
        path = os.path.dirname(os.path.abspath(__file__))
        fname = os.path.abspath(os.path.join(path,'../data/Data.csv'))
        
    name, ext = os.path.splitext(fname)
    if ext.lower() == ".sqlite":
        return readDataFromDatabase(fname=fname,Output=Output)
    elif ext.lower() == ".csv":
        return readDataFromExcel(fname=fname,Output=Output)
    else:
        messagebox.showwarning("Input File Extension","Only reading data of files with extensions: sqlite or csv is supported! \nYou are trying to read data from a file with extension: {}".format(ext))
        
          
def readDataFromExcel(fname=None,Output=True):
    """
    Read data from excel file.
    
    Parameters:
        :param fname: Path to input file.
        :type fname: str.
        :param Output: Boolean variable.
        :type Output: bool.
        
    """
    if fname is None:
        path = os.path.dirname(os.path.abspath(__file__))
        fname = os.path.abspath(os.path.join(path,'../data/sol.sqlite'))
        
    rows = []; count = 0
    with open(fname, 'r') as f:
        for line in f:
            ln = [s.strip() for s in line.split(',')]
            if count == 0:
                columns = ln
            else:
                rows.append(ln)
                
            count = count + 1
                
    if Output:
        print(columns)
        for row in rows:
            print(row)
                
    return rows,columns
    

def readDataFromDatabase(fname=None,Output=True):
    """
    Read data from database.
    
    Parameters:
        :param fname: Path to input file.
        :type fname: str.
        :param Output: Boolean variable.
        :type Output: bool.
        :returns: rows and column names from database.
    
    """
    if fname is None:
        path = os.path.dirname(os.path.abspath(__file__))
        dbfilename = os.path.abspath(os.path.join(path,'../data/sol.sqlite'))
        
    rows,columns = get_data(dbfilename)
    
    if Output:
        print_data(columns,rows)
    
    return rows,columns


def getNamesAndValues(arr):
    names = [];values = []
    for e in arr:
        els = e.split("=")
        if len(els) == 2:
            names.append(els[0].strip())
            values.append(els[1].strip())
        else:
            names.append(els[0].strip())
            values.append(0)
        
    return names, values


def importModel(file_path,startDate,endDate,shocks=None):
    """
    Write GUI form data to YAML template text file.
    
    Parameters:
        :param file_path: Path to output yaml file.
        :type file_path: str.
        :param startDate: Start date of simulation.
        :type startDate: str.
        :param endDate: End date of simulation.
        :type endDate: str.
        :param shocks: Dictionary of dates and corresponding shocks.
        :type shocks: dict.
        :return: Model object.
    
    """
    from datetime import datetime as dt
    from snowdrop.src.utils.getXmlData import readXmlModelFile
    from snowdrop.src.gui.mytable import m,showTable
    from snowdrop.src.model.factory import instantiate_model
    
    txtEqs,txtParams,txtParamsRange,txtEndogVars,txtExogVars,txtShocks,txtRange,txtFreq,txtModelName = readXmlModelFile(file_path)
        
    frequencies = {"Annually":"YS", "Quarterly":"QS", "Monthly":"MS", "Weekly":"W", "Daily":"D"} 
    freqs = ["Annually", "Quarterly", "Monthly", "Weekly", "Daily"]
    
    start = dt.strptime(startDate,"%Y/%m/%d")
    end = dt.strptime(endDate,"%Y/%m/%d")
    dates = pd.date_range(start=start,end=end,freq=frequencies[txtFreq]).date
    rng = [[start.year,start.month,start.day],[end.year,end.month,end.day]]
    T = len(dates)
            
    paramNames,paramValues = getNamesAndValues(txtParams)
    endogVariablesNames,endogVariablesValues = getNamesAndValues(txtEndogVars)
    exogVariablesNames,exogVariablesValues = getNamesAndValues(txtExogVars)
    shockNames, shockValues = getNamesAndValues(txtShocks)
    
    if shocks is None:
        vals=['0']*len(shockNames)
        showTable(title="Shock Editor",names=shockNames,vals=vals)
        shocks = m   
        for t in m:
            vals = m[t]
            for k in vals:
                sh = np.float(vals[k])
                if abs(sh) > 0:
                    print("Date {0}: shock {1}={2}".format(t,k,sh))
                
    calibration = {}
    eqs = txtEqs.split("\n")
    model_name = "Sirius Model"
    
    equations = []
    for eq in eqs:
        eq = eq.replace("{","(").replace("}",")").replace("^","**")
        if ":" in eq:
            ind = eq.index(":")
            equations.append(eq[1+ind:].strip())
        else:
            equations.append(eq)
    
    for name,val in zip(endogVariablesNames,endogVariablesValues):
        calibration[name] = float(val)
    for name,val in zip(exogVariablesNames,exogVariablesValues):
        calibration[name] = float(val)
    for name,val in zip(paramNames,paramValues):
        calibration[name] = float(val)
                
    symbols = {"variables": endogVariablesNames, "shocks": shockNames, "parameters": paramNames}
        
    periods = []
    # Get periods
    for k in shocks.keys():
        d = dt.strptime(k,"%Y/%m/%d").date()
        for j in range(T-1):
            d1 = dates[j]
            d2 = dates[1+j]
            if d >= d1 and d < d2:
                periods.append(j)
                
    shock_values = []
    # Get shocks list
    for d in shocks.keys():
        lst = []
        v = shocks[d]
        for name in shockNames:
            if name in v:
                lst.append(float(v[name]))
            else:
                lst.append(0)
        shock_values.append(lst)
            
    freq = [i for i in range(len(freqs)) if freqs[i] == txtFreq][0]
    options = {"shock_values": shock_values, "range": rng, "frequency": freq, "periods": periods}
    
    data = {"calibration": calibration, "equations": equations, "name": model_name, "options": options, "symbols": symbols}
    
    model = instantiate_model(data=data,filename=file_path)
    
    return model
        
    
def SaveToYaml(file=None,description="YAML Model",shock_names=[],shock_values=[],
             variables_names=[],variables_init_values=[],param_names=[],
             param_values=[],exog_var=[],equations=[],varLabels={},eqsLabels=[],
             comments=None,time_range="",freq="0",periods="",param_range="",bInp=False):
    """
    Write GUI form data to YAML template text file.
    
    Parameters:
        :param file: Path to output yaml file.
        :type file: str.
        :param description: Description of model.
        :type description: str.
        :param shock_names: List of shock names.
        :type shock_names: List.
        :param shock_values: List of shock values.
        :type shock_values: List.
        :param variables_names: Variables names.
        :type variables_names: List.
        :param variables_init_values: Initial values of variables.
        :type variables_init_values: List.
        :param param_names: Names of parameters.
        :type param_names: List.
        :param param_values: Values of parameters.
        :type param_values: List.
        :param exog_var: Exogenous variables.
        :type exog_var: List.
        :param equations: Model equations.
        :type equations: List.
        :param varLabels: Variables labels.
        :type varLabels: List.
        :param eqsLabels: Equations labels.
        :type eqsLabels: List.
        :param comments: Equations comments.
        :type comments: List.
        :param time_range: Time range.
        :type time_range: str.
        :param freq: Frequency of time series.
        :type freq: str.
        :param periods: Periods in which shocks are applied.
        :type periods: List.
        :param param_range: Range of parameters.
        :type param_range: List.
        :param bInp: Flag that is used to indicate that a steady-state and a dynamic models should be generated.
        :type bInp: bool.
        
    """
    if file is None:
        fout = 'template'
        path = os.path.dirname(os.path.abspath(__file__))
        fdir = os.path.abspath(os.path.join(path,'../../models'))
    else:
        name = os.path.basename(file)
        fout,ext = os.path.splitext(name)
        fdir = os.path.dirname(file)
        
    file_path = os.path.abspath(os.path.join(fdir,fout + '.yaml'))
    file_path_dyn = os.path.abspath(os.path.join(fdir,fout + '.yaml'))
    file_path_ss = os.path.abspath(os.path.join(fdir,fout + '_ss.yaml'))
    if not file is None:
        name,ext = os.path.splitext(file)
        if ext.lower() == ".inp":
            file_path_dyn = file.replace(name+ext,name+"_dyn.yaml")
            file_path_ss = file.replace(name+ext,name+"_ss.yaml")
    
    # Replace curly brackets
    if isinstance(equations,list):
        equations = "\n".join(equations)
    equations = equations.replace("{","(").replace("}",")")
    eqs = equations.split("\n")
    eqs = list(filter(None,eqs))
    
    # Check equations labels
    if not isinstance(eqsLabels,list) or not len(eqsLabels) == len(eqs):
        eqsLabels = [None]*len(eqs)
            
    temp = []
    if isinstance(exog_var,str):
        exog_var = exog_var.split("\n")
    for v in exog_var:
        ind = v.find("=")
        if ind >= 0:
            temp.append(v[:ind].strip())
    new_param_names = set(param_names+temp)-set(shock_names)
    
    
    if False and bInp and "_ss" in equations: #ag
        
        # Create dynamic model file
        Output(True,file_path_dyn,description,variables_names,shock_names,new_param_names,
           eqs,param_names,param_values,exog_var,variables_init_values,time_range,freq,periods,
           shock_values,param_range,eqsLabels=eqsLabels,varLabels=varLabels,comments=comments,bInp=bInp)
            
        # Create steady-state model file
        Output(False,file_path_ss,description,variables_names,shock_names,new_param_names,
           eqs,param_names,param_values,exog_var,variables_init_values,time_range,freq,periods,
           shock_values,param_range,eqsLabels=eqsLabels,varLabels=varLabels,comments=comments,bInp=bInp)
    
    else:
        
        # Create template model file
        Output(True,file_path,description,variables_names,shock_names,new_param_names,
           eqs,param_names,param_values,exog_var,variables_init_values,time_range,freq,periods,
           shock_values,param_range,eqsLabels=eqsLabels,varLabels=varLabels,comments=comments)
      
        
def Output(b,file_path,description,variables_names,shock_names,new_param_names,
           eqs,param_names,param_values,exog_var,variables_init_values,time_range,freq,
           periods,shock_values,param_range,eqsLabels=[],varLabels={},comments=[],bInp=False):
    """Output text to YAML model file."""
    if comments is None:
        comments = [None] * len(eqs)
    with open(file_path, 'w') as f:
        f.write('name: ' + description + '\n\n')
        f.write('symbols: \n')
        f.write('\n   variables : [' + ','.join(variables_names) + ']')
        f.write('\n\n   shocks : [' + ','.join(shock_names) + ']')
        f.write('\n\n   parameters : [' + ','.join(new_param_names) + ']\n')
        f.write('\nequations: \n')
        if b:
             for lb,eq in zip(comments,eqs):
                 if not lb is None:
                    f.write('   # ' + lb + '\n') 
                 if ":" in eq:
                    ind = eq.index(":")
                    lable = eq[:ind].strip()
                    eq = lable + " : " + eq[1+ind:]
                 if bInp and "_ss" in eq:
                     continue
                 e = eq.strip()
                 if e:
                    f.write('   - ' + e + '\n')
        else:    
            j = 0
            neqs = len(eqs)
            for i in range(neqs):
                if j > neqs - 2:
                    continue
                eq1 = eqs[j]
                eq2 = eqs[1+j]
                j += 1
                if "_ss" in eq2:
                    e = eq2.replace("_ss","").strip()
                    j += 1
                else:
                    e = eq1.strip()
                lastEq = e
                if e:
                    f.write('   - ' + e + '\n')
            eq2 = eqs[len(eqs)-1]
            e = eq2.replace("_ss","").strip()
            if e and lastEq != e:
                f.write('   - ' + e + '\n')
        f.write('\ncalibration:')
        f.write('\n\n   # PARAMETERS')
        for p,v in zip(param_names,param_values):
            if p and v:
                f.write('\n   ' + p + ': ' + v)
        f.write('\n\n   # EXOGENOUS VARIABLES')  
        for v in exog_var: 
            if v.strip():
                f.write('\n   ' + v.replace("="," : ").strip())
        f.write('\n\n   # STARTING VALUES OF ENDOGENOUS VARIABLES')
        for n,v in zip(variables_names,variables_init_values):
            f.write('\n   ' + n + ': ' + v)
        if varLabels:
            f.write('\n\nlabels:') 
            for k in varLabels:
                f.write('\n   ' + k + ': ' + varLabels[k])
        f.write('\n\noptions:')
        if time_range:
            if "[" in time_range:
                f.write('\n   range : ' + time_range)
            else:
                f.write('\n   T : ' + time_range[0])
        if len(shock_values) > 0:
            f.write('\n   frequency : ' + freq)
            if len(periods) > 0:
                f.write('\n   periods : ' + str(periods))
            f.write('\n\n   shock_values : [[' + ','.join([str(x) for x in shock_values]) + ']]')
        for p in param_range:
            p = p.replace("="," : ")
            if ':' in p:
                left,right = p.split(':')
                if '-' in right:
                    tmp = right.split('-')
                    right = '[' + tmp[0] + ',' + tmp[1] + ']'
                f.write('\n   ' + left.strip() + ': ' + right.replace(' ',''))
        
def getNamesValues(arr):
    """
    Return list of names and values.
    
    Parameters:
        :param arr: Array
        :type arr: List.
        :return: Variables names and values
        
    """
    names = []; values = []
    for e in arr:
        ind = e.find("=")
        if ind >= 0:
            names.append(e[:ind].strip())
            values.append(e[1+ind:].strip())
    return names,values
     
    
def tracefunc(frame, event, arg, indent=[0]):
    """Trace calls to a function."""
    if event == "call":
        indent[0] += 2
        print ("-" * indent[0] + "> call function ",  frame.f_code.co_name)
    elif event == "return":
        print( "<" + "-" * indent[0], "exit function ", frame.f_code.co_name)
        indent[0] -= 2
    return tracefunc

    
def deleteFiles(path_to_dir, pattern):
    """Delete files in a directory matching a pattern."""
    for f in os.listdir(path_to_dir):
        if re.search(pattern, f):
            try:
                os.remove(os.path.join(path_to_dir, f))
            except IOError as e:
                print(e)


def correctLabel(lbl):
    """
    Correct a variable name.
    
    For example, variable name x_minus_1 will be changed to x(-1)
    
    """
    ind1 = lbl.find("_plus_")
    ind2 = lbl.find("_minus_")
    if ind1 == -1 and ind2 == -1:
        return lbl
    ind3 = lbl.rfind("_")
    digit = digit1 = lbl[1+ind3:]
    if "(" in digit and ")" in digit:
        ind = digit.index("(")
        digit2 = digit[:ind]
        digit3 = digit[1+ind:].replace(")","")
        digit1 = str(int(digit2) + int(digit3))
            
    name = lbl.replace("_plus_"+digit,"("+digit1+")")
    name = name.replace("_minus_"+digit,"(-"+digit1+")")
    
    return name


def getMap(file_path):
    """Return map of model variable names to report variable names."""
    m = {}
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip("\n")
                arr = line.split(",")
                if len(arr) >= 2:
                    m[arr[0]] = arr[1]
        
    return m        
    

def correctVariablesNames(variable_names):
    """Correct names of FSGM model variables so that they are consistent with names of variables of Fame reports."""
    import re
    
    path = os.path.dirname(os.path.abspath(__file__))
    fname= os.path.abspath(os.path.join(path,'../../data/map.csv'))
    variablesMap = getMap(fname)
            
    for i in range(len(variable_names)):
        n = variable_names[i]
        ind = [x.start() for x in re.finditer("_", n)]
        v = n
        if len(ind) == 0:
            if n in variablesMap.keys():
               v = variablesMap[n]
        else:
            for j in ind:
                n1 = n[:j]
                n2 = n[1+j:]
                if n2 in variablesMap.keys():
                    v = n1 + "_" + variablesMap[n2]
                    break
        if not n == v:
            variable_names[i] = v 
            print(n," -> ",v)
    
    return variable_names


def correctHeaderOfCsvFile(fin,fout):
    """
    Correct header of Csv files so that names of variables are consistent with names of variables of Fame reports.
    """
    with open(fin, 'r') as f:
        lines = f.readlines()
        
    header = lines[0]
    new_variables_names = correctVariablesNames(header.split(","))
        
    lines[0] = ",".join(new_variables_names)
       
    with open(fout, 'w') as f:
        f.writelines(lines)
        
    return new_variables_names


def correctHeaders(fname):
    """Correct columns names."""
    file_path = os.path.dirname(os.path.abspath(__file__))
    fin = os.path.abspath(os.path.join(file_path,"../../data/" + fname))
    out_folder = os.path.abspath(os.path.join(file_path,"../../../FSGMreports/BASE/Data/"))
    if os.path.exists(fin) and os.path.exists(out_folder):
        fout = out_folder + fname
        correctHeaderOfCsvFile(fin=fin,fout=fout)


def getExogenousSeries(model,debug=False):
    """
    Convert exogenous series to 2D array.
    
    Parameters:
        :param model: Model object.
        :type model: Model.f
        :return: Pandas data frame.
    """
    frequencies = {"0":"YS","1":"QS","2":"MS","3":"W","4":"D"} #Annually,Quarterly,Monthly,Weekly,Daily
    
    freq = model.options["freq"] if "freq" in model.options else None
    if freq is None:
        if 'frequency' in model.options:
            freq = model.options['frequency']
        else:
            freq = 0
            
    data = model.symbolic.exog_data
    exog_data = dict()
    if bool(data):
        T = 0
        for k in data:
            T = max(model.T,T,len(data[k]))
        model.T = T
        start,end,rng,dates,_,_,_,_,T = simulationRange(model,freq=freq,T=T)   
        
        if isinstance(dates[0],int):
            index = np.arange(1,T+1)
            for i,exo in enumerate(data):
                ser = pd.Series(data=np.nan,index=index)
                x = data[exo]
                ser[x.index] = x
                ser[index<x.index[0]] = 0
                ser = ser.ffill().bfill().values
                exog_data[exo] = [ser[0]] + list(ser)
                
            return exog_data
                    
        for exo in data:
            ser = data[exo]
            if len(ser) > 0:
                ser = ser.resample(frequencies[str(freq)]).mean()
                index = ser.index
                val = []
                ind1 = ser.notna().idxmax()
                ind2 = ser.notna()[::-1].idxmax()
                for t in dates:
                    if t < index[0]:
                        v = ser[ind1]
                    elif t > index[-1]:
                        v = ser[ind2]
                    else:
                        v = ser[t]
                    if abs(v) < 1.e-10:
                        v = 1.e-10
                    val.append(v)
  
                val = np.array(val)
                ser2 = pd.Series(data=val,index=dates)
                ser2.bfill(inplace=True)
                ser2.ffill(inplace=True)
                ser2.fillna(1.e-10)
                val = ser2.values
                
                #TODO Period zero is starting values. Check if zero should be inserted at the beginning.
                exog_data[exo] = val                     
                      
            if debug and any(np.isnan(val)):
                cprint(f"Series {exo} contains NAN values.  Replacing Nans with previous or later values...","red")
                      
    return exog_data
                        
            

def simulationRange(model,freq=None,T=None):
    """
    Return and set the  date range of simulations.
    
    Parameters:
        :param model: Model object.
        :type model: Model.
        :param freq: Dates frequency.
        :type freq: str.
        :param T: Time span of simulations.
        :type T: int.
        :return: pandas date_range.
    """
    frequencies = {"0":"Y","1":"Q","2":"M","3":"W","4":"D"} #Annually,Quarterly,Monthly,Weekly,Daily
    start,end,rng,dates,Time,start_filter,end_filter,filter_rng = None,None,None,None,None,None,None,None
    
    if freq is None:
        if 'frequency' in model.options:
            freq = model.options['frequency']
        else:
            freq = 0
            
    if 'range' in model.options:
        rng = model.options['range']
        if not rng is None and len(rng)>1:
            start_filter = start = getDate(rng[0])
            end_filter = end = getDate(rng[1])
            rng = pd.period_range(start=start,end=end,freq=frequencies[str(freq)])
            if end.year-start.year < 200:
                dates = rng.to_timestamp()
            else:
                dates = [x.strftime('%m/%d/%Y') for x in rng]
            if T is None:
                Time = range(start.year,end.year+1)
            else:
                T = len(rng)
                rng = rng[:T]
                Time = range(start.year,end.year+T+1)
            #Time = dates.datestr2num(rng.strftime('%B %d, %Y'))
        else:
            if T is None:
                if 'T' in model.options:
                    T = model.options['T']
                else:
                    T = 101
            start = dt.date(2025,1,1)
            end = dt.date(2025+T,1,1)
            rng = pd.period_range(start=start,end=end,freq=frequencies[str(freq)])
            if end.year-start.year < 200:
                dates = rng.to_timestamp()
            else:
                dates = [x.strftime('%m/%d/%Y') for x in rng]
            Time = range(1,T+1)
    else:
        if T is None:
            if 'T' in model.options:
                T = model.options['T']
            else:
                T = 101
        start = 1
        end = T
        Time = list(range(1,T+1))
        rng = list(range(T))
        dates = rng
              
    if 'filter_range' in model.options:
        rngf = model.options['filter_range']
        if not rngf is None and len(rngf) > 1:
            start_filter = getDate(rngf[0])
            if freq == 0:
                previous_period = start_filter - relativedelta(years=1)
            elif freq == 1:
                previous_period = start_filter - relativedelta(months=3)
            elif freq == 2:
                previous_period = start_filter - relativedelta(months=1)
            elif freq == 3:
                previous_period = start_filter - relativedelta(weeks=1)
            elif freq == 4:
                previous_period = start_filter - relativedelta(days=1)
            if previous_period < start:
                cprint(f"Changing starting date '{start}' to '{previous_period}' to ensure that filtartion starts after starting date","red")
                start = previous_period
            end_filter = getDate(rngf[1])
            filter_rng = pd.period_range(start=start_filter,end=end_filter,freq=frequencies[str(freq)])
        
    model.date_range = dates
    if not T is None:
        model.T = T
        
    return start,end,rng,dates,start_filter,end_filter,filter_rng,Time,T
        

def getDate(d):
    if isinstance(d,str):
        #date = dt.datetime.strptime(d,'%Y-%m-%d')
        date = pd.to_datetime(d.replace(",","-").replace(".","-"))
    elif isinstance(d,int):
        date = dt.date(d,1,1)
    else:
        if len(d)==3: 
            date = dt.date(d[0],d[1],d[2])
        elif len(d)==2: 
            date = dt.date(d[0],d[1],1)
        elif len(d)==1: 
            date = dt.date(d[0],1,1)
        else:
            date = dt.date(d[0],1,1)
        date = pd.to_datetime(date)
    return date


def getPeriods(model,T,rng=None):
    """
    Return list of periods.
    
    Parameters:
        :param model: Model object.
        :type model: Model.
        :param T: Time span of simulations.
        :type T: int.
        :return: List of periods.
        
    """
    periods = None
    if 'periods' in model.options:
        per = model.options.get('periods',[])
        if not per is None and len(per) > 0:
            if isinstance(per[0],list):
                periods = []
                rngs = rng.to_timestamp()
                for p in per:
                    for i in range(T):
                        if len(p)==3: 
                            d = dt.date(p[0],p[1],p[2])
                        elif len(p)==2: 
                            d = dt.date(p[0],p[1],1)
                        elif len(p)==1: 
                            d = dt.date(p[0],1,1)
                        else:
                            d = dt.date(p[0],1,1)
                        for j in range(min(T-1,len(rngs)-1)):
                            d1 = rngs.date[j]
                            d2 = rngs.date[1+j]
                            if d >= d1 and d < d2:
                                periods.append(j)

            elif isinstance(per[0],str): 
                if per[0].isdigit():
                    periods = per
                elif not rng is None:
                    from dateutil.parser import parse
                    d = parse(per[0])
                    rngs = rng.to_timestamp()
                    for j in range(min(T-1,len(rngs)-1)):
                        d1 = rngs[j]
                        d2 = rngs[1+j]
                        if d >= d1 and d < d2:
                            periods = [j]
                            break
            elif isinstance(per[0],int):
                periods = per
            else:
                periods = [per]
                
        elif not rng is None:
            periods = []
            for i in range(T):
                d = per[i]
                if len(d)==3: 
                    d = dt.date(d[0],d[1],d[2])
                elif len(d)==2: 
                    d = dt.date(d[0],d[1],1)
                elif len(d)==1: 
                    d = dt.date(d[0],1,1)
                else:
                    d = dt.date(d[0],1,1)
                for j in range(min(T-1,len(rng)-1)):
                    d1 = rng.date[j]
                    d2 = rng.date[1+j]
                    if d >= d1 and d < d2:
                        periods.append(j)
                    
    return periods

 
def findVariableLag(x):
    """
    Find lag of a variable given its name.
    
    Parameters:
        :param x: Variable name.
        :type x: str.
        :return: Lag of a variable.
        
    """
    lag = 0
    if '_minus_' in x:
        ind = x.index('_minus_') + len('_minus_')
        x = x[ind:]
        if '(' in x:
            ind = x.index('(')
            s = x[:ind]
        else:
            s = x
        lag -= int(s)
    elif '_m' in x:
        ind = x.index('_m') + len('_m')
        x = x[ind:]
        if '_' in x:
            ind = x.index('_')
            x = x[:ind]
        if '(' in x:
            ind = x.index('(')
            s = x[:ind]
        else:
            s = x
        lag -= int(s)
    if "(" in x and ")" in x:
        ind1 = x.index("(")
        ind2 = x.index(")")
        lag += int(x[1+ind1:ind2])
    
    return lag  

 
def findVariableLead(x):
    """
    Find lead of a variable given its name.
    
    Parameters:
        :param x: Variable name.
        :type x: str.
        :return: Lead of a variable.
        
    """
    lead = 0
    if '_plus_' in x:
        ind = x.index('_plus_') + len('_plus_')
        x = x[ind:]
        if '(' in x:
            ind = x.index('(')
            s = x[:ind]
        else:
            s = x
        lead = int(s)
    elif '_p' in x:
        ind = x.index('_p') + len('_p')
        x = x[ind:]
        if '_' in x:
            ind = x.index('_')
            x = x[:ind]
        if '(' in x:
            ind = x.index('(')
            s = x[:ind]
        else:
            s = x
        lead = int(s)
    if "(" in x and ")" in x:
        ind1 = x.index("(")
        ind2 = x.index(")")
        lead += int(x[1+ind1:ind2])
 
    
    return lead


def compare(save,fnames,legends):
    """Compare time series in two excel files."""
    import matplotlib.pyplot as plt
    from snowdrop.src.utils.merge import merge

    # Read data
    dfs = []
    for i,fname in enumerate(fnames):
        file_path = os.path.abspath(os.path.join(working_dir,fname))
        df = pd.read_csv(file_path,header=0,index_col=0,parse_dates=True)
        df = df.iloc[2:].astype(float)
        df.index = pd.to_datetime(df.index)
        dfs.append(df)
        var = list(df.columns)
        if i == 0:
            variables = var
        else:
            variables = [x for x in variables if x in var]
            
    n = len(variables)
    if n==0:
        return
    
    files = []
    for k in range(0,6,n):
        plt.figure(figsize=(16,16))
        ii = 0
        for i in range(6):
            j = 6*k+i
            if j < n:
                name = variables[j]
                ax = plt.subplot(3,2,1+ii)
                m = 0
                for df in dfs:
                    ax.plot(df[name],label=legends[m])
                    m += 1
                plt.title(name,fontsize = 'x-large')
                plt.xlabel('Year')
                plt.legend(loc="best",fontsize='x-large')
                plt.grid(True)
                ii += 1
        
        file_path = os.path.abspath(os.path.join(working_dir,'results/compare_'+str(k+1)+'.pdf'))
        files.append(file_path)
        plt.savefig(file_path)
    
    if save:
        outputFile = os.path.abspath(os.path.join(working_dir,"results/compare.pdf"))
        merge(outputFile,files) 
              
        
def getVariableValue(eqs:list,v:str,var_names:list,m:dict) -> float:
    """
    Replaces variables in STEADY_STATE function with values.

    Parameters:
        eqs : list
            Equations.
        v : str
            Name of a variable.
        var_names : list
            List of variables names.
        m : dict
            Map of varaiables names and values.

    Returns:
        Value of a variable.
        
    """  
    for eq in eqs:
        eq = eq.replace(" ","")
        if eq.startswith(v+"="):
            for n in var_names:
                eq = eq.replace(n+"(-1)",n)
                eq = eq.replace(n+"(+1)",n)
                eq = eq.replace(n+"(1)",n)
            try:
                exec(eq,{"__builtins__":None},m)
                val = m[v]
            except:
                val = np.nan
            return val
    return np.nan

    
def replaceExpressions(eqs: list, m: dict) -> list:
    """
    Replaces variables in STEADY_STATE function with values.

    Parameters:
        eqs : list
            Equations.
        m : dict
            Map of varaiables names and values.

    Returns:
        Equations with replaced steady state values.

    """
    fn = "STEADY_STATE"
    n = len(fn)
    lst = list()
    for eq in eqs:
        s = ""
        while fn in eq:
            ind1 = eq.find(fn)
            ind2 = eq.find(")",1+ind1)
            s += eq[:ind1]
            arg = eq[ind1+n+1:ind2].strip()
            if ind1>=0 and ind2==-1:
                print("util.replaceExpressions error: missing closing parenthis.")
                break
            if arg in m:
                s += "{}".format(m[arg])
            else:
                s += fn + "('" + arg + "')"
            eq = eq[1+ind2:]
        s += eq
        # if fn in eqtn:
        #     print('eq:',eqtn,'\n   ',s)
        lst.append(s)
        
    return lst       
        
def saveLeadLagIncidence(lli,isLinearModel):
    """
    Store lead lag incidence matrix in a file.
    
    If model is non-linear delete the lead lag incidence file.

    Parameters:
        lli : numpy.array
            Matrix of variables leads and lags.
        isLinearModel : bool
            True if model is linear and False otherwise.
    
        Returns:
            None.

    """
    fpath = os.path.abspath(os.path.join(working_dir,"data/cpp/lli.csv"))
    if isLinearModel:
        with open(fpath, "w") as f:
            for i in range(len(lli)):
                f.write(','.join([str(x) for x in lli[i]]) + '\n')
    elif os.path.exists(fpath):
        os.remove(fpath)
        
            
        
def create_config_file(T,variables_names,initial_values,shock_names,shock_values,parameters_names,parameters_values,options):
    """
    Create files with data fot initial endogenous variables value, parameters and shock values.
    
    These files are created when model option GENERATE_CPP_CODE is set to True.

    Parameters:
        T : int.
            Number of time steps.
        variables_names : list.
            Names of endogenous variables.
        variables_names : list.
            Names of endogenous variables.
        initial_values : numpy array.
            Initial values of endogenous variables.
        shock_names : list.
            Names of shock variables.
        shock_values : list.
            Shocks values.
        parameters_names : list.
            Names of model parameters.
        parameters_values : numpy array.
            Parameters values.
        shocks : dictionary.
            Model options.

    Returns:
        None.

    """
    t = 0
    if "shock_values" in options:
        shock_values = options.get("shock_values")
        if isinstance(shock_values,list) and len(shock_values)==1:
            if isinstance(shock_values[0],list):
               shock_values = shock_values[0] 
    shock_values = np.array(shock_values)
    periods = options["periods"] if "periods" in options else None
    
    # Write variables initial values
    with open(os.path.join(working_dir,"data/cpp/initial_values.csv"), "w") as f:
        # Writing data to a file
        f.write("Period,"+",".join(variables_names)+"\n")
        if isinstance(initial_values,list):
            f.write("0,"+",".join([str(x) for x in initial_values])+"\n")
        else:
            if np.ndim(initial_values) == 1:
                f.write("0,"+",".join(initial_values.astype(str).tolist())+"\n") 
            else:
                f.write("0,"+",".join(initial_values[-1].astype(str).tolist())+"\n") 
                          
        
    # Write parameters values
    with open(os.path.join(working_dir,"data/cpp/parameters.csv"), "w") as f:
        # Writing data to a file
        f.write("Period,"+",".join(parameters_names)+"\n")
        if isinstance(parameters_values,list):
            f.write("0,"+",".join([str(x) for x in parameters_values])+"\n")
        else:
            if np.ndim(parameters_values) == 1:
                f.write("0,"+",".join(parameters_values.astype(str).tolist())+"\n") 
            else:
                for t in range(T):
                    f.write(str(t)+","+",".join(parameters_values[:,t].astype(str).tolist())+"\n") 
            
    # Write shocks values
    with open(os.path.join(working_dir,"data/cpp/shocks.csv"), "w") as f:
        # Writing data to a file
        f.write("Period,"+",".join(shock_names)+"\n")
        n_shocks = len(shock_names)
            
        # If periods is not present in model options it means that shock is permanent.
        # Then add zeros to shocks to indicate transient nature of shocks.
        k = 0
        # Period t=0 defines starting values.  So insert zero shocks...
        f.write("0,"+",".join(["0" for i in range(n_shocks)])+"\n")
        if periods is None:
            for t in range(1,T):
                if np.ndim(shock_values) == 1:
                    f.write(str(t)+","+",".join(shock_values.astype(str).tolist())+"\n")
                elif t < len(shock_values):
                    f.write(str(t)+","+",".join(shock_values[t].astype(str).tolist())+"\n")
                else:
                    f.write(str(t)+","+",".join(["0" for i in range(n_shocks)])+"\n")
                
        else:
            for t in range(1,T):
                if t in periods:
                    if np.ndim(shock_values) == 1:
                        f.write(str(t)+","+",".join(shock_values.astype(str).tolist())+"\n")
                    elif t < len(shock_values):
                        f.write(str(t)+","+",".join(shock_values[t].astype(str).tolist())+"\n")
                    k += 1        
                else:
                    f.write(str(t)+","+",".join(["0" for i in range(n_shocks)])+"\n")
                
            
def compareTrollFiles(fpath1,fpath2):
    """Compare steady state values in two excel files."""
    import pandas as pd
    import numpy as np
    
    # Read data
    file_path1 = os.path.abspath(os.path.join(working_dir,"snowdrop/data/Troll/FSGM3", fpath1))
    file_path2 = os.path.abspath(os.path.join(working_dir,"snowdrop/data/Troll/FSGM3", fpath2))
    fout = os.path.abspath(os.path.join(working_dir,"data/Troll/FSGM3/comparison.csv"))
    
    
    df1 = pd.read_csv(file_path1,header=0,index_col=0,parse_dates=True)
    df2 = pd.read_csv(file_path2,header=0,index_col=0,parse_dates=True)
    keys1,vals1 = df1.index,df1.iloc[:,0]
    keys2,vals2 = df2.columns,df2.iloc[-1]
    keys1 = [x.strip() for x in keys1]
    keys2 = [x.strip() for x in keys2]
    m1 = dict(zip(keys1,vals1))
    m2 = dict(zip(keys2,vals2))
    keys  = m1.keys() & m2.keys()
    arr   = []
    
    for k in keys:
        diff = m1[k]/m2[k]-1 if not m1[k]*m2[k]==0 else 0
        x = [k,m1[k],m2[k],diff]
        arr.append(x)
    
    data = np.array(arr)
    df = pd.DataFrame(data,columns=["Var","Troll","Framework","Relative Difference"])
    df.to_csv(fout)
    
    
def read_and_combine_text(fpath: str,_if_="!if",_else_="!else",_end_="!end",_import_="!import",
                          transition_variables="!transition_variables",
                          transition_shocks="!transition_shocks",
                          measurement_variables="!measurement_variables",
                          measurement_shocks="!measurement_shocks",
                          parameters="!parameters",
                          transition_equations="!transition_equations",
                          measurement_equations="!measurement_equations",
                          stop="!stop",Legend="Legend",
                          conditions: dict = None,debug=False,
                          ) -> str:
    """Combine separate parts of file text."""
    from snowdrop.src.misc.termcolor import cprint
    
    fdir = os.path.dirname(fpath) 
    lines = []; original_lines = []
    fname,ext = os.path.splitext(fpath)
    with open(fpath) as file:
        while line := file.readline():
            original_lines.append(line)
        
    m = {"Legend": Legend, "!stop": stop}
    special = list(m.keys())
    keyWords = [transition_variables,transition_shocks,
                measurement_variables,measurement_shocks,
                parameters,transition_equations,
                measurement_equations,Legend]
   
    key = None; content=[]; start=False
    for k in keyWords:
        m[k] = ""
        
    # Preparse !if, !else, !end directives
    for line in original_lines:
        ln = line.replace(";","").replace("\n","").strip()
        if ln in conditions:
            line = "% " + line.replace(";","")
        if ln.startswith(_if_):
            block1 = []; block2 = []; states = [1]
            expr = ln[3:].strip()
            condition = expr.replace("~","").strip().split(" ")[0].strip()
            start = True
            if condition in conditions:
                expr = expr.replace("~"," not ")
                expr = expr.replace(condition,str(conditions[condition]))
                b = include = eval(expr)
            else:
                expr = expr.replace("~","not ").replace("true","True").replace("false","False")
                b = include = eval(expr)
        elif start and ln.startswith(_else_):
            states.append(2)
            include = not include
        elif start and ln.startswith(_end_):
            start = False
            states.append(3)
            if b:
                content.extend(block1)
                if debug:
                    if condition in conditions:
                        cprint(f"Included block #1 for condition: {condition} = {conditions[condition]}","blue")
                    else:
                        cprint(f"Included block #1 for condition: {expr}","blue")
            else:
                content.extend(block2)
                if debug:
                    if condition in conditions:
                        cprint(f"Included block #2 for condition: {condition} = {conditions[condition]}","blue")
                    else:
                        cprint(f"Included block #2 for condition: {expr}","blue")
        else:
            if start:
                if include:
                    block1.append(line)
                elif 2 in states:
                    block2.append(line)
            else:
                content.append(line)
                
    # Import files
    for line in content:
        ln = line.strip()
        if ln.startswith(_import_):
            if "(" in ln and ")" in ln:
                ind1 = ln.index("(")
                ind2 = ln.index(")")
                fl = ln[1+ind1:ind2]
            file = os.path.abspath(os.path.join(fdir,fl))
            line = ["\n"]
            if os.path.exists(file):
                with open(file) as f:
                    line = f.readlines()
            lines.extend(line)
        else:
            lines.append(line)
    
    if debug:
        with open(fname + ".text","w") as file:
            file.write("".join(lines))
        
    
    # Combine text for each key word block        
    for line in lines:
        if ";" in ln:
            ind = ln.index(";")
            ln = ln[:ind]
        ln = line.strip()
        if ln in keyWords:
            key = line.replace("\n","").strip()
        elif not key is None:
            m[key] += line
            
    # Combine all pieces together
    txt = ""
    for k in keyWords:
        if not k in special:
            txt += "\n\n" + k + "\n" + m[k]
        
    with open(fname + ".txt","w") as file:
        file.write(txt)
        
    return txt
            

def getOutputFolderPath():
    name = "PLATFORM_OUTPUT_FOLDER"
    if name in os.environ:
        output_dir = os.environ[name]
    else:
        output_dir = os.path.abspath(os.path.join(working_dir,'../supplements'))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)      
    return output_dir
          
    
def compareResults(yy,variables,dates,filter_range,fdir):
    
     import matplotlib.pyplot as plt
     import pandas as pd
    
     # start.end = dates[0],dates[-1]
     filter_start,filter_end = filter_range[0],filter_range[-1]
     start = getDate(filter_start)
     end = getDate(filter_end)
     
     lbls = ["Python","Iris","Difference"]
     Outputs = ["Filter","Smooth"]
     for i,output in enumerate(Outputs):
         filtr = {}
         results = yy[i]
         for j,n in enumerate(variables):
             if "_plus_" in n or "_minus_" in n:
                 continue
             data = results[:,j]
             m = min(len(data),len(dates))
             ts = pd.Series(data[:m], dates[:m])
             ts = ts[start:end]
             ts.index = pd.to_datetime(ts.index,format='#Q')
             filtr[n] = ts
            
         # Read Iris results
         fpath = os.path.abspath(os.path.join(fdir, '../Iris/Ireland/data/kf_'+output+'.csv'))
         df = pd.read_csv(fpath,index_col=0,parse_dates=True)
         
         fig = plt.figure(figsize=(10,8))
         rows = 4; columns = 2
         for i,v in enumerate(variables):
             ax = plt.subplot(rows,columns,1+i)
             filtr[v].plot(ax=ax,linewidth=1,color='r')
             df[v][start:end].plot(ax=ax,linewidth=1,color='b')
             diff = df[v][start:end]-filtr[v]
             diff.plot(ax=ax,linewidth=1,color='k')
             plt.title(v)
             if i==0: plt.legend(lbls,loc="best",ncol=len(lbls))
             plt.grid(True)
             
         fig.suptitle(output,fontsize=25,fontweight='bold')
         plt.tight_layout()
         plt.show(block=False)
         fig.savefig(os.path.abspath(os.path.join(fdir,'../graphs/kf_comp_'+output+'.png')),dpi=600)
      
def output(fdir,names,*args):
    for i,arg in enumerate(args):  
        df = pd.DataFrame(arg) 
        df.to_csv(f"C:/Temp/{fdir}/{names[i]}.csv")
    import sys
    sys.exit(0)

class Encoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj,pd.Series):
            return obj.to_json()
            
        return json.JSONEncoder.default(self, obj)
    
def saveData(fdir, **kwargs):
    for k in kwargs:
        fpath = os.path.join(fdir,k+'.json') 
        if not os.path.exists(fdir):
            os.mkdir(fdir)
        data = kwargs[k]
        with open(fpath,'w') as f:
            if isinstance(data,pd.DatetimeIndex): 
                tmp = data.strftime('%Y-%m-%d').tolist()
                json_data ='["'+ '","'.join(tmp)+'"]'
                f.write(json_data)
            elif isinstance(data,dict):
                json_data = json.dumps(data,cls=Encoder)
                f.write(json_data)
            elif isinstance(data,np.ndarray):
                json_data = json.dumps(data.tolist())
                f.write(json_data)
            elif isinstance(data,list):
                json_data = json.dumps(data)
                f.write(json_data)
            else:
                json.dump(data,f)
           
    
def loadData(fdir, *args):
    from snowdrop.src.utils.equations import build_func
    
    arr = []; lfunc = []
    for k in args:
        fpath = os.path.join(fdir,k+'.json')
        with open(fpath,'r') as f:
            data = json.load(f)
            if k == 'lfunc':
                lfunc,_,_,_ = build_func(*data)
                arr.append(lfunc)
            elif k == "hist_rng":
                tmp = pd.to_datetime(data)
                arr.append(tmp)
            elif "_data" in k: 
                tmp = {}
                for k,x in data.items():
                    ts = pd.Series(json.loads(x))
                    index = [float(x) for x in ts.index]
                    ts.index = pd.to_datetime(index,unit='ms')
                    tmp[k] = ts.sort_index(inplace=False)
                arr.append(tmp)
            else:    
                arr.append(data)
    return arr
    
    
if __name__ == '__main__':   
     
    path = os.path.dirname(os.path.abspath(__file__))
    fpath = os.path.abspath(os.path.join(path, 'snowdrop/models/QPM/model_Lshocks.model'))
        
    txt = read_and_combine_text(fpath,conditions={"fiscalswitch": False,"wedgeswitch": True})
    print(txt)
    
    # model_file = os.path.abspath(os.path.join(working_dir,'snowdrop/models/Sirius/chile_model.xml'))
    
    # # Shocks
    # shocks = {'2025/1/1': {'e_dot_cpi':1, 'e_lgdp_gap':-1}}     
    
    # # Instantaiate a model
    # model = importModel(file_path=model_file,startDate='2000/1/1',endDate='2010/1/1',shocks=shocks) 
    # print(model)
    
    # from snowdrop.src.driver import run
    
    # output_variables = ['dot4_cpi','dot4_cpi_x','dot4_gdp','lgdp_gap','lx_gdp_gap','mci','rmc','rr','rr_gap']
    # run(model=model,output_variables=output_variables,Plot=True)
        
    # eq = "a = p*STEADY_STATE(b) + (1-b)*STEADY_STATE(b)**2"
    # eq = replaceExpressions([eq])
    # print(eq)
  
    # endog = ['E_WRL_PRODOIL_R','ISR_ACT_R','ISR_BREVAL_N','ISR_B_N','ISR_B_RAT','ISR_CFOOD_R','ISR_CNCOM_R','ISR_COIL_R','ISR_COM_FE_R','ISR_COM_RK_P','ISR_CPINCOM_P','ISR_CPIX_P','ISR_CPI_P','ISR_CURBAL_N','ISR_C_LIQ_R','ISR_C_OLG_R','ISR_C_R','ISR_C_RAT','ISR_DELTA','ISR_EPS','ISR_FACTFOOD_R','ISR_FACT_R','ISR_FXPREM','ISR_GC_N','ISR_GC_R','ISR_GC_RAT','ISR_GDEF_N','ISR_GDEF_RAT','ISR_GDEF_TAR','ISR_GDPINC_N','ISR_GDPSIZE','ISR_GDP_FE_R','ISR_GDP_N','ISR_GDP_R','ISR_GE_N','ISR_GISTOCK_R','ISR_GI_N','ISR_GI_R','ISR_GI_RAT','ISR_GNP_R','ISR_GOVCHECK','ISR_GSUB_N','ISR_GTARIFF_N','ISR_G_R','ISR_IFOODA_R','ISR_IFOOD_R','ISR_IMETALA_R','ISR_IMETAL_R','ISR_IM_R','ISR_INFCPI','ISR_INFCPIX','ISR_INFEXP','ISR_INFL','ISR_INFPIM','ISR_INFWAGE','ISR_INFWAGEEFF','ISR_INFWEXP','ISR_INT','ISR_INT10','ISR_INTC','ISR_INTCORP','ISR_INTCOST_N','ISR_INTCOST_RAT','ISR_INTGB','ISR_INTMP','ISR_INTMPU','ISR_INTNFA','ISR_INTRF','ISR_INTRF10','ISR_INTXM10','ISR_INVESTP_R','ISR_INVEST_R','ISR_INVEST_RAT','ISR_IOILA_R','ISR_IOIL_R','ISR_IT_R','ISR_IT_RAT','ISR_J','ISR_KG_R','ISR_K_R','ISR_LABH_FE_R','ISR_LABH_R','ISR_LAB_FE_R','ISR_LAB_R','ISR_LF_FE_R','ISR_LF_R','ISR_LSTAX_RAT','ISR_MET_RK_P','ISR_MKTPREM','ISR_MKTPREMSM','ISR_MPC','ISR_MPCINV','ISR_NFAREVAL_N','ISR_NFA_D','ISR_NFA_RAT','ISR_NPOPB_R','ISR_NPOPH_R','ISR_NPOP_R','ISR_NTRFPSPILL_FE_R','ISR_OILRECEIPT_N','ISR_OILSUB_N','ISR_PART','ISR_PARTH','ISR_PARTH_DES','ISR_PARTH_FE','ISR_PARTH_W','ISR_PART_DES','ISR_PART_FE','ISR_PCFOOD_P','ISR_PCOIL_P','ISR_PCW_P','ISR_PC_P','ISR_PFM_P','ISR_PFOOD_P','ISR_PGDP_P','ISR_PGDP_P_AVG','ISR_PG_P','ISR_PIMADJ_P','ISR_PIMA_P','ISR_PIM_P','ISR_PIT_P','ISR_PI_P','ISR_PMETAL_P','ISR_POIL_P','ISR_POIL_P_SUB','ISR_PRIMSUR_N','ISR_PRIMSUR_TAR','ISR_PRODFOOD_R','ISR_PSAVING_N','ISR_PXMF_P','ISR_PXMUNADJ_P','ISR_PXM_P','ISR_PXT_P','ISR_Q_P','ISR_R','ISR_R10','ISR_RC','ISR_RC0_SM','ISR_RC0_WM','ISR_RCI','ISR_RCORP','ISR_REER','ISR_RK_P','ISR_RPREM','ISR_R_NEUT','ISR_SOVPREM','ISR_SOVPREMSM','ISR_SUB_OIL','ISR_TAU_C','ISR_TAU_K','ISR_TAU_L','ISR_TAU_OIL','ISR_TAXC_N','ISR_TAXC_RAT','ISR_TAXK_N','ISR_TAXK_RAT','ISR_TAXLH_N','ISR_TAXL_N','ISR_TAXL_RAT','ISR_TAXOIL_N','ISR_TAX_N','ISR_TAX_RAT','ISR_TB_N','ISR_TFPEFFECT_R','ISR_TFPKGSPILL_FE_R','ISR_TFPSPILL_FE_R','ISR_TFP_FE_R','ISR_TFP_FE_R_AVG','ISR_TFP_R','ISR_TM','ISR_TPREM','ISR_TRANSFER_LIQ_N','ISR_TRANSFER_N','ISR_TRANSFER_OLG_N','ISR_TRANSFER_RAT','ISR_TRANSFER_TARG_N','ISR_TRANSFER_TARG_RAT','ISR_TRFPSPILL_FE_R','ISR_UFOOD_R','ISR_UNR','ISR_UNRH','ISR_UNRH_FE','ISR_UNR_FE','ISR_USA_SM','ISR_USA_WM','ISR_WAGEEFF_N','ISR_WAGEH_N','ISR_WAGE_N','ISR_WF_R','ISR_WH_R','ISR_WK_N','ISR_WO_R','ISR_W_R','ISR_W_R_AVG','ISR_XFOOD_R','ISR_XMA_R','ISR_XM_R','ISR_XT_R','ISR_XT_RAT','ISR_YCAP_N','ISR_YD_R','ISR_YLABH_N','ISR_YLAB_N','ISR_Z','ISR_Z_AVG','ISR_Z_NFA','PFOOD_P','PMETAL_P','POIL_P','RC0_ACT_R','RC0_BREVAL_N','RC0_B_N','RC0_B_RAT','RC0_CFOOD_R','RC0_CNCOM_R','RC0_COIL_R','RC0_COM_FE_R','RC0_COM_RK_P','RC0_CPINCOM_P','RC0_CPIX_P','RC0_CPI_P','RC0_CURBAL_N','RC0_C_LIQ_R','RC0_C_OLG_R','RC0_C_R','RC0_C_RAT','RC0_DELTA','RC0_EPS','RC0_FACTFOOD_R','RC0_FACTMETAL_R','RC0_FACTOIL_R','RC0_FACT_R','RC0_FXPREM','RC0_GC_N','RC0_GC_R','RC0_GC_RAT','RC0_GDEF_N','RC0_GDEF_RAT','RC0_GDEF_TAR','RC0_GDPINC_N','RC0_GDPSIZE','RC0_GDP_FE_R','RC0_GDP_N','RC0_GDP_R','RC0_GE_N','RC0_GISTOCK_R','RC0_GI_N','RC0_GI_R','RC0_GI_RAT','RC0_GNP_R','RC0_GOVCHECK','RC0_GSUB_N','RC0_GTARIFF_N','RC0_G_R','RC0_IFOODA_R','RC0_IFOOD_R','RC0_IMETALA_R','RC0_IMETAL_R','RC0_IM_R','RC0_INFCPI','RC0_INFCPIX','RC0_INFEXP','RC0_INFL','RC0_INFPIM','RC0_INFWAGE','RC0_INFWAGEEFF','RC0_INFWEXP','RC0_INT','RC0_INT10','RC0_INTC','RC0_INTCORP','RC0_INTCOST_N','RC0_INTCOST_RAT','RC0_INTGB','RC0_INTMP','RC0_INTMPU','RC0_INTNFA','RC0_INTRF','RC0_INTRF10','RC0_INTXM10','RC0_INVESTP_R','RC0_INVEST_R','RC0_INVEST_RAT','RC0_IOILA_R','RC0_IOIL_R','RC0_ISR_SM','RC0_ISR_WM','RC0_IT_R','RC0_IT_RAT','RC0_J','RC0_KG_R','RC0_K_R','RC0_LABH_FE_R','RC0_LABH_R','RC0_LAB_FE_R','RC0_LAB_R','RC0_LF_FE_R','RC0_LF_R','RC0_LSTAX_RAT','RC0_MET_RK_P','RC0_MKTPREM','RC0_MKTPREMSM','RC0_MPC','RC0_MPCINV','RC0_MROYALTIES_N','RC0_MROYALTY','RC0_NFAREVAL_N','RC0_NFA_D','RC0_NFA_RAT','RC0_NPOPB_R','RC0_NPOPH_R','RC0_NPOP_R','RC0_NTRFPSPILL_FE_R','RC0_OILPAY_N','RC0_OILRECEIPT_N','RC0_OILSHARF','RC0_OILSUB_N','RC0_PART','RC0_PARTH','RC0_PARTH_DES','RC0_PARTH_FE','RC0_PARTH_W','RC0_PART_DES','RC0_PART_FE','RC0_PCFOOD_P','RC0_PCOIL_P','RC0_PCW_P','RC0_PC_P','RC0_PFM_P','RC0_PFOOD_P','RC0_PGDP_P','RC0_PGDP_P_AVG','RC0_PG_P','RC0_PIMADJ_P','RC0_PIMA_P','RC0_PIM_P','RC0_PIT_P','RC0_PI_P','RC0_PMETAL_P','RC0_POIL_P','RC0_POIL_P_SUB','RC0_PRIMSUR_N','RC0_PRIMSUR_TAR','RC0_PRODFOOD_R','RC0_PRODMETAL_R','RC0_PRODOIL_R','RC0_PSAVING_N','RC0_PXMF_P','RC0_PXMUNADJ_P','RC0_PXM_P','RC0_PXT_P','RC0_Q_P','RC0_R','RC0_R10','RC0_RC','RC0_RCI','RC0_RCORP','RC0_REER','RC0_RK_P','RC0_ROYALTIES_N','RC0_ROYALTIES_RAT','RC0_ROYALTY','RC0_RPREM','RC0_R_NEUT','RC0_SOVPREM','RC0_SOVPREMSM','RC0_SUB_OIL','RC0_TAU_C','RC0_TAU_K','RC0_TAU_L','RC0_TAU_OIL','RC0_TAXC_N','RC0_TAXC_RAT','RC0_TAXK_N','RC0_TAXK_RAT','RC0_TAXLH_N','RC0_TAXL_N','RC0_TAXL_RAT','RC0_TAXOIL_N','RC0_TAX_N','RC0_TAX_RAT','RC0_TB_N','RC0_TFPEFFECT_R','RC0_TFPKGSPILL_FE_R','RC0_TFPSPILL_FE_R','RC0_TFP_FE_R','RC0_TFP_FE_R_AVG','RC0_TFP_R','RC0_TM','RC0_TPREM','RC0_TRANSFER_LIQ_N','RC0_TRANSFER_N','RC0_TRANSFER_OLG_N','RC0_TRANSFER_RAT','RC0_TRANSFER_TARG_N','RC0_TRANSFER_TARG_RAT','RC0_TRFPSPILL_FE_R','RC0_UFOOD_R','RC0_UMETAL_R','RC0_UNR','RC0_UNRH','RC0_UNRH_FE','RC0_UNR_FE','RC0_UOIL_R','RC0_USA_SM','RC0_USA_WM','RC0_WAGEEFF_N','RC0_WAGEH_N','RC0_WAGE_N','RC0_WF_R','RC0_WH_R','RC0_WK_N','RC0_WO_R','RC0_W_R','RC0_W_R_AVG','RC0_XFOOD_R','RC0_XMA_R','RC0_XMETAL_R','RC0_XM_R','RC0_XOIL_R','RC0_XT_R','RC0_XT_RAT','RC0_YCAP_N','RC0_YD_R','RC0_YLABH_N','RC0_YLAB_N','RC0_Z','RC0_Z_AVG','RC0_Z_NFA','RPFOOD','RPMETAL','RPMETAL_ADJ','RPMETAL_AVG','RPOIL','RPOIL_ADJ','RPOIL_AVG','USA_ACT_R','USA_BREVAL_N','USA_B_N','USA_B_RAT','USA_CFOOD_R','USA_CNCOM_R','USA_COIL_R','USA_COM_FE_R','USA_COM_RK_P','USA_CPINCOM_P','USA_CPIX_P','USA_CPI_P','USA_CURBAL_N','USA_C_LIQ_R','USA_C_OLG_R','USA_C_R','USA_C_RAT','USA_DELTA','USA_EPS','USA_FACTFOOD_R','USA_FACTMETAL_R','USA_FACTOIL_R','USA_FACT_R','USA_FXPREM','USA_GC_N','USA_GC_R','USA_GC_RAT','USA_GDEF_N','USA_GDEF_RAT','USA_GDEF_TAR','USA_GDPINC_N','USA_GDPSIZE','USA_GDP_FE_R','USA_GDP_N','USA_GDP_R','USA_GE_N','USA_GISTOCK_R','USA_GI_N','USA_GI_R','USA_GI_RAT','USA_GNP_R','USA_GOVCHECK','USA_GSUB_N','USA_GTARIFF_N','USA_G_R','USA_IFOODA_R','USA_IFOOD_R','USA_IMETALA_R','USA_IMETAL_R','USA_IM_R','USA_INFCPI','USA_INFCPIX','USA_INFEXP','USA_INFL','USA_INFPIM','USA_INFWAGE','USA_INFWAGEEFF','USA_INFWEXP','USA_INT','USA_INT10','USA_INTC','USA_INTCORP','USA_INTCOST_N','USA_INTCOST_RAT','USA_INTGB','USA_INTMP','USA_INTMPU','USA_INTNFA','USA_INTRF','USA_INTRF10','USA_INTXM10','USA_INVESTP_R','USA_INVEST_R','USA_INVEST_RAT','USA_IOILA_R','USA_IOIL_R','USA_ISR_SM','USA_ISR_WM','USA_IT_R','USA_IT_RAT','USA_J','USA_KG_R','USA_K_R','USA_LABH_FE_R','USA_LABH_R','USA_LAB_FE_R','USA_LAB_R','USA_LF_FE_R','USA_LF_R','USA_LSTAX_RAT','USA_MET_RK_P','USA_MKTPREM','USA_MKTPREMSM','USA_MPC','USA_MPCINV','USA_MROYALTIES_N','USA_MROYALTY','USA_NFAREVAL_N','USA_NFA_D','USA_NFA_RAT','USA_NPOPB_R','USA_NPOPH_R','USA_NPOP_R','USA_NTRFPSPILL_FE_R','USA_OILPAY_N','USA_OILRECEIPT_N','USA_OILSHARF','USA_OILSUB_N','USA_PART','USA_PARTH','USA_PARTH_DES','USA_PARTH_FE','USA_PARTH_W','USA_PART_DES','USA_PART_FE','USA_PCFOOD_P','USA_PCOIL_P','USA_PCW_P','USA_PC_P','USA_PFM_P','USA_PFOOD_P','USA_PGDP_P','USA_PGDP_P_AVG','USA_PG_P','USA_PIMADJ_P','USA_PIMA_P','USA_PIM_P','USA_PIT_P','USA_PI_P','USA_PMETAL_P','USA_POIL_P','USA_POIL_P_SUB','USA_PRIMSUR_N','USA_PRIMSUR_TAR','USA_PRODFOOD_R','USA_PRODMETAL_R','USA_PRODOIL_R','USA_PSAVING_N','USA_PXMF_P','USA_PXMUNADJ_P','USA_PXM_P','USA_PXT_P','USA_Q_P','USA_R','USA_R10','USA_RC','USA_RC0_SM','USA_RC0_WM','USA_RCI','USA_RCORP','USA_REER','USA_RK_P','USA_ROYALTIES_N','USA_ROYALTIES_RAT','USA_ROYALTY','USA_RPREM','USA_R_NEUT','USA_SOVPREM','USA_SOVPREMSM','USA_SUB_OIL','USA_TAU_C','USA_TAU_K','USA_TAU_L','USA_TAU_OIL','USA_TAXC_N','USA_TAXC_RAT','USA_TAXK_N','USA_TAXK_RAT','USA_TAXLH_N','USA_TAXL_N','USA_TAXL_RAT','USA_TAXOIL_N','USA_TAX_N','USA_TAX_RAT','USA_TB_N','USA_TFPEFFECT_R','USA_TFPKGSPILL_FE_R','USA_TFPSPILL_FE_R','USA_TFP_FE_R','USA_TFP_FE_R_AVG','USA_TFP_R','USA_TM','USA_TPREM','USA_TRANSFER_LIQ_N','USA_TRANSFER_N','USA_TRANSFER_OLG_N','USA_TRANSFER_RAT','USA_TRANSFER_TARG_N','USA_TRANSFER_TARG_RAT','USA_TRFPSPILL_FE_R','USA_UFOOD_R','USA_UMETAL_R','USA_UNR','USA_UNRH','USA_UNRH_FE','USA_UNR_FE','USA_UOIL_R','USA_WAGEEFF_N','USA_WAGEH_N','USA_WAGE_N','USA_WF_R','USA_WH_R','USA_WK_N','USA_WO_R','USA_W_R','USA_W_R_AVG','USA_XFOOD_R','USA_XMA_R','USA_XMETAL_R','USA_XM_R','USA_XOIL_R','USA_XT_R','USA_XT_RAT','USA_YCAP_N','USA_YD_R','USA_YLABH_N','USA_YLAB_N','USA_Z','USA_Z_AVG','USA_Z_NFA','WRL_GDP_FE_METAL_R','WRL_GDP_FE_OIL_R','WRL_GDP_FE_R','WRL_GDP_METAL_R','WRL_GDP_OIL_R','WRL_GDP_R','WRL_PRODFOOD_R','WRL_PRODMETAL_R','WRL_PRODOIL_R','WRL_XFOOD_R','WRL_XMETAL_R','WRL_XOIL_R','WTRADE_FOOD_N','WTRADE_FOOD_R','WTRADE_METAL_N','WTRADE_METAL_R','WTRADE_M_N','WTRADE_M_R','WTRADE_OIL_N','WTRADE_OIL_R']  
    # eq = "1+ISR_INTGB = ((1+ISR_INT-E_ISR_INT)/(1+ISR_MKTPREM)+E_ISR_INT)**(1-ISR_INTGB1)*(((1+ISR_INTXM10(-2))**(1/3))*((1+ISR_INTXM10(-1))**(1/3))*((1+ISR_INTXM10)**(1/3)))**ISR_INTGB1*(1+RES_ISR_INTGB)*(1+E_ISR_INTGB)"
    # mod_eqs, new_endog, map_new_endog, leads, lags = fixEquations([eq],endog,tagBeg="(",tagEnd=")")
    
    # print()
    # print(eq)
    # print()
    # print(mod_eqs[0])
    
    
