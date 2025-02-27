"""
Program for testing various models.

Created on Tue Mar 13 15:58:11 2018
@author: A.Goumilevski
"""
import os,sys
import pandas as pd

path = os.path.dirname(os.path.abspath(__file__))
working_dir = os.path.abspath(path + "/../../..")
sys.path.append(working_dir)
os.chdir(working_dir)

from snowdrop.src import driver
from snowdrop.src.utils.util import getExogenousSeries
from snowdrop.src.numeric.solver.util import checkSolution

def test_1(fname='models/TOY/RBC.yaml',decompose=False,title=None):

    fout = 'data/test.csv' # Results are saved in this file
    decomp = None # List of variables for which decomposition plots are produced
    output_variables = None #['PDOT','RR','RS','Y','PIE','LGDP','G','L_GDP','L_GDP_GAP']
    if not decompose:
        output_variables,decomp = decomp,output_variables
        
    # Path to model file
    file_path = os.path.abspath(os.path.join(working_dir, 'snowdrop', fname))
    
    # Function that runs simulations, model parameters estimation, MCMC sampling, etc...
    y,dates = \
    driver.run(fname=file_path,fout=fout,decomp_variables=decomp,
             output_variables=output_variables,header=title,
             Output=True,Plot=True,Solver="LBJ",
             graph_info=False,use_cache=False)


def test_2(fname='models/TOY/JLMP98.yaml',decompose=True,title=None):
    
    # Path to model file
    file_path = os.path.abspath(os.path.join(working_dir,'snowdrop', fname))
        
    # Create model object
    model = driver.importModel(fname=file_path,use_cache=False)
    
    # # Solving non-linear model by linear solver.  Just kidding...
    # from snowdrop.src.utils.equations import topology
    # model.isLinear = True
    # topology(model) 

    variables = model.symbols['variables']
    # Change model parameters
    cal = {'g':0.049,'p_pdot1':0.414,'p_pdot2': 0.196,'p_pdot3': 0.276,
            'p_rs1':3,'p_y1':0.304,'p_y2':0.098,'p_y3':0.315}
    model.calibration['parameters'] = list(cal.values())
    
    # Shocks
    model.options["periods"] = [1]
    model.options["shock_values"] = [0.02]
    
    # # User tune
    # m = {'PDOT': pd.Series([0.01,0.01],[2,3])}
    # shock_names = ['epdot']
    # model.swap(var1=m,var2=shock_names,reset=False)
    
    # Exogenous variables. Revision of Monetary Policy Rate.
    # The last exogenous process value is set for the rest of time periods
    exo = pd.Series([0.03,0.0],[4,5])
    exog_data = {'exo': exo}
    model.symbolic.exog_data = exog_data     
    model.calibration["exogenous"] = getExogenousSeries(model)
    
    # List of variables for which decomposition plots are produced
    decomp = ['PDOT','RR','RS','Y']
    output_variables = None #['PDOT','RR','RS','Y','PIE','LGDP','G','L_GDP','L_GDP_GAP']
    if not decompose:
        output_variables,decomp = decomp,output_variables
    
    # Function that runs simulations, model parameters estimation, MCMC sampling, etc...
    y,dates = driver.run(model=model,decomp_variables=decomp,
                         output_variables=output_variables, #Solver="ABLR",
                         header=title,Output=True,Plot=True)
    
    # Residuals of unrestricted equatinos without user's tunes
    err = checkSolution(model,periods=model.options["periods"],y=y)
    print(f"\nResiduals:\n{'                '.join(variables)}\n",err)
    
    # Print solution
    n = len(dates)
    df = pd.DataFrame(y[:n],dates)
    df.columns = model.symbols["variables"]
    print(f"\nSolution:\n {df}") 
    
if __name__ == '__main__':
    """
    The main test program.
    """
    fname = 'models/TOY/JLMP98.yaml'   # Simple monetary policy example
    #fname = 'models/TOY/RBC.yaml'      # Simple RBC model with deterministic shocks
    #fname = 'models/TOY/RBC1.yaml'     # Simple RBC model with stochastic shocks
    #fname = 'models/templates/five_regions.yaml'   # Template example
    #fname = 'models/templates/countries.yaml'   # Template example
    
    decompose=True
    
    test_1(title='RBC Model')
    test_2(fname=fname,decompose=decompose,title='Shock to Output and Revision of Nominal Interest Rate')
    
