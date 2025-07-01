import os, pandas as pd

working_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),"../../.."))
os.chdir(working_dir)

from snowdrop.src.driver import importModel, run, estimate
#from snowdrop.src.utils.util import simulationRange

ESTIMATE = False

if __name__ == '__main__':
    """
    The main test program.
    """
    ######### Load Model
    fname = "COVID19/gsw_model.yaml" # GSW model"
    # Path to model file
    model_path = os.path.abspath(os.path.join(working_dir, 'supplements/models', fname))
    
    # Create model object
    model = importModel(fname=model_path,Solver="Klein")
    model.anticipate = True
    
    # Path to data
    meas = os.path.abspath(os.path.join(working_dir, 'supplements/data/COVID19/country_data.xlsx'))
        
    output_variables = ['y','kpf','lab','c','unempl','inve','pinf','r','labstar'] 
    decomp = ['y','lab','inve','pinf']
    shock_values = model.options["shock_values"]
        
    var_names = model.symbols['variables']
    n = len(var_names)
    ind = [i for i in range(n) if var_names[i] in output_variables]
    
    if ESTIMATE:
        
        # Create model object
        model = importModel(fname=model_path,Solver="Klein",Filter="Durbin_Koopman",Smoother="Durbin_Koopman",Prior="Diffuse",model_info=True)
        model.anticipate = True
            
        model.options["range"] = [[2008,1,1],[2023,1,1]]
        model.options["filter_range"] = [[2008,1,1],[2019,12,31]]
          
        ###   Estimate model parameters
        yy,dates,epsilonhat,etahat = estimate(model=model,meas=meas,Plot=True,Output=True,sample=False,estimate_ML=True,output_variables=output_variables)
     
        # Save model parameters
        params = model.calibration["parameters"]
    
    
    # Create model object with perfect foresight solver
    model = importModel(fname=model_path,Solver="LBJ",model_info=False)
    
    if ESTIMATE:
        model.calibration["parameters"] = params
            
    # Set two periods shocks.
    model.options["periods"] = [1,2]
    model.options["shock_values"] = [[90,0,0,0,0,0,0,0,0],[90,0,0,0,0,0,0,0,0]]
     
    
    ######### Run forecast simulations without tunes.
    
    if False:
        y,dates = run(model=model,Plot=True,Output=True,output_variables=output_variables,decomp_variables=decomp,Tmax=9)
        
        m = {}
        for i in ind:
            m[var_names[i]] = pd.Series(y[:len(dates),i],dates)
        df = pd.DataFrame(m)
        #print('Forecast:')
        #print(df)
    
    
    ######### Run forecast simulations with tunes.
        
    # Create model object
    model = importModel(fname=model_path,Solver="Klein",model_info=False)
    model.anticipate = True
    
    
    if ESTIMATE:
        model.calibration["parameters"] = params
                
    # Set two periods shocks.
    model.options["periods"] = [1,2]
    model.options["shock_values"] = [[90,0,0,0,0,0,0,0,0],[90,0,0,0,0,0,0,0,0]]
     
    m = {"y": pd.Series([2.5,2.5,2.5],[3,4,5])}
    shock_names  = ['ey']
    model.swap(var1=m,var2=shock_names)
    
    fcast_tune,dates = run(model=model,Plot=True,Output=True,output_variables=output_variables,decomp_variables=decomp,Tmax=9)
    
    m = {}
    for i in ind:
        m[var_names[i]] = pd.Series(fcast_tune[:len(dates),i],dates)
    df = pd.DataFrame(m)
    print('Forecast with Tunes:')
    print(df["y"][:11])











