"""
Program for testing Potential Output Model Estimation.

Created on Tue Mar 13 15:58:11 2018
@author: A.Goumilevski
"""
import os
import sys

path = os.path.dirname(os.path.abspath(__file__))
working_dir = os.path.abspath(path+"/../../..")
sys.path.append(working_dir)
os.chdir(working_dir)

def estimate(fname='TOY/Ireland2004.yaml',fmeas='supplements/data/gpr_1948.csv'):

    from snowdrop.src.driver import importModel, estimate

    #fname = 'TOY/Ireland2004.yaml' # Ireland NK model example
    fout = os.path.abspath(os.path.join(working_dir,'supplements/data/results.csv')) # Results are saved in this file
    output_variables = None  # List of variables that will be plotted or displayed

    # Path to measurement data
    meas = os.path.abspath(os.path.join(working_dir,fmeas))
    
    # Path to model file
    file_path = os.path.abspath(os.path.join(working_dir, 'supplements/models', fname))

    # Create model object
    model = importModel(fname=file_path,Solver="Klein",
                        # Available Kalman filters: Durbin_Koopman,Particle 
                        Filter="Durbin_Koopman",Smoother="Durbin_Koopman",
                        #Filter="Non_Diffuse_Filter",Smoother="BrysonFrazier",
                        # Available priors: StartingValues,Diffuse,Equilibrium
                        Prior="Diffuse",
                        # Available methods: pymcmcstat-mh,pymcmcstat-am,pymcmcstat-dr,pymcmcstat-dram,emcee,particle-pmmh,particle-smc
                        SamplingMethod="pymcmcstat-dram",
                        measurement_file_path=meas,use_cache=False)
    
    # Estimate model parameters
    estimate(model=model,output_variables=output_variables,
             #estimate_Posterior=True,
             estimate_ML=True,
             # Available algorithms: Nelder-Mead, L-BFGS-B, TNC, SLSQP, Powell, and trust-constr
             algorithm="SLSQP",
             sample=True,Ndraws=10000,Niter=1000,burn=100,Parallel=False,
             fout=fout,Output=False,Plot=True)
   
if __name__ == '__main__':
    """
    The main test program.
    """
    estimate()