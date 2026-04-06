"""
Program for testing model parameters estimation.

Created on Tue Mar 13, 2018
@author: A.Goumilevski
"""
import os
import sys

path = os.path.dirname(os.path.abspath(__file__))
working_dir = os.path.abspath(path+"/..")
sys.path.append(working_dir)
os.chdir(working_dir)

def test(fname='TOY/Ireland2004.yaml',fmeas='supplements/data/gpr_1948.csv'):

    from snowdrop.src.driver import importModel, estimate

    #fname = 'TOY/Ireland2004.yaml' # Ireland NK model example
    fout = 'output/data/results.csv' # Results are saved in this file
    output_variables = None  # List of variables that will be plotted or displayed

    # Path to measurement data
    meas = os.path.abspath(os.path.join(working_dir,fmeas))
    
    # Path to model file
    file_path = os.path.abspath(os.path.join(working_dir, 'supplements/models', fname))

    # Create model object
    model = importModel(fname=file_path,Solver="Klein",
                        Filter="Durbin_Koopman",Smoother="Durbin_Koopman",
                        Prior="Diffuse",measurement_file_path=meas)
    
    # Estimate model parameters
    estimate(model=model,estimate_ML=True,
             fout=fout,output_variables=output_variables,
             Output=False,Plot=True)
   
if __name__ == '__main__':
    """
    The main test program.
    """
    test()