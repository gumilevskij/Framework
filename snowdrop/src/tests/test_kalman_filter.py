"""
Program for testing Kalman Filter.

Created on Tue Mar 13 15:58:11 2018
@author: A.Goumilevski
"""
import os
import sys

path = os.path.dirname(os.path.abspath(__file__))
working_dir = os.path.abspath(path+"/../../..")
sys.path.append(working_dir)
os.chdir(working_dir)

def test(fname='TOY/Ireland2004.yaml',fmeas='snowdrop/data/gpr_1948.csv'):
    from snowdrop.src.driver import importModel
    from snowdrop.src.driver import kalman_filter

    #fname = 'TOY/Ireland2004.yaml' # Peter Ireland model filter example
    #fname = 'TOY/MVF_US.yaml' # Multivariate Kalman filter example
    #fname = 'TOY/Test.yaml' # Car trajectory tracking example
    fout = 'data/results.csv' # Results are saved in this file
    output_variables = None #['DLGDP','LGDP','LGDP_BAR','PIE','UNR','UNR_GAP']   # List of variables that will be plotted or displayed
    decomp = None #['pie','r']

    # Path to measurement data
    meas = os.path.abspath(os.path.join(working_dir, fmeas))
    #meas = os.path.abspath(os.path.join(working_dir, 'snowdrop/data/dataForKalman.csv'))
    
    # Path to model file
    file_path = os.path.abspath(os.path.join(working_dir, 'snowdrop/models', fname))

    # Instantiate model object
    model = importModel(fname=file_path,
                        Solver="BinderPesaran",
                        #Solver="Klein,AndersonMoore,LBJ,ABLR,BinderPesaran,Villemot  
                        #Filter="Particle",Smoother="Durbin_Koopman",
                        #Filter="Unscented", Smoother="BrysonFrazier",
                        #Filter="Durbin_Koopman", Smoother="Durbin_Koopman",
                        Filter="Diffuse",Smoother="Diffuse",
                        Prior="Diffuse") #Prior="StartingValues",  Prior="Diffuse", 
          

    ### Run Kalman filter
    model.setStartingValues(hist=meas,bTreatMissingObs=False,debug=False)
    yy,dates,epsilonhat,etahat = kalman_filter(model=model,meas=meas,fout=fout,Output=True,Plot=True)
    

if __name__ == '__main__':
    """
    The main test program.
    """
    test()