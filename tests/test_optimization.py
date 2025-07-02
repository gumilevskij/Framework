"""
Program for testing Computable General Equilibrium models.

@author: A.Goumilevski
"""
import os
import sys

path = os.path.dirname(os.path.abspath(__file__))
working_dir = os.path.abspath(path + "/..")
sys.path.append(working_dir)
os.chdir(working_dir)

   
def test(fname='OPT/armington.yaml'):

    from snowdrop.src.driver import optimize

    Plot = not 'pytest' in sys.modules
    
    #fname = 'OPT/melitz.yaml' # Melitz model example 
    #fname = 'OPT/krugman.yaml' # Krugman model example 
    #fname = 'OPT/armington.yaml' # Armington model example 
    #fname = 'OPT/transport.yaml' # Transportation expenses minimization example 
    
    if Plot:
        plot_variables = ["c","Y","Q","P"]
    else:
        plot_variables = None

    # Path to model file.
    file_path = os.path.abspath(os.path.join(working_dir, 'supplements/models', fname))

    fout = os.path.abspath(os.path.join(working_dir,'supplements/data/OPT/opt.csv')) # Results are saved in this file

    # Run optimization routine.
    optimize(fpath=file_path,fout=fout,Output=True,plot_variables=plot_variables,model_info=False)
 
if __name__ == '__main__':
     """
     The main test program.
     """
     test()