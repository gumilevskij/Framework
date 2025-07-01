import os, sys
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.widgets import Slider

path = os.path.dirname(os.path.abspath(__file__))
working_dir = os.path.abspath(path + "/../../..")
sys.path.append(working_dir)
os.chdir(working_dir)

from snowdrop.src import driver
from matplotlib.animation import FuncAnimation

model = None

def run(val=None):
    global model
    
    decomp = ['PDOT','RR','RS','Y'] # List of variables for which decomposition plots are produced

    # Path to model file
    file_path = os.path.abspath(os.path.join(working_dir,'supplements/models/TOY/JLMP98.yaml'))
    
    # Create model object
    if model is None:
        model = driver.importModel(fname=file_path,Solver="LBJ")

    var_names = model.symbols['variables']
    shock_names = model.symbols['shocks']
    param_names = model.symbols['parameters']
    param_values = model.calibration['parameters'].copy()
    var_labels = model.symbols.get("variables_labels",{})
    variables = [var_labels[k] for k in var_labels if k in decomp]
    
    ind = param_names.index('p_pdot1')
    param_values[ind] = float(val)
    model.calibration['parameters'] = param_values
    
    # Function that runs simulations, model parameters estimation, MCMC sampling, etc...
    y,dates = driver.run(model=model,decomp_variables=variables,Output=False,Plot=False,use_cache=True)

    print("Plotting Endogenous Variables")
    plot(y=y,rng=dates,variables_names=variables,T=model.T)
    
def main():
    # Create initial plot
    fig, ax = plt.subplots(figsize=[10,8])
    plt.subplots_adjust(bottom=0.1)
    plt.tick_params(bottom=False, labelbottom=False) #remove ticks
    plt.tick_params(left=False, labelleft=False) #remove ticks
    plt.box(False)
    
    # Create slider
    axfreq = plt.axes([0.45,0.1,0.35,0.03])
    param_slider = Slider(axfreq,'Inflation persistence parameter (p_pdot1)  ',0.1,1.0,valinit=0.1,valstep=0.1)
    
    # Update plot function
    def update(val):
        run(val=param_slider.val)
        fig.canvas.draw_idle()
    
    # Connect slider to update function
    param_slider.on_changed(update)
    
    # Create the animation
    #ani = FuncAnimation(fig, run, interval=50)
    plt.show()
    
def plot(y,rng,variables_names,T=14):
    rows, columns = 2,2

    for m,n in enumerate(variables_names):
        ax = plt.subplot(rows,columns,1+m)
        ax.tick_params(axis='both', labelsize=12)
        series = pd.Series(data=y[:T,m], index=rng[:T])
        series.plot(ax=ax,lw=1,label=n)
        plt.title(n,fontsize=12)
        plt.grid(True)
        
    # Adjust the subplot parameters to place the chart in the upper part
    plt.subplots_adjust(bottom=0.2)
        
    
    plt.show()
        
if __name__ == '__main__':
    #run(fig=plt.subplots(figsize=[12,8]))
    main()
