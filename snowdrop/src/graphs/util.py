# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 14:28:27 2019

@author: A.Goumilevski
"""
import os
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from math import pi, ceil
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.dates as dates
import matplotlib.style as style
from matplotlib import ticker
from pandas.plotting import register_matplotlib_converters
import warnings
from snowdrop.src.misc.termcolor import cprint
warnings.filterwarnings("ignore", category=DeprecationWarning) 

register_matplotlib_converters()

path = os.path.dirname(os.path.abspath(__file__))
working_dir = os.path.abspath(path+"/../../../")   

#STYLE = "seaborn-darkgrid"
STYLE = "seaborn-v0_8-whitegrid"
K1 = 0; K2 = 0; K3 = 0

def plotEigenValues(path_to_dir,ev,show=True,save=True,ext="png"):
    """
    Plot eigen values.
    
    Parameters:
        :param path_to_dir: Path to the folder where figures are saved.
        :type path_to_dir: str.
        :param ev: Eigen values
        :type ev: np.array.
        :param show: Boolean variable.  If set to True shows graphs.
        :type show: bool.
        :param save: Boolean variable.  If set to True saves graphs.
        :type save: bool.colors
        :param ext: Format of the saved file.
        :type ext: str.
    """
    if save:
        figsize=(8, 4)
    else:
        figsize=(4, 2)
    style.use(STYLE)
    
    fig = plt.figure(figsize=figsize)
    plt.scatter(ev.real,ev.imag,linewidths=4)
    pts = np.linspace(-pi, pi, 100)
    plt.plot(np.cos(pts),np.sin(pts),'k')
    plt.plot([-1, 1],[0, 0],'k')
    plt.plot([0, 0],[-1, 1],'k')
    plt.box(True)
    plt.grid(True)
    plt.title('Eigen Values',fontsize = 'x-large')
    plt.xlabel('Real')
    plt.ylabel('Imaginary')
    
    if save:
        plt.savefig(os.path.join(path_to_dir,'Eigen_values.'+ext))
    if show: 
        plt.show(block=False)
        plt.close(fig)
     
 
def plotDecomposition(path_to_dir,model,y,variables_names,decomp_variables,periods,isKF=False,header=None,sizes=(2,2),Tmax=50,rng=None,figsize=None,show=True,save=True,ext="png"):
    """
    Plot contributions of different soures to endogenous variable.
    
    Parameters:
        :param path_to_dir: Path to the folder where figures are saved.
        :type path_to_dir: str.
        :param model: Model object.
        :type model: Model.
        :param y: Data array.
        :type y: array.
        :param variables_names: Variable names.
        :type variables_names: list.
        :param decomp_variables: decomposition variables names.
        :type decomp_variables: list.
        :param periods: List of periods of shocks.
        :type periods: list.
        :param header: Plot header.
        :type header: str.
        :param isKF: True if shocks are obtained by Kalman Filter.
        :type isKF: bool.
        :param sizes: Subplots dimensions.
        :type sizes: tuple.
        :param Tmax: Maximum number of periods to display in graphs.
        :type Tmax: int.
        :param rng: Time range.
        :type rng: list.
        :param sizes: Figure sizes.
        :type sizes: tuple.
        :param show: Boolean variable.  If set to True shows graphs.
        :type show: bool.
        :param save: Boolean variable.  If set to True saves graphs.
        :type save: bool.
        :param ext: Format of the saved file.
        :type ext: str.
        
    """
    import numpy as np
    from snowdrop.src.utils.util import deleteFiles,correctLabel
    from snowdrop.src.numeric.solver.util import decompose
    from snowdrop.src.utils.util import getMap
    from textwrap import wrap
    
    global K2
    K2 += 1

    var_labels = model.symbols.get("variables_labels",{})
    shocks = model.symbols["shocks"]
    exog = model.symbols["exogenous"]
    
    figs = []
    style.use(STYLE)
    
    years = dates.YearLocator()   # every year
    yearsFmt = dates.DateFormatter('%Y')
    barWidth = 0.5
    
    if decomp_variables is None:
        decomp_variables = variables_names
        
    decomp_variables = [x for x in decomp_variables if not "_plus_" in x and not "_minus_" in x]
     
    if not sizes is None:
        rows, columns = sizes
    else:
        if len(decomp_variables) == 1:
            rows = columns = 1
        elif len(decomp_variables) <= 2:   
            rows,columns = 2,1
        elif len(decomp_variables) <= 4:   
            rows,columns = 2,2
        elif len(decomp_variables) <= 9:   
            rows,columns = 3,3
        elif len(decomp_variables) <= 12:   
            rows,columns = 4,3
        else:
            rows, columns = sizes
            
    # Delete files
    if K2==1:
        deleteFiles(path_to_dir,"Decomposition")
    kmax = K2-1
    
    file_path = os.path.abspath(os.path.join(path_to_dir,"../data/dictionary/symbols_labels.csv"))
    if os.path.exists(file_path):
        symbolsMap = getMap(file_path)
    else:
        symbolsMap = {}
    
    alpha = 1.0
    fig = None
    if figsize is None:
        figsize=(10, 8)
    k = 0; m = 0
    for n in decomp_variables:
        if n in variables_names:
            c = 0
            if rng is None:
                T = min(len(y),Tmax)
            else:
                T = min(len(y),len(rng),Tmax)
            # Get decomposition data
            y = y[:T]
            data,var = decompose(model,y,variables_names,n,T,periods,isKF)
            var = [var_labels[k] if k in var_labels else k for k in var+shocks+exog]
            colors,m_colors = getColors(var)
            if len(data) > 0:
                j = [i for i,val in enumerate(variables_names) if val==n][0]
                if k%(rows*columns) == 0:
                    m = 0
                    fig = plt.figure(figsize=figsize)
                k += 1
                m += 1
                ax = plt.subplot(rows,columns,m)
                if rng is None:
                    ind = np.arange(T)   
                    series = pd.Series(data=y[:,j])
                else:
                    ind = rng[:T]
                    series = pd.Series(data=y[:,j], index=ind)
                h0 = series.plot(ax=ax,lw=3,color='k',label="TOTAL")
                
                # PLot bar graphs
                bottomPlus = None
                bottomMinus = None
                labels = ["TOTAL"] + [correctLabel(k) for k in data]
                for key in data:
                    z = np.array(data[key])
                    zPlus = 0.5*(z+np.abs(z))
                    zMinus = 0.5*(z-np.abs(z)) 
                    c += 1
                    c = c % len(colors)
                    color = colors[c]
                    # if key in m_colors: 
                    #     color = m_colors[key]
                    # Stacked bar plots
                    if bottomPlus is None and bottomMinus is None:
                        ax.bar(ind, zPlus, width=barWidth, align='center', alpha=alpha, color=color,edgecolor="black")
                        ax.bar(ind, zMinus, label = correctLabel(key), width=barWidth, align='center', alpha=alpha, color=color,edgecolor="black")
                        bottomPlus = zPlus
                        bottomMinus = zMinus
                    else:
                        ax.bar(ind, zPlus, bottom=bottomPlus, width=barWidth, align='center', alpha=alpha, color=color,edgecolor="black")
                        ax.bar(ind, zMinus, label = correctLabel(key), bottom=bottomMinus, width=barWidth,align='center',alpha=alpha,color=colors[c],edgecolor="black")
                        bottomPlus += zPlus
                        bottomMinus += zMinus
                ax.legend(loc="best",fontsize = 'medium')
                if n in symbolsMap:
                    plt.title("\n".join(wrap(symbolsMap[n])),fontsize = 'x-large')
                else:
                    var_label = var_labels[n] if n in var_labels else n
                    plt.title(var_label,fontsize = 'x-large')
                plt.grid(True)
                # format the ticks
                ax.xaxis.set_minor_locator(years)
                ax.xaxis.set_minor_formatter(yearsFmt)
                #plt.xticks(ind, [''], rotation=90)
    
        if k%(rows*columns) == 0 and not fig is None:
            fig.set_tight_layout(True)
            if not header is None:
                make_space_above(fig)
                fig.suptitle(header,fontsize=17,fontweight='normal')
            figs.append(fig)
            if save:
                plt.savefig(os.path.join(path_to_dir,'Decomposition_' + str(kmax+ceil(k/4)) + '.' + ext))
            if show: 
                plt.show(block=False)
                plt.close(fig)
                fig = None
            
    if k%(rows*columns)>0 and not fig is None:  
        fig.set_tight_layout(True)
        if not header is None:
            make_space_above(fig)
            fig.suptitle(header,fontsize=17,fontweight='normal')
        figs.append(fig) 
        if save:
            plt.savefig(os.path.join(path_to_dir,'Decomposition_' + str(kmax+ceil((1+k)/4)) + '.' + ext))
        if show: 
            plt.show(block=False)
            plt.close(fig)
            fig = None

    plt.close('all')
    return figs
    

def plotHistogram(path_to_dir,priors,samples,names,parameters,header=None,show=True,save=True,sizes=(2,2),ext="png"):
    """
    Plot histogram and kernel density function.
      
    Parameters:
        :param path_to_dir: Path to the folder where figures are saved.
        :type path_to_dir: str.
        :param priors: Model parameters priors.
        :type priors: dict.
        :param samples: Data array.
        :type samples: np.array.
        :param names: List of parameters names.
        :type names: list.
        :param parameters: List of parameters values.
        :type parameters: list.
        :param header: Plot header.
        :type header: str.
        :param sizes: Figure sizes.
        :type sizes: tuple.
        :param show: Boolean variable.  If set to True shows graphs.
        :type show: bool.
        :param save: Boolean variable.  If set to True saves graphs.
        :type save: bool.
        :param ext: Format of the saved file.
        :type ext: str.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    #from scipy import stats
    from snowdrop.src.utils.util import deleteFiles
    from snowdrop.src.utils.distributions import pdf
    from snowdrop.src.utils.distributions import getHyperParameters
        
    global K3
    K3 += 1
    # Delete files
    if K3==1:
        deleteFiles(path_to_dir,"Density")
        deleteFiles(path_to_dir,"Corner")
    kmax = K3-1
    
    size = len(parameters)
    if save:
        figsize=(16, 10)
    else:
        figsize=(10, 8)
   
    style.use(STYLE)
    
    rows,columns = sizes
    
    k = 0; m = 0
    for i in range(size):
        if k%(rows*columns) == 0:
            m = 0
            fig = plt.figure(figsize=figsize)
        k += 1
        m += 1
        ax = plt.subplot(rows,columns,m)
        
        # Get priors
        name = names[i]
        prior = priors[name]
        pars  = np.copy(prior['parameters'])
        distr = prior['distribution']
        lb,ub = pars[1],pars[2]
        
        # Get posteriors    
        data = samples[:,i]
        #nb,bins,patches = ax.hist(data,facecolor="green",alpha=0.75)
        sns.distplot(data,ax=ax,hist=True,kde=True,bins=10,color="darkblue",label="Posterior Distribution",hist_kws={"edgecolor":"black"},kde_kws={"linewidth":4})
        plt.ylabel("Density")
        plt.title("Histogram of {}".format(name.upper()),fontsize = 'x-large')
        pmin,pmax = plt.xlim()
        ymin,ymax = plt.ylim()
        
        # Get prior distribution
        num = 1000
        p = np.linspace(start=lb,stop=ub,num=num)
        x = list(); prior_pdf = list()
        
        # Get distribution hyperparameters
        if not distr.endswith("_hp"):
            pars[3],pars[4] = getHyperParameters(distr=distr,mean=pars[3],std=pars[4])
        
        xm = lb; ym = 0
        for j in range(num):
            y,b   = pdf(distr,p[j],pars)
            x.append(p[j])  
            prior_pdf.append(y)
            if y > ym:
                ym = y
                xm = p[j]
            if not b:
                cprint(f"plotHistogram: prior pdf calculation for '{distr}' distribution failed","red")
        
        if ymax < 10*ym:
            pmin = max(lb,min(pmin,pars[3]-2*pars[4]))
            pmax = min(ub,max(pmax,pars[3]+2*pars[4]))
       
        # Plot prior
        if len(prior_pdf)>0:
            ax.plot(x,prior_pdf,lw=3,color="y",label="Prior Distribution")
        plt.legend(fontsize = 'medium')
        plt.xlim([pmin,pmax])
        plt.grid(True)
        if k%(rows*columns) == 0:
            fig.set_tight_layout(True)
            if not header is None:
                make_space_above(fig)
                fig.suptitle(header,fontsize=17,fontweight='normal')
            if save:    
                fig.savefig(os.path.join(path_to_dir,"Density_" + str(kmax+ceil(k/4)) + '.' + ext))
            if show: 
                plt.show(block=False)
                plt.close(fig)
                fig = None
                
    if k%(rows*columns)>0:  
        fig.set_tight_layout(True)
        if not header is None:
            make_space_above(fig)
            fig.suptitle(header,fontsize=17,fontweight='normal')
        if save:
            plt.savefig(os.path.join(path_to_dir,'Density_' + str(kmax+ceil((1+k)/4)) + '.' + ext))
        if show: 
            plt.show(block=False)
            plt.close(fig)
            fig = None
                
            
    print("Two dimensional projections of multidimensional samples covariances")
    from corner import corner
    params = []           
    for i,k in enumerate(names):
        if k in priors and i < size:
            params.append(parameters[i])
            
    data = samples[:,:size]
    params_std = np.std(data, axis=0)
    if np.max(np.abs(params_std))>1.e-6:
        fig = corner(data,labels=names[:size],truths=params,smooth=True,color="blue")
        fig.suptitle('Projections of a parameter set in a multi-dimensional space\n', fontsize=18, y=1.05)
        fig.set_figheight(8)
        fig.set_figwidth(10)
        
        # Extract the axes
        axes = np.array(fig.axes)
        axes = axes.reshape((size, size))
                  
        # Empirical mean of the sample:
        params_mean = np.mean(samples, axis=0)
    
        # Loop over the diagonal
        for i in range(size):
            ax = axes[i, i]
            ax.axvline(params[i], color="r")
            ax.axvline(params_mean[i], color="b")
            for item in [ax.xaxis.label, ax.yaxis.label]:
                item.set_fontsize(15)
            
        # Loop over the histograms
        for yi in range(size):
            for xi in range(yi):
                ax = axes[yi, xi]
                ax.axvline(params[xi], color="r")
                ax.axvline(params_mean[xi], color="b")
                ax.axhline(params[yi], color="r")
                ax.axhline(params_mean[yi], color="b")
                ax.plot(params[xi], params[yi], "r")
                ax.plot(params_mean[xi], params_mean[yi], "b")
                for item in [ax.xaxis.label, ax.yaxis.label]:
                    item.set_fontsize(8)
                for item in (ax.get_xticklabels() + ax.get_yticklabels()):
                    item.set_fontsize(6)
        
        if save:    
            fig.set_tight_layout(True)
            fig.savefig(os.path.join(path_to_dir,"Corner." + ext))        
        if show: 
            plt.show(block=False)
            plt.close(fig)
            fig = None
        
    plt.close('all')
    
        
def plot(path_to_dir,data,variable_names,sizes=None,figsize=None,meas_values=None,meas_variables=None,lrx_filter=None,hp_filter=None,Tmax=1.e10,output_variables=None,var_labels={},prefix=None,steady_state=None,rng=None,rng_meas=None,irf=0,Npaths=1,header=None,show=True,save=True,ext="png"):
    """
    Plot 1-D graphs of macro variables.
    
    Parameters:
        :param path_to_dir: Path to the folder where figures are saved.
        :type path_to_dir: str.
        :param data: Data array.
        :type data: array.
        :param variable_names: Variables names.
        :type variable_names: list.
        :param sizes: Subplots dimensions.
        :type sizes: tuple.
        :param figsize: Figure size..
        :type figsize: tuple.
        :param meas_values: Measurement variables values.
        :type meas_values: list.
        :param meas_variables: Measurement variables names.
        :type meas_variables: list.
        :param lrx_filter: List of LRX filtered measurement values.
        :type lrx_filter: list.
        :param hp_filter: List of HP filtered measurement values.
        :type hp_filter: list.
        :param Tmax: Maximum number of periods to display in graphs.
        :type Tmax: int.
        :param output_variables: Output variable names.
        :type output_variables: list.
        :param var_labels: Labels of output variable names.
        :type var_labels: dict.
        :param prefix: Prefix of variables.  If set then display variable which name start with prefix.
        :type prefix: str.
        :param steady_state: Steady-state solution.
        :type steady_state: list.
        :param irf: Impulse-responce function variable.
        :type irf: int.
        :param Npaths: Number of paths.
        :type Npaths: int.
        :param header: Plot header.
        :type header: str.
        :param show: Boolean variable.  If set to True shows graphs.
        :type show: bool.
        :param save: Boolean variable.  If set to True saves graphs.
        :type save: bool.
        :param ext: Format of the saved file.
        :type ext: str.
        :returns: List of figures.
    """
    from textwrap import wrap
    from snowdrop.src.utils.util import deleteFiles
    from snowdrop.src.utils.util import getMap
    from snowdrop.src.utils.util import caseInsensitiveDict
    
    global K1
    K1 += 1
    
    if figsize is None:
        figsize=(16, 10)
    style.use(STYLE)
    
    if not sizes is None:
        rows, columns = sizes
    elif not output_variables is None:
        if len(output_variables) == 1:
            rows = columns = 1
        elif len(output_variables) == 2:
            rows, columns = 1,2
        elif len(output_variables) <= 4:   
            rows = columns = 2
        elif len(output_variables) <= 6:   
            rows = 3; columns = 2
        elif len(output_variables) <= 8:   
            rows = 4; columns = 2
        elif len(output_variables) <= 12:   
            rows = 4; columns = 3
        else:
            rows, columns = 4,3
    else:
        rows, columns = 3,3
    
    # years = dates.YearLocator()   # every year
    # yearsFmt = dates.DateFormatter('%Y')
    
    plot_vars = [x for x in variable_names if  "_minus_" not in x and "_plus_" not in x]
    
    if not prefix is None:
        plot_vars = [x for x in plot_vars if  x.startswith(prefix)]
                
        
    if K1==1:
        deleteFiles(path_to_dir,"Variables")
    kmax = K1 - 1
    
    file_path = os.path.abspath(os.path.join(working_dir,"supplements/data/dictionary/symbols_labels.csv"))
    if os.path.exists(file_path):
        symbolsMap = getMap(file_path)
    else:
        symbolsMap = {}
        
    k = 0; m = 0  
    figs = []
    
    if meas_variables is None:
        meas_vars = None
    else:
        meas_vars = [x.lower() for x in meas_variables]
    
    ciLabels = caseInsensitiveDict(var_labels)
    indices = sorted(range(len(variable_names)), key=lambda k: variable_names[k])
    
    for j in indices:
        n = variable_names[j]
        if not n in plot_vars:
            continue
        if not output_variables is None:
            if not n in output_variables:
                continue
        data_meas = None
        data_lrx_filtered = None
        data_hp_filtered = None
        if "_BAR" in n:
            ind = n.index("_BAR")
            n2 = n[:ind]
        else:
            n2 = n
        if not meas_vars is None:
            ind = None
            if n2.lower()+"_meas" in meas_vars:
                meas_var = n2.lower() + "_meas"
                ind = meas_vars.index(meas_var)
            elif "obs_" + n2.lower() in meas_vars:
                meas_var = "obs_" + n2.lower()
                ind = meas_vars.index(meas_var)
            if not ind is None:
                if not meas_values is None:
                    if isinstance(meas_values,pd.DataFrame):
                        data_meas = meas_values[meas_variables[ind]]
                    else:
                        data_meas = meas_values[:,ind]
                if not lrx_filter is None and len(lrx_filter) > ind:
                    data_lrx_filtered = lrx_filter[ind]
                if not hp_filter is None and len(hp_filter) > ind:
                    data_hp_filtered = hp_filter[ind]
        
        if k%(rows*columns) == 0:
            m = 0
            fig = plt.figure(figsize=figsize)
        k += 1
        m += 1
        ax = plt.subplot(rows,columns,m)
        ax.tick_params(axis='both', labelsize=12)
        for i in range(Npaths):
            if Npaths == 1:
                y = data[-1]
            else:
                y = data[i]
            dim1,dim2 = y.shape
            if rng is None:
                T = min(dim1,Tmax)
                series = pd.Series(data=y[:T,j])
                if Npaths == 1:
                    series.plot(ax=ax,lw=3,color='b',label=n)
                else:
                    series.plot(ax=ax,lw=1,label=n)
            else:
                T = min(dim1,len(rng),Tmax)
                series = pd.Series(data=y[:T,j], index=rng[:T])
                # if not rng_meas is None:
                #     mask = (series.index.to_timestamp() >= rng_meas[0]) & (series.index.to_timestamp() <= rng_meas[-1]) 
                #     series = series[mask]
                if Npaths == 1:
                    series.plot(ax=ax,lw=3,color='DarkBlue',label=n)
                else:
                    series.plot(ax=ax,lw=1,label=n)
            if not data_meas is None:
                if isinstance(data_meas,pd.Series):
                    data_meas.plot(ax=ax,lw=3,color='r',style='.',markersize=10,label="MEASUREMENT")
                elif rng_meas is None:
                    series_meas = pd.Series(data=data_meas)
                    series_meas.plot(ax=ax,lw=3,color='r',style='.',markersize=10,label="MEASUREMENT")
                else:
                    series_meas = pd.Series(data=data_meas, index=rng_meas[1:len(data_meas)])
                    series_meas.plot(ax=ax,lw=3,color='r',style='.',markersize=10,label="MEASUREMENT")
            if not data_lrx_filtered is None:
                if rng_meas is None:
                    series_meas = pd.Series(data=data_lrx_filtered)
                    series_meas.plot(ax=ax,lw=3,color='g',label="LRX FILTER")
                else:
                    nm = min(len(data_lrx_filtered),len(rng_meas))
                    series_meas = pd.Series(data=data_lrx_filtered[:nm], index=rng_meas[:nm])
                    series_meas.plot(ax=ax,lw=3,color='g',label="LRX FILTER")
            if not data_hp_filtered is None:
                if rng_meas is None:
                    series_meas = pd.Series(data=data_hp_filtered[1:])
                    series_meas.plot(ax=ax,lw=3,color='y',label="HP FILTER")
                else:
                    nm = min(len(data_hp_filtered),len(rng_meas))
                    series_meas = pd.Series(data=data_hp_filtered[:nm], index=rng_meas[:nm])
                    series_meas.plot(ax=ax,lw=3,color='y',label="HP FILTER")
        if Npaths == 1:
            plt.legend(fontsize = 'medium')
        xmin, xmax = plt.xlim() 
        ymin, ymax = plt.ylim() 
        if not steady_state is None:
            ssj = (1-irf)*steady_state[j]
            ax.plot([xmin,xmax],[ssj,ssj],'k',lw=3)
            
            
        if n in symbolsMap:
            plt.title("\n".join(wrap(symbolsMap[n])),fontsize=12)
        elif n.upper() in ciLabels:
            plt.title("\n".join(wrap(ciLabels[n.upper()])),fontsize=12)
        else:
            plt.title(n,fontsize=12)
        plt.grid(True)
        # format the ticks
        # ax.xaxis.set_minor_locator(years)
        # ax.xaxis.set_major_formatter(yearsFmt)
        # ax.xaxis.set_minor_formatter(yearsFmt)

        if k%(rows*columns) == 0:
            plt.tight_layout()
            if not header is None:
                make_space_above(fig)
                fig.suptitle(header,fontsize=17,fontweight='normal')
            figs.append(fig)
            if save:
                plt.savefig(os.path.join(path_to_dir,'Variables_' + str(kmax+ceil(k/(rows*columns))) + '.' + ext))
            if show: 
                plt.show(block=False) 
                #plt.close(fig)
            
    if k%(rows*columns) > 0:    
        plt.tight_layout(pad=2)
        if not header is None:
            make_space_above(fig)
            fig.suptitle(header,fontsize=17,fontweight='normal')
        figs.append(fig) 
        if save:
            plt.savefig(os.path.join(path_to_dir,'Variables_' + str(kmax+ceil((1+k)/(rows*columns))) + '.' + ext))
        if show: 
            plt.show(block=False) 
            #plt.close(fig)
        
    return figs   
 
   
def plotSteadyState(path_to_dir,variables,arr_ss,par_ss,sizes=None,fig_sizes=(12,10),save=False,show=False,ext="png"):
    """
    Plot steady state solution graphs.
    
    Parameters:
        :param path_to_dir: Path to the folder where figures are saved.
        :type path_to_dir: str.
        :param variables: List of variables.
        :type variables: list.
        :param arr_ss: Array of steady states.
        :type arr_ss: array.
        :param par_ss: List of parameter names.
        :type par_ss: list.
        :param sizes: Subplots dimensions.
        :type sizes: tuple.
        :param ext: Format of the saved file.
        :type ext: str.
    """
    style.use(STYLE)
    nVariables = len(variables)
    
    if not sizes is None:
        rows, columns = sizes
    else:
        ns = nVariables * len(par_ss)
        if ns == 1:
            rows,columns = 1,1
        elif ns == 2:
            rows,columns = 2,1
        elif ns <= 3:
            rows, columns = 3,1
        elif ns <= 4:
            rows, columns = 2,2
        elif ns <= 6:   
            rows,columns = 3,2
        elif ns <= 8:   
            rows,columns = 4,2
        elif ns <= 9:   
            rows,columns = 3,3
        elif ns <= 12:   
            rows = 4; columns = 3
        else:
            columns = 3
            rows = int(np.ceil(ns/columns))
            
    figs = []
    chunk = rows*columns
    m = 0

    for k in range(len(arr_ss)):
        arr = arr_ss[k]
        nss = len(arr)
        x = np.zeros((nss,1))
        y = np.zeros((nss,nVariables))
        for i in range(nss):
            ar = arr[i]
            x[i] = ar[0]
            y[i,:] = ar[1:]
        jj = 0
        for j in range(nVariables):
            if m%chunk == 0 or j == 0:
                jj = 0
                fig = plt.figure(figsize=fig_sizes)
                figs.append(fig)
            m += 1
            jj += 1
            ax = plt.subplot(rows,columns,jj)
            ax.plot(x,y[:,j],label=variables[j])
            plt.title('Steady-State Solution for "' + variables[j] + '"',fontsize = 'medium')
            plt.xlabel(par_ss[k])
            plt.grid(True)
            plt.tight_layout(pad = 5) 
            
    if save:
        plt.savefig(os.path.join(working_dir,'graphs/Steady_State.'+ext))
    if show: 
        plt.show(block=False) 
    
    return figs
        
 
def plotTimeSeries(path_to_dir,header,titles,labels,series,sizes=None,fig_sizes=(12,10),save=False,highlight=None,stacked=True,isLastLinePlot=True,zero_line=False,show=True,ext=None):
    """    
    Plot time series.
    
    Parameters:
        :param path_to_dir: Path to the folder where figures are saved.
        :type path_to_dir: str.
        :param header: List of variables.
        :type header: str.
        :param titles: List of graphs titles.
        :type titles: list.
        :param labels: List of graphs labels.
        :type labels: list.
        :param sizes: Subplots dimensions.
        :type sizes: tuple.
        :param save: If True saves plot in a pdf format.
        :type save: bool.
        :param highlight: The startong and ending dates of a highlighted region.
        :type highlight: list.
        :param stacked: If True plot stacked bar plots.
        :type stacked: bool.
        :param isLastLinePlot: If True the last series plot will be drawn as line plots.
        :type isLastLinePlot: bool.
        :param show: If True display plots.
        :type zero_line: bool.
        :param zero_line: If True adds zero line.
        :type show: bool.
        :param ext: Format of the saved file.
        :type ext: str.
    """
    #import matplotlib.gridspec as gridspec
    #from datetime import datetime 
    
    style.use(STYLE)
    colors = getColors()
    
    if not sizes is None:
        rows, columns = sizes
    else:
        if len(titles) == 1:
            rows,columns = 1,1
        elif len(titles) == 2:
            rows,columns = 2,1
        elif len(titles) <= 3:
            rows, columns = 3,1
        elif len(titles) <= 4:
            rows, columns = 2,2
        elif len(titles) <= 6:   
            rows,columns = 3,2
        elif len(titles) <= 8:   
            rows,columns = 4,2
        elif len(titles) <= 9:   
            rows,columns = 3,3
        elif len(titles) <= 12:   
            rows = 4; columns = 3
        else:
            columns = 3
            rows = int(np.ceil(len(titles)/columns))
        
    n_titles = len(titles)
    chunk = rows*columns
    n_chunks = int(np.ceil(n_titles/chunk))
    barWidth = 70
    alpha = 1.0
    e = None
    for k in range(n_chunks):
        fig = plt.figure(figsize=fig_sizes)
        _titles = titles[k*chunk:(k+1)*chunk]
        _series = series[k*chunk:(k+1)*chunk]
        for i,t in enumerate(_titles):
            tmin,tmax = None,None
            b = False
            if i < len(_series):
                entries = _series[i]
                if isinstance(entries,pd.Series):
                    entries = entries.dropna()
                    if tmin is None: 
                        tmin = min(entries.index)
                    else:
                        tmin = min(tmin,min(entries.index))
                    if tmax is None: 
                        tmax = max(entries.index)
                    else:
                        tmax = max(tmax,max(entries.index))
                    if i == 0:
                        plt.title(t,loc='center',wrap=True)
                        line, = plt.plot(entries,linewidth=3,color='k')
                    else:
                        line, = plt.plot(entries,linewidth=2,marker='',markersize=3,zorder=1)
                        line.set_dashes([2,2,4,2])
                    if zero_line:
                        plt.axhline(y=0,color='b',linestyle='-')
                        
                else:
                    if labels is None:
                        lbls = []
                    else:
                        lbls = labels[i] if i < len(labels) else []
                    n = len(entries)
                    c = -1
                    bottomPlus,bottomMinus = 0,0 
                    # width_ = 0.77/columns
                    # height_ = 0.6/rows
                    # left_ = (1-width_*columns)/2 + 0.9*(jj-1)/columns
                    # bottom_ = min(0.93-height_,0.93-0.9*ii/rows+0.05)
                    # if jj==columns:
                    #     jj = 0
                    #     ii += 1
                    #print(ii,jj,left_,bottom_)
                    ax = plt.subplot(rows,columns,1+i)
                    #pos = ax.get_position()
                    #print(pos)
                    #ax.set_position(np.array([left_,bottom_,width_,height_]))
                    #print([left_,bottom_,width_,height_])
                    ax.set_title(t,loc='center',wrap=True)
                    data = {}
                    for j in range(n):
                        if entries[j] is None:
                            continue
                        e = entries[j]
                        #e = entries[j].fillna(0)
                        if isinstance(e,pd.Series):
                            if tmin is None: 
                                tmin = min(e.index)
                            else:
                                tmin = min(tmin,min(e.index))
                            if tmax is None: 
                                tmax = max(e.index)
                            else:
                                tmax = max(tmax,max(e.index))
                            if j<len(lbls):
                                lb = lbls[j]
                                b = True
                            else:
                                lb = None
                            if n <= 2 or not stacked:
                                if lb is None:
                                    line, = ax.plot(e,linewidth=2,marker='',markersize=2,zorder=1)
                                else:
                                    line, = ax.plot(e,linewidth=2,marker='',markersize=2,zorder=1,label=lb)
                                if j > 1:
                                    line.set_dashes([2,2,4,2])
                                if zero_line:
                                    plt.axhline(y=0,color='k',linestyle='-')
                
                            else:
                                data[lb] = e
                                ind = e.index
                                zPlus = 0.5*(e+np.abs(e))
                                zMinus = 0.5*(e-np.abs(e)) 
                                c += 1
                                c = c % len(colors)
                                # Stacked bar plots
                                try:
                                    if j == 0:
                                        if not lb is None:
                                            plt.bar(x=ind,height=zPlus,label=lb,width=barWidth,align='center',alpha=alpha,color=colors[c],edgecolor="black")
                                        else:
                                            plt.bar(x=ind,height=zPlus,width=barWidth,align='center',alpha=alpha,color=colors[c],edgecolor="black")
                                        plt.bar(x=ind,height=zMinus,width=barWidth,align='center',alpha=alpha,color=colors[c],edgecolor="black")
                                        bottomPlus  = zPlus
                                        bottomMinus = zMinus
                                    elif j==n-1 and isLastLinePlot:
                                        if not lb is None:
                                            line, = ax.plot(e,linewidth=2,color='k',label=lb)
                                        else:
                                            line, = ax.plot(e,linewidth=2,color='k')
                                    else:
                                        if not lb is None:
                                            plt.bar(ind,zPlus,label=lb,bottom=bottomPlus[zPlus.index],width=barWidth,align='center',alpha=alpha,color=colors[c],edgecolor="black")
                                        else:    
                                            plt.bar(ind,zPlus,bottom=bottomPlus[zPlus.index],width=barWidth,align='center',alpha=alpha,color=colors[c],edgecolor="black")                                                                   
                                        plt.bar(ind,zMinus,bottom=bottomMinus[zMinus.index],width=barWidth,align='center',alpha=alpha,color=colors[c],edgecolor="black")
                                        bottomPlus += zPlus
                                        bottomMinus += zMinus
                                    if zero_line:
                                        plt.axhline(y=0,color='k',linestyle='-')              
                                except Exception as exc:
                                    cprint(f"plotTimeSeries: Series - {lb}, Error - {exc}","red")
                                    
                                        
                    ax.xaxis.set_tick_params(labelsize='medium')
                    if len(ax.xaxis.get_ticklabels()) >= 10:
                        [l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_ticklabels()) if i % 2 != 0]
                    # if not e is None and (isinstance(e.index,pd.DatetimeIndex) or isinstance(e.index[0],datetime)):
                    #     ax.xaxis.set_minor_locator(matplotlib.dates.MonthLocator([1,4,7,10]))
                    #     ax.xaxis.set_major_locator(matplotlib.dates.YearLocator())
                    #     ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y'))
            
            if b: 
                if len(data)>0:
                    plt.legend(loc="best",mode="expand",ncol=len(lbls),fontsize='medium')
                else:
                    plt.legend(loc="best",fontsize='medium')
            plt.grid(True)
            plt.xticks(rotation=45)
            plt.tight_layout()
                
            if not highlight is None:
                if isinstance(highlight[0],str):
                    from datetime import datetime as dt
                    x1 = dt.strptime(highlight[0],"%Y-%m-%d")
                    x2 = dt.strptime(highlight[1],"%Y-%m-%d")
                else:
                    x1 = highlight[0]
                    x2 = highlight[1]
                
                ax.axvspan(x1,x2,color='gray',alpha=0.5) 
    
        if len(titles) > 12:
            fig.autofmt_xdate(bottom=0.1, rotation=45, ha='right', which='major')
            
        make_space_above(fig)
        fig.suptitle(header,fontsize=17,fontweight='normal')
        if show:
            plt.show(block=False)
        
        _header = header if k==0 else header+"_"+str(k)
        if save:
            if ext is None:
                fig.savefig(os.path.join(path_to_dir,_header+".pdf"),dpi=600)
                fig.savefig(os.path.join(path_to_dir,_header+".png"),dpi=600)
            else:
                fig.savefig(os.path.join(path_to_dir,_header+"."+ext),dpi=600)
                
    return fig

 
def plotSeries(path_to_dir,header,titles,labels,series,sizes=None,xlabel=[],ylabel=[],plotType=None,fig_sizes=(12,10),save=False,show=True,ext=None):
    """    
    Plot time series.
    
    Parameters:
        :param path_to_dir: Path to the folder where figures are saved.
        :type path_to_dir: str.
        :param header: List of variables.
        :type header: str.
        :param titles: List of graphs titles.
        :type titles: list.
        :param labels: List of graphs labels.
        :type labels: list.
        :param sizes: Subplots dimensions.
        :type sizes: tuple.
        :param plotType: Type of plot, i.e., line plot, bar plot, etc.
        :type plotType: str.
        :param xlabel: X labels of plot.
        :type xlabel: list.
        :param ylabel: Y labels of plot.
        :type ylabel: list.
        :param save: If True saves plot in a pdf format.
        :type save: bool.
        :param isLastLinePlot: If True the last series plot will be drawn as line plots.
        :type isLastLinePlot: bool.
        :param show: If True display plots.
        :type show: bool.
        :param ext: Format of the saved file.
        :type ext: str.
    """
    #import matplotlib.gridspec as gridspec
    style.use(STYLE)
    colors = getColors()
    
    fig = plt.figure(figsize=fig_sizes)
    
    if not sizes is None:
        rows, columns = sizes
        if plotType is None:
            plotType = ["line"]*100
    else:
        if len(titles) == 1:
            rows = columns = 1
        elif len(titles) == 2:
            rows, columns = 2,1
        elif len(titles) <= 4:   
            rows = columns = 2
        elif len(titles) <= 6:   
            rows = 3; columns = 2
        elif len(titles) <= 8:   
            rows = 4; columns = 2
        elif len(titles) <= 9:   
            rows = columns = 3
        elif len(titles) <= 12:   
            rows = 4; columns = 3
        else:
            rows, columns = 4,3
        if plotType is None:
            plotType = ["line"]*len(titles)
 
    barWidth = 150
    alpha = 1.0
    for i,t in enumerate(titles):
        entries = series[i]
        n = len(entries)
        tmin,tmax = None,None
        if labels is None:
            lbls = []
        else:
            lbls = labels[i] if i < len(labels) else []
        xlb = xlabel[i] if i < len(xlabel) else None
        ylb = ylabel[i] if i < len(ylabel) else None
        ax = plt.subplot(rows,columns,1+i)
        b = False
        if plotType[i].lower() == "line":
            for j in range(n):
                if entries[j] is None:
                    continue
                e = entries[j].fillna(0)
                if j<len(lbls):
                    lb = lbls[j]
                    b = True
                else:
                    lb = None
                if j == 0:
                    plt.title(t)
                    line, = ax.plot(e,linewidth=3,color='k',label=lb)
                else:
                    line, = ax.plot(e,linewidth=2,label=lb,marker='',markersize=3,zorder=1)
                    line.set_dashes([2,2,4,2])
                
        elif plotType[i].lower() == "bar":
            c = -1
            bottomPlus,bottomMinus = 0,0 
            ax.set_title(t)
            data = {}
            for j in range(n):
                e = entries[j]
                if e is None:
                    continue
                e = e.fillna(0)
                if isinstance(e,pd.Series):
                    if tmin is None: 
                        tmin = min(e.index)
                    else:
                        tmin = min(tmin,min(e.index))
                    if tmax is None: 
                        tmax = max(e.index)
                    else:
                        tmax = max(tmax,max(e.index))
                    if j<len(lbls):
                        lb = lbls[j]
                        b = True
                    else:
                        lb = None
                    data[lb] = e
                    z = e.values
                    ind = e.index
                    zPlus = 0.5*(z+np.abs(z))
                    zMinus = 0.5*(z-np.abs(z)) 
                    c += 1
                    c = c % len(colors)
                    try:
                        # Stacked bar plots
                        if j == 0:
                            if not lb is None:
                                plt.bar(x=ind,height=zPlus,label=lb,width=barWidth,align='center',alpha=alpha,color=colors[c],edgecolor="black")
                            else:
                                plt.bar(x=ind,height=zPlus,width=barWidth,align='center',alpha=alpha,color=colors[c],edgecolor="black")
                            plt.bar(x=ind,height=zMinus,width=barWidth,align='center',alpha=alpha,color=colors[c],edgecolor="black")
                        else:
                            if not lb is None:
                                plt.bar(ind,zPlus,label=lb,bottom=bottomPlus,width=barWidth,align='center',alpha=alpha,color=colors[c],edgecolor="black")
                            else:    
                                plt.bar(ind,zPlus,bottom=bottomPlus,width=barWidth,align='center',alpha=alpha,color=colors[c],edgecolor="black")                                                                   
                            plt.bar(ind,zMinus,bottom=bottomMinus,width=barWidth,align='center',alpha=alpha,color=colors[c],edgecolor="black")
                        bottomPlus += zPlus
                        bottomMinus += zMinus
                    except Exception as exc:
                        cprint(f"plotSeries: Series - {lb}, Error - {exc}","red")
            
        elif plotType[i].lower() == "fill":
            c = -1 
            ax.set_title(t)
            zPrevious = 0
            for j in range(n):
                e = entries[j]
                if e is None:
                    continue
                e = e.fillna(0)
                if isinstance(e,pd.Series):
                    if j<len(lbls):
                        lb = lbls[j]
                        b = True
                    else:
                        lb = None
                    z = zPrevious + e.values
                    if not np.all(z==0):
                        ind = e.index 
                        c += 1
                        c = c % len(colors)
                        # Fill area
                        plt.fill_between(x=ind,y1=zPrevious,y2=z,label=lb,alpha=alpha,color=colors[c],edgecolor="black")
                        zPrevious = z            
                                
            if not xlb is None:
                ax.set_xlabel(xlb)
            if not ylb is None:
                ax.set_ylabel(ylb)
            ax.xaxis.set_tick_params(labelsize='medium')
        
        if b: 
            plt.legend(loc="best",fontsize='medium')
        plt.grid(True)
        plt.tight_layout()
            

    if len(titles) > 4:
        fig.autofmt_xdate()
        
    make_space_above(fig)
    fig.suptitle(header,fontsize=17,fontweight='normal')
    if show:
        plt.show(block=False)
    
    if save:
        if ext is None:
            fig.savefig(os.path.join(path_to_dir,header+".pdf"),dpi=600)
            fig.savefig(os.path.join(path_to_dir,header+".png"),dpi=600)
        else:
            fig.savefig(os.path.join(path_to_dir,header+"."+ext),dpi=600)
            
    return fig


def barPlot(path_to_dir,titles,data,labels=None,xLabel=None,yLabel=None,sizes=None,plot_variables=False,fig_sizes=(8,6),save=False,show=True,ext=None):
    """    
    Plot bar graph.
    
    Parameters:
        :param path_to_dir: Path to the folder where figures are saved.
        :type path_to_dir: str.
        :param titles: Title.
        :type titles: str or list.
        :param data: Data.
        :type data: numpy array.
        :param labels: List of graphs labels.
        :type labels: list.
        :param xLabel: X axis label.
        :type xLabel: str.
        :param yLabel: Y axis label.
        :type yLabel: str.
        :param sizes: Subplots dimensions.
        :type sizes: tuple.
        :param plot_variables: If True plot selected variables.
        :type plot_variables: bool.
        :param fig_sizes: Figure sizes.
        :type fig_sizes: Tuple.
        :param save: If True saves plot in a pdf format.
        :type save: bool.
        :param show: If True display plots.
        :type show: bool.
        :param ext: Format of the saved file.
        :type ext: str.
    """
    #import matplotlib.gridspec as gridspec
    style.use(STYLE)
    
    fig = plt.figure(figsize=fig_sizes)
    barWidth = 0.3
    alpha = 0.8
    if labels is None:
        labels = np.arange(1,1+len(data))
        
    if len(titles) == 1:
        rows = columns = 1
    elif len(titles) == 2:
        rows, columns = 2,1
    elif len(titles) <= 4:   
        rows = columns = 2
    elif len(titles) <= 6:   
        rows = 3; columns = 2
    elif len(titles) <= 8:   
        rows = 4; columns = 2
    elif len(titles) >= 9:   
        rows = columns = 3
    elif not sizes is None:
        rows, columns = sizes
    else:
        rows = columns = 3
    
    if plot_variables:
        for i,t in enumerate(titles):
            ax = plt.subplot(rows,columns,1+i)
            ax.set_title(t)
            zPlus = 0.5*(data[i]+np.abs(data[i]))
            zMinus = 0.5*(data[i]-np.abs(data[i])) 
            ax.bar(x=labels[i],height=zPlus,width=barWidth,align='center',alpha=alpha,edgecolor="black")
            ax.bar(x=labels[i],height=zMinus,width=barWidth,align='center',alpha=alpha,edgecolor="black")
            plt.grid(True)
            plt.tight_layout()
    
    else:
        if isinstance(data[0],(list,tuple,np.ndarray)):
            df = pd.DataFrame(data[:,0],columns=['Baseline'],index=labels)
            for i in range(1,len(data[0])):
                df["Scenario " + str(i)] = data[:,i]
            fig = df.plot(kind='bar').get_figure()
            plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f'))
            plt.gca().xaxis.set_tick_params(rotation=0)
        else:
            zPlus = 0.5*(data+np.abs(data))
            zMinus = 0.5*(data-np.abs(data)) 
            plt.bar(x=labels,height=zPlus,width=barWidth,align='center',alpha=alpha,edgecolor="black")
            plt.bar(x=labels,height=zMinus,width=barWidth,align='center',alpha=alpha,edgecolor="black")
            
        plt.title(titles)
        plt.grid(True)
        if not xLabel is None:
            plt.xlabel(xLabel)
        if not yLabel is None:
            plt.ylabel(yLabel)
        plt.tight_layout()
            
    if show:
        plt.show(block=False)
    if save:
        if isinstance(titles,list):
            header = "_".join(titles)
        else:
            header = titles
        if ext is None:
            fig.savefig(os.path.join(path_to_dir,header+".pdf"),dpi=600)
            fig.savefig(os.path.join(path_to_dir,header+".png"),dpi=600)
        else:
            fig.savefig(os.path.join(path_to_dir,header+"."+ext),dpi=600)
            
    return  
  
               
def plot_chain(path_to_dir,chains,names=None,title=None,sizes=[3,2],figsize=None,save=False,ext="pdf"):
    """
    Plot sampling chain for each parameter.

    Args:
        * **param path_to_dir** Path to the folder where figures 
          are saved
        * **chains** (:class:`~numpy.ndarray`): Sampling chain for each
          parameter
        * **names** (:py:class:`list`): List of strings - name of each
          parameter
        * **figsize** (:py:class:`list`): Specify figure size in inches
          [Width, Height]
        * **save* If True saves plot in a pdf format
        * **ext* Format of the saved file
    """
    rows, columns = sizes
    if names is None:
        rows, columns = sizes
    else:
        if len(names) == 1:
            rows = columns = 1
        elif len(names) == 2:
            rows, columns = 1,2
        elif len(names) <= 4:   
            rows = columns = 2
        elif len(names) <= 6:   
            rows = 2; columns = 3
        else:
            columns = 2
            rows = int(np.ceil(len(names)/columns))
    
    if figsize is None:
        figsize = [7, 5]
        
    nsimu, nparam = chains.shape  # number of rows, number of columns
    inds = np.arange(nsimu)
    fig = plt.figure(dpi=100, figsize=(figsize))  # initialize figure
    for ii in range(nparam):
        # define chain
        chain = chains[:, ii]  # check indexing
        # plot chain on subplot
        plt.subplot(rows, columns, ii+1)
        plt.plot(inds, chain, 'b')
        # format figure
        plt.xlabel('Iteration')
        plt.ylabel(str('{}'.format(names[ii])))
        if ii+1 <= rows*columns - columns:
            plt.xlabel('')
            
    if not title is None:
        make_space_above(fig)
        fig.suptitle(title,fontsize=17,fontweight='normal')
      
    # adjust spacing
    plt.tight_layout() 
    
    plt.show(block=False)
    if save:
        if ext is None:
            fig.savefig(os.path.join(path_to_dir,"Convergence.pdf"),dpi=600)
            fig.savefig(os.path.join(path_to_dir,"Convergence.png"),dpi=600)
        else:
            fig.savefig(os.path.join(path_to_dir,"Convergence."+ext),dpi=600)
    

def plot_pairwise_correlation(path_to_dir,chains,names=None,title=None,figsize=None,save=False,ext="pdf"):
    """
    Plot pairwise correlation for each parameter.

    Args:
        * **param path_to_dir** Path to the folder where figures 
          are saved
        * **chains** (:class:`~numpy.ndarray`): Sampling chain for each
          parameter
        * **names** (:py:class:`list`): List of strings - name of each
          parameter
        * **figsize** (:py:class:`list`): Specify figure size in inches
          [Width, Height]
        * **save* If True saves plot in a pdf format
        * **ext* Format of the saved file
    """
    nsimu, nparam = chains.shape  # number of rows, number of columns
    inds = np.arange(nsimu)
    if figsize is None:
        figsize = [12, 10]
    fig = plt.figure(dpi=100, figsize=(figsize))  # initialize figure
    for jj in range(2, nparam + 1):
        for ii in range(1, jj):
            chain1 = chains[inds, ii - 1]
            chain2 = chains[inds, jj - 1]
            # plot density on subplot
            ax = plt.subplot(nparam - 1, nparam - 1, (jj - 2)*(nparam - 1)+ii)
            plt.plot(chain1, chain2, '.b')
            # format figure
            if jj != nparam:  # rm xticks
                ax.set_xticklabels([])
            if ii != 1:  # rm yticks
                ax.set_yticklabels([])
            if ii == 1:  # add ylabels
                plt.ylabel(str('{}'.format(names[jj - 1])))
            if ii == jj - 1:
                if nparam == 2:  # add xlabels
                    plt.xlabel(str('{}'.format(names[ii - 1])))
                else:  # add title
                    plt.title(str('{}'.format(names[ii - 1])))
                    
    if not title is None:
        make_space_above(fig)
        fig.suptitle(title,fontsize=17,fontweight='normal')
        
    # adjust figure margins
    plt.tight_layout()
    
    plt.show(block=False)
    if save:
        if ext is None:
            fig.savefig(os.path.join(path_to_dir,"Correlation.pdf"),dpi=600)
            fig.savefig(os.path.join(path_to_dir,"Correlation.png"),dpi=600)
        else:
            fig.savefig(os.path.join(path_to_dir,"Correlation."+ext),dpi=600)
    

def plotTestedSeries(x1,dx1,x2,dx2,n1,n2):
    """
    Check endogenous variables cross-correlation and Granger cointegration.

    Parameters:
        x1 : numpy array
            Values of the first endogenous variables.
        dx1 : numpy array
            Values of the increments of the first endogenous variables.
        x2 : numpy array
            Values of the second endogenous variables.
        dx12 : numpy array
            Values of the increments of the second endogenous variables.

    """
    fig = plt.figure(figsize=(8,6))
    plt.subplot(2,2,2)
    plt.scatter(dx1,dx2,linewidths=2)
    plt.plot([np.min(dx1),np.max(dx1)],[np.min(dx2),np.max(dx2)],'r')
    plt.xlabel('Difference')
    plt.ylabel('Difference')
    plt.subplot(2,2,1)
    plt.scatter(x1,x2,linewidths=2)
    plt.plot([np.min(x1),np.max(x1)],[np.min(x2),np.max(x2)],'r')
    plt.xlabel('Level')
    plt.ylabel('Level')
    plt.subplot(2,2,3)
    plt.plot(x1)
    plt.title(n1)
    plt.ylabel('Level')
    plt.xlabel('Period')
    plt.subplot(2,2,4)
    plt.plot(x2)
    plt.title(n2)
    plt.ylabel('Level')
    plt.xlabel('Period')
    plt.suptitle(n1 + "  -  " + n2,fontsize = 'x-large')
    fig.set_tight_layout(True)


def plotSurface(path_to_dir,data,Time,variable_names,output_variables=None,prefix=None,Npaths=1,show=True,save=True,ext="png"):
    """
    Plot 2-D graphs of macro variables.
    
    Parameters:
        :param path_to_dir: Path to the folder where figures are saved.
        :type path_to_dir: str.
        :param data: Data array.
        :type data: array.
        :param Time: List of dates.
        :type Time: list.
        :param variable_names: Variables names.
        :type variable_names: list.
        :param output_variables: Output variable names.
        :type output_variables: list.
        :param prefix: Prefix of variables.  If set displays variable which name start with this prefix.
        :type prefix: str.
        :param Npaths: Number of paths.
        :type Npaths: int.
        :param show: Boolean variable.  If set to True shows graphs.
        :type show: bool.
        :param save: Boolean variable.  If set to True saves graphs.
        :type save: bool.
        :param ext: Format of the saved file.
        :type ext: str.
        :returns: List of figures.
    """
    from snowdrop.src.utils.util import deleteFiles
    from snowdrop.src.utils.util import getMap
    from textwrap import wrap
    
    dim1,dim2,dim3 = data[0].shape
    if dim1 == 1:
        return
    
    style.use(STYLE)
    
    years = dates.YearLocator()   # every year
    yearsFmt = dates.DateFormatter('%Y')
    T = len(Time)
        
    plot_vars = [x for x in variable_names if  "_minus_" not in x and "_plus_" not in x]
 
     # Surface Plot
    for j in range(1,Npaths):
        yIter = np.array(data[j])
        dim1t,dim2t,dim3t = yIter.shape
        dim1 = min(dim1,dim1t)
        dim2 = min(dim2,dim2t)
        dim3 = min(dim3,dim3t)
    yIter = np.zeros((dim1,dim2,dim3))    
    for j in range(Npaths):
        temp = np.array(data[j])/Npaths
        yIter = yIter + temp[0:dim1,0:dim2,0:dim3]
    
    Iterations = range(1,1+dim1)
    Xg,Yg = np.meshgrid(Time,Iterations)  
    
    k = 0; m = 0
    figs = []
    
    # Delete files
    deleteFiles(path_to_dir, "Convergence")
    
    file_path = os.path.abspath(os.path.join(working_dir,"supplements/data/dictionary/symbols_labels.csv"))
    if os.path.exists(file_path):
        symbolsMap = getMap(file_path)
    else:
        symbolsMap = {}
    
    # Sort plotting variables names in alphabetic order
    indices = sorted(range(len(variable_names)), key=lambda k: variable_names[k])
       
    for j in indices:
        n = variable_names[j]
        if not n in plot_vars:
            continue
        if not output_variables is None:
            if not n in output_variables:
                continue
        if k%4 == 0:
            m = 0
            fig = plt.figure(figsize=(10, 8))
        k += 1
        m += 1
        Variables_iter = np.zeros((dim1,T))
        y = yIter[0:dim1,:,j]
        for i in range(dim1):
            Variables_iter[i,:] = y[i,0:T]
    
        min_variables = np.min(Variables_iter)
        ax = fig.add_subplot(2,2,m, projection='3d')
        surf = ax.plot_surface(Xg,Yg,Variables_iter,cmap=cm.coolwarm,linewidth=0,antialiased=True)
        ax.contour(Xg,Yg,Variables_iter, zdir='z', offset=min_variables,cmap=cm.coolwarm)
        ax.contour(Xg,Yg,Variables_iter, zdir='x', offset=-1,cmap=cm.coolwarm)
        ax.contour(Xg,Yg,Variables_iter, zdir='y', offset=-1,cmap=cm.coolwarm)
        # format the ticks
        ax.xaxis.set_minor_locator(years)
        ax.xaxis.set_minor_formatter(yearsFmt)
        if n in symbolsMap:
            ax.set_title("\n".join(wrap(symbolsMap[n])))
        else:
            ax.set_title(n)
        ax.set_xlabel('Time')
        ax.set_ylabel('Iteration')
        fig.colorbar(surf, shrink=0.5, aspect=5)
        
        
        if k%4 == 0:
            plt.tight_layout()
            figs.append(fig)
            if save:
                plt.savefig(os.path.join(path_to_dir,'Convergence_' + str(ceil(k/4)) + '.' + ext))
            if show: 
                plt.show(block=False) 
                plt.close(fig)
            
    if k%4 > 0:    
        plt.tight_layout()
        figs.append(fig) 
        if save:
            plt.savefig(os.path.join(path_to_dir,'Convergence_' + str(ceil((1+k)/4)) + '.' + ext))
        if show: 
            plt.show(block=False) 
            plt.close(fig)
    
    return figs
     

def bar_plot(path_to_dir=None,var=[],var_names=[],par=None,par_names=None,title="",symbols=None,xLabel=None,yLabel=None,plot_variables=False,relative=False,sizes=None,fig_sizes=(8,6)):
    """Plot bar graphs."""
    
    from snowdrop.src.graphs.util import barPlot
    if path_to_dir is None:
        path_to_dir = os.path.abspath(os.path.join(working_dir,'graphs'))
    
    if plot_variables:
        title = []
        labels = []; data = []
        for v in var_names:
            arr = []; lbls = []
            for k in var:
                if k.startswith(v+"_"):
                    s = k[1+len(v):]
                    lbls.append(s.upper().replace("_","\n"))
                    arr.append(var[k])
            if len(lbls)>0:
                labels.append(lbls)
                t = symbols[v] if v in symbols else v
                title.append(t)
            if len(arr)>0:
                data.append(arr)
        data = np.array(data)
        
        if len(labels) > 0:
            barPlot(path_to_dir,title,data,labels,xLabel,yLabel,sizes=sizes,plot_variables=plot_variables,fig_sizes=fig_sizes,save=True,show=True,ext='png')
     
    else:
        vom_parameters = [x for x in par_names if x.startswith("vom_")]
        Y   = [x for x in var_names if x.startswith("Y_")]
        if len(vom_parameters) == len(Y):
            welfare = {}
            for p in vom_parameters:
                s = p[4:]
                if "Y_"+s in Y:
                    welfare[s.upper().replace("_","\n")] = par[p] * var["Y_"+s]
                    
            if bool(welfare):
                labels = list(welfare.keys())
                if relative:
                    data = np.array([100*(welfare[k][1]/welfare[k][0]-1) for k in labels])
                else:
                    data = np.array([welfare[k] for k in labels])
                barPlot(path_to_dir,title,data,labels,xLabel,yLabel,sizes=sizes,fig_sizes=fig_sizes,save=True,show=True,ext='png')
     
        
def plot3D(path_to_dir,data,variable_names,show=True,save=True,ext="png"):
    """
    Plot 3-D graphs of macro variables.
    
    This is a toy example of 3D plots.
    
    Parameters:
        :param path_to_dir: Path to the folder where figures are saved.
        :type path_to_dir: str.
        :param data: Data array.
        :type data: array.
        :param variable_names: Variables names.
        :type variable_names: list.
        :param show: Boolean variable.  If set to True shows graphs.
        :type show: bool.
        :param save: Boolean variable.  If set to True saves graphs.
        :type save: bool.
        :param ext: Format of the saved file.
        :type ext: str.
        :returns: List of figures.
    """
    colors = getColors()
    nd = data.ndim
    if nd == 3:
        ndim1,ndim2,ndim3 = data.shape
        if ndim3 >= 4:
            X = data[0,:,0]
            #Y = data[0,:,1]
            Z = data[0,:,3]
            Radius = Z
            n = len(X)
            Time = range(n)
            
            fig = plt.figure(figsize=(10, 8))
            ax = plt.axes(projection='3d')
            k = -1
            for i in range(1,n-1,ceil(n/5)):
                k += 1 
                c = k % len(colors)
                color = colors[c]
                r = Radius[i]
                # Draw a sphere
                u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
                x = Time[i] + r*np.cos(u)*np.sin(v)
                y = r*np.sin(u)*np.sin(v)
                z = r*np.cos(v)
                ax.plot_wireframe(x, y, z, color=color)
            
            
            ax.set_title(variable_names[3])
            ax.set_xlabel('Time')
            ax.set_ylabel(variable_names[0])
            ax.set_zlabel(variable_names[1])
            if save:
                plt.savefig(os.path.join(path_to_dir,'3D.'+ext))
            if show: 
                plt.show(block=False)
                plt.close(fig)
            
            
def plotImage(path_to_dir,fname):
    """
    Plot image file.
    
    Parameters:
        :param path_to_dir: Path to the folder where figures are saved.
        :type path_to_dir: str.
        :param fname: Image file name.
        :type fname: str.
    """
    from PIL import Image  

    fpath = os.path.abspath(os.path.join(path_to_dir,fname))
    img = Image.open(fpath)  
    img.show()
    
    
def make_space_above(fig,topmargin=1):
    """Increase figure size to make top margin (in inches) space for titles, without changing the axes sizes."""
    
    s = fig.subplotpars
    w, h = fig.get_size_inches()

    figh = h - (1-s.top)*h  + topmargin
    fig.subplots_adjust(bottom=s.bottom*h/figh, top=1-topmargin/figh)
    fig.set_figheight(figh)
    
    
def rgb(col):
    """Set up RGB Colours."""
    if col   == 'Turque':   R = 0;   G = 165; B = 185
    elif col == 'Blue':     R = 55;  G = 70;  B = 150
    elif col == 'DarkBlue': R = 64;  G = 86;  B = 106
    elif col == 'Pink':     R = 230; G = 30;  B = 105
    elif col == 'Black':    R = 0;   G = 0;   B = 0   
    elif col == 'Red':      R = 255; G = 0;   B = 0  
    elif col == 'Orange':   R = 245; G = 150; B = 20
    elif col == 'Yellow':   R = 255; G = 220; B = 50 
    elif col == 'Purple':   R = 130; G = 55;  B = 140
    elif col == 'Green':    R = 0;   G = 255; B = 219
    elif col == 'Brown':    R = 146; G = 146; B = 146
    elif col == 'Grey':     R = 128; G = 128; B = 128
    elif col == 'White':    R = 255; G = 255; B = 255
    
    return (R/255.,G/255.,B/255.)


def getColors(variables=[]):
    """Define RGB colors."""
    DarkBlue  = rgb('DarkBlue')
    Blue      = rgb('Blue')
    Green     = rgb('Green')
    Brown     = rgb('Brown')
    Yellow    = rgb('Yellow'),
    Turque    = rgb('Orange')
    Pink      = rgb('Pink')
    Orange    = rgb('Orange')
    Red       = rgb('Red')
    Grey      = rgb('Grey')
    Turque    = rgb('Turque')
    Purple    = rgb('Purple')
    White     = rgb('White')
    Black     = rgb('Black')
    
    colors = [Blue,Yellow,Orange,Green,Brown,Pink,Turque,Purple,Red,DarkBlue,Grey,White,Black]
    n  = len(colors)
    mp = dict()
    
    if len(variables) > 0:    
        for i,v in enumerate(variables):
            ii = i%n
            mp[v] = colors[ii]
        return colors,mp
    else:
        return colors