# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 13:49:49 2018

@author: A.Goumilevski
"""

import sys, os

path = os.path.dirname(os.path.abspath(__file__))
working_dir = os.path.abspath(os.path.join(path,".."))
if os.path.exists(working_dir):
    sys.path.append(working_dir)

import re
import datetime as dt 
import tkinter as tk
from tkinter import filedialog as fd
from tkinter import messagebox
from tkinter import ttk
#from tkinter import W,N,E,S,BOTH
import numpy as np
from snowdrop.src.utils.util import getNamesAndValues
#from snowdrop.src.utils.interface import *
   
minWidth = 1400
minHeight = 650
maxWidth = 1600
maxHeight = 800

strDescription = "Description:"
strEqs = "Equations:"
strSession = "Session"
strInput = "Input"
strParams = "Parameters:"
strParamsRange = "Parameters Range:"
strEndogVarInitValues = "Endogenous Variables Starting Values:"
strExogVariables = "Exogenous Variables Values:"
strShocks = "Shocks:"
strTimeRange = "Time Range:"
strFreq = "Frequency:"
strFreqList = ['Annually','Quarterly','Monthly','Weekly','Daily']
strSteadyState = "Find Steady-State Solution for Parameters Range:"
strSteadyStateSolution = "Steady-State Solution:"
strSimulationResults = "Simulation Results:"

showSessionTab = False

class Application(tk.Frame):
     
    def __init__(self, master=None,file_path=None):
        self.containers = {}
        self.master = master
        tk.Frame.__init__(self, master)
        if file_path is None:
            txtDescr="";txtEqs="";txtParams="";txtParamsRange="";txtEndogVars="";txtExogVars="";txtShocks="";txtRange="";txtFreq="";eqLabels=""
        else:
            txtDescr,txtEqs,txtParams,txtParamsRange,txtEndogVars,txtExogVars,txtShocks,txtRange,txtFreq,eqLabels,comments = readFile(file_path)
        self.description = txtDescr
        self.createWidgets(txtDescr,txtEqs,txtParams,txtParamsRange,txtEndogVars,txtExogVars,txtShocks,txtRange,txtFreq)
        self.tabContainer = None
        self.tableContainer = None
        self.eqLabels = eqLabels
        self.input_file = None
        self.history_file = None
        self.comments = None
        self.pack()

    def OpenFile(self):
        """ Open file dialog."""
        #fdir = os.path.abspath(os.path.join(working_dir,"../models/Sirius"))
        fdir = os.path.abspath(os.path.join(working_dir,"../models/Troll/FSGM3"))
        file_path = fd.askopenfilename(initialdir=fdir)
        self.input_file = os.path.abspath(file_path)
        
        for key in self.containers.keys():
            tab= self.containers[key]
            tab.destroy()
            
        txtDescr,txtEqs,txtParams,txtParamsRange,txtEndogVars,txtExogVars,txtShocks,txtRange,txtFreq,eqLabels,comments = readFile(file_path)
        self.description = txtDescr
        self.createWidgets(txtDescr,txtEqs,txtParams,txtParamsRange,txtEndogVars,txtExogVars,txtShocks,txtRange,txtFreq)
        self.comments = comments
        
        if "tab2" in self.containers:
            tab = self.containers["tab2"]   
            self.notebook.select(tab)
        
    def OpenHistoryFile(self):
        """Open history file dialog."""
        file_path = fd.askopenfilename()
        self.history_file = file_path
            
    def RestoreSession(self):
        """Read session file."""
        path = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.abspath(os.path.join(path,'../../data/session.txt'))
        obj = self.containers[strInput]
        obj.delete('1.0', tk.END)
        with open(file_path, 'r') as f:
            for line in f:
                ln = line.strip() + "\n"
                obj.insert(tk.END,ln)
        
    def Save(self):
        """Save GUI data into template.txt files and template.yaml files."""
        if checkNumberOfEquationsAndVariables(self):
            SaveYamlOutput(self)
            #SaveTemplateOutput(self)
        
    def SaveSession(self):
        """Save user input to a text file."""
        path = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.abspath(os.path.join(path, '../../data/session.txt'))
        obj = self.containers[strInput]
        session = obj.get("1.0",tk.END)
        with open(file_path, 'w') as f:
            for line in session.split("\n"):
                f.write(line + "\n")
           
    def FindSteadyState(self):
        """
        Finds the steady state solution
        """
        from snowdrop.src.driver import importModel
        from snowdrop.src.driver import findSteadyStateSolution
        from snowdrop.src.driver import findSteadyStateSolutions

        # Save GUI data to template file
        self.Save()
        # Run simulations
        fdir = os.path.dirname(self.input_file)
        name = os.path.basename(self.input_file)
        fout,ext = os.path.splitext(name)
        fname = os.path.abspath(os.path.join(fdir,fout+'.yaml'))
        # Create model
        model = importModel(fname)
        # Get variables names
        variables = model.symbols['variables']
        order = np.argsort(variables)
        
        param_range = getParamRange(self)
        if param_range.strip():
            findSteadyStateSolutions(model=model,Plot=False,Output=False)
        else:
            data = findSteadyStateSolution(model=model,Output=False)    
            data = np.around(data, decimals=3)   
            arr1 = []; arr2 = []
            for i in order:
                v = variables[i]
                if "_plus_" in v or "_minus_" in v:
                    continue
                arr1.append(v)
                arr2.append(data[0][i] if data[0][i] != 0 else data[1][i])
            
            data = np.column_stack((arr1,arr2))
            
            tab = self.containers["tab3"]
            if not self.tabContainer is None:
                self.tabContainer.destroy()
            
            # Pupulate widgets of this tab  
            container = tk.Frame(tab)
            
            self.cl = tk.Button(container, fg = "black", bg = "white")
            self.cl["text"] = "CLOSE"
            self.cl["command"] = self.Close
            self.cl.pack(side=tk.RIGHT,padx=5,pady=5)
            
            self.cl = tk.Button(container, fg = "black", bg = "white")
            self.cl["text"] = "CLEAN"
            self.cl["command"] = self.CleanResults
            self.cl.pack(side=tk.RIGHT,padx=5,pady=5)
            
            if not self.tableContainer is None:
                self.tableContainer.destroy()
                
            self.tableContainer = createTableWidget(tab,container,strSteadyStateSolution,columns=['Name','Value'],values=data,minheight=40,width=200,minwidth=50,anchor="w",stretch=True,side=tk.TOP,adjust_heading_to_content=True)
            container.pack(side=tk.BOTTOM,fill=tk.BOTH,expand=True) 
            self.tabContainer = container

            self.notebook.select(tab)
                             
            
    def GetImpulseResponseFunctions(self):
        """Find and plot impulse response functions."""
        from snowdrop.src.driver import plot,getImpulseResponseFunctions
        
        # Save GUI data to template file
        self.Save()
        
        # Remove figure tabs
        self.removeFigures()
        
        fdir = os.path.dirname(self.input_file)
        name = os.path.basename(self.input_file)
        fout,ext = os.path.splitext(name)
        fname = os.path.abspath(os.path.join(fdir,fout+'.yaml'))
        
        path = os.path.dirname(os.path.abspath(__file__))
        path_to_dir = os.path.abspath(os.path.join(path,'../../graphs'))
        
        # Run simulations
        time,data,columns,rng = getImpulseResponseFunctions(fname=fname,Plot=False,Output=False)
        
        # Get figures
        figs = plot(path_to_dir=path_to_dir,data=data,variable_names=columns,sizes=(2,2),figsize=(6,4),rng=rng,show=False,save=False)
         
        # Output data 
        data = np.around(data[0], decimals=3)
        if len(time) > 0:
            columns = ['Date'] + columns
            dates = []
            for d in time:
                dates.append(dt.datetime.strftime(d,'%m/%d/%Y'))
            n = min(len(dates),len(data))
            data = np.column_stack( (dates[:n],data[:n]) )
        
        arr1 = []; arr2 = []
        for i,v in enumerate(columns):
            if "_plus_" in v or "_minus_" in v:
                continue
            arr1.append(v)
            arr2.append(data[:,i])
            
        columns = arr1                         
        data = np.array(arr2).T
                
        tab = self.containers["tab3"]
        if not self.tabContainer is None:
            self.tabContainer.destroy()
        
        # Pupulate widgets of this tab  
        container = tk.Frame(tab)
        
        self.cl = tk.Button(container, fg = "black", bg = "white")
        self.cl["text"] = "CLOSE",
        self.cl["command"] = self.Close
        self.cl.pack(side=tk.RIGHT,padx=5,pady=5)
        
        self.cl = tk.Button(container, fg = "black", bg = "white")
        self.cl["text"] = "CLEAN",
        self.cl["command"] = self.CleanResults
        self.cl.pack(side=tk.RIGHT,padx=5,pady=5)
        
        if not self.tableContainer is None:
            self.tableContainer.destroy()
        
        self.tableContainer = createTableWidget(tab,container,strSimulationResults,columns=columns,values=data,minheight=40,side=tk.TOP,adjust_heading_to_content=True)
        container.pack(side=tk.BOTTOM,fill=tk.BOTH,expand=True) 
        self.tabContainer = container
        
        container = tk.Frame(tab)
        self.notebook.select(tab)
        
        #import matplotlib
        #matplotlib.use('TkAgg')
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        
        # Place figures in new tabs
        for i,fig in enumerate(figs):
            tabControl = self.notebook
            tab = ttk.Frame(tabControl)
            tabName = 'Figure #' + str(1+i)
            self.containers[tabName] = tab
            tabControl.add(tab,text=tabName)
            canvas = FigureCanvasTkAgg(fig, master=tab)
            canvas.get_tk_widget().pack(expand=tk.TRUE)
            canvas.draw()
            
        tab = self.containers["tab2"]   
        self.notebook.select(tab)
           
    def Clean(self):  
        """
        Cleans session and command text boxes
        """
        obj = self.containers[strInput]
        obj.delete('1.0', tk.END)
        
        obj = self.containers[strSession]
        obj.config(state=tk.NORMAL)
        obj.delete('1.0', tk.END)
        obj.config(state=tk.DISABLED)
        
        self.removeFigures()
        
               
    def CleanResults(self):
        """Clean content of tab #3."""                
        if not self.tabContainer is None:
            self.tabContainer.destroy()
            
        if not self.tableContainer is None:
            self.tableContainer.destroy()
            
        self.removeFigures()
         
        tab = self.containers["tab2"]   
        self.notebook.select(tab)
                
        
    def ProcessCommands(self):
        """Run commands that user enetered in the Input text box."""
        from io import StringIO
        
        # create file-like string to capture output
        old_stdout = sys.stdout
        sys.stdout = StringIO() 
        my_stdout = sys.stdout
        
        ansi_escape = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')
        
        obj = self.containers[strInput]
        inp = obj.get("1.0",tk.END)
        buf = []
        for line in inp.split("\n"):
            cmd = line.replace("\n","").strip()
            if cmd:
                try:
                    buf.append(">> " + cmd)
                    exec(cmd)
                    value = my_stdout.getvalue()
                    # Remove non-ansi characters
                    value = ansi_escape.sub('', value)
                    buf.append(value)
                    my_stdout.truncate(0)
                except NameError as err:
                    buf.append("Name error: {0}".format(err) + "\n")
                except SyntaxError as err:
                    buf.append("Syntax error: {0}".format(err) + "\n")
                except OSError as err:
                    buf.append("OS error: {0}".format(err) + "\n")
                except ValueError as err:
                    buf.append("Value error: {0}".format(err) + "\n")
                except:
                    buf.append("Error: {0}".format(sys.exc_info()[0]) + "\n")        
            
        sys.stdout = old_stdout  
           
        # Populate session text box        
        obj = self.containers[strSession]
        obj.config(state=tk.NORMAL)
        obj.delete('1.0', tk.END)  
        obj.insert(tk.END,"\n".join(buf))
        obj.config(state=tk.DISABLED)
                       
        
    def Run(self):
        """Run simulations."""
        from snowdrop.src.driver import run,importModel
        from snowdrop.src.graphs.util import plot,plotDecomposition
        
        # Save GUI data to template file
        self.Save()
        
        # Remove figure tabs
        self.removeFigures()
        
        fdir = os.path.dirname(self.input_file)
        name = os.path.basename(self.input_file)
        fout,ext = os.path.splitext(name)
        fname = os.path.abspath(os.path.join(fdir,fout+'.yaml'))
        
        path = os.path.dirname(os.path.abspath(__file__))
        path_to_dir = os.path.abspath(os.path.join(path,'../../graphs'))
        
        # Run simulations
        model = importModel(fname=fname)
        rng,data = run(model=model,Plot=False,Output=False)
        columns = model.symbols["variables"]
    
        # Get figures
        figs1 = plot(path_to_dir=path_to_dir,data=data,variable_names=columns,sizes=(2,2),figsize=(12,8),rng=rng,show=False,save=False)        
        decomp = ['dot4_cpi','dot4_cpi_x','dot4_gdp','lgdp_gap','lx_gdp_gap','mci','rmc','rr','rr_gap']
        #decomp = columns
        figs2 = plotDecomposition(path_to_dir=path_to_dir,model=model,y=data[-1],s=columns,decomp_variables=decomp,periods=rng,rng=rng,sizes=(2,2),figsize=(12,8),show=False,save=False)        
        figs = figs1 + figs2   
        
        indices = sorted(range(len(columns)), key=lambda k: columns[k])
        variable_names = [columns[i] for i in indices if not "_plus_" in columns[i] and not "_minus_" in columns[i]]
        indices = [i for i,x in enumerate(columns) if x in variable_names]
                         
        # Output data
        data = np.around(data[-1], decimals=3)
        data = data[:,indices]
        if len(rng) > 0:
            columns = ['Date'] + variable_names
            dates = []
            for d in rng:
                dates.append(dt.datetime.strftime(d,'%m/%d/%Y'))
            n = min(len(dates),len(data))
            data = np.column_stack( (dates[:n],data[:n]) )
            
        arr1 = []; arr2 = []
        for i,v in enumerate(columns):
            if "_plus_" in v or "_minus_" in v:
                continue
            arr1.append(v)
            arr2.append(data[:,i])
                
        columns = arr1                         
        data = np.array(arr2).T
        
        tab = self.containers["tab3"]
        if not self.tabContainer is None:
            self.tabContainer.destroy()
        
        # Pupulate widgets of this tab  
        container = tk.Frame(tab)
        
        self.cl = tk.Button(container, fg = "black", bg = "white")
        self.cl["text"] = "CLOSE",
        self.cl["command"] = self.Close
        self.cl.pack(side=tk.RIGHT,padx=5,pady=5)
        
        self.cl = tk.Button(container, fg = "black", bg = "white")
        self.cl["text"] = "CLEAN",
        self.cl["command"] = self.CleanResults
        self.cl.pack(side=tk.RIGHT,padx=5,pady=5)
        
        if not self.tableContainer is None:
            self.tableContainer.destroy()
        
        self.tableContainer = createTableWidget(tab,container,strSimulationResults,columns=columns,values=data,minheight=40,side=tk.TOP,adjust_heading_to_content=True)
        container.pack(side=tk.BOTTOM,fill=tk.BOTH,expand=True) 
        self.tabContainer = container
        
        container = tk.Frame(tab)
        self.notebook.select(tab)
        
        # Place figures in new tabs
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        
        for i,fig in enumerate(figs):
            tabControl = self.notebook
            tab = ttk.Frame(tabControl)
            tabName = 'Figure #' + str(1+i)
            self.containers[tabName] = tab
            tabControl.add(tab,text=tabName)
            canvas = FigureCanvasTkAgg(fig, master=tab)
            canvas.get_tk_widget().pack(expand=tk.TRUE)
            canvas.draw()

    def Close(self):
       """Close the GUI."""
       global root
       root.destroy()
       root.quit()
       root = None
       
       
    def CloseFrame(self):
       """Close the frame."""
       self.root.destroy()
       self.root = None
      
        
    def Reset(self):
       """Reset the GUI."""
       self.destroy()
       app = Application(root)
       app.master.title("Equations Editor")
       app.master.minsize(minWidth, minHeight)
       app.master.maxsize(maxWidth, maxHeight)
               
       tab = self.containers["tab2"]   
       self.notebook.select(tab)
                  
       
    def getEquationLabels(self):
        """Return equation labels and definition for Troll models."""
        eqs = ""
        if not self.eqLabels is None:
            for lb, eq in self.eqLabels.iteritems():
                eqs = eqs + lb + " : " + eq + "\n"
            
        return eqs
    
    def removeFigures(self):
        """
        Removes figures tab
        """
        keys = []
        for key in self.containers.keys():
            tab = self.containers[key]
            if key.startswith("Figure"):
                keys.append(key)
                
        for key in keys:
            tab = self.containers[key]
            tab.destroy()
            self.containers.pop(key,None)
               
    def getEquation(self,label):
        """
        Returns equation definition by label for Troll models
        """
        eq = ""
        if not self.eqLabels is None and label in self.eqLabels.keys():
            eq = self.eqLabels[label]
            
        return eq           
     
    def createWidgets(self,txtDescr,txtEqs,txtParams,txtParamsRange,txtEndogVars,txtExogVars,txtShocks,txtRange,txtFreq):
        """
        Creates GUI widgets
        """
        self.description = txtDescr       
        # Pupulate widgets of tab1
        tabControl = ttk.Notebook(self)
        self.notebook = tabControl
        if showSessionTab:
            tab1 = ttk.Frame(tabControl)
            tabControl.add(tab1, text='Session')
            self.containers["tab1"] = tab1
        tab2 = ttk.Frame(tabControl)
        tabControl.add(tab2, text='Model')
        self.containers["tab2"] = tab2
        tab3 = ttk.Frame(tabControl)
        tabControl.add(tab3, text='Results')
        self.containers["tab3"] = tab3
        tabControl.pack(expand=True, fill="both")
        
        if showSessionTab:
            container = tk.Frame(tab1)        
            self.run = tk.Button(container, fg = "blue", bg = "white")
            self.run["text"] = "EXECUTE COMMANDS"
            self.run["command"] = self.ProcessCommands
            self.run.pack(side=tk.RIGHT,padx=5,pady=5)
                    
            self.ss = tk.Button(container, fg = "brown", bg = "white")
            self.ss["text"] = "LOAD SESSION COMMANDS"
            self.ss["command"] = self.RestoreSession
            self.ss.pack(side=tk.RIGHT,padx=5,pady=5)
            container.pack(side=tk.BOTTOM,fill=tk.BOTH,expand=True)
        
            #imgFilePath = os.path.abspath(os.path.join(path + '../../img/Save_32x32.png'))
            #img = tk.PhotoImage(file=imgFilePath)
            self.sv = tk.Button(container, fg = "brown", bg = "white")
            self.sv["text"] = "SAVE SESSION"
            self.sv["command"] = self.SaveSession
            self.sv.pack(side=tk.RIGHT,padx=5,pady=5)
            
            self.cl = tk.Button(container, fg = "green", bg = "white")
            self.cl["text"] = "CLEAN"
            self.cl["command"] = self.Clean
            self.cl.pack(side=tk.RIGHT,padx=5,pady=5)
            
            self.cl = tk.Button(container, fg = "green", bg = "white")
            self.cl["text"] = "CLOSE"
            self.cl["command"] = self.Close
            self.cl.pack(side=tk.RIGHT,padx=5,pady=5)
                  
            container = tk.Frame(tab1)
            createTextBoxWidget(self,container,label=strSession,text="",width=180,height=20,scrollBar=True,disabled=True,side=tk.TOP,background="lightyellow",foreground="black") 
            container.pack(side=tk.TOP,fill=tk.BOTH,expand=True)
            
            container = tk.Frame(tab1)
            createTextBoxWidget(self,container,label=strInput,text="",width=180,height=20,scrollBar=True,side=tk.TOP,selectbackground="black",inactiveselectbackground="black") 
            container.pack(side=tk.TOP,fill=tk.BOTH,expand=True)

        # Populate widgets of tab2  
        container = tk.Frame(tab2)
        self.run2 = tk.Button(container, fg = "blue", bg = "white")
        self.run2["text"] = "RUN SIMULATIONS"
        self.run2["command"] = self.Run
        self.run2.pack(side=tk.RIGHT,padx=5,pady=5)
        
        self.ss2 = tk.Button(container, fg = "blue", bg = "white")
        self.ss2["text"] = "IMPULSE RESPONSE FUNCTION"
        self.ss2["command"] = self.GetImpulseResponseFunctions
        self.ss2.pack(side=tk.RIGHT,padx=5,pady=5)
        
        self.ss2 = tk.Button(container, fg = "blue", bg = "white")
        self.ss2["text"] = "FIND STEADY STATE"
        self.ss2["command"] = self.FindSteadyState
        self.ss2.pack(side=tk.RIGHT,padx=5,pady=5)
        
        #self.ss = tk.Button(container, fg = "brown", bg = "white")
        #self.ss["text"] = "OPEN HISTORY FILE"
        #self.ss["command"] = self.OpenHistoryFile
        #self.ss.pack(side=tk.RIGHT,padx=5,pady=5)
        
        self.ss = tk.Button(container, fg = "brown", bg = "white")
        self.ss["text"] = "OPEN MODEL FILE"
        self.ss["command"] = self.OpenFile
        self.ss.pack(side=tk.RIGHT,padx=5,pady=5)
        
        #imgFilePath = os.path.abspath(os.path.join(path, '../../img/Save_32x32.png'))
        #img = tk.PhotoImage(file=imgFilePath)
        self.sv = tk.Button(container, fg = "green", bg = "white")
        self.sv["text"] = "SAVE TEMPLATE"
        self.sv["command"] = self.Save
        self.sv.pack(side=tk.RIGHT,padx=5,pady=5)
        
        self.rs = tk.Button(container, fg = "green", bg = "white")
        self.rs["text"] = "RESET"
        self.rs["command"] = self.Reset
        self.rs.pack(side=tk.RIGHT,padx=5,pady=5)
        
        self.cl2 = tk.Button(container, fg = "green", bg = "white")
        self.cl2["text"] = "CLOSE"
        self.cl2["command"] = self.Close
        self.cl2.pack(side=tk.RIGHT,padx=5,pady=5)
        container.pack(side=tk.BOTTOM,fill=tk.BOTH,expand=True)

        container = tk.Frame(tab2)
        createTextBoxWidget(self,container,label=strEqs,text=txtEqs,width=180,height=20,scrollBar=True,scroll="both",side=tk.TOP) 
        container.pack(side=tk.TOP,fill=tk.BOTH,expand=True)
        
        container = tk.Frame(tab2)
        createTextBoxWidget(self,container,label=strTimeRange,text=txtRange,width=40,height=3,side=tk.RIGHT)
        createListBoxWidget(self,container,label=strFreq,items=strFreqList,selected_item=txtFreq,width=30,height=5,side=tk.RIGHT)
        createTextBoxWidget(self,container,label=strSteadyState,text=txtParamsRange,width=40,height=3,side=tk.RIGHT)
        var = tk.IntVar()
        var.set(1)
        container.pack(side=tk.BOTTOM,fill=tk.BOTH,expand=True)
        
        container = tk.Frame(tab2)
        createTextBoxWidget(self,container,label=strShocks,text=txtShocks,width=40,height=10,scrollBar=True,side=tk.RIGHT)    
        createTextBoxWidget(self,container,label=strEndogVarInitValues,text=txtEndogVars,width=40,height=10,scrollBar=True,side=tk.RIGHT) 
        createTextBoxWidget(self,container,label=strExogVariables,text=txtExogVars,width=40,height=10,scrollBar=True,side=tk.RIGHT) 
        createTextBoxWidget(self,container,label=strParams,text=txtParams,width=40,height=10,scrollBar=True,side=tk.BOTTOM)  
        container.pack(side=tk.BOTTOM,fill=tk.BOTH,expand=True) 
### End of class
                        
def createTableWidget(self,parent,label,columns,values,minheight,side=tk.BOTTOM,adjust_heading_to_content=False,width=None,minwidth=None,anchor=None,stretch=None):
    """
    Creates table widget
    """
    
    #from snowdrop.src.gui.table import Table
    from snowdrop.src.gui.multiColumnListBox import Multicolumn_Listbox

    container = tk.Frame(self)
    createLabel(container,label,side=tk.TOP)
        
    table = Multicolumn_Listbox(container,columns,height=minheight,stripped_rows = ("white","#f2f2f2"),cell_anchor="center",adjust_heading_to_content=adjust_heading_to_content)
    for index in range(values.shape[1]):
        table.configure_column(index,width=width,minwidth=minwidth,anchor=anchor,stretch=stretch)
    table.interior.pack()
    nrow = len(values)
    for i in range(nrow):
        table.insert_row(list(values[i,:]))
                 
    #table = Table(container,columns,minheight=minheight,height=500)
    #table.pack(expand=True,side=tk.TOP,padx=1,pady=1)
    #table.set_data(values)
    
    container.pack(side=side,fill=tk.BOTH,expand=True) 
    
    return container
    
def createLabel(container,label,side=tk.TOP):
    """
    Creates label widget
    """
    container.eqsLabel = tk.Label(container)
    container.eqsLabel["text"] = label
    container.eqsLabel.pack(side=side)
    
def createTextBoxWidget(self,parent,label,text,width,height,scrollBar=False,scroll="y",disabled=False,side=tk.TOP,background="white",foreground="black",selectbackground="black",inactiveselectbackground="white"):
    """
    Creates text box widget
    """
    container = tk.Frame(parent)
    createLabel(container,label,side=tk.TOP)
    
    if scrollBar: 
        container.sbTb = tk.Scrollbar(container,orient="vertical")
        container.sbTb.pack({"side": "right","fill":"y"})
        if scroll == "both":
            container.sbTb2 = tk.Scrollbar(container,orient="horizontal")
            container.sbTb2.pack({"side": "bottom","fill":"x"})
            
    container.tb = tk.Text(container,width=width,height=height,background=background,foreground=foreground,selectbackground=selectbackground,inactiveselectbackground=inactiveselectbackground)
    for i in range(len(text)):
        container.tb.insert(tk.END,text[i].strip('\t'))
    container.tb.pack(side=tk.TOP,fill=tk.BOTH,expand=tk.TRUE,padx=10,pady=10)
    
    if scrollBar: 
        container.tb.configure(yscrollcommand=container.sbTb.set)
        container.sbTb.config(command=container.tb.yview)
        if scroll == "both":
            container.tb.configure(xscrollcommand=container.sbTb2.set)
            container.sbTb2.config(command=container.tb.xview)
    
    if disabled: 
        container.tb.config(state=tk.DISABLED)
    container.pack(side=side,fill=tk.BOTH,expand=tk.TRUE)
    
    self.containers[label] = container.tb
    return container.tb
        
def createListBoxWidget(self,parent,label,selected_item,items,width,height,scrollBar=False,side=tk.TOP):
    """
    Creates list box widget
    """
    container = tk.Frame(parent)
    createLabel(container,label,side=tk.TOP)
    
    if scrollBar: 
        container.sbLb = tk.Scrollbar(container,orient="vertical")
        container.sbLb.pack({"side": "right","fill":"y"})
    container.lb = tk.Listbox(container,width=width,height=height)
    for i in items:
        container.lb.insert(tk.END, i)
    container.lb.pack(side=tk.TOP,padx=10,pady=10)
    item = re.sub('\s+', '', selected_item)
    if item in items:
        ind = items.index(item)
    else: 
        ind = 0
    container.lb.select_set(ind)
    
    if scrollBar: 
        container.lb.configure(yscrollcommand=container.sbLb.set)
        container.sbLb.config(command=container.lb.yview)
    container.pack(side=side,fill=tk.BOTH,expand=True)
    
    self.containers[label] = container.lb
    return container.lb
   
def createCheckBoxWidget(self,parent,label,text,flag,width,height,side=tk.TOP):
    """
    Creates check box widget
    """
    container = tk.Frame(parent)
    createLabel(container,label,side=tk.TOP)
    
    container.cb = tk.Checkbutton(container,text=text,variable=flag,width=width,height=height)
    container.cb.pack(side=tk.TOP,padx=10,pady=10)
    container.pack(side=side,fill=tk.BOTH,expand=True)
     
    self.containers[label] = container.cb
    return container.cb
    
def createRadioButtonWidget(self,parent,label,text,width,height,side=tk.TOP):
    """
    Creates radio button widget
    """
    container = tk.Frame(parent)
    createLabel(container,label,side=tk.TOP)
    
    container.rb = tk.Radiobutton(container,text=text,value=text,width=width,height=height)
    container.rb.pack(side=tk.TOP,padx=10,pady=10)
    container.pack(side=side,fill=tk.BOTH,expand=True)
    
    self.containers[label] = container.rb
    return container.rb
                
def readFile(file_path):
    """
    Reads data file
    """
    from snowdrop.src.utils.getDynareData import readDynareModelFile   
    from snowdrop.src.utils.getIrisData import readIrisModelFile
    from snowdrop.src.utils.getTemplateData import readTemplateFile
    from snowdrop.src.utils.getTrollData import readTrollModelFile
    from snowdrop.src.utils.getXmlData import readXmlModelFile
    from snowdrop.src.utils.getYamlData import readYamlModelFile
    
    eqLabels = None; comments = None
    fname, ext = os.path.splitext(file_path)
    name,_ = os.path.splitext(os.path.basename(file_path))
    if ext.lower() in [".inp",".src"]:
        txtEqs,txtParams,txtParamsRange,txtEndogVars,txtExogVars,txtShocks,txtRange,txtFreq,varLabels,eqLabels,modelName,comments,undefined_parameters = readTrollModelFile(file_path)
        if modelName:
            txtDescr = modelName
        else:
            txtDescr = 'Troll Model ' + name
        if len(undefined_parameters):
            messagebox.showwarning("Warning",f"{len(undefined_parameters)} parameters were not defined and were set to one.")           
    elif ext.lower() == ".mod":
        txtEqs,txtParams,txtParamsRange,txtEndogVars,txtExogVars,txtShocks,txtRange,txtFreq,modelName = readDynareModelFile(file_path)
        if modelName:
            txtDescr = modelName
        else:
            txtDescr = 'Dynare Model ' + name
    elif ext.lower() == ".model":   
        txtEqs,txtParams,txtParamsRange,txtEndogVars,txtExogVars,txtShocks,txtRange,txtFreq,modelName = readIrisModelFile(file_path)
        if modelName:
            txtDescr = modelName
        else:
            txtDescr = 'Iris Model ' + name
    elif ext.lower() == ".txt": 
        txtEqs,txtParams,txtParamsRange,txtEndogVars,txtExogVars,txtShocks,txtRange,txtFreq,modelName = readTemplateFile(file_path)   
        if modelName:
            txtDescr = modelName
        else:
            txtDescr = 'Text File ' + name
    elif ext.lower() == ".yaml": 
        txtEqs,txtParams,txtParamsRange,txtEndogVars,txtExogVars,txtShocks,txtRange,txtFreq,modelName = readYamlModelFile(file_path)   
        if modelName:
            txtDescr = modelName
        else:
            txtDescr = 'YAML Model ' + name
    elif ext.lower() == ".xml": 
        txtEqs,txtParams,txtParamsRange,txtEndogVars,txtExogVars,txtShocks,txtRange,txtFreq,modelName = readXmlModelFile(file_path)   
        if modelName:
            txtDescr = modelName
        else:
            txtDescr = 'XML Model ' + name
    else:
        txtEqs="";txtParams="";txtParamsRange="";txtEndogVars="";txtExogVars="";txtShocks="";txtRange="";txtFreq=""
        txtDescr = 'UnknownL Model File Type'
        messagebox.showwarning("Unknown Model File Extension","Only the following file extensions are supported: inp/mod/model/yaml/txt!  You are trying to open file with extension: {}".format(ext))
        
    return txtDescr,txtEqs,txtParams,txtParamsRange,txtEndogVars,txtExogVars,txtShocks,txtRange,txtFreq,eqLabels,comments 


def checkNumberOfEquationsAndVariables(self):
    """
    Performs check of equality of the number of equations 
    and of the number of variables
    """
    input_file = self.input_file
    b = not input_file is None and input_file.endswith(".inp")
    var = getVariables(self)
    var = var.split("\n")
    if b:
        var = [x for x in var if x.strip() and not "_ss" in x]
    n_var = len(var)
    eqs = getEquations(self)
    eqs = eqs.split("\n")
    if b:
        eqs = [x for x in eqs if x.strip() and not "_ss" in x]
    n_eqs = len(eqs)
    if n_var < n_eqs:
        msg = "endogenous variables"
    elif n_var > n_eqs:
        msg = "equations"
    if n_var != n_eqs: 
        messagebox.showerror("Warning","Number of equations: {} and number of endogenous variables: {} is different. Please add {}.".format(n_eqs,n_var,msg))
        return True
    else:
        return True
    
def getDescription(self):
    """Return Description."""
    description = self.description
    return description

def getShocks(self):
    obj = self.containers[strShocks]
    shocks = obj.get("1.0",tk.END)
    return shocks
    
def getShockNamesAndValues(self):
    shocks = getShocks(self)
    shock_names = []; shock_values = []
    names, values = getNamesAndValues(shocks)
    for n,v in zip(names, values):
        if not "Date" in n:
            shock_names.append(n)
            shock_values.append(v)
        
    return shock_names, shock_values

def getVariableNamesAndInitialValues(self):  
    var = getVariables(self)
    return getNamesAndValues(var)

def getVariables(self): 
    obj = self.containers[strEndogVarInitValues]
    init_values = obj.get("1.0",tk.END)
    return init_values

def getExogVariables(self):
    obj = self.containers[strExogVariables]
    exog_var = obj.get("1.0",tk.END)
    return exog_var

def getParameters(self):
    obj = self.containers[strParams]
    params = obj.get("1.0",tk.END)
    return params
    
def getParameterNamesAndValues(self): 
    params = getParameters(self)
    return getNamesAndValues(params)

def getParamRange(self):
    obj = self.containers[strSteadyState]
    param_range = obj.get("1.0",tk.END)
    return param_range
    
def getEquations(self):
    obj = self.containers[strEqs]
    equations = obj.get("1.0",tk.END)
    return equations

def getTimeRange(self):
    obj = self.containers[strTimeRange]
    time_range = obj.get("1.0",tk.END)
    return time_range

def getFormattedTimeRange(self):
    time_range = ''
    rng = getTimeRange(self)
    rng = rng.strip("\n")
    rng = rng.split("-")
    if len(rng) == 2:
        d1 = rng[0].strip()
        d1 = dt.datetime.strptime(d1,'%m/%d/%Y')
        d2 = rng[1].strip()
        d2 = dt.datetime.strptime(d2,'%m/%d/%Y')
        time_range = '[[' + str(d1.year) + ',' + str(d1.month) + ',' + str(d1.day) + '],[' + str(d2.year) + ',' + str(d2.month) + ',' + str(d2.day) + ']]'
    else:
        time_range = rng
    return time_range

def getFrequency(self):
    try:
        obj = self.containers[strFreq]
        selection = obj.curselection()
        freq = obj.get(selection[0])
        ind = strFreqList.index(freq)
        if ind == -1:
            ind = 0
    except:
        ind = 0
    return str(ind)

def getPeriods(self):
    periods = ''
    shocks = getShocks(self).split("\n")
    ind = -1
    for shock in shocks:
        ind = shock.find('Date')
        if ind >= 0:
            txt = shock[1+ind:]
            ind2 = txt.find(":")
            if ind2 >= 0:
                txt = txt[1+ind2:].strip()
                d = dt.datetime.strptime(txt,'%m/%d/%Y')
                periods = periods + '[' + str(d.year) + ',' + str(d.month) + ',' + str(d.day) + '],'
    periods = periods[:-1]
    if ind >= 0:
        periods = '[' + periods[0:len(periods)-1] + ']'
    return '[' + periods + ']'
  
def output(self,f,label,txt):
    #print(txt)
    f.write(label)
    f.write('\n')
    f.writelines(txt + '\n')
    f.write('\n')
    
         
def SaveYamlOutput(self):
    """
    Writes GUI data to YAML template text file
    """
    from snowdrop.src.utils.util import  SaveToYaml

    description = self.description
    shock_names,shock_values = getShockNamesAndValues(self)
    variables_names,variables_init_values = getVariableNamesAndInitialValues(self)
    param_names,param_values = getParameterNamesAndValues(self)
    exog_var = getExogVariables(self)
    equations = getEquations(self)
    time_range = getFormattedTimeRange(self)
    freq = getFrequency(self)
    periods = getPeriods(self)
    param_range = getParamRange(self)
    input_file = self.input_file
    bInp = not input_file is None and input_file.endswith(".inp")
    
    SaveToYaml(file=self.input_file,description=description,shock_names=shock_names,shock_values=shock_values,
             variables_names=variables_names,variables_init_values=variables_init_values,comments=self.comments,
             param_names=param_names,param_values=param_values,exog_var=exog_var,equations=equations,
             time_range=time_range,freq=freq, periods=periods,param_range=param_range,bInp=bInp)
            
            
def SaveTemplateOutput(self):
    """
    Wtites GUI data to template text file
    """
    path = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.abspath(os.path.join(path, '../../models/template.txt'))
    
    description = getDescription(self)
    eqs = getEquations(self)
    params = getParameters(self)
    initValues = getVariables(self) 
    exogVariables = getExogVariables(self)
    shocks = getShocks(self)
    timeRange = getTimeRange(self)
    freq = getFrequency(self)
      
    with open(file_path, 'w') as f:
        output(self,f,strDescription,description) 
        output(self,f,strEqs,eqs) 
        output(self,f,strParams,params) 
        output(self,f,strEndogVarInitValues,initValues)  
        output(self,f,strExogVariables,exogVariables)  
        output(self,f,strShocks,shocks) 
        output(self,f,strTimeRange,timeRange)  
        output(self,f,strFreq,freq) 

if __name__ == '__main__':
    path = os.path.dirname(os.path.abspath(__file__))
    working_dir = os.path.abspath(os.path.join(path,"../"))
    sys.path.append(working_dir)
    file_path = os.path.abspath(os.path.join(path,'../../models/PAMISV01.inp'))
    
    root = tk.Tk()
    app = Application(root)
    app.master.title("Equations Editor")
    app.master.minsize(minWidth, minHeight)
    app.master.maxsize(maxWidth, maxHeight)
    app.mainloop()
