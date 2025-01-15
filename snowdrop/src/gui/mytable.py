# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 18:13:56 2019

@author: agoumilevski
"""
import numpy as np
import tkinter as tk    
from tkinter import NSEW,END,RIDGE
from tkinter import Entry 
from tkinter import messagebox
from math import ceil
    
minWidth = 600
minHeight = 300
maxWidth = 1000
maxHeight = 600
N = 0
m = {} 
data = [] 


def getValues(container,names=[],vals=[]):
    N = len(names)
    n_columns = 3
    n_rows = ceil(N/n_columns)
    data.clear()
    
    lb = tk.Label(container)
    lb.grid(row=0, column=0)
    lb = tk.Label(container)
    lb.grid(row=1, column=0)
    lb = tk.Label(container)
    lb["text"] = "Date"
    lb.grid(row=2, column=0, sticky=NSEW)
    e = Entry(container,relief=RIDGE)
    e.grid(row=2, column=1, sticky=NSEW)
    e.insert(END, "2017/1/1")
    data.append(e)
    lb = tk.Label(container)
    lb.grid(row=3, column=0)
            
    k = 0
    for i in range(n_rows):
        for j in range(n_columns):
            k += 1
            if k >= N:
                break
            # Populate entry name
            lb = tk.Label(container)
            lb["text"] = names[k]
            lb.grid(row=i+4, column=2*j, sticky=NSEW)
            e = Entry(container,relief=RIDGE)
            e.grid(row=i+4, column=2*j+1, sticky=NSEW)
            e.insert(END, vals[k])
            data.append(e)
            
    lb = tk.Label(container)
    lb.grid(row=n_rows+6, column=0)
    lb = tk.Label(container)
    lb.grid(row=n_rows+7, column=0)
       
    # Local functions
    def onSave():
        e = data[0]
        dt = e.get()
        vals = {}
        for i in range(1,N):
            e = data[i]
            d = e.get().strip()
            name = names[i-1]
            if len(d) > 0:
                vals[name] = d
            else:
                vals[name] = np.nan
        m[dt] = vals
        
        messagebox.showinfo("Shocks Editor","Shocks saved.")
 
        
    def onClean():
        m.clear()
        for i in range(1,N):
            e = data[i]
            e.delete(0,END)
            e.insert(0,"0")
            
        messagebox.showinfo("Shocks Editor","Shocks cleaned.")
            
    def onClose():
        container.destroy()
    
    bt1 = tk.Button(container,text='Save',width=2,height=1,command=onSave)
    bt1.grid(row=10+n_rows,column=2, sticky=NSEW)
    bt2 = tk.Button(container,text='Clean',width=2,height=1,command=onClean)
    bt2.grid(row=10+n_rows,column=3, sticky=NSEW)
    bt3 = tk.Button(container,text='Close',width=2,height=1,command=onClose)
    bt3.grid(row=10+n_rows,column=4, sticky=NSEW)
    
    return

def showTable(title="Table Editor",names=[],vals=[]):
    m.clear()
    data.clear()
    root = tk.Tk()
    root.title(title)
    root.minsize(minWidth, minHeight)
    root.maxsize(maxWidth, maxHeight)
   
    getValues(container=root,names=names,vals=vals)
    root.mainloop()
    return
    
if __name__ == '__main__':
    """
    Main entry point
    """
       
    names=['RES_L_GDP_BAR','RES_L_GDP_GAP','RES_DL_GDP_BAR','RES_PIE','RES_PIE_TAR',
           'RES_UNR_GAP','RES_UNR_BAR','RES_G_UNR_BAR','RES_CAPU_BAR','RES_DL_CU_BAR']
    vals=['0']*len(names)
    showTable(title="Shock Editor",names=names,vals=vals)
    print(m)