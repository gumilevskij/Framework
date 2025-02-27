# -*- coding: utf-8 -*-
"""
Created on Tue May 21 19:29:30 2019

@author: A.Goumilevski
"""
def showDialog(msg,timeout=5000,title="DIGNAR-19 Toolkit"):
    """Simple implementation of dialog window.

    Parameters:
        msg : str
            Message to be shown.
        timeout : int
            Timeout in milliseconds.

    Returns:
        bool
            True if left button is clicked.

    """
    import tkinter as tk
    root = tk.Tk()
    
    class MyDialog(tk.Frame):
        
        def __init__(self,master=None,msg=None):
            root.after(timeout,root.destroy)
            self.master = master
            self.yes_no = None
            label = tk.Label(root, text=msg) 
            label.pack(side="top",fill="both",expand=True, padx=20, pady=20) 
            button = tk.Button(root, text="Close",command=lambda: self.ok()) 
            button.pack(side=tk.LEFT,padx=20,pady=5,expand=False) 
            button.config(height=2,width=10)
            button = tk.Button(root, text="No",command=lambda: self.no()) 
            button.pack(side=tk.RIGHT,padx=20,pady=5,expand=False) 
            button.config(height=2,width=10)
            
        def ok(self):
            self.yes_no="yes"
            root.destroy()
            
        def no(self):
            self.yes_no="No"
            # Do nothing
            # root.destroy()
            
    def dialog(msg,title):
        app = MyDialog(root,msg)
        app.master.minsize(400,150)
        app.master.maxsize(450,200)
        app.master.title(title)
        root.mainloop()
        return app.yes_no

    
    ret = dialog(msg=msg,title=title)
    return ret == "yes"

    
if __name__ == '__main__':
    showDialog(msg="Test",timeout=5000,title="Information")