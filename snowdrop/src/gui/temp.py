# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 16:15:09 2018

@author: agoumilevski
"""

import matplotlib
import numpy as np
from matplotlib.figure import Figure
from tkinter import *

class mclass:

    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    matplotlib.use('TkAgg')
    def __init__(self,  window):
        self.window = window
        self.box = Entry(window)
        self.button = Button (window, text="plot", command=self.plot)
        self.box.pack ()
        self.button.pack()

    def plot (self):
        x=np.array ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        v= np.array ([16,16.31925,17.6394,16.003,17.2861,17.3131,19.1259,18.9694,22.0003,22.81226])
        p= np.array ([16.23697,     17.31653,     17.22094,     17.68631,     17.73641 ,    18.6368,
            19.32125,     19.31756 ,    21.20247  ,   22.41444   ,  22.11718  ,   22.12453])

        fig = Figure(figsize=(6,6))
        a = fig.add_subplot(111)
        a.scatter(v,x,color='red')
        a.plot(p, range(2 +max(x)),color='blue')
        a.invert_yaxis()

        a.set_title ("Estimation Grid", fontsize=16)
        a.set_ylabel("Y", fontsize=14)
        a.set_xlabel("X", fontsize=14)

        canvas = FigureCanvasTkAgg(fig, master=self.window)
        canvas.get_tk_widget().pack()
        canvas.draw()


if __name__ == "__main__":
    window= Tk()
    start= mclass (window)
    window.mainloop()
