#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 20:14:35 2024

@author: alexei
"""

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np

# Create initial plot
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)

x = np.arange(0, 2*np.pi, 0.01)
line, = ax.plot(x, np.sin(x))

# Create slider
axfreq = plt.axes([0.25, 0.1, 0.65, 0.03])
freq_slider = Slider(axfreq, 'Frequency', 0.1, 10.0, valinit=1.0)


# Update plot function
def update(val):
    line.set_ydata(np.sin(freq_slider.val * x))
    fig.canvas.draw_idle()

# Connect slider to update function
freq_slider.on_changed(update)

plt.show()