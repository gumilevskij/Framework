# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 17:44:13 2019

@author: AGoumilevski
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

def test():
        
    # Default delta is large because that makes it fast, and it illustrates
    # the correct registration between image and contours.
        
    delta = 0.5
    extent = (0, 2, 0, 2)
    X = np.arange(extent[0], extent[1], delta)
    Y = np.arange(extent[2], extent[3], delta)
    PDOT = np.meshgrid(X, Y)[0]
    PDOT_old,Y_old = PDOT,Y
    RR = 0.0
    RS = 0.0
    
    g = 0.049
    p_pdot1 = 0.414
    p_pdot2 = 0.196
    p_pdot3 = 0.276
    p_rs1 = 3.000
    p_y1 = 0.304
    p_y2 = 0.098
    p_y3 = 0.315
    
    for i in range(1):
        PDOT = (PDOT - (1-p_pdot1)*PDOT_old - p_pdot2*(g*g/(g-Y) - g) - p_pdot3*(g*g/(g-Y_old) - g)) / p_pdot1
        RR = RS - p_pdot1*PDOT - (1-p_pdot1)*PDOT_old
        RS = p_rs1*PDOT + Y_old
        #Y = p_y1*Y_old - p_y2*RR - p_y3*RR
    
    Z = PDOT
    
    # Boost the upper limit to avoid truncation errors.
    levels = np.arange(-2.0, 1.601, 0.4)
    
    norm = cm.colors.Normalize(vmax=abs(Z).max(), vmin=-abs(Z).max())
    cmap = cm.PRGn
    
    fig, axes = plt.subplots(nrows=2, ncols=2)
    fig.subplots_adjust(hspace=0.3)
    axs = axes.flatten()
    
    cset1 = axs[0].contourf(X, Y, Z, levels, norm=norm,
                         cmap=cm.get_cmap(cmap, len(levels) - 1))
    # It is not necessary, but for the colormap, we need only the
    # number of levels minus 1.  To avoid discretization error, use
    # either this number or a large number such as the default (256).
    
    # If we want lines as well as filled regions, we need to call
    # contour separately; don't try to change the edgecolor or edgewidth
    # of the polygons in the collections returned by contourf.
    # Use levels output from previous call to guarantee they are the same.
    
    cset2 = axs[0].contour(X, Y, Z, cset1.levels, colors='k')
    
    # We don't really need dashed contour lines to indicate negative
    # regions, so let's turn them off.
    
    for c in cset2.collections:
        c.set_linestyle('solid')
    
    # It is easier here to make a separate call to contour than
    # to set up an array of colors and linewidths.
    # We are making a thick green line as a zero contour.
    # Specify the zero level as a tuple with only 0 in it.
    
    cset3 = axs[0].contour(X, Y, Z, (0,), colors='g', linewidths=2)
    axs[0].set_title('Filled contours')
    fig.colorbar(cset1, ax=axs[0])
    
    
    axs[1].imshow(Z, extent=extent, cmap=cmap, norm=norm)
    axs[1].contour(Z, levels, colors='k', origin='upper', extent=extent)
    axs[1].set_title("Image, origin 'upper'")
    
    axs[2].imshow(Z, origin='lower', extent=extent, cmap=cmap, norm=norm)
    axs[2].contour(Z, levels, colors='k', origin='lower', extent=extent)
    axs[2].set_title("Image, origin 'lower'")
    
    # We will use the interpolation "nearest" here to show the actual
    # image pixels.
    # Note that the contour lines don't extend to the edge of the box.
    # This is intentional. The Z values are defined at the center of each
    # image pixel (each color block on the following subplot), so the
    # domain that is contoured does not extend beyond these pixel centers.
    im = axs[3].imshow(Z, interpolation='nearest', extent=extent,
                    cmap=cmap, norm=norm)
    axs[3].contour(Z, levels, colors='k', origin='image', extent=extent)
    ylim = axs[3].get_ylim()
    axs[3].set_ylim(ylim[::-1])
    axs[3].set_title("Origin from rc, reversed y-axis")
    fig.colorbar(im, ax=axs[3])
    
    fig.tight_layout()
    plt.show()

            
if __name__ == '__main__':
    """
    The main test program.
    """
    test()