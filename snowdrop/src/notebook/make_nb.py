"""Create a notebook that contains code from a script.

Run as:  python make_nb.py my_script.py
"""
import os,sys

import nbformat
from nbformat.v4 import new_notebook, new_code_cell

path = os.path.dirname(os.path.abspath(__file__))

def convert():
    """Convert python code to jupyter notebook."""
    fname = sys.argv[1]
    name,ext = os.path.splitext(fname)
        
    nb = new_notebook()
    with open(path+os.path.sep+fname) as f:
        code = f.read()
    
    nb.cells.append(new_code_cell(code))
    nbformat.write(nb, path+os.path.sep+name+'.ipynb')
    
    
if __name__ == '__main__':
    if len(sys.argv) > 1:
        convert()