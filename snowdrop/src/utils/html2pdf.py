#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 12:00:24 2021

@author: alexei
"""

import os
import pdfkit
from snowdrop.src.utils.merge import merge

path = os.path.dirname(os.path.abspath(__file__))
output_file = os.path.abspath(os.path.join(path,"../../api_docs/_build/html/Platform.pdf"))
dir1 = os.path.join(path,"../../api_docs/_build/html")
dir2 = os.path.join(dir1,"source")
dirs = [dir1,dir2]


def convertHTMLsToPFDs(dir,pdf_files,generatePdf=True):
    """Traverse folder and convert html document to pdf one by one."""
    
    for file in os.listdir(dir):
        if file.endswith(".html"):
            fname = os.path.abspath(os.path.join(dir,file))
            pdf = fname[:-4]+"pdf"
            pdf_files.append(pdf)
            print(pdf)
            if generatePdf: # or not os.path.exists(pdf):
                pdfkit.from_file(fname,pdf)
            
    return pdf_files
    
    
if __name__ == "__main__":
    """Read all html documents in folders, gerenerate pfds and create masted pdf document."""
   
    pdf_files = []
    for d in dirs:
        pdf_files = convertHTMLsToPFDs(d,pdf_files)

    pdf_files = ['index.pdf','info.pdf','source/modules.pdf','source/src.dignar.pdf',
                 'source/src.epidemic.pdf','source/src.graphs.pdf','source/src.gui.pdf',
                 'source/src.pdf','source/src.info.pdf','source/src.misc.pdf','source/src.model.pdf',
                 'source/src.notebook.pdf','source/src.numeric.bayes.pdf','source/src.numeric.calibration.pdf',
                 'source/src.numeric.dp.pdf','source/src.numeric.filters.pdf',
                 'source/src.numeric.grids.pdf','source/src.numeric.pdf','source/src.numeric.ml.pdf',
                 'source/src.numeric.sa.pdf','source/src.numeric.solver.pdf','source/src.olg.pdf',
                 'source/src.preprocessor.pdf','source/src.samples.pdf',
                 'source/src.tests.pdf','source/src.utils.pdf','genindex.pdf']
    
    pdf_files = [os.path.abspath(os.path.join(dir1,x)) for x in pdf_files]

    merge(output_file,pdf_files)
        