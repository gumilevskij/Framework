"""
Prints model object in pdf file format.
"""

import os
import pickle
from pdflatex import PDFLaTeX
from snowdrop.src.misc.text2latex import getLatexDocument

path = os.path.dirname(os.path.abspath(__file__))
working_dir = os.path.join(path,"..")


def print_model(model_name,fout):
    """
    Create PDF from .tex file.
    
    Parameters:
        model_name : str
            Name of model file
        fout : str
            Outtput file path

    Returns:
        None.

    """
    ### Read model file
    fpath = os.path.abspath(os.path.join(working_dir,"../models",model_name))
    
    text = getLatexDocument(fpath)
    
    pdfl = PDFLaTeX.from_texfile(fpath)
    pdf, log, completed_process = pdfl.create_pdf(keep_pdf_file=True, keep_log_file=True)
    
    with open(fout, "wb") as f:
        pickle.dump(text, f)
    
    
if __name__ == '__main__':
    """Test program."""
    
    # Model name
    name = 'MVF_US.yaml'
    fout = os.path.abspath(os.path.join(working_dir,"../results",name[:-5]+".pdf"))
    print_model(model_name = name, fout = fout)
    
    