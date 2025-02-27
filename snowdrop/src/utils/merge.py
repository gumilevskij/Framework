# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 19:30:31 2017
@author: agoumilevski

This module merges PDF files

Usage:
merge  -output Output_File -files Comma_Separated_List
"""

import sys,os
import warnings
import argparse
from snowdrop.src.misc.termcolor import cprint

try:
    from PyPDF2 import PdfMerger
except:
    from PyPDF2 import PdfFileMerger as PdfMerger

path = os.path.dirname(os.path.abspath(__file__))
working_dir = os.path.abspath(path+"\\..\\..")

def merge(output_file,pdf_files):
    """Merge several pdf files into one."""
    reader = None
    try:
        merger = PdfMerger()
        for pdf in pdf_files:
            if not os.path.isfile(pdf):
                cprint("File {} does not exist".format(pdf),"red")
            reader = open(pdf, 'rb')
            merger.append(reader)
        with open(output_file, "wb") as fout:
            merger.write(fout)
        if not reader is None:
            reader.close() 
        del reader
        print('Generated report ' + output_file)
    except:
        msg = "Unable to generate report " + output_file + '   Error: ' + str(sys.exc_info()[0])
        warnings.warn(msg)
    
    
def main(argv):
    """This is a main function."""
    parser = argparse.ArgumentParser(description='Merges PDF files')
    parser.add_argument('-files', help="pdf files")
    parser.add_argument('-output', help="output pdf")
    args = parser.parse_args()
    pdf_files = args.files.split(',')
    output_file = args.output
    merge(output_file,pdf_files)
    
    
if __name__ == "__main__":
    main(sys.argv[1:])
