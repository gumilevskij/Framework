# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 19:30:31 2017

@author: agoumilevski

This module merges PDF files

Usage:
merge  -output Output_File -dir directory -files Comma_Separated_List -titles Comma_Separated_List -footnotes  Comma_Separated_List -scale  Comma_Separated_List
"""

import sys,os,io
from math import ceil
from datetime import datetime, timedelta
import argparse
from PyPDF2 import PdfFileWriter, PdfFileReader
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

path = os.path.dirname(os.path.abspath(__file__))
working_dir = os.path.abspath(path+"\\..\\..")

def skipPage(pageObj):
    """Skip a page."""
    str = 'Latest Update:'
    content = pageObj.extractText()
    ind = content.find('Latest Update:')
    if (ind < 0):
        return False
    ind += len(str)
    latest_update = content[1+ind:ind+6]
    #print(latest_update)
    month = latest_update[:2]
    day = latest_update[3:]
    month = int(month)
    day = int(day)
    now = datetime.now()
    latest_update = datetime(now.year, month, day, 0, 0)
    if (latest_update < now-timedelta(days=7)):
        return True
    else:
        return False
    
    
def createPdf(texts,header):
    """Create a pdf object."""
    packet = io.BytesIO()
    # create a new PDF with Reportlab
    can = canvas.Canvas(packet, pagesize=letter)
    if header:
        can.setFont("Calibri",15)
        can.drawCentredString(300, 760, texts[0])
    else:
        can.setFont("Calibri",10)
        for i,txt in enumerate(texts):
            can.drawString(30, max(1,750-11*i), txt)
    can.save()
    #move to the beginning of the StringIO buffer
    packet.seek(0)
    pdfObj = PdfFileReader(packet)
    
    return pdfObj


def createPdfs(text,header=True):
    """Create a pdf object with multiple pages."""
    texts = text.split('^')
    size = 60
    n = len(texts)
    m = ceil(n/size)
    if n <= size:
        pdfObj = createPdf(texts,header)
    else:
        pdfObj = PdfFileWriter()
        for i in range(m):
            texts_block = texts[i*size:min(n,(i+1)*size)]
            newObj = createPdf(texts_block,header).getPage(0)
            pdfObj.addPage(newObj)
            
    return pdfObj


def merge(output_file,folder,pdf_files,titles,scales,footnotes='',allPages='',align=''):
    """Merge pdf files."""
    pdfmetrics.registerFont(TTFont('Calibri', 'Calibri.ttf'))
    #print('output_file:',output_file,'; dir:',folder,'; pdf_files:',pdf_files,'; titles:',titles,'; footnotes:',footnotes,'; scale:',scales,'; allPages:',allPages)
    if type(pdf_files).__name__ == "str": 
        pdf_files = pdf_files.split(',')  
    if type(titles).__name__ == "str": 
        titles = titles.split(',')
    if type(footnotes).__name__ == "str": 
        footnotes = footnotes.split(';')
    if type(scales).__name__ == "str": 
        scales = scales.split(',')
    if type(align).__name__ == "str": 
        shifts = align.split(';')
    if type(allPages).__name__ == "str": 
        allpages = allPages.split(',')
    try:
        pdfWriter  = PdfFileWriter()
        for i,f in enumerate(pdf_files):   
            path_to_file = os.path.abspath(folder) + '\\' + f
            pdfReader = PdfFileReader(open(path_to_file,'rb'))
            numPages = pdfReader.numPages
            if (i<len(titles)):
                title = titles[i]
            else:
                title = '' 
            if (i<len(footnotes)):
                footnote = footnotes[i]
            else:
                footnote = ''
            if (i<len(scales)):
                scale = float(scales[i])
            else:
                scale = 1.0  
            if (i<len(shifts)):
                shift = shifts[i].split(',')
                if (len(shift) > 1):
                    shift = [float(shift[0]),float(shift[1])]
                else:
                    shift = [0,0]
            else:
                shift = [0,0] 
            if (i<len(allpages) and allpages[i]):
                outputAllPages = int(allpages[i])
            else:
                outputAllPages = 0    
            for pageNum in range(numPages):
                #print(f,pageNum,numPages)
                newPage = pdfReader.getPage(pageNum)
                if (outputAllPages == 1):
                    skip = False
                else:
                    skip =  skipPage(newPage)
                if (not skip):
                    if pageNum == 0:
                        pdf_title = createPdfs(title,True)
                    else:
                        pdf_title = createPdfs('')
                    pageObj = pdf_title.getPage(0)
                    pageObj.mergeScaledTranslatedPage(newPage,scale,shift[0],shift[1]) 
                    pdfWriter.addPage(pageObj)
                    if (len(footnote)>0 and pageNum==numPages-1): 
                        pdf_footer = createPdfs(footnote,False)
                        numPages2 = pdf_footer.getNumPages()
                        for pageNum2 in range(numPages2):
                            newPage = pdf_footer.getPage(pageNum2)
                            pdfWriter.addPage(newPage)
                            
        with open(os.path.abspath(folder)  + '\\' + output_file, "wb") as fout:
            pdfWriter.write(fout)
            
        print('Generated report ' + os.path.abspath(folder)  + '\\' + output_file)
    except Exception as e:
        print("Unable to generate report " + output_file)
        print('File: ' + f)
        print('   Error: ' + str(sys.exc_info()[0]))
        print(e)
        raise e


def findCommand(cmd,txt):
    v = [s for s in txt if cmd in s]
    if len(v) == 0:
        return ""
    else:
        return v[0].replace(cmd,"").strip("\n").strip()
        
    
def driver(file_path):
    """Driver program."""
    # Read file
    with open(file_path, 'r') as f:
        txt = f.readlines()
		
    # Get arguments
    output_file = findCommand('-output', txt)
    folder = findCommand ('-dir', txt)
    files = findCommand ('-pdf_files', txt)
    titles = findCommand ('-titles', txt)
    footnotes = findCommand ('-footnotes', txt)
    scales = findCommand ('-scale', txt)
    allpages = findCommand ('-allpages', txt)
    align  = findCommand('-align', txt)
    if not titles:
        titles=''
    if not footnotes:
        footnotes=''
    if not scales:
        scales='1'
    if not align:
        align = ';'
    if not allpages:
        allpages=''    
		
    # Call merge script    
    merge(output_file,folder,files,titles,footnotes,scales,allpages,align)    
    
    
def main(argv):
    """
    This is a main function.
    """  
    parser = argparse.ArgumentParser(description='Merges PDF files')
    parser.add_argument('-cmd_path', help="path to cmd file")
    parser.add_argument('-output', help="output pdf")
    parser.add_argument('-dir', help="pdf files folder")
    parser.add_argument('-files', help="pdf files")
    parser.add_argument('-titles', help="titles of pdf files")
    parser.add_argument('-footnotes', help="footnotes of pdf files")
    parser.add_argument('-scale', help="scale measure of pdf files")
    parser.add_argument('-allpages', help="flag that indicates if all pages of odf file should be included ")
    parser.add_argument('-align', help="horizontal and vertical offset of pdf files")
    args = parser.parse_args()
    cmd_path = args.cmd_path
    if not cmd_path is None and os.path.exists(cmd_path):
       driver(cmd_path)
       return
    output_file = args.output
    folder = args.dir
    files = args.files
    titles = args.titles
    footnotes = args.footnotes
    scales = args.scale
    allpages = args.allpages
    align = args.align
    if titles is None:
        titles=''
    if footnotes is None:
        footnotes=''
    if scales is None:
        scales='1'
    if align is None:
        align = ';'
    if allpages is None:
        allpages=''
        
    merge(output_file,folder,files,titles,footnotes,scales,allpages,align)
    
    
if __name__ == "__main__":
    main(sys.argv[1:])
