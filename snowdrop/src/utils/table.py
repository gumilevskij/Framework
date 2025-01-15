# -*- coding: utf-8 -*-
"""
Created on Thu May  2 18:05:37 2019

@author: AGoumilevski
"""
import os
import numpy as np
from reportlab.platypus.flowables import Flowable
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, inch
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate
from reportlab.platypus import Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from snowdrop.src.misc.termcolor import cprint
# from reportlab.pdfbase import pdfmetrics
# from reportlab.pdfbase.ttfonts import TTFont

#pdfmetrics.registerFont(TTFont('DejaVuSerif', 'DejaVuSerif.ttf'))

path = os.path.dirname(os.path.abspath(__file__))
    
class TR(Flowable):
    """Rotate a text in a table cell."""
    
    def __init__(self, text ):
        Flowable.__init__(self)
        self.text=text
        
    def draw(self):
        canv = self.canv
        canv.rotate(90)
        canv.drawString( 0, -1, self.text)
        
            
class TB(Flowable):
    """Use bold italic font for a text in a table cell."""
    
    def __init__(self, text ):
        Flowable.__init__(self)
        self.text=text
        
    def draw(self):
        canv = self.canv
        canv.setFont('Times-BoldItalic',16)
        canv.drawString( 0, -1, self.text)
		
def matrix_to_pdf(headers,colNames,rowNames,matrices,fout):
    """
    Create a simple pdf document from a two-dimensional array.
    
    Parameters:
        :param header: Header of a table.
        :type header: str.
        :param colNames: Names of table columns.
        :type colNames: list.
        :param rowNames: Names of table rows.
        :type rowNames: list.
        :param matrices: Two dimensional matrix.
        :type matrices: np.array.
        :param table_rng: Range of dates.
        :type table_rng: list.
        :param fout: Path to pdf document.
        :type fout: str.
        :returns: Simulation results.
    """
    styleSheet = getSampleStyleSheet()
    doc = SimpleDocTemplate(fout, pagesize=(800,800))
	
    # container for the 'Flowable' objects
    elements = []
    
    ncol = len(colNames)
    n = len(headers)
    columns = []
    for c in colNames:
        columns.append(TR(c))
    
    for i in range(n):
        header = headers[i]
        matrix = matrices[i]
        nrow = len(matrix)
        
        data= [['']*(1+ncol),['']+columns,['']*(1+ncol)]
        for j,r in enumerate(rowNames):
            s = [str(round(x,2)) for x in matrix[j]]
            data.append([r] + s)
        data.append(['']*(1+ncol))
            
        txt = '''<para align=center spaceb=3 size=16><b>{0}</b></para>'''.format(header)
        p = Paragraph(txt,styleSheet["BodyText"])
        elements.append(p)
        p = Paragraph('',styleSheet["BodyText"])
        elements.append(p)
                
        t = Table(data,[0.7*inch]+ncol*[0.5*inch],
                  [0.01*inch,0.35*inch,0.01*inch]+[0.35*inch]+(nrow-2)*[0.3*inch]+[0.35*inch]+[0.01*inch])
        t.setStyle(TableStyle([('ALIGN',(0,0),(0,-1),'LEFT'),
                               ('ALIGN',(1,1),(-1,-1),'CENTER'),
                               ('VALIGN',(1,1),(-1,1),'BOTTOM'),
                               ('HALIGN',(1,1),(-1,1),'CENTER'),
                               ('BACKGROUND',(0,0),(-1,0), colors.black),
                               ('BACKGROUND',(0,2),(-1,2), colors.black),
                               ('BACKGROUND',(0,-1),(-1,-1), colors.black),
    						   ('TEXTCOLOR',(1,1),(-1,-1),colors.black),
    						   ('FONTSIZE',(0,0),(-1,1),12),
    						   ('FONTSIZE',(0,0),(0,-1),12),
    						   ('FONTSIZE',(1,2),(-1,-1),10),
    						   ]))
    
        elements.append(t)
        p = Paragraph('',styleSheet["BodyText"])
        elements.append(p)
        elements.append(p)
        elements.append(p)
    
	# write the document to disk
    doc.build(elements)

        
def table_to_pdf(header,colNames,rowNames,matrix,table_rng,fout,date=None):
    """
    Create a pdf document from a two-dimensional matrix.
    
    Parameters:
        :param header: Header of a table.
        :type header: str.
        :param colNames: Names of table columns.
        :type colNames: list.
        :param rowNames: Names of table rows.
        :type rowNames: list.
        :param matrix: Two dimensional matrix.
        :type matrix: np.array.
        :param table_rng: Range of dates.
        :type table_rng: list.
        :param fout: Path to pdf document.
        :type fout: str.
        :param date: Date at which vertical line in a table is drawn.
        :type date: datetime.date. 
        :returns: Simulation results.
    """
    styleSheet = getSampleStyleSheet()
    doc = SimpleDocTemplate(fout, pagesize=(1200,850))
	
    # Container for the 'Flowable' objects
    elements = []
    ncol = len(colNames)
    nrow = len(rowNames)
    if not date is None:
        lst = [i for i,x in enumerate(table_rng) if x==date]
        if len(lst) > 0:
            n = 1+lst[0]
        else:
            n = 1
        col = colors.black
    else:
        n = 1
        col = colors.white
    n = min(n,len(colNames))
    
    m = 0
    for r in rowNames:
        m = max(m,len(r))
            
    data= [['']*(m+1+ncol),['']*m+colNames[:n]+['']+colNames[n:],['']*(m+1+ncol)]
    for j,r in enumerate(rowNames):
        if j >= len(matrix):
            continue
        mat = matrix[j]
        try:
            if len(mat)==0 and len(r)==0:
                data.append([""]*(m+1+ncol))
            elif len(mat)==0 and len(r)>0:
                if len(r)==1:
                    data.append([TB(r[0])]+[""]*(m+ncol))
                elif len(r)==2:    
                    data.append([TB(r[0]),TB(r[1])]+[""]*(m-1+ncol))
            elif len(mat)>0 and len(r)==0:
                mt = [mat[x] if x in mat else np.nan for x in table_rng]
                s = [str(round(np.real(x),1)) if np.imag(x)==0 else str(np.around(x,1)) for x in mt]
                s = [x.replace("nan","") for x in s]
                s = ['{0: >5}'.format(x) for x in s]
                data.append([""]*m+s[:n]+[""]+s[n:])
            else:
                mt = [mat[x] if x in mat else np.nan for x in table_rng]
                s = [str(round(np.real(x),1)) if np.imag(x)==0 else str(np.around(x,1)) for x in mt]
                s = [x.replace("nan","") for x in s]
                s = ['{0: >5}'.format(x) for x in s]
                data.append(r+(m-len(r))*[""]+s[:n]+[""]+s[n:])
        except Exception as exc:
            cprint(f"table_to_pdf: {exc}","red")
            
    data.append(['']*(3+ncol))
        
    txt = '''<para align=center spaceb=3 size=22><b>{0}</b></para>'''.format(header)
    p = Paragraph(txt,styleSheet["BodyText"])
    elements.append(p)
    p = Paragraph('',styleSheet["BodyText"])
    elements.append(p)
    elements.append(p)
    elements.append(p)
            
    t = Table(data,[2*inch]+(m-1)*[2*inch]+n*[0.8*inch]+[0.01*inch]+(ncol-n)*[0.8*inch],
              [0.01*inch,0.4*inch,0.01*inch]+[0.4*inch]+(nrow-2)*[0.23*inch]+[0.4*inch]+[0.01*inch])
    t.setStyle(TableStyle([('ALIGN',(0,0),(0,-1),'LEFT'),
                           ('ALIGN',(1,1),(-1,-1),'CENTER'),
                           ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
                           ('VALIGN',(0,-1),(-1,-1),'BOTTOM'),
                           ('HALIGN',(1,1),(-1,1),'CENTER'),
                           ('BACKGROUND',(0,0),(-1,0), colors.black),
                           ('BACKGROUND',(0,2),(-1,2), colors.black),
                           ('BACKGROUND',(0,-1),(-1,-1), colors.black),
                           ('BACKGROUND',(n+m,0),(n+m,-1), col),
						   ('TEXTCOLOR',(1,1),(-1,-1),colors.black),
						   ('FONTSIZE',(0,0),(-1,1),16),
						   ('FONTSIZE',(0,0),(0,-1),14),
						   ('FONTSIZE',(1,2),(-1,-1),14)
						   ]))

    elements.append(t)
    p = Paragraph('',styleSheet["BodyText"])
    elements.append(p)
    elements.append(p)
    elements.append(p)
    elements.append(p)
    
	# write the document to disk
    doc.build(elements)


def test():
    """Create a sample pdf document with complex cell values."""
    doc = SimpleDocTemplate(path + "\\..\\..\\results\\complex_cell_values.pdf", pagesize=letter)
	# container for the 'Flowable' objects
    elements = []

    styleSheet = getSampleStyleSheet()

    I = Image(path + "\\..\\..\\img\\replogo.gif")
    I.drawHeight = 1.25*inch*I.drawHeight / I.drawWidth
    I.drawWidth = 1.25*inch
    P0 = Paragraph('''
				   <b>A pa<font color=red>r</font>a<i>graph</i></b>
				   <super><font color=yellow>1</font></super>''',
				   styleSheet["BodyText"])
    P = Paragraph('''
		<para align=center spaceb=3>The <b>ReportLab Left
		<font color=red>Logo</font></b>
		Image</para>''',
		styleSheet["BodyText"])
    data= [['A', 'B', 'C', P0, 'D'],
		   ['00', '01', '02', [I,P], '04'],
		   ['10', '11', '12', [P,I], '14'],
		   ['20', '21', '22', '23', '24'],
		   ['30', '31', '32', '33', '34']]

    t=Table(data,style=[('GRID',(1,1),(-2,-2),1,colors.green),
						('BOX',(0,0),(1,-1),2,colors.red),
						('LINEABOVE',(1,2),(-2,2),1,colors.blue),
						('LINEBEFORE',(2,1),(2,-2),1,colors.pink),
						('BACKGROUND', (0, 0), (0, 1), colors.pink),
						('BACKGROUND', (1, 1), (1, 2), colors.lavender),
						('BACKGROUND', (2, 2), (2, 3), colors.orange),
						('BOX',(0,0),(-1,-1),2,colors.black),
						('GRID',(0,0),(-1,-1),0.5,colors.black),
						('VALIGN',(3,0),(3,0),'BOTTOM'),
						('BACKGROUND',(3,0),(3,0),colors.limegreen),
						('BACKGROUND',(3,1),(3,1),colors.khaki),
						('ALIGN',(3,1),(3,1),'CENTER'),
						('BACKGROUND',(3,2),(3,2),colors.beige),
						('ALIGN',(3,2),(3,2),'LEFT'),
	])
    t._argW[3]=1.5*inch

    elements.append(t)
	# write the document to disk
    doc.build(elements)


def test2():
    """Create a sample pdf document with some values."""
    matrix = [[0.1,0.3,1.2,.02,0,0],[0.1,0.3,1.2,.02,0,0],[0.1,0.3,1.2,.02,0,0],[0.1,0.3,1.2,.02,0,0]]
    headers  = ['YoY Inflation','QoQ Core Inflation','Nom. Exch. Rate, 100*log','Policy Rate','Output Gap']
    colNames = ['t+1','t+2','t+3','t+4','t+5','t+6']
    rowNames = ['Mean','Median','Std','SE']
    fout = os.path.abspath(path + "\\..\\..\\graphs\\statistics.pdf")
    data = []
    for i in range(len(headers)):
        data.append(matrix)
    
    matrix_to_pdf(headers,colNames,rowNames,data,fout)
    
    
if __name__ == '__main__':
    """
    The main program
    """
    test2()
        