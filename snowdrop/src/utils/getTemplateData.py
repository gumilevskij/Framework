import os
import re

def readTemplateFile(file_path=None):
    """
    Read data from a template text file
    """
                
    if file_path is None:
        path = os.path.dirname(os.path.abspath(__file__))
        file_path = path + '/../models/template.txt'
    #print(file_path)
    txt=[];txtEqs=[]; txtParams=[]; txtParamsRange=[];txtEndogVars=[];txtExogVars=[];txtShocks=[]
    txtRange=''; txtFreq='';txtDescription=''
    strEqs='Equations'
    strParams = 'Parameters'
    strEndogVariables = 'Endogenous'
    strExogVariables = 'Exogenous'
    strShocks = 'Shocks'
    strTimeRange = 'Time'
    strFreq = 'frequency'
    header = None
    with open(file_path, 'r') as f:
        for line in f:
            ln = line.strip()
            if ln in [strEqs,strParams,strShocks,strTimeRange,strFreq]:
                header = ln
                txt = []
            elif len(txt) > 0 and not ln:
                if header == strEqs:
                    txtEqs = txt
                elif header == strParams:
                    txtParams = txt
                elif header == strEndogVariables:
                    txtEndogVars = txt
                elif header == strExogVariables:
                    txtExogVars = txt
                elif header == strShocks:
                    txtShocks = txt
                elif header == strTimeRange:
                    txtRange = txt[0]
                elif header == strFreq:
                    txtFreq = txt[0]
            elif len(ln)>0:
                 txt.append(line)
                    
    endog = []; par = []
    delimiters = " ", ",", ";", "*", "/", "+", "-", ":","(", ")","^"
    regexPattern = '|'.join(map(re.escape, delimiters))
    arr = re.split(regexPattern," ".join(txtEqs))
    for e in arr:
        el = e.strip()
        if "'p" in el:
            par.append(el.strip("'p"))
        elif "'n" in el:
            endog.append(el.strip("'n"))
            
    #print(txtEqs,txtParams,txtParamsRange,txtEndogVars,txtExogVars,txtShocks,txtRange,txtFreq,txtDescription)  
    return txtEqs,txtParams,txtParamsRange,txtEndogVars,txtExogVars,txtShocks,txtRange,txtFreq,txtDescription
   