import os
import numpy as np
from datetime import date


def readXmlModelFile(file_path=None,bFillValues=True):
    """Read XML model file."""
    from xml.etree import ElementTree as ET
    
    if file_path is None:
        path = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.abspath(os.path.join(path,'../../examples/models/Sirius//morocco_model.xml'))
        
    txtEqs="";txtParams="";txtParamsRange="";txtEndogVars=[];txtExogVars="";txtShocks="";txtRange='';txtFreq='' 
    
    doc = ET.parse(file_path).getroot()
    itemlist = doc.findall("./modelname")
    txtModelName = itemlist[0].text
   
    # Get variables names and types
    itemlist = doc.findall("./symbols/symbol")
    endog = []; exog = []
    for i in itemlist:
        n = i.attrib["name"]
        t = i.attrib["type"]
        if t == "endo":
            endog.append(n)
        elif t == "shock":
            exog.append(n)
          
    # Get equations
    itemlist = doc.findall("./equations/equation")
    eq_labels = []; eqs = []
    for i in itemlist:
        child = i.find("label")
        label = child.text
        child = i.find("expression")
        eq = child.text
        eq_labels.append(label)
        eqs.append(eq)
        
    # Get variable initial values
    itemlist = doc.findall("./values/levels/value")
    var_values = {}
    for i in itemlist:
        n = i.attrib["name"]
        v = i.text
        var_values[n] = v
         
    # Get parameters
    itemlist = doc.findall("./values/parameters/value")
    par_values = {}
    for i in itemlist:
        n = i.attrib["name"]
        v = i.text
        par_values[n] = v
        
    itemlist = doc.findall("./values/trends/value")
    par_trends = {}
    for i in itemlist:
        n = i.attrib["name"]
        v = i.text
        par_trends[n] = v
                   
    itemlist = doc.findall("./autoexogenise/pair")
    shick_ref = {}
    for i in itemlist:
        n = i.attrib["endo"]
        v = i.attrib["shock"]
        shick_ref[n] = v
    
    freq = doc.attrib["freq"]
        
    if bFillValues:   
        if freq == "a":
            txtFreq = "Annually"
        elif freq == "q":
            txtFreq = "Quarterly"
        elif freq == "m":
            txtFreq = "Monthly"
        elif freq == "w":
            txtFreq = "Weekly"
        elif freq == "d":
            txtFreq = "Daily"
            
        for i,eq in enumerate(eqs):
            if i < len(eq_labels):
                lbl = eq_labels[i]
            txtEqs = txtEqs + lbl + " : " + eqs[i] 
            if i < len(eq_labels)-1:
                txtEqs += "\n"
            
        for k,v in var_values.items():
            if k in endog:
                txtEndogVars.append(k + " = " + v)
            elif k in exog:
                txtShocks = txtShocks + k + " = " + v + "\n" 
               
        for k,v in par_values.items():
            txtParams = txtParams + k + " = " + v + "\n"
      
        txtEndogVars = set(txtEndogVars)
        txtEndogVars = sorted(txtEndogVars,key=str.lower)
        txtEndogVars = "\n".join(txtEndogVars)
        
        if not "Date" in txtShocks:
            today = date.today()
            txtShocks = f"Date : {today.month}/{today.day}/{today.year} \n\n" + txtShocks
            
        txtRange = "01/01/2025 - 01/01/2040"
        
        return txtEqs,txtParams,txtParamsRange,txtEndogVars,txtExogVars,txtShocks,txtRange,txtFreq,txtModelName
        
    else:
        if freq == "a":
            frequency = 0
        elif freq == "q":
            frequency = 1
        elif freq == "m":
            frequency = 2
        elif freq == "w":
            frequency = 3
        elif freq == "d":
            frequency = 4
        
        for k, v in par_values.items():
            par_values[k] = float(v)
            
        params = list(par_values.keys())
        params_values = np.array(list(par_values.values()))
        
        variables = []; variables_values = []
        for k in var_values:
            if not k in exog:
               variables.append(k) 
               variables_values.append(var_values[k])
        
        calibration = {**par_values,**var_values}
        
        for i,eq in enumerate(eqs):
            eqs[i] = "0 = " + eqs[i].replace("{","(").replace("}",")").replace(" ","")
            
        return eqs,variables,variables_values,exog,params,params_values,calibration,eq_labels,frequency
   
if __name__ == "__main__":
    path = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(path, '../../../samples/models/morocco_model.xml')
    
    txtEqs,txtParams,txtParamsRange,txtEndogVars,txtExogVars,txtShocks,txtRange,txtFreq,txtModelName = readXmlModelFile(file_path)
