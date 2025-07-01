import sys
import os 
import warnings
import re
import glob
from datetime import date
from snowdrop.src.misc.termcolor import cprint
from snowdrop.src.utils.util import getNamesValues

known_functions = ['exp','sqrt','SQRT','log','LOG','MAX','Max','MIN','abs','Heaviside','IfThenElse','IfThen']
   
path = os.path.dirname(os.path.abspath(__file__))
working_dir = os.path.abspath(os.path.join(path,"../../.."))

def sortOrder(elem):
    ind = elem.find("=")
    if ind > 0:
        el = elem[:ind]
    else:
        el = elem
        
    return el
    

def findVars(var,vals,excl={}):
    """
    Search a variable in a list of known values and assigns value if found.
    """
    m = {}
    for val in vals:
        if "=" in val:
            ind = val.index("=")
            e = val[:ind].strip()
            if e in var:
                m[e] = val.strip()

    for e in excl:
        if e in m:
            m.pop(e)
              
    names = [*m]
    values = list(m.values())
    
    return values,names,m
            

def funcToLowerCase(eq):
    """
    Convert function name to lower case.
    """
    delimiters = "+","-","*","/","**","^","(", ")","=",":"
    regexPattern = '|'.join(map(re.escape, delimiters))
    arr = re.split(regexPattern,eq)
    arr = list(filter(None,arr))
    for e in arr:
        ind = e.find("'F")
        if ind > 0:
            e1 = e[:ind]
            if e1.strip() not in ['Max','Heaviside']:
                e2 = e1.lower()
                eq = eq.replace(e,e2)
            
    return eq        
    

def handleDiffOperator(n,expr,excl=None):
    """Handles Troll's difference operator."""
    new_expr = ""
    delimiters = "+","-","*","/","**","^", "(",")","="
    delimiters += tuple(known_functions)
    regexPattern = '|'.join(map(re.escape, delimiters))
    arr = re.split(regexPattern,expr)
    arr = list(filter(None,arr))
    ind = 0; ind1 = 0
    for e in arr:
        ind = expr.find(e,ind1)
        new_expr +=  expr[ind1:ind]
        # Check if an element is a number
        ind1 = ind+len(e)
        if not e.isdigit():
            ind2 = e.find("'F")
            if ind2 > 0:
                new_expr += e
            else:
                new_expr += e + "(" + str(-n) + ")"
        else:
            nc = int(e) 
            if new_expr[-1] == "-":
                nc = -nc
                new_expr = new_expr[:-5] + str(nc-n)
            elif new_expr[-1] == "+":
                new_expr = new_expr[:-5] + str(nc-n)
            else:
                new_expr += str(nc-n)
            
    new_expr += expr[ind1:]
    
    if not excl is None:
        b = True
        for e in excl:
            if e in expr:
                b = False
                break
        if b:
            new_expr = "(" + expr + "-(" + new_expr + "))"
        else:
            new_expr = expr
    else:
        new_expr = "(" + expr + "-(" + new_expr + "))"
    
    return new_expr


def handleProductOrSumOperator(expr,operator,excl=None):
    """ Handles Troll's product and sum operators."""
    new_expr = ""
    
    delimiters = " ", "=", "TO", ":"
    regexPattern = '|'.join(map(re.escape, delimiters))
    # Extract digits from expression
    arr = re.split(regexPattern,expr)
    digits = []
    for d in arr:
        match = re.findall(r"[-+]?\d+$",d)
        match = set(match)
        for m in match:
            digits.append(int(m))
            
    if len(digits) < 2:
        warnings.warn("\nThe left or the right boundaries for index i of the expression: " + expr + " are not set.")
        return expr
    
    i1 = digits[0]
    i2 = digits[1]
    ind = expr.find(":")
    ex = "(" + expr[1+ind:].replace(" ","")
    arr = []
    # We don't replace PRODUCT operator for parameters
    b = True
    if not excl is None:
        for e in excl:
            if e in ex:
                b = False
                break
    if b:
        for i in range(i1,1+i2):
            if i == 0:
                e = ex.replace("(i)","")
            else:
                e = ex.replace("(i)","(" + str(i) + ")")
            arr.append(e)
    else:
        e = ex.replace("(i)","")
        arr.append(e)
        
    if operator == "PRODUCT":
        new_expr = " * ".join(arr) 
    elif operator== "SUM":
        new_expr = " + ".join(arr)
    else:
        warnings.warn("Operator " + operator + " is not suppoted")
    
    return "(" + new_expr + ")"
              

def emulateModelLanguageOperators(equation,operator,excl=None):
    """
    Emulate Troll modelling language special operators like DEL([n],expr) for n-period difference
    """
    equation = equation.replace("'N","").replace("'n","").replace("'X","").replace("'x","").replace("'P","").replace("'p","")
    ind = equation.find(operator + "(")
    if ind == -1:
        # Nothing to do
        return equation  
    
    new_eq = equation[:ind]
    rest_eq = equation
    while ind >=0:
        rest_eq = rest_eq[ind+len(operator):]
        arr1 = []; arr2 = []
        for m in re.finditer(r"\(", rest_eq):
            arr1.append(m.start())
        for m in re.finditer(r"\)", rest_eq):
            arr2.append(m.start())
        arr = sorted(arr1 + arr2)
        s = 0
        count = 0
        for ind in arr:
            if ind in arr1:
                s += 1
                count += 1
            elif ind in arr2:
                s -= 1
                count += 1
            if s == 0 and count > 0:
                break
        expr = rest_eq[:1+ind]
        l = len(expr)
        if operator == "DEL":
            # Find difference n-period
            expr = expr.replace(" ","")
            ind = expr.find(":")
            if ind == -1:
                n = 1
            else:
                ex = expr[1:ind]
                if re.match(r"\d+$", ex):
                    n = int(ex)
                    expr = "(" + expr[1+ind:]
            expr = expr.replace(" ","")
            new_expr = handleDiffOperator(n,expr)
        if operator in ["PRODUCT","SUM"]:
            expr = expr.strip()
            new_expr = handleProductOrSumOperator(expr,operator,excl=excl)
        new_eq +=  new_expr 
        ind = rest_eq.find(operator + "(")
        if ind >= 0:
            new_eq += rest_eq[l:ind]
        else:
            new_eq += rest_eq[l:]    
        rest_eq = rest_eq[l:]
        ind = rest_eq.find(operator + "(")
        
    return new_eq	
	

def readTrollModelFile(file_path=None,bFillValues=True,debug=False):
    """
    Parse TROLL model file.
    """
    today = date.today()
    delimiters = " ",",",";",":","*","/","+","dzero"
    regexPattern = '|'.join(map(re.escape, delimiters))
    
    file_path = os.path.abspath(file_path)
    if "/" in file_path:
        arr = file_path.split("/")
    elif "\\" in file_path:
        arr = file_path.split("\\")
    fdir = os.path.abspath(working_dir+"/data/Troll/"+arr[-2])
    # fdb = os.path.abspath(fdir+"/start.mat") 
    m = {}; b = False
    
    # if os.path.exists(fdb):
    #     cprint(f"Reading parameters from a database file: \n {fdb}","blue")
    #     m = readDatabase(fdb,year=today.year)
 
    fl = os.path.abspath(os.path.join(fdir,"env.csv"))
    if os.path.exists(fl):
        cprint(f"Reading parameters from a steady state file: \n {fl}\n","blue")
        with open(fl,"r") as f:
            lines = f.readlines()
            for line in lines:
                arr = line.split(",")
                arr = list(filter(None,arr))
                if len(arr) > 1:
                    key = arr[0].strip(); val = arr[1].replace("\n","").strip()
                    try:
                        m[key] = float(val)
                    except:
                        if val in m:
                            m[key] = m[val]
    
    files = [os.path.abspath(f) for f in glob.glob(fdir+"/*startvals.src")]
    cprint(f"Reading parameters from a source file: \n {', '.join(files)}\n","blue")
    for fl in files:
        with open(fl,"r") as f:
            lines = f.readlines()
            for line in lines:
                b = (fl.endswith(".src") and "do " in line and "=" in line) \
                    or (fl.endswith(".txt") and "=" in line)
                if b:
                    arr = re.split(regexPattern,line)
                    arr = list(filter(None,arr))
                    ind = arr.index("=")
                    if ind > 0 and len(arr) > ind+1:
                        key = arr[ind-1]
                        if "_&" in key:
                            key = key[:key.index("_&")]
                        if '""' in key:
                            key = key[2+key.index('""'):]
                        val = arr[ind+1].replace("\n","").replace("'","").replace('"','').strip()
                        try:
                            m[key] = float(val)
                        except:
                            if val in m:
                                m[key] = m[val]
                      
    strEndogenous="Endogenous";strExogenous="Exogenous";strParams="Parameters"
    strEqs="Equations";strResiduals="Residuals";strShocks="Shocks"
    strConstants="Constants";strErrorTerms="Error Terms";strCalibration="Calibration"
    
    txt=[];txtEndog=[];txtExog=[];txtEqs='';txtEquations=[];txtParams=[];
    varsLabels=[];txtErrorTerms=[];comments=[];eqsLabels=[];nameEndog=[]
    txtConstants=[];txtShocks=[];txtResiduals=[];txtCalibration=[];labels={}
    txtParamsRange="";txtRange="";txtFreq="";description=""
    header = None
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            ln1 = re.sub(':','',line).strip()
            ln2 = re.sub('TROLL Command:','',line).strip()
            # Replace dzero
            ln2 = ln2.replace("dzero","0").strip()
            if "name:" in line:
                ind = ln1.find("name:")
                description = ln1[ind+5:].strip()
            if ln1 in [strEndogenous,strExogenous,strParams,strEqs,strResiduals,strShocks,strConstants,strErrorTerms,strCalibration]:
                header = ln1
                txt = []   
            elif len(txt) > 0 and not ln2:
                if header == strEqs:
                    txtEquations = txt
                elif header == strEndogenous:
                    txtEndog = txt
                elif header == strExogenous:
                    txtExog = txt
                elif header == strParams:
                    txtParams = txt
                elif header == strResiduals:
                    txtResiduals = txt
                elif header == strShocks:
                    txtShocks = txt
                elif header == strErrorTerms:
                    txtErrorTerms = txt
                elif header == strConstants:
                    txtConstants = txt
                elif header == strCalibration:
                    txtCalibration = txt
                header = None
                txt = []  
            elif len(ln2) > 0 and not "//" in ln2:
                txt.append(ln2)
     
    param_names = []
    for p in txtParams:
        ind = p.find("=")
        if ind>=0:
            param_names.append(p[:ind].strip())
            
    shock_names=[]       
    for s in txtShocks:
        ind = s.find("=")
        if ind>=0:
            shock_names.append(s[:ind].strip())
            
    bHasEqNumbers = False
    arr=[];txt="";lb=""
    for eq in txtEquations:
        ind = eq.find(" ")
        n = eq[:ind]
        ind = eq.find(":")
        if  re.match(r"\d+$", n) and ind > 0: # check for the beginning of equation line   
            bHasEqNumbers = True
            if len(txt) > 0:
                arr.append(txt)
            lb = eq[:ind]
            e = eq[1+ind:]
            ind = lb.find(" ")
            if ind > 0:
                lb = lb[1+ind:].strip()
            varsLabels.append(lb)
            txt = eq.strip()
            key = lb
            b = txt.endswith("DEL") or txt.endswith("DEL(")
        else:
            ind = eq.find(":")
            if ind >= 0 and not b and not "DEL(" in eq and not "TO " in eq:
                key = eq[:ind].strip()
                if not "(" in key and not ")" in key:
                    val = eq[1+ind:].strip()
                    comments.append(val)
                    labels[key] = val
            if key == lb:
                txt += eq.strip()
            b = txt.endswith("DEL") or txt.endswith("DEL(")
            
    # Add the last one        
    arr.append(txt)  
    
    # Handle If-Then-Else operator
    for i,eq in enumerate(arr):
        e = eq.lower()
        if "if " in e and " else " in e and " then " in e:
            arr[i] = handleIfThenElseOperator(eq,m)
            
    # Handle PNORM operator
    for i,eq in enumerate(arr):
        e = eq.lower().replace(" ","")
        if "pnorm(" in e :
            arr[i] = handlePNORMfunction(eq)
            
    if bHasEqNumbers:
        txtEqs = arr.copy()
    else:
        txtEqs = txtEquations
    
    # Check for TROLL notation of endogeneous variables and parameters 
    # These are: 'p - for parameter, 'n - for endogenous variable, 'x - for exogenous variable                        
    endog = []; exog = []; par = []
    delimiters = " ",",",";","*","/","+","-",":","(",")","^","="
    regexPattern = '|'.join(map(re.escape, delimiters))
    
    for i,eq in enumerate(txtEqs):
        # Break equations into elements
        arr = set(re.split(regexPattern,eq))
        for e in arr:
            el = e.strip()
            if "'P" in el or "'p" in el:
                par.append(el.replace("'P","").replace("'p",""))
            elif "'N" in el or "'n" in el:
                endog.append(el.replace("'N","").replace("'n",""))
            elif "'X" in el or "'x" in el:
                exog.append(el.replace("'X","").replace("'x",""))
        txtEqs[i] = eq.replace("'P","").replace("'p","").replace("'N","").replace("'n","").replace("'X","").replace("'x","")
    
    # Break calibration values into elements
    arr = set(re.split(regexPattern," ".join(txtCalibration)))
    for e in arr:
        el = e.strip()  
        el2 = el.replace("-","").replace(".","")
        if not el2.isdigit():   
            if not el in endog and not el in exog and not el in par:
                exog.append(el)    
        
    endog = list(set(endog))
    endog = list(filter(None,endog))
    exog  = list(set(exog))
    exog  = list(filter(None,exog))
    par   = list(set(par))
    par   = list(filter(None,par))
    
    # Handle Troll operators
    new_eqs = []
    count = 0  
    for eq in txtEqs:
        count += 1
        new_eq = funcToLowerCase(eq)
        if "PRODUCT" in new_eq:
            new_eq = emulateModelLanguageOperators(equation=new_eq,operator="PRODUCT",excl=param_names)
        if "SUM" in new_eq:
            new_eq = emulateModelLanguageOperators(equation=new_eq,operator="SUM",excl=param_names)
        if "DEL" in new_eq:
            new_eq = emulateModelLanguageOperators(equation=new_eq,operator="DEL")
        new_eq = new_eq.replace("'F","").replace("'f","") # function name notations
        ind = new_eq.find("=")
        left = new_eq[:ind]
        right = new_eq[1+ind:]       
        new_eq = left.strip() + " = " + right.replace(" ","")
        # Fix equation so that we don't have leads and lags for exogenous variables
        match = re.findall(r"\((\s?[+-]?\d+\s?)\)",new_eq)
        match = set(match)
        arr = re.split(regexPattern,new_eq)
        for e in arr:
            if e in exog and not e in endog:
                if e + "(" in new_eq:
                    for m in match:
                        old_exog = e + "(" + m + ")"
                        new_eq = new_eq.replace(old_exog,e)
        new_eqs.append(new_eq)
        
    replace_func = {"EXP":"exp","LOG":"log","MAX":"Max"}
    # Replace EXP() function with exp()
    for f in replace_func:
        for i,eq in enumerate(new_eqs):
            ind = eq.index(f+"(") if f+"(" in eq else -1
            while not ind==-1 and ind<len(eq) :
                e = eq[ind-1]
                if e==" " or e=="+" or e=="-" or e=="*" or e=="/" or e=="(":
                    eq = eq[:ind] + replace_func[f] + "(" + eq[ind+4:]
                if not f+"(" in eq[ind+1:]:
                    break
                ind = eq.index(f+"(",ind+1)
            new_eqs[i] = eq
        
    txtEqs = new_eqs
           
    if len(txtEndog) == 0:
        for eq in txtEqs:
            if ":" in eq:
                ind = eq.index(":")
                txt = eq[:ind]
                lst = list(filter(None,txt.split(" ")))
                if len(lst) == 1:
                    var = lst[0].strip()
                elif len(lst) >= 1:
                    var = lst[1].strip()
                else:
                    var = None
                if not var is None:
                    if var.lower().startswith("ss_"):
                        var = var[3:]
                    if not var in endog:
                        endog.append(var)
         
        lst = []    
        for v in endog:
            if v in m:
                txtEndog.append(f"{v} = {m[v]}")
            else:
                txtEndog.append(f"{v} = 1")
                lst.append(v)
        if len(lst) > 0:
            cprint(f"Setting starting values of endogenous variables to default value 1 : \n{lst}\n","red")
                
        mp = {}; nameEndog = endog.copy()
        
    else:
        # Assign initial values
        txtEndog,nameEndog,mp = findVars(var=endog,vals=set(txtEndog+txtExog+txtErrorTerms+txtCalibration)) 
      
    #txtParams += txtResiduals
        
    delimiters = " ","+","-","*","/","**","^","(",")","=",":",">","<",",","1e","1.e"
    delimiters += tuple(known_functions)
    regexPattern = '|'.join(map(re.escape, delimiters))
    allSymbols = []
    for eq in new_eqs:
        if ":" in eq:
            ind = eq.index(":")
            eq = eq[1+ind:]
        arr = re.split(regexPattern,eq)
        arr = list(filter(None,arr))
        for e in arr:
            if not e in ['DEL','PRODUCT','SUM']:
                try:
                    float(e)
                except:
                    if not e in allSymbols and not e in ["Positive","Negative"]:
                        allSymbols.append(e)
        
    # Add missing shocks
    missingShockNames = []; missingShockvalues = []
    for eq in new_eqs:
        eq = eq.replace(" ","")
        ind = -1
        arr = re.split(regexPattern,eq)
        arr = list(filter(None,arr))
        for v in arr:
            if v.startswith("E_") and not v in nameEndog:
                missingShockNames.append(v)
                missingShockvalues.append(v + " = 0")
                
    missingShockNames = list(set(missingShockNames))
    missingShockvalues = list(set(missingShockvalues))  
    shock_names+= missingShockNames   
    shock_names = list(set(shock_names)) 
    txtShocks += missingShockvalues
    txtShocks = list(set(txtShocks))
    
    allEqs = " ".join(txtEqs)
    
    # Assign initial values
    txtExog,nameExog,_ = findVars(var=exog,vals=set(txtExog+txtParams+txtConstants+txtErrorTerms+txtResiduals+txtCalibration),excl=(nameEndog+param_names+shock_names)) 
    if len(par) > 0:
        txtParams,nameParameters,_ = findVars(var=par,vals=txtParams+txtCalibration,excl=(nameEndog+nameExog+shock_names)) 
    else:  
        # Break equations into elements
        nameParameters = []; undefinedParameters = []
        arr = re.split(regexPattern,allEqs)
        arr = set(filter(None,arr))
        for e in arr:
            try:
                float(e)
            except:
                if not e in nameEndog and not e in shock_names and not e in exog and not e in ["Positive","Negative"]:
                    nameParameters.append(e)
                    if e in m and e in allSymbols:
                        txtParams.append(f"{e} = {m[e]}")
                    elif e in allSymbols:   
                        undefinedParameters.append(e) 
           
        undefinedParameters = sorted(undefinedParameters)
            
        if undefinedParameters:
            if len(undefinedParameters) > 0:
                cprint(f"{len(undefinedParameters)} parameters were not defined!\nSetting missing parameters values to 0.5","red")
            
            for e in undefinedParameters:
                txtParams.append(f"{e} = 0.5")
            
            cprint(f"{', '.join(undefinedParameters[:20])}","red")
            if len(undefinedParameters) > 20:
                cprint("...","red")
                 
        # Remove leads and lags brackets from parameters
        for i,eq in enumerate(txtEqs):
            for par in undefinedParameters:
                if par+"(" in eq:
                    ind1 = eq.index(par+"(")
                    rest = eq[ind1+len(par):]
                    if "(" in rest:
                        ind2 = findRightBracket(rest)
                        eq = eq[:ind1] + par + rest[ind2:]           
                    else:
                        cprint(f"Error removing parameters' leads and lags brackets in {eq}","r")
                        sys.exit(-1)
            txtEqs[i] = eq
         
    param_names = [x[:x.index("=")].strip() for x in txtParams]
    # Fix equations in case there are paramters with leads and lags
    delimiters = " ","+","-","*","/","**","^","=",":",">","<"
    delimiters += tuple(known_functions)
    regexPattern = '|'.join(map(re.escape, delimiters))
    np = 0
    for i,eq in enumerate(txtEqs):
        arr = re.split(regexPattern,eq)
        arr = filter(None,arr)
        for e in arr:
            if "(" in e:
                while e.startswith("("):
                    e = e[1:]
                if "(" in e:
                    ind = e.index("(")
                    p = e[:ind]
                    if p in param_names:
                        if p+"(" in eq:
                            ind1 = eq.index(p+"(")
                            ind2 = eq.index(")",ind1)
                            par  = eq[ind1:1+ind2]
                            eq = eq.replace(par,p)
                            np += 1
                            if debug:
                                if np==1:
                                    cprint("Removing time index from parameters:","blue")
                                print(f"  {par} --> {p}")
        if ":" in eq:
            ind = eq.index(":")
            left = eq[:ind]
            arr = left.split(" ")
            label = arr[-1]
            eqsLabels.append(label)
        txtEqs[i] = eq
    
    if np > 0: 
        cprint(f"Removed time index from {np} parameters\n","blue")
    
    if 0 < len(eqsLabels) < len(txtEqs):
        cprint(f"\nNumber of labels {len(eqsLabels)} is lesser than the number of equations {len(txtEqs)}","red")
    
    undefined = list(set(allSymbols) - set(nameParameters) - set(nameEndog) - set(exog) - set(shock_names))
    if bool(undefined):
        cprint(f"\n\n{len(undefined)} undefined symbols:","red")
        cprint(f"{', '.join(undefined[:20])}","red")
        if len(undefined) > 20:
            cprint("...","red")
        
        
    endog_names,endog_values = getNamesValues(txtEndog)
    param_names,param_values = getNamesValues(txtParams) 
        
    eqs = "\n".join(txtEqs)
    for v in endog_names:
        if v in eqs:
            eqs = eqs.replace("(+"+v+")(","+"+v+"(").replace("(-"+v+")","-"+v+"(")
        
    if bFillValues:
        
        # Sort variables by name
        txtEndog = sorted(txtEndog,key=sortOrder)
        txtExog = sorted(txtExog,key=sortOrder)
        txtParams = sorted(txtParams,key=sortOrder)
        
        # Get list of  non-assigned endogenous variables
        missingEndogNames = set(endog).difference(set(nameEndog))
        if len(missingEndogNames) > 0 :
                warnings.warn("Missing Endogenous Varibales:" + str(missingEndogNames))
            
        # Sort shocks by name
        txtShocks = sorted(txtShocks,key=sortOrder)
                    
        if not "Date" in txtShocks:
            txtShocks.insert(0,f"Date : {today.month}/{today.day}/{today.year} \n") 
            
        # Convert list to strings
        #eqs = "\n".join(txtEqs)
        endog = "\n".join(txtEndog)
        endog = endog.replace("0+","")
        exog = "\n".join(txtExog)
        exog = exog.replace("0+","")
        shocks = "\n".join(txtShocks)
        shocks = shocks.replace("0+","")
        params = "\n".join(txtParams)
        params = params.replace("0+","")
               
        if not txtRange:
            txtRange = f"01/01/{today.year} - 01/01/{101+today.year}"
            
        return eqs,params,txtParamsRange,endog,exog,shocks,txtRange,txtFreq,varsLabels,eqsLabels,description,comments,undefinedParameters
           
    else:
        
        for k in mp:
            val = mp[k]
            ind = val.index("=")
            try:
                m[k] = float(val[1+ind:])
            except:
                pass
        eqs = []
        for eq in txtEqs:
            ind = eq.index(":")
            k = eq[1+ind:].replace(" ","")
            eqs.append(k)
        shocks = []
        for sh in txtShocks:
            ind = sh.index("=")
            k = sh[:ind].strip()
            shocks.append(k)
        params = []
        for p in txtParams:
            ind = p.index("=")
            k = p[:ind].strip()
            v = p[1+ind:].strip()
            if k in allEqs:
                params.append(k)
            try:
                m[k] = float(v)
            except:
                pass
        exog = []
        for e in txtExog:
            ind = e.index("=")
            k = e[:ind].strip()
            if not k in params:
                params.append(k)
            exog.append(k)
            try:
                v = e[1+ind:].strip()
                m[k] = float(v)
            except:
                pass
            
            temp = []

        temp = []
        for v in txtExog:
            ind = v.find("=")
            if ind >= 0:
                temp.append(v[:ind].strip())
                 
        #params = param_names+temp
        
        return eqs,endog_names,endog_values,nameExog,param_names,param_values,shocks,labels,eqsLabels,comments,m,undefinedParameters
                
    
def handleElement(name,desc,e,arr):
    """
    Creates a link to an element and a reference to this element
    """
    ne = "<a href = '" + name + "'//><br//><br//>"
    link = "<h3 href = '" + name + "'//><br//><br//>"
    if e in desc.keys():
        link += desc[e] + "<br//>"
    ind = arr.find(e)
    if ind >= 0:
        v = arr[ind]
        link += v + "<br//>"
    
    return ne,link


def createHTMLDocument(fout,fdict,description,varsLabels,eqs,params,endog,exog,shocks):
    """
    Creates HTML document describing equations, variables, and parameters
    """
    path = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(path,'../../docs/',fout)
    dict_path = os.path.join(path,'../../data/dictionary',fdict)
        
    # Read dictionary file
    desc = {}
    with open(dict_path, 'r') as f:
        for line in f:
            arr = line.split(",")
            if len(arr) >= 2:
                desc[arr[0]] = arr[1]
        
    delimiters = "+", "-", "*", "/", "**", "^",  "(", ")", "="
    regexPattern = '|'.join(map(re.escape, delimiters))
    
    endog_names,endog_values = getNamesValues(endog)
    exog_names,exog_values = getNamesValues(exog)
    param_names,param_values = getNamesValues(params)
    shock_names,shock_values = getNamesValues(shocks)
    
    links = []
    
    with open(out_path, 'w') as f:
        for eq in eqs:
            arr = re.split(regexPattern,eq)
            arr = list(filter(None,arr))
            new_eq = eq; ne = ""; link = ""
            for e in arr:
                if e in endog_names:
                    ne,link = handleElement("Endogenous Variable",desc,e,endog)
                elif e in exog_names:  
                    ne,link = handleElement("Exogenous Variable",desc,e,exog)
                elif e in param_names:  
                    ne,link = handleElement("Parameter",desc,e,params)
                elif e in shock_names: 
                    ne,link = handleElement("Shocks",desc,e,shocks)
                new_eq = new_eq.replace(e,ne)
                links.append(link)
            f.write(new_eq + "\n")
            
        for link in links:
            f.write(link + "\n")
                            
            
def findRightBracket(expr):
    """Find position of right bracket."""
    ind = len(expr)
    if "(" in expr:
        left_brackets = [x.start() for x in re.finditer(r"\(",expr)]
        right_brackets = [x.start() for x in re.finditer(r"\)",expr)]
        nl = len(left_brackets)
        nr = len(right_brackets)
        if nl <= nr:
            ml = dict(zip(left_brackets,[1]*nl))
            mr = dict(zip(right_brackets,[-1]*nr))
            mp = {**ml,**mr}
            brackets = sorted(left_brackets+right_brackets)
            s = 0
            for br in brackets:
                s += mp[br]
                if s == 0:
                    ind = br+1
                    break
    else:
        cprint(f"Unbalanced parenthesis in expression {expr}","r")
        sys.exit(-1)
        
    return ind


def handlePNORMfunction(expr):
    """Handle Troll's PNORM function."""
    e = expr.lower()
    ind1 = e.index("pnorm")
    rest_expr = expr[ind1+5:]
    
    if "(" in rest_expr:
        ind2 = findRightBracket(rest_expr)
        arg = rest_expr[:ind2]
        ex = expr[:ind1] + " exp(-0.5*" + arg + "**2) " + rest_expr[ind2:]           
    else:
        cprint(f"Error handling PNORM function in {expr}","r")
        sys.exit(-1)
    
    return ex
    

def handleIfThenElseOperator(expr,m):
    """Handle Troll's If-Then-Else operator."""
    
    e = expr.lower()
    ind1 = e.index("if ")
    ind2 = e.index(" then ")
    ind3 = e.index(" else ")
    ind4 = ind3 + 6
    conditional_expr = expr[ind1+3:ind2].replace("ABSV","abs")
    than_expr = expr[ind2+6:ind3]
    rest_expr = expr[ind3+6:]
    ind5 = findRightBracket(rest_expr)
    else_expr = rest_expr[:ind5]
    
    # user_defined_functions = {"ABSV":abs}
    # env = {**m,**user_defined_functions}
    # b = eval(conditional_expr,env,m)
    # ex = than_expr if b else else_expr
    
    if ">" in conditional_expr:
        conditional_expr = "Positive(" + conditional_expr.replace(">","-") + ")"
    elif "<" in conditional_expr:
        conditional_expr = "Negative(" + conditional_expr.replace("<","-") + ")"

    if "(" in rest_expr:
        ex = expr[:ind1] + " IfThenElse(" + conditional_expr + ", " + than_expr + ", " + else_expr + ") " + rest_expr[1+ind4:]           
    else:
        ex = expr[:ind1] + " IfThenElse(" + conditional_expr + ", " + than_expr + ", " + rest_expr + ") "

    return ex


def readDatabase(fpath,year=2022,series_names=None):
    """Load Matlab file."""
    import numpy as np
    import pandas as pd
    from mat4py import loadmat
    
    db = loadmat(fpath)['db']
    m = {}; i = 0; err = 0
    for k in db:
        if series_names is None or k in series_names:
            try:
                arr = np.array(db[k]['data'])
                m[k] = pd.Series(arr[:,2],arr[:,0])[year]
                i += 1
            except:
                err += 1
                pass
        else:
            pass
            #cprint(f"Missing value of {k}","red")
    del db
    return m
    

def summary(all_sizes):
    import numpy as np
    sizes = sorted(set(all_sizes))
    sizes = np.array(sizes)   
    for x in sizes:
        n = sum(all_sizes==x)
        if n == 1:
            print(f"{n} component of size {x}")
        else:
            print(f"{n} components of size {x}")
  
        
def getTrollModel(fpath,Solver=None,calibration={},options={},use_cache=False,check=True,return_interface=False,debug=False):
    """
    Reads Troll model file and instantiates this model.

    Args:
        fpath : str.
            Path to Troll model file.
        Solver:
            Name of solver.
        calibration : dict.
            Map with values of calibrated parameters and starting values of endogenous variables.
        options : dict, optional
            Dictionary of options. The default is empty dictionary.
        use_cache : bool, optional
            If True reads previously saved model from a file of model dump.
        return_interface : bool, optional
            If True returns `Interface' object.
        debug : bool, optional
            If set to True prints information on Iris model file sections. The default is False.

        Returns:
            Model object.
    
    """
    import numpy as np
    from snowdrop.src.preprocessor.util import updateFiles
    from snowdrop.src.model.factory import getModel
    
    fname, ext = os.path.splitext(fpath)
    model_path = fpath.replace(fname+ext,fname+".bin")
    model_file_exist = os.path.exists(model_path)
    
    if use_cache and model_file_exist:
        
        from snowdrop.src.utils.interface import loadModel
        from snowdrop.src.model.settings import SolverMethod
        from snowdrop.src.preprocessor.util import updateFiles
    
        model = loadModel(model_path)
        updateFiles(model,path+"/../preprocessor")
        
        # Update model variables and parameters values
        variables = model.symbols["variables"]
        parameters = model.symbols["parameters"]
        mv = np.copy(model.calibration['variables'])
        mp = np.copy(model.calibration['parameters'])
        
        for i,k in enumerate(variables):
            if k in calibration:
                mv[i] = calibration[k]
            # Set un-assigned values
            if np.isnan(mv[i]):
                if "_minus_" in variables[i]:
                    ind = variables[i].index("_minus_")
                    v = variables[i][:ind]
                    j = variables.index(v)
                    mv[i] = mv[j]
                elif "_plus_" in variables[i]:
                    ind = variables[i].index("_plus_")
                    v = variables[i][:ind]
                    j = variables.index(v)
                    mv[i] = mv[j]
                else:
                    cprint(f"Setting missing value of variable {variables[i]} to 1.","red")
                    mv[i] = 1
                    
        for i,k in enumerate(parameters):
            if k in calibration:
                mp[i] = calibration[k]
            # Set un-assigned values
            elif np.isnan(mp[i]):
                cprint(f"Setting missing value of parameter {parameters[i]} to 1.","red")
                mp[i] = 1
        
        model.calibration['variables'] = mv
        model.calibration['parameters'] = mp
        
        if not Solver is None:
            if Solver.upper() == "SIMS":
                model.SOLVER = SolverMethod.Sims
                model.isLinear = True
            elif Solver.upper() == "VILLEMOT":
                model.SOLVER = SolverMethod.Villemot
                model.isLinear = True
            elif Solver.upper() == "KLEIN":
                model.SOLVER = SolverMethod.Klein
                model.isLinear = True
            elif Solver.upper() == "ANDERSONMOORE":
                model.SOLVER = SolverMethod.AndersonMoore
                model.isLinear = True
            elif Solver.upper() == "BINDERPESARAN":
                model.SOLVER = SolverMethod.BinderPesaran
                model.isLinear = True
            elif Solver.upper() == "BENES":
                model.SOLVER = SolverMethod.Benes
                model.isLinear = True
            elif Solver.upper() == "LBJ":
                model.SOLVER = SolverMethod.LBJ
                model.isLinear = False
            elif Solver.upper() == "LBJAX":
                model.SOLVER = SolverMethod.LBJax
                model.isLinear = False
                model.jaxdiff = True
            elif Solver.upper() == "ABLR":
                model.SOLVER = SolverMethod.ABLR
                model.isLinear = False
            elif Solver.upper() == "FAIRTAYLOR":
                model.SOLVER = SolverMethod.FairTaylor
                model.isLinear = False
            elif Solver.upper() == "BLOCK":
                model.SOLVER = SolverMethod.Block
                model.isLinear = False
            else:
                model.SOLVER = SolverMethod.LBJ
                model.isLinear = False
           
        # Update python generated files   
        updateFiles(model,path+"/../preprocessor")
    
    else:
        
        name = os.path.basename(fpath)
        infos = {'name': name,'filename' : fpath}   
        
        eqs,variables,var_values,exogVars,parameters,par_values,shocks,var_labels,eqsLabels,comments,m,_  \
            = readTrollModelFile(fpath,bFillValues=False,debug=debug)
        
        if debug:  
            equations = '\n'.join(eqs)
            print(f"\nParameters:\n{parameters}")
            print(f"\nTransition Shocks:\n{shocks}")
            print(f"\nTransition variables:\n{variables}")
            print(f"\nTransition equations:\n{equations}\n")
            print(f"\n\nLabels of variables:\n{var_labels}")
           
        calibration["variables"]  = var_values
        calibration["parameters"] = par_values
                
        model = getModel(name=name,Solver=Solver,eqs=eqs,variables=variables,parameters=parameters,
                         shocks=shocks,calibration=calibration,var_labels=var_labels,
                         eqs_labels=eqsLabels,check=check,return_interface=return_interface,
                         options=options,infos=infos)
        if return_interface:
            return model
            
        model.symbolic.labels = eqsLabels
        model.symbols["variables_labels"] = var_labels
        model.symbols["equations_comments"] = comments
        variables = model.symbols['variables']
        var_values = model.calibration['variables']
        mv = np.zeros(len(variables))
        mv[:len(var_values)] = var_values
        mp = np.copy(model.calibration['parameters'])
                
        for i,k in enumerate(variables):
            if k in m and np.isnan(mv[i]):
                mv[i] = m[k]
                  
        for i,k in enumerate(variables):
            if k in calibration:
                mv[i] = calibration[k]
            # Set un-assigned values
            elif np.isnan(mv[i]):
                if "_minus_" in variables[i]:
                    ind = variables[i].index("_minus_")
                    v = variables[i][:ind]
                    j = variables.index(v)
                    mv[i] = mv[j]
                elif "_plus_" in variables[i]:
                    ind = variables[i].index("_plus_")
                    v = variables[i][:ind]
                    j = variables.index(v)
                    mv[i] = mv[j]
                else:
                    cprint(f"Setting missing value of variable {variables[i]} to 1.","red")
                    mv[i] = 1
                    
        for i,k in enumerate(parameters):
            if k in m and np.isnan(mp[i]):
                mp[i] = m[k]
                        
        for i,k in enumerate(parameters):
            if k in calibration:
                mp[i] = calibration[k]
            # Set un-assigned values
            elif np.isnan(mp[i]):
                cprint(f"Setting missing value of parameter {parameters[i]} to 1.","red")
                mp[i] = 1
                
        model.calibration['variables'] = mv
        model.calibration['parameters'] = mp
        
        # Serialize model into file
        from snowdrop.src.utils.interface import saveModel
        saveModel(model_path,model)
    
    return model


def getEqsInfo(eqs,eqsLabels):
    """
    Display information on model equations blocks. 

    Parameters
    ----------
    eqs : TYPE
        DESCRIPTION.
    eqsLabels : TYPE
        DESCRIPTION.

    Returns
    -------
    None.
    """
    
    m = dict()
    delimiters = "+","-","*","/","**","^", "(",")","="," "
    delimiters += tuple(known_functions)
    regexPattern = '|'.join(map(re.escape, delimiters))
    eqs = txtEqs.split("\n")
    print(f"\nNumber of equations {len(eqs)} and of endogenous variables {len(variables_names)}")
    for i,eq in enumerate(eqs):
        ind   = eq.index("=")
        eqLabel = eqsLabels[i]
        if eqLabel.lower().startswith("ss_"): 
            eqLabel = eqLabel[3:]
        left  = eq[:ind].strip() if eqLabel.isdigit() else eqLabel
        right = eq[1+ind:].strip()
        arr   = re.split(regexPattern,right)
        arr   = list(filter(None,arr))
        m[left] = [x for x in arr if x in variables_names]
                
    mv = dict((v,k) for k,v in enumerate(variables_names))
    
    from snowdrop.src.info.graph import Graph
    
    g = Graph(m=mv)
    for k in m:
        for v in m[k]:
            g.addEdge(k,v)
    
    import networkx as nx
    G = nx.DiGraph()
    G.add_nodes_from(mv)
    for k in m:
        nds = m[k]
        for k2 in nds:
            G.add_edge(k,k2)
            
            
    sc = [len(c) for c in sorted(nx.strongly_connected_components(G),key=len,reverse=False)]
    print(f"\nNumber of strongly connected components {len(sc)}")
    summary(sc)
    
    cn = [len(c) for c in sorted(nx.connected_components(G.to_undirected()),key=len,reverse=False)]
    print(f"\nNumber of connected components {len(cn)}")  
    summary(cn)
    
    ac = [len(c) for c in sorted(nx.attracting_components(G),key=len,reverse=False)]
    print(f"\nNumber of attracting components {len(ac)}")
    summary(ac)
    
    
    bc = [len(c) for c in sorted(nx.biconnected_components(G.to_undirected()),key=len,reverse=False)]
    print(f"\nNumber of biconnected components {len(bc)}")
    summary(bc)
    
    wc = [len(c) for c in sorted(nx.weakly_connected_components(G),key=len,reverse=False)]
    print(f"\nNumber of weekly connected components {len(wc)}")
    
    summary(wc)


if __name__ == '__main__':
    
    path = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(working_dir,'models/Troll/FSGM3/ISRMOD.inp')
    
    name = os.path.basename(file_path)
    fout,ext = os.path.splitext(name)
    fdir = os.path.dirname(file_path)
    fpath = os.path.abspath(os.path.join(fdir,fout + '.yaml'))

    txtEqs,txtParams,txtParamsRange,txtEndog,txtExog,txtShocks,txtRange, \
    txtFreq,varsLabels,eqsLabels,description,comments,undef_params = readTrollModelFile(file_path)
    variables_names,variables_init_values = getNamesValues(txtEndog.split('\n'))
    param_names,param_values = getNamesValues(txtParams.split('\n'))     
    shock_names,shock_values = getNamesValues(txtShocks.split('\n'))   
    endog_names,endog_values = getNamesValues(txtEndog.split('\n'))

    #prefix = set([x[:x.index("_")] if "_" in x else x for x in variables_names])
    #cprint(f"\nPrefixes of variables: {list(prefix)}\n","green")
    #print(txtEndog)
    
    getEqsInfo(eqs=txtEqs.split("\n"),eqsLabels=eqsLabels)
    
    from snowdrop.src.utils.util import SaveToYaml
    arr = file_path.split("/")
    description='Model '+ arr[-3] + "/" + arr[-2] + "/" + fout + ext
    SaveToYaml(file=fpath,description=description,shock_names=shock_names,shock_values=shock_values,
            variables_names=variables_names,variables_init_values=variables_init_values,
            param_names=param_names,param_values=param_values,exog_var=txtExog,equations=txtEqs,
            eqsLabels=eqsLabels,comments=comments,time_range="[[2023,1,1],[2128,1,1]]",periods="[[2024,1,1]]")
    
