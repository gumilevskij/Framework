import os,re

path = os.path.dirname(os.path.abspath(__file__))


def readDynareModelFile(file_path,conditions={},bFillValues=True):
    """
    Reads data from DYNARE model file
    """
    delimiters = " ", ",", ";", "*", "/", ":", "+", "-", "="
    regexPattern = '|'.join(map(re.escape, delimiters))
    
    if file_path is None:
        file_path = os.path.abspath(os.path.join(path,'../models/ICD/MPAF/model.mod'))
        
    txt=[];txtEqs=[];txtParams=[];txtParamsRange='';txtInitVal=[];txtEndogVars=[];txtExogVars=[]
    txtEndogVarValues=[];txtExogVarValues=[];txtShocks=[];labels=[];txtEstParams=[]
    txtRange='';txtFreq='';txtDescription=''
    header=None; ln=None; content=[]; start=False
    
    # Preparse @#if, @#else, @#endif directives
    with open(file_path, 'r') as f:
        for line in f:
            ln = line.strip()
            if ln.startswith("@#if"):
                block1 = []; block2 = []
                expr = ln[3:]
                condition = expr.replace("~","").strip().split(" ")[0].strip()
                if condition in conditions:
                    start = True
                    expr = expr.replace("~"," not ")
                    expr = expr.replace(condition,str(conditions[condition]))
                    b = include = eval(expr)
                else:
                    start = True
                    expr = expr.replace("~","not ").replace("true","True").replace("false","False")
                    b = include = eval(expr)
            elif start and ln.startswith("!@#else"):
                include = not include
            elif start and ln.startswith("@#endif"):
                start = False
                if b:
                    content.extend(block1)
                    # if condition in conditions:
                    #     cprint(f"Included block #1 for condition: {condition} = {conditions[condition]}","blue")
                    # else:
                    #     cprint(f"Included block #1 for condition: {expr}","blue")
                else:
                    content.extend(block2)
                    # if condition in conditions:
                    #     cprint(f"Included block #2 for condition: {condition} = {conditions[condition]}","blue")
                    # else:
                    #     cprint(f"Included block #2 for condition: {expr}","blue")
            else:
                if start:
                    if include:
                        block1.append(line)
                    else:
                        block2.append(line)
                else:
                    content.append(line)
                    
    for line in content:
        ln = line.replace("lambda","lmbda").replace("{","(").replace("}",")").replace("^","**").strip()
        if ln.startswith("//") or ln.startswith("#") or ln.startswith("%") or not bool(ln):
            continue
        if "#" in ln:
            ind = ln.index("#")
            ln = ln[:ind]
        if "//" in ln:
            ind = ln.index("//")
            ln = ln[:ind]
        if ln.startswith("model"):
            header = ln
            txt = [] 
        elif ln.startswith("initval") or ln.startswith("shocks") or ln.startswith("estimated_params") \
            or ln.startswith("var")   or ln.startswith("varexo") or ln.startswith("parameters"):
            header = ln    
            #print(header)
            txt = [] 
            if " " in ln:
                ind = ln.index(" ")
                ln2 = ln[ind:]
            else:
                ln2 = ln
            ln2 = ln2.replace(";","")
            if "//" in ln2:
                ind = ln2.index("//")
                ln2 = ln2[:ind]
            if "#" in ln2:
                ind = ln2.index("#")
                ln2 = ln2[:ind]
            ln2 = ln2.strip()
            txt = re.split(';|,| ',ln2)
            txt = [x for x in txt if bool(x.strip())]
            if "initval" in ln and not bool(txtInitVal):
                txtInitVal = txt
            elif "shocks" in ln and not bool(txtShocks):
                txtShocks = txt
            elif "estimated_params" in ln and not bool(txtEstParams):
                txtEstParams = txt
            elif "varexo" in ln and not bool(txtExogVars):
                txtExogVars = txt
            elif "var" in ln and not bool(txtEndogVars):
                txtEndogVars = txt
            elif "parameters" in ln and not bool(txtParams):
                txtParams = txt
        elif "end;" in ln and not bool(txtEqs):
            if "model" in header:
                txt = "\n".join(txt).replace("\t","").replace(" ","").split('\n')
                arr = []
                for t in txt:
                    arr.append(t)
                    if ";" in t:
                        txtEqs.append(" ".join(arr).replace(";",""))
                        arr = []
        elif bool(header) and not bool(ln):
            if "initval" in header and not bool(txtInitVal):
                txt = " ".join(txt).replace(';','').replace(","," ").split(" ")
                txtInitVal = [x.strip() for x in txt if "=" in x]
            elif "shocks" in header and not bool(txtShocks):
                txt = " ".join(txt).replace('var','').replace(';',' ').replace(","," ").split(" ")
                txtShocks = [x.strip() for x in txt]
            elif "estimated_params" in header and not bool(txtEstParams):
                txtEstParams = " ".join(txt).replace(';','').replace(","," ").split(' ')
            elif "varexo" in header and not bool(txtExogVars):
                txt = " ".join(txt).replace(';',' ').replace(","," ").split(" ")
                txtExogVars = [x.strip() for x in txt if bool(x)]
            elif "var" in header and not bool(txtEndogVars):
                txt = " ".join(txt).replace(';',' ').replace(","," ").split(" ")
                txtEndogVars = [x.strip() for x in txt if bool(x)]
            elif "parameters" in header and not bool(txtParams):
                txt = [x.replace(';','').replace(","," ") for x in txt if not "=" in x]
                txt = " ".join(txt).split(" ")
                txtParams = [x.strip() for x in txt if bool(x.replace("\n",""))]
        else:
            txt.append(ln)


    if bFillValues:
        txtShocks += txtExogVars; txtExogVars = []
        txtShocks = " ".join(txtShocks).split(" ")
    txtEndogVars = [x.strip() for x in txtEndogVars if not "=" in x and bool(x)]
    txtExogVars = [x.strip() for x in txtExogVars if not "=" in x and bool(x)]
    txtParams = [x.strip() for x in txtParams if not "=" in x and bool(x)]
    txtEndogVars = " ".join(txtEndogVars).replace(';',' ').replace(","," ").split(" ")
    txtExogVars = " ".join(txtExogVars).replace(';',' ').replace(","," ").split(" ")
    txtParams = " ".join(txtParams).replace(';',' ').replace(","," ").split(" ")
    txtEndogVars = [x for x in txtEndogVars if bool(x)]
    txtExogVars = [x for x in txtExogVars if bool(x)]
    txtParams = [x for x in txtParams if bool(x)]
    
    mapParams = {}
    with open(file_path, 'r') as f:
        for line in f:
            ln = line.strip()
            if ln.startswith("//") or ln.startswith("#"):
                continue
            if "#" in ln:
                ind = ln.index("#")
                ln = ln[:min(len(ln)-1,ind)].strip()
            if "//" in ln:
                ind = ln.index("//")
                ln = ln[:min(len(ln)-1,ind)].strip()
            if "=" in line:
                ind = ln.find('=')
                left  = ln[:ind].strip()
                right = ln[1+ind:].replace(";","").strip()
                if left in txtParams: 
                    if re.match(r'^-?\d+(?:\.\d+)?$', right) is None:
                        mapParams[left] = right
                    else:
                        mapParams[left] = float(right)
                
                   
    for eq in txtEqs:
        if '=' in eq:
            ind = eq.index('=')
            arr = re.split(regexPattern,eq[:ind])
            arr = list(filter(None,arr))
            if bool(arr):
                if len(arr) == 1:
                    labels.append(arr[0])
                else:
                    labels.append(None)
                    
    if bFillValues:
        
        txtEqs = "\n".join(txtEqs)
        for v in txtInitVal:
            ind = v.find('=')
            if ind >= 0:
                n = v[:ind].strip()
                if n in txtEndogVars:
                    txtEndogVarValues.append(v)
                
        for v in txtInitVal:
            ind = v.find('=')
            if ind >= 0:
                n = v[:ind].strip()
                if n in txtExogVars:
                    txtExogVarValues.append(v)
                
        txtParams = []
        for k,v in mapParams.items():
            txtParams.append(k + " = " + str(v))
            
        txtParams = "\n".join(txtParams)
                
        arr = []
        for v in txtEndogVarValues:
            ind = v.find('=')
            n = v[:ind].strip()
            arr.append(n)
         
        if len(arr) > 0:
            for n in txtEndogVars:
                if not n in arr:
                    txtEndogVarValues.append(n + ' = 0')
         
        if bool(txtEndogVarValues):
            txtEndogVars = txtEndogVarValues
        else:
            arr = []
            for n in txtEndogVars:
                arr.append(n + ' = 0')
            txtEndogVars = arr
        txtEndogVars = "\n".join(txtEndogVars)
        
        arr = []
        for n in txtShocks:
            arr.append(n.replace(",","").replace(";","") + ' = 0')
        txtShocks = arr
                    
        if not "Date" in txtShocks:
            txtShocks.insert(0,"Date : 01/01/2026\n") 
            
        txtShocks = "\n".join(txtShocks)
            
        if not txtRange:
            txtRange = "01/01/2025 - 01/01/2125"
          
        return txtEqs,txtParams,txtParamsRange,txtEndogVars,txtExogVars,txtShocks,txtRange,txtFreq,txtDescription
    
    else:
        
        calibration = {}
        for v in txtInitVal:
            if "=" in v:
                arr = v.split("=")
                var = arr[0].replace(" ","")
                val = arr[1].replace(";","").replace(" ","")
                try:
                    calibration[var] = float(val)
                except:
                    calibration[var] = val
            
        calibration = {**calibration, **mapParams}
            
        return txtEqs,txtEndogVars,txtEndogVarValues,txtExogVars,txtExogVarValues,txtParams,calibration,labels
    

def getDynareModel(fpath,calibration,options={},use_cache=False,debug=False):
    """
    Reads Dynare model file and instantiates this model.

    Args:
        fpath : str.
            Path to Iris model file.
        calibration : dict.
            Map with values of calibrated parameters and starting values of endogenous variables.
        options : dict, optional
            Dictionary of options. The default is empty dictionary object.        
        use_cache : bool, optional
            If True reads previously saved model from a file of model dump.
        debug : bool, optional
            If set to True prints information on Iris model file sections. The default is False.

    Returns:
        model : Model.
            Model object.
    """
    from snowdrop.src.model.factory import getModel
    
    file_path = os.path.abspath(os.path.join(path,fpath))
    fname, ext = os.path.splitext(file_path)
    model_path = file_path.replace(fname+ext,fname+".bin")
    model_file_exist = os.path.exists(model_path)
    
    if use_cache and model_file_exist:
        
        from snowdrop.src.utils.interface import loadModel
        from snowdrop.src.preprocessor.util import updateFiles
        
        model = loadModel(model_path)
        updateFiles(model,path+"/../preprocessor")
        
        # Update model variables and parameters values
        variables = model.symbols["variables"]
        parameters = model.symbols["parameters"]
        mv = model.calibration['variables'] 
        mp = model.calibration['parameters']
        for i,k in enumerate(variables):
            if k in calibration:
                mv[i] = calibration[k]
        for i,k in enumerate(parameters):
            if k in calibration:
                mp[i] = calibration[k]
        
        model.calibration['variables'] = mv
        model.calibration['parameters'] = mp
    
    else:
        
        name = os.path.basename(file_path)
        infos = {'name': name,'filename' : file_path}   
        
        eqs,variables,variables_values,shocks,shock_values,params,mapCalibration,labels = \
            readDynareModelFile(file_path=file_path,bFillValues=False)
        
        calibration = {**calibration, **mapCalibration}
        
        if bool(labels):
            var_labels = dict()
            for var,lbl in zip(variables,labels):
                var_labels[var] = lbl
        else:
            var_labels = {}
            
        if debug:  
            print("\nTransition variables:\n{}".format(variables))
            #print("\n\nLabels of variables:\n{}".format(labels))
            print("\nTransition Shocks:\n{}".format(shocks))
            print("\nParameters:\n{}".format(params)) 
            print("\nCalibration:\n{}".format(calibration)) 
            print("\nEquations:\n{}\n\n".format(eqs)) 
            
        model = getModel(name=name,eqs=eqs,variables=variables,parameters=params,shocks=shocks,
                          calibration=calibration,var_labels=var_labels,options=options,infos=infos)
                
        # Serialize model into file
        from snowdrop.src.utils.interface import saveModel
        saveModel(model_path,model)
    
    return model

    
if __name__ == '__main__':
    """The main program."""
    file_path = os.path.abspath(os.path.join(path,'../../../supplements/models/MPAF/model.mod'))
    model = getDynareModel(fpath=file_path,calibration={},debug=True)
   