import os,re
from snowdrop.src.misc.termcolor import cprint
from snowdrop.src.utils.util import read_and_combine_text

path = os.path.dirname(os.path.abspath(__file__))


def handleMovingAverageOperator(n,expr):
    """Handles IRIS moving average operator."""
    new_expr = ""
    sign = 1 if n > 0 else -1
    for i in range(abs(n)):
        if i == 0:
            new_expr += "(" + expr + ")"
        else:
            new_expr += "+(" + expr + "(" + str(sign*i) + "))"
        
    new_expr = "(" + new_expr + ") / " + str(abs(n))
    return new_expr
 
def handleMovingSumOperator(n,expr):
    """Handles IRIS moving sum operator."""
    new_expr = ""
    sign = 1 if n > 0 else -1
    for i in range(abs(n)):
        if i == 0:
            new_expr += "(" + expr + ")"
        else:
            new_expr += "+(" + expr + "(" + str(sign*i) + "))"
        
    new_expr = "(" + new_expr + ") "
    return new_expr            
     
def handleDiffOperator(n,expr):
    """Handles IRIS difference operator."""
    new_expr = " ((" + expr + ") - (" + expr + "(" + str(n) + "))) "
    return new_expr            
                            
def handleDiffLogOperator(n,expr):
    """Handles IRIS difference of log operator."""
    new_expr = " (log(" + expr + ") - log(" + expr + "(" + str(n) + "))) "
    return new_expr            
                     
def emulateModelLanguageOperators(equation,operator):
    """
    Emulate IRIS modelling language special operators like movavg(expr,n) for n-period moving average
    """
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
        if operator in ["movavg","movsum"]:
            # Find difference n-period
            expr = expr.replace(" ","")
            ind = expr.find(",")
            if ind == -1:
                n = -1
            else:
                ex = expr[1+ind:].replace(")","")
                if re.match(r"\d+$", ex):
                    n = int(ex)
                    expr = expr[1:ind]
            if operator == "movavg":
                new_expr = handleMovingAverageOperator(n,expr)
            elif operator == "movsum":
                new_expr = handleMovingSumOperator(n,expr)
        elif operator == "difflog":
            # Find difference n-period
            expr = expr.replace(" ","")
            ind = expr.find(",")
            if ind == -1:
                n = 4
            else:
                ex = expr[1+ind:].replace(")","")
                if re.match(r"\d+$", ex):
                    n = int(ex)
                    expr = expr[1:ind]
            if operator == "diff":
                new_expr = handleDiffOperator(n,expr)
            elif operator == "difflog":
                new_expr = handleDiffLogOperator(n,expr)
        new_eq +=  new_expr    
        rest_eq = rest_eq[l:]
        ind = rest_eq.find(operator + "(")
        
    return new_eq	


def readIrisModelFile(file_path,conditions={},bFillValues=True,
                      strVariables = ["!variables","!transition_variables"],strShocks = ["!shocks","!transition_shocks"],
                      strParameters = "!parameters",strEquations = ["!equations","!transition_equations"],
                      strMeasurementVariables="!measurement_variables",strMeasurementEquations="!measurement_equations",
                      strMeasuarementShocks="!measurement_shocks",strLegend="legend"):
    """Read IRIS model file."""
    
    delimiters = " ", ",", ";", "*", "/", ":", "+", "-", "="
    regexPattern = '|'.join(map(re.escape, delimiters))
        
    txt=[];txtEqs=[];txtParams=[];txtParamsRange=[];txtEndogVars=[];
    txtShocks=[];txtMeasVar=[];txtMeasEqs=[];txtMeasShocks=[];txtLegend=[]
    txtRange='';txtFreq='';txtDescription=''
    
    if len(conditions) == 0:
        with open(file_path,  encoding='utf8') as f:
            text = f.read()
            text = text.replace("^","**")
    else:
        text = read_and_combine_text(file_path,conditions=conditions)
    lines = text.split("\n")
                
    # Parse model file
    for line in lines:
        ln = line.strip()
        if ln.startswith("%") or ln.startswith("//")  or ln.startswith("#"):
            continue
        ln2 = ln.strip(";")
        #print ("-" + ln2 + "-")
        if ln.startswith("!") or ln.lower().startswith("legend"):
            header = ln
            txt = []
        elif len(txt) > 0 and not bool(ln2):
            if header in strEquations:
                txtEqs = '\n'.join(txt)
            elif header in strVariables:
                txtEndogVars = '\n'.join(txt).replace(';','').split('\n')
            elif header in strParameters:
                txtParams = '\n'.join(txt).replace(';','').replace(',',' ').split('\n')
            elif header in strShocks:
                txtShocks = '\n'.join(txt).replace('var ','').replace(';','').replace(',',' ').split('\n')
            elif header in strMeasurementVariables:
                txtMeasVar = '\n'.join(txt).replace(';','').split('\n')
            elif header in strMeasurementEquations:
                txtMeasEqs = '\n'.join(txt).replace(';','').split('\n')
            elif header in strMeasuarementShocks:
                txtMeasShocks = '\n'.join(txt).replace(';','').split('\n')
            elif header.lower() in strLegend:
                txtLegend= '\n'.join(txt).replace(';','').split('\n')
        elif bool(ln2):
            txt.append(ln2.replace('EXP(','exp(').replace('LOG(','log(').replace('SIN(','sin(').replace('COS(','cos('))

    # Process the last line
    if header in strEquations:
        txtEqs = '\n'.join(txt)
    elif header in strMeasurementEquations:
        txtMeasEqs = '\n'.join(txt).replace(';','').split('\n')
    elif header in strVariables:
        txtEndogVars = '\n'.join(txt).replace(';',' ').replace('\t',' ').split('\n')
    elif header in strMeasurementVariables:
        txtMeasVar = '\n'.join(txt).replace(';',' ').replace('',' ').replace('\t',' ').split('\n')
    elif header in strParameters:
        txtParams = '\n'.join(txt).replace(';','').replace(',',' ').split('\n')
    elif header in strShocks:
        txtShocks = '\n'.join(txt).replace('var ','').replace(';','').replace(',',' ').split('\n')
    elif header in strMeasuarementShocks:
        txtMeasShocks = '\n'.join(txt).replace('var ','').replace(';','').replace(',',' ').split('\n')
    elif header in strLegend:
        txtLegend = '\n'.join(txt).replace(';','').replace(',',' ').split('\n')
        
    txtEqs = txtEqs.replace("{","(").replace("}",")").replace("=#","=").replace("..."," ").split('\n')
    eqs = [x.replace("\t","").replace(" ","") for x in txtEqs if not "'" in x]
    txtMeasEqs = [x.replace("\t","").replace(" ","") for x in txtMeasEqs if not "'" in x]
    
    var = []; labels = []
    for t in txtEndogVars:
        s = t.replace('\t',' ').strip()
        arr = [s for s in re.split(r"[);\W]+", s)]
        v = arr[-1]
        var.append(v)
        lbl = t.replace(v,"").replace('"','').replace("'","").strip()
        labels.append(lbl)
        
    txtEndogVars = var
   
    txtEqs = []; ss = {}; eqtn = None
    operators = ["movavg","movsum","diff","difflog"]
    for i,eq in enumerate(eqs):
        if '=' in eq:
            if not eqtn is None:
                txtEqs.append(eqtn)
            eqtn = eq
            # ind = eq.index('=')
            # arr = re.split(regexPattern,eq[:ind])
            # arr = list(filter(None,arr))
            # if bool(arr):
            #     if len(arr) == 1:
            #         labels.append(arr[0])
            #     else:
            #         labels.append(str(i))
        else:
            eqtn += " " + eq
        if "!!" in eqtn:
            ind = eqtn.index("!!")
            tmp = eqtn[1+ind:]
            eqtn = eqtn[:ind]
            if "=" in tmp:
                ind2 = tmp.index("=")
                k = tmp[1:ind2].strip()
                val = tmp[1+ind2:].strip()
                ss[eqtn] = (k,val)
        # Handle IRIS operators
        for operator in operators:
            while operator in eqtn:
                eqtn = emulateModelLanguageOperators(equation=eqtn,operator=operator)
            
    # Append the last one        
    txtEqs.append(eqtn)
    
    meas_var = []; meas_labels = []
    for t in txtMeasVar:
        if "'" in t:
            ind = t.rindex("'")
            meas_labels.append(t[:ind])
            t = t[1+ind:]
        t = t.replace(","," ")
        meas_var.append(t)
    meas_var = ' '.join(meas_var).split(' ')
    txtMeasVar = [x.strip() for x in meas_var if bool(x.strip())]
    
    meas_eqs = []
    for t in txtMeasEqs:
        if "'" in t:
            ind = t.rindex("'")
            t = t[1+ind:]
        t = t.replace(","," ")
        meas_eqs.append(t)
    meas_eqs = ' '.join(meas_eqs).split(' ')
    txtMeasEqs = [x.strip() for x in meas_eqs if bool(x.strip())]
        
    meas_shocks = []
    for t in txtMeasShocks:
        if "'" in t:
            ind = t.rindex("'")
            t = t[1+ind:]
        t = t.replace(","," ")
        meas_shocks.append(t)
    meas_shocks = ' '.join(meas_shocks).split(' ')
    txtMeasShocks = [x.strip() for x in meas_shocks if bool(x.strip())]
        
    shock_var = []; shock_labels = []
    for t in txtShocks:
        if "'" in t:
            ind = t.rindex("'")
            shock_labels.append(t[:ind])
            t = t[1+ind:]
        shock_var.append(t)
    shock_var = ' '.join(shock_var).split(' ')
    txtShocks = [x.strip() for x in shock_var if bool(x.strip())]
    
    param_var = []; param_labels = []
    for t in txtParams:
        if "'" in t:
            ind = t.rindex("'")
            param_labels.append(t[:ind])
            t = t[1+ind:]
        if "=" in t:
            ind = t.index("=")
            t = t[:ind]
        param_var.append(t.strip())
    param_var = ' '.join(param_var).split(' ')
    txtParams = [x.strip() for x in param_var if bool(x.strip())]
    
    
    if bFillValues:
        
        delimiters = " ", ",", ";"
        regexPattern = '|'.join(map(re.escape, delimiters))
            
        arr = []
        for p in txtParams:
            arr2 = re.split(regexPattern,p)
            for v in arr2:
                if v:
                    arr.append(v.strip() + " = 1")
        txtParams = "\n".join(arr)   
                
        arr = []
        for t in txtEndogVars:
            arr.append(t.strip() + " = 1")
        txtEndogVars = "\n".join(arr)
            
        arr = []
        for s in txtShocks:
            arr.append(s.strip() + " = 0")
        txtShocks = arr
            
        if not "Date" in txtShocks:
            txtShocks.insert(0,"Date : 01/01/2020") 
            
        if not txtRange:
            txtRange = "01/01/2020 - 01/01/2100"
        
        txtEqs = "\n".join(txtEqs)
        txtShocks = "\n".join(txtShocks)
        
        return txtEqs,txtParams,txtParamsRange,txtEndogVars,txtMeasVar,txtShocks,txtRange,txtFreq,txtDescription
    
    else:
        return txtEqs,txtMeasEqs,txtParams,txtEndogVars,txtMeasVar,txtMeasEqs,txtShocks,txtMeasShocks,ss,labels


def getIrisModel(fpath,calibration={},options={},conditions={},use_cache=False,
                     tag_variables=["!variables","!transition_variables"],tag_shocks = ["!shocks","!transition_shocks"],
                     tag_parameters = "!parameters",tag_equations = ["!equations","!transition_equations"],
                     tag_measurement_variables="!measurement_variables",
                     tag_measurement_equations="!measurement_equations",
                     tag_measurement_shocks="!measurement_shocks",
                     check=True,debug=False):
    """
    Reads Iris model file and instantiates this model.

    Args:
        fpath : str.
            Path to Iris model file.
        calibration : dict.
            Map with values of calibrated parameters and starting values of endogenous variables.
        options : dict, optional
            Dictionary of options. The default is empty dictionary.
        conditions : dict, optional
            Choose block of code based on logical condition. The default is empty list.
        use_cache : bool, optional
            If True reads previously saved model from a file of model dump.
        tag_variables : list, optional
            Tag for endogenous variables section. The default is "!transition_variables".
        tag_shocks : str, optional
            Tag for shock variables section. The default is "!transition_shocks".
        tag_parameters : str, optional
            Tag for parameters section. The default is "!parameters".
        tag_equations : TYPE, optional
            Tag for equations section. The default is "!transition_equations".
        tag_measurement_variables : str, optional
            Tag for measurement variables section. The default is "!measurement_variables".
        tag_measurement_equations : str, optional
            Tag for measurement equations section. The default is "!measurement_equations".
        debug : bool, optional
            If set to True prints information on Iris model file sections. The default is False.

        Returns:
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
            elif "_plus_" in k:
                ind = k.index("_plus_")
                kk  = k[:ind]
                if kk in calibration:
                    mv[i] = calibration[kk]
            elif "_minus_" in k:
                ind = k.index("_minus_")
                kk  = k[:ind]
                if kk in calibration:
                    mv[i] = calibration[kk]
                    
        for i,k in enumerate(parameters):
            if k in calibration:
                mp[i] = calibration[k]
        
        model.calibration['variables'] = mv
        model.calibration['parameters'] = mp
        model.symbolic.calibration_dict = {**model.symbolic.calibration_dict,**calibration}
        
    else:
        
        name = os.path.basename(file_path)
        infos = {'name': name,'filename' : file_path}   
        
        eqs,measEqs,params,variables,measVar,measEqs,shocks,measShocks,ss,labels = \
            readIrisModelFile(file_path=file_path,conditions=conditions,bFillValues=False,
                              strVariables=tag_variables,strShocks=tag_shocks,strParameters=tag_parameters,strEquations=tag_equations,
                              strMeasurementVariables=tag_measurement_variables,strMeasurementEquations=tag_measurement_equations,
                              strMeasuarementShocks=tag_measurement_shocks)  
            
        var_labels = dict(zip(variables,labels))
        eqs_labels = [""]*len(eqs)
        for i,eq in enumerate(eqs):
            if "=" in eq:
                ind = eq.index("=")
                lbl = eq[:ind].strip()
                eqs_labels[i] = lbl
                
        labels = '\n'.join(labels)
        if debug:  
            equations = '\n'.join(eqs)
            meas_equations = '\n'.join(measEqs)
            print(f"\nParameters:\n{params}")
            print(f"\nTransition Shocks:\n{shocks}")
            print(f"\nTransition Variables:\n{variables}")
            print(f"\nTransition Equations:\n{equations}\n")
            print(f"\n\nLabels of Variables:\n{var_labels}")
            print(f"\nMeasurement Variables:\n{measVar}")
            print(f"\nMeasurement Equations:\n{meas_equations}\n")
            print(f"\nMeasurement Shocks:\n{measShocks}\n")
            if len(ss) > 0:
                cprint(f"\nModel file defines the following steady states of variables: \n{ss}\n","red")
            
        model = getModel(name=name,eqs=eqs,ss=ss,meas_eqs=measEqs,variables=variables,parameters=params,shocks=shocks,
                         meas_variables=measVar,calibration=calibration,var_labels=var_labels,meas_shocks=measShocks,
                         eqs_labels=eqs_labels,check=check,options=options,infos=infos)
        model.symbolic.labels = labels
        
        # Serialize model into file
        from snowdrop.src.utils.interface import saveModel
        saveModel(model_path,model)
    
    return model

if __name__ == '__main__':
    """The main program."""
    fpath = os.path.abspath(os.path.join(path,'../../../supplements/models/TOY/RBC.model'))
    model = getIrisModel(fpath=fpath,debug=False)
    print(model)
   
 