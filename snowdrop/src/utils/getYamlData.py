import os
import re

path = os.path.dirname(os.path.abspath(__file__))


def readYamlModelFile(file_path=None,strVariables="variables",strMeasVariables="measurement_variables",
                      strMeasShocks="measurement_shocks",strShocks="shocks",strParameters="parameters",
                      strEquations="equations",strCalibration="calibration",strOptions="options",
                      strValues="values",strRange="range",strFrequency="frequency"):
    """
    Parse YAML model file.
    """
    if file_path is None:
        path = os.path.dirname(os.path.abspath(__file__))
        file_path = path + '/../../models/template.yaml'      
    
    txt=[];txtEqs=[];txtParams=[];txtParamsRange=[];txtEndogVars=[];txtMeasVars=[];txtMeasShocks=[]
    txtExogVars=[];txtCalibration=[];txtShocks=[];txtOptions=[]
    txtRange='';txtFreq='';txtDescription=''  
    
    frequencies = {"0":"Annually","1":"Quarterly","2":"Monthly","3":"Weekly","4":"Daily"}
    strName = "name"
    
    header = None
    with open(file_path, 'r') as f:
        for line in f:
            ln = line.strip()
            ln1 = ln.replace(":"," ")
            ind = ln1.find(" ")
            ln1 = ln1[:ind].strip()
            ln2 = ln[ind:].replace(":","").replace("[","").replace("]","").strip()
            if ln1 in [strEquations,strCalibration,strOptions]:
                header = ln1
                txt = []
            elif len(txt) > 0 and not ln:
                if header == strEquations:
                    txtEqs = txt
                elif header == strCalibration:
                    txtCalibration = txt
                elif header == strOptions:
                    txtOptions = txt
                txt = []  
            else:
                if ln1 == strName:
                    txtDescription = re.split('; |, | ',ln2)
                    txtDescription =  " ".join(txtDescription )
                elif ln1 == strVariables:
                    txtEndogVars = re.split('; |, | ',ln2)
                    txtEndogVars =  " ".join(txtEndogVars)
                elif ln1 == strMeasVariables:
                    txtMeasVars = re.split('; |, | ',ln2)
                    txtMeasVars =  " ".join(txtMeasVars)
                elif ln1 == strShocks:
                    txtShocks = re.split('; |, | ',ln2)
                    txtShocks =  " ".join(txtShocks).strip()
                elif ln1 == strMeasShocks:
                    txtMeasShocks = re.split('; |, | ',ln2)
                    txtMeasShocks =  " ".join(txtMeasShocks).strip()
                elif ln1 == strParameters:
                    txtParams = re.split('; |, | ',ln2)
                    txtParams =  " ".join(txtParams)
                elif ln:
                     txt.append(ln.strip(';').strip())
    

    # Process last line
    if header == strOptions and txt:
        txtOptions = txt  
            
    eqs = []                
    for e in txtEqs:
        eq = e.strip()
        if eq[0] == "-":
            eq = eq[1:]
        eqs.append(eq)
        
    txtEqs = "\n".join(eqs) 
    
    arrEndogVars = txtEndogVars.split(",")
    arrShocks = txtShocks.split(",")
    arrParams = txtParams.split(",")
    
        
    endogVars = []; exogVars = []; params = []
    for line in txtCalibration:
        ln = line.strip()
        ind = ln.find(":")
        ln1 = ln[:ind].strip()
        if ln1 in arrEndogVars:
            endogVars.append(line.replace(":"," = ").strip())
        elif ln1 in arrShocks:
            exogVars.append(line.replace(":"," = ").strip())
        elif ln1 in arrParams:
            params.append(line.replace(":"," = ").strip())
    
    txtEndogVars = "\n".join(endogVars) 
    txtExogVars = "\n".join(exogVars) 
    txtParams = "\n".join(params)
     
    shocks = []
    for line in  txtOptions:
        ln = line.strip()
        ind = ln.find(":")
        ln1 = ln[:ind].strip()
        ln2 = ln[1+ind:].strip()
        if ln1 == strRange:
            txtRange = ln2
        elif ln1 == strFrequency:
            freq = ln2.strip()
            if freq in frequencies.keys():
                txtFreq = frequencies[freq]
            else:
                txtFreq = "Annually"
        elif ln1 == strValues:
            values = ln2.replace("[","").replace("]",",").strip()
            values = values.replace(",,",",").split(",")
            i = 0
            for sh in txtShocks.split(","):
                if sh and i < len(values):
                    shocks.append(sh.strip() + " = " + values[i])
                    i = i + 1
        elif ln1 in txtParams:
            txtParamsRange.append(ln.replace(":"," = "))
            
    arr = []
    for sh in shocks:
        if "=" in sh:
            arr.append(sh)
        else:
            arr.append(sh + " = 0")
            
    if len(arr) > 0:                
        txtShocks = "\n".join(arr)    
        
    if not "Date" in txtShocks:
        txtShocks = "Date : 01/01/2001\n" + txtShocks 
                
        
    if txtRange:
        arr = txtRange.replace("[","").replace("]","").split(",")
        if len(arr) == 6:
            txtRange = str(arr[1]) + "/" + str(arr[2])  + "/" +  str(arr[0]) + " - " + str(arr[4]) + "/" + str(arr[5])  + "/" +  str(arr[3])
        
    else:
        txtRange = "01/01/2000 - 01/01/2100"
        
    #print(txtEqs,txtParams,txtParamsRange,txtEndogVars,txtExogVars,txtShocks,txtRange,txtFreq,txtDescription)  
    return txtEqs,txtParams,txtParamsRange,txtEndogVars,txtExogVars,txtShocks,txtRange,txtFreq,txtDescription
  
    
def getYamlModel(fpath,calibration={},labels={},options={},use_cache=False,debug=False):
    """
    Reads Yaml model file and instantiates this model.

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
    from model.model import Model
    from model.factory import import_model
    
    file_path = os.path.abspath(os.path.join(path,fpath))
    fname, ext = os.path.splitext(file_path)
    model_path = file_path.replace(fname+ext,fname+".bin")
    model_file_exist = os.path.exists(model_path)
    
    if use_cache and model_file_exist:
        
        from utils.interface import loadModel
        from preprocessor.util import updateFiles
        
        model = loadModel(model_path)
        updateFiles(model,path+"/../preprocessor")
        
        # Update model variables and parameters values
        variables = model.symbols["variables"]
        parameters = model.symbols["parameters"]
        shocks = model.symbols["shocks"]
        mv = model.calibration['variables'] 
        mp = model.calibration['parameters']
        ms = model.calibration['shocks']
        for i,k in enumerate(variables):
            if k in calibration:
                mv[i] = calibration[k]
        for i,k in enumerate(parameters):
            if k in calibration:
                mp[i] = calibration[k]
        for i,k in enumerate(shocks):
            if k in calibration:
                ms[i] = calibration[k]
        
        model.calibration['variables'] = mv
        model.calibration['parameters'] = mp
        model.options['shock_values'] = ms
        
    
    else:
        
        name = os.path.basename(file_path)
        infos = {'name': name,'filename' : file_path}   
        
        interface = import_model(fname=file_path)
        interface.calibration_dict = {**interface.calibration_dict,**calibration}
        
        variables = interface.symbols["variables"]
        shocks = interface.symbols["shocks"]
        params = interface.symbols["parameters"]
        eqs = interface.equations
        
        if debug:  
            print("\nTransition variables:\n{}".format(variables))
            print("\nShock variables:\n{}".format(shocks))
            print("\nParameters:\n{}".format(params)) 
            print("\nEquations:\n{}\n\n".format(eqs)) 
            
        model = Model(interface, infos=infos)
        
        mv = dict(zip(variables,model.calibration["variables"]))
        mp = dict(zip(params,model.calibration["parameters"]))
        ms = dict(zip(shocks,model.calibration["shocks"]))
        
        if debug:  
            print(f"\n\nTransition variables:\n{mv}")
            print(f"\nShocks:\n{ms}")
            print("\nParameters:\n{mp}")
        
        # Serialize model into file
        from utils.interface import saveModel
        saveModel(model_path,model)
    
    return model


if __name__ == "__main__":
    
    path = os.path.dirname(os.path.abspath(__file__))
    file_path = path + '/../../models/Toy/RBC.yaml'  
        
    txtEqs,txtParams,txtParamsRange,txtEndogVars,txtExogVars,txtShocks,txtRange,txtFreq,txtDescription = readYamlModelFile(file_path=file_path)
    
    print("Exogenous variables:")
    print(txtExogVars)
    print()
    print("Parameters:")
    print(txtParams)
    print()
    print("Shocks:")
    print(txtShocks)
    
    

    
        