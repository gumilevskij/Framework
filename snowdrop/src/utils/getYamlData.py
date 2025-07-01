import os
import re

path = os.path.dirname(os.path.abspath(__file__))

def readYamlModelFile(file_path=None,strVariables="variables",strMeasVariables="measurement_variables",
                      strMeasShocks="measurement_shocks",strShocks="shocks",strParameters="parameters",
                      strEquations="equations",strCalibration="calibration",strOptions="options",
                      strExogenous="exogenous",strValues="values",strRange="range",strFrequency="frequency",
                      strLabels="labels"):
    """
    Parse YAML model file.
    """
    import yaml
    
    if file_path is None:
        path = os.path.dirname(os.path.abspath(__file__))
        file_path = path + '/../../../supplements/models/template.yaml'      
    
    arrParamsRange=[];arrEndogVars=[];arrMeasVars=[];arrMeasShocks=[];arrExogVars=[]
    mapCalibration={};arrShocks=[];mapOptions={};mapLabels={};arrParamsRange=[];arrRange=[]
    txtEqs='';txtParams='';txtExogVars='';txtRange='';txtFreq='';txtDescription='';txtShocks='' 
    
    frequencies = {"0":"Annually","1":"Quarterly","2":"Monthly","3":"Weekly","4":"Daily"}
    
    with open(file_path, 'r') as f:
        data = yaml.load(f,Loader=yaml.FullLoader)
        if 'name' in data:
            txtDescription = data['name']
        if strEquations in data:
            txtEqs = data[strEquations]
            txtEqs = "\n".join(txtEqs) 
        if strCalibration in data:
            mapCalibration = data[strCalibration]
        if strOptions in data:
            mapOptions = data[strOptions]
        if strLabels in data:
            mapLabels = data[strLabels]
        if 'symbols' in data:
            if strVariables in data['symbols']:
                arrEndogVars = data['symbols'][strVariables]
            if strParameters in data['symbols']:
                arrParams = data['symbols'][strParameters]
            if strShocks in data['symbols']:
                arrShocks = data['symbols'][strShocks]
            if strExogenous in data['symbols']:
                arrExogVars = data['symbols'][strExogenous]
            if strMeasVariables in data['symbols']:
                arrMeasVars = data['symbols'][strMeasVariables]
            if strMeasShocks in data['symbols']:
                arrMeasShocks = data['symbols'][strMeasShocks] 
            
    
    arrEndogVars = [f'{k} = {mapCalibration[k]}' if k in mapCalibration else k for k in arrEndogVars ]
    arrParams = [f'{k} = {mapCalibration[k]}' if k in mapCalibration else k for k in arrParams]
    arrShocks = [f'{k} = {mapCalibration[k]}' if k in mapCalibration else k + ' = 0' for k in arrShocks]
    arrExogVars = [f'{k} = {mapCalibration[k]}' if k in mapCalibration else k + ' = 0' for k in arrExogVars]
    
    txtEndogVars = "\n".join(arrEndogVars) 
    txtExogVars = "\n".join(arrExogVars) 
    txtParams = "\n".join(arrParams)
    txtShocks = "\n".join(arrShocks)
     
    for k in mapOptions:
        ln1 = k
        ln2 = mapOptions[k]
        if ln1 == strRange:
            arrRange = ln2
        elif ln1 == strFrequency:
            freq = str(ln2).strip()
            if freq in frequencies.keys():
                txtFreq = frequencies[freq]
            else:
                txtFreq = "Annually"
        elif ln1 in arrParams:
            arrParamsRange.append(f" {ln1} = {ln2}")
            
    txtShocks = "Date : 01/01/2026\n" + txtShocks 
                    
    if len(arrRange) == 2:
        arr1 = arrRange[0]
        arr2 = arrRange[1]
        if ',' in arr1:
            arr1 = arr1.split(',')
            arr2 = arr2.split(',')
        if '/' in arr1:
            arr1 = arr1.split('/')
            arr2 = arr2.split('/')
        txtRange = str(arr1[1]) + "/" + str(arr1[2])  + "/" +  str(arr1[0]) + " - " + str(arr2[1]) + "/" + str(arr2[2])  + "/" +  str(arr2[0])
        
    else:
        txtRange = "01/01/2025 - 01/01/2125"
         
    return txtEqs,txtParams,arrParamsRange,txtEndogVars,txtExogVars,txtShocks,mapLabels,txtRange,txtFreq,txtDescription
  
 
    
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
    file_path = os.path.abspath(path + '/../../../supplements/models/TOY/JLMP98.yaml')  
        
    txtEqs,txtParams,txtParamsRange,txtEndogVars,txtExogVars,txtShocks,labels,txtRange,txtFreq,txtDescription = readYamlModelFile(file_path=file_path)
    
    print("Exogenous variables:")
    print(txtExogVars)
    print()
    print("Parameters:")
    print(txtParams)
    print()
    print("Shocks:")
    print(txtShocks)
    
    

    
        