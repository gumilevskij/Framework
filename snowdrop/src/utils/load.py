# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 19:27:09 2018

@author: agoumilevski
"""
import os
import sys
import re
import pandas as pd
import numpy as np
from snowdrop.src.misc.termcolor import cprint
import ruamel.yaml as yaml
version = yaml.__version__


def read_file_or_url(url,labels={},conditions={}):
    """
    Returns content of file or url.
    
        :param url: Path to file or URL address.
        :type url: str.
        :returns:  File or URL content.
    """
    import os
    from urllib import request
    
    substVars = {}
    if 'http' in url:
        txt = request.urlopen(url).read()
        txt = txt.decode('utf8') # not very robust
    else:
        # must be a file
        with open(url,  encoding='utf8') as f:
            txt = f.read()
            lst = None
            bEquations = bVariables = bParameters = False
            if "@include" in txt:
                dir_name = os.path.dirname(url)
                lst = []
                arr = txt.split("\n")
                for line in arr:
                    if line.strip().startswith("#"):
                        lst.append(line)
                    elif "@include" in line:
                        ind  = line.index("@include")
                        fl   = list(filter(None,re.split(" |,",line[ind+8:])))[0]
                        file = os.path.abspath(os.path.join(dir_name,fl))
                        if os.path.exists(file):
                            incl,lbls = read_file_or_url(file,labels)
                        else:
                            incl = ""
                        # Remove lines with comments
                        section = incl.split("\n")
                        section = [x[:x.index("//")] if "//" in x else x for x in section if not x.strip().startswith("//")]
                        incl = "\n".join(section)
                        # Special treatment for Dynare/Iris model files
                        if fl.endswith(".mod") or fl.endswith(".model"):
                            keyWords = ["in","lambda"]
                            bVariables = "variables:" in line.replace(" ","").lower()
                            bParameters = "parameters:" in line.replace(" ","").lower()
                            bEquations = "equations:" in line.replace(" ","").lower()
                            if bVariables and ";" in incl:
                                incl = incl[:incl.index(";")]
                            incl,labels = convert(incl,labels,bEquations,substitute=substVars,conditions=conditions)
                            if bVariables:
                                if incl.startswith("var,"):
                                    incl = incl[4:]
                                incl,substVars = substitute(incl,keyWords,substVars)
                                incl = drop_parenthesis(incl)
                            if bParameters:
                                incl,substVars = substitute(incl,keyWords,substVars)

                        ind  = line.index("@include")
                        beg  = line[:ind] + " \n" if bEquations else line[:ind]
                        ind  = line.index(fl)
                        end  = line[ind+len(fl):]
                        lst.append(beg + incl + end)
                    else:
                        lst.append(line)
                        
            if not lst is None:
                txt = "\n".join(lst)

    return txt,labels


def drop_parenthesis(s):
    import re
    if ('(') in s and (')') in s:
        ind1 = [m.start() for m in re.finditer(r'\(',s)]
        ind2 = [m.start() for m in re.finditer(r'\)',s)]
        j1 = 0; x = ""
        for i in range(len(ind1)):
            j2 = ind1[i]
            x += s[j1:j2]
            j1 = 1+ind2[i]
        x += s[1+ind2[-1]:]
    else:
        x = s
    
    x = x.replace(",,",",")
    if x.startswith(","): x = x[1:]
    if x.endswith(","): x = x[:-1]
    
    return x
        
    
def substitute(variables,keyWords,subst={}):
    """
    Replaces variables names that belong to the key words in Python.
    
        :param variables: Path to file or URL address.
        :type variables: str.
        :param keyWords: Path to file or URL address.
        :type keyWords: str.
        :returns: variable name appended with underscore.
    """
    var = []
    for v in variables.split(","):
        if v in keyWords:
            subst[v] = v+"_"
            var.append(v+"_")
        elif v is None:
            var.append("")
        else:
            var.append(v)
    var = ",".join(var)
            
    return var,subst
    

def convert(txt,labels,bEquations=False,substitute=None,conditions={}):
    """
    Convert Dynare/Iris model files to yaml format.
    
        :param txt: Path to model file
        :type txt: str.
        :param labels: Labels
        :type labels: dict.
        :returns:  Converted text
    """
    start1 = False; start2 = False; b = False; bModel = False
    lst = []; eqs_labels= []; lbls={}; label = None
    if "model" in txt or "varexo" in txt or "parameters" in txt:
        arr = txt.split("\n")
        for line in arr:
            ln = line.strip()
            ln = ln.replace("^","**")
            if ln.startswith("#") or ln.startswith("//") or len(ln.strip())==0:
                continue
            if ln == "model;":
                i = 0
                bModel = start1 = True
                continue
            elif ln in ["var","varexo","parameters"]:
                b = start2 = True
                continue
            else:
                if ln in [";","end;"]:
                    start1 = start2 = False
                    break
                    
            ln = ln.replace(";","").replace("'","")
            if start1 and len(ln) > 0 and not ln.startswith("---") and not ln.startswith("/*") and not ln.startswith("*/"):
                if ln.startswith("["):
                    i += 1
                    if "=" in ln:
                        ind1 = ln.index("=")
                    else:
                        ind1 = ln.index("[")
                    if "]" in ln:
                        ind2 = ln.index("]")
                    else:
                        ind2 = len(ln)
                    label = ln[1+ind1:ind2].replace(":","")
                else:
                    lst.append("   - " + ln)
                    if not label is None:
                        eqs_labels.append(label)
                    
            elif start2:
                if " " in ln:
                    ind1 = ln.index(" ")
                    ln1 = ln[:ind1]
                    ln2 = ln[1+ind1:]
                    lst.append(ln1)
                    if "=" in ln2:
                        ind2 = ln2.index("=")
                        label = ln2[1+ind2:]
                    else:
                        label = ln2
                    label = label.replace("'","").replace("(","").replace(")","").replace("[","").replace("]","")
                    lbls[ln1] = label
                else:
                    lst.append(ln)
        
        if b:
            text = ",".join(lst)
        else:
            text = "\n".join(lst)
    else:
        if bEquations:
            delimiters = " ", "#", "*", "/", "+", "-", "^", "{", "}", "(", ")", "="
            regexPattern = '|'.join(map(re.escape, delimiters))
            lstEqs = []; definitions = {}; i = 0
            lst = list(filter(None,txt.split("\n")))
            for x in lst:
                x = x.strip()
                if x.startswith("//"):
                    continue
                elif "[" in x:
                    ind1 = x.index("=")
                    ind2 = x.index("]")
                    x = x[1+ind1:ind2]
                    label = x.replace("'","")
                else:
                    i += 1
                    eq = x.replace(";","")
                    arr = re.split(regexPattern,eq)
                    arr = list(filter(None,arr))
                    if "%" in eq:
                        ind = eq.index("%")
                        eq = eq[:ind]
                    elif x.startswith("#"):
                        ind = eq.index("=")
                        key = eq[1:ind].strip()
                        rhs = eq[1+ind:].strip()
                        rhs = modifyEquation(rhs,substitute,arr)
                        rhs = modifyEquation(rhs,definitions,arr)
                        definitions[key] = "(" + rhs + ")"
                        continue
                    eq = modifyEquation(eq,definitions,arr)
                    eq = modifyEquation(eq,substitute,arr)
                    lstEqs.append(eq)
                    if not label is None:
                        lbls[i] = label
                    else:
                        lbls[i] = ""
                    label = None
            lstEqs = parseIfStatements(lstEqs,conditions)
            for i,eq in enumerate(lstEqs):
                lstEqs[i] = "   - " + eq
            text = "\n" + "\n".join(lstEqs)
        else:
            tmp = txt.split("\n")
            tmp = [x.strip() for x in tmp if not x.strip()=="var" or len(x.strip())==0]
            var = [x[:x.index(" ")] for x in tmp if " " in x]
            tmp = [x[1+x.index("="):] if "=" in x else "" for x in tmp ]
            tmp = [x[1+x.index("'"):] for x in tmp if "'" in x]
            tmp = [x[:x.index("'")] for x in tmp if "'" in x]
            tmp = [x.replace("'",", ").replace(";","; ") for x in tmp]
            lbls = dict(zip(var,tmp))
            lst = list(filter(None,txt.split(" ")))
            lst = [x.replace(",","").replace("\n",",") for x in lst if not "%" in x]
            lst = list(filter(None,lst))
            text = ",".join(lst)
            text = text.replace(";","").replace(",,",",")
            if text.startswith(","): text = text[1:]
            if text.endswith(","): text = text[:-1]
    
    labels = {**labels,**lbls}
    return text,labels
              
            
def modifyEquation(eq,m,arr):
    """
    Replace variables in equations.
    
        :param eq: Equation
        :type eq: str
        :param m: Dictionary of original and modified variables
        :type m: dict
        :param arr: Variables
        :type arr: list
        :returns:  Modified equation
    """
    ind1 = 0; new_eq = ""
    for v in arr:
        if v in m:
            ind2  = eq.index(v,ind1) 
            new_eq += eq[ind1:ind2] + m[v]
            ind1 = ind2 + len(v)
    
    if ind1 == 0:
        new_eq = eq
    else:
        new_eq += eq[ind1:]
            
    return new_eq
    

def parseIfStatements(eqs,conditions={}):
    """
    Preparse @#if, @#else, @#endif directives
    
        :param eqs: Equations
        :type eqs: list.
        :param conditions: Directives
        :type conditions: dict.
        :returns:  Preparsed list of equations
    """
    content = []; start = False
    for eqtn in eqs:
        eq = eqtn.strip()
        if eq.startswith("@#if"):
            block = []
            expr = eq[4:].replace("||","or").replace("&&","&").replace("&","and").strip()
            start = True
            try:
                include = eval(expr,globals(),conditions)
            except:
                include = False
        elif start and eq.startswith("@#else"):
            include = not include
        elif start and eq.startswith("@#endif"):
            start = False
            if include:
                content.extend(block)
        else:
            if start:
                if include:
                    block.append(eqtn)
            else:
                content.append(eqtn)
                
    return content   
            
                    
def build(indices,categories,txt,out=[],n=0):
    """
    Recursively iterates thru a list of indices and categories, 
    substitutes an index in a variable name with a corresponding category, 
    and builds a list of new variables.

    Parameters:
        :param indices: List of indeces.
        :type indices: list.
        :param categories: List of categories.
        :type categories: list.
        :param txt: Text.
        :type txt: string.
        :param n: Number of calls.
        :type n: int.
        :returns: list object.
    """
    n += 1
    if len(indices) == 0 or len(categories) == 0:
        return n,out
    
    ind = indices.pop()
    cat = categories.pop()
    
    # Loop over categories
    sub = "{"+ind+"}"
    if n == 1:
        lst = out.copy()
        if sub in txt:
            for c in cat:
                lst.append(txt.replace(sub,c))
        else:
            lst.append(txt)
    else:
        lst = []
        for t in out:
            if sub in t:
                for c in cat:
                    lst.append(t.replace(sub,c))
            else:
                lst.append(t)
                
    i,out = build(indices,categories,txt,lst,n)
    
    return i,out
            
                
def buildEquations(sets,equations,variables=[],comments=[]):
    """
    Parse sets and equations and build a list of equations and variables.
    
    Parameters:
        :param sets: List of categories.
        :type sets: list.
        :param equations: List of equations.
        :type equations: list.
        :param variables: New variables.
        :type variables: list.
        :returns: list object.
    """
    eqs=[]; eqs_ss=[]; eqs_files = []
    
    ### Canonical form of yaml file format with sets definition suggested by Benjamin
    for e in equations:
        if isinstance(e,dict):
            for k in e.keys():
                if k == 'file':
                    eqs_files = e.get('file', [])
                    continue
                v = e[k]
                if isinstance(v,dict):
                    st  = v.get('set', [])
                    if isinstance(st,str):
                        st = [st]
                    indices  = v.get('index', [])
                    if isinstance(indices,str):
                        indices = [indices]
                    if len(st) != len(indices):
                        raise IndexError("Length of indeces {} and sets {} is different".format(len(st),len(indices)))                 
                    categories = []
                    for s in st:
                        if s in sets.keys():
                            # Get list of categories for given set
                            cat = sets[s]
                            if isinstance(cat,str):
                                cat = [cat]
                            categories.append(cat)
                    # Loop over current set
                    eq = v.get('eq', '').strip()
                    n_eqs,eqs = build(indices.copy(),categories.copy(),eq,eqs)
                    eq_ss  = v.get('eq_ss', '').strip()
                    n_eqs_ss,eqs_ss = build(indices.copy(),categories.copy(),eq_ss,eqs_ss)
                    endo  = v.get('endo', '')
                    n_variables,variables = build(indices.copy(),categories.copy(),endo,variables) 
       
    if len(eqs) > 0:
        return eqs,eqs_ss,variables,[],eqs_files,None,True
        
    ### Simplified form of yaml file with sets definition
    import re
    from snowdrop.src.model.util import expand
    
    indices = [x.split(" ")[-1].split("(")[0].strip() for x in sets.keys()]
    
    new_sets = {}
    for k in sets:
        arr = list(filter(None,k.split(" ")))
        k1 = k[:-len(arr[-1])].strip()
        indx = arr[-1].strip()
        if "(" in indx and ")" in indx:
            ind1 = indx.index("(")
            ind2 = indx.index(")")
            k2 = indx[1+ind1:ind2].strip()
            k3 = indx[:ind1].strip()
        else:
            k2 = None
            k3 = indx
        if isinstance(sets[k],str) and sets[k] in new_sets:
            new_sets[k3] = sets[sets[k]]
        else:
            new_sets[k3] = sets[k]
        # Check that all elements of map for key=k3 are subset of elements of this map for key=k2
        if not k2 is None:
            diff = set(new_sets[k3]) - set(new_sets[k2])
            if len(diff) > 0:
                diff = ",".join(diff)
                cprint(f"\nMisspecified elements of set '{k1}': extra elements - {diff}.","red")
                sys.exit()
                
    delimiters = " ", ",", ";", "*", "/", ":", "+", "-", "^", "{", "}", "="
    regexPattern = '|'.join(map(re.escape, delimiters))
    
    # Get list of new variables
    index = np.zeros(len(equations),dtype=bool)
    exclude = []
    for i,eq in enumerate(equations):
        if isinstance(eq,dict):
            for k in eq:
                eqtn = eq[k]
            eq = eqtn
        arr = re.split(regexPattern,eq)
        arr = list(filter(None,arr))
        for v in arr:
            v = v.replace(")(",")")
            if v in variables:
                index[i] = True
                for k in new_sets:
                    if "("+k+")" in v:
                        ind = v.index("("+k+")")
                        for x in new_sets[k]:
                            var = v[:ind] + "_" + x
                            if not var in variables:
                                variables.append(var)
                        if not v in exclude:
                            exclude.append(v)
                            
    # Expand ariables
    variables = [v for v in variables if not v in exclude]
    
    # Expand equtions
    equations = [eq for i,eq in enumerate(equations) if index[i] ]
    eqs     = expand(new_sets,indices,equations,loop=True) 
    
    # Expand comments
    cmnts   = [c.replace(" ","_",) for c in comments]
    cmnts   = expand(new_sets,indices,cmnts,loop=True)  
    cmnts   = [c.replace("\n","").replace("_"," ").replace(":",":    ") for c in cmnts]
            
    return eqs,eqs_ss,variables,cmnts,eqs_files,index,False
   
    
def sortIndex(sets,variables):
    """
    Sorts variables by category.
    
    Parameters:
        :param sets: List of categories.
        :type sets: list.
        :param variables: New variables.
        :type variables: list.
        :returns: sorted list.
    """
    import numpy as np 
    
    n = 0
    for k in sets.keys():
        cat = sets[k]
        if isinstance(cat,str):
            n += 1
        else:
            n += len(cat)
        
    nv = len(variables)
    indices = [0]*nv
    lst = []     
    m = n**nv
    for k in sets.keys():
        # Get list of categories for a given set
        cat = sets[k]
        if isinstance(cat,str):
            cat = [cat]  
        for c in cat:
            for i,v in enumerate(variables):
                key = "_" + c
                if key in v: # and not v in lst:
                    indices[i] -= m + i
                    lst.append(v)
                    #print(v,c,m,i,indices[i])
            m /= nv
                
    sort_index = np.argsort(indices) 
    #print(variables)
    #print(indices)            
            
    return sort_index
 
                           
def loadYaml(path,txt):
    """
    Parses a model file, resolves reference to files and parses these files, and builds a model dictionary object.

    Parameters:
        :param path: Path to model files.
        :type path: str.
        :param txt: Text of model file.
        :type txt: str.
        :returns: dict object.
    """
    from snowdrop.src.preprocessor.language import minilang
 
    for C in minilang:
        k = C.__name__
        name = f'!{k}'
        yaml.add_constructor(name, C.constructor)
 
    txt = txt.replace('^', '**')
    
    # Load equations comments
    if version >= '0.18':
        data = yaml.YAML(typ='rt').load(txt)
    else:
        data = yaml.load(txt, Loader=yaml.RoundTripLoader)
    eqs  = data['equations']
    comments = []
    if bool(eqs):
        for idx, obj in enumerate(eqs):
            comment_token = eqs.ca.items.get(idx)
            if not comment_token is None:
                comment = comment_token[0].value.replace("\\n","")
                if bool(comment.strip()):
                    comments.append(comment)
                else:
                    comments.append(' ')
    
    if version >= '0.18':
        import snowdrop.src.preprocessor.language as module
        def my_constructor(loader,node):
            t = node.tag.replace("!","")
            value = loader.construct_mapping(node)
            class_ = getattr(module, t)
            return class_(**value)
        obj = yaml.YAML(typ='safe')
        obj.constructor.add_constructor(tag='!Normal',constructor=my_constructor)
        obj.constructor.add_constructor(tag='!MvNormal',constructor=my_constructor)
        obj.constructor.add_constructor(tag='!LogNormal',constructor=my_constructor)
        obj.constructor.add_constructor(tag='!Beta',constructor=my_constructor)
        obj.constructor.add_constructor(tag='!Binomial',constructor=my_constructor)
        obj.constructor.add_constructor(tag='!Gamma',constructor=my_constructor)
        obj.constructor.add_constructor(tag='!Logistic',constructor=my_constructor)
        obj.constructor.add_constructor(tag='!Uniform',constructor=my_constructor)
        obj.constructor.add_constructor(tag='!Cartesian',constructor=my_constructor)
        data = obj.load(txt)
    else:
        data = yaml.load(txt, Loader=yaml.Loader)
    
    # Symbols section
    symbols = data['symbols']
    sets = symbols.get('sets', {})
    variables = symbols.get('variables',[])
    endo1 = [x for x in variables if "(" in x]
    endo2 = [x for x in variables if not "(" in x]
    
    # Equations section
    equations = data['equations']
    
    new_eqs,new_eqs_ss,new_variables,new_comments,eqs_files,ind,bFormat = buildEquations(sets=sets,equations=equations.copy(),variables=endo1,comments=comments)
    if isinstance(eqs_files,str):
            eqs_files = [eqs_files]
               
    for file in eqs_files:
        if os.path.exists(path + '\\' + file):
            t,labels = read_file_or_url(path + '\\' + file) 
            t = t.replace('^', '**')
            if version >= '0.18':
                sect = yaml.YAML(typ='safe').load(t)
            else:
                sect = yaml.load(t, Loader=yaml.Loader)
            equations = sect['equations']
            new_f_eqs,new_f_eqs_ss,new_f_variables,new_comments,new_eqs_files,ind,bFormat  = buildEquations(sets=sets,equations=equations.copy(),variables=endo1,comments=comments)
            new_eqs += new_f_eqs
            new_eqs_ss += new_f_eqs_ss
            new_variables += new_f_variables
    

    if bFormat:
        if bool(new_variables):
            # Sort variables and equations by country
            index = sortIndex(sets,new_variables)
            new_variables = [new_variables[i] for i in index] 
            new_eqs = [new_eqs[i] for i in index] 
            new_eqs_ss = [new_eqs_ss[i] for i in index] 
            symbols['variables'] += new_variables
            #print(new_variables)
            
            if bool(new_eqs):
                equations += new_eqs
                equations = [e for e in equations if isinstance(e,str)]  
                data['equations'] = equations
    else:
        # Sort variables and equations by country
        index = sortIndex(sets,new_variables)
        new_variables = [new_variables[i] for i in index]  
        new_eqs = [new_eqs[i] for i in index] 
        new_comments = [new_comments[i] for i in index] 
        symbols['variables'] = endo2 + new_variables
        eqtns = [eq for i,eq in enumerate(equations) if not ind[i]]
        data['equations'] = eqtns + new_eqs
        
    if bool(new_eqs_ss):
        data['equations_ss'] = new_eqs_ss 
        
    if len(new_comments) > 0:
        if len(new_comments) < len(data['equations']):
            comments = ['']*(len(data['equations'])-len(new_comments)) + new_comments
        data['eqComments'] = comments
        
    # Shocks section
    shocks = data.pop('shocks', {})
    shocks_files = shocks.pop('file', [])
    if isinstance(shocks_files,str):
        shocks_files = [shocks_files]
                 
    # Parameters section
    params = data.pop('parameters', {})
    params_files = params.pop('file', [])
    if isinstance(params_files,str):
            params_files = [params_files]
    new_params = {}
    for file in params_files:
        new_params = loadFile(path + '\\' + file) 
            
    if 'parameters' in params:
        symbols['parameters'] = params['parameters']          
    if bool(new_params):
        symbols['parameters'] += new_params
    
    # Calibration section
    cal = data.get('calibration', {})
    cal_files = cal.pop('file', [])
    if isinstance(cal_files,str):
            cal_files = [cal_files]
    cal_files += shocks_files
    new_cal = {}
    for file in cal_files:
        fpath = os.path.abspath(os.path.join(path,file))
        new_cal = loadFile(fpath,new_cal)
          
    if bool(new_cal):
        data['calibration'] = {**cal, **new_cal}
        
    return data
                      

def loadFile(path,calibration={},names=[],bShocks=False):
    """
    Parse a file and populate model's calibration dictionary.

    Parameters:
        :param path: Path to file.
        :type path: str.
        :param calibration: Map of variables names and values.
        :type calibration: dict.
        :param names: Names of variables.
        :type names: list.
        :param bShocks: True if loading shocks values file and False otherwise.
        :type bShocks: bool.
        :returns: dictionary object.
    """
    from numpy import log, exp, sin, cos, tan, sqrt
    
    m = {'log':log, 'exp':exp, 'sin':sin, 'cos':cos, 'tan':tan, 'sqrt':sqrt}
 
    if not os.path.exists(path):
        cprint("\nFile: " + path + " does not exist!\nExitting...","red")
        sys.exit()
    
    new_cal = {}; d = {}
    name,ext = os.path.splitext(path)
    
    if ext.lower() == ".yaml":
        txt,labels = read_file_or_url(path)
        if version >= '0.18':
            data = yaml.YAML(typ='safe').load(txt)
        else:
            data = yaml.load(txt, Loader=yaml.Loader)
        for k in data:
            v = data[k]
            if not bool(names):
                try:
                    new_cal[k] = float(v)
                except ValueError:
                    if v in new_cal:
                        new_cal[k] = new_cal[v]
                    else:
                        new_cal[k] = 0
                        d[k] = v
            else:
                if k in names:
                    try:
                        new_cal[k] = float(v)
                    except ValueError:
                        if v in new_cal:
                            new_cal[k] = new_cal[v]
                        else:
                            new_cal[k] = 0
                            d[k] = v
                else:                    
                    try:
                        new_cal[k] = float(v)
                    except ValueError:
                        if v in new_cal:
                            new_cal[k] = new_cal[v]
                        else:
                            new_cal[k] = 0
                    
                                
    elif ext.lower() == ".csv" or ext.lower() == ".xlsx" or ext.lower() == ".xls":
        new_cal = loadTimeSeries(path)
        
    elif ext.lower() == ".txt":
        txt = []
        with open(path,"r") as f:
            count = 0
            for line in f:
                count += 1
                ln = line.strip().rstrip("\n")
                if ln.startswith("#") or len(ln) == 0:
                    continue
                ln = ln.replace(" ","").replace("="," ").replace(":"," ")
                if "#" in ln:
                    ind = ln.index("#")
                    ln = ln[:ind]
                txt.append(ln)
                
        for i,t in enumerate(txt):
            arr = t.split(" ")
            arr = list(filter(None,arr))
            if len(arr) == 2:
                k = arr[0]
                v = arr[1].split(",")
                if not bool(names):
                    try:
                        arr = [float(x) for x in v]
                        if len(arr) == 1:
                            arr = arr[0]
                        new_cal[k] = arr
                    except ValueError:
                        if v in new_cal:
                            new_cal[k] = new_cal[v]
                        else:
                            new_cal[k] = 0
                            d[k] = v
                else:
                    if k in names:
                        try:
                            arr = [float(x) for x in v]
                            if len(arr) == 1:
                                arr = arr[0]
                            new_cal[k] = arr
                        except ValueError:
                            if v in new_cal:
                                new_cal[k] = new_cal[v]
                            else:
                                new_cal[k] = 0
                                d[k] = v
                    else:     
                        new_cal[k] = [0]
       
    elif ext.lower() == ".sqlite":
        from snowdrop.src.utils.db import get_data
        rows,columns = get_data(path)
        if not bool(rows) or columns is None:
            cprint(f"Failed to retrieve data from database, {path}","red")
        data = np.array(rows)    
        for i,col in enumerate(columns):
            new_cal[col] = data[:,i]
        
    newCal =  new_cal.copy()
    for k in d:
        v = d[k]
        x = eval(v,newCal,m) 
        new_cal[k] = x            
     
    if bShocks:
        # Get max length
        Nt = 0; arr = [] 
        for k in new_cal:
            v = new_cal[k]
            Nt = max(Nt,len(v))
        periods = list(range(1,Nt+1))
        for i in range(Nt): 
            values = []
            for k in names:
                if k in new_cal:
                    v = new_cal[k]
                    if i < len(v):
                        values.append(v[i])
                    else:
                        values.append(0)
                else:
                    # cprint(f"Shock name {k} is not defined","red")
                    values.append(0)
            arr.append(values)
            
        return arr,periods
    
    else:
        
        for k in new_cal:
            if isinstance(k,dict):
                calibration = {**calibration, **new_cal[k]}
        calibration = {**calibration, **new_cal}
        
        return calibration
    
                      
def loadTimeSeries(path,dates=None):
    """
    Parse a file and create a time series object.

    Parameters:
        :param path: Path to file.
        :type path: str.
        :param dates: Dates.
        :type dates: date.
        :returns: Time series object.
    """
    name,ext = os.path.splitext(path)
    if ext == ".csv":
        df = pd.read_csv(path,index_col=0,header=0,parse_dates=True)
    elif ext == ".xlsx" or ext == ".xls":
        df = pd.read_excel(path,header=0,index_col=0,parse_dates=True)
    elif ext == ".txt":
        df = pd.read_csv(path,sep=" ",index_col=0,header=0,parse_dates=True)
    
    m = {}
    
    if len(df) == 1:
        row = df.iloc[0]
        for i,col in enumerate(df.columns):
            m[col] = float(row[i])
    else: # For more than one row treat data as time series
        for col in df.columns:
            data = df[col]
            dts = data.index
            vals = data.values.astype('float')
            if dates is None:
                if isinstance(dts,pd.DatetimeIndex):
                    values = vals
                else:    
                    size = max(dts)
                    if type(size) is int:
                        values = np.zeros(size)
                        for dt,v in zip(dts,vals):
                            values[dt-1:] = v
            else:
                series = pd.Series(data=0,index=dates)
                for j in range(len(dts)-1):
                    d1 = dts[j]
                    d2 = dts[1+j]
                    v = vals[j]
                    for d in dates:  
                        if d >= d1 and d < d2:
                            series[d:] = v
                            break
                values = series.values
            
            arr = []
            for v in values:
                arr.append(float(v))
                    
            m[col] = arr
            
    return m


def getCalibration(fpath,names=None):
    """
    Parses a model file, and builds a model calibration dictionary object.

    Parameters:
        :param fpath: Path to model files.
        :type fpath: str.
        :param names: Endogenous variables names.
        :type names: str.
        :returns: dict object.
    """
    from numpy import log, exp, sin, cos, tan, sqrt
    
    m = {'log':log, 'exp':exp, 'sin':sin, 'cos':cos, 'tan':tan, 'sqrt':sqrt}
    
    txt,labels = read_file_or_url(fpath)
    if version >= '0.18':
        data = yaml.YAML(typ='safe').load(txt)
    else:
        data = yaml.load(txt, Loader=yaml.Loader)
    data = data["calibration"]
    
    cal = dict(); d = dict()
    for k in data:
        v = data[k]
        if not bool(names):
            try:
                cal[k] = float(v)
            except ValueError:
                d[k] = v
                cal[k] = 0
        else:
            if k in names:
                try:
                    cal[k] = float(v)
                except ValueError:
                    d[k] = v
                    cal[k] = 0
            else:                    
                try:
                    cal[k] = float(v)
                except ValueError:
                    cal[k] = 0
                    d[k] = v
      
    for k in d:
        v = d[k]
        x = eval(v,cal.copy(),m) 
        cal[k] = x  
        
    return cal

    
if __name__ == '__main__':
    """
    The main program.
    """
    path = os.path.dirname(os.path.abspath(__file__))
    working_dir = path+"\\.."
    
    # fname = 'Test.yaml'
    # file_path = os.path.abspath(os.path.join(working_dir,'../templates/'+fname))
    
    # with open(file_path, 'r') as fin:
    #     txt = fin.read()
    #     loadYaml(txt)
      
        
    fname = 'sol.sqlite'
    file_path = os.path.abspath(os.path.join(working_dir,'../data/db/'+fname))
    m = loadFile(file_path)
    print(m)
