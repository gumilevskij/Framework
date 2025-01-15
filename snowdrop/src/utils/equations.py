# -*- coding: utf-8 -*-
"""
Created on Wed May 16 16:30:50 2018
@author: agoumilevski
"""
import os
import re
import numpy as np
import sympy as sy  
from collections import OrderedDict

path = os.path.dirname(os.path.abspath(__file__))
working_dir = os.path.abspath(path + "/../../..")
os.chdir(working_dir)

from snowdrop.src.misc.termcolor import cprint
    
mp1 = None; mp2 = None
        
delimiters = " ", ",", ";", "*", "/", ":", "+", "-", "^", "{", "}", "(", ")", "="
regexPattern = '|'.join(map(re.escape, delimiters))
      

def findVar(txt,line,shocks=[]):
    """Parse text string and returns name, value of a variable."""
    if not "//DO " in line and not "*/" in line:
        #arr = txt.split(' ')
        arr = re.split(' |;|,',txt)
        for i in range(len(arr)):
            if len(arr[i]) > 0:
                ind1 = line.find(arr[i],0)
                ind2 = line.find(';',ind1)
                if ind1 >= 0 and ind2 >= 0:
                    var = line[ind1:ind2]
                    j = var.find('=')
                    if j >= 0:
                        name = var[:j].strip()
                        value = var[1+j:].strip()
                        isShock = name in shocks
                        if name == arr[i]:
                            var = var.replace('dzero','0')
                            var = ' '.join(var.split()) 
                            var = var + '\n'
                            return name,value,isShock
    return None,None,False
    
    
def replaceEq(eq,arr,var,old_var,new_var):
    """
    Replace variables in equation.
    
    Parameters:
        :param eq: Equation.
        :type eq: str.
        :param arr: Symbols of an equation.
        :type arr: list.
        :param var: Variable name.
        :type var: str.
        :param old_var: Old variable name.
        :type old_var: str.
        :param new_var: New variable name.
        :type new_var: str.
        :return: New equation.
    """
    new_eq = "" #eq.replace(old_var,new_var) 
    ind1 = 0; beg = 0
    for v in arr:
        ind  = eq.index(v,ind1)
        ind1 = ind+len(v)
        if ")" in eq[ind:]:
            ind2 = 1+eq.index(")",ind1)
            if ind2 <= len(eq) and eq[ind:ind2] == old_var:
                end = ind
                new_eq += eq[beg:end] + new_var
                beg = end + len(old_var)
        
    if beg == 0:
        new_eq = eq
    else:
        new_eq += eq[beg:]
            
    return new_eq


# def _fixEquation(eq):
#     """
#     Replace exponent ** operator with power() function.
    
#     https://stackoverflow.com/questions/37077052/changing-operator-to-power-function-using-parsing
    
#     Parameters:
#         :param eq: Equation.
#         :type eq: str.
#     """
#     from pyparsing import Regex,Word,Forward,Combine,infixNotation
#     from pyparsing import Optional,opAssoc,oneOf
#     from pyparsing import alphas,alphanums
    
#     # define some basic operand expressions
#     number = Regex(r'\d+(\.\d*)?([Ee][+-]?\d+)?')
#     ident = Word(alphas+'_', alphanums+'_')
    
#     # forward declare our overall expression, since a slice could 
#     # contain an arithmetic expression
#     expr = Forward()
#     slice_ref = '[' + expr + ']'
    
#     # define our arithmetic operand
#     operand = number | Combine(ident + Optional(slice_ref))
    
#     # parse actions to convert parsed items
#     def convert_to_pow(tokens):
#         tmp = tokens[0][:]
#         ret = tmp.pop(-1)
#         tmp.pop(-1)
#         while tmp:
#             base = tmp.pop(-1)
#             # hack to handle '**' precedence ahead of '-'
#             if base.startswith('-'):
#                 ret = '-pow(%s,%s)' % (base[1:], ret)
#             else:
#                 ret = 'pow(%s,%s)' % (base, ret)
#             if tmp:
#                 tmp.pop(-1)
#         return ret
    
#     def unary_as_is(tokens):
#         return '(%s)' % ''.join(tokens[0])
    
#     def as_is(tokens):
#         return '%s' % ''.join(tokens[0])
    
#     # simplest infixNotation - may need to add a few more operators, but start with this for now
#     arith_expr = infixNotation( operand,
#         [
#         ('-', 1, opAssoc.RIGHT, as_is),
#         ('**', 2, opAssoc.LEFT, convert_to_pow),
#         ('-', 1, opAssoc.RIGHT, unary_as_is),
#         (oneOf("* /"), 2, opAssoc.LEFT, as_is),
#         (oneOf("+ -"), 2, opAssoc.LEFT, as_is),
#         ])
    
#     # now assign into forward-declared expr
#     expr <<= arith_expr.setParseAction(lambda t: '(%s)' % ''.join(t))
    
#     eq = expr.transformString(eq)[1:-1]
#     return eq

def fixEquation(eq):
    """
    Replace exponent ** operator with power() function.
    
    Parameters:
        :param eq: Equation.
        :type eq: str.
    """
    if not "**" in eq:
        return eq
    else:
        while ("**" in eq):
            eq = fixExpression(eq)
        return eq
    
    
def fixExpression(eq):
    """
    Replace exponent ** operator with power() function.
    
    Parameters:
        :param eq: Equation.
        :type eq: str.
    """
    import copy
    from ast import Pow,Name,parse,copy_location
    from ast import Call,NodeTransformer,Load
    from snowdrop.src.preprocessor.codegen import to_source  

    class ExponentTransformer(NodeTransformer):
        def visit_BinOp(self, node):
            if isinstance(node.op, Pow):
                new_node = Call(func=Name(id='pow', ctx=Load()),
                                args=[node.left, node.right],
                                keywords = [],
                                starargs = None, kwargs= None
                                )
                copy_location(new_node, node)
                return new_node
            else:
                self.generic_visit(node)
            return node       

    if  "**" in eq:
        cp = copy.deepcopy(eq)
        syntax_tree = parse(cp)
        et = ExponentTransformer()
        expr = et.visit(syntax_tree)
        eq = to_source(expr)
            
    return eq
      
    
def fixEquations(eqs,endog,params=None,tagBeg="(",tagEnd=")",b=False):
    """
    Replace curly brackets in lead/lag variables expressions.
    
    Adds new equations and new variables if leads/lags greater than one.
    
    Parameters:
        :param eqs: List of equations.None
        :type eqs: list.
        :param endog: List of engogenous variables.
        :type endog: list.
        :param params: List of parameters names.
        :type params: list.
        :param tagBeg: The beginning tag of lead/lag variable.
        :type tagBeg: str.
        :param tagEnd: The ending tag of lead/lag variable.
        :type tagEnd: str.
        :param b: Flag.  If False it converts equations to a form that have variable 
                  with leads and lags of no more than one.
               If True it converts equations to a form that have variable 
                  with leads of no more than one and no lags.
        :type b: Bool.
    """
    if b:
        eqs = modifyEquations(eqs=eqs,variables=endog)
        

    ex_leads = []; ex_lags = []; leads = {}; lags = {}
    eq_leads = {}; eq_lags = {}; par_leads = {}; par_lags = {}
    mod_eqs = []; new_endog = []; mod_endog = {}
    
    for i,eq in enumerate(eqs):
        eq = eq.replace(" ","");
        ind = -1
        arr = re.split(regexPattern,eq)
        arr = list(filter(None,arr))
        for v in arr:
            ind = max(1+ind,eq.find(v,1+ind))
            if v in endog and not eq.find(v+tagBeg,ind) == -1:
                ind1 = max(ind,eq.find(v+tagBeg,ind)) + len(v)
                ind2 = max(ind1,eq.find(tagEnd,ind1))
                if ind1 >= 0 and ind2 > ind1:
                    lead_lag = eq[1+ind1:ind2].replace(" ","")
                    ex_var = v + tagBeg + lead_lag + tagEnd
                    match = re.match(r"[-+]?\d+",lead_lag)
                    if not match is None:
                        try:
                            i_lead_lag = int(lead_lag)
                            if i_lead_lag > 1:
                                mod_var = v + "_plus_" + str(i_lead_lag-1)
                                new_endog.append(mod_var)
                                mod_endog[ex_var] = mod_var
                            elif i_lead_lag < -1: 
                                mod_var = v + "_minus_" + str(-i_lead_lag-1)
                                new_endog.append(mod_var)
                                mod_endog[ex_var] = mod_var
                            if i_lead_lag == 1:
                                ex_leads.append(v)
                            if i_lead_lag == -1:
                                ex_lags.append(v)
                            if i_lead_lag > 1:
                                if i in eq_leads:
                                    eq_leads[i].append([v,i_lead_lag])
                                else:
                                    eq_leads[i] = [[v,i_lead_lag]] 
                                if v in leads.keys():
                                    leads[v] = max(leads[v],i_lead_lag)
                                else:
                                    leads[v] = i_lead_lag
                            if i_lead_lag < -1:
                                if i in eq_lags:
                                    eq_lags[i].append([v,i_lead_lag])
                                else:
                                    eq_lags[i] = [[v,i_lead_lag]]
                                if v in lags.keys():  
                                    lags[v] = min(i_lead_lag,lags[v])
                                else:
                                    lags[v] = i_lead_lag
                        except:
                            pass
                        
            if v in params:
                ind1 = max(ind,eq.find(v+tagBeg,ind)) + len(v)
                ind2 = max(ind1,eq.find(tagEnd,ind1))
                if ind1 >= 0 and ind2 > ind1:
                    lead_lag = eq[1+ind1:ind2].replace(" ","")
                    match = re.match(r"[-+]?\d+",lead_lag)
                    if not match is None:
                        try:
                            i_lead_lag = int(lead_lag)
                            if i_lead_lag > 0:
                                if i in par_leads:
                                    par_leads[i].append([v,i_lead_lag])
                                else:
                                    par_leads[i] = [[v,i_lead_lag]] 
                            if i_lead_lag < 0:
                                if i in par_lags:
                                    par_lags[i].append([v,i_lead_lag])
                                else:
                                    par_lags[i] = [[v,i_lead_lag]]
                        except:
                            pass            
     
    for i,eq in enumerate(eqs):
        eq = eq.replace(" ","");   
        
        # Replace lead/lag variables in equations
        if i in eq_leads:
            for x in eq_leads[i]:
                k,j = x
                new_var = k + "_plus_" + str(j-1) + "(+1)"
                old_var = k + "(+" + str(j) + ")"
                arr = re.split(regexPattern,eq)
                arr = list(filter(None,arr))
                eq = replaceEq(eq,arr,var=k,old_var=old_var,new_var=new_var)
                old_var = k + "(" + str(j) + ")"  
                #eq = eq.replace(old_var,new_var) 
                arr = re.split(regexPattern,eq)
                arr = list(filter(None,arr))
                eq = replaceEq(eq,arr,var=k,old_var=old_var,new_var=new_var)
        if i in eq_lags:
            for x in eq_lags[i]:
                k,j = x  
                new_var = k + "_minus_" + str(-j-1) + "(-1)"
                old_var = k + "(-" + str(-j) + ")"
                arr = re.split(regexPattern,eq)
                arr = list(filter(None,arr))
                eq = replaceEq(eq,arr,var=k,old_var=old_var,new_var=new_var)
        
        # Make sure parameters don't have leads/lags
        if i in par_leads:
            for x in par_leads[i]:
                k,j = x
                new_var = k
                old_var = k + "(+" + str(j) + ")"
                arr = re.split(regexPattern,eq)
                arr = list(filter(None,arr))
                eq = replaceEq(eq,arr,var=k,old_var=old_var,new_var=new_var)
                old_var = k + "(" + str(j) + ")"  
                #eq = eq.replace(old_var,new_var) 
                arr = re.split(regexPattern,eq)
                arr = list(filter(None,arr))
                eq = replaceEq(eq,arr,var=k,old_var=old_var,new_var=new_var)
        if i in par_lags:
            for x in par_lags[i]:
                k,j = x  
                new_var = k
                old_var = k + "(-" + str(-j) + ")"
                arr = re.split(regexPattern,eq)
                arr = list(filter(None,arr))
                eq = replaceEq(eq,arr,var=k,old_var=old_var,new_var=new_var)
            
        mod_eqs.append(eq)

    map_new_endog = {}    
    # Add new variables
    for k,v in leads.items():
        for i in range(1,v):
            var = k + "_plus_" + str(i)
            new_endog.append(var)
            if not k in map_new_endog.keys():
                map_new_endog[k] = []
            map_new_endog[k].append(var)
            
    for k,v in lags.items():
        for i in range(1,-v):
            var = k + "_minus_" + str(i)
            new_endog.append(var)
            if not k in map_new_endog.keys():
                map_new_endog[k] = []
            map_new_endog[k].append(var)
    
    new_endog = sorted(list(set(new_endog)),key=str.lower)
                    
    # Add new equations
    new_eqs = []
    for k,v in leads.items():
        for i in range(v-1):
            if i == 0:
                var1 = k + "(+1)"
                var2 = k + "_plus_1"
            else:
                var1 = k + "_plus_" + str(i) + "(+1)"
                var2 = k + "_plus_" + str(1+i)
            new_eqs.append(var2 + " = " + var1)
                
    for k,v in lags.items():
        for i in range(-v-1):
            if i == 0:
                var1 = k + "(-1)"
                var2 = k + "_minus_1"
            else:
                var1 = k + "_minus_" + str(i) + "(-1)"
                var2 = k + "_minus_" + str(1+i)
            new_eqs.append(var2 + " = " + var1)
    
    new_eqs = sorted(new_eqs,key=str.lower)
    mod_eqs = mod_eqs + new_eqs 
            
    return mod_eqs, new_endog, map_new_endog, leads, lags


def modifyEquations(eqs,variables):
    """
    Modify equations so that the new equations do not have lags, but only leads variables.
    
    Parameters:
        :param eqs: List of equations.
        :type eqs: list.
        :param eqs: List of variables.
        :type eqs: list.
        :return: New equations.
    """
    delimiters = " ", ",", ";", "*", "/", ":", "+", "-", "^", "{", "}", "(", ")", "="
    regexPattern = '|'.join(map(re.escape, delimiters))
    delimiters_ = " ", ",", ";", "*", "/", ":", "+", "-", "^" , "="
    regexPattern_ = '|'.join(map(re.escape, delimiters_))
    
    mod_eqs = []
    for eq in eqs:
        j = 0
        eq_ = eq.replace(" ","").replace("(1)","(+1)")
        arr = re.split(regexPattern_,eq_)
        arr = list(filter(None,arr))
        max_lead,min_lag,var_lead,var_lag,mp = getMaxLeadAndLag(eq=eq_,arr=arr,variables=variables,var_lead={},var_lag={})
        if min_lag == 0:
            mod_eqs.append(eq)
        else:
            new_eq = eq.replace(" ","")
            size = len(new_eq)
            arr = re.split(regexPattern,new_eq)
            arr = list(filter(None,arr))
            match = re.findall(r"\((\s?[+-]?\d+\s?)\)",new_eq)
            for e in match:
                n = int(e.strip())
                m = n - min_lag
                for v in variables:
                    old_var = v + "(" + e + ")"
                    new_var = v + "(" + str(m) + ")"
                    # Replace lead/lag variables 
                    if old_var in new_eq:
                        arr = re.split(regexPattern,new_eq)
                        arr = list(filter(None,arr))
                        new_eq = replaceEq(new_eq,arr,v,old_var,new_var)
                        
            # Replace variables that are not leads/lags variables
            for v in arr:  
                if v in variables:
                    i = new_eq.find(v,j)
                    j = i + len(v)
                    if j <= size-1 and new_eq[j] != "(":
                        new_eq = new_eq[:j] + "(" + str(-min_lag) + ")" + new_eq[j:]
                        
            # Replace variables that are not leads/lags variables
            for v in variables:
                if v in eq:
                    new_eq = new_eq.replace(v+"(0)",v)
                        
            mod_eqs.append(new_eq)
            
    return mod_eqs


def getMaxLeadAndLag(eq,arr,variables,var_lead={},var_lag={},m={},max_lead=0,min_lag=0,tagBeg=None,tagEnd=None):
    """
    Return leads and lags and a map of leads and lags of variables in equation.
    
    Parameters:
        :param eq: Equation.
        :type eq: str.
        :param var_lead: Lead variables.
        :type var_lead: Dictionary.
        :param var_lag: Lag variables.
        :type var_lag: Dictionary.
        :param m: Dictionary of variables frequencies in equations.
        :type m: Dictionary.
        :param max_lead: Initial max lead.
        :type max_lead: int.
        :param min_lag: Initial min lag.
        :type min_lag: int.
        :param variables: List of variables.
        :type var_lag: list.
        :return: Max lead and minimum lag of variables in equation.
    
    """
    n = 0
    if tagBeg is None or tagEnd is None:
        tagBeg = "("
        tagEnd = ")"
        
    for v in arr:
        if v in variables:
            ind = eq.find(v)
            eq = eq[ind:]
            if eq.startswith(v+tagBeg):
                ind1 = len(v)
                eq = eq[1+ind1:]
                ind2 = eq.find(tagEnd)
                e = eq[:ind2]
                eq = eq[1+ind2:]
                match = re.match(r"[-+]?\d+",e)
                if not match is None:
                    var = v + "(" + e + ")"
                    n = int(e)
                    if v in m:
                        m[v] += 1
                    else:
                        m[v] = 1
                    eq = eq.replace(var,"",1)
                    if n > 0:
                        if var in var_lead.keys():
                            var_lead[var] += 1
                        else:
                            var_lead[var] = 1
                    elif n < 0:
                        if var in var_lag.keys():
                            var_lag[var] += 1
                        else:
                            var_lag[var] = 1
            else:
                eq = eq[len(v):]
        else:
            eq = eq[len(v):]
            
        max_lead = max(max_lead,n)
        min_lag = min(min_lag,n)
            
    return max_lead,min_lag,var_lead,var_lag,m


def getMaxLeadsLags(eqs,variables):
    """
    Return maximum leads and minimum lags of variables in equations.
    
    Parameters:
        :param eqs: Model equations.
        :type eqs: list.
        :param variables: List of variables.
        :type variables: list.
        :return: Maximum leads and minimum lags of equations
    
    """
    var_lead = {}
    var_lag = {}
    m = {}
    max_lead = 0
    min_lag = 0
    
    delimiters = " ", ",", ";", "*", "/", ":", "+", "-", "^" , "=", "(", ")"
    regexPattern = '|'.join(map(re.escape, delimiters))
    
    for eq in eqs:  
        eq_ = eq.replace(" ","").replace("(1)","(+1)")
        arr = re.split(regexPattern,eq_)
        arr = list(filter(None,arr))
        max_lead,min_lag,var_lead,var_lag,m = getMaxLeadAndLag(eq_,arr,variables,var_lead,var_lag,m,max_lead,min_lag)
      
    np_eqs = []    
    np_var = [k for k in m.keys() if m[k] == 1]
    for eq in eqs:
        if '=' in eq:
            ind = eq.find('=')
            left = eq[:ind].strip()
            right = eq[1+ind:].strip()
            if left in np_var and right in var_lead.keys():
                np_eqs.append(eq)
        
    for k in var_lead.keys():
        for eq in np_eqs:
            if k in eq:
               var_lead[k] -= 1
               
    # n_fwd_looking_var = 0
    # for k in var_lead:
    #     v = var_lead[k]
    #     ind1 = k.find('(')
    #     ind2 = k.find(')')
    #     n = k[1+ind1:ind2]
    #     if v > 0:
    #         n_fwd_looking_var += 1
            
    # n_bkwd_looking_var = 0
    # for k in var_lag:
    #     v = var_lag[k]
    #     ind1 = k.find('(')
    #     ind2 = k.find(')')
    #     n = k[1+ind1:ind2]
    #     if v > 0:
    #         n_bkwd_looking_var += 1
            
    n_fwd_looking_var = len(var_lead)
    n_bkwd_looking_var = len(var_lag)
        
    return max_lead,min_lag,n_fwd_looking_var,n_bkwd_looking_var,var_lead,var_lag
    

def getStateVariables(eqs,variables,shocks=None,min_lag=None):
    """
    Return state variables and equations ordering numbers that these state variables belong to.
    
    State variables are variables that appear in equaions at the previous period and 
    possibly at the current period.
    
    Parameters:
        :param eqs: Model equations.
        :type eqs: list.
        :param variables: List of variables.
        :type variables: list.
        :param shocks: List of shock variables.
        :type shocks: list.
        :param min_lag: Maximum lag of endogenous variables.
        :type min_lag: int.
        :return: State variables and equations numbers.
    
    """
    state_vars = []
    eqs_number = []
    delimiters = " ", ",", ";", "*", "/", ":", "+", "-", "^", "{", "}", "(", ")", "="
    regexPattern = '|'.join(map(re.escape, delimiters))
    delimiters_ = " ", ",", ";", "*", "/", ":", "+", "-", "^" , "="
    regexPattern_ = '|'.join(map(re.escape, delimiters_))
    
        
    for i,eq in enumerate(eqs):
        # Get lagged variables 
        eq_ = eq.replace(" ","").replace("(1)","(+1)")
        arr = re.split(regexPattern_,eq_)
        arr = list(filter(None,arr))
        lead_var,lag_var = getMaxLeadAndLag(eq_,arr,variables,var_lead={},var_lag={})[2:4]
        lead_variables = list(lead_var.keys())
        lag_variables = list(lag_var.keys())
        for v in lead_variables:
            eq = eq.replace(v,"")
        for v in lag_variables:
            eq = eq.replace(v,"")
        arr = re.split(regexPattern,eq)
        arr = list(filter(None,arr))
        b = True
        if b:
            #print(1+i,lag_variables)
            for v in lag_variables:
                if not min_lag is None or "("+str(min_lag)+")" in v:
                    ind = v.index("(")
                    v = v[:ind]
                    if v in arr:
                        if not v in state_vars:
                            state_vars.append(v)
                        if not i in eqs_number:
                            eqs_number.append(i)
                   
    return state_vars, eqs_number
    

def getVariablesRowsAndColumns(lead_lags,rows,columns,include=None,exclude=None):
    """
    Return equations rows and columns of variables.
    
    Parameters:
        :param lead_lags: Variables leads and lags.
        :type lead_lags: list.
        :param rows: List of rows.
        :type rows: list.
        :param columns: List of columns.
        :type columns: list.
        :param include: Comma separated list of variables to include.
        :type include: str.
        :param exclude: Comma separated list of variables to exclude.
        :type exclude: str.
    """
    Rows = []; Columns = []
    if include is None:
        includeList = None
    elif ',' in include:
        includeList = include.split(',')
    else:
        includeList = [include]
    if exclude is None:
        excludeList = None
    elif ',' in exclude:
        excludeList = exclude.split(',')
    else:
        excludeList = [exclude]
    
    for row in np.unique(rows):
        LL = [lead_lags[i] for i,x in enumerate(rows) if x==row]
        R = [rows[i] for i,x in enumerate(rows) if x==row]
        C = [columns[i] for i,x in enumerate(rows) if x==row]
        if include is None and exclude is None:
            Rows.extend(R)
            Columns.extend(C)
        if not include is None and exclude is None:
            ind = [i for i,x in enumerate(LL) if x in includeList]
            if len(ind) > 0:
                Rows.extend([R[i] for i in ind])
                Columns.extend([C[i] for i in ind])
        if include is None and not exclude is None:
            ind = [i for i,x in enumerate(LL) if not x in excludeList]
            if len(ind) > 0:
                Rows.extend([R[i] for i in ind])
                Columns.extend([C[i] for i in ind])
        elif not include is None and not exclude is None:
            ind = [i for i,x in enumerate(LL) if x in includeList and not x in excludeList]
            if len(ind) > 0:
                Rows.extend([R[i] for i in ind])
                Columns.extend([C[i] for i in ind])
                
    return np.unique(Rows),np.unique(np.sort(Columns))
    

def getIncidenceMap(endog,model=None,eqs=None,tagBeg="(",tagEnd=")"):
    """
    Get maps of variables rows and columns.
    
    Parameters:
        :param model: Model.
        :type model: `Model` object.
        :param endog: List of endogenous variables.
        :type endog: list.
        :param eqs: List of equations.
        :type eqs: list.
        :param tagBeg: The beginning tag of lead/lag variable.
        :type tagBeg: str.
        :param tagEnd: The ending tag of lead/lag variable.
        :type tagEnd: str.
    """
    if not model is None:
        eqs = model.functions_src["f_dynamic_txt"]
        eqs = eqs.replace("__m1_","(-1)")
        eqs = eqs.replace("__p1_","(1)")
        eqs = eqs.replace("__","(0)")
        eqs = eqs.replace(" ","")
        eqs = eqs.split("\n")
        
    delimiters = " ", ",", ";", "*", "/", ":", "+", "-", "^", "{", "}", "(", ")"
    regexPattern = '|'.join(map(re.escape, delimiters))
      
    m = OrderedDict()
    n = len(eqs)
            
    for row,eqtn in enumerate(eqs):
        if "=" in eqtn:
            i = eqtn.index("=")
            eq = eqtn[1+i:].replace(" ","")
        else:
            eq = eqtn.replace(" ","")
        ind = -1
        arr = re.split(regexPattern,eq)
        arr = list(filter(None,arr))
        for v in arr:
            match = re.match(r"[-+]?\d+",v) # skip integer tokens
            if match is None:
                if v in endog:
                    ind = eq.find(v)
                    eq  = eq[ind:]
                    col = endog.index(v)
                    if eq.startswith(v+tagBeg):
                        ind1 = eq.find(v+tagBeg)
                        ind2 = eq.find(tagEnd,ind1)
                        if ind1 >= 0 and ind2 > ind1:
                            lead_lag = eq[1+len(v)+ind1:ind2]
                            match = re.match(r"[-+]?\d+",lead_lag)
                            if not match is None:
                                try:
                                    i_lead_lag = int(lead_lag)
                                    if i_lead_lag == 0:
                                        col += n
                                    elif i_lead_lag == -1:
                                        col += 2*n
                                    val = (i_lead_lag,row,col)
                                    if not v in m:
                                        m[v] = []
                                    m[v].append(val)
                                except:
                                    pass
                        else:
                            val = (0,row,col)
                            if not v in m:
                                m[v] = []
                            m[v].append(val) 
                    else:
                        val = (0,row,col)
                        if not v in m:
                            m[v] = []
                        m[v].append(val)
            eq = eq[len(v):]
            
    return m
    
    
def getTopology(model,tagBeg="(",tagEnd=")"):
    """
    Get maps of variables rows and columns.
    
    Parameters:
        :param model: The Model object.
        :type model: instance of class Model.
        :param tagBeg: The beginning tag of lead/lag variable.
        :type tagBeg: str.
        :param tagEnd: The ending tag of lead/lag variable.
        :type tagEnd: str.
    """
    if not model.Topology is None:
        mp = model.Topology
    else:    
        mp = {}
        endog = model.symbols["variables"]
        m = getIncidenceMap(endog=endog,model=model,tagBeg=tagBeg,tagEnd=tagEnd)
        
        for k in m:
            vals = set(m[k])
            lst = []
            for v in vals:
                if v[0] > 0:
                    lst.append("+{0}:{1},{2}".format(v[0],v[1],v[2]))
                else:
                    lst.append("{0}:{1},{2}".format(v[0],v[1],v[2]))
            mp[k] = lst    
            
        model.Topology = mp
        
    return mp


def getLeadLagIncidence(model):
    """
    Build lead-lag incidence matrix.
    
    This matrix has three rows and variable number of columns (it is the same as number variables).
    The first row lists positions of variables in the model equations for lag variables, the second -
    position of current variables, and the third - position of lead variables.

    Parameters:
        :param model: The Model object.
        :type model: instance of class Model.
    """
    if model.lead_lag_incidence is None:
        variables = model.symbols["variables"]
        n = len(variables)
        lead_lag_incidence = np.zeros(shape=(3,n))
        lead_lag_incidence[:] = np.nan
        
        mp = getTopology(model) 
        for k in mp:
            col = variables.index(k)
            v = mp[k]
            for e in v:
                ind = e.index(':')
                lead_lag = int(e[:ind])
                row = lead_lag + 1
                e = e[1+ind:]
                ind = e.index(',')
                var_col = int(e[1+ind:])
                lead_lag_incidence[row,col] = var_col
      
        model.lead_lag_incidence = lead_lag_incidence
        
    return model.lead_lag_incidence
    

def getVarRowsIncidence(model):
    """
    Return variables rows incidence.

    Parameters:
        :param model: The Model object.
        :type model: instance of class Model.
    """
    row_incidence = {}
    
    mp = getTopology(model) 
    n = len(mp)
    for k in mp:
        v = mp[k]
        for e in v:
            ind = e.index(':')
            e = e[1+ind:]
            ind = e.index(',')
            var_row = int(e[:ind])
            var_col = int(e[1+ind:]) % n
            if var_col in row_incidence:
                rows = row_incidence[var_col] + [var_row]
                row_incidence[var_col] = sorted(list(set(rows)))
            else:
                row_incidence[var_col] = [var_row]
  
    model.var_rows_incidence = row_incidence
    return row_incidence


def getTopologyOfVariables(model,tagBeg="(",tagEnd=")"):
    """
    Define different types of endogenous variables.
    
    This function defines topology of variables described in Sebastien Villemot's paper:
    "Solving rational expectations model at first order: What Dynare does"
    Please see https://www.dynare.org/wp-repo/dynarewp002.pdf
    
    Parameters:
        :param model: The Model object.
        :type model: instance of class Model.
        :param tagBeg: The beginning tag of lead/lag variable.
        :type tagBeg: str.
        :param tagEnd: The ending tag of lead/lag variable.
        :type tagEnd: str.
    """
    state = {}; static = {}; mixed = {}; dynamic = {}
    backward = {}; purely_backward = {}; forward = {}; purely_forward = {}
        
    mp = getTopology(model,tagBeg,tagEnd) 
        
    i = -1
    for k in mp:
        i += 1
        v = mp[k]
        vals = ','.join(v)
        lead_lags = []; rows = []; columns = []
        for e in v:
            ind = e.index(':')
            lead_lags.append(e[:ind])
            e = e[1+ind:]
            ind = e.index(',')
            rows.append(int(e[:ind]))
            columns.append(int(e[1+ind:]))
        
        if model.max_lead > 0 and model.min_lag < 0:
            if '-1:' in vals and not '+1:' in vals:
                purely_backward[k] = getVariablesRowsAndColumns(lead_lags=lead_lags,rows=rows,columns=columns,include='-1,0')
            if '-1:' in vals and '0:' in vals:
                backward[k] = getVariablesRowsAndColumns(lead_lags=lead_lags,rows=rows,columns=columns,include='-1,0',exclude='+1')
            if '+1:' in vals and not '-1:' in vals:
                purely_forward[k] = getVariablesRowsAndColumns(lead_lags=lead_lags,rows=rows,columns=columns,include='+1,0')
            if '+1:' in vals and '0:' in vals:
                forward[k] = getVariablesRowsAndColumns(lead_lags=lead_lags,rows=rows,columns=columns,include='+1,0',exclude='-1')
            if '+1:' in vals and '-1:' in vals:
                mixed[k] = getVariablesRowsAndColumns(lead_lags=lead_lags,rows=rows,columns=columns,include='+1,0,-1')
            if '-1:' in vals and '0:' in vals:
                state[k] = getVariablesRowsAndColumns(lead_lags=lead_lags,rows=rows,columns=columns,include='-1,0')
            if '0:' in vals and not '+1:' in vals and not '-1:' in vals:
                static[k] = getVariablesRowsAndColumns(lead_lags=lead_lags,rows=rows,columns=columns,include='0')
            else:
                dynamic[k] = getVariablesRowsAndColumns(lead_lags=lead_lags,rows=rows,columns=columns)
                
        elif model.max_lead == 0 and model.min_lag < 0:
            purely_forward[k] = []; forward[k] = []
            if '-1:' in vals and not '0:' in vals:
                purely_backward[k] = getVariablesRowsAndColumns(lead_lags=lead_lags,rows=rows,columns=columns,include='-1')
            if '-1:' in vals:
                backward[k] = getVariablesRowsAndColumns(lead_lags=lead_lags,rows=rows,columns=columns,include='-1,0')
            if '-1:' in vals and '0:' in vals:
                state[k] = mixed[k] = getVariablesRowsAndColumns(lead_lags=lead_lags,rows=rows,columns=columns,include='-1,0')
            if '0:' in vals and not '-1:' in vals:
                static[k] = getVariablesRowsAndColumns(lead_lags=lead_lags,rows=rows,columns=columns,include='0',exclude='-1')
            else:
                dynamic[k] = getVariablesRowsAndColumns(lead_lags=lead_lags,rows=rows,columns=columns,include='-1,0')
                
        elif model.max_lead > 0 and model.min_lag == 0:
            purely_backward[k] = []; backward[k] = []
            if '+1:' in vals and not '0:' in vals:
                purely_forward[k] = getVariablesRowsAndColumns(lead_lags=lead_lags,rows=rows,columns=columns,include='+1',exclude='0')
            if '+1:' in vals:
                forward[k] = getVariablesRowsAndColumns(lead_lags=lead_lags,rows=rows,columns=columns,include='+1,0')
            if '+1:' in vals and '0:' in vals:
                state[k] = mixed[k] = getVariablesRowsAndColumns(lead_lags=lead_lags,rows=rows,columns=columns,include='+1,0')
            if '0:' in vals and not '+1:' in vals:
                static[k] = getVariablesRowsAndColumns(lead_lags=lead_lags,rows=rows,columns=columns,include='0')
            else:
                dynamic[k] = getVariablesRowsAndColumns(lead_lags=lead_lags,rows=rows,columns=columns,include='+1,0')
                
        elif model.max_lead == 0 and model.min_lag == 0:
            if i==1: cprint("Model does not have leads or lags... Skipping variables topology definition!","green")
            purely_backward[k] = None; backward[k] = None; purely_forward[k]  = None; forward[k] = None
            mixed[k] = None; state[k] = None; static[k] = None; dynamic[k]  = None
                
    backward = {**purely_backward, **mixed}
    #forward = {**purely_forward, **mixed}         
          
    model.topology = {}
    model.topology["state"] = state
    model.topology["static"] = static
    model.topology["mixed"] = mixed
    model.topology["dynamic"] = dynamic
    model.topology["backward"] = backward
    model.topology["purely_backward"] = purely_backward
    model.topology["forward"] = forward
    model.topology["purely_forward"] = purely_forward
    
    return model
    

def getLeadVariables(m,endog):
    """
    Get a list of lead endogenous variables.
    
    Parameters:
        :param m: Map of variables names and row and column position in Jacobian matrix.
        :type m: dict.
        :param endog: List of endogenous variables.
        :type endog: list.
        :return: List of lead endogenous variables.
    """

    lst = []
    for k in m:
        vals = set(m[k])
        for v in vals:
            if v[0] > 0:
                lst.append(k)
                
    lst = list(set(lst))
    return lst


def getRowsColumns(n1,n2,m,exclRows=None,exclColumns=None):
    """
    Find rows number and columns number for endogenous variables.
    
    Parameters:
        :param m: Map of variables names and lead/lags values and equations row and column numbers.
        :type m: dict.
        :param exclRows: Rows to exclude.
        :type exclRows: list.
        :param exclColumns: Columns to exclude.
        :type exclColumns: list.
    """
    Columns = set(); Rows = set()    
    for k in m:
        if not m[k] is None and len(m[k]) > 0:
            rows,columns = m[k]
            Rows.update(rows)
            for c in columns:
                if c >= n1 and c < n2:
                    Columns.add(c-n1)
        
    if not exclRows is None:
        Rows -= set(exclRows)
    if not exclColumns is None:
        Columns -= set(exclColumns)
        
    Rows    = list(Rows);    Rows.sort()
    Columns = list(Columns); Columns.sort()
    
    return Rows,Columns

    
def getColumns(n,m,var,exclRows=None,exclColumns=None):
    """
    Find columns number for endogenous variables.
    
    Parameters:
        :param m: Map of variables names and lead/lags values and equations row and column numbers.
        :type m: dict.
        :param exclRows: Rows to exclude.
        :type exclRows: list.
        :param exclColumns: Columns to exclude.
        :type exclColumns: list.
    """
    mp = {}  
    excl_rows = [] if exclRows is None else exclRows
    excl_columns = [] if exclColumns is None else exclColumns
    for k in m:
        rows,columns = m[k]
        for r,c in zip(rows,columns):
            if not r in excl_rows and not c in excl_columns and c >= n and c < 2*n:
                if r in mp:
                    mp[r].append(c-n)
                else:
                    mp[r] = [c-n]
    keys = m.keys()
    positions = []
    for k in keys:
        ind = var.index(k)
        if ind < n:
            positions.append(ind)

    return mp,positions


def topology(model):
    """
    Find rows number and columns number for different types of endogenous variables.
    
    Parameters:
        :param model: The Model object.
        :type model: instance of class Model.
    """
    getTopologyOfVariables(model)
    static = model.topology["static"] 
    state = model.topology["state"]
    dynamic = model.topology["dynamic"]
    mixed = model.topology["mixed"]
    backward = model.topology["backward"] 
    purely_backward = model.topology["purely_backward"]
    forward = model.topology["forward"]
    purely_forward = model.topology["purely_forward"]
    variables = model.symbols["variables"]
    n = len(variables)
    
    ###  ------------------------------- Find rows and columns of:
    # static endogenous variables
    static_rows,static_columns = getRowsColumns(n1=n,n2=2*n,m=static)
    # state endogenous variables
    state_rows,state_columns = getRowsColumns(n1=n,n2=2*n,m=state,exclRows=static_rows,exclColumns=static_columns)
    # dynamic endogenous variables
    dynamic_rows,dynamic_columns = getRowsColumns(n1=n,n2=2*n,m=dynamic,exclRows=static_rows,exclColumns=static_columns)
    # mixed endogenous variables
    mixed_rows,mixed_columns = getRowsColumns(n1=n,n2=2*n,m=mixed,exclRows=static_rows,exclColumns=static_columns)
    # backward endogenous variables
    backward_rows,backward_columns = getRowsColumns(n1=2*n,n2=3*n,m=backward,exclRows=static_rows,exclColumns=static_columns)
    # purely backward endogenous variables
    purely_backward_rows,purely_backward_columns = getRowsColumns(n1=2*n,n2=3*n,m=purely_backward,exclRows=static_rows,exclColumns=static_columns)
    # foreward endogenous variables
    forward_rows,forward_columns = getRowsColumns(n1=0,n2=n,m=forward,exclRows=static_rows,exclColumns=static_columns)
    # purely foreward endogenous variables
    purely_forward_rows,purely_forward_columns = getRowsColumns(n1=0,n2=n,m=purely_forward,exclRows=static_rows,exclColumns=static_columns)
    
    #model.n_fwd_looking_var = len(forward_rows)
    model.topology["static_rows"] = static_rows
    model.topology["state_rows"] = state_rows
    model.topology["state_columns"] = state_columns
    model.topology["static_rows"] = static_rows
    model.topology["static_columns"] = static_columns
    model.topology["dynamic_rows"] = dynamic_rows
    model.topology["dynamic_columns"] = dynamic_columns
    model.topology["forward_rows"] = forward_rows
    model.topology["forward_columns"] = forward_columns
    model.topology["purely_forward_rows"] = purely_forward_rows   
    model.topology["purely_forward_columns"] = purely_forward_columns
    model.topology["backward_rows"] = backward_rows
    model.topology["backward_columns"] = backward_columns
    model.topology["purely_backward_columns"] = purely_backward_columns
    model.topology["mixed_columns"] = mixed_columns
    model.topology["mixed_rows"] = mixed_rows
    
    debug = False
    if debug:
        from snowdrop.src.misc.termcolor import cprint
        cprint('')
        cprint('state_columns: ' + str(state_columns),'green')
        cprint('static_columns: ' + str(static_columns),'green')
        cprint('static_rows: ' + str(static_rows),'green')
        cprint('dynamic_columns: ' + str(dynamic_columns),'green')
        cprint('forward_columns: ' + str(forward_columns),'green')
        cprint('purely_forward_columns: ' + str(purely_forward_columns),'green')
        cprint('backward_columns: ' + str(backward_columns),'green')
        cprint('purely_backward_columns: ' + str(purely_backward_columns),'green')
        cprint('mixed_columns: ' + str(mixed_columns),'green')
              
    return model             
               
 
def getForwardBackwardRowsColumns(model):
    """Return row and column numbers of forward and backward endogenous variables.""" 
    var = model.symbols["variables"]
    n = len(var)
      
    rowsForward = model.topology["forward_rows"]
    colsForward = model.topology["forward_columns"]
    rowsBackward = [i for i in range(n) if not i in rowsForward]
    colsBackward = [i for i in range(n) if not i in colsForward]  
    
    return rowsBackward,colsBackward,rowsForward,colsForward


def getStableUnstableVariables(model,debug=True):
    """
    Return stable and unstable endogenous variables.
    
    Parameters:
        :param model: Model object.
        :type model: Model.
        :returns: Array of starting values
    """ 
    unstableColumns = []; unstableRows = []; var_unstable = []
    variables = model.symbols["variables"]
    n = len(variables)
    mp = getTopology(model) 
    
    for k in mp:
        v = mp[k]
        if "+" in " ".join(v):
            for e in v:
                ind = e.index(':')
                lead = int(e[:ind])
                if lead > 0:
                    e = e[1+ind:]
                    ind = e.index(',')
                    row = int(e[:ind])
                    col = int(e[1+ind:]) % n
                    unstableRows.append(row)
                    unstableColumns.append(col)
                    var_unstable.append(k)
    
    unstableColumns = np.unique(unstableColumns)
    unstableRows = np.unique(unstableRows)
    stableColumns = [i for i in range(n) if not i in unstableColumns]   
    stableRows = [i for i in range(n) if not i in unstableRows]  
    n_stable = len(stableColumns)
    n_unstable = len(unstableColumns)
    var_stable = [v for v in variables if not v in var_unstable]
        
    if debug:
        print()
        print("Number of stable variables {0}:".format(n_stable))
        print("Stable variables: ",sorted(var_stable))
        print()
        print("Number of unstable variables {0}:".format(n_unstable))
        print("Unstable variables: ",sorted(var_unstable))
        print()
    
    return var_stable,var_unstable,stableColumns,stableRows,unstableColumns,unstableRows


def getVariablesOrder(model,debug=False):
    """
    Return stable and unstable endogenous variables.
    
    Parameters:
        :param model: Model object.
        :type model: Model.
        :returns: Array of starting values
    """ 
    import pandas as pd
    
    variables = model.symbols["variables"]
    n = len(variables)
    mp = getTopology(model) 
    
    meas_var = model.symbols.get("measurement_variables",[])
    measCol=[]; measRow=[];staticCol=[]; staticRow=[]; backwardCol=[]
    forwardCol=[]; forwardRow=[]; mixedCol=[]; mixedRow=[]; backwardRow=[]
    
    ### First pass
    for k in mp:
        v = mp[k]
        x = " ".join(v)
        if k in meas_var:
            for e in v:
                ind = e.index(':')
                lead_lag = int(e[:ind])
                e = e[1+ind:]
                ind = e.index(',')
                row = int(e[:ind])
                col = int(e[1+ind:]) % n
                measCol.append(col)
                measRow.append(row)
        elif not "-" in x and not "+" in x:
            for e in v:
                ind = e.index(':')
                lead_lag = int(e[:ind])
                e = e[1+ind:]
                ind = e.index(',')
                row = int(e[:ind])
                col = int(e[1+ind:]) % n
                staticCol.append(col)
                staticRow.append(row)
        elif "-" in x and "+" in x:
            for e in v:
                ind = e.index(':')
                lead_lag = int(e[:ind])
                e = e[1+ind:]
                ind = e.index(',')
                row = int(e[:ind])
                col = int(e[1+ind:]) % n
                if not lead_lag == 0:
                    mixedCol.append(col)
                    mixedRow.append(row)
        elif "-" in x:
            for e in v:
                ind = e.index(':')
                lead_lag = int(e[:ind])
                e = e[1+ind:]
                ind = e.index(',')
                row = int(e[:ind])
                col = int(e[1+ind:]) % n
                if not lead_lag < 0:
                    backwardCol.append(col)
                    backwardRow.append(row)
        elif "+" in x:    
            for e in v:
                ind = e.index(':')
                lead_lag = int(e[:ind])
                e = e[1+ind:]
                ind = e.index(',')
                row = int(e[:ind])
                col = int(e[1+ind:]) % n
                if not lead_lag > 0:
                    forwardCol.append(col)
                    forwardRow.append(row)
    
    measCol = list(np.unique(measCol))
    measRow = list(np.unique(measRow))
    staticCol = list(np.unique(staticCol))
    staticRow = list(np.unique(staticRow))
    backwardCol = list(np.unique(backwardCol))
    backwardRow = list(np.unique(backwardRow))
    mixedCol = list(np.unique(mixedCol))
    mixedRow = list(np.unique(mixedRow))
    forwardCol = list(np.unique(forwardCol))
    forwardRow = list(np.unique(forwardRow))
        
    meas_var = [variables[i] for i in range(n) if i in measCol]
    backward_var = [variables[i] for i in range(n) if i in backwardCol]
    static_var = [variables[i] for i in range(n) if i in staticCol]
    mixed_var = [variables[i] for i in range(n) if i in mixedCol]
    forward_var = [variables[i] for i in range(n) if i in forwardCol]
    varOrder = meas_var+static_var+backward_var+mixed_var+forward_var
    
    # Make sure row numbers for different group of variables do not everlap.
    staticRow = [i for i in staticRow if not i in measRow]
    backwardRow = [i for i in backwardRow if not i in measRow]
    mixedRow = [i for i in mixedRow if not i in measRow and not i in backwardRow]
    forwardRow = [i for i in forwardRow if not i in measRow and not i in backwardRow and not i in mixedRow]
        
    columns = measCol+staticCol+backwardCol+mixedCol+forwardCol
    rows = np.unique(measRow+staticRow+backwardRow+mixedRow+forwardRow)
    missed_rows = [i for i in range(n) if not i in rows]
    
    # Second pass
    if len(rows) < len(columns):
        for k in mp:
            v = mp[k]
            x = " ".join(v)
            if not "-" in x and not "+" in x:
                for e in v:
                    ind = e.index(':')
                    lead_lag = int(e[:ind])
                    e = e[1+ind:]
                    ind = e.index(',')
                    row = int(e[:ind])
                    col = int(e[1+ind:]) % n
                    if row in missed_rows:
                        staticCol.append(col)
                        staticRow.append(row)
            elif "-" in x and "+" in x:
                for e in v:
                    ind = e.index(':')
                    lead_lag = int(e[:ind])
                    e = e[1+ind:]
                    ind = e.index(',')
                    row = int(e[:ind])
                    col = int(e[1+ind:]) % n
                    if row in missed_rows:
                        mixedCol.append(col)
                        mixedRow.append(row)
            elif "-" in x:
                for e in v:
                    ind = e.index(':')
                    lead_lag = int(e[:ind])
                    e = e[1+ind:]
                    ind = e.index(',')
                    row = int(e[:ind])
                    col = int(e[1+ind:]) % n
                    if row in missed_rows:
                        backwardCol.append(col)
                        backwardRow.append(row)
            elif "+" in x:    
                for e in v:
                    ind = e.index(':')
                    lead_lag = int(e[:ind])
                    e = e[1+ind:]
                    ind = e.index(',')
                    row = int(e[:ind])
                    col = int(e[1+ind:]) % n
                    if row in missed_rows:
                        forwardCol.append(col)
                        forwardRow.append(row)
        
    staticRow = [i for i in staticRow if not i in measRow]
    backwardRow = [i for i in backwardRow if not i in measRow]
    mixedRow = [i for i in mixedRow if not i in measRow and not i in backwardRow]
    forwardRow = [i for i in forwardRow if not i in measRow and not i in backwardRow and not i in mixedRow]
    
    colOrder = list(pd.unique(measCol+staticCol+backwardCol+mixedCol+forwardCol))
    rowOrder = list(pd.unique(staticCol+measCol+backwardCol+mixedCol+forwardCol))
    
    invColOrder = [colOrder.index(i) for i in range(n)]
    invRowOrder = [rowOrder.index(i) for i in range(n)]
    
    if debug:
        print()
        print("\nOrdered variables:",varOrder)
        print("\nMeasurement variables:",meas_var)
        print("\nStatic variables:",static_var)
        print("\nBackward variables:",backward_var)
        print("\nMixed variables:",mixed_var)
        print("\nForward variables:",forward_var)
        print()
        missed_var = [variables[i] for i in range(n) if not i in rowOrder]
        if len(missed_var) > 0:
            print("Missed variables:")
            print(missed_var)
            print()
        # testColOrder = np.all(np.array([colOrder[i] for i in invColOrder]) == np.arange(n))
        # testRowOrder = np.all(np.array([rowOrder[i] for i in invRowOrder]) == np.arange(n))
        # print("Test column order:",testColOrder)
        # print("Test row order:",testRowOrder)
           
    n_static = len(staticCol)
    n_meas = len(measCol)
    mf = np.arange(n_static,n_static+n_meas)
    
    Z = np.zeros((n_meas,n))
    Z[:,n_static:n_static+n_meas] = np.eye(n_meas)
    
    return Z, mf, colOrder, rowOrder, invColOrder, invRowOrder


arr = None
def check_presence(symb,equations):
    """
    Check if symbol is defined in equations.

    Parameters:
        symb : str
            Symbol.
        equations : str
            Equations.

    Returns:
        True if symbol is defined in equations and False otherwise

    """
    # global arr
    # if arr is None:
    #     delimiters = " ", ",", ";", "*", "/", ":", "+", "-", "^", "{", "}", "(", ")","[","]"
    #     regexPattern = '|'.join(map(re.escape, delimiters))
    #     arr = re.split(regexPattern,equations)
    #     arr = list(filter(None,arr))
    
    b = symb in equations
    
    return b


def getMap(endog,equations,eqLabels=None):   
    """
    Get dictionary of row numbers as keys, and equation labels and endogenous variables as values.
    
    Parameters:
        :param endog: Endogenous variables.
        :type edog: list.
        :param equations: Equations.
        :type equations: list.
        :param eqLabels: Equations labels.
        :type eqLables: list.
        :return: Dictionary with rows numbers as keys, and equation label
                 and an array of equation endogenous variables as values.
    """
    import re
    global mp1, mp2
    
    if not mp1 is None and not mp2 is None:
        return mp1, mp2
    
    mp1 = {}; mp2 = {}
    delimiters = " ", ",", ";", "*", "/", ":", "+", "-", "^", "{", "}","(", ")","="
    regexPattern = '|'.join(map(re.escape, delimiters))
    
    for row,eqtn in enumerate(equations):
        eq  = eqtn.replace(" ","")
        if not eqLabels is None and row < len(eqLabels):
            var = eqLabels[row]
        else:
            ind = eq.index("=") if "=" in eq else -1
            lhs = eq[:ind].strip()
            if ind >= 0:
                arr = re.split(regexPattern,lhs)
                arr = [x for x in arr if x in endog]
                if len(arr) > 0:
                    var = arr[0]
        if var in endog:
            arr = re.split(regexPattern,eq)
            arr = [x for x in arr if x in endog]
            lst = []
            for v in arr:
                ind1 = eq.index(v)
                if eq[min(ind1+len(v),len(eq)-1)] == "(":
                    ind2 = eq.index(")",ind1)
                    numbr = eq[ind1+len(v)+1:ind2]
                    num = numbr.replace("+","").replace("-","")
                    if num.isdigit():
                        lst.append((v,int(numbr)))
                    else:
                        lst.append((v,0))
                    eq = eq[1+ind2:]
                elif not v==var:
                    lst.append((v,0))
                    eq = eq[ind1+len(v):]
                else:
                    eq = eq[ind1+len(v):]
            if len(lst) > 0:
                mp1[row] = (var,lst)
            if len(arr) > 0:
                mp2[row] = (var,set(arr))
            
    return mp1,mp2


def getRHS(eqs,variables,eqLabels=None,b=True,debug=False):
    """
    Return right-hand-side of equations.
    
    Parameters:
        :param eqs: Model equations.
        :type eqs: list.
        :param variables: Endogenous variables.
        :type variables: list.
        :param eqLabels: Labels of equations.
        :type eqLabels: list.
        :returns: Right hand side terms.
    """ 
    import re
    delimiters = ",","+","-","*","/","(",")","__"
    regexPattern = '|'.join(map(re.escape, delimiters))
    
    rhs_eqs = []; endog = []
    n = len(eqs)
        
    if b is None:
        # Right-hand-side equations
        for i,eq in enumerate(eqs):
            arr = str.split(eq,'=')
            if len(arr) == 2:
                left  = arr[0].strip()
                right = arr[1].strip()
                if right == '0':
                    right = left
            else:
                right = eq
            rhs_eqs.append(right)
    
    else:
                
        from snowdrop.src.preprocessor.symbolic import stringify
        from snowdrop.src.preprocessor.function_compiler_sympy import normalize_equations
        
        lst_eqs = []
        for eq in eqs:
            arr = str.split(eq,'=')
            left = arr[0].strip()
            if len(arr) > 1:
                right = arr[1].strip()
                lst_eqs.append(left + " - (" + right + ")")
            else:
                lst_eqs.append(left)
        
        syms = [(v,1) for v in variables] \
             + [(v,0) for v in variables] \
             + [(v,-1) for v in variables]
        vv = [s[0] for s in syms]
        eqs_sym = normalize_equations(lst_eqs,vv,True)
     
        ii = 0
        for i,eqtn in enumerate(eqs):
            eq = eqtn.replace(" ","")
            arr = str.split(eq,'=')
            bRow = False
            if len(arr) == 2:
                left = arr[0]
                right = arr[1]
                arr2 = re.split(regexPattern,left)
                arr2 = [x for x in arr2 if x in variables]
                var = arr2[0]
                if len(arr2) > 1:
                    bRow = True
                elif "(" in left or ")" in left or "+" in left or "-" in left or "*" in left or "/" in left or "^" in left :
                    bRow = True
            else:
                right = eq
                if not eqLabels is None and i < len(eqLabels) and eqLabels[i] in variables:
                    var = eqLabels[i] 
                else:
                    var = None
            endog.append(var)
                
            if bRow: 
                if ii == 0 and debug:
                    print("Started conversion of equations.")
                ii += 1
                max_lead = 0  
                while var+"(" in eq:
                    ind1 = eq.index(var+"(")
                    eq = eq[1+ind1+len(var):]
                    if ")" in eq:
                        ind2 = eq.index(")")
                        numbr = int(eq[:ind2])
                        max_lead = max(max_lead,numbr)
                
                x = stringify((var,max_lead))
                s = sy.symbols(x)
                if debug: 
                    print(f"\n{i+1}, {s}:  {eqs_sym[i]} = 0")
                sol = sy.solve(eqs_sym[i],s)
               
                if len(sol) > 0:
                    rhs = str(sol[0])
                    if debug: 
                        cprint(f"   {s} = {rhs}","blue")
                    rhs_eqs.append(str(normalize_equations([rhs],vv,True)[0]))
                else:
                    rhs_eqs.append(right)
            else:
                rhs_eqs.append(str(normalize_equations([right],vv,True)[0]))
        if ii > 0 and debug:
            print(f"Converted {ii} equations.\n")

    return rhs_eqs,endog


# def replaceEquation(eq,var):
#     """
#     Replaces lead/lag variables in equations with equivalent representation.
#     For eaxmple: var(-1) -> var__m1_ ; var(1) -> var__p1_  ; var( -> var__.
    
#     Parameters:
#         :param eq: Model equation.
#         :type eq: str.
#         :param var: Variables.
#         :type var: list.
#         :returns: Normalized equation.
#     """ 
#     delimiters = " ", ",", ";", "*", "/", ":", "+", "-", "^", "=", "{", "}", "(", ")", "[", "]"
#     regexPattern = '|'.join(map(re.escape, delimiters))
    
#     arr = re.split(regexPattern,eq)
#     arr = list(filter(None,arr))
#     new_eq = ""
#     ind = 0; ind1 = 0; beg = 0
#     for v in arr:
#         if v in var:
#             ind  = eq.index(v,ind1)
#             ind1 = ind+len(v)
#             if ind1 < len(eq):
#                 end = ind
#                 if eq[ind1] == "(":
#                     ind2 = eq.index(")",ind1)
#                     shift = eq[1+ind1:ind2]
#                     new_eq += eq[beg:end] + v + \
#                         ("__m" + str(abs(int(shift))) + "_" if "-" in shift else "__p" + str(abs(int(shift))) + "_")
#                     beg = end + len(v) + (ind2-ind1+1)
#                 else:
#                     new_eq += eq[beg:end] + v + "__"
#                     beg = end + len(v)
#             else:
#                 ind1 = ind+len(v)
#                 new_eq += eq[beg:ind1] + "__"
#                 beg = ind1
#                 end = beg + 2
                
#         # else:
#         #     end = ind
#         #     ind1 = end + len(v)
            
#     if beg == 0:
#         new_eq = eq
#     else:
#         rest = eq[beg:]
#         arr = re.split(regexPattern,rest)
#         arr = list(filter(None,arr))
#         br = False
#         for v in arr:
#             if v in var:
#                 ind  = rest.index(v)
#                 new_eq += rest[:ind]
#                 new_eq += rest + "__"
#                 br = True
#                 break
#         if not br:
#             new_eq += rest
            
#     return new_eq
            
            
def get_steady_state_equations(eqs,endog,params,shocks=[],exog=[]):
    """
    Build steady state equations.
    
    Parameters
    ----------
    eqs : list
        Dynamic equations.
    endog : list
        Endogenous variables that exclude added auxillary variables.
    params : list
        Parameters.
    shocks : list
        Shocks names.
    exog : list
        Exogenous variables.

    Returns
    -------
    ss_equations : list.
        Transformed equations with removed time subscript.
    """  
    delimiters = " ", ",", ";", "*", "/", ":", "+", "-", "=", "(", ")","^"
    regexPattern = '|'.join(map(re.escape, delimiters))
    
    ss_equations = []
    for i,eqtn in enumerate(eqs):
        eq = eqtn.replace(" ","")
        arr = re.split(regexPattern,eq)
        arr = list(filter(None,arr))
        arr = [x for x in arr if x in endog or x in params or x in shocks or x in exog]
        ind = 0
        # Replace lead/lag variables or shocks with current values
        for v in arr:
            if v in eq[ind:]:
                ind1 = eq.index(v,ind)
                k = ind1 + len(v)
                if 1+k < len(eq) and eq[k] == "(" and ")" in eq[1+k:]:
                    ind2 = eq.index(")",ind1)
                    num = eq[1+k:ind2]
                    match = re.match(r"[-+]?\d+",num)
                    if not match is None and v+"("+num+")" in eq:
                        eq = eq.replace(v+"("+num+")",v)
                ind = ind1 + len(v) 
                
        ss_equations.append(eq)
        
    #print("get_steady_state_equations:","(-1)" in "\n".join(ss_equations))
    #ind = [i for i,x in enumerate(ss_equations) if "(-1)" in x]
    
    return ss_equations
        
 
def aggregateEqs(variables,aggregation,freq,aggr_freq,m_aggr):
    """Build aggregated equations and endogenous variables for M/Q/A frequencies."""
    new_eqs = []; new_vars = []; new_exog_vars = []
            
    # Add aggregation equations
    for v,aggr in zip(variables,aggregation): 
        if v in m_aggr:
            if freq == "monthly" and aggr_freq == "quarterly":
                if aggr == "sum":
                    newEq = f"SHK_{v}[t] = {v}_AGGREGATED[t] - ({v}[max(0,t-1)]+{v}[max(0,t-2)]+{v}[max(0,t-3)])"
                elif aggr == "mean":
                    newEq = f"SHK_{v}[t] = {v}_AGGREGATED[t] - ({v}[max(0,t-1)]+{v}[max(0,t-2)]+{v}[max(0,t-3)])/3"
                elif aggr == "geomean":
                    newEq = f"SHK_{v}[t] = {v}_AGGREGATED[t] - ({v}[max(0,t-1)]*{v}[max(0,t-2)]*{v}[max(0,t-3)])**(1/3.)"
                elif aggr == "first":
                    newEq = f"SHK_{v}[t] = {v}_AGGREGATED[t] - {v}[max(0,t-3)]"
                elif aggr == "last":
                    newEq = f"SHK_{v}[t] = {v}_AGGREGATED[t] - {v}[max(0,t-1)]"
            elif freq == "quarterly" and aggr_freq == "annual":
                if aggr == "sum":
                    newEq = f"SHK_{v}[t] = {v}_AGGREGATED[t] - ({v}[max(0,t-1)]+{v}[max(0,t-2)]+{v}[max(0,t-3)]+{v}[max(0,t-4)])"
                elif aggr == "mean":
                    newEq = f"SHK_{v}[t] = {v}_AGGREGATED[t] - ({v}[max(0,t-1)]+{v}[max(0,t-2)]+{v}[max(0,t-3)]+{v}[max(0,t-4)])/4"
                elif aggr == "geomean":
                    newEq = f"SHK_{v}[t] = {v}_AGGREGATED[t] - ({v}[max(0,t-1)]*{v}[max(0,t-2)]*{v}[max(0,t-3)]*{v}[max(0,t-4)])**(1/4.)"
                elif aggr == "first":
                    newEq = f"SHK_{v}[t] = {v}_AGGREGATED[t] - {v}[max(0,t-4)]"
                elif aggr == "last":
                    newEq = f"SHK_{v}[t] = {v}_AGGREGATED[t] = {v}[max(0,t-1)]"
            elif freq == "monthly" and aggr_freq == "annual":
                if aggr == "sum":
                    newEq = f"SHK_{v}[t] = {v}_AGGREGATED[t] - ({v}[max(0,t-1)]+{v}[max(0,t-2)]+{v}[max(0,t-3)]+{v}[max(0,t-4)]+{v}[max(0,t-5)]+{v}[max(0,t-6)]+{v}[max(0,t-7)]+{v}[max(0,t-8)]+{v}[max(0,t-9)]+{v}[max(0,t-10)]+{v}[max(0,t-11)]+{v}[max(0,t-12)])"
                elif aggr == "mean":
                    newEq = f"SHK_{v}[t] = {v}_AGGREGATED[t] - ({v}[max(0,t-1)]+{v}[max(0,t-2)]+{v}[max(0,t-3)]+{v}[max(0,t-4)]+{v}[max(0,t-5)]+{v}[max(0,t-6)]+{v}[max(0,t-7)]+{v}[max(0,t-8)]+{v}[max(0,t-9)]+{v}[max(0,t-10)]+{v}[max(0,t-11)]+{v}[max(0,t-12)])/12"
                elif aggr == "geomean":
                    newEq = f"SHK_{v}[t] = {v}_AGGREGATED[t] - ({v}[max(0,t-1)]*{v}[max(0,t-2)]*{v}[max(0,t-3)]*{v}[max(0,t-4)]*{v}[max(0,t-5)]*{v}[max(0,t-6)]*{v}[max(0,t-7)]*{v}[max(0,t-8)]*{v}[max(0,t-9)]*{v}[max(0,t-10)]*{v}[max(0,t-11)]*{v}[max(0,t-12)])**(1/12.)"
                elif aggr == "first":
                    newEq = f"SHK_{v}[t] = {v}_AGGREGATED[t] - {v}[max(0,t-12)]"
                elif aggr == "last":
                    newEq = f"SHK_{v}[t] = {v}_AGGREGATED[t] - {v}[max(0,t-1)]"
            new_eqs.append(newEq)
            new_vars.append(f"SHK_{v}[t]")
            new_exog_vars.append(f"{v}_AGGREGATED")
                    
    return new_eqs,new_vars,new_exog_vars 
            
       
def processFormulas(v,formulas):
    for formula in formulas:
        if formula == "pch" or formula == "pchy":
            return f"100*({v}/{v}(-1)-1)"
    return v


def transformEq(eqtn,var,formulas=[],b=None):
    """
    Replace endogenous variables v in equations as following,  v -> v*scale and build steady state equations.

    Parameters
    ----------
    eqtn : str
        Equation.
    var : list
        Endogenous variables.
    formula : str, optional
        Formula, for example, percentage change 'pchy'. The default is None.
    b : bool, optional
        If True normalize equation, i.e. replace variables lead/lag with var__p/m# substring. The default is None.

    Returns
    -------
    New equation and new steady-state equation.

    """
    br1 = "("; br2 = ")"
    eq = eqtn.replace("{","(").replace("}",")")
    arr = re.split(regexPattern,eq)
    arr = list(filter(None,arr))
    new_eq = ""; ss_eq = ""
    
    ind1 = 0; beg = 0; j = -1
    for i in range(len(arr)):
        j += 1
        if j < len(arr)-1:
            v = arr[j]
            if v in var+formulas:
                ind  = eq.index(v,ind1)
                ind1 = ind+len(v)
                if ind1 < len(eq):
                    end = ind
                    if v in formulas and "(" in eq[ind:ind+5]:
                        vn = arr[i+1]
                        expr = processFormulas(vn,formulas)
                        new_eq += eq[beg:end] + "(" + expr + ")"
                        j += 1
                        ind1 += 2 + len(vn)
                        beg = ind1
                    elif eq[ind1] == br1:
                        ind2 = eq.index(br2,ind1)
                        shift = eq[1+ind1:ind2]
                        if b is None:
                            new_eq += eq[beg:end] + v + br1 + shift + br2
                        elif b:
                            if "-" in shift:
                                tmp = f"__m{abs(int(shift))}_"
                            else:
                                tmp = f"__p{abs(int(shift))}_"
                            new_eq += eq[beg:end] + v + tmp
                        else:
                            tmp = "[t" + eq[1+ind1:ind2] + "]"
                            new_eq += eq[beg:end] + v + tmp
                        beg = end + len(v) + (ind2-ind1+1)
                    else:
                        if b is None:
                            new_eq += eq[beg:end] + v
                        elif b:
                            new_eq += eq[beg:end] + v
                        else:
                            new_eq += eq[beg:end] + v + "[t]"
                        beg = end + len(v)
                    
    if beg == 0:
        new_eq = eq
    else:
        rest = eq[beg:]
        arr = re.split(regexPattern,rest)
        arr = list(filter(None,arr))
        br = False
        for v in arr:
            if v in var:
                ind  = rest.index(v)
                new_eq += rest[:ind]
                if b is None:
                    new_eq += rest[ind:]
                elif b:
                    new_eq += rest[ind:]
                else:
                    new_eq += rest[ind:] + "[t]"
                br = True
                break
        if not br:
            new_eq += rest

    # # Steady state equations
    # arr = re.split(regexPattern,eq)
    # arr = list(filter(None,arr))
    # ind1 = 0; beg = 0; j = -1
    # for i in range(len(arr)):
    #     j += 1
    #     if j < len(arr)-1:
    #         v = arr[j]
    #         if v in var+formulas:
    #             ind  = eq.index(v,ind1)
    #             ind1 = ind+len(v)
    #             if ind1 < len(eq):
    #                 end = ind
    #                 if v in formulas and "(" in eq[ind:ind+5]:
    #                     vn = arr[i+1]
    #                     ss_eq += eq[beg:end] + "0"
    #                     j += 1
    #                     ind1 += 2 + len(vn)
    #                     beg = ind1
    #                 elif ind1 < len(eq):
    #                     if eq[ind1] == br1:
    #                         ind2 = eq.index(br2,ind1)
    #                         shift = eq[1+ind1:ind2]
    #                         ss_eq += eq[beg:end] + v
    #                         beg = end + len(v) + (ind2-ind1+1)
    #                     else:
    #                         ss_eq += eq[beg:end] + v
    #                         beg = end + len(v)
    # if beg == 0:
    #     ss_eq = eq
    # else:
    #     rest = eq[beg:]
    #     arr = re.split(regexPattern,rest)
    #     arr = list(filter(None,arr))
    #     br = False
    #     for v in arr:
    #         ind  = rest.index(v)
    #         ss_eq += rest[:ind]
    #         if b is None:
    #             ss_eq += rest[ind:]
    #         else:
    #             ss_eq += rest[ind:]  + "__"
    #         br = True
    #         break
    #     if not br:
    #         ss_eq += rest
            

    # Transform equation
    if b:
        new_eq = normalizeEq(new_eq,var)
    
    return new_eq


def normalizeEq(eq,var):
    """Normalize equation such that it has the form: var = F(var,...)."""
    from sympy import Symbol,solve

    ind = eq.index("=")
    lhs = eq[:ind]
    rhs = eq[1+ind:]
    if "(" in lhs or ")" in lhs or "*" in lhs or "/" in lhs or "+" in lhs or "-" in lhs:
        arr = re.split(regexPattern,lhs)
        arr = list(filter(None,arr))
        for v in arr:
            if v in var:
                break
            
        # Solve equation algebraically with sympy
        x = Symbol(f"{v}")
        eq = f"{lhs} - _y_"
        expr = solve(eq, x)
        expr = str(expr[0])
        expr = expr.replace("_y_","("+rhs+")")
        eq = f"{v} = {expr}"
        
    return eq


def build_func(var,exog,params,eqs):
    """Build array of functions."""
    lfunc = []; largs = []; lnames = []; leqs = []
    for eq in eqs:
        func,name,args = str2func(var,exog,params,eq)
        if not name in params:
            lfunc.append(func)
            largs.append(args)
            lnames.append(name)
            leqs.append(eq)
        
    return lfunc,lnames,largs,leqs
 
    
def str2func(var_names,exog_names,param_names,eqtn):
    """Convert string to Python function."""
    from sympy import symbols,lambdify
    
    delimiters = " ", ",", ";", "*", "/", ":", "+", "-", "^", "=", "{", "}", "(", ")", "[", "]"
    regexPattern = "|".join(map(re.escape, delimiters))
      
    ind = eqtn.index("=")
    left = eqtn[:ind] 
    right = eqtn[1+ind:]
    arr = re.split(regexPattern,left)
    arr = list(filter(None,arr))
    arr2 = re.split(regexPattern,right)
    arr2 = list(filter(None,arr2))
    variables = []; args = []
    
    # Add variables
    for var in arr:
        if var in var_names:
            break
            
    for v in arr2:
        if v in var_names and not v in args:
            variables.append(symbols(v))
            args.append(v)
        if "__m" in v:
            ind = v.index("__m")
            vv = v[:ind]
            if vv in var_names and not v in variables:
                variables.append(symbols(v))
                args.append(v)
        elif "__p" in v:
            ind = v.index("__p")
            vv = v[:ind]
            if vv in var_names and not v in variables:
                variables.append(symbols(v))
                args.append(v)
                
    # Add exogenous variables
    for v in arr2:
        if "__m" in v:
            ind = v.index("__m")
            vv = v[:ind]
        elif "__p" in v:
            ind = v.index("__p")
            vv = v[:ind]
        else:
            vv = None
        if v in exog_names and not v in args:
            variables.append(symbols(v))
            args.append(v)
        if vv in exog_names and not v in args:
            variables.append(symbols(v))
            args.append(v)
            
    # Add parameters
    for v in arr2:
        if v in param_names and not v in args:
            variables.append(symbols(v))
            args.append(v)
          
    func = lambdify(variables,right,"numpy")
    return func, var, args

def getExogMatrix(eqs,exog):
    """
    Compute matrix of exogenous variables.

    Parameters
    ----------
    eqs : list
        Equations.
    exog : list
        Exogenous variables.

    Returns
    -------
    Matrix of exogenous variables.

    """
    n = len(eqs)
    m = len(exog)
    W = np.zeros((n,m))
    
    for i,eq in enumerate(eqs):
        arr = re.split(regexPattern,eq)
        arr = list(filter(None,arr))
        for j,exo in enumerate(exog):
            if exo in arr: # eq
                W[i,j] = 1 
                
    return W
                
if __name__ == '__main__':
    """Main entry point."""
    # variables = ["a","b","c"]
    # eqs = []
    # eq = "a(1 ) + b( +21) = c( -3 )"
    # eqs.append(eq)
    # eq = " a(+10) + b( -20 ) = g( 30 ) "
    # eqs.append(eq)
    
    # max_lead,min_lag,n_fwd_looking_var,n_bkwd_looking_var,var_lead,var_lag = getMaxLeadsLags(eqs,variables)
    # print(max_lead,min_lag)
    
    # endog = ['L_GDP', 'L_GDP_BAR', 'DL_GDP_BAR', 'L_GDP_GAP', 'PIE', 'PIE_TAR', 'PIE_EXP', 'UNR', 'UNR_BAR', 'UNR_GAP', 'G_UNR_BAR']
    # eqs=['PIE = lambda1*PIE(+1) + (1-lambda1)*PIE(-1)+ lambda3*L_GDP_GAP']
    # mod_eqs = modifyEquations(eqs=eqs,variables=endog)
    # print(mod_eqs)
    
    # eq = "au__ - (-lockdown_policy*theta_lockdown + 1)**2*(ci__*cs__*i__m1_*pi1*s__m1_ + i__m1_*ni__*ns__*pi2*s__m1_ + i__m1_*pi3*s__m1_)"
    # #print(fixEquation(eq))
    # eq = "F__*pie__**(1/(gam - 1))*xi*((-pie__**(1/(gam - 1))*xi + 1)/(1 - xi))**(1 - gam)*(1 - gam)/(pie__*(gam - 1)*(-pie__**(1/(gam - 1))*xi + 1))"
    # eq = "(x**2/9+y**3)**5"
    # print(fixEquation(eq))
    
    # eq = """log(ISR_COM_FE_R) = 0+ISR_COM1*(((log(RPOIL_PERM(-2)/RPOIL_PERM_BAR*ISR_SUB_OIL(-2))+PERSIST_E_ISR_RPOIL_CYC*E_ISR_RPOIL_CYC(-2)+E_ISR_RPOIL_PERM(-2))/3)+((log(RPOIL_PERM(-1)/RPOIL_PERM_BAR*ISR_SUB_OIL(-1))+PERSIST_E_ISR_RPOIL_CYC*E_ISR_RPOIL_CYC(-1)+E_ISR_RPOIL_PERM(-1))/3)+((log(RPOIL_PERM/RPOIL_PERM_BAR*ISR_SUB_OIL)+PERSIST_E_ISR_RPOIL_CYC*E_ISR_RPOIL_CYC+E_ISR_RPOIL_PERM)/3))+ISR_COM2*(log(RPOIL_CYC*ISR_SUB_OIL)+E_ISR_RPOIL_CYC)+ISR_COM4*(((log(RPMETAL_PERM(-2)/RPMETAL_PERM_BAR)+PERSIST_E_ISR_RPMETAL_CYC*E_ISR_RPMETAL_CYC(-2)+E_ISR_RPMETAL_PERM(-2))/3)+((log(RPMETAL_PERM(-1)/RPMETAL_PERM_BAR)+PERSIST_E_ISR_RPMETAL_CYC*E_ISR_RPMETAL_CYC(-1)+E_ISR_RPMETAL_PERM(-1))/3)+((log(RPMETAL_PERM/RPMETAL_PERM_BAR)+PERSIST_E_ISR_RPMETAL_CYC*E_ISR_RPMETAL_CYC+E_ISR_RPMETAL_PERM)/3))+ISR_COM5*(log(RPMETAL_CYC)+E_ISR_RPMETAL_CYC)"""       
    # endog = ['E_WRL_PRODOIL_CYC_R', 'ISR_ACT_R', 'ISR_BREVAL_N', 'ISR_B_N', 'ISR_B_RAT', 'ISR_CFOOD_R', 'ISR_CNCOM_R', 'ISR_COIL_R', 'ISR_COMEFFECT_R', 'ISR_COM_FE_R', 'ISR_CPI13', 'ISR_CPINCOM_P', 'ISR_CPIX_P', 'ISR_CPI_P', 'ISR_CURBAL_N', 'ISR_C_LIQ_R', 'ISR_C_OLG_R', 'ISR_C_R', 'ISR_C_RAT', 'ISR_DELTA', 'ISR_EPS', 'ISR_FACTFOOD_R', 'ISR_FACT_R', 'ISR_FXPREM', 'ISR_GC_N', 'ISR_GC_R', 'ISR_GC_RAT', 'ISR_GDEF_N', 'ISR_GDEF_RAT', 'ISR_GDEF_TAR', 'ISR_GDPINC_N', 'ISR_GDPSIZE', 'ISR_GDP_FE_R', 'ISR_GDP_N', 'ISR_GDP_R', 'ISR_GE_N', 'ISR_GISTOCK_R', 'ISR_GI_N', 'ISR_GI_R', 'ISR_GI_RAT', 'ISR_GNP_R', 'ISR_GOVCHECK', 'ISR_GSUB_N', 'ISR_GTARIFF_N', 'ISR_G_R', 'ISR_IFOODA_R', 'ISR_IFOOD_R', 'ISR_IMETALA_R', 'ISR_IMETAL_R', 'ISR_IM_R', 'ISR_INFCPI', 'ISR_INFCPIX', 'ISR_INFEXP', 'ISR_INFL', 'ISR_INFPIM', 'ISR_INFWAGE', 'ISR_INFWAGEEFF', 'ISR_INFWEXP', 'ISR_INT', 'ISR_INT10', 'ISR_INTC', 'ISR_INTCORP', 'ISR_INTCOST_N', 'ISR_INTCOST_RAT', 'ISR_INTGB', 'ISR_INTMP', 'ISR_INTMPU', 'ISR_INTNFA', 'ISR_INTRF', 'ISR_INTRF10', 'ISR_INTXM10', 'ISR_INVESTP_R', 'ISR_INVEST_R', 'ISR_INVEST_RAT', 'ISR_IOILA_R', 'ISR_IOIL_R', 'ISR_IT_R', 'ISR_IT_RAT', 'ISR_J', 'ISR_KG_R', 'ISR_K_R', 'ISR_LABH_FE_R', 'ISR_LABH_R', 'ISR_LAB_FE_R', 'ISR_LAB_R', 'ISR_LF_FE_R', 'ISR_LF_R', 'ISR_LSTAX_RAT', 'ISR_MKTPREM', 'ISR_MKTPREMSM', 'ISR_MPC', 'ISR_MPCINV', 'ISR_NFAREVAL_N', 'ISR_NFA_D', 'ISR_NFA_RAT', 'ISR_NPOPB_R', 'ISR_NPOPH_R', 'ISR_NPOP_R', 'ISR_NTRFPSPILL_FE_R', 'ISR_OILRECEIPT_N', 'ISR_OILSUB_N', 'ISR_PART', 'ISR_PARTH', 'ISR_PARTH_DES', 'ISR_PARTH_FE', 'ISR_PART_DES', 'ISR_PART_FE', 'ISR_PCFOOD_P', 'ISR_PCOIL_P', 'ISR_PCW_P', 'ISR_PC_P', 'ISR_PFM_P', 'ISR_PFOOD_P', 'ISR_PGDP_P', 'ISR_PGDP_P_AVG', 'ISR_PG_P', 'ISR_PIMADJ_P', 'ISR_PIMA_P', 'ISR_PIM_P', 'ISR_PIT_P', 'ISR_PI_P', 'ISR_PMETAL_CYC_P', 'ISR_PMETAL_P', 'ISR_PMETAL_PERM_P', 'ISR_POIL_CYC_P', 'ISR_POIL_P', 'ISR_POIL_PERM_P', 'ISR_POIL_P_SUB', 'ISR_PRIMSUR_N', 'ISR_PRIMSUR_TAR', 'ISR_PRODFOOD_CYC_R', 'ISR_PRODFOOD_PERM_R', 'ISR_PRODFOOD_R', 'ISR_PSAVING_N', 'ISR_PXMF_P', 'ISR_PXMUNADJ_P', 'ISR_PXM_P', 'ISR_PXT_P', 'ISR_Q_P', 'ISR_R', 'ISR_R10', 'ISR_RC', 'ISR_RC0_SM', 'ISR_RC0_WM', 'ISR_RCI', 'ISR_RCORP', 'ISR_REER', 'ISR_RK8', 'ISR_RK_P', 'ISR_RPREM', 'ISR_R_NEUT', 'ISR_SOVPREM', 'ISR_SOVPREMSM', 'ISR_SUB_OIL', 'ISR_TAU_C', 'ISR_TAU_K', 'ISR_TAU_L', 'ISR_TAU_OIL', 'ISR_TAXC_N', 'ISR_TAXC_RAT', 'ISR_TAXK_N', 'ISR_TAXK_RAT', 'ISR_TAXLH_N', 'ISR_TAXL_N', 'ISR_TAXL_RAT', 'ISR_TAXOIL_N', 'ISR_TAX_N', 'ISR_TAX_RAT', 'ISR_TB_N', 'ISR_TFPEFFECT_R', 'ISR_TFPKGSPILL_FE_R', 'ISR_TFPSPILL_FE_R', 'ISR_TFP_FE_R', 'ISR_TFP_FE_R_AVG', 'ISR_TFP_R', 'ISR_TM', 'ISR_TPREM', 'ISR_TRANSFER_LIQ_N', 'ISR_TRANSFER_N', 'ISR_TRANSFER_OLG_N', 'ISR_TRANSFER_RAT', 'ISR_TRANSFER_TARG_N', 'ISR_TRANSFER_TARG_RAT', 'ISR_TRFPSPILL_FE_R', 'ISR_UFOOD_R', 'ISR_UNR', 'ISR_UNRH', 'ISR_UNRH_FE', 'ISR_UNR_FE', 'ISR_USA_SM', 'ISR_USA_WM', 'ISR_WAGEEFF_N', 'ISR_WAGEH_N', 'ISR_WAGE_N', 'ISR_WF_R', 'ISR_WH_R', 'ISR_WK_N', 'ISR_WO_R', 'ISR_W_R', 'ISR_W_R_AVG', 'ISR_XFOOD_R', 'ISR_XMA_R', 'ISR_XM_R', 'ISR_XT_R', 'ISR_XT_RAT', 'ISR_YCAP_N', 'ISR_YD_R', 'ISR_YLABH_N', 'ISR_YLAB_N', 'ISR_Z', 'ISR_Z_AVG', 'ISR_Z_NFA', 'PFOOD_P', 'PMETAL_CYC_P', 'PMETAL_P', 'PMETAL_PERM_P', 'POIL_CYC_P', 'POIL_P', 'POIL_PERM_P', 'RC0_ACT_R', 'RC0_BREVAL_N', 'RC0_B_N', 'RC0_B_RAT', 'RC0_CFOOD_R', 'RC0_CNCOM_R', 'RC0_COIL_R', 'RC0_COMEFFECT_R', 'RC0_COM_FE_R', 'RC0_CPI13', 'RC0_CPINCOM_P', 'RC0_CPIX_P', 'RC0_CPI_P', 'RC0_CURBAL_N', 'RC0_C_LIQ_R', 'RC0_C_OLG_R', 'RC0_C_R', 'RC0_C_RAT', 'RC0_DELTA', 'RC0_EPS', 'RC0_FACTFOOD_R', 'RC0_FACTMETAL_R', 'RC0_FACTOIL_R', 'RC0_FACT_R', 'RC0_FXPREM', 'RC0_GC_N', 'RC0_GC_R', 'RC0_GC_RAT', 'RC0_GDEF_N', 'RC0_GDEF_RAT', 'RC0_GDEF_TAR', 'RC0_GDPINC_N', 'RC0_GDPSIZE', 'RC0_GDP_FE_R', 'RC0_GDP_N', 'RC0_GDP_R', 'RC0_GE_N', 'RC0_GISTOCK_R', 'RC0_GI_N', 'RC0_GI_R', 'RC0_GI_RAT', 'RC0_GNP_R', 'RC0_GOVCHECK', 'RC0_GSUB_N', 'RC0_GTARIFF_N', 'RC0_G_R', 'RC0_IFOODA_R', 'RC0_IFOOD_R', 'RC0_IMETALA_R', 'RC0_IMETAL_R', 'RC0_IM_R', 'RC0_INFCPI', 'RC0_INFCPIX', 'RC0_INFEXP', 'RC0_INFL', 'RC0_INFPIM', 'RC0_INFWAGE', 'RC0_INFWAGEEFF', 'RC0_INFWEXP', 'RC0_INT', 'RC0_INT10', 'RC0_INTC', 'RC0_INTCORP', 'RC0_INTCOST_N', 'RC0_INTCOST_RAT', 'RC0_INTGB', 'RC0_INTMP', 'RC0_INTMPU', 'RC0_INTNFA', 'RC0_INTRF', 'RC0_INTRF10', 'RC0_INTXM10', 'RC0_INVESTP_R', 'RC0_INVEST_R', 'RC0_INVEST_RAT', 'RC0_IOILA_R', 'RC0_IOIL_R', 'RC0_ISR_SM', 'RC0_ISR_WM', 'RC0_IT_R', 'RC0_IT_RAT', 'RC0_J', 'RC0_KG_R', 'RC0_K_R', 'RC0_LABH_FE_R', 'RC0_LABH_R', 'RC0_LAB_FE_R', 'RC0_LAB_R', 'RC0_LF_FE_R', 'RC0_LF_R', 'RC0_LSTAX_RAT', 'RC0_MKTPREM', 'RC0_MKTPREMSM', 'RC0_MPC', 'RC0_MPCINV', 'RC0_MROYALTIES_N', 'RC0_MROYALTY', 'RC0_NFAREVAL_N', 'RC0_NFA_D', 'RC0_NFA_RAT', 'RC0_NPOPB_R', 'RC0_NPOPH_R', 'RC0_NPOP_R', 'RC0_NTRFPSPILL_FE_R', 'RC0_OILPAY_N', 'RC0_OILRECEIPT_N', 'RC0_OILSHARF', 'RC0_OILSUB_N', 'RC0_PART', 'RC0_PARTH', 'RC0_PARTH_DES', 'RC0_PARTH_FE', 'RC0_PART_DES', 'RC0_PART_FE', 'RC0_PCFOOD_P', 'RC0_PCOIL_P', 'RC0_PCW_P', 'RC0_PC_P', 'RC0_PFM_P', 'RC0_PFOOD_P', 'RC0_PGDP_P', 'RC0_PGDP_P_AVG', 'RC0_PG_P', 'RC0_PIMADJ_P', 'RC0_PIMA_P', 'RC0_PIM_P', 'RC0_PIT_P', 'RC0_PI_P', 'RC0_PMETAL_CYC_P', 'RC0_PMETAL_P', 'RC0_PMETAL_PERM_P', 'RC0_POIL_CYC_P', 'RC0_POIL_P', 'RC0_POIL_PERM_P', 'RC0_POIL_P_SUB', 'RC0_PRIMSUR_N', 'RC0_PRIMSUR_TAR', 'RC0_PRODFOOD_CYC_R', 'RC0_PRODFOOD_PERM_R', 'RC0_PRODFOOD_R', 'RC0_PRODMETAL_CYC_R', 'RC0_PRODMETAL_PERM_R', 'RC0_PRODMETAL_R', 'RC0_PRODOIL_CYC_R', 'RC0_PRODOIL_PERM_R', 'RC0_PRODOIL_R', 'RC0_PSAVING_N', 'RC0_PXMF_P', 'RC0_PXMUNADJ_P', 'RC0_PXM_P', 'RC0_PXT_P', 'RC0_Q_P', 'RC0_R', 'RC0_R10', 'RC0_RC', 'RC0_RCI', 'RC0_RCORP', 'RC0_REER', 'RC0_RK8', 'RC0_RK_P', 'RC0_ROYALTIES_N', 'RC0_ROYALTY', 'RC0_RPREM', 'RC0_R_NEUT', 'RC0_SOVPREM', 'RC0_SOVPREMSM', 'RC0_SUB_OIL', 'RC0_TAU_C', 'RC0_TAU_K', 'RC0_TAU_L', 'RC0_TAU_OIL', 'RC0_TAXC_N', 'RC0_TAXC_RAT', 'RC0_TAXK_N', 'RC0_TAXK_RAT', 'RC0_TAXLH_N', 'RC0_TAXL_N', 'RC0_TAXL_RAT', 'RC0_TAXOIL_N', 'RC0_TAX_N', 'RC0_TAX_RAT', 'RC0_TB_N', 'RC0_TFPEFFECT_R', 'RC0_TFPKGSPILL_FE_R', 'RC0_TFPSPILL_FE_R', 'RC0_TFP_FE_R', 'RC0_TFP_FE_R_AVG', 'RC0_TFP_R', 'RC0_TM', 'RC0_TPREM', 'RC0_TRANSFER_LIQ_N', 'RC0_TRANSFER_N', 'RC0_TRANSFER_OLG_N', 'RC0_TRANSFER_RAT', 'RC0_TRANSFER_TARG_N', 'RC0_TRANSFER_TARG_RAT', 'RC0_TRFPSPILL_FE_R', 'RC0_UFOOD_R', 'RC0_UMETAL_R', 'RC0_UNR', 'RC0_UNRH', 'RC0_UNRH_FE', 'RC0_UNR_FE', 'RC0_UOIL_R', 'RC0_USA_SM', 'RC0_USA_WM', 'RC0_WAGEEFF_N', 'RC0_WAGEH_N', 'RC0_WAGE_N', 'RC0_WF_R', 'RC0_WH_R', 'RC0_WK_N', 'RC0_WO_R', 'RC0_W_R', 'RC0_W_R_AVG', 'RC0_XFOOD_R', 'RC0_XMA_R', 'RC0_XMETAL_R', 'RC0_XM_R', 'RC0_XOIL_R', 'RC0_XT_R', 'RC0_XT_RAT', 'RC0_YCAP_N', 'RC0_YD_R', 'RC0_YLABH_N', 'RC0_YLAB_N', 'RC0_Z', 'RC0_Z_AVG', 'RC0_Z_NFA', 'RPFOOD', 'RPFOOD_CYC', 'RPFOOD_PERM', 'RPMETAL', 'RPMETAL_CYC', 'RPMETAL_CYC_AVG', 'RPMETAL_PERM', 'RPMETAL_PERM_AVG', 'RPOIL', 'RPOIL_CYC', 'RPOIL_CYC_AVG', 'RPOIL_PERM', 'RPOIL_PERM_AVG', 'USA_ACT_R', 'USA_BREVAL_N', 'USA_B_N', 'USA_B_RAT', 'USA_CFOOD_R', 'USA_CNCOM_R', 'USA_COIL_R', 'USA_COMEFFECT_R', 'USA_COM_FE_R', 'USA_CPI13', 'USA_CPINCOM_P', 'USA_CPIX_P', 'USA_CPI_P', 'USA_CURBAL_N', 'USA_C_LIQ_R', 'USA_C_OLG_R', 'USA_C_R', 'USA_C_RAT', 'USA_DELTA', 'USA_EPS', 'USA_FACTFOOD_R', 'USA_FACTMETAL_R', 'USA_FACTOIL_R', 'USA_FACT_R', 'USA_FXPREM', 'USA_GC_N', 'USA_GC_R', 'USA_GC_RAT', 'USA_GDEF_N', 'USA_GDEF_RAT', 'USA_GDEF_TAR', 'USA_GDPINC_N', 'USA_GDPSIZE', 'USA_GDP_FE_R', 'USA_GDP_N', 'USA_GDP_R', 'USA_GE_N', 'USA_GISTOCK_R', 'USA_GI_N', 'USA_GI_R', 'USA_GI_RAT', 'USA_GNP_R', 'USA_GOVCHECK', 'USA_GSUB_N', 'USA_GTARIFF_N', 'USA_G_R', 'USA_IFOODA_R', 'USA_IFOOD_R', 'USA_IMETALA_R', 'USA_IMETAL_R', 'USA_IM_R', 'USA_INFCPI', 'USA_INFCPIX', 'USA_INFEXP', 'USA_INFL', 'USA_INFPIM', 'USA_INFWAGE', 'USA_INFWAGEEFF', 'USA_INFWEXP', 'USA_INT', 'USA_INT10', 'USA_INTC', 'USA_INTCORP', 'USA_INTCOST_N', 'USA_INTCOST_RAT', 'USA_INTGB', 'USA_INTMP', 'USA_INTMPU', 'USA_INTNFA', 'USA_INTRF', 'USA_INTRF10', 'USA_INTXM10', 'USA_INVESTP_R', 'USA_INVEST_R', 'USA_INVEST_RAT', 'USA_IOILA_R', 'USA_IOIL_R', 'USA_ISR_SM', 'USA_ISR_WM', 'USA_IT_R', 'USA_IT_RAT', 'USA_J', 'USA_KG_R', 'USA_K_R', 'USA_LABH_FE_R', 'USA_LABH_R', 'USA_LAB_FE_R', 'USA_LAB_R', 'USA_LF_FE_R', 'USA_LF_R', 'USA_LSTAX_RAT', 'USA_MKTPREM', 'USA_MKTPREMSM', 'USA_MPC', 'USA_MPCINV', 'USA_MROYALTIES_N', 'USA_MROYALTY', 'USA_NFAREVAL_N', 'USA_NFA_D', 'USA_NFA_RAT', 'USA_NPOPB_R', 'USA_NPOPH_R', 'USA_NPOP_R', 'USA_NTRFPSPILL_FE_R', 'USA_OILPAY_N', 'USA_OILRECEIPT_N', 'USA_OILSHARF', 'USA_OILSUB_N', 'USA_PART', 'USA_PARTH', 'USA_PARTH_DES', 'USA_PARTH_FE', 'USA_PART_DES', 'USA_PART_FE', 'USA_PCFOOD_P', 'USA_PCOIL_P', 'USA_PCW_P', 'USA_PC_P', 'USA_PFM_P', 'USA_PFOOD_P', 'USA_PGDP_P', 'USA_PGDP_P_AVG', 'USA_PG_P', 'USA_PIMADJ_P', 'USA_PIMA_P', 'USA_PIM_P', 'USA_PIT_P', 'USA_PI_P', 'USA_PMETAL_CYC_P', 'USA_PMETAL_P', 'USA_PMETAL_PERM_P', 'USA_POIL_CYC_P', 'USA_POIL_P', 'USA_POIL_PERM_P', 'USA_POIL_P_SUB', 'USA_PRIMSUR_N', 'USA_PRIMSUR_TAR', 'USA_PRODFOOD_CYC_R', 'USA_PRODFOOD_PERM_R', 'USA_PRODFOOD_R', 'USA_PRODMETAL_CYC_R', 'USA_PRODMETAL_PERM_R', 'USA_PRODMETAL_R', 'USA_PRODOIL_CYC_R', 'USA_PRODOIL_PERM_R', 'USA_PRODOIL_R', 'USA_PSAVING_N', 'USA_PXMF_P', 'USA_PXMUNADJ_P', 'USA_PXM_P', 'USA_PXT_P', 'USA_Q_P', 'USA_R', 'USA_R10', 'USA_RC', 'USA_RC0_SM', 'USA_RC0_WM', 'USA_RCI', 'USA_RCORP', 'USA_REER', 'USA_RK8', 'USA_RK_P', 'USA_ROYALTIES_N', 'USA_ROYALTY', 'USA_RPREM', 'USA_R_NEUT', 'USA_SOVPREM', 'USA_SOVPREMSM', 'USA_SUB_OIL', 'USA_TAU_C', 'USA_TAU_K', 'USA_TAU_L', 'USA_TAU_OIL', 'USA_TAXC_N', 'USA_TAXC_RAT', 'USA_TAXK_N', 'USA_TAXK_RAT', 'USA_TAXLH_N', 'USA_TAXL_N', 'USA_TAXL_RAT', 'USA_TAXOIL_N', 'USA_TAX_N', 'USA_TAX_RAT', 'USA_TB_N', 'USA_TFPEFFECT_R', 'USA_TFPKGSPILL_FE_R', 'USA_TFPSPILL_FE_R', 'USA_TFP_FE_R', 'USA_TFP_FE_R_AVG', 'USA_TFP_R', 'USA_TM', 'USA_TPREM', 'USA_TRANSFER_LIQ_N', 'USA_TRANSFER_N', 'USA_TRANSFER_OLG_N', 'USA_TRANSFER_RAT', 'USA_TRANSFER_TARG_N', 'USA_TRANSFER_TARG_RAT', 'USA_TRFPSPILL_FE_R', 'USA_UFOOD_R', 'USA_UMETAL_R', 'USA_UNR', 'USA_UNRH', 'USA_UNRH_FE', 'USA_UNR_FE', 'USA_UOIL_R', 'USA_WAGEEFF_N', 'USA_WAGEH_N', 'USA_WAGE_N', 'USA_WF_R', 'USA_WH_R', 'USA_WK_N', 'USA_WO_R', 'USA_W_R', 'USA_W_R_AVG', 'USA_XFOOD_R', 'USA_XMA_R', 'USA_XMETAL_R', 'USA_XM_R', 'USA_XOIL_R', 'USA_XT_R', 'USA_XT_RAT', 'USA_YCAP_N', 'USA_YD_R', 'USA_YLABH_N', 'USA_YLAB_N', 'USA_Z', 'USA_Z_AVG', 'USA_Z_NFA', 'WRL_GDP_FE_METAL_R', 'WRL_GDP_FE_OIL_R', 'WRL_GDP_FE_R', 'WRL_GDP_METAL_R', 'WRL_GDP_OIL_R', 'WRL_GDP_R', 'WRL_PRODFOOD_CYC_R', 'WRL_PRODFOOD_PERM_R', 'WRL_PRODFOOD_R', 'WRL_PRODMETAL_CYC_R', 'WRL_PRODMETAL_PERM_R', 'WRL_PRODMETAL_R', 'WRL_PRODOIL_CYC_R', 'WRL_PRODOIL_PERM_R', 'WRL_PRODOIL_R', 'WRL_XFOOD_R', 'WRL_XMETAL_R', 'WRL_XOIL_R', 'WTRADE_FOOD_N', 'WTRADE_FOOD_R', 'WTRADE_METAL_N', 'WTRADE_METAL_R', 'WTRADE_M_N', 'WTRADE_M_R', 'WTRADE_OIL_N', 'WTRADE_OIL_R']
    # params=  ['USA_INTNFA1', 'ISR_EPS_TAR', 'RC0_IM5', 'FILTER_RC0_MPC', 'ISR_SPILLM1', 'RC0_R_NEUT_BAR', 'RC0_XX_PRIMSUR_N', 'PERSISTCOM_ISR', 'ISR_GPROD_SS', 'CONSTANT_USA_PG_P', 'CONSTANT_RPOIL_PERM', 'USA_GC_RAT_BAR', 'FILTER_WTRADE_FOOD_R', 'CONSTANT_USA_FXPREM', 'CONSTANT_RC0_CPINCOM_P', 'CONSTANT_USA_MKTPREM', 'FILTER_USA_PRIMSUR_TAR', 'USA_GDP_N_BAR', 'RC0_COM5', 'RES_ISR_GNP_R', 'USA_COEFPGDP', 'ISR_Q1', 'RC0_IM1', 'FILTER_USA_INTNFA', 'RES_RPOIL_PERM_AVG', 'FILTER_ISR_PXM_P', 'RC0_XX_TRANSFER_N', 'USA_XMETAL2', 'USA_COEFW', 'RES_ISR_INTCORP', 'CONSTANT_USA_CPI_P', 'RC0_GAMMA', 'RES_RPMETAL_CYC_AVG', 'RES_RC0_PCOIL_P', 'USA_COEFTFP', 'ISR_EBASE', 'ISR_INTCOST_RAT_BAR', 'FILTER_USA_PARTH', 'USA_GILRSW', 'RES_RC0_ROYALTIES_N', 'CONSTANT_ISR_GDEF_TAR', 'RES_USA_WO_R', 'ISR_XM2', 'PERSIST_E_RC0_RPMETAL_CYC', 'USA_BNSHK', 'RC0_CFOOD3', 'USA_COM4', 'ISR_BGG4', 'USA_CFOOD3', 'RC0_WEDGEFOOD_P', 'RC0_INTGAP', 'RC0_INFCPIX_TAR', 'FILTER_USA_IMETAL_R', 'ISR_USA_SOIL', 'RC0_RK5', 'USA_GNP_R_BAR', 'FILTER_ISR_PARTH', 'USA_EXP1', 'CONSTANT_RC0_PC_P', 'ISR_XX_TRANSFER_TARG_N', 'RES_ISR_INT10', 'USA_IM5', 'RES_USA_INTRF', 'USA_UNR2', 'USA_TAXL_RAT_BAR', 'USA_BGG3', 'RC0_WT', 'USA_PRODMETAL1', 'CONSTANT_ISR_IOIL_R', 'USA_CPI4', 'RC0_PIM1', 'FILTER_ISR_K_R', 'RES_RC0_Z_NFA', 'USA_PXM1', 'ISR_PARTH_ADJ_STAR', 'RPMETAL_CYC_BAR', 'ISR_IMETAL3', 'RC0_KTAXLRSW', 'FILTER_RC0_TPREM', 'RES_USA_PARTH_FE', 'FILTER_WTRADE_FOOD_N', 'CONSTANT_RC0_IMETAL_R', 'USA_INTINFL', 'RC0_W_R_BAR', 'RC0_INTCOST_RAT_BAR', 'RC0_TAXK_RAT_BAR', 'RC0_CPI7', 'TRDEMETAL_N', 'USA_Q2', 'CONSTANT_ISR_C_OLG_R', 'USA_RK4', 'CONSTANT_RC0_PCW_P', 'USA_RC0_SMETAL', 'ISR_COEFPGDP', 'RES_ISR_C_LIQ_R', 'USA_NSPILLM1', 'FILTER_ISR_TPREM', 'ISR_PSUB_XM', 'ISR_COIL2', 'RC0_CPI4', 'RC0_POIL_P_SUB_BAR', 'FILTER_USA_TAU_OIL', 'RES_ISR_Q_P', 'RC0_GI_N_EXOG', 'USA_Z_BAR', 'ISR_GI_R_BAR', 'RC0_BGG4', 'RES_RC0_Z_AVG', 'ISR_INTMP_BAR', 'USA_COM5', 'USA_LAPH', 'USA_COM1', 'USA_B_RAT_BAR', 'FILTER_ISR_CPI_P', 'ISR_COEFPARTH', 'ISR_IOIL1', 'RES_USA_PCOIL_P', 'CONSTANT_RC0_TPREM', 'RC0_IFOOD1', 'ISR_W_R_BAR', 'RC0_TARGDAMP', 'USA_BETA', 'RC0_CTAXLRSW', 'RC0_XX_TAU_L', 'FILTER_RC0_PRIMSUR_TAR', 'WRL_GDP_OIL_R_BAR', 'FILTER_ISR_PRIMSUR_TAR', 'RES_RC0_INFWEXP', 'ISR_XX_PRIMSUR_N', 'WRL_GDP_R_BAR', 'RC0_ISR_S', 'USA_GI_MPROP', 'ISR_XFOOD1', 'RC0_INTMP_BAR', 'USA_BREVAL1', 'USA_IOIL4', 'ISR_IFOOD4', 'ISR_TRANSFER_RAT_EXOG', 'RC0_IMETAL2', 'RC0_PXM1', 'RC0_TARIFF_IM', 'RES_USA_ROYALTIES_N', 'USA_COSTOIL_P', 'CONSTANT_USA_PXMF_P', 'ISR_MNFACTOR', 'ISR_TAXLBASE_N_BAR', 'RC0_IM_SPILL_BAR', 'USA_PARTH2', 'RC0_DAMP_GDP_GAP', 'USA_IOIL2', 'USA_PRODMETAL_R_EXOG', 'RC0_COM4', 'USA_EPS_TAR', 'ISR_TAU_L_BAR', 'RC0_GI_RAT_BAR', 'RC0_XX_TRANSFER_TARG_N', 'USA_PSUB_XM', 'CONSTANT_USA_NPOPB_R', 'ISR_SIGMA', 'FILTER_USA_RPREM', 'USA_PRODOIL2', 'ISR_VMETAL', 'USA_SUB_XM', 'RC0_PRODMETAL1', 'USA_CFOOD1', 'CONSTANT_USA_OILPAY_N', 'USA_INTLAG', 'RPMETAL_PERM_BAR', 'FILTER_RC0_PCW_P', 'USA_SOV1', 'ISR_TAXCBASE_N_BAR', 'USA_GIDAMP', 'USA_IMETAL1', 'PERSIST_E_RC0_RPOIL_CYC', 'RC0_IMETAL3', 'USA_RC0_S', 'CONSTANT_WTRADE_M_N', 'RC0_XMETAL1', 'ISR_IM1', 'RES_RC0_GNP_R', 'RES_ISR_PGDP_P_AVG', 'TRDEFOOD_N', 'ISR_CPI2', 'ISR_KGTFPRAT', 'USA_CFOOD4', 'ISR_W2', 'RES_USA_INTC', 'RES_ISR_PRODFOOD_R', 'USA_SIGMA', 'RC0_UNR3', 'RC0_INTGR', 'USA_SPILLM1', 'FILTER_ISR_COIL_R', 'RPMETAL1', 'ISR_XM4', 'RC0_TSPILLX1', 'RES_ISR_CURBAL_N', 'RC0_NSPILLX1', 'FILTER_RC0_XOIL_R', 'USA_COM2', 'RES_POIL_PERM_P', 'USA_SPILLX1', 'TRDEOIL_R', 'ISR_CFOOD1', 'RC0_SPILLX1', 'RC0_COSTMETAL_P', 'ISR_R_NEUT_BAR', 'USA_PRODOIL_R_EXOG', 'USA_TOIL', 'FILTER_RC0_PG_P', 'RC0_EBASE', 'CONSTANT_USA_PI_P', 'RC0_CPI2', 'USA_CGBFX', 'RES_USA_MROYALTIES_N', 'RC0_GC_N_EXOG', 'CONSTANT_ISR_MKTPREM', 'RC0_TMETAL', 'ISR_B_STAR', 'ISR_GC_MPROP', 'RC0_XM1', 'USA_TFP_FE_R_BAR', 'ISR_TAXC_RAT_BAR', 'USA_UNR3', 'USA_PARTH4', 'RES_RC0_INT10', 'CONSTANT_ISR_CFOOD_R', 'RES_USA_C_LIQ_R', 'RC0_NFANSHK', 'ISR_WEDGEOIL_P', 'FILTER_USA_TPREM', 'FILTER_ISR_IOIL_R', 'ISR_XFOOD2', 'ISR_C_LAMBDA', 'RES_USA_CURBAL_N', 'RC0_W1', 'ISR_CPI7', 'FILTER_USA_PXM_P', 'ISR_BETA', 'TEMP_USA', 'CONSTANT_USA_INTNFA', 'RC0_XMETAL3', 'WRL_PRODFOOD_R_BAR', 'CONSTANT_USA_XFOOD_R', 'RC0_XFOOD4', 'FILTER_RC0_CFOOD_R', 'RC0_SOV2', 'RES_USA_NFA_D', 'ISR_INTEPS', 'ISR_XX_TRANSFER_N', 'ISR_LAPH', 'CONSTANT_RC0_SOVPREM', 'RC0_PRODOIL1', 'RC0_CHI', 'USA_XOIL4', 'CONSTANT_ISR_K_R', 'USA_GDP_R_BAR', 'USA_XX_GI_N', 'RES_USA_INFCPIX', 'ISR_INTWT', 'RC0_CPI9', 'RC0_EXP1', 'FILTER_USA_DELTA', 'FILTER_RC0_RPREM', 'RPOIL3', 'ISR_IMETAL1', 'PERSIST_E_USA_RPOIL_CYC', 'CONSTANT_RC0_OILPAY_N', 'ISR_XX_TAU_C', 'RC0_XFOOD2', 'CONSTANT_ISR_XFOOD_R', 'CONSTANT_WTRADE_FOOD_R', 'ISR_IM5', 'RC0_IMETAL1', 'USA_PRODFOODSHARE', 'RC0_XX_GI_N', 'RES_RC0_RC', 'USA_UOIL_R_BAR', 'ISR_TAXK_RAT_BAR', 'ISR_CPI4', 'CONSTANT_ISR_IM_R', 'FILTER_USA_XFOOD_R', 'RC0_IOIL2', 'USA_C_MPROP', 'FILTER_RC0_XM_R', 'FILTER_RC0_K_R', 'RC0_CPI1', 'ISR_GNP_R_BAR', 'USA_C_LIQ_R_BAR', 'FILTER_USA_PCW_P', 'ISR_PCOILBASE', 'FILTER_RPFOOD_PERM', 'CONSTANT_RC0_PXM_P', 'RC0_VMETAL', 'FILTER_ISR_IMETAL_R', 'RES_ISR_PIM_P', 'RES_ISR_R_NEUT', 'FILTER_ISR_PG_P', 'RES_ISR_BREVAL_N', 'ISR_XM_MPROP', 'RC0_USA_SOIL', 'RC0_XM2', 'USA_CTAXLRSW', 'ISR_PIM2', 'CONSTANT_ISR_TFP_FE_R', 'USA_CPI1', 'USA_XM_SPILL_BAR', 'RC0_INTC1', 'USA_COIL2', 'USA_RK2', 'CONSTANT_RC0_PARTH', 'FILTER_RC0_FXPREM', 'FILTER_ISR_INTNFA', 'USA_RK5', 'FILTER_USA_GDEF_TAR', 'USA_XOIL2', 'ISR_PARTH4', 'ISR_TSPILLM1', 'RC0_COEFPARTH', 'RC0_IMETAL5', 'FILTER_ISR_C_OLG_R', 'RES_ISR_PARTH_DES', 'CONSTANT_ISR_IFOOD_R', 'USA_XX_GC_N', 'USA_DAMP_GDP_GAP', 'RES_ISR_WH_R', 'USA_XX_TRANSFER_TARG_N', 'FILTER_USA_SOVPREM', 'ISR_TAXOILBASE_N_BAR', 'RC0_B_STAR', 'ISR_COEFTFP', 'USA_TAXC_RAT_BAR', 'CONSTANT_USA_DELTA', 'USA_PCFOODBASE', 'CONSTANT_USA_OILSHARF', 'ISR_TMETAL', 'ISR_BGG1', 'CONSTANT_USA_SOVPREM', 'ISR_PARTH5', 'RC0_PRODMETAL2', 'USA_CTAXDAMP', 'ISR_CTAXLRSW', 'RC0_DELTAG', 'USA_WEDGEFOOD_P', 'ISR_UNR3', 'RES_ISR_INFEXP', 'USA_XOIL_R_BAR', 'FILTER_RC0_PXM_P', 'TEMP_E_USA_RPFOOD_CYC', 'FILTER_RC0_NPOPB_R', 'USA_INTGR', 'USA_XMETAL1', 'ISR_WT', 'USA_C_LAMBDA', 'RES_USA_PARTH_DES', 'ISR_USA_L', 'USA_ALPHA_KG', 'FILTER_ISR_FXPREM', 'ISR_EXP1', 'USA_CFOOD2', 'USA_BGG4', 'RC0_BNSHK', 'ISR_XX_GC_N', 'RES_USA_W_R_AVG', 'RES_RC0_PARTH_FE', 'RC0_CTAXDAMP', 'USA_TARGLRSW', 'RC0_COEFZ', 'ISR_INTC1', 'CONSTANT_USA_XMETAL_R', 'USA_TRANSFERDAMP', 'RES_RC0_TAXOIL_N', 'CONSTANT_RPMETAL_PERM', 'CONSTANT_RC0_CPI_P', 'RC0_XX_OILSUB_N', 'RC0_XM4', 'USA_INTCORP1', 'RC0_XM_MPROP', 'USA_RC0_SOIL', 'ISR_CPI3', 'USA_IM3', 'ISR_ALPHA_FE', 'CONSTANT_USA_ROYALTY', 'RPOIL_PERM_BAR', 'ISR_RK1', 'RC0_C_MPROP', 'RES_USA_BREVAL_N', 'USA_ISR_S', 'RC0_TARGLRSW', 'RES_RC0_INTMP', 'RES_USA_Z_NFA', 'ISR_TAU_K_EXOG', 'RC0_WEDGEOIL_P', 'RC0_C_OLG_R_BAR', 'CONSTANT_USA_K_R', 'ISR_KGSPILLM1', 'USA_TAU_C_EXOG', 'FILTER_WTRADE_M_N', 'RC0_TAXKBASE_N_BAR', 'RC0_W3', 'USA_W_R_BAR', 'CONSTANT_RC0_RPREM', 'RES_RC0_PIM_P', 'RES_ISR_INTRF', 'RPMETAL4', 'RES_ISR_PARTH_FE', 'RC0_IFOOD4', 'ISR_SOV2', 'FILTER_RC0_TFP_FE_R', 'FILTER_ISR_CFOOD_R', 'ISR_TRANSFERSHARE', 'CONSTANT_ISR_NPOPB_R', 'FILTER_RC0_MROYALTY', 'USA_IMETAL4', 'CONSTANT_USA_MPC', 'FILTER_ISR_XFOOD_R', 'RC0_IM4', 'TEMP_ISR', 'ISR_TRANSFERDAMP', 'WRL_GDP_METAL_R_BAR', 'ISR_B_RAT_BAR', 'USA_CPI7', 'USA_LTAXDAMP', 'RC0_XM3', 'USA_TAU_L_EXOG', 'ISR_RK4', 'USA_DAMP_DEBT', 'USA_NPOP_R_EXOG', 'USA_IM4', 'ISR_IMETAL5', 'CONSTANT_USA_PXM_P', 'USA_TARIFF_IM', 'RPFOOD2', 'RC0_COEFW', 'TEMP_E_ISR_TAXK_RAT', 'RC0_Z_BAR', 'CONSTANT_ISR_RPREM', 'ISR_TARGLRSW', 'RC0_LSTAX_N', 'USA_XMETAL4', 'ISR_CFOOD4', 'ISR_INVEST_MPROP', 'RES_RC0_RCORP', 'USA_XOIL3', 'FILTER_RC0_C_OLG_R', 'PERSISTCOM_USA', 'USA_XOIL1', 'RES_USA_OILSUB_N', 'FILTER_ISR_CPINCOM_P', 'ISR_XX_GI_N', 'RC0_B_RAT_BAR', 'ISR_ABSRATROOT', 'ISR_QUOTA_IM', 'RC0_T', 'CONSTANT_USA_IM_R', 'USA_CPI9', 'CONSTANT_ISR_PXM_P', 'FILTER_ISR_XM_R', 'ISR_W3', 'USA_W3', 'RC0_COM2', 'RC0_CFOOD1', 'ISR_INTNFA1', 'ISR_GI_MPROP', 'RC0_PARTH5', 'FILTER_RC0_IFOOD_R', 'USA_TRANSFER_TARG_RAT_BAR', 'USA_RC0_SFOOD', 'RES_USA_INFEXP', 'CONSTANT_ISR_CPINCOM_P', 'ISR_RK5', 'USA_XM1', 'ISR_OILRECEIPT_N_BAR', 'ISR_CGBFX', 'USA_PXM2', 'USA_XM3', 'USA_POIL_P_SUB_BAR', 'FILTER_ISR_UNRH_FE', 'RC0_UNR2', 'FILTER_RC0_PC_P', 'USA_TAXOILBASE_N_BAR', 'RC0_SOV1', 'ISR_TAXL_RAT_BAR', 'RES_RC0_PRODMETAL_CYC_R', 'ISR_XM3', 'USA_XX_TAU_L', 'FILTER_ISR_IM_R', 'FILTER_RC0_OILPAY_N', 'RES_RC0_PRODFOOD_PERM_R', 'ISR_C_MPROP', 'ISR_GCDAMP', 'RES_USA_PRODOIL_R', 'USA_ISR_L', 'ISR_TSPILLX1', 'ISR_COM2', 'FILTER_ISR_PCW_P', 'RC0_XFOOD3', 'CONSTANT_RPFOOD_PERM', 'RC0_XMETAL4', 'FILTER_RC0_XFOOD_R', 'ISR_TFP_FE_R_BAR', 'FILTER_WTRADE_METAL_R', 'RC0_USA_SMETAL', 'USA_GPROD_SS', 'RES_RC0_OILSUB_N', 'RC0_LTAXDAMP', 'USA_INTGB1', 'USA_CPI11', 'CONSTANT_RC0_GDEF_TAR', 'FILTER_RC0_GDEF_TAR', 'FILTER_USA_IFOOD_R', 'USA_RK7', 'RC0_SIGMA', 'RC0_XOIL2', 'RES_USA_PRODFOOD_R', 'RC0_PARTH1', 'RC0_W2', 'FILTER_USA_ROYALTY', 'RES_RPMETAL_CYC', 'RC0_TSPILLM1', 'USA_COIL4', 'RPMETAL2', 'RC0_GREAL_SS', 'RC0_NSPILLM1', 'FILTER_USA_XMETAL_R', 'RES_RC0_INTRF', 'ISR_WEDGEFOOD_P', 'USA_KGTFPRATROOT', 'RES_USA_INTGB', 'CONSTANT_RC0_XOIL_R', 'ISR_XFOOD3', 'CONSTANT_RC0_K_R', 'RC0_INVEST_MPROP', 'ISR_IM3', 'RC0_PCOILBASE', 'RES_RC0_INTC', 'CONSTANT_WTRADE_FOOD_N', 'RC0_GCDAMP', 'RC0_PRODMETAL_R_EXOG', 'RC0_CPI12', 'CONSTANT_WTRADE_METAL_R', 'RC0_PIM2', 'CONSTANT_ISR_PARTH', 'FILTER_ISR_PXMF_P', 'RC0_CPI6', 'RC0_TRANSFERDAMP', 'RC0_COEFTFP', 'ISR_TAU_L_EXOG', 'RC0_CGBFX', 'ISR_TAU_C_EXOG', 'RC0_CPI11', 'RC0_TAU_C_EXOG', 'RES_RC0_INT', 'ISR_RC0_SMETAL', 'ISR_T', 'ISR_INTLAG', 'CONSTANT_RC0_ROYALTY', 'RES_ISR_INFWEXP', 'RPFOOD1', 'USA_NFANSHK', 'RC0_PCFOOD_P_BAR', 'RES_USA_GDEF_N', 'USA_PI1', 'USA_GI_RAT_BAR', 'RC0_KTAXDAMP', 'RES_RC0_INTCORP', 'ISR_POIL_P_SUB_BAR', 'FILTER_RC0_PI_P', 'CONSTANT_RC0_COIL_R', 'ISR_XM_SPILL_BAR', 'RC0_LTAXLRSW', 'FILTER_ISR_TFP_FE_R', 'ISR_TFOOD', 'FILTER_USA_K_R', 'RC0_TAU_L_EXOG', 'USA_TAXK_RAT_BAR', 'FILTER_USA_UNRH_FE', 'ISR_COIL4', 'PERM', 'RC0_INTGB1', 'FILTER_ISR_MPC', 'ISR_GDP_R_BAR', 'ISR_IMETAL2', 'USA_CHI', 'RC0_TAU_L_BAR', 'USA_MTFACTOR', 'ISR_KGTFPRATROOT', 'USA_IOIL3', 'FILTER_RC0_IOIL_R', 'RES_USA_UNRH', 'ISR_COIL1', 'USA_PRODOIL1', 'ISR_TFPRATROOT', 'RES_RC0_PCFOOD_P', 'ISR_USA_SMETAL', 'FILTER_ISR_GDEF_TAR', 'FILTER_RC0_ROYALTY', 'RC0_XOIL_R_BAR', 'USA_TRANSFER_RAT_EXOG', 'RC0_UOIL_R_BAR', 'CONSTANT_ISR_PRIMSUR_TAR', 'PERSIST_E_ISR_RPOIL_CYC', 'FILTER_RC0_COIL_R', 'FILTER_USA_OILSHARF', 'RC0_BGG3', 'RC0_XFOOD1', 'ISR_GREAL_SS', 'RC0_INTINFL', 'FILTER_USA_MROYALTY', 'RC0_COIL4', 'ISR_KTAXDAMP', 'ISR_BNSHK', 'FILTER_USA_IM_R', 'FILTER_USA_OILPAY_N', 'RES_USA_OILRECEIPT_N', 'RC0_GNP_R_BAR', 'TEMP_E_USA_TAXK_RAT', 'RES_RC0_UNRH', 'RC0_INTLAG', 'ISR_IM2', 'ISR_CPI6', 'ISR_IFOOD1', 'RES_ISR_W_R_AVG', 'USA_CPI5', 'ISR_RC0_L', 'RES_RC0_PRODOIL_R', 'USA_BGG2', 'RC0_XOIL4', 'USA_XX_TAU_C', 'CONSTANT_USA_CFOOD_R', 'USA_GCDAMP', 'ISR_TOIL', 'ISR_W1', 'ISR_SUB_XM', 'USA_IOIL1', 'CONSTANT_RC0_DELTA', 'CONSTANT_USA_TAU_OIL', 'ISR_IM4', 'USA_INTCOST_RAT_BAR', 'ISR_INTCORP1', 'USA_TAXCBASE_N_BAR', 'ISR_IOIL5', 'RC0_IFOOD5', 'ISR_C_OLG_R_BAR', 'ISR_RK7', 'CONSTANT_ISR_PI_P', 'RES_ISR_OILSUB_N', 'USA_IM1', 'RPMETAL3', 'CONSTANT_ISR_PCW_P', 'RES_RC0_TFP_FE_R_AVG', 'ISR_IFOOD5', 'RES_RC0_PRODMETAL_R', 'RC0_IM3', 'RES_RC0_PGDP_P_AVG', 'ISR_XX_LB', 'ISR_KTAXLRSW', 'RC0_GC_RAT_BAR', 'CONSTANT_RC0_MKTPREM', 'USA_PARTH5', 'FILTER_ISR_PC_P', 'CONSTANT_RC0_MROYALTY', 'ISR_GI_RAT_BAR', 'RC0_TRANSFER_TARG_RAT_BAR', 'ISR_DELTAG', 'RC0_IMETAL4', 'ISR_COM4', 'ISR_COEFZ', 'USA_T', 'RC0_XMETAL2', 'ISR_NSPILLM1', 'RC0_COIL3', 'ISR_GCLRSW', 'CONSTANT_RC0_INTNFA', 'FILTER_USA_CPI_P', 'ISR_IOIL3', 'ISR_TRANSFER_TARG_RAT_BAR', 'USA_W2', 'RC0_USA_SFOOD', 'RES_USA_RCORP', 'ISR_KREVAL', 'USA_IFOOD5', 'ISR_NSPILLX1', 'CONSTANT_USA_IMETAL_R', 'ISR_CFOOD2', 'FILTER_USA_NPOPB_R', 'USA_VMETAL', 'ISR_DAMP_DEBT', 'RC0_FLOOR', 'RC0_KGTFPRAT', 'USA_KGSPILLM1', 'CONSTANT_ISR_XM_R', 'ISR_GI_N_EXOG', 'RC0_MTFACTOR', 'RES_ISR_PCFOOD_P', 'USA_XX_TRANSFER_N', 'USA_XFOOD2', 'ISR_INFCPIX_TAR', 'RC0_INTCORP1', 'CONSTANT_USA_XOIL_R', 'CONSTANT_USA_COIL_R', 'USA_IFOOD3', 'USA_DELTAG', 'RC0_COM1', 'CONSTANT_ISR_DELTA', 'RES_RC0_INFWAGEEFF', 'ISR_CPI9', 'USA_COEFZ', 'CONSTANT_RC0_OILSHARF', 'CONSTANT_ISR_PC_P', 'USA_INTEPS', 'ISR_SPILLX1', 'RES_ISR_NFA_D', 'RC0_OILRECEIPT_N_BAR', 'RES_USA_RC', 'TRDEFOOD_R', 'RC0_INTNFA1', 'USA_FLOOR', 'CONSTANT_USA_IFOOD_R', 'FILTER_ISR_MKTPREM', 'ISR_RK6', 'USA_TSPILLM1', 'ISR_PARTH1', 'USA_KTAXLRSW', 'CONSTANT_USA_CPINCOM_P', 'RES_RC0_PRODFOOD_R', 'USA_GAMMA', 'ISR_XFOOD4', 'ISR_LSTAX_N', 'ISR_BGG2', 'USA_INVEST_MPROP', 'RPMETAL5', 'FILTER_RPOIL_PERM', 'CONSTANT_RC0_TAU_OIL', 'FILTER_USA_MKTPREM', 'USA_ISR_SFOOD', 'RC0_XX_TAU_C', 'ISR_CPI8', 'ISR_COM5', 'USA_IFOOD4', 'USA_VOIL', 'RC0_TAXC_RAT_BAR', 'USA_Q1', 'RC0_CPI5', 'RC0_VOIL', 'RC0_TFOOD', 'FILTER_RC0_SOVPREM', 'USA_TAU_L_BAR', 'RES_ISR_OILRECEIPT_N', 'CONSTANT_ISR_FXPREM', 'ISR_INTGAP', 'ISR_C_LIQ_R_BAR', 'RC0_BGG1', 'RC0_BREVAL1', 'CONSTANT_ISR_INTNFA', 'ISR_IFOOD2', 'PERSISTCOM_RC0', 'RC0_BGG2', 'USA_BGG1', 'RC0_USA_S', 'USA_TSPILLX1', 'RC0_ISR_L', 'USA_WEDGEOIL_P', 'USA_W1', 'RC0_MNFACTOR', 'RES_RPOIL_CYC_AVG', 'FILTER_RC0_DELTA', 'CONSTANT_RC0_IOIL_R', 'RC0_Q1', 'RES_RC0_C_LIQ_R', 'USA_NPOPH_R_BAR', 'RC0_ALPHA_KG', 'ISR_BREVAL1', 'USA_SOV2', 'ISR_TFPRAT', 'PERM_E_MPC', 'USA_KGTFPRAT', 'USA_INTC1', 'FILTER_RC0_CPI_P', 'RES_USA_PRODOIL_CYC_R', 'USA_IMETAL2', 'WRL_PRODOIL_R_BAR', 'RES_USA_INT', 'USA_PIM2', 'RC0_Q2', 'RES_RC0_GDEF_N', 'TEMP_E_RPFOOD_CYC', 'USA_IFOOD2', 'USA_LTAXLRSW', 'ISR_GC_N_EXOG', 'ISR_LTAXDAMP', 'ISR_INTINFL', 'RC0_PRODOIL2', 'ISR_PI1', 'USA_OILRECEIPT_N_BAR', 'USA_PIM1', 'PERSIST_E_USA_RPMETAL_CYC', 'FILTER_RC0_XMETAL_R', 'CONSTANT_ISR_SOVPREM', 'ISR_GAMMA', 'ISR_POIL_P_BAR', 'CONSTANT_RC0_MPC', 'ISR_NPOPH_R_BAR', 'ISR_IOIL2', 'FILTER_USA_MPC', 'RES_RC0_OILRECEIPT_N', 'RES_ISR_INFCPIX', 'USA_XX_LB', 'ISR_Z_BAR', 'RC0_GI_R_BAR', 'ISR_PRODFOODSHARE', 'RES_ISR_INFWAGEEFF', 'FILTER_USA_FXPREM', 'RES_USA_PRODMETAL_CYC_R', 'RC0_IFOOD3', 'USA_XM2', 'RPOIL2', 'RC0_CPI3', 'RES_RPMETAL_PERM_AVG', 'RC0_IOIL4', 'FILTER_RC0_INTNFA', 'FILTER_RC0_UNRH_FE', 'RC0_GDP_N_BAR', 'RC0_CFOOD4', 'RES_ISR_TFP_FE_R_AVG', 'RC0_PRODFOODSHARE', 'FILTER_RC0_OILSHARF', 'USA_XM_MPROP', 'ISR_FLOOR', 'RES_ISR_EPS', 'RC0_KGSPILLM1', 'USA_COIL3', 'CONSTANT_RC0_XFOOD_R', 'RC0_COEFPGDP', 'TEMP_E_RC0_RPFOOD_CYC', 'ISR_CFOOD3', 'RES_USA_EPS', 'RES_USA_PRODFOOD_PERM_R', 'TEMP_E_RC0_TAXK_RAT', 'FILTER_ISR_IFOOD_R', 'RES_USA_PCFOOD_P', 'USA_PRODMETAL2', 'FILTER_ISR_DELTA', 'RES_ISR_PCOIL_P', 'RC0_GPROD_SS', 'RC0_BETA', 'ISR_PCFOOD_P_BAR', 'RC0_RK4', 'USA_PCOILBASE', 'FILTER_ISR_RPREM', 'RES_RPFOOD_CYC', 'RC0_TRANSFERSHARE', 'RC0_QUOTA_IM', 'WRL_PRODMETAL_R_BAR', 'RC0_TRANSFERLRSW', 'CONSTANT_ISR_CPI_P', 'CONSTANT_RC0_PI_P', 'USA_GC_N_EXOG', 'ISR_IFOOD3', 'CONSTANT_RC0_UNRH_FE', 'ISR_TAXKBASE_N_BAR', 'RES_RC0_WH_R', 'RC0_TAXL_RAT_BAR', 'ISR_XM1', 'ISR_XX_GDEF_N', 'FILTER_RC0_IMETAL_R', 'CONSTANT_RC0_IM_R', 'RC0_XM_SPILL_BAR', 'ISR_CPI12', 'RC0_XX_GDEF_N', 'CONSTANT_USA_GDEF_TAR', 'FILTER_USA_TFP_FE_R', 'ISR_XX_TAU_L', 'USA_QUOTA_IM', 'ISR_PXM1', 'TEMP_RC0', 'USA_TAXKBASE_N_BAR', 'USA_C_OLG_R_BAR', 'FILTER_USA_PXMF_P', 'USA_COEFPARTH', 'ISR_UNR1', 'ISR_GC_RAT_BAR', 'ISR_COEFW', 'RES_ISR_UNRH', 'USA_RK6', 'TRDEOIL_N', 'CONSTANT_RC0_XM_R', 'RC0_TFPRAT', 'ISR_WEXP1', 'USA_GREAL_SS', 'CONSTANT_RC0_PG_P', 'PERSIST_E_ISR_RPMETAL_CYC', 'USA_POIL_P_BAR', 'CONSTANT_USA_IOIL_R', 'TRDEM_N', 'USA_RK1', 'USA_R_NEUT_BAR', 'CONSTANT_RC0_FXPREM', 'USA_IOIL5', 'ISR_RK2', 'USA_MFACTOR', 'USA_PROB', 'ISR_SOV1', 'RES_ISR_INTGB', 'USA_UNR1', 'RES_RC0_PARTH_DES', 'FILTER_USA_XOIL_R', 'FILTER_ISR_NPOPB_R', 'CONSTANT_RC0_IFOOD_R', 'RC0_SUB_XM', 'RPOIL_ROOT', 'USA_INTGAP', 'RC0_TRANSFER_TARG_RAT_EXOG', 'RES_ISR_INTC', 'RES_USA_PIM_P', 'CONSTANT_ISR_PXMF_P', 'RC0_GCLRSW', 'RES_RC0_INFEXP', 'ISR_RC0_SOIL', 'ISR_CPI11', 'CONSTANT_WTRADE_M_R', 'RC0_CFOOD2', 'ISR_MTFACTOR', 'ISR_BGG3', 'RC0_COSTOIL_P', 'ISR_TRANSFER_RAT_BAR', 'USA_XX_PRIMSUR_N', 'RC0_PI1', 'ISR_CHI', 'ISR_GILRSW', 'USA_ALPHA_FE', 'RC0_TRANSFER_RAT_EXOG', 'RC0_CPI8', 'RES_ISR_WO_R', 'USA_IFOOD1', 'RC0_XX_TAU_K', 'USA_IM_SPILL_BAR', 'USA_EBASE', 'CONSTANT_USA_TFP_FE_R', 'ISR_INTGB1', 'USA_XM4', 'RES_RC0_WO_R', 'FILTER_WTRADE_M_R', 'FILTER_RC0_IM_R', 'FILTER_RC0_MKTPREM', 'RC0_ISR_SFOOD', 'RC0_SPILLM1', 'RC0_RK6', 'RC0_INTEPS', 'ISR_CPI1', 'FILTER_WTRADE_OIL_R', 'CONSTANT_ISR_IMETAL_R', 'USA_GC_MPROP', 'FILTER_WTRADE_OIL_N', 'RES_RC0_INTGB', 'RC0_PARTH4', 'CONSTANT_USA_PC_P', 'USA_TARGDAMP', 'ISR_IM_SPILL_BAR', 'FILTER_ISR_SOVPREM', 'RES_USA_Q_P', 'USA_TFPRATROOT', 'RC0_IOIL5', 'CONSTANT_RC0_CFOOD_R', 'CONSTANT_RC0_NPOPB_R', 'FILTER_USA_CFOOD_R', 'RC0_COIL2', 'RES_ISR_GDEF_N', 'RES_ISR_RC', 'RPOIL5', 'ISR_Q2', 'ISR_MKT1', 'ISR_TRANSFERLRSW', 'RC0_UNR1', 'RC0_CPI10', 'FILTER_WTRADE_METAL_N', 'ISR_PARTH2', 'ISR_CPI5', 'CONSTANT_WTRADE_METAL_N', 'USA_WT', 'RES_USA_INFWAGEEFF', 'USA_TRANSFERSHARE', 'ISR_VOIL', 'USA_KREVAL', 'ISR_LTAXLRSW', 'CONSTANT_ISR_TPREM', 'RC0_MFACTOR', 'RC0_PCFOODBASE', 'ISR_XX_OILSUB_N', 'USA_XX_OILSUB_N', 'ISR_PIM1', 'RC0_PSUB_XM', 'USA_ABSRATROOT', 'RC0_XX_GC_N', 'RC0_USA_L', 'USA_XFOOD4', 'FILTER_USA_PC_P', 'RES_RC0_PRODOIL_CYC_R', 'RES_RC0_R_NEUT', 'RC0_TAXLBASE_N_BAR', 'RC0_KGTFPRATROOT', 'CONSTANT_USA_MROYALTY', 'RC0_C_LAMBDA', 'ISR_ALPHA_KG', 'RC0_NPOPH_R_BAR', 'TEMP_E_ISR_RPFOOD_CYC', 'USA_XX_GDEF_N', 'RES_RC0_INFCPIX', 'ISR_TARGDAMP', 'CONSTANT_USA_PRIMSUR_TAR', 'ISR_UNR2', 'FILTER_USA_COIL_R', 'RPOIL_CYC_BAR', 'RC0_WEXP1', 'RC0_PARTH_ADJ_STAR', 'RC0_ALPHA_FE', 'ISR_NFANSHK', 'ISR_IOIL4', 'USA_KTAXDAMP', 'ISR_RC0_S', 'RES_USA_R_NEUT', 'RC0_IOIL1', 'RC0_TFP_FE_R_BAR', 'ISR_RC0_SFOOD', 'CONSTANT_USA_RPREM', 'TRDEM_R', 'RC0_DAMP_DEBT', 'RES_RC0_NFA_D', 'USA_GCLRSW', 'FILTER_USA_C_OLG_R', 'CONSTANT_RC0_XMETAL_R', 'CONSTANT_WTRADE_OIL_R', 'CONSTANT_USA_XM_R', 'USA_TAU_K_EXOG', 'RC0_LAPH', 'ISR_PROB', 'RC0_KREVAL', 'RES_USA_INT10', 'FILTER_USA_PG_P', 'RES_USA_TAXOIL_N', 'RC0_TAU_K_EXOG', 'FILTER_RC0_CPINCOM_P', 'CONSTANT_ISR_COIL_R', 'CONSTANT_ISR_TAU_OIL', 'RC0_XOIL5', 'USA_INTWT', 'RC0_IOIL3', 'CONSTANT_ISR_MPC', 'USA_PARTH3', 'ISR_CTAXDAMP', 'FILTER_RC0_PXMF_P', 'RES_ISR_Z_AVG', 'USA_ABSRAT', 'RC0_XX_LB', 'RC0_POIL_P_BAR', 'CONSTANT_RC0_TFP_FE_R', 'RES_RC0_Q_P', 'ISR_GIDAMP', 'USA_INFCPIX_TAR', 'RC0_ABSRAT', 'ISR_TARIFF_IM', 'USA_CPI12', 'RC0_TAXCBASE_N_BAR', 'USA_RC0_L', 'RC0_RK7', 'RC0_IFOOD2', 'CONSTANT_USA_C_OLG_R', 'RPFOOD3', 'ISR_XX_TAU_K', 'FILTER_USA_PI_P', 'ISR_DAMP_GDP_GAP', 'RC0_RK2', 'USA_TMETAL', 'RC0_NPOP_R_EXOG', 'USA_WEXP1', 'FILTER_ISR_TAU_OIL', 'ISR_COIL3', 'USA_CPI3', 'RC0_GI_MPROP', 'RES_ISR_INT', 'USA_PCFOOD_P_BAR', 'USA_IMETAL3', 'USA_TFPRAT', 'RES_RC0_MROYALTIES_N', 'USA_CPI2', 'CONSTANT_RC0_C_OLG_R', 'RES_ISR_RCORP', 'USA_PARTH_ADJ_STAR', 'RC0_PXM2', 'USA_TRANSFER_RAT_BAR', 'ISR_CPI10', 'FILTER_RC0_TAU_OIL', 'RC0_GILRSW', 'ISR_TRANSFER_TARG_RAT_EXOG', 'RES_RPOIL_CYC', 'USA_CPI10', 'RES_USA_PRODMETAL_R', 'USA_GI_N_EXOG', 'RC0_TAXOILBASE_N_BAR', 'CONSTANT_RC0_PRIMSUR_TAR', 'USA_XFOOD1', 'USA_XFOOD3', 'RC0_TOIL', 'FILTER_RPMETAL_PERM', 'ISR_IMETAL4', 'RPOIL1', 'RES_USA_INTMP', 'ISR_NPOP_R_EXOG', 'RES_ISR_Z_NFA', 'USA_PARTH1', 'RC0_PRODOIL_R_BAR', 'USA_INTMP_BAR', 'USA_LSTAX_N', 'ISR_MFACTOR', 'CONSTANT_RC0_PXMF_P', 'USA_TRANSFERLRSW', 'FILTER_RC0_PARTH', 'RC0_XOIL1', 'RC0_GIDAMP', 'CONSTANT_USA_PARTH', 'ISR_COM1', 'ISR_ABSRAT', 'USA_XMETAL3', 'USA_CPI8', 'RC0_ABSRATROOT', 'RC0_C_LIQ_R_BAR', 'ISR_USA_S', 'RC0_IM2', 'USA_B_STAR', 'USA_MKT1', 'RPOIL4', 'RC0_MKT1', 'TRDEMETAL_R', 'CONSTANT_ISR_PG_P', 'USA_NSPILLX1', 'RES_USA_GNP_R', 'RES_ISR_INTMP', 'FILTER_ISR_PI_P', 'CONSTANT_ISR_UNRH_FE', 'RES_USA_INTCORP', 'RES_RC0_W_R_AVG', 'USA_PRODOIL_R_BAR', 'RC0_INTWT', 'RC0_PRODOIL_R_EXOG', 'RES_USA_Z_AVG', 'USA_IMETAL5', 'CONSTANT_USA_PCW_P', 'RC0_PROB', 'RC0_EPS_TAR', 'CONSTANT_WTRADE_OIL_N', 'RC0_PARTH2', 'RES_RC0_EPS', 'ISR_INTGR', 'RC0_RK1', 'RES_USA_INFWEXP', 'USA_TFOOD', 'USA_COSTMETAL_P', 'RC0_PARTH3', 'USA_TAXLBASE_N_BAR', 'FILTER_USA_XM_R', 'ISR_GDP_N_BAR', 'FILTER_USA_CPINCOM_P', 'USA_XOIL5', 'USA_XX_TAU_K', 'RES_RC0_BREVAL_N', 'ISR_PARTH3', 'RC0_XOIL3', 'RC0_COIL1', 'USA_TRANSFER_TARG_RAT_EXOG', 'USA_IM2', 'RES_USA_WH_R', 'RC0_GC_MPROP', 'RC0_GDP_R_BAR', 'FILTER_USA_IOIL_R', 'CONSTANT_USA_TPREM', 'USA_GI_R_BAR', 'RC0_TRANSFER_RAT_BAR', 'RES_USA_PGDP_P_AVG', 'USA_MNFACTOR', 'RES_ISR_TAXOIL_N', 'CONSTANT_USA_UNRH_FE', 'ISR_PXM2', 'USA_CPI6', 'ISR_USA_SFOOD', 'USA_COIL1', 'RC0_TFPRATROOT', 'RES_ISR_PRODFOOD_PERM_R', 'RES_USA_TFP_FE_R_AVG', 'ISR_PCFOODBASE', 'RES_PMETAL_PERM_P']
    
    # mod_eqs, new_endog, map_new_endog, leads, lags = fixEquations(eqs=[eq],endog=endog,params=params)
    # print(mod_eqs)
    
    # eq = """ISR_PARTH_FE = (ISR_PARTH_FE(-1)/(ISR_NPOPH_R_BAR(-1)/ISR_NPOPH_R(-1))**ISR_PARTH5-RES_ISR_PARTH_FE-E_ISR_PARTH_FE(-1)-E_ISR_PART_FE(-1)*ISR_NPOP_R(-1)/ISR_NPOPH_R(-1))**ISR_COEFPARTH*ISR_PARTH_ADJ_STAR**(1-ISR_COEFPARTH)*(ISR_NPOPH_R_BAR/ISR_NPOPH_R)**ISR_PARTH5+RES_ISR_PARTH_FE+E_ISR_PARTH_FE+E_ISR_PART_FE*ISR_NPOP_R/ISR_NPOPH_R"""
    # endog = ['E_WRL_PRODOIL_CYC_R', 'ISR_ACT_R', 'ISR_BREVAL_N', 'ISR_B_N', 'ISR_B_RAT', 'ISR_CFOOD_R', 'ISR_CNCOM_R', 'ISR_COIL_R', 'ISR_COMEFFECT_R', 'ISR_COM_FE_R', 'ISR_CPI13', 'ISR_CPINCOM_P', 'ISR_CPIX_P', 'ISR_CPI_P', 'ISR_CURBAL_N', 'ISR_C_LIQ_R', 'ISR_C_OLG_R', 'ISR_C_R', 'ISR_C_RAT', 'ISR_DELTA', 'ISR_EPS', 'ISR_FACTFOOD_R', 'ISR_FACT_R', 'ISR_FXPREM', 'ISR_GC_N', 'ISR_GC_R', 'ISR_GC_RAT', 'ISR_GDEF_N', 'ISR_GDEF_RAT', 'ISR_GDEF_TAR', 'ISR_GDPINC_N', 'ISR_GDPSIZE', 'ISR_GDP_FE_R', 'ISR_GDP_N', 'ISR_GDP_R', 'ISR_GE_N', 'ISR_GISTOCK_R', 'ISR_GI_N', 'ISR_GI_R', 'ISR_GI_RAT', 'ISR_GNP_R', 'ISR_GOVCHECK', 'ISR_GSUB_N', 'ISR_GTARIFF_N', 'ISR_G_R', 'ISR_IFOODA_R', 'ISR_IFOOD_R', 'ISR_IMETALA_R', 'ISR_IMETAL_R', 'ISR_IM_R', 'ISR_INFCPI', 'ISR_INFCPIX', 'ISR_INFEXP', 'ISR_INFL', 'ISR_INFPIM', 'ISR_INFWAGE', 'ISR_INFWAGEEFF', 'ISR_INFWEXP', 'ISR_INT', 'ISR_INT10', 'ISR_INTC', 'ISR_INTCORP', 'ISR_INTCOST_N', 'ISR_INTCOST_RAT', 'ISR_INTGB', 'ISR_INTMP', 'ISR_INTMPU', 'ISR_INTNFA', 'ISR_INTRF', 'ISR_INTRF10', 'ISR_INTXM10', 'ISR_INVESTP_R', 'ISR_INVEST_R', 'ISR_INVEST_RAT', 'ISR_IOILA_R', 'ISR_IOIL_R', 'ISR_IT_R', 'ISR_IT_RAT', 'ISR_J', 'ISR_KG_R', 'ISR_K_R', 'ISR_LABH_FE_R', 'ISR_LABH_R', 'ISR_LAB_FE_R', 'ISR_LAB_R', 'ISR_LF_FE_R', 'ISR_LF_R', 'ISR_LSTAX_RAT', 'ISR_MKTPREM', 'ISR_MKTPREMSM', 'ISR_MPC', 'ISR_MPCINV', 'ISR_NFAREVAL_N', 'ISR_NFA_D', 'ISR_NFA_RAT', 'ISR_NPOPB_R', 'ISR_NPOPH_R', 'ISR_NPOP_R', 'ISR_NTRFPSPILL_FE_R', 'ISR_OILRECEIPT_N', 'ISR_OILSUB_N', 'ISR_PART', 'ISR_PARTH', 'ISR_PARTH_DES', 'ISR_PARTH_FE', 'ISR_PART_DES', 'ISR_PART_FE', 'ISR_PCFOOD_P', 'ISR_PCOIL_P', 'ISR_PCW_P', 'ISR_PC_P', 'ISR_PFM_P', 'ISR_PFOOD_P', 'ISR_PGDP_P', 'ISR_PGDP_P_AVG', 'ISR_PG_P', 'ISR_PIMADJ_P', 'ISR_PIMA_P', 'ISR_PIM_P', 'ISR_PIT_P', 'ISR_PI_P', 'ISR_PMETAL_CYC_P', 'ISR_PMETAL_P', 'ISR_PMETAL_PERM_P', 'ISR_POIL_CYC_P', 'ISR_POIL_P', 'ISR_POIL_PERM_P', 'ISR_POIL_P_SUB', 'ISR_PRIMSUR_N', 'ISR_PRIMSUR_TAR', 'ISR_PRODFOOD_CYC_R', 'ISR_PRODFOOD_PERM_R', 'ISR_PRODFOOD_R', 'ISR_PSAVING_N', 'ISR_PXMF_P', 'ISR_PXMUNADJ_P', 'ISR_PXM_P', 'ISR_PXT_P', 'ISR_Q_P', 'ISR_R', 'ISR_R10', 'ISR_RC', 'ISR_RC0_SM', 'ISR_RC0_WM', 'ISR_RCI', 'ISR_RCORP', 'ISR_REER', 'ISR_RK8', 'ISR_RK_P', 'ISR_RPREM', 'ISR_R_NEUT', 'ISR_SOVPREM', 'ISR_SOVPREMSM', 'ISR_SUB_OIL', 'ISR_TAU_C', 'ISR_TAU_K', 'ISR_TAU_L', 'ISR_TAU_OIL', 'ISR_TAXC_N', 'ISR_TAXC_RAT', 'ISR_TAXK_N', 'ISR_TAXK_RAT', 'ISR_TAXLH_N', 'ISR_TAXL_N', 'ISR_TAXL_RAT', 'ISR_TAXOIL_N', 'ISR_TAX_N', 'ISR_TAX_RAT', 'ISR_TB_N', 'ISR_TFPEFFECT_R', 'ISR_TFPKGSPILL_FE_R', 'ISR_TFPSPILL_FE_R', 'ISR_TFP_FE_R', 'ISR_TFP_FE_R_AVG', 'ISR_TFP_R', 'ISR_TM', 'ISR_TPREM', 'ISR_TRANSFER_LIQ_N', 'ISR_TRANSFER_N', 'ISR_TRANSFER_OLG_N', 'ISR_TRANSFER_RAT', 'ISR_TRANSFER_TARG_N', 'ISR_TRANSFER_TARG_RAT', 'ISR_TRFPSPILL_FE_R', 'ISR_UFOOD_R', 'ISR_UNR', 'ISR_UNRH', 'ISR_UNRH_FE', 'ISR_UNR_FE', 'ISR_USA_SM', 'ISR_USA_WM', 'ISR_WAGEEFF_N', 'ISR_WAGEH_N', 'ISR_WAGE_N', 'ISR_WF_R', 'ISR_WH_R', 'ISR_WK_N', 'ISR_WO_R', 'ISR_W_R', 'ISR_W_R_AVG', 'ISR_XFOOD_R', 'ISR_XMA_R', 'ISR_XM_R', 'ISR_XT_R', 'ISR_XT_RAT', 'ISR_YCAP_N', 'ISR_YD_R', 'ISR_YLABH_N', 'ISR_YLAB_N', 'ISR_Z', 'ISR_Z_AVG', 'ISR_Z_NFA', 'PFOOD_P', 'PMETAL_CYC_P', 'PMETAL_P', 'PMETAL_PERM_P', 'POIL_CYC_P', 'POIL_P', 'POIL_PERM_P', 'RC0_ACT_R', 'RC0_BREVAL_N', 'RC0_B_N', 'RC0_B_RAT', 'RC0_CFOOD_R', 'RC0_CNCOM_R', 'RC0_COIL_R', 'RC0_COMEFFECT_R', 'RC0_COM_FE_R', 'RC0_CPI13', 'RC0_CPINCOM_P', 'RC0_CPIX_P', 'RC0_CPI_P', 'RC0_CURBAL_N', 'RC0_C_LIQ_R', 'RC0_C_OLG_R', 'RC0_C_R', 'RC0_C_RAT', 'RC0_DELTA', 'RC0_EPS', 'RC0_FACTFOOD_R', 'RC0_FACTMETAL_R', 'RC0_FACTOIL_R', 'RC0_FACT_R', 'RC0_FXPREM', 'RC0_GC_N', 'RC0_GC_R', 'RC0_GC_RAT', 'RC0_GDEF_N', 'RC0_GDEF_RAT', 'RC0_GDEF_TAR', 'RC0_GDPINC_N', 'RC0_GDPSIZE', 'RC0_GDP_FE_R', 'RC0_GDP_N', 'RC0_GDP_R', 'RC0_GE_N', 'RC0_GISTOCK_R', 'RC0_GI_N', 'RC0_GI_R', 'RC0_GI_RAT', 'RC0_GNP_R', 'RC0_GOVCHECK', 'RC0_GSUB_N', 'RC0_GTARIFF_N', 'RC0_G_R', 'RC0_IFOODA_R', 'RC0_IFOOD_R', 'RC0_IMETALA_R', 'RC0_IMETAL_R', 'RC0_IM_R', 'RC0_INFCPI', 'RC0_INFCPIX', 'RC0_INFEXP', 'RC0_INFL', 'RC0_INFPIM', 'RC0_INFWAGE', 'RC0_INFWAGEEFF', 'RC0_INFWEXP', 'RC0_INT', 'RC0_INT10', 'RC0_INTC', 'RC0_INTCORP', 'RC0_INTCOST_N', 'RC0_INTCOST_RAT', 'RC0_INTGB', 'RC0_INTMP', 'RC0_INTMPU', 'RC0_INTNFA', 'RC0_INTRF', 'RC0_INTRF10', 'RC0_INTXM10', 'RC0_INVESTP_R', 'RC0_INVEST_R', 'RC0_INVEST_RAT', 'RC0_IOILA_R', 'RC0_IOIL_R', 'RC0_ISR_SM', 'RC0_ISR_WM', 'RC0_IT_R', 'RC0_IT_RAT', 'RC0_J', 'RC0_KG_R', 'RC0_K_R', 'RC0_LABH_FE_R', 'RC0_LABH_R', 'RC0_LAB_FE_R', 'RC0_LAB_R', 'RC0_LF_FE_R', 'RC0_LF_R', 'RC0_LSTAX_RAT', 'RC0_MKTPREM', 'RC0_MKTPREMSM', 'RC0_MPC', 'RC0_MPCINV', 'RC0_MROYALTIES_N', 'RC0_MROYALTY', 'RC0_NFAREVAL_N', 'RC0_NFA_D', 'RC0_NFA_RAT', 'RC0_NPOPB_R', 'RC0_NPOPH_R', 'RC0_NPOP_R', 'RC0_NTRFPSPILL_FE_R', 'RC0_OILPAY_N', 'RC0_OILRECEIPT_N', 'RC0_OILSHARF', 'RC0_OILSUB_N', 'RC0_PART', 'RC0_PARTH', 'RC0_PARTH_DES', 'RC0_PARTH_FE', 'RC0_PART_DES', 'RC0_PART_FE', 'RC0_PCFOOD_P', 'RC0_PCOIL_P', 'RC0_PCW_P', 'RC0_PC_P', 'RC0_PFM_P', 'RC0_PFOOD_P', 'RC0_PGDP_P', 'RC0_PGDP_P_AVG', 'RC0_PG_P', 'RC0_PIMADJ_P', 'RC0_PIMA_P', 'RC0_PIM_P', 'RC0_PIT_P', 'RC0_PI_P', 'RC0_PMETAL_CYC_P', 'RC0_PMETAL_P', 'RC0_PMETAL_PERM_P', 'RC0_POIL_CYC_P', 'RC0_POIL_P', 'RC0_POIL_PERM_P', 'RC0_POIL_P_SUB', 'RC0_PRIMSUR_N', 'RC0_PRIMSUR_TAR', 'RC0_PRODFOOD_CYC_R', 'RC0_PRODFOOD_PERM_R', 'RC0_PRODFOOD_R', 'RC0_PRODMETAL_CYC_R', 'RC0_PRODMETAL_PERM_R', 'RC0_PRODMETAL_R', 'RC0_PRODOIL_CYC_R', 'RC0_PRODOIL_PERM_R', 'RC0_PRODOIL_R', 'RC0_PSAVING_N', 'RC0_PXMF_P', 'RC0_PXMUNADJ_P', 'RC0_PXM_P', 'RC0_PXT_P', 'RC0_Q_P', 'RC0_R', 'RC0_R10', 'RC0_RC', 'RC0_RCI', 'RC0_RCORP', 'RC0_REER', 'RC0_RK8', 'RC0_RK_P', 'RC0_ROYALTIES_N', 'RC0_ROYALTY', 'RC0_RPREM', 'RC0_R_NEUT', 'RC0_SOVPREM', 'RC0_SOVPREMSM', 'RC0_SUB_OIL', 'RC0_TAU_C', 'RC0_TAU_K', 'RC0_TAU_L', 'RC0_TAU_OIL', 'RC0_TAXC_N', 'RC0_TAXC_RAT', 'RC0_TAXK_N', 'RC0_TAXK_RAT', 'RC0_TAXLH_N', 'RC0_TAXL_N', 'RC0_TAXL_RAT', 'RC0_TAXOIL_N', 'RC0_TAX_N', 'RC0_TAX_RAT', 'RC0_TB_N', 'RC0_TFPEFFECT_R', 'RC0_TFPKGSPILL_FE_R', 'RC0_TFPSPILL_FE_R', 'RC0_TFP_FE_R', 'RC0_TFP_FE_R_AVG', 'RC0_TFP_R', 'RC0_TM', 'RC0_TPREM', 'RC0_TRANSFER_LIQ_N', 'RC0_TRANSFER_N', 'RC0_TRANSFER_OLG_N', 'RC0_TRANSFER_RAT', 'RC0_TRANSFER_TARG_N', 'RC0_TRANSFER_TARG_RAT', 'RC0_TRFPSPILL_FE_R', 'RC0_UFOOD_R', 'RC0_UMETAL_R', 'RC0_UNR', 'RC0_UNRH', 'RC0_UNRH_FE', 'RC0_UNR_FE', 'RC0_UOIL_R', 'RC0_USA_SM', 'RC0_USA_WM', 'RC0_WAGEEFF_N', 'RC0_WAGEH_N', 'RC0_WAGE_N', 'RC0_WF_R', 'RC0_WH_R', 'RC0_WK_N', 'RC0_WO_R', 'RC0_W_R', 'RC0_W_R_AVG', 'RC0_XFOOD_R', 'RC0_XMA_R', 'RC0_XMETAL_R', 'RC0_XM_R', 'RC0_XOIL_R', 'RC0_XT_R', 'RC0_XT_RAT', 'RC0_YCAP_N', 'RC0_YD_R', 'RC0_YLABH_N', 'RC0_YLAB_N', 'RC0_Z', 'RC0_Z_AVG', 'RC0_Z_NFA', 'RPFOOD', 'RPFOOD_CYC', 'RPFOOD_PERM', 'RPMETAL', 'RPMETAL_CYC', 'RPMETAL_CYC_AVG', 'RPMETAL_PERM', 'RPMETAL_PERM_AVG', 'RPOIL', 'RPOIL_CYC', 'RPOIL_CYC_AVG', 'RPOIL_PERM', 'RPOIL_PERM_AVG', 'USA_ACT_R', 'USA_BREVAL_N', 'USA_B_N', 'USA_B_RAT', 'USA_CFOOD_R', 'USA_CNCOM_R', 'USA_COIL_R', 'USA_COMEFFECT_R', 'USA_COM_FE_R', 'USA_CPI13', 'USA_CPINCOM_P', 'USA_CPIX_P', 'USA_CPI_P', 'USA_CURBAL_N', 'USA_C_LIQ_R', 'USA_C_OLG_R', 'USA_C_R', 'USA_C_RAT', 'USA_DELTA', 'USA_EPS', 'USA_FACTFOOD_R', 'USA_FACTMETAL_R', 'USA_FACTOIL_R', 'USA_FACT_R', 'USA_FXPREM', 'USA_GC_N', 'USA_GC_R', 'USA_GC_RAT', 'USA_GDEF_N', 'USA_GDEF_RAT', 'USA_GDEF_TAR', 'USA_GDPINC_N', 'USA_GDPSIZE', 'USA_GDP_FE_R', 'USA_GDP_N', 'USA_GDP_R', 'USA_GE_N', 'USA_GISTOCK_R', 'USA_GI_N', 'USA_GI_R', 'USA_GI_RAT', 'USA_GNP_R', 'USA_GOVCHECK', 'USA_GSUB_N', 'USA_GTARIFF_N', 'USA_G_R', 'USA_IFOODA_R', 'USA_IFOOD_R', 'USA_IMETALA_R', 'USA_IMETAL_R', 'USA_IM_R', 'USA_INFCPI', 'USA_INFCPIX', 'USA_INFEXP', 'USA_INFL', 'USA_INFPIM', 'USA_INFWAGE', 'USA_INFWAGEEFF', 'USA_INFWEXP', 'USA_INT', 'USA_INT10', 'USA_INTC', 'USA_INTCORP', 'USA_INTCOST_N', 'USA_INTCOST_RAT', 'USA_INTGB', 'USA_INTMP', 'USA_INTMPU', 'USA_INTNFA', 'USA_INTRF', 'USA_INTRF10', 'USA_INTXM10', 'USA_INVESTP_R', 'USA_INVEST_R', 'USA_INVEST_RAT', 'USA_IOILA_R', 'USA_IOIL_R', 'USA_ISR_SM', 'USA_ISR_WM', 'USA_IT_R', 'USA_IT_RAT', 'USA_J', 'USA_KG_R', 'USA_K_R', 'USA_LABH_FE_R', 'USA_LABH_R', 'USA_LAB_FE_R', 'USA_LAB_R', 'USA_LF_FE_R', 'USA_LF_R', 'USA_LSTAX_RAT', 'USA_MKTPREM', 'USA_MKTPREMSM', 'USA_MPC', 'USA_MPCINV', 'USA_MROYALTIES_N', 'USA_MROYALTY', 'USA_NFAREVAL_N', 'USA_NFA_D', 'USA_NFA_RAT', 'USA_NPOPB_R', 'USA_NPOPH_R', 'USA_NPOP_R', 'USA_NTRFPSPILL_FE_R', 'USA_OILPAY_N', 'USA_OILRECEIPT_N', 'USA_OILSHARF', 'USA_OILSUB_N', 'USA_PART', 'USA_PARTH', 'USA_PARTH_DES', 'USA_PARTH_FE', 'USA_PART_DES', 'USA_PART_FE', 'USA_PCFOOD_P', 'USA_PCOIL_P', 'USA_PCW_P', 'USA_PC_P', 'USA_PFM_P', 'USA_PFOOD_P', 'USA_PGDP_P', 'USA_PGDP_P_AVG', 'USA_PG_P', 'USA_PIMADJ_P', 'USA_PIMA_P', 'USA_PIM_P', 'USA_PIT_P', 'USA_PI_P', 'USA_PMETAL_CYC_P', 'USA_PMETAL_P', 'USA_PMETAL_PERM_P', 'USA_POIL_CYC_P', 'USA_POIL_P', 'USA_POIL_PERM_P', 'USA_POIL_P_SUB', 'USA_PRIMSUR_N', 'USA_PRIMSUR_TAR', 'USA_PRODFOOD_CYC_R', 'USA_PRODFOOD_PERM_R', 'USA_PRODFOOD_R', 'USA_PRODMETAL_CYC_R', 'USA_PRODMETAL_PERM_R', 'USA_PRODMETAL_R', 'USA_PRODOIL_CYC_R', 'USA_PRODOIL_PERM_R', 'USA_PRODOIL_R', 'USA_PSAVING_N', 'USA_PXMF_P', 'USA_PXMUNADJ_P', 'USA_PXM_P', 'USA_PXT_P', 'USA_Q_P', 'USA_R', 'USA_R10', 'USA_RC', 'USA_RC0_SM', 'USA_RC0_WM', 'USA_RCI', 'USA_RCORP', 'USA_REER', 'USA_RK8', 'USA_RK_P', 'USA_ROYALTIES_N', 'USA_ROYALTY', 'USA_RPREM', 'USA_R_NEUT', 'USA_SOVPREM', 'USA_SOVPREMSM', 'USA_SUB_OIL', 'USA_TAU_C', 'USA_TAU_K', 'USA_TAU_L', 'USA_TAU_OIL', 'USA_TAXC_N', 'USA_TAXC_RAT', 'USA_TAXK_N', 'USA_TAXK_RAT', 'USA_TAXLH_N', 'USA_TAXL_N', 'USA_TAXL_RAT', 'USA_TAXOIL_N', 'USA_TAX_N', 'USA_TAX_RAT', 'USA_TB_N', 'USA_TFPEFFECT_R', 'USA_TFPKGSPILL_FE_R', 'USA_TFPSPILL_FE_R', 'USA_TFP_FE_R', 'USA_TFP_FE_R_AVG', 'USA_TFP_R', 'USA_TM', 'USA_TPREM', 'USA_TRANSFER_LIQ_N', 'USA_TRANSFER_N', 'USA_TRANSFER_OLG_N', 'USA_TRANSFER_RAT', 'USA_TRANSFER_TARG_N', 'USA_TRANSFER_TARG_RAT', 'USA_TRFPSPILL_FE_R', 'USA_UFOOD_R', 'USA_UMETAL_R', 'USA_UNR', 'USA_UNRH', 'USA_UNRH_FE', 'USA_UNR_FE', 'USA_UOIL_R', 'USA_WAGEEFF_N', 'USA_WAGEH_N', 'USA_WAGE_N', 'USA_WF_R', 'USA_WH_R', 'USA_WK_N', 'USA_WO_R', 'USA_W_R', 'USA_W_R_AVG', 'USA_XFOOD_R', 'USA_XMA_R', 'USA_XMETAL_R', 'USA_XM_R', 'USA_XOIL_R', 'USA_XT_R', 'USA_XT_RAT', 'USA_YCAP_N', 'USA_YD_R', 'USA_YLABH_N', 'USA_YLAB_N', 'USA_Z', 'USA_Z_AVG', 'USA_Z_NFA', 'WRL_GDP_FE_METAL_R', 'WRL_GDP_FE_OIL_R', 'WRL_GDP_FE_R', 'WRL_GDP_METAL_R', 'WRL_GDP_OIL_R', 'WRL_GDP_R', 'WRL_PRODFOOD_CYC_R', 'WRL_PRODFOOD_PERM_R', 'WRL_PRODFOOD_R', 'WRL_PRODMETAL_CYC_R', 'WRL_PRODMETAL_PERM_R', 'WRL_PRODMETAL_R', 'WRL_PRODOIL_CYC_R', 'WRL_PRODOIL_PERM_R', 'WRL_PRODOIL_R', 'WRL_XFOOD_R', 'WRL_XMETAL_R', 'WRL_XOIL_R', 'WTRADE_FOOD_N', 'WTRADE_FOOD_R', 'WTRADE_METAL_N', 'WTRADE_METAL_R', 'WTRADE_M_N', 'WTRADE_M_R', 'WTRADE_OIL_N', 'WTRADE_OIL_R']
    # params=  ['USA_INTNFA1', 'ISR_EPS_TAR', 'RC0_IM5', 'FILTER_RC0_MPC', 'ISR_SPILLM1', 'RC0_R_NEUT_BAR', 'RC0_XX_PRIMSUR_N', 'PERSISTCOM_ISR', 'ISR_GPROD_SS', 'CONSTANT_USA_PG_P', 'CONSTANT_RPOIL_PERM', 'USA_GC_RAT_BAR', 'FILTER_WTRADE_FOOD_R', 'CONSTANT_USA_FXPREM', 'CONSTANT_RC0_CPINCOM_P', 'CONSTANT_USA_MKTPREM', 'FILTER_USA_PRIMSUR_TAR', 'USA_GDP_N_BAR', 'RC0_COM5', 'RES_ISR_GNP_R', 'USA_COEFPGDP', 'ISR_Q1', 'RC0_IM1', 'FILTER_USA_INTNFA', 'RES_RPOIL_PERM_AVG', 'FILTER_ISR_PXM_P', 'RC0_XX_TRANSFER_N', 'USA_XMETAL2', 'USA_COEFW', 'RES_ISR_INTCORP', 'CONSTANT_USA_CPI_P', 'RC0_GAMMA', 'RES_RPMETAL_CYC_AVG', 'RES_RC0_PCOIL_P', 'USA_COEFTFP', 'ISR_EBASE', 'ISR_INTCOST_RAT_BAR', 'FILTER_USA_PARTH', 'USA_GILRSW', 'RES_RC0_ROYALTIES_N', 'CONSTANT_ISR_GDEF_TAR', 'RES_USA_WO_R', 'ISR_XM2', 'PERSIST_E_RC0_RPMETAL_CYC', 'USA_BNSHK', 'RC0_CFOOD3', 'USA_COM4', 'ISR_BGG4', 'USA_CFOOD3', 'RC0_WEDGEFOOD_P', 'RC0_INTGAP', 'RC0_INFCPIX_TAR', 'FILTER_USA_IMETAL_R', 'ISR_USA_SOIL', 'RC0_RK5', 'USA_GNP_R_BAR', 'FILTER_ISR_PARTH', 'USA_EXP1', 'CONSTANT_RC0_PC_P', 'ISR_XX_TRANSFER_TARG_N', 'RES_ISR_INT10', 'USA_IM5', 'RES_USA_INTRF', 'USA_UNR2', 'USA_TAXL_RAT_BAR', 'USA_BGG3', 'RC0_WT', 'USA_PRODMETAL1', 'CONSTANT_ISR_IOIL_R', 'USA_CPI4', 'RC0_PIM1', 'FILTER_ISR_K_R', 'RES_RC0_Z_NFA', 'USA_PXM1', 'ISR_PARTH_ADJ_STAR', 'RPMETAL_CYC_BAR', 'ISR_IMETAL3', 'RC0_KTAXLRSW', 'FILTER_RC0_TPREM', 'RES_USA_PARTH_FE', 'FILTER_WTRADE_FOOD_N', 'CONSTANT_RC0_IMETAL_R', 'USA_INTINFL', 'RC0_W_R_BAR', 'RC0_INTCOST_RAT_BAR', 'RC0_TAXK_RAT_BAR', 'RC0_CPI7', 'TRDEMETAL_N', 'USA_Q2', 'CONSTANT_ISR_C_OLG_R', 'USA_RK4', 'CONSTANT_RC0_PCW_P', 'USA_RC0_SMETAL', 'ISR_COEFPGDP', 'RES_ISR_C_LIQ_R', 'USA_NSPILLM1', 'FILTER_ISR_TPREM', 'ISR_PSUB_XM', 'ISR_COIL2', 'RC0_CPI4', 'RC0_POIL_P_SUB_BAR', 'FILTER_USA_TAU_OIL', 'RES_ISR_Q_P', 'RC0_GI_N_EXOG', 'USA_Z_BAR', 'ISR_GI_R_BAR', 'RC0_BGG4', 'RES_RC0_Z_AVG', 'ISR_INTMP_BAR', 'USA_COM5', 'USA_LAPH', 'USA_COM1', 'USA_B_RAT_BAR', 'FILTER_ISR_CPI_P', 'ISR_COEFPARTH', 'ISR_IOIL1', 'RES_USA_PCOIL_P', 'CONSTANT_RC0_TPREM', 'RC0_IFOOD1', 'ISR_W_R_BAR', 'RC0_TARGDAMP', 'USA_BETA', 'RC0_CTAXLRSW', 'RC0_XX_TAU_L', 'FILTER_RC0_PRIMSUR_TAR', 'WRL_GDP_OIL_R_BAR', 'FILTER_ISR_PRIMSUR_TAR', 'RES_RC0_INFWEXP', 'ISR_XX_PRIMSUR_N', 'WRL_GDP_R_BAR', 'RC0_ISR_S', 'USA_GI_MPROP', 'ISR_XFOOD1', 'RC0_INTMP_BAR', 'USA_BREVAL1', 'USA_IOIL4', 'ISR_IFOOD4', 'ISR_TRANSFER_RAT_EXOG', 'RC0_IMETAL2', 'RC0_PXM1', 'RC0_TARIFF_IM', 'RES_USA_ROYALTIES_N', 'USA_COSTOIL_P', 'CONSTANT_USA_PXMF_P', 'ISR_MNFACTOR', 'ISR_TAXLBASE_N_BAR', 'RC0_IM_SPILL_BAR', 'USA_PARTH2', 'RC0_DAMP_GDP_GAP', 'USA_IOIL2', 'USA_PRODMETAL_R_EXOG', 'RC0_COM4', 'USA_EPS_TAR', 'ISR_TAU_L_BAR', 'RC0_GI_RAT_BAR', 'RC0_XX_TRANSFER_TARG_N', 'USA_PSUB_XM', 'CONSTANT_USA_NPOPB_R', 'ISR_SIGMA', 'FILTER_USA_RPREM', 'USA_PRODOIL2', 'ISR_VMETAL', 'USA_SUB_XM', 'RC0_PRODMETAL1', 'USA_CFOOD1', 'CONSTANT_USA_OILPAY_N', 'USA_INTLAG', 'RPMETAL_PERM_BAR', 'FILTER_RC0_PCW_P', 'USA_SOV1', 'ISR_TAXCBASE_N_BAR', 'USA_GIDAMP', 'USA_IMETAL1', 'PERSIST_E_RC0_RPOIL_CYC', 'RC0_IMETAL3', 'USA_RC0_S', 'CONSTANT_WTRADE_M_N', 'RC0_XMETAL1', 'ISR_IM1', 'RES_RC0_GNP_R', 'RES_ISR_PGDP_P_AVG', 'TRDEFOOD_N', 'ISR_CPI2', 'ISR_KGTFPRAT', 'USA_CFOOD4', 'ISR_W2', 'RES_USA_INTC', 'RES_ISR_PRODFOOD_R', 'USA_SIGMA', 'RC0_UNR3', 'RC0_INTGR', 'USA_SPILLM1', 'FILTER_ISR_COIL_R', 'RPMETAL1', 'ISR_XM4', 'RC0_TSPILLX1', 'RES_ISR_CURBAL_N', 'RC0_NSPILLX1', 'FILTER_RC0_XOIL_R', 'USA_COM2', 'RES_POIL_PERM_P', 'USA_SPILLX1', 'TRDEOIL_R', 'ISR_CFOOD1', 'RC0_SPILLX1', 'RC0_COSTMETAL_P', 'ISR_R_NEUT_BAR', 'USA_PRODOIL_R_EXOG', 'USA_TOIL', 'FILTER_RC0_PG_P', 'RC0_EBASE', 'CONSTANT_USA_PI_P', 'RC0_CPI2', 'USA_CGBFX', 'RES_USA_MROYALTIES_N', 'RC0_GC_N_EXOG', 'CONSTANT_ISR_MKTPREM', 'RC0_TMETAL', 'ISR_B_STAR', 'ISR_GC_MPROP', 'RC0_XM1', 'USA_TFP_FE_R_BAR', 'ISR_TAXC_RAT_BAR', 'USA_UNR3', 'USA_PARTH4', 'RES_RC0_INT10', 'CONSTANT_ISR_CFOOD_R', 'RES_USA_C_LIQ_R', 'RC0_NFANSHK', 'ISR_WEDGEOIL_P', 'FILTER_USA_TPREM', 'FILTER_ISR_IOIL_R', 'ISR_XFOOD2', 'ISR_C_LAMBDA', 'RES_USA_CURBAL_N', 'RC0_W1', 'ISR_CPI7', 'FILTER_USA_PXM_P', 'ISR_BETA', 'TEMP_USA', 'CONSTANT_USA_INTNFA', 'RC0_XMETAL3', 'WRL_PRODFOOD_R_BAR', 'CONSTANT_USA_XFOOD_R', 'RC0_XFOOD4', 'FILTER_RC0_CFOOD_R', 'RC0_SOV2', 'RES_USA_NFA_D', 'ISR_INTEPS', 'ISR_XX_TRANSFER_N', 'ISR_LAPH', 'CONSTANT_RC0_SOVPREM', 'RC0_PRODOIL1', 'RC0_CHI', 'USA_XOIL4', 'CONSTANT_ISR_K_R', 'USA_GDP_R_BAR', 'USA_XX_GI_N', 'RES_USA_INFCPIX', 'ISR_INTWT', 'RC0_CPI9', 'RC0_EXP1', 'FILTER_USA_DELTA', 'FILTER_RC0_RPREM', 'RPOIL3', 'ISR_IMETAL1', 'PERSIST_E_USA_RPOIL_CYC', 'CONSTANT_RC0_OILPAY_N', 'ISR_XX_TAU_C', 'RC0_XFOOD2', 'CONSTANT_ISR_XFOOD_R', 'CONSTANT_WTRADE_FOOD_R', 'ISR_IM5', 'RC0_IMETAL1', 'USA_PRODFOODSHARE', 'RC0_XX_GI_N', 'RES_RC0_RC', 'USA_UOIL_R_BAR', 'ISR_TAXK_RAT_BAR', 'ISR_CPI4', 'CONSTANT_ISR_IM_R', 'FILTER_USA_XFOOD_R', 'RC0_IOIL2', 'USA_C_MPROP', 'FILTER_RC0_XM_R', 'FILTER_RC0_K_R', 'RC0_CPI1', 'ISR_GNP_R_BAR', 'USA_C_LIQ_R_BAR', 'FILTER_USA_PCW_P', 'ISR_PCOILBASE', 'FILTER_RPFOOD_PERM', 'CONSTANT_RC0_PXM_P', 'RC0_VMETAL', 'FILTER_ISR_IMETAL_R', 'RES_ISR_PIM_P', 'RES_ISR_R_NEUT', 'FILTER_ISR_PG_P', 'RES_ISR_BREVAL_N', 'ISR_XM_MPROP', 'RC0_USA_SOIL', 'RC0_XM2', 'USA_CTAXLRSW', 'ISR_PIM2', 'CONSTANT_ISR_TFP_FE_R', 'USA_CPI1', 'USA_XM_SPILL_BAR', 'RC0_INTC1', 'USA_COIL2', 'USA_RK2', 'CONSTANT_RC0_PARTH', 'FILTER_RC0_FXPREM', 'FILTER_ISR_INTNFA', 'USA_RK5', 'FILTER_USA_GDEF_TAR', 'USA_XOIL2', 'ISR_PARTH4', 'ISR_TSPILLM1', 'RC0_COEFPARTH', 'RC0_IMETAL5', 'FILTER_ISR_C_OLG_R', 'RES_ISR_PARTH_DES', 'CONSTANT_ISR_IFOOD_R', 'USA_XX_GC_N', 'USA_DAMP_GDP_GAP', 'RES_ISR_WH_R', 'USA_XX_TRANSFER_TARG_N', 'FILTER_USA_SOVPREM', 'ISR_TAXOILBASE_N_BAR', 'RC0_B_STAR', 'ISR_COEFTFP', 'USA_TAXC_RAT_BAR', 'CONSTANT_USA_DELTA', 'USA_PCFOODBASE', 'CONSTANT_USA_OILSHARF', 'ISR_TMETAL', 'ISR_BGG1', 'CONSTANT_USA_SOVPREM', 'ISR_PARTH5', 'RC0_PRODMETAL2', 'USA_CTAXDAMP', 'ISR_CTAXLRSW', 'RC0_DELTAG', 'USA_WEDGEFOOD_P', 'ISR_UNR3', 'RES_ISR_INFEXP', 'USA_XOIL_R_BAR', 'FILTER_RC0_PXM_P', 'TEMP_E_USA_RPFOOD_CYC', 'FILTER_RC0_NPOPB_R', 'USA_INTGR', 'USA_XMETAL1', 'ISR_WT', 'USA_C_LAMBDA', 'RES_USA_PARTH_DES', 'ISR_USA_L', 'USA_ALPHA_KG', 'FILTER_ISR_FXPREM', 'ISR_EXP1', 'USA_CFOOD2', 'USA_BGG4', 'RC0_BNSHK', 'ISR_XX_GC_N', 'RES_USA_W_R_AVG', 'RES_RC0_PARTH_FE', 'RC0_CTAXDAMP', 'USA_TARGLRSW', 'RC0_COEFZ', 'ISR_INTC1', 'CONSTANT_USA_XMETAL_R', 'USA_TRANSFERDAMP', 'RES_RC0_TAXOIL_N', 'CONSTANT_RPMETAL_PERM', 'CONSTANT_RC0_CPI_P', 'RC0_XX_OILSUB_N', 'RC0_XM4', 'USA_INTCORP1', 'RC0_XM_MPROP', 'USA_RC0_SOIL', 'ISR_CPI3', 'USA_IM3', 'ISR_ALPHA_FE', 'CONSTANT_USA_ROYALTY', 'RPOIL_PERM_BAR', 'ISR_RK1', 'RC0_C_MPROP', 'RES_USA_BREVAL_N', 'USA_ISR_S', 'RC0_TARGLRSW', 'RES_RC0_INTMP', 'RES_USA_Z_NFA', 'ISR_TAU_K_EXOG', 'RC0_WEDGEOIL_P', 'RC0_C_OLG_R_BAR', 'CONSTANT_USA_K_R', 'ISR_KGSPILLM1', 'USA_TAU_C_EXOG', 'FILTER_WTRADE_M_N', 'RC0_TAXKBASE_N_BAR', 'RC0_W3', 'USA_W_R_BAR', 'CONSTANT_RC0_RPREM', 'RES_RC0_PIM_P', 'RES_ISR_INTRF', 'RPMETAL4', 'RES_ISR_PARTH_FE', 'RC0_IFOOD4', 'ISR_SOV2', 'FILTER_RC0_TFP_FE_R', 'FILTER_ISR_CFOOD_R', 'ISR_TRANSFERSHARE', 'CONSTANT_ISR_NPOPB_R', 'FILTER_RC0_MROYALTY', 'USA_IMETAL4', 'CONSTANT_USA_MPC', 'FILTER_ISR_XFOOD_R', 'RC0_IM4', 'TEMP_ISR', 'ISR_TRANSFERDAMP', 'WRL_GDP_METAL_R_BAR', 'ISR_B_RAT_BAR', 'USA_CPI7', 'USA_LTAXDAMP', 'RC0_XM3', 'USA_TAU_L_EXOG', 'ISR_RK4', 'USA_DAMP_DEBT', 'USA_NPOP_R_EXOG', 'USA_IM4', 'ISR_IMETAL5', 'CONSTANT_USA_PXM_P', 'USA_TARIFF_IM', 'RPFOOD2', 'RC0_COEFW', 'TEMP_E_ISR_TAXK_RAT', 'RC0_Z_BAR', 'CONSTANT_ISR_RPREM', 'ISR_TARGLRSW', 'RC0_LSTAX_N', 'USA_XMETAL4', 'ISR_CFOOD4', 'ISR_INVEST_MPROP', 'RES_RC0_RCORP', 'USA_XOIL3', 'FILTER_RC0_C_OLG_R', 'PERSISTCOM_USA', 'USA_XOIL1', 'RES_USA_OILSUB_N', 'FILTER_ISR_CPINCOM_P', 'ISR_XX_GI_N', 'RC0_B_RAT_BAR', 'ISR_ABSRATROOT', 'ISR_QUOTA_IM', 'RC0_T', 'CONSTANT_USA_IM_R', 'USA_CPI9', 'CONSTANT_ISR_PXM_P', 'FILTER_ISR_XM_R', 'ISR_W3', 'USA_W3', 'RC0_COM2', 'RC0_CFOOD1', 'ISR_INTNFA1', 'ISR_GI_MPROP', 'RC0_PARTH5', 'FILTER_RC0_IFOOD_R', 'USA_TRANSFER_TARG_RAT_BAR', 'USA_RC0_SFOOD', 'RES_USA_INFEXP', 'CONSTANT_ISR_CPINCOM_P', 'ISR_RK5', 'USA_XM1', 'ISR_OILRECEIPT_N_BAR', 'ISR_CGBFX', 'USA_PXM2', 'USA_XM3', 'USA_POIL_P_SUB_BAR', 'FILTER_ISR_UNRH_FE', 'RC0_UNR2', 'FILTER_RC0_PC_P', 'USA_TAXOILBASE_N_BAR', 'RC0_SOV1', 'ISR_TAXL_RAT_BAR', 'RES_RC0_PRODMETAL_CYC_R', 'ISR_XM3', 'USA_XX_TAU_L', 'FILTER_ISR_IM_R', 'FILTER_RC0_OILPAY_N', 'RES_RC0_PRODFOOD_PERM_R', 'ISR_C_MPROP', 'ISR_GCDAMP', 'RES_USA_PRODOIL_R', 'USA_ISR_L', 'ISR_TSPILLX1', 'ISR_COM2', 'FILTER_ISR_PCW_P', 'RC0_XFOOD3', 'CONSTANT_RPFOOD_PERM', 'RC0_XMETAL4', 'FILTER_RC0_XFOOD_R', 'ISR_TFP_FE_R_BAR', 'FILTER_WTRADE_METAL_R', 'RC0_USA_SMETAL', 'USA_GPROD_SS', 'RES_RC0_OILSUB_N', 'RC0_LTAXDAMP', 'USA_INTGB1', 'USA_CPI11', 'CONSTANT_RC0_GDEF_TAR', 'FILTER_RC0_GDEF_TAR', 'FILTER_USA_IFOOD_R', 'USA_RK7', 'RC0_SIGMA', 'RC0_XOIL2', 'RES_USA_PRODFOOD_R', 'RC0_PARTH1', 'RC0_W2', 'FILTER_USA_ROYALTY', 'RES_RPMETAL_CYC', 'RC0_TSPILLM1', 'USA_COIL4', 'RPMETAL2', 'RC0_GREAL_SS', 'RC0_NSPILLM1', 'FILTER_USA_XMETAL_R', 'RES_RC0_INTRF', 'ISR_WEDGEFOOD_P', 'USA_KGTFPRATROOT', 'RES_USA_INTGB', 'CONSTANT_RC0_XOIL_R', 'ISR_XFOOD3', 'CONSTANT_RC0_K_R', 'RC0_INVEST_MPROP', 'ISR_IM3', 'RC0_PCOILBASE', 'RES_RC0_INTC', 'CONSTANT_WTRADE_FOOD_N', 'RC0_GCDAMP', 'RC0_PRODMETAL_R_EXOG', 'RC0_CPI12', 'CONSTANT_WTRADE_METAL_R', 'RC0_PIM2', 'CONSTANT_ISR_PARTH', 'FILTER_ISR_PXMF_P', 'RC0_CPI6', 'RC0_TRANSFERDAMP', 'RC0_COEFTFP', 'ISR_TAU_L_EXOG', 'RC0_CGBFX', 'ISR_TAU_C_EXOG', 'RC0_CPI11', 'RC0_TAU_C_EXOG', 'RES_RC0_INT', 'ISR_RC0_SMETAL', 'ISR_T', 'ISR_INTLAG', 'CONSTANT_RC0_ROYALTY', 'RES_ISR_INFWEXP', 'RPFOOD1', 'USA_NFANSHK', 'RC0_PCFOOD_P_BAR', 'RES_USA_GDEF_N', 'USA_PI1', 'USA_GI_RAT_BAR', 'RC0_KTAXDAMP', 'RES_RC0_INTCORP', 'ISR_POIL_P_SUB_BAR', 'FILTER_RC0_PI_P', 'CONSTANT_RC0_COIL_R', 'ISR_XM_SPILL_BAR', 'RC0_LTAXLRSW', 'FILTER_ISR_TFP_FE_R', 'ISR_TFOOD', 'FILTER_USA_K_R', 'RC0_TAU_L_EXOG', 'USA_TAXK_RAT_BAR', 'FILTER_USA_UNRH_FE', 'ISR_COIL4', 'PERM', 'RC0_INTGB1', 'FILTER_ISR_MPC', 'ISR_GDP_R_BAR', 'ISR_IMETAL2', 'USA_CHI', 'RC0_TAU_L_BAR', 'USA_MTFACTOR', 'ISR_KGTFPRATROOT', 'USA_IOIL3', 'FILTER_RC0_IOIL_R', 'RES_USA_UNRH', 'ISR_COIL1', 'USA_PRODOIL1', 'ISR_TFPRATROOT', 'RES_RC0_PCFOOD_P', 'ISR_USA_SMETAL', 'FILTER_ISR_GDEF_TAR', 'FILTER_RC0_ROYALTY', 'RC0_XOIL_R_BAR', 'USA_TRANSFER_RAT_EXOG', 'RC0_UOIL_R_BAR', 'CONSTANT_ISR_PRIMSUR_TAR', 'PERSIST_E_ISR_RPOIL_CYC', 'FILTER_RC0_COIL_R', 'FILTER_USA_OILSHARF', 'RC0_BGG3', 'RC0_XFOOD1', 'ISR_GREAL_SS', 'RC0_INTINFL', 'FILTER_USA_MROYALTY', 'RC0_COIL4', 'ISR_KTAXDAMP', 'ISR_BNSHK', 'FILTER_USA_IM_R', 'FILTER_USA_OILPAY_N', 'RES_USA_OILRECEIPT_N', 'RC0_GNP_R_BAR', 'TEMP_E_USA_TAXK_RAT', 'RES_RC0_UNRH', 'RC0_INTLAG', 'ISR_IM2', 'ISR_CPI6', 'ISR_IFOOD1', 'RES_ISR_W_R_AVG', 'USA_CPI5', 'ISR_RC0_L', 'RES_RC0_PRODOIL_R', 'USA_BGG2', 'RC0_XOIL4', 'USA_XX_TAU_C', 'CONSTANT_USA_CFOOD_R', 'USA_GCDAMP', 'ISR_TOIL', 'ISR_W1', 'ISR_SUB_XM', 'USA_IOIL1', 'CONSTANT_RC0_DELTA', 'CONSTANT_USA_TAU_OIL', 'ISR_IM4', 'USA_INTCOST_RAT_BAR', 'ISR_INTCORP1', 'USA_TAXCBASE_N_BAR', 'ISR_IOIL5', 'RC0_IFOOD5', 'ISR_C_OLG_R_BAR', 'ISR_RK7', 'CONSTANT_ISR_PI_P', 'RES_ISR_OILSUB_N', 'USA_IM1', 'RPMETAL3', 'CONSTANT_ISR_PCW_P', 'RES_RC0_TFP_FE_R_AVG', 'ISR_IFOOD5', 'RES_RC0_PRODMETAL_R', 'RC0_IM3', 'RES_RC0_PGDP_P_AVG', 'ISR_XX_LB', 'ISR_KTAXLRSW', 'RC0_GC_RAT_BAR', 'CONSTANT_RC0_MKTPREM', 'USA_PARTH5', 'FILTER_ISR_PC_P', 'CONSTANT_RC0_MROYALTY', 'ISR_GI_RAT_BAR', 'RC0_TRANSFER_TARG_RAT_BAR', 'ISR_DELTAG', 'RC0_IMETAL4', 'ISR_COM4', 'ISR_COEFZ', 'USA_T', 'RC0_XMETAL2', 'ISR_NSPILLM1', 'RC0_COIL3', 'ISR_GCLRSW', 'CONSTANT_RC0_INTNFA', 'FILTER_USA_CPI_P', 'ISR_IOIL3', 'ISR_TRANSFER_TARG_RAT_BAR', 'USA_W2', 'RC0_USA_SFOOD', 'RES_USA_RCORP', 'ISR_KREVAL', 'USA_IFOOD5', 'ISR_NSPILLX1', 'CONSTANT_USA_IMETAL_R', 'ISR_CFOOD2', 'FILTER_USA_NPOPB_R', 'USA_VMETAL', 'ISR_DAMP_DEBT', 'RC0_FLOOR', 'RC0_KGTFPRAT', 'USA_KGSPILLM1', 'CONSTANT_ISR_XM_R', 'ISR_GI_N_EXOG', 'RC0_MTFACTOR', 'RES_ISR_PCFOOD_P', 'USA_XX_TRANSFER_N', 'USA_XFOOD2', 'ISR_INFCPIX_TAR', 'RC0_INTCORP1', 'CONSTANT_USA_XOIL_R', 'CONSTANT_USA_COIL_R', 'USA_IFOOD3', 'USA_DELTAG', 'RC0_COM1', 'CONSTANT_ISR_DELTA', 'RES_RC0_INFWAGEEFF', 'ISR_CPI9', 'USA_COEFZ', 'CONSTANT_RC0_OILSHARF', 'CONSTANT_ISR_PC_P', 'USA_INTEPS', 'ISR_SPILLX1', 'RES_ISR_NFA_D', 'RC0_OILRECEIPT_N_BAR', 'RES_USA_RC', 'TRDEFOOD_R', 'RC0_INTNFA1', 'USA_FLOOR', 'CONSTANT_USA_IFOOD_R', 'FILTER_ISR_MKTPREM', 'ISR_RK6', 'USA_TSPILLM1', 'ISR_PARTH1', 'USA_KTAXLRSW', 'CONSTANT_USA_CPINCOM_P', 'RES_RC0_PRODFOOD_R', 'USA_GAMMA', 'ISR_XFOOD4', 'ISR_LSTAX_N', 'ISR_BGG2', 'USA_INVEST_MPROP', 'RPMETAL5', 'FILTER_RPOIL_PERM', 'CONSTANT_RC0_TAU_OIL', 'FILTER_USA_MKTPREM', 'USA_ISR_SFOOD', 'RC0_XX_TAU_C', 'ISR_CPI8', 'ISR_COM5', 'USA_IFOOD4', 'USA_VOIL', 'RC0_TAXC_RAT_BAR', 'USA_Q1', 'RC0_CPI5', 'RC0_VOIL', 'RC0_TFOOD', 'FILTER_RC0_SOVPREM', 'USA_TAU_L_BAR', 'RES_ISR_OILRECEIPT_N', 'CONSTANT_ISR_FXPREM', 'ISR_INTGAP', 'ISR_C_LIQ_R_BAR', 'RC0_BGG1', 'RC0_BREVAL1', 'CONSTANT_ISR_INTNFA', 'ISR_IFOOD2', 'PERSISTCOM_RC0', 'RC0_BGG2', 'USA_BGG1', 'RC0_USA_S', 'USA_TSPILLX1', 'RC0_ISR_L', 'USA_WEDGEOIL_P', 'USA_W1', 'RC0_MNFACTOR', 'RES_RPOIL_CYC_AVG', 'FILTER_RC0_DELTA', 'CONSTANT_RC0_IOIL_R', 'RC0_Q1', 'RES_RC0_C_LIQ_R', 'USA_NPOPH_R_BAR', 'RC0_ALPHA_KG', 'ISR_BREVAL1', 'USA_SOV2', 'ISR_TFPRAT', 'PERM_E_MPC', 'USA_KGTFPRAT', 'USA_INTC1', 'FILTER_RC0_CPI_P', 'RES_USA_PRODOIL_CYC_R', 'USA_IMETAL2', 'WRL_PRODOIL_R_BAR', 'RES_USA_INT', 'USA_PIM2', 'RC0_Q2', 'RES_RC0_GDEF_N', 'TEMP_E_RPFOOD_CYC', 'USA_IFOOD2', 'USA_LTAXLRSW', 'ISR_GC_N_EXOG', 'ISR_LTAXDAMP', 'ISR_INTINFL', 'RC0_PRODOIL2', 'ISR_PI1', 'USA_OILRECEIPT_N_BAR', 'USA_PIM1', 'PERSIST_E_USA_RPMETAL_CYC', 'FILTER_RC0_XMETAL_R', 'CONSTANT_ISR_SOVPREM', 'ISR_GAMMA', 'ISR_POIL_P_BAR', 'CONSTANT_RC0_MPC', 'ISR_NPOPH_R_BAR', 'ISR_IOIL2', 'FILTER_USA_MPC', 'RES_RC0_OILRECEIPT_N', 'RES_ISR_INFCPIX', 'USA_XX_LB', 'ISR_Z_BAR', 'RC0_GI_R_BAR', 'ISR_PRODFOODSHARE', 'RES_ISR_INFWAGEEFF', 'FILTER_USA_FXPREM', 'RES_USA_PRODMETAL_CYC_R', 'RC0_IFOOD3', 'USA_XM2', 'RPOIL2', 'RC0_CPI3', 'RES_RPMETAL_PERM_AVG', 'RC0_IOIL4', 'FILTER_RC0_INTNFA', 'FILTER_RC0_UNRH_FE', 'RC0_GDP_N_BAR', 'RC0_CFOOD4', 'RES_ISR_TFP_FE_R_AVG', 'RC0_PRODFOODSHARE', 'FILTER_RC0_OILSHARF', 'USA_XM_MPROP', 'ISR_FLOOR', 'RES_ISR_EPS', 'RC0_KGSPILLM1', 'USA_COIL3', 'CONSTANT_RC0_XFOOD_R', 'RC0_COEFPGDP', 'TEMP_E_RC0_RPFOOD_CYC', 'ISR_CFOOD3', 'RES_USA_EPS', 'RES_USA_PRODFOOD_PERM_R', 'TEMP_E_RC0_TAXK_RAT', 'FILTER_ISR_IFOOD_R', 'RES_USA_PCFOOD_P', 'USA_PRODMETAL2', 'FILTER_ISR_DELTA', 'RES_ISR_PCOIL_P', 'RC0_GPROD_SS', 'RC0_BETA', 'ISR_PCFOOD_P_BAR', 'RC0_RK4', 'USA_PCOILBASE', 'FILTER_ISR_RPREM', 'RES_RPFOOD_CYC', 'RC0_TRANSFERSHARE', 'RC0_QUOTA_IM', 'WRL_PRODMETAL_R_BAR', 'RC0_TRANSFERLRSW', 'CONSTANT_ISR_CPI_P', 'CONSTANT_RC0_PI_P', 'USA_GC_N_EXOG', 'ISR_IFOOD3', 'CONSTANT_RC0_UNRH_FE', 'ISR_TAXKBASE_N_BAR', 'RES_RC0_WH_R', 'RC0_TAXL_RAT_BAR', 'ISR_XM1', 'ISR_XX_GDEF_N', 'FILTER_RC0_IMETAL_R', 'CONSTANT_RC0_IM_R', 'RC0_XM_SPILL_BAR', 'ISR_CPI12', 'RC0_XX_GDEF_N', 'CONSTANT_USA_GDEF_TAR', 'FILTER_USA_TFP_FE_R', 'ISR_XX_TAU_L', 'USA_QUOTA_IM', 'ISR_PXM1', 'TEMP_RC0', 'USA_TAXKBASE_N_BAR', 'USA_C_OLG_R_BAR', 'FILTER_USA_PXMF_P', 'USA_COEFPARTH', 'ISR_UNR1', 'ISR_GC_RAT_BAR', 'ISR_COEFW', 'RES_ISR_UNRH', 'USA_RK6', 'TRDEOIL_N', 'CONSTANT_RC0_XM_R', 'RC0_TFPRAT', 'ISR_WEXP1', 'USA_GREAL_SS', 'CONSTANT_RC0_PG_P', 'PERSIST_E_ISR_RPMETAL_CYC', 'USA_POIL_P_BAR', 'CONSTANT_USA_IOIL_R', 'TRDEM_N', 'USA_RK1', 'USA_R_NEUT_BAR', 'CONSTANT_RC0_FXPREM', 'USA_IOIL5', 'ISR_RK2', 'USA_MFACTOR', 'USA_PROB', 'ISR_SOV1', 'RES_ISR_INTGB', 'USA_UNR1', 'RES_RC0_PARTH_DES', 'FILTER_USA_XOIL_R', 'FILTER_ISR_NPOPB_R', 'CONSTANT_RC0_IFOOD_R', 'RC0_SUB_XM', 'RPOIL_ROOT', 'USA_INTGAP', 'RC0_TRANSFER_TARG_RAT_EXOG', 'RES_ISR_INTC', 'RES_USA_PIM_P', 'CONSTANT_ISR_PXMF_P', 'RC0_GCLRSW', 'RES_RC0_INFEXP', 'ISR_RC0_SOIL', 'ISR_CPI11', 'CONSTANT_WTRADE_M_R', 'RC0_CFOOD2', 'ISR_MTFACTOR', 'ISR_BGG3', 'RC0_COSTOIL_P', 'ISR_TRANSFER_RAT_BAR', 'USA_XX_PRIMSUR_N', 'RC0_PI1', 'ISR_CHI', 'ISR_GILRSW', 'USA_ALPHA_FE', 'RC0_TRANSFER_RAT_EXOG', 'RC0_CPI8', 'RES_ISR_WO_R', 'USA_IFOOD1', 'RC0_XX_TAU_K', 'USA_IM_SPILL_BAR', 'USA_EBASE', 'CONSTANT_USA_TFP_FE_R', 'ISR_INTGB1', 'USA_XM4', 'RES_RC0_WO_R', 'FILTER_WTRADE_M_R', 'FILTER_RC0_IM_R', 'FILTER_RC0_MKTPREM', 'RC0_ISR_SFOOD', 'RC0_SPILLM1', 'RC0_RK6', 'RC0_INTEPS', 'ISR_CPI1', 'FILTER_WTRADE_OIL_R', 'CONSTANT_ISR_IMETAL_R', 'USA_GC_MPROP', 'FILTER_WTRADE_OIL_N', 'RES_RC0_INTGB', 'RC0_PARTH4', 'CONSTANT_USA_PC_P', 'USA_TARGDAMP', 'ISR_IM_SPILL_BAR', 'FILTER_ISR_SOVPREM', 'RES_USA_Q_P', 'USA_TFPRATROOT', 'RC0_IOIL5', 'CONSTANT_RC0_CFOOD_R', 'CONSTANT_RC0_NPOPB_R', 'FILTER_USA_CFOOD_R', 'RC0_COIL2', 'RES_ISR_GDEF_N', 'RES_ISR_RC', 'RPOIL5', 'ISR_Q2', 'ISR_MKT1', 'ISR_TRANSFERLRSW', 'RC0_UNR1', 'RC0_CPI10', 'FILTER_WTRADE_METAL_N', 'ISR_PARTH2', 'ISR_CPI5', 'CONSTANT_WTRADE_METAL_N', 'USA_WT', 'RES_USA_INFWAGEEFF', 'USA_TRANSFERSHARE', 'ISR_VOIL', 'USA_KREVAL', 'ISR_LTAXLRSW', 'CONSTANT_ISR_TPREM', 'RC0_MFACTOR', 'RC0_PCFOODBASE', 'ISR_XX_OILSUB_N', 'USA_XX_OILSUB_N', 'ISR_PIM1', 'RC0_PSUB_XM', 'USA_ABSRATROOT', 'RC0_XX_GC_N', 'RC0_USA_L', 'USA_XFOOD4', 'FILTER_USA_PC_P', 'RES_RC0_PRODOIL_CYC_R', 'RES_RC0_R_NEUT', 'RC0_TAXLBASE_N_BAR', 'RC0_KGTFPRATROOT', 'CONSTANT_USA_MROYALTY', 'RC0_C_LAMBDA', 'ISR_ALPHA_KG', 'RC0_NPOPH_R_BAR', 'TEMP_E_ISR_RPFOOD_CYC', 'USA_XX_GDEF_N', 'RES_RC0_INFCPIX', 'ISR_TARGDAMP', 'CONSTANT_USA_PRIMSUR_TAR', 'ISR_UNR2', 'FILTER_USA_COIL_R', 'RPOIL_CYC_BAR', 'RC0_WEXP1', 'RC0_PARTH_ADJ_STAR', 'RC0_ALPHA_FE', 'ISR_NFANSHK', 'ISR_IOIL4', 'USA_KTAXDAMP', 'ISR_RC0_S', 'RES_USA_R_NEUT', 'RC0_IOIL1', 'RC0_TFP_FE_R_BAR', 'ISR_RC0_SFOOD', 'CONSTANT_USA_RPREM', 'TRDEM_R', 'RC0_DAMP_DEBT', 'RES_RC0_NFA_D', 'USA_GCLRSW', 'FILTER_USA_C_OLG_R', 'CONSTANT_RC0_XMETAL_R', 'CONSTANT_WTRADE_OIL_R', 'CONSTANT_USA_XM_R', 'USA_TAU_K_EXOG', 'RC0_LAPH', 'ISR_PROB', 'RC0_KREVAL', 'RES_USA_INT10', 'FILTER_USA_PG_P', 'RES_USA_TAXOIL_N', 'RC0_TAU_K_EXOG', 'FILTER_RC0_CPINCOM_P', 'CONSTANT_ISR_COIL_R', 'CONSTANT_ISR_TAU_OIL', 'RC0_XOIL5', 'USA_INTWT', 'RC0_IOIL3', 'CONSTANT_ISR_MPC', 'USA_PARTH3', 'ISR_CTAXDAMP', 'FILTER_RC0_PXMF_P', 'RES_ISR_Z_AVG', 'USA_ABSRAT', 'RC0_XX_LB', 'RC0_POIL_P_BAR', 'CONSTANT_RC0_TFP_FE_R', 'RES_RC0_Q_P', 'ISR_GIDAMP', 'USA_INFCPIX_TAR', 'RC0_ABSRAT', 'ISR_TARIFF_IM', 'USA_CPI12', 'RC0_TAXCBASE_N_BAR', 'USA_RC0_L', 'RC0_RK7', 'RC0_IFOOD2', 'CONSTANT_USA_C_OLG_R', 'RPFOOD3', 'ISR_XX_TAU_K', 'FILTER_USA_PI_P', 'ISR_DAMP_GDP_GAP', 'RC0_RK2', 'USA_TMETAL', 'RC0_NPOP_R_EXOG', 'USA_WEXP1', 'FILTER_ISR_TAU_OIL', 'ISR_COIL3', 'USA_CPI3', 'RC0_GI_MPROP', 'RES_ISR_INT', 'USA_PCFOOD_P_BAR', 'USA_IMETAL3', 'USA_TFPRAT', 'RES_RC0_MROYALTIES_N', 'USA_CPI2', 'CONSTANT_RC0_C_OLG_R', 'RES_ISR_RCORP', 'USA_PARTH_ADJ_STAR', 'RC0_PXM2', 'USA_TRANSFER_RAT_BAR', 'ISR_CPI10', 'FILTER_RC0_TAU_OIL', 'RC0_GILRSW', 'ISR_TRANSFER_TARG_RAT_EXOG', 'RES_RPOIL_CYC', 'USA_CPI10', 'RES_USA_PRODMETAL_R', 'USA_GI_N_EXOG', 'RC0_TAXOILBASE_N_BAR', 'CONSTANT_RC0_PRIMSUR_TAR', 'USA_XFOOD1', 'USA_XFOOD3', 'RC0_TOIL', 'FILTER_RPMETAL_PERM', 'ISR_IMETAL4', 'RPOIL1', 'RES_USA_INTMP', 'ISR_NPOP_R_EXOG', 'RES_ISR_Z_NFA', 'USA_PARTH1', 'RC0_PRODOIL_R_BAR', 'USA_INTMP_BAR', 'USA_LSTAX_N', 'ISR_MFACTOR', 'CONSTANT_RC0_PXMF_P', 'USA_TRANSFERLRSW', 'FILTER_RC0_PARTH', 'RC0_XOIL1', 'RC0_GIDAMP', 'CONSTANT_USA_PARTH', 'ISR_COM1', 'ISR_ABSRAT', 'USA_XMETAL3', 'USA_CPI8', 'RC0_ABSRATROOT', 'RC0_C_LIQ_R_BAR', 'ISR_USA_S', 'RC0_IM2', 'USA_B_STAR', 'USA_MKT1', 'RPOIL4', 'RC0_MKT1', 'TRDEMETAL_R', 'CONSTANT_ISR_PG_P', 'USA_NSPILLX1', 'RES_USA_GNP_R', 'RES_ISR_INTMP', 'FILTER_ISR_PI_P', 'CONSTANT_ISR_UNRH_FE', 'RES_USA_INTCORP', 'RES_RC0_W_R_AVG', 'USA_PRODOIL_R_BAR', 'RC0_INTWT', 'RC0_PRODOIL_R_EXOG', 'RES_USA_Z_AVG', 'USA_IMETAL5', 'CONSTANT_USA_PCW_P', 'RC0_PROB', 'RC0_EPS_TAR', 'CONSTANT_WTRADE_OIL_N', 'RC0_PARTH2', 'RES_RC0_EPS', 'ISR_INTGR', 'RC0_RK1', 'RES_USA_INFWEXP', 'USA_TFOOD', 'USA_COSTMETAL_P', 'RC0_PARTH3', 'USA_TAXLBASE_N_BAR', 'FILTER_USA_XM_R', 'ISR_GDP_N_BAR', 'FILTER_USA_CPINCOM_P', 'USA_XOIL5', 'USA_XX_TAU_K', 'RES_RC0_BREVAL_N', 'ISR_PARTH3', 'RC0_XOIL3', 'RC0_COIL1', 'USA_TRANSFER_TARG_RAT_EXOG', 'USA_IM2', 'RES_USA_WH_R', 'RC0_GC_MPROP', 'RC0_GDP_R_BAR', 'FILTER_USA_IOIL_R', 'CONSTANT_USA_TPREM', 'USA_GI_R_BAR', 'RC0_TRANSFER_RAT_BAR', 'RES_USA_PGDP_P_AVG', 'USA_MNFACTOR', 'RES_ISR_TAXOIL_N', 'CONSTANT_USA_UNRH_FE', 'ISR_PXM2', 'USA_CPI6', 'ISR_USA_SFOOD', 'USA_COIL1', 'RC0_TFPRATROOT', 'RES_ISR_PRODFOOD_PERM_R', 'RES_USA_TFP_FE_R_AVG', 'ISR_PCFOODBASE', 'RES_PMETAL_PERM_P']
    
    # mod_eqs, new_endog, map_new_endog, leads, lags = fixEquations(eqs=[eq],endog=endog,params=params)
    # print(mod_eqs)
    
    # eq = 'x**2+a*x**4 + c + y'
    # x = 'x'
    # eqs = ['E8_dot4_cpi_ber=u1*(dot_cpi(-1) +dot_cpi(-2) +dot_cpi(-3) +dot_cpi(-4))/4 +u2*ss_target +(1-u1-u2)*E8_dot4_cpi_ber(-1) +e_E8_dot4_cpi_ber', 'lcpi_ber=lcpi_ber(-4)+E8_dot4_cpi_ber', 'lgdp_gap=a1*lgdp_gap(+1)+a2*lgdp_gap(-1)- a3*(lrr_gap+0*term40_gap)+a4*lz_gap+a5*lx_gdp_gap+ a6*(lqforcmd_gap-0.15*lqoil_gap)+e_lgdp_gap', 'gg=j1*gg(-1)+(1-j1)*ss_g+e_g', 'lgdp_eq=lgdp_eq(-1)+gg/4- theta*((lgdp_eq(-1)-lgdp_eq(-2))-gg(1)/4)+e_lgdp_eq', 'dot_gdp=4*(lgdp-lgdp(-1))', 'dot4_gdp=lgdp-lgdp(-4)', 'dot_gdp_eq=4*(lgdp_eq-lgdp_eq(-1))', 'lgdp_gap=lgdp-lgdp_eq', 'lcpi=w_food*lcpi_food +w_elec*lcpi_elec +w_petr*lcpi_petr +w_goodsx*lcpi_goodsx +w_serv*lcpi_serv +e_lcpi', 'lcpi_core*(w_goodsx+w_serv)=w_goodsx*lcpi_goodsx +w_serv*lcpi_serv +e_lcpi_core', 'dot_cpi_serv=(1-b11-b12)*((1-b16)*dot_cpi(+1)+b16*E8_dot4_cpi_ber)+ b11*dot_cpi_serv(-1)+b12*dot_pm_serv+ rmc_serv+e_dot_cpi_serv', 'dot_cpi_goodsx=(1-b21-b22)*((1-b26)*dot_cpi(+1)+b26*E8_dot4_cpi_ber)+ b21*dot_cpi_goodsx(-1)+b22*dot_pm_goodsx+ rmc_goodsx+e_dot_cpi_goodsx', 'rmc_serv=b13*lgdp_gap+b14*lrulc_gap+b15*lz_gap', 'rmc_goodsx=b23*lgdp_gap+b24*lrulc_gap+b25*lz_gap', 'dot_pm_serv=c1*dot_pm_serv(-1)+ (1-c1)*(dot_s+dot_x_cpi-dot_z_eq)+e_dot_pm_serv', 'dot_pm_goodsx=c1*dot_pm_goodsx(-1)+ (1-c1)*(dot_s+dot_x_cpi-dot_z_eq)+e_dot_pm_goodsx', 'dot_cpi_food=(1-b31-b32-b36)*((1-b37)*dot_cpi(+1)+b37*E8_dot4_cpi_ber)+ b31*dot_cpi_food(-1)+b32*(dot_food+dot_s)+rmc_food+ b36*dot4_cpi_petr+e_dot_cpi_food', 'rmc_food=b33*lgdp_gap+b34*lrulc_gap+b35*(lqfood_gap+lz_gap)', 'dot_cpi_elec=b41*dot_cpi_elec(-1)+(1-b41)*dot_cpi(+1) +b42*rmc_elec+e_dot_cpi_elec', 'rmc_elec=b43*lgdp_gap+(1-b43)*lrulc_gap', 'dot_cpi_petr=dot_petrol+e_dot_cpi_petr', 'lpetrol=w_bfp*lbfp+(1-w_bfp)*lpettax+e_lpetrol', 'dot_bfp=dot_oil+dot_s+e_dot_bfp', 'dot_pettax=h6*dot_pettax(-1)+(1-h6)*ss_dot_pettax+e_dot_pettax', 'target=t1*target(-1)+(1-t1)*ss_target+e_target', 'lcpi=lcpi(-1)+dot_cpi/4', 'lcpi_core=lcpi_core(-1)+dot_cpi_core/4', 'lcpi_serv=lcpi_serv(-1)+dot_cpi_serv/4', 'lcpi_goodsx=lcpi_goodsx(-1)+dot_cpi_goodsx/4', 'lcpi_food=lcpi_food(-1)+dot_cpi_food/4', 'lcpi_elec=lcpi_elec(-1)+dot_cpi_elec/4', 'lcpi_petr=lcpi_petr(-1)+dot_cpi_petr/4', 'dot4_cpi=lcpi-lcpi(-4)', 'dot4_cpi_core=lcpi_core-lcpi_core(-4)', 'dot4_cpi_serv=lcpi_serv-lcpi_serv(-4)', 'dot4_cpi_goodsx=lcpi_goodsx-lcpi_goodsx(-4)', 'dot4_cpi_food=lcpi_food-lcpi_food(-4)', 'dot4_cpi_elec=lcpi_elec-lcpi_elec(-4)', 'dot4_cpi_petr=lcpi_petr-lcpi_petr(-4)', 'dot4_pm_goodsx=(dot_pm_goodsx+dot_pm_goodsx(-1)+ dot_pm_goodsx(-2)+dot_pm_goodsx(-3))/4', 'dot4_pm_serv=(dot_pm_serv+dot_pm_serv(-1)+ dot_pm_serv(-2)+dot_pm_serv(-3))/4', 'dot_petrol=(lpetrol-lpetrol(-1))*4', 'dot4_petrol=(lpetrol-lpetrol(-4))', 'lbfp=lbfp(-1)+dot_bfp/4', 'dot4_bfp=(lbfp-lbfp(-4))', 'lpettax=lpettax(-1)+dot_pettax/4', 'dot4_pettax=(lpettax-lpettax(-4))', 'edot4_cpi=(dot4_cpi(+1)+dot4_cpi(+2)+dot4_cpi(+3)+dot4_cpi(+4))/4', 'edot40_cpi=(edot4_cpi+edot4_cpi(+4)+edot4_cpi(+8)+ edot4_cpi(+12)+edot4_cpi(+16)+edot4_cpi(+20)+ edot4_cpi(+24)+edot4_cpi(+28)+edot4_cpi(+32)+ edot4_cpi(+36))/10', 'w_bfp=w_bfp(-1)+e_w_bfp', 'w_food=w_food(-1)+e_w_food', 'w_petr=w_petr(-1)+e_w_petr', 'w_elec=w_elec(-1)+e_w_elec', 'w_goodsx=w_goodsx(-1)+e_w_goodsx', 'w_serv=w_serv(-1)+e_w_serv', 'dot_w=(1-w1-w2)*dot_w(+1)+w1*dot_w(-1)+ w2*(dot_cpi(-1)+dot_rw_eq)-w3*lrulc_gap+ e_dot_w', 'lrulc_gap=lrw_gap-(lgdp_gap-lemp_gap)', 'lemp_gap=k1*lemp_gap(-1)+k2*lgdp_gap(-1)+e_lemp_gap', 'gg_dot_rw_eq=h8*gg_dot_rw_eq(-1)+(1-h8)*(gg-gg_dot_emp_eq)+e_dot_rw_eq', 'lrw_eq=lrw_eq(-1)+gg_dot_rw_eq/4- theta1*((lrw_eq(-1)-lrw_eq(-2))-gg_dot_rw_eq(1)/4)+e_lrw_eq', 'gg_dot_emp_eq=h1*gg_dot_emp_eq(-1)+(1-h1)*ss_dot_emp_eq+e_dot_emp_eq', 'lemp_eq=lemp_eq(-1)+gg_dot_emp_eq/4- theta2*((lemp_eq(-1)-lemp_eq(-2))-gg_dot_emp_eq(1)/4)+e_lemp_eq', 'lw=lw(-1)+dot_w/4', 'dot4_w=(dot_w+dot_w(-1)+dot_w(-2)+dot_w(-3))/4', 'lrw=lrw(-1)+dot_rw/4', 'dot_rw=dot_w-dot_cpi', 'dot4_rw=(dot_rw+dot_rw(-1)+dot_rw(-2)+dot_rw(-3))/4', 'dot_rw_eq=4*(lrw_eq-lrw_eq(-1))', 'lrw_gap=lrw-lrw_eq', 'dot_emp=4*(lemp-lemp(-1))', 'dot4_emp=lemp-lemp(-4)', 'dot_emp_eq=4*(lemp_eq-lemp_eq(-1))', 'lemp_gap=lemp-lemp_eq', 'lulc=lulc(-1)+dot_ulc/4', 'dot_ulc=dot_w+dot_emp-dot_gdp', 'dot4_ulc=(dot_ulc+dot_ulc(-1)+dot_ulc(-2)+dot_ulc(-3))/4', 'ls=e1*ls(+1)+(1-e1)*(ls(-1)+ 2/4*(target-ss_dot_x_cpi+dot_z_eq))+ (-rn+x_rn+prem)/4+e_ls', 'prem=h0*prem(-1)+(1-h0)*prem_eq+e_prem', 'gg_dot_z_eq=h5*gg_dot_z_eq(-1)+(1-h5)*(ss_dot_z_eq-g2*dot_qforcmd_eq)+e_dot_z_eq', 'lz_eq=lz_eq(-1)+gg_dot_z_eq/4- theta3*((lz_eq(-1)-lz_eq(-2))-gg_dot_z_eq(1)/4)+e_lz_eq', 'dot_z_eq(+1)=rr_eq-x_rr_eq-prem_eq+shock_prem', 'prem_eq=hh0*prem_eq(-1)+(1-hh0)*ss_prem_eq+e_prem_eq', 'rr_eq=h4*rr_eq(-1)+(1-h4)*(ss_rr_eq)+e_rr_eq', 'rr40=((rr)+(rr(1))+(rr(2))+(rr(3))+(rr(4))+(rr(5))+(rr(6))+(rr(7))+(rr(8))+(rr(9))+(rr(10))+(rr(11))+(rr(12))+(rr(13))+(rr(14))+(rr(15))+(rr(16))+(rr(17))+(rr(18))+(rr(19))+(rr(20))+(rr(21))+(rr(22))+(rr(23))+(rr(24))+(rr(25))+(rr(26))+(rr(27))+(rr(28))+(rr(29))+(rr(30))+(rr(31))+(rr(32))+(rr(33))+(rr(34))+(rr(35))+(rr(36))+(rr(37))+(rr(38))+(rr(39))) / 40', 'rr40_eq=((rr_eq)+(rr_eq(1))+(rr_eq(2))+(rr_eq(3))+(rr_eq(4))+(rr_eq(5))+(rr_eq(6))+(rr_eq(7))+(rr_eq(8))+(rr_eq(9))+(rr_eq(10))+(rr_eq(11))+(rr_eq(12))+(rr_eq(13))+(rr_eq(14))+(rr_eq(15))+(rr_eq(16))+(rr_eq(17))+(rr_eq(18))+(rr_eq(19))+(rr_eq(20))+(rr_eq(21))+(rr_eq(22))+(rr_eq(23))+(rr_eq(24))+(rr_eq(25))+(rr_eq(26))+(rr_eq(27))+(rr_eq(28))+(rr_eq(29))+(rr_eq(30))+(rr_eq(31))+(rr_eq(32))+(rr_eq(33))+(rr_eq(34))+(rr_eq(35))+(rr_eq(36))+(rr_eq(37))+(rr_eq(38))+(rr_eq(39))) / 40', 'lrr=lrrw01*rr+lrrw40*rr40', 'lrr_eq=lrrw01*rr_eq+lrrw40*rr40_eq', 'lrr_gap=lrr-lrr_eq', 'term40=h4*term40(-1)+(1-h4)*ss_term40+ h7*(prem-ss_prem_eq)+e_term40', 'term40_gap=term40-ss_term40', 'rn40=rr40+edot40_cpi+term40', 'rn=f1*rn(-1)+ (1-f1)*(rn_neutral+ f2*((dot4_cpi(+3)+dot4_cpi(+4)+dot4_cpi(+5))/3-ss_target)+ f3*lgdp_gap)+e_rn', 'ls=ls(-1)+dot_s/4', 'dot4_s=(dot_s+dot_s(-1)+dot_s(-2)+dot_s(-3))/4', 'lz=ls+lx_cpi-lcpi', 'dot_z=4*(lz-lz(-1))', 'dot_z_eq=4*(lz_eq-lz_eq(-1))', 'lz_gap=lz-lz_eq', 'rn_neutral=rr_eq+target', 'rr=rn-dot4_cpi(+1)', 'rr_gap=rr-rr_eq', 'rr_core=rn-dot4_cpi_core(+1)', 'rr_core_gap=rr_core-rr_eq', 'dot_forcmd=4*(lforcmd-lforcmd(-1))', 'lqforcmd=lforcmd-lx_cpi', 'lqforcmd=lqforcmd_eq+lqforcmd_gap', 'dot_qforcmd_eq=4*(lqforcmd_eq-lqforcmd_eq(-1))', 'lqforcmd_gap=g4*lqforcmd_gap(-1)+e_lqforcmd_gap', 'dot_qforcmd_eq=g4*dot_qforcmd_eq(-1)+(1-g4)*(ss_dot_qforcmd_eq)+e_dot_qforcmd_eq', 'dot_food=4*(lfood-lfood(-1))', 'dot4_food=(dot_food+dot_food(-1)+dot_food(-2)+dot_food(-3))/4', 'lqfood=lfood-lx_cpi', 'lqfood=lqfood_eq+lqfood_gap', 'dot_qfood_eq=h2*dot_qfood_eq(-1)+(1-h2)*(ss_dot_qfood_eq)+e_dot_qfood_eq', 'dot_qfood_eq=4*(lqfood_eq-lqfood_eq(-1))', 'lqfood_gap=h2*lqfood_gap(-1)+e_lqfood_gap', 'dot_oil=4*(loil-loil(-1))', 'dot4_oil=(loil-loil(-4))', 'lqoil=loil-lx_cpi', 'lqoil=lqoil_eq+lqoil_gap', 'dot_qoil_eq=h2*dot_qoil_eq(-1)+(1-h2)*(ss_dot_qoil_eq)+e_dot_qoil_eq', 'dot_qoil_eq=4*(lqoil_eq-lqoil_eq(-1))', 'lqoil_gap=h2*lqoil_gap(-1)+e_lqoil_gap', 'x_rn=h3*x_rn(-1)+(1-h3)*(x_rr_eq+dot_x_cpi)+e_x_rn', 'x_rr=x_rn-dot4_x_cpi(+1)', 'x_rr_eq=h3*x_rr_eq(-1)+(1-h3)*ss_x_rr_eq+e_x_rr_eq', 'x_rr_gap=x_rr-x_rr_eq', 'dot_x_cpi=h3*dot_x_cpi(-1)+(1-h3)*ss_dot_x_cpi+e_dot_x_cpi', 'lx_cpi=lx_cpi(-1)+dot_x_cpi/4', 'lx_gdp=lx_gdp(-1)+dot_x_gdp/4', 'dot4_x_gdp=(dot_x_gdp+dot_x_gdp(-1)+dot_x_gdp(-2)+dot_x_gdp(-3))/4', 'dot4_x_cpi=(dot_x_cpi+dot_x_cpi(-1)+dot_x_cpi(-2)+dot_x_cpi(-3))/4', 'lx_gdp_eq=lx_gdp_eq(-1)+g_x/4+e_lx_gdp_eq', 'g_x=jx1*g_x(-1)+(1-jx1)*ss_g_x+e_g_x', 'dot_x_gdp_eq=4*(lx_gdp_eq-lx_gdp_eq(-1))', 'lx_gdp_gap=lx_gdp-lx_gdp_eq', 'lx_gdp_gap=h3*lx_gdp_gap(-1)+e_lx_gdp_gap', 'E0_dot4_cpi=dot4_cpi', 'E1_dot_cpi=dot_cpi(+1)', 'E1_dot4_cpi=dot4_cpi(+1)', 'E3_dot4_cpi=dot4_cpi(+3)', 'E4_dot4_cpi=dot4_cpi(+4)', 'E5_dot4_cpi=dot4_cpi(+5)', 'E1_lgdp_gap=lgdp_gap(+1)', 'E1_dot4_x_cpi=dot4_x_cpi(+1)', 'E1_dot_z_eq=dot_z_eq(+1)', 'E1_ls=ls(+1)']
    # variables = ['E8_dot4_cpi_ber', 'lcpi_ber', 'lgdp', 'lgdp_eq', 'lgdp_gap', 'dot_gdp', 'dot4_gdp', 'dot_gdp_eq', 'gg', 'lcpi', 'dot_cpi', 'dot4_cpi', 'lcpi_core', 'dot_cpi_core', 'dot4_cpi_core', 'lcpi_serv', 'dot_cpi_serv', 'dot4_cpi_serv', 'rmc_serv', 'dot_pm_serv', 'dot4_pm_serv', 'lcpi_goodsx', 'dot_cpi_goodsx', 'dot4_cpi_goodsx', 'rmc_goodsx', 'dot_pm_goodsx', 'dot4_pm_goodsx', 'lcpi_food', 'dot_cpi_food', 'dot4_cpi_food', 'rmc_food', 'lcpi_elec', 'dot_cpi_elec', 'dot4_cpi_elec', 'rmc_elec', 'lcpi_petr', 'dot_cpi_petr', 'dot4_cpi_petr', 'lpetrol', 'dot_petrol', 'dot4_petrol', 'lpettax', 'dot_pettax', 'dot4_pettax', 'lbfp', 'dot_bfp', 'dot4_bfp', 'w_bfp', 'target', 'edot4_cpi', 'edot40_cpi', 'w_serv', 'w_goodsx', 'w_food', 'w_elec', 'w_petr', 'lw', 'dot_w', 'dot4_w', 'lrw_gap', 'dot_rw_eq', 'gg_dot_rw_eq', 'dot_rw', 'dot4_rw', 'lrw', 'lrw_eq', 'lemp', 'lemp_eq', 'lemp_gap', 'dot_emp', 'dot4_emp', 'dot_emp_eq', 'gg_dot_emp_eq', 'lrulc_gap', 'lulc', 'dot_ulc', 'dot4_ulc', 'ls', 'dot_s', 'dot4_s', 'prem', 'prem_eq', 'shock_prem', 'rn', 'rr', 'rr_eq', 'rr_gap', 'rn_neutral', 'rn40', 'rr40', 'rr40_eq', 'lrr', 'lrr_eq', 'lrr_gap', 'term40', 'term40_gap', 'rr_core', 'rr_core_gap', 'lz', 'lz_eq', 'lz_gap', 'dot_z', 'dot_z_eq', 'gg_dot_z_eq', 'lfood', 'dot_food', 'dot4_food', 'lqfood', 'lqfood_eq', 'lqfood_gap', 'dot_qfood_eq', 'lforcmd', 'dot_forcmd', 'lqforcmd', 'lqforcmd_eq', 'lqforcmd_gap', 'dot_qforcmd_eq', 'loil', 'dot_oil', 'dot4_oil', 'lqoil', 'lqoil_eq', 'lqoil_gap', 'dot_qoil_eq', 'lx_gdp', 'lx_gdp_eq', 'lx_gdp_gap', 'dot_x_gdp', 'dot4_x_gdp', 'dot_x_gdp_eq', 'g_x', 'x_rn', 'x_rr', 'x_rr_eq', 'x_rr_gap', 'lx_cpi', 'dot_x_cpi', 'dot4_x_cpi', 'E1_dot_cpi', 'E1_dot4_cpi', 'E3_dot4_cpi', 'E4_dot4_cpi', 'E5_dot4_cpi', 'E1_lgdp_gap', 'E1_dot4_x_cpi', 'E1_dot_z_eq', 'E0_dot4_cpi', 'E1_ls']
    # parameters = ['u1', 'u2', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'j1', 'theta', 'b11', 'b12', 'b13', 'b14', 'b15', 'b16', 'b21', 'b22', 'b23', 'b24', 'b25', 'b26', 'b31', 'b32', 'b33', 'b34', 'b35', 'b36', 'b37', 'b41', 'b42', 'b43', 'c1', 't1', 'h6', 'ss_target', 'ss_dot_pettax', 'w1', 'w2', 'w3', 'k1', 'k2', 'h1', 'h8', 'ss_dot_emp_eq', 'ss_prem_eq', 'theta1', 'theta2', 'ss_rr_eq', 'ss_dot_z_eq', 'ss_g', 'ss_term40', 'lrrw01', 'lrrw40', 'e1', 'f1', 'f2', 'f3', 'g2', 'h0', 'h4', 'h5', 'h7', 'hh0', 'h9', 'theta3', 'ss_dot_qfood_eq', 'ss_dot_qforcmd_eq', 'ss_dot_qoil_eq', 'g4', 'h2', 'ss_dot_x_cpi', 'ss_x_rr_eq', 'ss_g_x', 'jx1', 'h3']
    #ss_eqs = get_steady_state_equations(eqs,variables,parameters)
    
    eqs = ['GGR_ST=GGR*(NPGDP_OVER_HEADLINE^ELAS_C_ST)*(NPGDP_OVER_HEADLINE(-1)^ELAS_L_ST)']
    variables = ['GGR','NPGDP_OVER_HEADLINE','ELAS_C_ST','ELAS_L_ST']
    parameters = []
    print(eqs)
    ss_eqs = get_steady_state_equations(eqs,variables,parameters)
    print(ss_eqs)
    
    
    
    # eq = 'ISR_INTCORP__ - (((ISR_RPREM__ + 1)*(ISR_RPREM__p1_ + 1)*(ISR_RPREM_plus_1__p1_ + 1)*(ISR_RPREM_plus_2__p1_ + 1)*(ISR_RPREM_plus_3__p1_ + 1)*(ISR_RPREM_plus_4__p1_ + 1)*(ISR_RPREM_plus_5__p1_ + 1)*(ISR_RPREM_plus_6__p1_ + 1)*(ISR_RPREM_plus_7__p1_ + 1)*(ISR_RPREM_plus_8__p1_ + 1))**0.1*(ISR_INT10__ + 1))**ISR_INTCORP1*((ISR_INT__ + 1)*(ISR_RPREM__ + 1))**(1 - ISR_INTCORP1)*(E_ISR_INTCORP + 1)*(RES_ISR_INTCORP + 1) + 1'
    # x = 'ISR_INTCORP__'
    # s   = sy.symbols(x)
    # sol = sy.solve(eq,s)
               
    # rhs = str(sol[0])
    # print(rhs)
    
    # eq = '(10000000*NFIP)=(10000000*NFI)-(10000000*NFIG)'
    # var = ['NFI','NFIG']
    # print(replaceEquation(eq,var))
    
    
    
    # from snowdrop.src.utils.util import findVariableLead
    # from snowdrop.src.utils.util import findVariableLag
    
    # i = -1; T = 10
    # var = ["x"]; me = {"x":[0]*(1+T)} 
    # params = ["a"]; mp = {"a":1.5}
    # eqs = ["x = a*x__m1_+2"]
    # lfunc,lnames,largs,leqs = build_func(var,params,eqs)
    
    # m = dict()
    # for x in lnames:
    #     m[x] = np.zeros(T)
        
    # for func,arg in zip(lfunc,largs):
    #     i += 1
    #     for t in range(1,T):
    #         lst = list()
    #         for j in range(len(arg)):
    #             a = arg[j]
    #             lead = findVariableLead(a)
    #             lag  = findVariableLag(a)
    #             tt = t + lead + lag
    #             if "__" in a:
    #                 a = a[:a.index("__")] 
    #             y = eval(a,m,mp)
    #             if a in var:
    #                 y = y[tt]
    #             lst.append(y)
    #         if len(lst) > 0:
    #             u = func(*lst)
    #         else:
    #             u = func()
    #         m[x][t] = u
            
    # print(m[x])
    
        
        
