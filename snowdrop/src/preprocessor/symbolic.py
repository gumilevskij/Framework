"""
 All symbolic functions take ast expression trees (not expressions) as input.
 This one can be constructed as : ast.parse(s).body[0].value
"""

import os
import copy
import ast
from yaml import ScalarNode

known_functions = ['log','exp','sin','cos','tan','max','min','Max','Min','Heaviside','abs','sign','IfThenElse','IfThen','Positive','Negative','LambertW','myzif']
Expression = ast.Expr

from ast import UnaryOp, UAdd, USub, Name, Load, Call
from ast import NodeTransformer
from typing import Tuple, List
from snowdrop.src.preprocessor.language import functions as functions_dict
from snowdrop.src.preprocessor.codegen import to_source

functions = list(functions_dict.keys())


def list_variables(expr: Expression, funs: List[str]=None, vars: List[str]=None) -> List[Tuple[str,int]]:

    if funs is None: 
        funs=[]
    if vars is None: 
        vars=[]
    l = ListSymbols(known_functions=functions+funs, known_variables=vars)
    l.visit(expr)
    if l.problems:
        e = Exception('Symbolic error.')
        e.problems = l.problems
        raise e
    return [v[0] for v in l.variables]


def time_shift(expr: Expression, n, vars: List[str] = []) -> Expression:
    """
    Shifts timing in equations variables.
    
        Example:
        time_shift(:(a+b(1)+c),1,[:b,:c]) == :(a+b(2)+c(1))
    """
    eexpr = copy.deepcopy(expr)
    return TimeShiftTransformer(shift=n, variables=vars).visit(eexpr)


def stringify_variable(arg: Tuple[str, int]) -> str:
    """
    Stringify a variable.
    
    This method encodes varaible name with its lead or lag.
    """
    s = arg[0]
    date = arg[1]
    if date == 0:
        #return '{}__'.format(s)
        return s+'__'
        #return f'{s}__'
    elif date <= 0:
        #return '{}__m{}_'.format(s,-date)
        return s+'__m'+str(-date)+'_'
        #return f'{s}__m{-date}_'
    elif date > 0:
        #return '{}__{}_'.format(s,date)
        return s+'__p'+str(date)+'_'
        #return f'{s}__{date}_'


def log_stringify_variable(arg: Tuple[str, int]) -> str:
    """
    Return variable  with a log function of time shifted variable.
    
    This method encodes varaible name with its lead or lag.
    """
    s = arg[0]
    date = arg[1]
    if date == 0:
        #return '{}__'.format(s)
        return 'log('+s+'__)'
        #return f'{s}__'
    elif date <= 0:
        #return '{}__m{}_'.format(s,-date)
        return 'log('+s+'__m'+str(-date)+'_)'
        #return f'{s}__m{-date}_'
    elif date > 0:
        #return '{}__{}_'.format(s,date)
        return 'log('+s+'__p'+str(date)+'_)'
        #return f'{s}__{date}_'
        

def stringify_parameter(p: str) -> str:
    """Stringify a parameter."""
    return '{}'.format(p)


def stringify(arg) -> str:
    """Stingify a variable or a parameter."""
    if isinstance(arg, str):
        return stringify_parameter(arg)
    elif isinstance(arg, tuple):
        if len(arg)==2 and isinstance(arg[0],str) and isinstance(arg[1],int):
            return stringify_variable(arg)
    raise Exception("Unknown canonical form: {}".format(arg))


def stringify_symbol(arg) -> str:
    """Stingify symbol."""
    if isinstance(arg, str):
        return stringify_parameter(arg)
    elif isinstance(arg, tuple):
        if len(arg) == 2 and isinstance(arg[0], str) and isinstance(arg[1], int):
            return stringify_variable(arg)
    raise Exception("Unknown canonical form: {}".format(arg))


def destringify(s: str, variables: List[str] = []) -> Tuple[int, int]:
    """Find leads and lags of a variable from its name."""
    i = 0
    v = ''
    if "__" in s:
        ind = s.rindex("__")
        if not s.endswith("__"):
            lead_lag = s[ind:]
            v = s[:ind]
            lead_lag = lead_lag.replace("_","")
            if "m" in lead_lag:
                lead_lag = lead_lag[1:]
                if lead_lag.isnumeric():
                    i = - int(lead_lag)
            if "p" in lead_lag:
                lead_lag = lead_lag[1:]
                if lead_lag.isnumeric():
                    i = int(lead_lag)
            elif lead_lag.isnumeric():
                i = int(lead_lag)
        s = s[:ind]
        if "_plus_" in s:
            ind = s.rindex("_plus_")
            lead_lag = s[ind+6:]
            v = s[:ind]
            if lead_lag.isnumeric():
                i += int(lead_lag)
        elif "_minus_" in s:
            ind = s.rindex("_minus_")
            lead_lag = s[ind+7:]
            v = s[:ind]
            if lead_lag.isnumeric():
                i -= int(lead_lag)
    else:
       v = s
            
    if v in variables:
        j = variables.index(v)
    else:
        j = 0

    return (j,i)


def parse_string(text, start=None):

    from lark.lark import Lark
    from lark.exceptions import UnexpectedInput, UnexpectedCharacters
    
    DIR_PATH, this_filename = os.path.split(__file__)
    DATA_PATH = os.path.join(DIR_PATH, "grammar.lark")
    
    grammar = open(DATA_PATH, "rt", encoding="utf-8").read()
    parser = Lark(
        grammar,
        start=[
            "start",
            "variable",
            "equation_block",
            "assignment_block",
            "complementarity_block",
        ],
    )
    
    if start is None:
        start = "start"

    if isinstance(text, ScalarNode):
        if text.tag != "tag:yaml.org,2002:str":
            #     raise Exception(f"Don't know how to parse node {text}")
            txt = text.value
        else:
            if text.start_mark is None:
                txt = text.value
            else:
                buffer = text.end_mark.buffer
                i1 = text.start_mark.pointer
                i2 = text.end_mark.pointer
                txt = buffer[i1:i2]
                if text.style in (">", "|"):
                    txt = txt[1:]

    else:
        txt = text

    try:
        return parser.parse(txt, start)

    except (UnexpectedInput, UnexpectedCharacters) as e:

        if isinstance(text, ScalarNode):
            sm = text.start_mark
            # em = text.end_mark
            if text.style not in (">", "|"):
                new_column = sm.column + e.column
                new_line = sm.line + e.line
            else:
                new_line = sm.line + e.line
                new_column = e.column
            newargs = list(e.args)
            newargs[0] = e.args[0].replace(f"line {e.line}", f"line {new_line}")
            newargs[0] = newargs[0].replace(f"col {e.column}", f"col {new_column}")
            e.args = tuple(newargs)
            e.line = new_line
            e.column = new_column

        raise e
    
        
def normalize(expr: Expression, variables: List[str] = [])->Expression:
    """Replace calls to variables by their time subscripts."""
    try:
        en = ExpressionNormalizer(variables=variables)
        cp = copy.deepcopy(expr)
        e = en.visit(cp)
    except:
        print("Error: ")
        print(to_source(expr))
        e = None
    return e

    
def log_normalize(expr: Expression, variables: List[str] = [], log_variables: List[str] = [])->Expression:
    """Replace calls to variables by their time subscripts."""
    try:
        en = ExpressionLogNormalizer(variables=variables,log_variables=log_variables)
        cp = copy.deepcopy(expr)
        e = en.visit(cp)
    except:
        print("Error: ")
        print(to_source(expr))
        e = None
    return e


def std_tsymbol(tsymbol):
    """Return string encoded with leads/lags."""
    s, date = tsymbol
    if date == 0:
        return '_{}_'.format(s)
    elif date <= 0:
        return '_{}_m{}_'.format(s, str(-date))
    elif date >= 0:
        return '_{}__{}_'.format(s, str(date))


class StandardizeDatesSimple(NodeTransformer):
    """Replaces calls to variables by time subscripts."""

    def __init__(self, variables):

        self.variables = variables
        # self.variables = tvariables # ???

    def visit_Name(self, node):
        """Visitor for Name node."""
        name = node.id
        if name in self.variables:
            return Name(id=std_tsymbol((name,0)),ctx=Load())
        else:
            return node

    def visit_Call(self, node):
        """Visitor for Call node."""
        name = node.func.id
        args = node.args[0]

        if name in self.variables:
            if isinstance(args, UnaryOp):
                # we have s(+1)
                if (isinstance(args.op, UAdd)):
                    args = args.operand
                    date = args.n
                elif (isinstance(args.op, USub)):
                    args = args.operand
                    date = -args.n
                else:
                    raise Exception("Unrecognized subscript.")
            else:
                date = args.n
            newname = std_tsymbol((name, date))
            if newname is not None:
                return Name(newname, Load())

        else:

            # , keywords=node.keywords, starargs=node.starargs, kwargs=node.kwargs)
            return Call(func=node.func, args=[self.visit(e) for e in node.args], keywords=[])



class TimeShiftTransformer(ast.NodeTransformer):
    def __init__(self, variables, shift=0):

        self.variables = variables
        self.shift = shift

    def visit_Name(self, node):
        """Visitor for Name node."""
        name = node.id
        if name in self.variables:
            if self.shift==0 or self.shift=='S':
                return ast.parse(name).body[0].value
            else:
                return ast.parse('{}({})'.format(name,self.shift)).body[0].value
        else:
             return node

    def visit_Call(self, node):
        """Visitor for Call node."""
        name = node.func.id
        args = node.args[0]

        if name in self.variables:
            if isinstance(args, UnaryOp):
                # we have s(+1)
                if (isinstance(args.op, UAdd)):
                    args = args.operand
                    date = args.n
                elif (isinstance(args.op, USub)):
                    args = args.operand
                    date = -args.n
                else:
                    raise Exception("Unrecognized subscript.")
            else:
                date = args.n
            if self.shift =='S':
                return ast.parse('{}'.format(name)).body[0].value
            else:
                new_date = date+self.shift
                if new_date != 0:
                    return ast.parse('{}({})'.format(name,new_date)).body[0].value
                else:
                    return ast.parse('{}'.format(name)).body[0].value
        else:

            # , keywords=node.keywords,  kwargs=node.kwargs)
            return Call(func=node.func, args=[self.visit(e) for e in node.args], keywords=[])




def timeshift(expr, variables, shift):

    eexpr = copy.deepcopy(expr)
    return TimeShiftTransformer(variables, shift).visit(eexpr)


class Compare:
    """
    Compares two ast tree instances.
    
    .. currentmodule: preprocessor
    """

    def __init__(self):
        self.d = {}

    def compare(self, A, B):
        """Compare two nodes."""
        if isinstance(A, ast.Name) and (A.id[0] == '_'):
            if A.id not in self.d:
                self.d[A.id] = B
                return True
            else:
                return self.compare(self.d[A.id], B)
        if not (A.__class__ == B.__class__): return False
        if isinstance(A, ast.Name):
            return A.id == B.id
        elif isinstance(A, ast.Call):
            if not self.compare(A.func, B.func): return False
            if not len(A.args)==len(B.args): return False
            for i in range(len(A.args)):
                if not self.compare(A.args[i], B.args[i]): return False
            return True
        elif isinstance(A, ast.Num):
            return A.n == B.n
        elif isinstance(A, ast.Expr):
            return self.compare(A.value, B.value)
        elif isinstance(A, ast.Module):
            if not len(A.body)==len(B.body): return False
            for i in range(len(A.body)):
                if not self.compare(A.body[i], B.body[i]): return False
            return True
        elif isinstance(A, ast.BinOp):
            if not isinstance(A.op, B.op.__class__): return False
            if not self.compare(A.left, B.left): return False
            if not self.compare(A.right, B.right): return False
            return True
        elif isinstance(A, ast.UnaryOp):
            if not isinstance(A.op, B.op.__class__): return False
            return self.compare(A.operand, B.operand)
        elif isinstance(A, ast.Subscript):
            if not self.compare(A.value, B.value): return False
            return self.compare(A.slice, B.slice)
        elif isinstance(A, ast.Index):
            return self.compare(A.value, B.value)
        elif isinstance(A, ast.Compare):
            if not self.compare(A.left, B.left): return False
            if not len(A.ops)==len(B.ops): return False
            for i in range(len(A.ops)):
                if not self.compare(A.ops[i], B.ops[i]): return False
            if not len(A.comparators)==len(B.comparators): return False
            for i in range(len(A.comparators)):
                if not self.compare(A.comparators[i], B.comparators[i]): return False
            return True
        elif isinstance(A, ast.In):
            return True
        elif isinstance(A, (ast.Eq, ast.LtE)):
            return True
        else:
            print(A.__class__)
            raise Exception("Not implemented")


def compare(a,b):
    """Compare two nodes."""
    comp = Compare()
    val = comp.compare(a,b)
    return val

def match(m,s):
    comp = Compare()
    val = comp.compare(m,s)
    d = comp.d
    if len(d) == 0:
        return val
    else:
        return d


class ListNames(ast.NodeVisitor):
    def __init__(self):
        self.found = []
    def visit_Name(self, name):
        self.found.append(name.id)

def get_names(expr):
    ln = ListNames()
    ln.visit(expr)
    return [e for e in ln.found]

def eval_scalar(tree):
    try:
        if isinstance(tree, ast.Num):
            return tree.value
        elif isinstance(tree, ast.UnaryOp):
            if isinstance(tree.op, ast.USub):
                return -tree.operand.n
            if isinstance(tree.op, ast.UAdd):
                return tree.operand.n
        else:
            raise Exception("Don't know how to do that.")
    except:
        raise Exception("Don't know how to do that.")


class ExpressionChecker(ast.NodeVisitor):
    """
    Checks AST expressions.
    
    .. currentmodule: preprocessor
    
    """
    def __init__(self, spec_variables, known_functions, known_constants):
        self.spec_variables = spec_variables
        self.known_functions = known_functions
        self.known_constants = known_constants
        self.functions = []
        self.variables = []
        self.problems = []

    def visit_Call(self, call):
        name = call.func.id
        colno = call.func.col_offset
        if name in self.spec_variables:
            try:
                assert(len(call.args)==1)
                n = eval_scalar(call.args[0])
                allowed_timing = self.spec_variables[name]
                if allowed_timing is None or (n in allowed_timing):
                    self.variables.append((name, n, call.func.col_offset))
                else:
                    self.problems.append([name,n,colno,'incorrect_timing',allowed_timing])
            except Exception as e:
                print(e)
                self.problems.append([name,None,colno,'timing_error'])

        elif name in self.known_functions:
            self.functions.append((name, colno))
            for e in call.args:
                self.visit(e)
        else:
            self.problems.append([name, None, colno,'unknown_function'])

    def visit_Name(self, name):
        # colno = name.colno
        colno = name.col_offset
        n = 0
        name = name.id
        if name in self.spec_variables:
            allowed_timing = self.spec_variables[name]
            if (allowed_timing is None) or (n in allowed_timing):
                self.variables.append((name, n, colno))
            else:
                self.problems.append([name,n,colno,'incorrect_timing',allowed_timing])
        elif name not in self.known_constants:
            self.problems.append([name,0,colno,'unknown_variable'])

def check_expression(expr, spec_variables, known_functions=[]):

    from snowdrop.src.preprocessor.language import functions, constants
    func = list(functions.keys()) + known_functions

    ch = ExpressionChecker(spec_variables, func, constants)
    ch.visit(expr)
    return dict(
        functions = ch.functions,
        variables = ch.variables,
        problems = ch.problems
    )
    

class ExpressionNormalizer(NodeTransformer):
    """
    Replaces calls to variables by time subscripts.
    
    .. currentmodule: preprocessor.symbolic
    
    """

    def __init__(self, variables=None, functions=None):

        self.variables = variables if variables is not None else []
        if functions is None:
            self.functions = [e for e in functions_dict.keys()]
        else:
            self.functions = functions

    def visit_Name(self, node):

        name = node.id
        # if name self.functions:
        #     return node
        if name in self.variables:
            return Name(id=stringify_variable((name,0)), ctx=Load())
        else:
            return Name(id=stringify_parameter(name), ctx=Load())

    def visit_Call(self, node):

        name = node.func.id
        args = node.args[0]

        if name in self.variables or name not in self.functions:
            try:
                date = eval_scalar(args)
            except:
                raise Exception("Unrecognized subscript: name ={} args={}".format(name,args))
            newname = stringify_variable((name, date))
            if newname is not None:
                return Name(newname, Load())
        else:
            return Call(func=node.func, args=[self.visit(e) for e in node.args], keywords=[])


class ExpressionLogNormalizer(NodeTransformer):
    """
    Replaces calls to variables by log function of their time subscripts.
    
    .. currentmodule: preprocessor.symbolic
    
    """

    def __init__(self, variables=[], log_variables=[], functions=None):

        self.variables = variables
        self.log_variables = log_variables
        if functions is None:
            self.functions = [e for e in functions_dict.keys()]
        else:
            self.functions = functions

    def visit_Name(self, node):

        name = node.id
        # if name self.functions:
        #     return node
        if name in self.log_variables:
            return Name(id=log_stringify_variable((name,0)), ctx=Load())
        elif name in self.variables:
            return Name(id=stringify_variable((name,0)), ctx=Load())
        else:
            return Name(id=stringify_parameter(name), ctx=Load())

    def visit_Call(self, node):

        name = node.func.id
        args = node.args[0]

        if name in self.variables or name not in self.functions:
            try:
                date = eval_scalar(args)
            except:
                raise Exception("Unrecognized subscript: name ={} args={}".format(name,args))
            if name in self.log_variables:
                newname = log_stringify_variable((name, date))
            else:
                newname = stringify_variable((name, date))
            if newname is not None:
                return Name(newname, Load())
        else:
            return Call(func=node.func, args=[self.visit(e) for e in node.args], keywords=[])
        
        
class ListSymbols(ast.NodeVisitor):
    """
    Creates a lists of symbols by visiting each Call object in ast expression tree.
    
    .. currentmodule: preprocessor.symbolic
    
    """
    def __init__(self, known_functions=[], known_variables=[]):
        self.known_functions = known_functions
        self.known_variables = known_variables
        self.functions = []
        self.variables = []
        self.constants = []
        self.problems = []

    def visit_Call(self, call):
        name = call.func.id
        colno = call.func.col_offset
        if name in self.known_functions:
            self.functions.append((name, colno))
            [self.visit(e) for e in call.args]
        else:
            try:
                assert(len(call.args) == 1)
                n = int(eval_scalar(call.args[0]))
                self.variables.append(((name, n), colno))
            except:
                if name in self.known_variables + [vv[0][0] for vv in self.variables]:
                    self.problems.append([name, 0, colno, 'incorrect subscript'])
                else:
                    self.problems.append([name, 0, colno, 'unknown_function'])
                # [self.visit(e) for e in call.args]

    def visit_Name(self, name):
        # colno = name.colno
        colno = name.col_offset
        name = name.id
        if name in self.known_variables:
            self.variables.append(((name, 0), colno))
        elif name in self.known_functions:
            self.problems.append([name, colno, 'function_not_called'])
        else:
            self.constants.append((name, colno))
            
# def get_variables(variables, expr):
#     ln = ListVariables(variables)
#     ln.visit(expr)
#     return ln.found

# def get_functions(variables, expr):
#     ln = ListVariables(variables)
#     ln.visit(expr)
#     return ln.functions


# class ExpressionChecker(ast.NodeVisitor):
#
#     def __init__(self, variables, functions):
#
#         self.allowed_variables = variables
#         self.functions = functions
#         self.found = []
#         self.problems = []
#
#     def visit_Call(self, call):
#         name = call.func.id
#         if name in self.variables:
#             assert(len(call.args)==1)
#             print(call.args[0])
#             n = eval_scalar(call.args[0])
#             self.found.append((name, n))
#         elif name in self.functions:
#             self.functions.append(name)
#             for e in call.args:
#                 self.visit(e)
#         else:
#             for e in call.args:
#                 self.visit(e)
#
#     def visit_Name(self, name):
#         name = name.id
#         if name in self.variables:
#             self.found.append((name,0))
#
# def check_expression(expr, variables, functions):
#     ec = ExpressionChecker(variables, functions)
#     pbs = ec.visit(ec)

if __name__ == '__main__':
    """
    The main program
    """ 
    var = ['g', 'p_pdot1', 'p_pdot2', 'p_pdot3', 'p_rs1', 'p_y1', 'p_y2', 'p_y3', 'p_pdot1__m1_']
    s = var[1]
    x = stringify((s,-1))
    print(x)
    s = var[-1]
    sd = destringify(s,var)
    print(s,sd)
    