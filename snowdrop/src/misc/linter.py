"""Set of utilities to check correctness of model file syntax."""

import ast
import json

import ruamel.yaml as ry
from ruamel.yaml.comments import CommentedSeq
from snowdrop.src.preprocessor.symbolic import check_expression
from collections import OrderedDict
from snowdrop.src.preprocessor.recipes import recipes
from snowdrop.src.misc.termcolor import colored
from snowdrop.src.misc.termcolor import cprint

known_symbol_types = recipes['symbols']
version = ry.__version__

class ModelException(Exception):
    """Model exception class."""
    type = 'error'


def check_symbol_validity(s):
    """
    Check symbol validity.
    
    Parameters:
        :param s: Symbol
        :type s: str.
        :returns:
    """
    import ast
    val = ast.parse(s).body[0].value
    assert(isinstance(val, ast.Name))


def check_symbols(data):
    """
    Check model symbols validity.
    
    Parameters:
        :param data: Model content.
        :type data: Dictionary.
        :returns: List of exceptions if any.
        
    Can raise three types of exceptions:
      - unknown symbol
      - invalid symbol
      - already declared
    """
    exceptions = []
    symbols = data['symbols']
    cm_symbols = symbols
    already_declared = {}  # symbol: symbol_type, position

    for key, values in cm_symbols.items():
        # (start_line, start_column, end_line, end_column) of the key
        if key not in known_symbol_types:
            exc = ModelException("Unknown symbol type '{}'".format(key))
            if hasattr(cm_symbols,"lc"):
                l0, c0, l1, c1 = cm_symbols.lc.data[key]
                exc.pos = (l0, c0, l1, c1)
                # print(l0,c0,l1,c1)
            exceptions.append(exc)
            assert(isinstance(values, CommentedSeq))

        for i, v in enumerate(values):
            l0 = c0 = None
            try:
                check_symbol_validity(v)
            except:
                exc = ModelException("Invalid symbol '{}'".format(v))
                if hasattr(values,'lc'):
                    (l0, c0) = values.lc.data[i]
                    length = len(v)
                    l1 = l0
                    c1 = c0 + length
                    exc.pos = (l0, c0, l1, c1)
                exceptions.append(exc)
            if v in already_declared:
                ll = already_declared[v]
                if hasattr(v,'lc'):
                    exc = ModelException(
                        "Symbol '{}' already declared as '{}'. (pos {})".format(
                            v, ll[0], (ll[1][0] + 1, ll[1][1])
                        )
                    )
                    exc.pos = (l0, c0, l1, c1)
                else:
                    exc = ModelException(
                        "Symbol '{}' already declared as '{}'.".format(v, ll[0])
                    )
                    
                exceptions.append(exc)
            else:
                already_declared[v] = (key, (l0, c0))

    return exceptions


def check_extra_symbols(data):
    """
    Check extra variables declaration.
    
    Parameters:
        :param data: Model content.
        :type data: Dictionary.
        :returns: List of exceptions if any.
    """
    import re
    
    delimiters = " ", ",", ";", "*", "/", ":", "+", "-", "^", "{", "}", "(", ")", "="
    regexPattern = '|'.join(map(re.escape, delimiters))
    equations = data['equations']
    all_symbols = list()
    
    for eq in equations:
        if isinstance(eq,dict):
            for k in eq.keys():
                eq = eq[k]
        arr = re.split(regexPattern,eq)
        arr = list(filter(None,arr))
        all_symbols.extend(arr)
        
    symbols = data['symbols']
    if hasattr(symbols,"lc"):
        l0, c0, l1, c1 = symbols.lc.data['variables']
        pos0 = (l0, c0, l1, c1)
    else:
        pos0 = []
    exceptions = []
    variables = data['symbols']['variables']
    for v in variables:
        if not v in all_symbols:
            exc = ModelException("An extra variable {} was declared. Please remove it.".format(v))
            exc.pos = pos0
            exc.type = 'error'
            exceptions.append(exc)
            
    return exceptions


def check_equations(data):
    """
    Check correctness of model equations.
    
    Parameters:
        :param data: Model content.
        :type data: Dictionary.
        :returns: List of exceptions if any.
    """
    if hasattr(data,'lc'):
        pos0 = data.lc.data['equations']
    else:
        pos0 = []
    equations = data['equations']

    exceptions = []
    specs = recipes['specs']

    for eq_type in specs.keys():
        if (eq_type not in equations) and (not specs.get('optional', True)):
            exc = ModelException("Missing equation type {}.".format(eq_type))
            exc.pos = pos0
            exceptions.append(exc)

    return exceptions


def check_variables(data):
    """
    Check number of endogenous variables.
    
    Parameters:
        :param data: Model content.
        :type data: Dictionary.
        :returns: List of exceptions if any.
    """
    equations = data['equations']
    variables = data['symbols']['variables']
    if 'log_variables' in data['symbols']:
        variables += data['symbols']['log_variables']
    shocks = data['symbols']['shocks'] if 'shocks' in data['symbols'] else []
    parameters = data['symbols']['parameters'] if 'parameters' in data['symbols'] else []
    n_eqs = len(equations)
    n_vars = len(variables)

    exceptions = []
    if not n_eqs == n_vars:
        exc = ModelException(f"The number of equations ({n_eqs}) and the number of endogenous variables ({n_vars}) are different!\n")
        exc.type = 'error'
        exceptions.append(exc)
     
    # Check duplicates of endogenous and exogenous variables    
    duplicate = []
    for n in variables:
        if n in shocks:
            duplicate.append(n)
            
    if len(duplicate) > 0:
        common = ",".join(duplicate)
        exc = ModelException(f"Endogenous and exogenous variables have common members {common}!\n")
        exc.type = 'error'
        exceptions.append(exc)
            
    # Check duplicates of endogenous variables and parameters   
    duplicate = []
    for n in variables:
        if n in parameters:
            duplicate.append(n)
            
    if len(duplicate) > 0:
        common = ",".join(duplicate)
        exc = ModelException(f"Endogenous variables and parameters have common members {common}!\n")
        exc.type = 'error'
        exceptions.append(exc)

    return exceptions


def check_definitions(data):
    """
    Check correctness of model definitions.
    
    Parameters:
        :param data: Model content.
        :type data: Dictionary.
        :returns: List of exceptions if any.
    """
    if 'definitions' not in data:
        return []
    definitions = data['definitions']
    if definitions is None:
        return []

    exceptions = []
    known_symbols = sum(data['symbols'].values(), [])

    allowed_symbols = {v: (0,) for v in known_symbols}
    if 'parameters' in data['symbols']:
        for p in data['symbols']['parameters']:
            allowed_symbols[p] = (0,)

    new_definitions = OrderedDict()
    for k, v in definitions.items():
        pos = definitions.lc.data[k]
        if k in known_symbols:
            exc = ModelException(
                'Symbol {} has already been defined as a model symbol.'.format(k))
            exc.pos = pos
            exceptions.append(exc)
            continue
        if k in new_definitions:
            exc = ModelException(
                'Symbol {} cannot be defined twice.'.format(k))
            exc.pos = pos
            exceptions.append(exc)
            continue
        try:
            check_symbol_validity(k)
        except:
            exc = ModelException("Invalid symbol '{}'".format(k))
            exc.pos = pos
            exceptions.append(exc)

            # pos = equations[eq_type].lc.data[n]
        try:
            expr = ast.parse(str(v))

            # print(allowed_symbols)
            check = check_expression(expr, allowed_symbols)
            # print(check['problems'])
            for pb in check['problems']:
                name, t, offset, err_type = [pb[0], pb[1], pb[2], pb[3]]
                if err_type == 'timing_error':
                    exc = Exception(
                        'Timing for variable {} could not be determined.'.format(pb[0]))
                elif err_type == 'incorrect_timing':
                    exc = Exception(
                        'Variable {} cannot have time {}. (Allowed: {})'.format(name, t, pb[4]))
                elif err_type == 'unknown_function':
                    exc = Exception(
                        'Unknown variable/function {}.'.format(name))
                elif err_type == 'unknown_variable':
                    exc = Exception(
                        'Unknown variable/parameter {}.'.format(name))
                else:
                    print(err_type)
                exc.pos = (pos[0], pos[1] + offset, pos[0],
                           pos[1] + offset + len(name))
                exc.type = 'error'
                exceptions.append(exc)

            new_definitions[k] = v

            allowed_symbols[k] = (0,)  # TEMP
            # allowed_symbols[k] = None

        except SyntaxError as e:
            pp = pos  
            exc = ModelException("Syntax Error.")
            exc.pos = [pp[0], pp[1] + e.offset, pp[0], pp[1] + e.offset]
            exceptions.append(exc)

    return exceptions


def check_calibration(data):
    """
    Check correctness of model calibration.
    
    Parameters:
        :param data: Model content.
        :type data: Dictionary.
        :returns: List of exceptions if any.
    """
    
    # what happens here if symbols are not clean ?
    exceptions = []
    #all_symbols = []
    symbols = data['symbols']
    if hasattr(data,'lc'):
        pos0 = data.lc.data['calibration']
    else:
        pos0 = []
    if not 'calibration' in data:
        return exceptions
    calibration = data['calibration']
    # for v in symbols.values():
    #     all_symbols += v
    if "parameters" in symbols:
        for s in symbols["parameters"]:
            if not s in calibration:
                # should skip invalid symbols there
                exc = ModelException(
                    "Parameter {} has not been set in a model file".format(s))
                exc.pos = pos0
                exc.type = 'warning'
                exceptions.append(exc)
    lv = list()
    for s in symbols["variables"]:
        if not s in calibration:
            lv.append(s)
    if bool(lv):
        cprint("\nInitial values of endogenous variables:\n {} \nare not set in a model file!\n\nPlease set these variables either by passing a csv file with historical data\nto importModel() function or by calling setStartingValues() method.".format(",".join(lv)),"red")
    for s in calibration.keys():
        val = str(calibration[s])
        try:
            ast.parse(val)
        except SyntaxError as e:
            exc = ModelException("Syntax error raised while parsing expression: " + val)
            if hasattr(calibration,'lc'):
                pos = calibration.lc.data[s]
                exc.pos = [pos[0], pos[1] + e.offset, pos[0], pos[1] + e.offset]
            exceptions.append(exc)
    return exceptions


def check_all(data):
    """
    Check correctness of model file.
    
    Parameters:
        :param data: Model content.
        :type data: Dictionary.
        :returns: List of exceptions if any.
    """
    def serious(exsc): return ('error' in [e.type for e in exsc])

    exceptions = check_infos(data)
    if serious(exceptions):
        return exceptions
    
    exceptions = check_symbols(data)
    if serious(exceptions):
        return exceptions
    
    if not "bellman_equation" in data:
        exceptions += check_extra_symbols(data)
        if serious(exceptions):
            return exceptions
        exceptions += check_variables(data)
        if serious(exceptions):
            return exceptions
                        
    exceptions += check_definitions(data)
    if serious(exceptions):
        return exceptions
    
    exceptions += check_equations(data)
    if serious(exceptions):
        return exceptions
    
    exceptions += check_calibration(data)
    if serious(exceptions):
        return exceptions
    
    return exceptions


def human_format(err):
    """
    Highlights exceptions represntation by red color if it is an error and by yellow color if it is warning..
    
    Parameters:
        :param err: Error.
        :type err: ModelException.
        :returns:
    """
    err_type = err['type']
    err_type = colored(err_type, color=(
        'red' if err_type == 'error' else 'yellow'))
    if hasattr(err,'range'):
        err_range = str([e + 1 for e in err['range'][0]])[1:-1]
        return '{:7}: {:6}: {}'.format(
            err_type,
            err_range,
            err['text']
        )
    else:
        return '{:7}: {}'.format(
            err_type,
            err['text']
        )
        


def check_infos(data):
    """
    Check model info.
    
    Parameters:
        :param data: Model info.
        :type data: Dictionary.
        :returns: List of exceptions if any.
    """
    exceptions = []
    if 'name' not in data:
        exc = ModelException("Missing field: 'name'.")
        exc.pos = (0, 0, 0, 0)
        exc.type = 'warning'
        exceptions.append(exc)
    return exceptions


def lint(txt, source='<string>', format='human'):
    """
    Convert model file text to Python objects. Check syntax of model file for any errors.
    
    Parameters:
        :param txt: Model file content.
        :type txt: str.
        :param source: Source type.
        :type source: str.
        :param format: Format of exceptions display.
        :type format: str.
        :returns: Exceptions if any, otherwise a ruamel.yaml object.
    """
    # raise ModelException if it doesn't work correctly
    try:
        if version >= '0.18':
            data = ry.YAML(typ='rt').load(txt)
        else:
            data = ry.load(txt, ry.RoundTripLoader)
    except Exception as e:
        cprint(e,'red')
        return []  # should return parse error

    if not ('symbols' in data or 'equations' in data or 'calibration' in data):
        # this is probably not a yaml filename
        output = []
    else:
        try:
            exceptions = check_all(data)
        except Exception as e:
            # raise(e)
            exc = ModelException("Linter Error: Uncaught Exception - {}".format(e))
            exc.pos = [0, 0, 0, 0]
            exc.type = 'error'
            exceptions = [exc]

        output = []
        for k in exceptions:
            try:
                err_type = k.type
            except:
                err_type = 'error'
            if hasattr(k,'pos'): 
                output.append({
                    'type': err_type,
                    'source': source,
                    'range': ((k.pos[0], k.pos[1]), (k.pos[2], k.pos[3])),
                    'text': k.args[0]
                })
            else:
                output.append({
                    'type': err_type,
                    'source': source,
                    'text': k.args[0]
                })

    if format == 'json':
        return (json.dumps(output))
    elif format == 'human':
        return (str.join("\n", [human_format(e) for e in output]))
    elif not format:
        return output
    else:
        raise ModelException("Unkown format {}.".format(format))

