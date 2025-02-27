import ast

class Compare:

    def __init__(self):
        self.d = {}

    def compare(self, A, B):
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


def compare_strings(a,b):
    t1 = ast.parse(a)
    t2 = ast.parse(b)
    comp = Compare()
    val = comp.compare(t1,t2)
    d = comp.d
    return val

def match(m,s):
    if isinstance(m,str):
        m = ast.parse(m).body[0].value
    if isinstance(s,str):
        s = ast.parse(s).body[0].value
    comp = Compare()
    val = comp.compare(m,s)
    d = comp.d
    if len(d) == 0:
        return val
    else:
        return d

import ast
class ReplaceExpectation(ast.NodeTransformer):
    import ast
    def visit_Subscript(self, node):
        from ast import Expr, Call, Name, Load, Lambda, arguments, arg
        m = match('E[ _expr | _x in _set]', node)
        if m:
            x_s = m['_x'].id # name of dummy vaqriable
            expr = m['_expr']
            sset = m['_set']
            res = Call(func=Name(id='Ex', ctx=Load()),
                    args=[Lambda(args=arguments(args=[arg(arg=x_s, annotation=None)],
                    vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]), body=expr), sset],
                    keywords=[], starargs=None, kwargs=None)
            return res
        m = match('Sum[ _expr | _x in _set]', node)
        if m:
            x_s = m['_x'].id # name of dummy vaqriable
            expr = m['_expr']
            sset = m['_set']
            res = Call(func=Name(id='Sum', ctx=Load()),
                    args=[Lambda(args=arguments(args=[arg(arg=x_s, annotation=None)],
                    vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]), body=expr), sset],
                    keywords=[], starargs=None, kwargs=None)
            return res

        return node


def test_strings():

    tests = [
    ['b(_x)','b(1)', True],
    ['_x(1)','b(1)', True],
    ['_x(2)','b(1)', False],
    ['a + b + c', 'a + b + c', True],
    ['a + b + d', 'a + b + c', False],
    ['a + b + _d','a + b + c',  True],
    ['a + _d + _d','a + b + c', False],
    ['a + _d + _d','a + c + c', True],
    ['a*_x+c*_x**2+d*_y',  'a*x+c*x**2+d*y', True],
    ['a*_x+c*_x**2+d*_y',  'a*x+c*y**2+d*z', False],
    ['-_a(_x) + _b(_x)',  '-u(-1)+v(-1)', True],
    ['-_a(_x) + _b(_x)',  '-u(11)+v(11)', True],
    ['-_a(_x) + _b(_x)',  '-u(1)+v(2)', False],
    ['E[_x]',  'E[1]', True],
    ['E[2]',  'E[1]', False],
    ['E[1]',  'X[1]', False],
    ['a in _x',  'a in b', True],
    ['a in b',  'a in c', False],
    ['a | _x',  'a | c', True],
    ['a | b',  'a | c', False],
    ['E[ _x | _y in _z]',  'E[ a | b in c]', True],
    ['E[ a | b in c]',  'E[ a | b in d]', False],
    ['E[ _x | _y in _z]',  'E[ f(a) | b in c]', True],
    ['E[ _x | _y in _z]',  'E[ f(a)^2 | b in c]', True],
    # ['E[ f(s) ]',  'E[ _x ]', True],
    ['_a == _b | _x <= _y <= _z', 'a == b | x <= y <= z', True],
    ['_eq | _comp','phi[s] + (-(f[s]-min_f))*1000    |  phi[s]', True],
    ['_eq | _comp','phi[s] + (-(f[s]-min_f))*1000    | 0 <= phi[s]', True],
    ['_eq | 0 <= _x','phi[s] + (-(f[s]-min_f))*1000  | 0 <= phi[s]', True],
    # ['_cond : _eq | _comp' ,  's>0 : a == 8 | x<=0<=y', True],
    ]
    replacement_rules = [
        ['E[ expr | x in S(s)]', 'Esp(lambda x: expr, S(s))'],
        ['Sum[ expr | x in H(s)]', 'Sum(lambda x: expr, H(s)']
    ]

    for l in tests:
        print('{} : {} : {} : {}'.format(compare_strings(l[0], l[1]), l[2], l[0],l[1]))
        # assert(compare_strings(l[0], l[1]) ==  l[2])


def fix_equation(eq):
    eqq = ReplaceExpectation().visit(eq)
    return eqq
#
#
# if __name__ == "__main__":
#     test_strings()
#     res = match("E[ _i | _v in _s]", "E[ y(1)*2 | y(1) in S(s)]")
#     res = match('_a == _b | _x <= _y <= _z', 'a == b | x <= y <= z')
#     if match("a_","b"):
#         print("yes")
#     if match("a","a"):
#         print("yes")
#     else:
#         print("no")
