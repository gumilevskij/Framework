"""
Classes define user functions.

Created on Thu Mar 11 22:08:13 2021

@author: A.Goumilevski
"""
import os
import sympy as sp
#from numpy import abs as Abs,sign,exp
from sympy import Function,simplify,Abs,sign,exp,erf
from sympy import Ge,Gt,Le,Lt,Eq,N,Heaviside,Min,Max
from sympy.abc import x,y,z


# Positive = sp.lambdify(x, sign(x)+sign(Abs(x))) 
# Negative = sp.lambdify(x, -sign(x)+sign(Abs(x))) 

class Positive(Function):
    """Evaluate if expression is positive.""" 
    
    @classmethod
    def eval(cls,x):
        """
        Evaluate expression `x`.

        Parameters:
            cls : `Function` object
                Instance of `Function` class.
            x : float
                Expression value.

        Returns:
            f : float
                Returns one if x is non-negative and zero otherwise.
        """
        cls.x = x
        if sign(x)+sign(Abs(x)):
            f = 1
        else:
            f = 0
        return f
    
    @classmethod
    def fdiff(cls,argindex):
        """
        Compute drivative of `a` or `b` expression.

        Parameters:
            cls : `Function` object
                Instance of `Function` class.
            argindex : int
                Indexes of the args, starting at 1

        Returns:
            diff : Expression
                Returns zero.
        """
        return 0
    
class Negative(Function):
    """Evaluate if expression is negative.""" 
    
    @classmethod
    def eval(cls,x):
        """
        Evaluate expression `x`.

        Parameters:
            cls : `Function` object
                Instance of `Function` class.
            x : float
                Expression value.

        Returns:
            f : float
                Returns one if x is negative and zero otherwise.

        """
        if sign(x)+sign(Abs(x)):
            f = 0
        else:
            f = 1
        return f
    
    @classmethod
    def fdiff(cls,argindex):
        """
        Compute drivative of `a` or `b` expression.

        Parameters:
            cls : `Function` object
                Instance of `Function` class.
            argindex : int
                Indexes of the args, starting at 1

        Returns:
            diff : Expression
                   Returns zero.
        """
        return 0
    
    
class PNORM(Function):
    """Evaluate Troll's normal function.""" 
    
    @classmethod
    def eval(cls,x):
        """
        Return value based on condition evaluation.

        Parameters:
            cls : `Function` object
                Instance of `Function` class.
            condition : float
                Expression value.
            x : float
                Expression value.

        Returns:
            f : float
                Returns `x` if `condition` is non-negative and zero otherwise.

        """
        cls.x = x
        f = exp(-0.5*x*x)
        return f
    
    @classmethod
    def fdiff(cls,z):
        """
        Compute drivative of `z` expression.

        Parameters:
            cls : `Function` object
                Instance of `Function` class.
            z : symbol
                Take derivative w.r.t. symbol `z`

        Returns:
            diff : Expression
                Derivative w.r.t `z`.

        """
        diff = -z*exp(-0.5*z*z)
        return diff
    
    
class IfThen(Function):
    """Evaluate codition and return value based on this condition.""" 
    
    @classmethod
    def eval(cls,condition,x):
        """
        Return value based on condition evaluation.

        Parameters:
            cls : `Function` object
                Instance of `Function` class.
            condition : float
                Expression value.
            x : float
                Expression value.

        Returns:
            f : float
                Returns `x` if `condition` is non-negative and zero otherwise.

        """
        cls.condition = condition
        cls.x = x
        if Positive(condition):
            f = x
        else:
            f = 0
        return f
      
    @classmethod
    def fdiff(cls,argindex):
        """
        Compute drivative of `x` expression.

        Parameters:
            cls : `Function` object
                Instance of `Function` class.
            argindex : int
                Indexes of the args, starting at 1

        Returns:
            diff : Expression
                Derivative w.r.t `x` if `condition` is non-negative and zero otherwise.
        """
        if Positive(cls.condition):
            diff = sp.diff(cls,cls.x)
        else:
            diff = 0
        return diff
    
    
class IfThenElse(Function):    
    """Evaluate codition and return value based on this condition.""" 
    
    @classmethod
    def eval(cls,condition,a,b):
        """
        Return value based on condition evaluation.

        Parameters:
            cls : `Function` object
                Instance of `Function` class.
            condition : float
                Expression value.
            a : float
                Expression value.
            b : float
                Expression value.

        Returns:
            f : float
                Returns `a` if `condition` is non-negative and `b` otherwise.

        """
        cls.condition = condition
        cls.a = a
        cls.b = b
        if Positive(condition):
            return a
        else:
            return b
        
    @classmethod
    def fdiff(cls,argindex):
        """
        Compute drivative of `a` or `b` expression.

        Parameters:
            cls : `Function` object
                Instance of `Function` class.
            argindex : int
                Indexes of the args, starting at 1

        Returns:
            diff : Expression
                Derivative w.r.t `a` if `condition` is non-negative and derivative w.r.t `b`  otherwise.
        """
        if argindex == 1:
            diff = 0
        elif Positive(condition):
            if argindex == 2:
                diff = sp.diff(cls,cls.a)
            elif argindex == 3:
                diff = sp.diff(cls,cls.b)
        else:
            if argindex == 2:
                diff = sp.diff(cls,cls.b)
            elif argindex == 3:
                diff = sp.diff(cls,cls.a)
            
        return diff
    

class myzif(Function):

    @classmethod
    def eval(cls,x):
        """
        Return value based on user's function evaluation.

        Parameters:
            cls : `Function` object
                Instance of `Function` class.
            x : float
                Expression value.
        """
        y = x*(1+erf(3.*x))/2.
        return y


class Min1(Function):

    @classmethod
    def eval(cls,a,b):
        """
        Return minimum of two values.

        Parameters:
            cls : `Function` object
                Instance of `Function` class.
            a : float
                Expression value.
            b : float
                Expression value.
        """
        if a <= b:
            y = a
        else:
            y = b
        return y


class Max1(Function):

    @classmethod
    def eval(cls,a,b):
        """
        Return minimum of two values.

        Parameters:
            cls : `Function` object
                Instance of `Function` class.
            a : float
                Expression value.
            b : float
                Expression value.
        """
        if a >= b:
            y = a
        else:
            y = b
        return y
    
    
def updateFiles(model,path):
    """
    Update python generated files.

    Parameters:
        model : Model
            Model object.
        path : str
            Path to folder.
    """
    try:
        src_dynamic = model.functions_src['f_dynamic_src']
        src_sparse = model.functions_src['f_sparse_src']
        src_ss = model.functions_src['f_steady_src']
        src_jacob = model.functions_src['f_jacob_src']
        src_rhs = model.functions_src['f_rhs_src']
        
        file_path = os.path.abspath(os.path.join(path,'f_dynamic.py'))
        with open(file_path, 'w') as f: 
            f.write(src_dynamic)
        
        if not src_sparse is None:
            file_path = os.path.abspath(os.path.join(path,'f_sparse.py'))
            with open(file_path, 'w') as f: 
                f.write(src_sparse)
        
        file_path = os.path.abspath(os.path.join(path,'f_steady.py'))
        with open(file_path, 'w') as f: 
            f.write(src_ss)
        
        file_path = os.path.abspath(os.path.join(path,'f_jacob.py'))
        with open(file_path, 'w') as f: 
            f.write(src_jacob)
            
        file_path = os.path.abspath(os.path.join(path,'f_rhs.py'))
        with open(file_path, 'w') as f: 
            f.write(src_rhs)
    except:
        pass
        
        
if __name__ == "__main__":
    "Test user functions."
    z = -1
    
    # f = PNORM
    # s=f(x)
    # print(s)
    # s=sp.diff(f(x),x)
    # print(s)
    
    # f = sp.lambdify(x, x>0)
    # s = f(z)
    # print(s)
    
    # f = Positive
    # s = f(z)
    # print(s)
    # s = sp.diff(f(x),x)
    # print(s)
    # print()   
    
    # f = Negative
    # s = f(z)
    # print(s)
    # s = sp.diff(f(x),x)
    # print(s)
    # print()
    
    # f = IfThen
    # s = f(z,x**2*(x+y))
    # print(s)
    # s = sp.diff(f(z,x**2*(x+y)),x)
    # print(s)
    # s = sp.diff(f(z,x**2*(x+y)),y)
    # print(s)
    # print()
    
    # f = IfThenElse
    # s=f(z,x**2*(x+y),y**3)
    # print(s)
    # s=sp.diff(f(z,x**2*(x+y),y**3),x)
    # print(s)
    # s=sp.diff(f(z,x**2*(x+y),y**3),y)
    # print(s)

    # ns = {"IfThenElse":IfThenElse,"Positive":Positive,"z":z}
    # e = "IfThenElse(z,x**2*(x+y),y**3)"
    # eq = sp.sympify(e,locals=ns)
    # s = eq.diff(x)
    # print(s)
    
    # f = myzif 
    # s=f(x)
    # print(s)
    # s=sp.diff(f(x),x)
    # print(s)
    
    # eq = Min(x**2*(x+y),y**3)
    # print(eq.diff(y))
    # e = "Min(x**2*(x+y),y**3)"
    # eq = sp.sympify(e)
    # s = eq.diff(x)
    # print(s)
    # print(Min(1,2))
    
    ns={"T":x,"T_rule":y,"T_ceiling":z}
    e = "T-Min(T_rule, T*T_ceiling)"
    #eq = sp.sympify(e)
    eq = sp.sympify(e,locals=ns)
    print(eq.diff(x))
    
