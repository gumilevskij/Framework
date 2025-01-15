"""
Defines structure of LanguageElement class. 
"""
import math
import ruamel.yaml as yaml
from sympy import Max,Min,Heaviside,Abs,sign
from snowdrop.src.preprocessor.util import IfThen, IfThenElse, Positive, Negative
#from snowdrop.src.preprocessor.util import PNORM
from snowdrop.src.preprocessor import objects

functions = {
    'abs':        abs,
    'log':        math.log,
    'exp':        math.exp,
    'sin':        math.sin,
    'cos':        math.cos,
    'tan':        math.tan,
    'Max':        Max,
    'Min':        Min,
    'sign':       sign,
    'Heaviside':  Heaviside,
    'Abs':        Abs,
    'IfThen':     IfThen,
    'IfThenElse': IfThenElse,
    'Positive':   Positive,
    'Negative':   Negative
#    'PNORM':      PNORM
}

constants = { 'pi': math.pi }

class LanguageElement(dict):
    """
    Class LanguageElement.
    
    .. currentmodule: preprocessor
    """
    
    @classmethod
    def constructor(cls, loader, node):
        """Constructor."""
        value = loader.construct_mapping(node)
        return cls(**value)

    def check(self):
        """Inspect call signature of a callable object and its return annotation."""
        import inspect
        sig = inspect.signature(self.baseclass)
        sig.bind_partial(self)

    def eval(self, d={}):
        """Evaluate Numeric node."""
        from snowdrop.src.preprocessor.symbolic_eval import NumericEval
        ne = NumericEval(d)
        args = ne.eval_dict(self)
        obj = self.baseclass(**args)
        return obj

    def __repr__(self):
        """Return object representation."""
        s = super().__repr__()
        n = self.baseclass.__name__
        return "{}(**{})".format(n, s)

    def __str__(self):
        """Human readable object representation."""
        n = self.baseclass.__name__
        c = str.join(", ", ["{}={}".format(k, v) for k, v in self.items()])
        return "{}({})".format(n, c)

    
class Domain(LanguageElement):
    """Domain class."""
    baseclass = objects.Domain
  
class Normal(LanguageElement):
    """Normal distribution class."""
    baseclass = objects.Normal
    
class MvNormal(LanguageElement):
    """Multivariate normal distribution class."""
    baseclass = objects.MvNormal
    
class LogNormal(LanguageElement):
    """Lognormal distribution class."""
    baseclass = objects.LogNormal
    
class Beta(LanguageElement):
    """Beta distribution class."""
    baseclass = objects.Beta
    
class Binomial(LanguageElement):
    """Binomial distribution class."""
    baseclass = objects.Binomial
    
class Gamma(LanguageElement):
    """Gamma distribution class."""
    baseclass = objects.Gamma
    
class Logistic(LanguageElement):
    """Logistic distribution class."""
    baseclass = objects.Logistic
    
class Uniform(LanguageElement):
    """Uniform distribution class."""
    baseclass = objects.Uniform
    
class Cartesian(LanguageElement):
    """Cartesian grid class."""
    baseclass = objects.Cartesian

minilang = [Normal,MvNormal,LogNormal,Beta,Binomial,Gamma,Logistic,Uniform,Domain,Cartesian]

types=[]
for c in minilang:
    t = c.__name__
    types.append(t)
    

if __name__ == '__main__':
    """Entry point."""
    
    for C in minilang:
         k = C.__name__
         name = '!{}'.format(k)
         yaml.add_constructor(name, C.constructor)
 
    txt = """
    distribution: !MvNormal
        Sigma: [[0.3]]
        mu: [0.1]
    """
    data = yaml.load(txt, Loader=yaml.Loader)
    dis = data['distribution']
    print(dis)
      
    # txt = """
    # distribution: !Normal
    #     mean: 0.0
    #     scale: 0.1
    # """
    # data = yaml.load(txt, Loader=yaml.Loader)
    # dis = data['distribution']
    # print(dis)

    # dis.__repr__()
     
    # from snowdrop.src.preprocessor.symbolic_eval import NumericEval
    # txt = """
    # grid: !Cartesian
    # a: [x,10]
    # b: [y,20]
    # orders: [50,50]
    # """

    # data = yaml.load(txt, Loader=yaml.Loader)
    # grid = data['grid']

    # d = dict(x=20, y=30, sig_z=0.001)
    # ne = NumericEval(d, minilang=minilang)
    # ne.eval(d)
    
    # cart = grid.eval(d)
    # dd = dis.eval(d)
    # ne.eval(data['grid'])
    # ndata = ne.eval(data)
    # data['grid']
    # ndata['grid']
