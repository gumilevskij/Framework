import numpy as np

from numpy.random import multivariate_normal
from numpy.random import uniform,normal,lognormal,beta,binomial,gamma,logistic
#from numpy.random import chisquare,laplace,poisson,pareto,wald,weibull

### Multivariate distributions
class MvNormal:
    """Multivariate normal distribution class."""

    def __init__(self, mean=[0], cov=[[1.0]]):
        self.mean = np.array(mean, dtype=float)
        self.cov = np.atleast_2d( np.array(cov, dtype=float) )

    def simulate(self, N, T):
        mean = self.mean
        cov = self.cov
        sim = multivariate_normal(mean=mean,cov=cov,size=N*T)
        return sim.reshape((T,N,len(mean)))


### Univariate distributions
class Normal:
    """Univariate Normal distribution class."""

    def __init__(self, loc=0, scale=1.0):
        self.loc = loc
        self.scale = scale

    def simulate(self, T):
        loc = self.loc
        scale = self.scale
        sim = normal(loc=loc,scale=scale,size=T)
        return sim
    
    
class LogNormal:
    """ Univariate LogNormal distribution class."""

    def __init__(self, Mu=0.0, Sigma=1.0):
        self.Sigma = Sigma
        self.Mu = Mu

    def simulate(self, T):
        Sigma = self.Sigma
        Mu = self.Mu
        sim = lognormal(mean=Mu,sigma=Sigma,size=T)
        return sim
    
    
class Beta:
    """Univariate Beta distribution class."""

    def __init__(self, a, b):
        self.a = a
        self.b = b

    def simulate(self, T):
        a = self.a
        b = self.b
        sim = beta(a=a,b=b,size=T)
        return sim
    
    
class Binomial:
    """Univariate Binomial distribution class."""

    def __init__(self, n, p):
        self.n = n
        self.p = p

    def simulate(self, T):
        n = self.n
        p = self.p
        sim = binomial(n=n,p=p,size=T)
        return sim
    
    
class Gamma:
    """ Univariate Gamma distribution class."""

    def __init__(self, shape, scale=1.0):
        self.shape = shape
        self.scale = scale

    def simulate(self, T):
        shape = self.shape
        scale = self.scale
        sim = gamma(shape=shape,scale=scale,size=T)
        return sim
          
class Logistic:
    """Univariate Logistic distribution class."""

    def __init__(self, loc=0.0, scale=1.0):
        self.loc = loc
        self.scale = scale

    def simulate(self, T):
        loc = self.loc
        scale = self.scale
        sim = logistic(loc=loc,scale=scale,size=T)
        return sim
                
    
class Uniform:
    """Univariate Uniform distribution class."""

    def __init__(self, low=0, high=1):
        self.low = low
        self.high = high

    def simulate(self, T):
        low = self.low
        high = self.high
        sim = uniform(low=low,high=high,size=T)
        return sim

