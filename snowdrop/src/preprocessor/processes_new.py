import numpy as np
from numpy.random import default_rng
#from numpy.random import multivariate_normal
#from numpy.random import uniform,normal,lognormal,beta,binomial,gamma,logistic
#from numpy.random import chisquare,laplace,poisson,pareto,wald,weibull

rng = default_rng()

### Multivariate distributions
class MvNormal:
    """Multivariate normal distribution class."""
    def __init__(self, mean=[0], cov=[[1.0]]):
        """
        Constructor for `MvNormal` normal class.

        Parameters:
            mean : 1-D array_like, optional
                Mean values. The default is [0].
            cov : 2-D array_like, optional
                Covariance matrix of the distribution. The default is [[1.0]].
        """
        self.mean = np.array(mean, dtype=float)
        self.cov = np.atleast_2d( np.array(cov, dtype=float) )

    def simulate(self, N, T):
        """
        Return a sample draw from multivariate normal distribution.

        Parameters:
            N : int
                Sample size.
            T : int
                Time.

        Returns:
            ndarray or scalar
                Drawn samples from the parameterized multivariate normal distribution.
        """
        mean = self.mean
        cov = self.cov
        result = rng.MvNormal(mean=mean,cov=cov,size=N*T)
        return  result.reshape((T,N,len(mean)))


### Univariate distributions
class Normal:
    """Normal distribution class."""
    def __init__(self, loc=0, scale=1.0):
        """
        Constructor for `Normal` class.

        Parameters:
            loc : float or array_like of floats, optional
                Mean of the distribution. The default is 0.
            scale : float or array_like of floats, optional
                Standard deviation of the distribution. The default is 1.

        """
        self.loc = loc
        self.scale = scale

    def simulate(self, T):
        """
        Return a sample draw from normal distribution.

        Parameters:
            T : int
                Time.

        Returns:
             result : ndarray or scalar
                Drawn samples from the parameterized normal distribution.

        """
        loc = self.loc
        scale = self.scale
        result = rng.Normal(loc=loc,scale=scale,size=T)
        return  result
    
    
class LogNormal:
    """ Univariate LogNormal distribution class."""
    def __init__(self, Mu=0.0, Sigma=1.0):
        """
        Constructor for `LogNormal` class.

        Parameters:
            Mu : float or array_like of floats, optional
                Mean value of the underlying normal distribution. The default is 0.0.
            Sigma : float or array_like of floats, optional
                Standard deviation. The default is 1.0.

        """
        self.Sigma = Sigma
        self.Mu = Mu

    def simulate(self, T):
        """
        Return a sample draw from log-normal distribution.

        Parameters:
            T : int
                Time.

        Returns:
             result : ndarray or scalar
                 Drawn samples from the parameterized log-normal distribution..

        """
        Sigma = self.Sigma
        Mu = self.Mu
        result = rng.LogNormal(mean=Mu,sigma=Sigma,size=T)
        return  result
    
    
class Beta:
    """Univariate Beta distribution class."""
    def __init__(self, a, b):
        """
        Constructor for `Beta` class.

        Parameters:
            a : float or array_like of floats.
                Alpha, positive (>0).
            b : float or array_like of floats.
                Beta, positive (>0).
`
        """
        self.a = a
        self.b = b

    def simulate(self, T):
        """
        Return a sample draw from beta distribution.

        Parameters:
            T : int
                Time.

        Returns:
             result : ndarray or scalar
                Drawn samples from the parameterized beta distribution.

        """
        a = self.a
        b = self.b
        result = rng.Beta(a=a,b=b,size=T)
        return  result
    
    
class Binomial:
    """Binomial distribution class."""
    def __init__(self, n, p):
        """
        Constructor for `Binomial` class.

        Parameters:
            n : int or array_like of ints.
                Parameter of the distribution, non-negative.
            p : float or array_like of floats.
                Parameter of the distribution, 0 <= p <= 1.

        """
        self.n = n
        self.p = p

    def simulate(self, T):
        """
        Return a sample draw from binomial distribution.

        Parameters:
            T : int
                Time.

        Returns:
             result : ndarray or scalar
                Drawn samples from the parameterized binomial distribution.

        """
        n = self.n
        p = self.p
        result = rng.Binomial(n=n,p=p,size=T)
        return  result
    
    
class Gamma:
    """ Univariate Gamma distribution class."""
    def __init__(self, shape, scale=1.0):
        """
        Constructor for  `Gamma` class.

        Parameters:
            shape : float or array_like of floats.
                The shape of the gamma distribution.
            scale : float or array_like of floats, optional.
                The scale of the gamma distribution. Must be non-negative. The default is 1.0.
        
        """
        self.shape = shape
        self.scale = scale

    def simulate(self, T):
        """
        Return a sample draw from gamma distribution.

        Parameters:
            T : int
                Time.

        Returns:
             result : ndarray or scalar
                Drawn samples from the parameterized gamma distribution.

        """
        shape = self.shape
        scale = self.scale
        result = rng.Gamma(shape=shape,scale=scale,size=T)
        return  result
          
    
class Logistic:
    """Univariate Logistic distribution class."""
    def __init__(self, loc=0.0, scale=1.0):
        """
        Constructor for `Logistic` class.

        Parameters:
            loc : float or array_like of floats, optional.
                Parameter of the distribution. The default is 0.
            scale : loat or array_like of floats, optional.
                Parameter of the distribution. Must be non-negative. The default is 1.
            
        Returns:
            None.

        """
        self.loc = loc
        self.scale = scale

    def simulate(self, T):
        """
        Return a sample draw from logistic distribution.

        Parameters:
            T : int
                Time.

        Returns:
             result : ndarray or scalar
                Drawn samples from the parameterized logistics distribution.

        """
        loc = self.loc
        scale = self.scale
        result = rng.Logistic(loc=loc,scale=scale,size=T)
        return  result
                
    
class Uniform:
    """Univariate Uniform distribution class."""
    def __init__(self, low=0, high=1):
        """
        Constructor for `Uniform` class.

        Parameters:
            low : float or array_like of floats, optional.
                Lower boundary of the output interval. The default is 0.
            high : float or array_like of floats, optional.
                Upper boundary of the output interval. . The default is 1.

        """
        self.low = low
        self.high = high

    def simulate(self, T):
        """
        Return a sample draw from uniform distribution.

        Parameters:
            T : int
                Time.

        Returns:
             result : ndarray or scalar
                Drawn samples from the parameterized uniform distribution.

        """
        low = self.low
        high = self.high
        result = rng.Uniform(low=low,high=high,size=T)
        return  result
