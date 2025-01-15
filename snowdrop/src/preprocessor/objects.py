"""Wrapper for probability distribution objects."""

import numpy as np
from snowdrop.src.preprocessor.processes import Normal,MvNormal,LogNormal,Beta
from snowdrop.src.preprocessor.processes import Binomial,Gamma,Logistic,Uniform
from snowdrop.src.preprocessor.processes import Cartesian

MvNormal = MvNormal
Normal = Normal
LogNormal = LogNormal
Beta = Beta
Binomial = Binomial
Gamma = Gamma
Logistic = Logistic
Uniform = Uniform
Cartesian = Cartesian


class Domain(dict):
    """Domain class."""

    def __init__(self, **kwargs):
        super().__init__()
        for k, w in kwargs.items():
            v = kwargs[k]
            self[k] = np.array(v, dtype=float)

    @property
    def min(self):
        return np.array([self[e][0] for e in self.states])

    @property
    def max(self):
        return np.array([self[e][1] for e in self.states])


if __name__ == '__main__':
    """Main entry point."""
    normal = MvNormal(mean=[1.5,-2.0],cov=[[0.3,-0.1],[0.05,0.01]])
    results = normal.simulate(2,10)
    print(results)
    
    normal = Normal(loc=-2.0,scale=0.3)
    results = normal.simulate(10)
    print(results)
