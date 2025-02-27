#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  2 19:43:41 2021

@author: alexei
"""
import sys
import numpy as np
from scipy import stats
from scipy.special import betaln
from scipy.special import gammaln
from particles import distributions
from snowdrop.src.misc.termcolor import cprint

    
class StructDist(distributions.StructDist):
    """Distribution with structured arrays as inputs and outputs."""
    def logpdf(self, theta):
        l = 0.
        for par, law in self.laws.items():
            cond_law = law(theta) if callable(law) else law
            l += cond_law.logpdf(theta[par])
        return l
       

def pdf(distr,x,params):
    """
    Return probability density function for a set of distributions.

    Parameters:
        distr : str.
            Shape of distribution.
        x : float.
            Value of variable.
        params : list.
            Parameters of distribution.

    Returns:
        Estimation of probability density function.

    """
    b = True
    likelihood = 0
    lower_bound = params[1]
    upper_bound = params[2]
    
    if distr.endswith("_hp"):
        distr = distr[:-3]
    
    if x < lower_bound or x > upper_bound:
        likelihood = 1.e-20
        
    elif distr == 'uniform_pdf':
        #likelihood = stats.uniform.pdf(x=x,loc=lower_bound,scale=lower_bound+upper_bound)
        likelihood = 1./(params[4]-params[3])
        
    elif distr == 'normal_pdf':
        likelihood = stats.norm.pdf(x=x,loc=params[3],scale=params[4])
    
    elif distr == 'lognormal_pdf':
        if x <= 0:
            b = False
            likelihood = 1.e-20
        else:
            #likelihood = stats.lognorm.pdf(x=x-lower_bound,s=params[3],loc=params[2],scale=1)
            likelihood = 1./((x-lower_bound)*params[4]*np.sqrt(2*np.pi)) * np.exp(-0.5*((np.log(x-lower_bound)-params[3])/params[4])**2)
    
    elif distr == 'gamma_pdf':
        if x < 0 or params[3] <= 0:
            b = False
            likelihood = 1.e-20
        else:
            #likelihood = stats.gamma.pdf(x=x-lower_bound,a=params[3],scale=params[4])
            ldens= -gammaln(params[3])-params[3]*np.log(params[4])+(params[3]-1)*np.log(x-lower_bound)-(x-lower_bound)/params[4]
            likelihood = np.exp(ldens)
            
    elif distr == 'logit_pdf':
        likelihood = stats.logistic.pdf(x=x)
        
    elif distr == 'inv_gamma_pdf':
        if x < 0 or params[3] <= 0:
            b = False
            likelihood = 1.e-20
        else:
            likelihood = inverse_gamma_pdf(x=x-lower_bound,a=params[3],b=params[4])
 
    elif distr == 'inv_gamma1_pdf':
        # Evaluates the logged INVERSE-GAMMA-1 PDF at x.
        # X ~ IG1(s,nu) if X = sqrt(Y) where Y ~ IG2(s,nu) and Y = inv(Z) with Z ~ G(nu/2,2/s) (Gamma distribution)
        # See L. Bauwens, M. Lubrano and J-F. Richard [1999, appendix A] for more details.
        if x < 0:
            b = False
            likelihood = 1.e-20
        else:
            ldens = np.log(2)-gammaln(0.5*params[4])-0.5*params[4]*(np.log(2)-np.log(params[3]))-(params[4]+1)*np.log(x-lower_bound)-0.5*params[3]/(x-lower_bound)**2
            likelihood = np.exp(ldens)

    elif distr == 'inv_gamma2_pdf':
        # Evaluates the logged INVERSE-GAMMA-2 PDF at x.
        # X ~ IG2(s,nu) if X = inv(Z) where Z ~ G(nu/2,2/s) (Gamma distribution)
        # See L. Bauwens, M. Lubrano and J-F. Richard [1999, appendix A] for more details.
        if x < 0:
            b = False
            likelihood = 1.e-20
        else:
            ldens = -gammaln(0.5*params[4])-(0.5*params[4])*(np.log(2)-np.log(params[3]))-0.5*(params[4]+2)*np.log(x-lower_bound)-0.5*params[3]/(x-lower_bound)
            likelihood = np.exp(ldens)
            
    elif distr == 'beta_pdf':
        if x < 0:
            b = False
            likelihood = 1.e-20
        else:
            #likelihood = stats.beta.pdf(x=x-lower_bound,a=params[3],b=params[4])
            ldens = -betaln(params[3],params[4])+(params[3]-1)*np.log(x-lower_bound)+(params[4]-1)*np.log(upper_bound-x)-(params[3]+params[4]-1)*np.log(upper_bound-lower_bound)
            likelihood = np.exp(ldens)
            
    elif distr == 't_pdf':
        if params[3] < 0:
            b = False
            likelihood = 1.e-20
        else:
            likelihood = stats.t.pdf(x=x,df=params[3],scale=params[4])
    
    elif distr == 'weibull_pdf':
        if x < 0 or params[3] <= 0 or params[4] <= 0:
            b = False
            likelihood = 1.e-20
        else:
            #likelihood = stats.weibull_min.pdf(x=x/params[3],c=params[4]) / params[3]
            x0 = x-params[5] if len(params) > 5 else x
            x1 = x/params[4]
            x2 = x1**params[3]
            ldens = np.log(params[3])-params[3]*np.log(params[4])+(params[3]-1)*np.log(x0)-x2
            likelihood = np.exp(ldens)

    elif distr == 'inv_weibull_pdf':
        if x < 0 or params[3] <= 0 or params[4] <= 0:
            b = False
            likelihood = 1.e-20
        else:
            c = params[5] if len(params) > 5 else 0
            likelihood = stats.invweibull.pdf(x=(x-c)/params[3],c=params[4])
    
    elif distr == 'wishart_pdf':
        if params[3] < 0 or params[4] <= 0:
            b = False
            likelihood = 1.e-20
        else:
            likelihood = stats.wishart.pdf(x=x,df=params[3],scale=params[4])
    
    elif distr == 'inv_wishart_pdf':
        if params[3] < 0 or params[4] <= 0:
            b = False
            likelihood = 1.e-20
        else:
            likelihood = stats.invwishart.pdf(x=x,df=params[3],scale=params[4])
            
    else:
        raise Exception(f"Distribution {distr} is not implemented yet!")
        
    return likelihood,b    


def inverse_gamma_pdf( x,a,b):
    """
    Calculate inverse gamma probability density function.
        
    Returns the inverse gamma probability density
    function with shape and scale parameters a and b, 
    respectively, at the values in x.

    Parameters:
        x : float
            Input value.
        a : float
            Shape parameter.
        b : float
            Scale parameter.

    Returns:
        y : float
            Inverse gamma probability density function.
        
    """
    from scipy.special import gamma
    
    y = b**a/gamma(a)*x**(-a-1)*np.exp(-b/x)
    
    return y 


def getMcParameter(name,distr,params):
    """
    Return parameter of a distribution.

    Parameters:
        name : str.
            Name of parameter.
        distr : str.
            Shape of distribution.
        x : float.
            Value of variable.
        params : list.
            Parameters of distribution.

    Returns:
        pmc3 parameter.

    """
    import pymc3 as pm
    
    if distr == 'uniform_pdf':
        p   = pm.Uniform(name=name,lower=params[1],upper=params[2])
    elif distr == 'normal_pdf':
        p   = pm.Normal(name=name,mu=params[3],tau=1/params[4]**2)
    elif distr == 'lognormal_pdf':
        p   = pm.Lognormal(name=name,mu=params[3],tau=1/params[4]**2)
    elif distr == 'gamma_pdf':
        p   = pm.Gamma(name=name,alpha=params[3],beta=1/params[4])
    elif distr == 'inv_gamma_pdf':
        p   = pm.InverseGamma(name=name,alpha=params[3],beta=params[4])
    elif distr == 'beta_pdf':
        p   = pm.Beta(name=name,alpha=params[3],beta=params[4])
    elif distr == 't_pdf':
        p   = pm.StudentT(name=name,mu=params[3],nu=params[4])
    elif distr == 'weibull_pdf':
        p   = pm.Weibull(name=name,alpha=params[3],beta=params[4])
    elif distr == 'wishart_pdf':
        p   = pm.Wishart(name=name,nu=params[3],V=params[4])
    else:
        raise Exception(f"Distribution {distr} is not implemented yet!")
 
    return p   

    
def getParticleParameter(distr,params):
    """
    Return parameter of a distribution.

    Parameters:
        name : str.
            Name of parameter.
        distr : str.
            Shape of distribution.
        x : float.
            Value of variable.
        params : list.
            Parameters of distribution.

    Returns:
        Particles parameter.

    """
    import particles.distributions as dists
    
    if distr.endswith("_hp"):
        distr = distr[:-3]
        
    lower_bound = params[1]
    upper_bond  = params[2]
    
    if not distr.endswith("_hp"):
        params[3],params[4] = getHyperParameters(distr=distr,mean=params[3],std=params[4])
   
        
    if distr == 'uniform_pdf':
        p   = dists.Uniform(a=params[1],b=params[2])
    elif distr == 'normal_pdf':
        p   = dists.Normal(loc=params[3],scale=params[4])
    elif distr == 'truncated_normal_pdf':
        p = dists.TruncNormal(mu=params[3],sigma=params[4],a=params[1],b=params[2])
    elif distr == 'lognormal_pdf':
        p   = dists.LogNormal(loc=params[3],scale=params[4])
    elif distr == 'binomial_pdf':
        p   = dists.Binomial(n=params[3],p=params[4])
    elif distr == 'negative_binomial_pdf':
        p   = dists.NegativeBinomial(n=params[3],p=params[4])
    elif distr == 'gamma_pdf':
        p   = dists.Gamma(a=params[3],b=params[4])
    elif distr == 'inv_gamma_pdf':
        p   = dists.InvGamma(a=params[3],b=params[4])
    elif distr == 'beta_pdf':
        p   = dists.Beta(a=params[3],b=params[4])
    elif distr == 't_pdf':
        if len(params) < 6:
            p = dists.Student(loc=params[3],scale=params[4])
        else:
            p = dists.Student(df=params[3],loc=params[4],scale=params[5])
    elif distr == 'poisson_pdf':
        p   = dists.Poisson(rate=params[3])
    elif distr == 'logistic_pdf':
        p   = dists.Logistic(loc=params[3],scale=params[4])
    elif distr == 'laplace_pdf':
        p   = dists.Laplace(loc=params[3],scale=params[4])
    elif distr == 'inv_gamma1_pdf':
        # Evaluates the logged INVERSE-GAMMA-1 PDF at x.
        # X ~ IG1(s,nu) if X = sqrt(Y) where Y ~ IG2(s,nu) and Y = inv(Z) with Z ~ G(nu/2,2/s) (Gamma distribution)
        # See L. Bauwens, M. Lubrano and J-F. Richard [1999, appendix A] for more details.
        class InvGamma1(dists.InvGamma):
            """Inverse Gamma1(a,b) distribution."""
            def __init__(self, a=1., b=1.):
                self.a = a
                self.b = b
            def logpdf(self, x):
                ldens = np.exp(np.log(2)-gammaln(0.5*params[4])-0.5*params[4]*(np.log(2)-np.log(params[3]))-(params[4]+1)*np.log(x-lower_bound)-0.5*params[3]/(x-lower_bound)**2)
                likelihood = np.exp(ldens)
                return likelihood
        p = InvGamma1(a=params[3],b=params[4])
    elif distr == 'inv_gamma2_pdf':
        # Evaluates the logged INVERSE-GAMMA-2 PDF at x.
        # X ~ IG2(s,nu) if X = inv(Z) where Z ~ G(nu/2,2/s) (Gamma distribution)
        # See L. Bauwens, M. Lubrano and J-F. Richard [1999, appendix A] for more details.
        class InvGamma2(dists.InvGamma):
            """Inverse Gamma1(a,b) distribution."""
            def __init__(self, a=1., b=1.):
                self.a = a
                self.b = b
            def logpdf(self, x):
                ldens = -gammaln(0.5*params[4])-(0.5*params[4])*(np.log(2)-np.log(params[3]))-0.5*(params[4]+2)*np.log(x-lower_bound)-0.5*params[3]/(x-lower_bound)
                likelihood = np.exp(ldens)
                return likelihood
        p = InvGamma2(a=params[3],b=params[4])
    else:
        raise Exception(f"Distribution {distr} is not implemented yet!")
 
    return p  

def getHyperParameters(distr,mean,std,x0=(0.5,0.5),lb=0,ub=1):
    """Return hyper-parameters given distribution mean and variance.
    
    normal_pdf,lognormal_pdf,beta_pdf,gamma_pdf,t_pdf,weibull_pdf,inv_gamma_pdf,inv_weibull_pdf,wishart_pdf,inv_wishart_pdf
    """
    from scipy.special import gamma
    from scipy.optimize import root
    from scipy.optimize import least_squares
    global func
    
    if distr.endswith("_hp"):
        distr = distr[:-3]
        
    if distr == 'uniform_pdf':
        def func(x):
            a,b = x
            return (a+b)/2-mean, (b-a)**2/12-std**2
        #return least_squares(func,x0).x
        m  = mean - std*np.sqrt(3.0)
        s  = np.sqrt(mean + std*np.sqrt(3.0))
        return m,s
    elif distr == 'normal_pdf':
        return mean,std
        def func(x):
            mu,sigma = x
            # TODO: check this formulas
            return mu-mean,sigma-std**2
        return least_squares(func,x0,bounds=((-np.inf,0),(np.inf,np.inf))).x
    elif distr == 'lognormal_pdf':
        def func(x):
            mu,sigma = x
            return np.exp(mu+sigma**2/2)-mean, (np.exp(sigma**2)-1)*np.exp(2*mu+sigma**2)-std**2        
        return least_squares(func,x0,bounds=((-np.inf,0),(np.inf,np.inf))).x
    elif distr == 'gamma_pdf':
        def func(x):
            kappa,theta = x
            return kappa*theta-mean, kappa*theta**2-std**2
        #return least_squares(func,x0,bounds=((0,0),(np.inf,np.inf))).x
        mu = mean-lb
        b  = std**2/mu;
        a  = mu/b
        return a,b
    elif distr == 'inv_gamma_pdf':
        def func(x):
            alpha,beta = x
            return beta/(alpha-1)-mean, beta**2/(alpha-1)**2/(alpha-2)-std**2
        # return least_squares(func,x0,bounds=((0,0),(np.inf,np.inf))).x
        nu   = 2*(2+mean**2/std**2)
        s    = 2*mean*(1+mean**2/std**2)
        return nu,s
    elif distr == 'inv_gamma1_pdf':
        def func(x):
            y = np.log(2*mu2) - np.log((sigma2+mu2)*(x-2)) + 2*( gammaln(x/2)-gammaln((x-1)/2) )
            return y
        sigma2 = std**2
        mu  = mean
        mu2 = mu**2
        # use_fzero_flag
        # nu  = np.sqrt(2*(2+mu2/std))
        # s   = 2*mu*(1+mu**2/std**2)
        # nu  = root(fun=func,x0=1).x
        nu  = np.sqrt(2*(2+mu2/sigma2))
        nu2 = 2*nu
        nu1 = 2
        err  = func(nu)
        err2 = func(nu2)
        if err2 > 0: # Too short interval.
            while nu2 < 1e12: # Shift the interval containing the root.
                nu1  = nu2
                nu2  = nu2*2
                err2 = func(nu2)
                if err2<0:
                    break
            if err2>0:
                cprint('inverse_gamma_specification:: Failed to find interval containing function sign change! Please check if prior variance is not too small compared to the prior mean...','red')
                sys.exit(-1)
        
        # Solve for nu using the secant method.
        while abs(nu2/nu1-1) > 1e-14:
            if err > 0:
                nu1 = nu
                if nu < nu2:
                    nu = nu2
                else:
                    nu = 2*nu
                    nu2 = nu
            else:
                nu2 = nu
            nu =  (nu1+nu2)/2
            err = func(nu)
        s  = (sigma2+mu2)*(nu-2)
        return s,nu

    elif distr == 'inv_gamma2_pdf':
        def func(x):
            alpha,beta = x
            return beta/(alpha-1)-mean, beta**2/(alpha-1)**2/(alpha-2)-std**2
        # return least_squares(func,x0,bounds=((0,0),(np.inf,np.inf))).x
        mu,s = mean,std
        nu   = 2*(2+mu**2/std**2)
        s    = 2*mu*(1+mu**2/std**2)
        return nu,s
    elif distr == 'logit_pdf':
        def func(x):
            mu,s = x
            return mu-mean, np.pi**2*s**2/3-std**2
        return least_squares(func,x0,bounds=((-np.inf,0),(np.inf,np.inf))).x
    elif distr == 'beta_pdf':
        def func(x):
            alpha,beta = x
            return alpha/(alpha+beta)-mean, alpha*beta/(alpha+beta)**2/(alpha+beta+1)-std**2
        #func = lambda x: (x[0]/(x[0]+x[1])-mean, x[0]*x[1]/(x[0]+x[1])**2/(x[0]+x[1]+1)-std**2)
        #return least_squares(func,x0,bounds=((0,0),(np.inf,np.inf))).x
        mu = (mean-lb)/(ub-lb)
        sigma2 = std**2/(ub-lb)**2
        a = (1-mu)*mu*mu/sigma2-mu
        b = a*(1/mu-1)
        return a,b
    elif distr == 't_pdf':
        def func(x):
            if isinstance(x,list):
                mu,nu = x
                return mu-mean,nu/(nu-2)-std**2
            else:
                return x/(x-2)-std**2
        return least_squares(func,x0,bounds=((-np.inf,0),(np.inf,np.inf))).x
    elif distr == 'weibull_pdf':
        # def func(x):
        #     lmbda,k = x
        #     return lmbda*gamma(1+1/k)-mean, lmbda**2*(gamma(1+2/k)-(gamma(1+1/k))**2)-std**2
        # return least_squares(func,x0,bounds=((0,0),(np.inf,np.inf))).x
        def func(x):
            y = gammaln(1+2./x) - 2*gammaln(1+1./x) - np.log(1+sigma2/mu2)
            return y
        mu = mean-lb
        mu2 = mu*mu
        sigma2 = std**2
        shape = root(fun=func,x0=1).x
        scale = mu/gamma(1+1/shape)
        return shape,scale
    elif distr == 'wishart_pdf':
        def func(x):
            v,n = x
            return n*v-mean, 2*n*v**2-std**2
        return least_squares(func,x0,bounds=((0,0),(np.inf,np.inf))).x
    elif distr == 'inv_wishart_pdf': 
        def func(x):
            psi,df = x
            return psi/(df-2)-mean, 2*psi**2/(df-2)**2/(df-4)-std**2
        return least_squares(func,x0,bounds=((0,0),(np.inf,np.inf))).x
    # elif distr == 'inv_weibull_pdf':
    #     def func(mu,sigma):
    #         lmbda,k = x
    #     return lmbda*gamma(1+1/k)-mean, lmbda**2*(gamma(1+2/k)-(gamma(1+1/k))**2)-std**2
    #     return least_squares(func,x0,bounds=((0,0),(np.inf,np.inf))).x
    return None,None   
 
def compute_prior_mode(hyperparameters,distr):
    """ This function computes the mode of the prior distribution given the (two, three or four) hyperparameters
         of the prior distribution.
        
         INPUTS
           hyperparameters     [double]    1*n vector of hyper parameters.
        
         OUTPUTS
           m       [double]    scalar or 2*1 vector, the prior mode.
        
         note:
         [1] The size of the vector of hyperparameters is 3 when the Gamma or Inverse Gamma is shifted and 4 when
             the support of the Beta distribution is not [0,1].
         [2] The hyperparameters of the uniform distribution are the lower and upper bounds.
         [3] The uniform distribution has an infinity of modes. In this case the function returns the prior mean.
         [4] For the beta distribution we can have 1, 2 or an infinity of modes.

"""
    eps = 1.e-10
    if distr == 'beta_pdf':
        if (hyperparameters[0]>1 and hyperparameters[1]>1):
            m = (hyperparameters(1)-1)/(hyperparameters(1)+hyperparameters(2)-2) 
        elif (hyperparameters[0]<1 and hyperparameters[1]<1):
            m = [0,1] 
        elif ( hyperparameters[0]<1 and hyperparameters[1]>1-eps ) or ( abs(hyperparameters[0]-1)<2*eps and hyperparameters[1]>1 ):
            m = 0
        elif ( hyperparameters[0]>1 and hyperparameters[1]<1+eps ) or ( abs(hyperparameters[0]-1)<2*eps and hyperparameters[1]<1 ):
            m = 1
        elif ( abs(hyperparameters[0]-1)<2*eps and abs(hyperparameters[1]-1)<2*eps ): # Uniform distribution!
            m = 0.5 
        if len(hyperparameters)==4:
            m *= (hyperparameters[3]-hyperparameters[2]) + hyperparameters[2] 
    elif distr == 'inv_gamma_pdf':
        if hyperparameters[0]<1:
            m = 0
        else:
            m = (hyperparameters[0]-1)*hyperparameters[1]
        if len(hyperparameters)>2:
            m += hyperparameters[2]
    elif distr == 'normal_pdf':
        m = hyperparameters(1)
    elif distr == 'inv_gamma1_pdf':
        m = 1/np.sqrt((hyperparameters[1]+1)/hyperparameters[0])
        if len(hyperparameters)>2:
            m += hyperparameters[2]
    elif distr == 'uniform_pdf':
        m = hyperparameters[0]
    elif distr == 'inv_gamma2_pdf':
        m = hyperparameters[0]/(hyperparameters[1]+2) 
        if len(hyperparameters)>2:
            m += hyperparameters[2]
    elif distr == 'weibull_pdf':
        if hyperparameters[0]<=1:
            m = 0
        else:
            m = hyperparameters[1]*((hyperparameters[0]-1)/hyperparameters[0])**(1/hyperparameters[0])
        if len(hyperparameters)>2:
            # Add location parameter
            m += hyperparameters[2] 
    else:
        cprint(f'Unknown prior shape {distr}!','red')
        sys.exit()
    return m
    

def test1():
    import matplotlib.pyplot as plt
    
    distributions = ["uniform_pdf_hp","normal_pdf_hp","lognormal_pdf_hp",
                     "gamma_pdf_hp","inv_gamma_pdf_hp","beta_pdf_hp",
                     "t_pdf_hp","weibull_pdf_hp","wishart_pdf_hp",
                     "inv_wishart_pdf_hp"]
                     
    Plot = True
    if Plot:
        fig, axes = plt.subplots(3,3,figsize=(14,12))
        
    x = 0.5; m = 0
    params = [0.6,0.1,1.0,0.2,0.1]

    for distr in distributions:
        pars = np.copy(params)
        if "wishart" in distr:
            pars[3] = 3
           
        y,b = pdf(distr,x,pars)
        print(f"{b}: {distr}, x={x}, y={y:.3f}, params={params}")
        
        if Plot:
            v = []; z = []
            for i in range(100):
                xx = (1+i)/100.
                y,b = pdf(distr,xx,pars)
                if b:
                    z.append(xx)
                    v.append(y)
            zz = np.array(z)
            values = np.array(v)
            m += 1
            if m <= 9:
                plt.subplot(3,3,m)
                plt.plot(zz,values,lw=2)
                plt.box(True)
                plt.grid(True)
                plt.title(distr,fontsize=20)
                plt.rc('xtick', labelsize=15) 
                plt.rc('ytick', labelsize=15) 
                plt.tight_layout()
                
def test2():
    import matplotlib.pyplot as plt
    global func
    
    priors = [
        ["omega",    0.625,  0, 1,  "beta_pdf",      0.20,      0.10],
        ["alphax",   0.25,   0, 1,  "beta_pdf",      0.10,      0.05],
        ["alphapie", 0.25,   0, 1,  "beta_pdf",      0.10,      0.05],
        ["rhoa",     0.8078, 0, 1,    "beta_pdf",      0.85,      0.10],
        ["rhoe",     0.9873, 0, 1,    "beta_pdf",      0.85,      0.10],
        ["rhopie",   0.6017, 0, 1,    "gamma_pdf",     0.30,      0.10],
        ["rhog",     0.4240, 0, 1,    "gamma_pdf",     0.30,      0.10],
        ["rhox",     0.0770, 0, 1,    "gamma_pdf",     0.25,      0.0625],
        ["std_epsa", 0.150,  0, 0.2, "inv_gamma1_pdf", 0.03,      1.80],
        ["std_epse", 0.0008, 0, 0.0003, "inv_gamma1_pdf", 0.0000727, 0.0000582],
        ["std_epsz", 0.01,   0, 0.03, "inv_gamma1_pdf", 0.005,     0.275],
        ["std_epsr", 0.0030, 0, 0.02, "inv_gamma1_pdf", 0.005,     0.004417],
        ]
                     
    Plot = True
    if Plot:
        fig, axes = plt.subplots(4,3,figsize=(14,12))
        
    x = 0.5; m = 0
    params = [0.6,0.1,1.0,0.2,0.1]

    for prior in priors:
        name = prior[0]
        distr = prior[4]
        params = prior[1:4] + prior[5:]
        pars = np.copy(params)  
        lb=pars[1]; ub=pars[2]
        pars[3],pars[4] = getHyperParameters(distr=distr,mean=pars[3],std=pars[4],lb=pars[1],ub=pars[2])
        f = 1 #np.sqrt(np.sum(np.array(func((pars[3],pars[4])))**2))
        y,b = pdf(distr,x,pars)
        print(f"{name}: {distr}, err={f:.2e}, params={pars[3]:.2e},{pars[4]:.2e}")
        
        if Plot:
            v = []; z = []
            for i in range(100):
                xx = (1+i)/100.
                xx = lb + (ub-lb)*xx
                y,b = pdf(distr,xx,pars)
                if b:
                    z.append(xx)
                    v.append(y)
            zz = np.array(z)
            values = np.array(v)
            m += 1
            if m <= 12:
                plt.subplot(4,3,m)
                plt.plot(zz,values,lw=2)
                plt.box(True)
                plt.grid(True)
                plt.title(distr,fontsize=20)
                plt.rc('xtick', labelsize=15) 
                plt.rc('ytick', labelsize=15) 
                plt.title(name+", "+distr)
                plt.tight_layout()
                
                
if __name__ == '__main__':
    """Main program."""
    
    test1()
    print()
    test2()