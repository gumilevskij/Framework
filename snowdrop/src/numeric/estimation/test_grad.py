import os,sys,warnings
warnings.filterwarnings('ignore')
import autograd.numpy as np
from autograd import grad, jacobian, hessian
from autograd.scipy.stats import norm
from scipy.optimize import minimize
import statsmodels.api as sm
import pandas as pd
import time

path = os.path.dirname(os.path.abspath(__file__))
working_dir = os.path.abspath(path+"\\..\\..\\")

# suppose values for x and y are as follows
x, y = 2.0, 3.0
    
def f(x, y):
    return x**(.8) * y**(.2)

# first derivatives - for f(x,y), x is position 0 (default)  
#                         and y is position 1
dfdx = grad(f)
dfdy = grad(f, 1)


# evaluate the gradiant at x, y
derivs = np.array([dfdx(x,y), dfdy(x,y)])
print("Gradiant: ", derivs)

# analytical derivatives
a_dfdx = lambda x, y: .8*x**(-.2)*y**(.2)
a_dfdy = lambda x, y: .2*x**(.8)*y**(-.8)

print("Analytical Gradient: ", np.array([ a_dfdx(x,y), a_dfdy(x,y)])) 

# second derivates
d2fdxdx = grad(dfdx)
d2fdydy = grad(dfdy, 1)
d2fdxdy = grad(dfdx, 1)
d2fdydx = grad(dfdy)

# number of observations
N = 5000
# number of parameters
K = 10
# true parameter values
beta = 2 * np.random.randn(K)
# true error std deviation
sigma =  2

def datagen(N, beta, sigma):
    """
    Generates data for OLS regression.
    Inputs:
    N: Number of observations
    beta: K x 1 true parameter values
    sigma: std dev of error
    """
    K = beta.shape[0]
    x_ = 10 + 2 * np.random.randn(N,K-1)
    # x is the N x K data matrix with column of ones
    #   in the first position for estimating a constant
    x = np.c_[np.ones(N),x_]
    # y is the N x 1 vector of dependent variables
    y = x.dot(beta) + sigma*np.random.randn(N)
    return y, x

y, x  = datagen(N, beta, sigma)

def neg_loglike_(theta):
    beta = theta[:-1]
    sigma = theta[-1]
    mu = np.dot(x,beta)
    ll = -N/2 * np.log(2*np.pi*sigma**2) - (1/(2*sigma**2)) * np.sum((y - mu)**2)
    return -1 * ll

def neg_loglike(theta):
    beta = theta[:-1]
    # transform theta[-1]
    # so that sigma > 0
    sigma = np.exp(theta[-1])
    mu = np.dot(x,beta)
    ll = norm.logpdf(y,mu,sigma).sum()
    return -1 * ll

def test():
    # derivates of neg_loglike
    jacobian_  = jacobian(neg_loglike)
    hessian_ = hessian(neg_loglike)
    
    # evaluate the gradiant at true theta
    # theta = [beta log(sigma)]
    theta = np.append(beta,np.log(sigma))
    
    theta_start = np.append(np.zeros(beta.shape[0]),0.0)
    res1 = minimize(neg_loglike, theta_start, method = 'BFGS', \
    	       options={'disp': False}, jac = jacobian_)
    print("Convergence Achieved: ", res1.success)
    print("Number of Function Evaluations: ", res1.nfev)
    
    # estimated parameters
    theta_autograd = res1.x
    
    # for std errors, calculate the information matrix
    # using the autograd hessian
    information1 = np.transpose(hessian_(theta_autograd))
    se1 = np.sqrt(np.diagonal(np.linalg.inv(information1))) 
    
    # Put Results in a DataFrame
    results_a = pd.DataFrame({'Parameter':theta_autograd,'Std Err':se1})
    names = ['beta_'+str(i) for i in range(K)]
    names.append('log(Sigma)')
    results_a['Variable'] = names
    results_a['Model'] = "MLE Autograd"
    
    print(jacobian_(theta_autograd))
    print(jacobian_(theta))
    
    ## Comparison with OLS and Non-Autograd MLE
    
    # Using scipy OLS
    res_ols = sm.OLS(y,x).fit()
    
    # Put Results in a DataFrame
    results_o = pd.DataFrame({'Parameter':res_ols.params,
    			  'Std Err':res_ols.HC0_se})
    names = ['beta_'+str(i) for i in range(K)]
    results_o['Variable'] = names
    results_o['Model'] = "OLS"
    
    res2 = minimize(neg_loglike, theta_start, method = 'BFGS', \
    	       options={'disp': False}) 
    se2 = np.sqrt(np.diag(res2.hess_inv))
    theta2 = res2.x
    
    # Put Results in a DataFrame
    results_ = pd.DataFrame({'Parameter':theta2,'Std Err':se2})
    names = ['beta_'+str(i) for i in range(K)]
    names.append('log(Sigma)')
    results_['Variable'] = names
    results_['Model'] = "MLE"
    
    print("Convergence Achieved: ", res1.success)
    print("Number of Function Iterations: ", res2.nfev)
    print("Gradiant: ", jacobian_(res2.x))
    
    # combine results and print
    results_a.set_index(['Variable','Model'],inplace=True)
    results_o.set_index(['Variable','Model'],inplace=True)
    results_.set_index(['Variable','Model'],inplace=True)
    df_ = results_o.append(results_a).append(results_).unstack()
    
    print("Parameters")
    print(df_['Parameter'].head(K+1))
    
    # combine results and print
    # results_a.set_index(['Variable','Model'],inplace=True)
    # results_o.set_index(['Variable','Model'],inplace=True)
    # results_.set_index(['Variable','Model'],inplace=True)
    df_ = results_o.append(results_a).append(results_).unstack()
    
    print("Parameters")
    print(df_['Parameter'].head(K+1))
    
    ##Speed Up
    for k in [10, 50, 100]:
         beta = np.random.randn(k)
         theta_start = np.append(beta,1.0)
         y, x = datagen(N, beta, 1)
         print("Time for " + str(k) +" Parameters")
         start = time.time()
         minimize(neg_loglike, theta_start, method = 'BFGS', options={'disp': False}, jac = jacobian_)
         t1 = time.time()-start
         start = time.time()
         minimize(neg_loglike, theta_start, method = 'BFGS', options={'disp': False})   
         t2 = time.time()-start
         print(" %s seconds   vs   %s seconds  -  performance increase: %f" % (t1,t2,t2/t1))
    		
    df = pd.DataFrame(.001*np.c_[[214, 573, 1190],[961, 12000, 46600]],columns=['With Autograd','Without Autograd'],index=[10, 50, 100])
    df['Speed Up'] = df['Without Autograd']/df['With Autograd']
    df.index.name = "K"
    df.head()

if __name__ == '__main__':
    test()

		