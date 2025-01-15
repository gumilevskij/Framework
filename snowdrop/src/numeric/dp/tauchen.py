import scipy.stats as st
import scipy as sp


def tauchen_hussey(N,mu,rho,sigma,baseSigma):
	""" 
	Find a Markov chain whose sample paths approximate those of the AR(1) process
                
        .. math:: 
			z_{t+1} = (1-rho)*mu + rho*z_{t} + eps_{t+1}
                    
	Here eps is the normal noise with stddev sigma

	Format:     
                Z, Zprob = tauchen_hussey(N,mu,rho,sigma,m)

	Input:      
                N:         scalar, number of nodes for Z
    		    
                mu:        scalar, unconditional mean of process
    		    
                rho:       scalar
    		    
                sigma:     scalar, std. dev. of epsilons
    		    
                baseSigma: scalar, std. dev. used to calculate Gaussian quadrature weights and nodes, i.e. to build the grid. 
                
                It is recommended to you use:
                    .. math:: 
                            baseSigma = w*sigma +(1-w)*sigmaZ 
                where 
                    .. math:: 
                        sigmaZ = sigma/sqrt(1-rho^2), 
                        
                and      
                    .. math::
                        w = 0.5 + rho/4
                        
                Tauchen & Hussey recommend baseSigma = sigma, and also mention baseSigma = sigmaZ.

	Output:     
                Z:       N*1 vector, nodes for Z
				Zprob:   N*N matrix, transition probabilities

	Author:		
                Benjamin Tengelsen, Brigham Young University (python) Martin Floden, Stockholm School of Economics (original) January 2007 (updated August 2007)

	This procedure is an implementation of Tauchen and Hussey's
	algorithm, Econometrica (1991, Vol. 59(2), pp. 371-396)
	
    """
	Z     = sp.zeros((N,1))
	Zprob = sp.zeros((N,N))
	[Z,w] = GaussNorm(N,mu,baseSigma**2)
	for i in range(N):
		for j in range(N):
			EZprime    = (1-rho)*mu + rho*Z[i]
			Zprob[i,j] = w[j] * st.norm.pdf(Z[j],EZprime,sigma) / st.norm.pdf(Z[j],mu,baseSigma)
		
	for i in range(N):
		Zprob[i,:] = Zprob[i,:] / sum(Zprob[i,:])
		
	return Z.T,Zprob


def GaussNorm(n,mu,s2):
	""" 
	Find Gaussian nodes and weights for the normal distribution
	n  = # nodes
	mu = mean
	s2 = variance
	"""
	[x0,w0] = GaussHermite(n)
	x = x0*sp.sqrt(2.*s2) + mu
	w = w0/sp.sqrt(sp.pi)
	return [x,w]

	
def GaussHermite(n):
	"""
	Gauss Hermite nodes and weights following 'Numerical Recipes for C' 
	"""

	MAXIT = 10
	EPS   = 3e-14
	PIM4  = 0.7511255444649425

	x = sp.zeros((n,1))
	w = sp.zeros((n,1))

	m = int((n+1)/2)
	for i in range(m):
		if i == 0:
			z = sp.sqrt((2.*n+1)-1.85575*(2.*n+1)**(-0.16667))
		elif i == 1:
			z = z - 1.14*(n**0.426)/z
		elif i == 2:
			z = 1.86*z - 0.86*x[0]
		elif i == 3:
			z = 1.91*z - 0.91*x[1]
		else:
			z = 2*z - x[i-1]
		
		for iter in range(MAXIT):
			p1 = PIM4
			p2 = 0.
			for j in range(n):
				p3 = p2
				p2 = p1
				p1 = z*sp.sqrt(2./(j+1))*p2 - sp.sqrt(float(j)/(j+1))*p3
			pp = sp.sqrt(2.*n)*p2
			z1 = z
			z = z1 - p1/pp
			if sp.absolute(z-z1) <= EPS:
				break
		
		if iter>MAXIT:
			raise ValueError('too many iterations')
		x[i,0]     = z
		x[n-i-1,0] = -z
		w[i,0]     = 2./pp/pp
		w[n-i-1,0] = w[i]
	
	x = x[::-1]
	return [x,w]
