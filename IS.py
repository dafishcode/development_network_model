import numpy as np 
import scipy.stats as stat
import matplotlib.pyplot as plt



#========================================
def powerlaw(n,lam):
#========================================
    zeta=np.sum(1.0/np.arange(a,b+1)**lam)
    return(n**(-lam)/zeta)

#========================================
def lognormal(n,mu,sig):
#========================================
    return(1.0/n/np.sqrt(2*np.pi*sig**2)*np.exp(-(np.log(n)-mu)**2/(2*sig**2)))


#========================================
def LogLikelihood(lam, sizes, M, a, b):
#========================================
    """
    Calculate loglikelihood for power law given the data
    Likelihoods across all random draws, given your data
    """
    #normalisation factor for all lambda draws - normalises a distribution to sum of probability = 1 (by summing across all possible values in density)
    zetamat=np.power.outer(1.0/np.arange(a,b+1),lam) #Matrix of normalisation constants for each lambda draw, at each size: each row =  size**-current lambda, for every size from max to min
    zeta=np.sum(zetamat,0) #Norm vector - for each lambda draw - sum of norm constants for entire size max-min range at each lambda
    norm=-M*np.log(zeta) #Contribution of zeta to the likelihood
    nprod=-lam*np.sum(np.log(sizes)) #Loglikelihood calculation, given the data
    loglik=nprod+norm #Normalised loglikelihood
    return(loglik) 

#=============================================
def LogLikelihood_LN(mu,sig, sizes, M , a, b):
#=============================================
    """
    Calculate loglikelihood for lognormal given the data
    Likelihoods across all random draws, given your data
    """
    T1 = -np.sum(np.log(sizes))
    T2_mat = np.subtract.outer(np.log(sizes),mu)**2
    T2 = -np.sum(T2_mat,0)/(2*sig**2)
    T0 = -M*np.log(np.sqrt(2*np.pi) * sig )
    loglik=T0+T1+T2
    return(loglik) 

#=============================================
def IS(npart, sizes, M, a, b):
#=============================================

    """
    IMPORTANCE SAMPLER - for power law - monte carlo sampling from two different distributions
    OUTPUT - posterior average exponent, log marginal likelihood, effective sample size = how good is the sampler
    """

    lambda_sample=np.random.uniform(0.1,5,npart) #randomly sample lambda (exponent values)

    #Weights - loglikelihoods of your data for each lambda * (weight by) log probability of drawing each lambda sample from the prior, divided by log probability of drawing each lambda sample from the proposal
    #Likelihood of data weighted by prior expectation of lambda, and proposal expected lambda - cancel out the effect of the proposal
    weights=LogLikelihood(lambda_sample, sizes, M, a, b)+stat.norm.logpdf(lambda_sample,1,3)-stat.uniform.logpdf(lambda_sample,0.1,5)
    maxw=np.max(weights)
    w2 = np.exp(weights-maxw)
    w2_sum = np.sum(w2)
    ESS=1.0/(np.sum((w2/w2_sum)**2))
    mean_lambda = np.dot(lambda_sample,w2)/w2_sum #average of the lambda value for the posterior distribution
    #marginal likelihood = empirical means of all the weights
    marglik = maxw + np.log(np.sum(np.exp(weights-maxw)))-np.log(npart) #Take the exponent of logs to unlog, before summing/then divded by ncounts
    return([mean_lambda, marglik, LogLikelihood(lambda_sample, sizes, M , a, b), ESS])

#IMPORTANCE SAMPLER - for lognormal - monte carlo sampling from two different distributions
#OUTPUT - posterior average exponent, log marginal likelihood, effective sample size = how good is the sampler
#=============================================
def IS_LN(npart, sizes, M, a, b):
#=============================================

    """
    IMPORTANCE SAMPLER - for lognormal - monte carlo sampling from two different distributions
    OUTPUT - posterior average exponent, log marginal likelihood, effective sample size = how good is the sampler
    """

    mu_sample = np.random.uniform(-2.0,2.0,npart) #randomly sample mu
    sig_sample = np.random.uniform(0.1,5.0,npart) #randomly sample sigma
    weights=LogLikelihood_LN(mu_sample,sig_sample, sizes, M, a, b) 
    maxw=np.max(weights)
    w2 = np.exp(weights-maxw)
    w2_sum = np.sum(w2)
    ESS=1.0/(np.sum((w2/w2_sum)**2))
    wmax_ID=np.argmax(w2)
    mean_mu = mu_sample[wmax_ID]
    mean_sig = sig_sample[wmax_ID]
    #marginal likelihood = empirical means of all the weights
    marglik = maxw + np.log(np.sum(np.exp(weights-maxw)))-np.log(npart)
    return([mean_mu,mean_sig, marglik,  LogLikelihood_LN(mu_sample, sig_sample, sizes, M, a, b), ESS])

#=============================================
def plot_samples(npart):
#=============================================
    lambda_sample=np.random.uniform(0.1,5,npart)
    weights=LogLikelihood(lambda_sample)
    maxw=np.max(weights)
    w2 = np.exp(weights-maxw)
    plt.hist(lambda_sample,weights=w2,bins=np.linspace(2.5,2.8))
    plt.show()
    
#=============================================
def plotcomp(lam,mu,sig):
#=============================================

    x = np.linspace(a,b,40) 
    plt.hist(sizes,40,log=True,density=True)
    plt.plot(x,powerlaw(x,lam))
    plt.plot(x,lognormal(x,mu,sig))
    plt.show()