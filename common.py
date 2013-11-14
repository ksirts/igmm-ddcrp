'''
Created on Nov 11, 2013

@author: kairit
'''

from __future__ import division
import math, random
from numpy.linalg import inv, slogdet, cholesky
from scipy.stats import chi2
import numpy.random as npr
from scipy.special import multigammaln, gammaln
from scipy.misc import logsumexp
from numpy import trace, dot, ones
import numpy as np

def cholupdate(R,x,sign):
    p = np.size(x)
    x = x.T
    for k in range(p):
        if sign == '+':
            r = np.sqrt(R[k,k]**2 + x[k]**2)
        elif sign == '-':
            r = np.sqrt(R[k,k]**2 - x[k]**2)
        c = r/R[k,k]
        s = x[k]/R[k,k]
        R[k,k] = r
        if sign == '+':
            R[k,k+1:p] = (R[k,k+1:p] + s*x[k+1:p])/c
        elif sign == '-':
            R[k,k+1:p] = (R[k,k+1:p] - s*x[k+1:p])/c
        x[k+1:p]= c*x[k+1:p] - s*R[k, k+1:p]
    return R

def invwishartrand(nu, phi):
    invphi = inv(phi)
    wishdraw = wishartrand(nu, invphi)
    return inv(wishdraw)

def wishartrand(nu, phi):
    dim = phi.shape[0]
    chol = cholesky(phi)
    #nu = nu+dim - 1
    #nu = nu + 1 - np.arange(1,dim+1)
    foo = np.zeros((dim,dim))
    
    for i in range(dim):
        for j in range(i+1):
            if i == j:
                foo[i,j] = np.sqrt(chi2.rvs(nu-(i+1)+1))
            else:
                foo[i,j]  = npr.normal(0,1)
    return dot(chol, dot(foo, dot(foo.T, chol.T)))

def logNormalize(probs):
    Z = logsumexp(probs)
    return np.exp(probs - Z)

def sampleIndex(probs):
    p = random.random() * sum(probs)
    sum_ = 0
    for i in range(len(probs)):
        sum_ += probs[i]
        if p < sum_:
            return i
    assert False
    
def mylogsumexp(probs):
    probs = np.array(probs)
    a = np.amax(probs)
    return a + np.log(np.exp(probs - a).sum())



class LoginvWishartPdf(object):
    
    def __init__(self, Lambda, nu):
        self.invlambda = inv(Lambda)
        self.Lambda = Lambda
        self.nu = nu
        self.d = Lambda.shape[0]
        self.Z = nu * self.d / 2 * math.log(2) - nu / 2 * slogdet(Lambda)[1] + multigammaln(nu / 2, self.d)
        
    def __call__(self, x, xdet):
        prob = (self.nu + self.d + 1) / 2 * xdet
        prob -= 0.5 * trace(dot(self.Lambda, x))
        prob -= self.Z
        return prob
    
class MultivariateNormalLikelihood(object):
    
    def __init__(self, d):
        self.partialZ = d / 2 * math.log(2 * math.pi)
        

    def __call__(self, s, ss, N, mu, precision, logdet):
        ddM = ss + N * np.dot(mu, mu.T) - 2 * np.dot(s, mu.T)
        prob = -0.5 * np.multiply(ddM, precision).sum()
        Z = N * (self.partialZ - 0.5 * logdet)
        return prob - Z
    
class MultivariateStudentT(object):
    
    def __init__(self, d, nu, mu, Lambda):
        self.nu = nu
        self.d = d
        self.mu = mu
        self.precision = inv(Lambda)
        self.logdet = slogdet(Lambda)[1]
        self.Z = gammaln(nu / 2) + d / 2 * (math.log(nu) + math.log(math.pi)) - gammaln((nu + d) / 2)
        
        
    def __call__(self, x):
        diff = (x - self.mu)
        term = 1. / self.nu * np.dot(np.dot(diff.T, self.precision), diff)[0][0]
        second = -(self.nu + self.d) / 2 * math.log(1 + term)
        prob = -0.5 * self.logdet + second
        return  prob - self.Z
    
def logmvstprob(x, mu, nu, d, Lambda):
    diff = x - mu
    prob = gammaln((nu + d) / 2)
    prob -= gammaln(nu / 2)
    prob -= d / 2 * (math.log(nu) + math.log(math.pi))
    prob -= 0.5 * slogdet(Lambda)[1]
    prob -= (nu + d) / 2. * math.log(1 + 1. / nu * np.dot(np.dot(diff.T, inv(Lambda)), diff)[0][0])
    return prob
    
class Constants(object):
        
    def __init__(self, dim, mean, alpha, Lambda, pruning, kappa, a=1, priorth=-10, seq=False):
        self.nu0 = dim + 1
        self.mu0 = mean
        self.alpha = alpha
        self.logalpha = math.log(alpha)
        self.lambda0 = Lambda * np.identity(dim)
        self.kappa0 = kappa
        self.a = a
        self.pruningfactor = pruning
        self.invlambda0 = inv(self.lambda0)
        self.priorth = priorth
        self.seq = seq
        self.kappa0_outermu0 = kappa * np.dot(mean, mean.T)
        self.logdet = slogdet(self.invlambda0)[1]
        self.precision = self.kappa0 * (self.nu0 - dim - 1) / (self.kappa0 + 1) * self.invlambda0
        self.changeParams = 10


class State(object):
    
    def __init__(self, vocab, data, con):
        self.data = data
        self.n, self.d = data.shape
        self.vocab = vocab
        self.con = con
        self.assignments = -1 * ones(self.n, dtype=np.int)
        self.loginvWishartPdf = LoginvWishartPdf(con.lambda0, con.nu0)
        self.mvNormalLL = MultivariateNormalLikelihood(self.d)
        
    def initialize(self):
        self.K = 0
        self.mu = np.zeros((self.con.pruningfactor, self.d, 1), np.float)
        self.precision = np.zeros((self.con.pruningfactor, self.d, self.d), np.float)
        self.logdet = np.zeros(self.con.pruningfactor, np.float)
        self.counts = np.zeros(self.con.pruningfactor, np.int)
        self.dd = np.zeros((self.n, self.d, self.d), dtype=float)
        self.s = np.zeros((self.con.pruningfactor, self.d, 1), np.float)
        self.ss = np.zeros((self.con.pruningfactor, self.d, self.d), np.float)
        self.cluster_likelihood = np.zeros(self.con.pruningfactor, np.float)
        self.paramprobs = np.zeros(self.con.pruningfactor, np.float)
        self.denom = np.zeros(self.con.pruningfactor)
    
    def resampleParams(self):
        for t in range(self.K):
            mu, precision = self.sampleNewParams(t)
            self.mu[t] = mu
            self.precision[t] = precision
            precdet = slogdet(precision)[1]
            self.logdet[t] = precdet
            ll = self.mvNormalLL(self.s[t], self.ss[t], self.counts[t], mu, precision, self.logdet[t])
            self.cluster_likelihood[t] = ll
            paramprob = self.loginvWishartPdf(precision, precdet)
            paramprob += self.mvNormalLL(mu, dot(mu, mu.T), 1, self.con.mu0, self.con.kappa0 * precision, self.d * math.log(self.con.kappa0) + precdet)
            self.paramprobs[t] = paramprob
                
    def sampleNewParams(self, t):
        n = self.counts[t]
        nun = self.con.nu0 + n
        kappan = self.con.kappa0 + n
        mun = (self.con.kappa0 * self.con.mu0 + self.s[t]) / kappan
        lambdan = self.con.lambda0 + self.ss[t] + self.con.kappa0_outermu0 - kappan * dot(mun, mun.T)
        precision = wishartrand(nun, inv(lambdan))
        mu = npr.multivariate_normal(mun.T[0], inv(kappan * precision))[:,None]
        return mu, precision
    
    def integrateOverParameters(self, n, s, ss):       
        kappan = self.con.kappa0 + n
        nun = self.con.nu0 + n
        mun = (self.con.kappa0 * self.con.mu0 + s) / kappan
        lambdan = self.con.lambda0 + ss + self.con.kappa0_outermu0 - kappan * np.dot(mun, mun.T)
            
        ll = self.d / 2 * math.log(self.con.kappa0) + self.con.nu0 / 2 * slogdet(self.con.lambda0)[1]
        ll -= n * self.d / 2 * math.log(math.pi) + self.d / 2 * math.log(kappan) + nun / 2 * slogdet(lambdan)[1]
        for j in range(1, self.d + 1):
            ll += gammaln((nun + 1 - j) / 2) - gammaln((self.con.nu0 + 1 - j) / 2)
        return ll
    
    def posteriorPredictive(self, s_, ss_, t):
        s = self.s[t] + s_
        ss = self.ss[t] + ss_
        num = self.integrateOverParameters(self.counts[t] + 1, s, ss)
        #denom = self.integrateOverParameters(self.counts[t], self.s[t], self.ss[t])
        #assert (denom - self.denom[t]) < 1e-10
        res = num - self.denom[t]
        return res
    
    def assertCounts(self):
        for t in xrange(self.K):
            assert self.counts[t] == (self.assignments == t).sum()
            
    def assertAssignments(self):
        for i, followers in enumerate(self.sit_behind):
            t = self.assignments[i]
            for item in followers:
                if self.assignments[item] != t:
                    print t, self.assignments[item], item
                    assert self.assignments[item] == t

    def numClusters(self):
        return self.K

    def histogram(self):
        return ' '.join(map(str, self.counts[:self.K]))
    
    def getIndices(self, t):
        return [i for i in xrange(self.n) if self.assignments[i] == t]
