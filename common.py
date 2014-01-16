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
#from scipy.misc import logsumexp
from numpy import trace, dot, ones
import numpy as np

#from choldate import cholupdate, choldowndate


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
    p = random.random() * probs.sum()
    sum_ = 0
    for i in range(len(probs)):
        sum_ += probs[i]
        if p < sum_:
            return i
    assert False
    
def logsumexp(probs):
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
        ddM = ss + N * np.outer(mu, mu) - 2 * np.outer(s, mu)
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
        diff = (x - self.mu)[:,None]
        term = 1. / self.nu * np.dot(np.dot(diff.T, self.precision), diff)[0][0]
        second = -(self.nu + self.d) / 2 * math.log(1 + term)
        prob = -0.5 * self.logdet + second
        return  prob - self.Z
    
def logmvstprob(x, mu, nu, d, Lambda):
    diff = x - mu[:,None]
    prob = gammaln((nu + d) / 2)
    prob -= gammaln(nu / 2)
    prob -= d / 2 * (math.log(nu) + math.log(math.pi))
    prob -= 0.5 * slogdet(Lambda)[1]
    prob -= (nu + d) / 2. * math.log(1 + 1. / nu * np.dot(np.dot(diff.T, inv(Lambda)), diff)[0][0])
    return prob
    
class Constants(object):
        
    def __init__(self, nu, mean, out, alpha, scale, pruning, kappa, a=1, priorth=-10, seq=False):
        self.nu0 = nu
        self.mu0 = mean
        self.alpha = alpha ** a
        self.logalpha = math.log(alpha) * a
        self.lambda0 = scale
        self.kappa0 = kappa
        self.a = a
        self.pruningfactor = pruning
        self.invlambda0 = inv(self.lambda0)
        self.priorth = priorth
        self.seq = seq
        self.kappa0_outermu0 = kappa * np.outer(mean, mean)
        self.logdet = slogdet(self.lambda0)[1]
        #self.precision = self.kappa0 * (self.nu0 - dim - 1) / (self.kappa0 + 1) * self.invlambda0
        self.changeParams = 10
        self.out = out


class State(object):
    
    def __init__(self, vocab, data, con):
        self.data = data
        self.n, self.d = data.shape
        self.vocab = vocab
        self.con = con
        self.loginvWishartPdf = LoginvWishartPdf(con.lambda0, con.nu0)
        self.mvNormalLL = MultivariateNormalLikelihood(self.d)
        
    def initialize(self):
        self.K = 0
        self.mu = np.zeros((self.con.pruningfactor, self.d), np.float)
        self.precision = np.zeros((self.con.pruningfactor, self.d, self.d), np.float)
        self.logdet = np.zeros(self.con.pruningfactor, np.float)
        self.counts = np.zeros(self.con.pruningfactor, np.int)
        self.dd = np.zeros((self.n, self.d, self.d), dtype=float)
        self.s = np.zeros((self.con.pruningfactor, self.d), np.float)
        self.ss = np.zeros((self.con.pruningfactor, self.d, self.d), np.float)
        self.cluster_likelihood = np.zeros(self.con.pruningfactor, np.float)
        self.paramprobs = np.zeros(self.con.pruningfactor, np.float)
        self.denom = np.zeros(self.con.pruningfactor)
        self.cholesky = np.zeros((self.con.pruningfactor, self.d, self.d), np.float)
    
    def resampleParams(self):
        for t in range(self.K):
            mu, precision = self.sampleNewParams(t)
            self.mu[t] = mu
            self.precision[t] = precision
            precdet = slogdet(precision)[1]
            self.logdet[t] = precdet
            ll = self.mvNormalLL(self.s[t], self.ss[t], self.counts[t], mu, precision, self.logdet[t])
            self.cluster_likelihood[t] = ll
            paramprob = self.param_probs(t)
            self.paramprobs[t] = paramprob
                
    def sampleNewParams(self, t):
        n = self.counts[t]
        nun = self.con.nu0 + n
        kappan = self.con.kappa0 + n
        mun = (self.con.kappa0 * self.con.mu0 + self.s[t]) / kappan
        lambdan = self.con.lambda0 + self.ss[t] + self.con.kappa0_outermu0 - kappan * np.outer(mun, mun)
        precision = wishartrand(nun, inv(lambdan))
        mu = npr.multivariate_normal(mun, inv(kappan * precision))
        return mu, precision
    
    def integrateOverParameters(self, n, s, ss, logdet=None):       
        kappan = self.con.kappa0 + n
        nun = self.con.nu0 + n
        if logdet is None:
            mun = (self.con.kappa0 * self.con.mu0 + s) / kappan
            lambdan = self.con.lambda0 + ss + self.con.kappa0_outermu0 - kappan * np.outer(mun, mun)
            logdet = slogdet(lambdan)[1]
            
        ll = self.d / 2 * math.log(self.con.kappa0) + self.con.nu0 / 2 * self.con.logdet
        ll -= n * self.d / 2 * math.log(math.pi) + self.d / 2 * math.log(kappan) + nun / 2 * logdet
        for j in range(1, self.d + 1):
            ll += gammaln((nun + 1 - j) / 2) - gammaln((self.con.nu0 + 1 - j) / 2)
        return ll
    
    def printPosteriorVariance(self, f, t):
        kappan = self.con.kappa0 + self.counts[t]
        mun = (self.con.kappa0 * self.con.mu0 + self.s[t]) / kappan
        lambdan = self.con.lambda0 + self.ss[t] + self.con.kappa0_outermu0 - kappan * np.outer(mun, mun)
        nun = self.con.nu0 + self.counts[t]
        var = lambdan / (nun - self.d - 1)
        res = ""
        for i in range(self.d):
            for j in range(self.d):
                res += str(round(var[i, j], 2)) + "\t"
            res = res.strip()
            res += '\n'
        res += '\n'
        header = str(t) + " : " + str(self.counts[t]) + '\n'
        f.write(header)
        sumdiag = np.sum(np.diag(var))
        row = "trace of cov: " + str(round(sumdiag, 2)) + '\n'
        f.write(row)
        row = "scaled by cluster size: " + str(round(sumdiag * self.counts[t], 2)) + '\n'
        f.write(row)
        avgoffdiag = np.sum(var - np.diag(np.diag(var))) / (self.d * (self.d-1))
        row = "avg off-diagonal: " + str(round(avgoffdiag, 2)) + '\n'
        f.write(row)
        f.write(res)
        
    def posterorVariance(self, t):
        kappan = self.con.kappa0 + self.counts[t]
        mun = (self.con.kappa0 * self.con.mu0 + self.s[t]) / kappan
        lambdan = self.con.lambda0 + self.ss[t] + self.con.kappa0_outermu0 - kappan * np.outer(mun, mun)
        nun = self.con.nu0 + self.counts[t]
        var = lambdan / (nun - self.d - 1)
        return var
    
    def sampleVariance(self, t):
        Q = self.ss[t] - np.outer(self.s[t], self.s[t]) / self.counts[t]
        mean = Q/ (self.counts[t] - 1)
        return mean
        
    
    def posteriorPredictive(self, n, s_, ss_, t):
        s = self.s[t] + s_
        ss = self.ss[t] + ss_
        
        n_ = self.counts[t]
        kappan = self.con.kappa0 + n_
        nun = self.con.nu0 + n_

        kappa_star = self.con.kappa0 + self.counts[t] + n
        nu_star = self.con.nu0 + self.counts[t] + n

        mu_star = (self.con.kappa0 * self.con.mu0 + s) / kappa_star
        lambda_star = self.con.lambda0 + ss + self.con.kappa0_outermu0 - kappa_star * np.outer(mu_star, mu_star)

        res = self.d / 2 * math.log(kappan) + nun / 2 * self.logdet[t]
        res -= n * self.d/2 * math.log(math.pi) + self.d / 2 * math.log(kappa_star) + nu_star / 2 * slogdet(lambda_star)[1]
        for j in range(1, self.d + 1):
            res += gammaln((nu_star + 1 - j) / 2) - gammaln((nun + 1 - j) / 2)
      
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
    
    def param_probs(self, t):
        prob = (self.con.nu0 + self.d + 2) / 2 * self.logdet[t]
        diff = (self.mu[t] - self.con.mu0)[:,None]
        prob -= self.con.kappa0 / 2 * np.dot(np.dot(diff.T, self.precision[t]), diff)
        prob -= 0.5 * np.trace(np.dot(self.precision[t], self.con.lambda0))
        return prob
    
    def param_probabilites(self):
        return self.paramprobs.sum()
    
    def likelihood(self):
        return self.cluster_likelihood.sum()
    
    def likelihood_int(self):
        ll = 0.0
        for t in range(self.K):
            ll += self.integrateOverParameters(self.counts[t], self.s[t], self.ss[t])
        return ll
    
    def prepareSuffixes(self, featfn):
        self.featlist = []
        with open(featfn + '.featlist') as f:
            for line in f:
                self.featlist.append(line.strip())
        
        self.w = np.zeros(len(self.featlist))

        self.featsets = []
        with open(featfn + '.featsets') as f:
            for line in f:
                self.featsets.append(map(int, line.split()))

        self.features = np.load(featfn + '.feats.npy')
        
        self.featcounts = []
        for i in range(self.n):
            counts = np.bincount(self.features[i])
            counts[0] -= 1
            ii = np.nonzero(counts)[0]
            self.featcounts.append(zip(ii, counts[ii]))
            
    def setParams(self):
        for i in range(len(self.featsets)):
            term = 0.0
            for f in self.featsets[i]:
                term += self.w[f]
            self.featsetvals[i] = term
