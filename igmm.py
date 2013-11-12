'''
Created on Aug 9, 2013

@author: Kairit
'''
from __future__ import division

import math, random
import numpy as np
import numpy.random as npr
from numpy.linalg import inv, slogdet
from scipy.special import gammaln
import argparse
import sys

from common import Constants
from common import State
from common import invwishartrand, logNormalize, sampleIndex

def generateData(Lambda, alpha):
    means = []
    sigmas = []
    
    num_comp = 4
    dim = 2
    nu = dim
    Lambda = Lambda * np.identity(dim)
    #sigma = np.identity(dim)
    #priormeans = np.array(range(num_comp)) * 20
    for t in range(num_comp):
        sigma = invwishartrand(nu, Lambda)
        sigmas.append(sigma)
        #mu = np.array(random.sample(range(60), 2))
        mu = np.zeros(2)
        mean = npr.multivariate_normal(mu, sigma)
        means.append(mean)
    #prop = npr.dirichlet([1] * num_comp, 1)[0]
    data = []
    labels = []
    K = 0
    for i in range(100):
        comp = random.randrange(num_comp)
        point = npr.multivariate_normal(means[comp], sigma)
        data.append(point)
        labels.append(comp)
    return np.array(data), labels, means, sigmas


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
        term = 1 / self.nu * np.dot(np.dot(diff.T, self.precision), diff)[0][0]
        second = -(self.nu + self.d) / 2 * math.log(1 + term)
        prob = -0.5 * self.logdet + second
        return  prob - self.Z

        
class IGMMState(State):
    
    def __init__(self, vocab, data, con):
        State.__init__(self, vocab, data, con)
        std_nu = self.con.nu0 - self.d + 1
        std_S = (self.con.kappa0 + 1) / (self.con.kappa0 * std_nu) * self.con.lambda0
        self.mvStudentT = MultivariateStudentT(self.d, std_nu, self.con.mu0, std_S)
        

    def initialize(self):
        super(IGMMState, self).initialize()

        self.integrated = np.zeros(self.n, np.float)
        self.countsgammaln = np.zeros(self.con.pruningfactor, np.float)

        for i in range(self.n):
            if self.K < self.con.pruningfactor and random.random() < self.con.alpha / (i + self.con.alpha):
                ind = self.K
                self.K += 1      
            else:
                ind = random.sample(self.assignments[:i], 1)[0]
                self.countsgammaln[ind] += math.log(self.counts[ind])
            self.assignments[i] = ind
            self.counts[ind] += 1
            d = self.data[i][:,None]
            self.s[ind] += d
            self.dd[i] = np.dot(d, d.T)
            self.ss[ind] += self.dd[i]
            self.integrated[i] = self.mvStudentT(d)
            '''
            kappan = self.con.kappa0 + 1
            mun = (self.con.kappa0 * self.con.mu0 + d) / kappan
            lambdan = self.con.lambda0 + self.dd[i] + self.con.kappa0_outermu0 - kappan * np.dot(mun, mun.T)
            
            integrated2 = self.d / 2 * math.log(self.con.kappa0) + self.con.nu0 / 2 * slogdet(self.con.lambda0)[1]
            integrated2 -= self.d / 2 * math.log(math.pi) + self.d / 2 * math.log(self.con.kappa0 + 1) + (self.con.nu0 + 1) / 2 * slogdet(lambdan)[1]
            for j in range(1, self.d + 1):
                integrated2 += gammaln((self.con.nu0 + 2 - j) / 2) - gammaln((self.con.nu0 + 1 - j) / 2)
            '''
 
        self.probs = np.zeros(self.con.pruningfactor)    
        
            
    def resampleData(self):
        permuted = range(self.n)
        random.shuffle(permuted)
        for i in permuted:
            self.removeItem(i)
            newt = self.sample(i)
            self.addItem(i, newt)

                
    def removeItem(self, i):
        t = self.assignments[i]
        item = self.data[i][:,None]
        self.s[t] -= item
        self.ss[t] -= self.dd[i]
        self.counts[t] -= 1
        assert self.counts[t] >= 0
        ll = self.mvNormalLL(item, self.dd[i], 1, self.mu[t], self.precision[t], self.logdet[t])
        self.cluster_likelihood[t] -= ll
        if self.counts[t] == 0:
            assert abs(self.cluster_likelihood[t]) <= 10e-10
        if self.counts[t] > 0:
            self.countsgammaln[t] -= math.log(self.counts[t])

        if self.counts[t] == 0:
            self.removeCluster(t)

    def removeCluster(self, t):
        self.K -= 1
        if t != self.K:
            self.mu[t] = self.mu[self.K]
            self.precision[t] = self.precision[self.K]
            self.logdet[t] = self.logdet[self.K]
            self.s[t] = self.s[self.K]
            self.ss[t] = self.ss[self.K]
            self.counts[t] = self.counts[self.K]
            self.countsgammaln[t] = self.countsgammaln[self.K]
            self.cluster_likelihood[t] = self.cluster_likelihood[self.K]
            self.paramprobs[t] = self.paramprobs[self.K]
            self.assignments = [t if x==self.K else x for x in self.assignments]
        self.mu[self.K].fill(0.)
        self.precision[self.K].fill(0.)
        self.logdet[self.K].fill(0.)
        self.s[self.K].fill(0)
        self.ss[self.K].fill(0.)
        self.counts[self.K].fill(0)
        self.countsgammaln[self.K].fill(0.)
        self.cluster_likelihood[self.K].fill(0.)
        self.paramprobs[self.K].fill(0.)
        
    
    def addItem(self, i, t):
        self.s[t] += self.data[i][:,None]
        self.ss[t] += self.dd[i]
        if self.counts[t] > 0:
            self.countsgammaln[t] += math.log(self.counts[t])
        self.counts[t] += 1
        self.assignments[i] = t
        
        if t == self.K:
            self.K += 1
            mu, precision = self.sampleNewParams(t)
            self.mu[t] = mu
            self.precision[t] = precision
            self.logdet[t] = slogdet(precision)[1] 
            paramprob = self.loginvWishartPdf(precision, self.logdet[t])
            paramprob += self.mvNormalLL(mu, np.dot(mu, mu.transpose()), 1, self.con.mu0, self.con.kappa0 * precision, self.d * math.log(self.con.kappa0) + self.logdet[t])
            self.paramprobs[t] = paramprob
        ll = self.mvNormalLL(self.data[i][:,None], self.dd[i], 1, self.mu[t], self.precision[t], self.logdet[t])
        self.cluster_likelihood[t] += ll
                
        
    def sample(self, i):
        s = self.data[i][:,None]
        ss = self.dd[i]
        
        probs = []
        for t in range(self.K):
            prior = math.log(self.counts[t])
            ll = self.mvNormalLL(s, ss, 1, self.mu[t], self.precision[t], self.logdet[t])
            probs.append(ll + prior)
        if self.K < self.con.pruningfactor:
            prior = self.con.logalpha
            ll = self.integrated[i]
            probs.append(ll + prior)
        normed = logNormalize(probs)
        return sampleIndex(normed)    
    
    def logprob(self):
        ll = self.cluster_likelihood.sum()
        prior =  self.K * self.con.logalpha + self.countsgammaln.sum()
        base = self.paramprobs.sum()
        return prior, base, ll 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='infinite Gaussian Mixture Model')
    parser.add_argument('-D', '--data', help='data file name')
    parser.add_argument('-O', '--out', help='output file name')
    parser.add_argument('-V', '--vocab', help='vocabulary file name')
    parser.add_argument('-a', '--alpha', type=float, default=1.0, help='concentration parameter for DP prior')
    parser.add_argument('-L', '--Lambda', type=float, default=1.0, help='value for Inverse-Wishart scale matrix diagonal')
    parser.add_argument('-P', '--pruning', type=int, default=100, help="maximum number of clusters induced")
    parser.add_argument('-I', '--iter', type=int, default=100, help="number of Gibbs iterations")
    parser.add_argument('-k', '--kappa', type=float, default=0.1, help="number of pseudo-observations")
    args = parser.parse_args()

    data = np.load(args.data)
    mean = np.mean(data, axis=0)[:,None]
    vocab = open(args.vocab).read().split()

    con = Constants(data.shape[1], mean, args.alpha, args.Lambda, args.pruning, args.kappa)
    state = IGMMState(vocab, np.array(data), con)
    
    state.initialize()
    state.resampleParams()
    prior, base, ll = state.logprob()
    prob = prior + base + ll
    sys.stderr.write( "> iter 0:\t" + str(round(prior)) + '\t' + str(round(base)) + '\t' 
                      + str(round(ll)) + '\t' +str(round(prob)) + '\t' + str(state.numClusters()) + '\t' + state.histogram() + '\n')
    for i in range(args.iter):
        state.resampleData()
        state.resampleParams()
        
        if (i + 1) % 1 == 0:
            prior, base, ll = state.logprob()
            prob = prior + base + ll
            sys.stderr.write( "> iter " + str(i+1) + ":\t" + str(round(prior)) + '\t' + str(round(base)) + '\t' 
                      + str(round(ll)) + '\t' +str(round(prob)) + '\t' + str(state.numClusters()) + '\t' + state.histogram() + '\n')
        
    

    with open(args.out, 'w') as f:
        for i, item in enumerate(state.assignments):
            #f.write(vocab[i] + '\t' + str(item) + '\n')
            f.write(str(i) + '\t' + str(item) + '\n')

