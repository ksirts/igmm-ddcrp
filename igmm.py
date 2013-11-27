#!/usr/bin/env python

'''
Created on Aug 9, 2013

@author: Kairit
'''
from __future__ import division

import math, random
import numpy as np
import scipy as sp
from numpy.linalg import slogdet
import optparse
import sys, time, subprocess

from common import MultivariateStudentT, Constants
from common import State
from common import logNormalize, sampleIndex


        
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
            self.s[ind] += self.data[i]
            self.dd[i] = np.outer(self.data[i], self.data[i])
            self.ss[ind] += self.dd[i]
            self.integrated[i] = self.mvStudentT(self.data[i])
 
        self.probs = np.zeros(self.con.pruningfactor)    
        
            
    def resampleData(self):
        permuted = range(self.n)
        random.shuffle(permuted)
        for i in permuted:
            self.removeItem(i)
            assert self.counts.sum() == self.n - 1
            newt = self.sample(i)
            self.addItem(i, newt)
            assert self.counts.sum() == self.n

                
    def removeItem(self, i):
        t = self.assignments[i]
        self.s[t] -= self.data[i]
        self.ss[t] -= self.dd[i]
        self.counts[t] -= 1
        assert self.counts[t] >= 0
        ll = self.mvNormalLL(self.data[i], self.dd[i], 1, self.mu[t], self.precision[t], self.logdet[t])
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
        self.logdet[self.K] = 0.
        self.s[self.K].fill(0)
        self.ss[self.K].fill(0.)
        self.counts[self.K] = 0.
        self.countsgammaln[self.K] = 0.
        self.cluster_likelihood[self.K] = 0.
        self.paramprobs[self.K] = 0.
        
    
    def addItem(self, i, t):
        self.s[t] += self.data[i]
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
            paramprob = self.param_probs(t)
            self.paramprobs[t] = paramprob
        ll = self.mvNormalLL(self.data[i], self.dd[i], 1, self.mu[t], self.precision[t], self.logdet[t])
        self.cluster_likelihood[t] += ll
                
        
    def sample(self, i):
        s = self.data[i]
        ss = self.dd[i]
        
        probs = []
        for t in range(self.K):
            prior = math.log(self.counts[t])
            
            ll = self.mvNormalLL(s, ss, 1, self.mu[t], self.precision[t], self.logdet[t])
            probs.append(prior + ll)
        if self.K < self.con.pruningfactor:
            prior = self.con.logalpha

            ll = self.integrated[i]     
            probs.append(prior + ll)
        normed = logNormalize(probs)
        return sampleIndex(normed)    
    
    def prior_prob(self):
        return self.K * self.con.logalpha + self.countsgammaln.sum()

    
class IGMMStateIntegrated(IGMMState):
    
    def initialize(self):
        super(IGMMStateIntegrated, self).initialize()
        
        for t in range(self.K):
            #self.denom[t] = self.integrateOverParameters(self.counts[t], self.s[t], self.ss[t])
            n = self.counts[t]
            kappan = self.con.kappa0 + n
            mun = (self.con.kappa0 * self.con.mu0 + self.s[t]) / kappan
            lambdan = self.con.lambda0 + self.ss[t] + self.con.kappa0_outermu0 - kappan * np.outer(mun, mun)
            self.logdet[t] = slogdet(lambdan)[1]
        
    
    def removeItem(self, i):
        t = self.assignments[i]
        self.s[t] -= self.data[i]
        self.ss[t] -= self.dd[i]
        self.counts[t] -= 1
        assert self.counts[t] >= 0
        if self.counts[t] > 0:
            self.countsgammaln[t] -= math.log(self.counts[t])
            n = self.counts[t]
            kappan = self.con.kappa0 + n
            mun = (self.con.kappa0 * self.con.mu0 + self.s[t]) / kappan
            lambdan = self.con.lambda0 + self.ss[t] + self.con.kappa0_outermu0 - kappan * np.outer(mun, mun)
            self.logdet[t] = slogdet(lambdan)[1]

        #self.denom[t] = self.integrateOverParameters(self.counts[t], self.s[t], self.ss[t])

        if self.counts[t] == 0:
            self.removeCluster(t)
            
    def removeCluster(self, t):
        self.K -= 1
        if t != self.K:
            self.s[t] = self.s[self.K]
            self.ss[t] = self.ss[self.K]
            self.counts[t] = self.counts[self.K]
            self.countsgammaln[t] = self.countsgammaln[self.K]
            self.logdet[t] = self.logdet[self.K]
            self.assignments = [t if x==self.K else x for x in self.assignments]
        self.s[self.K].fill(0)
        self.ss[self.K].fill(0.)
        self.counts[self.K] = 0.
        self.countsgammaln[self.K] = 0.
        self.logdet[self.K] = 0.
        
    def addItem(self, i, t):
        self.s[t] += self.data[i]
        self.ss[t] += self.dd[i]
        if self.counts[t] > 0:
            self.countsgammaln[t] += math.log(self.counts[t])
        self.counts[t] += 1
        self.assignments[i] = t
        
        n = self.counts[t]
        kappan = self.con.kappa0 + n
        mun = (self.con.kappa0 * self.con.mu0 + self.s[t]) / kappan
        lambdan = self.con.lambda0 + self.ss[t] + self.con.kappa0_outermu0 - kappan * np.outer(mun, mun)
        self.logdet[t] = slogdet(lambdan)[1]
        
        if t == self.K:
            self.K += 1
            
            
    def sample(self, i):
        probs = []
        for t in range(self.K):
            prior = math.log(self.counts[t])
            ll = self.posteriorPredictive(set([i]), self.data[i], self.dd[i], t)
            probs.append(prior + ll)
        if self.K < self.con.pruningfactor:
            prior = self.con.logalpha

            ll = self.integrated[i]     
            probs.append(prior + ll)
        normed = logNormalize(probs)
        return sampleIndex(normed) 
    

    def resampleParams(self):
        pass
            

    

if __name__ == '__main__':
    start = time.clock()
    parser = optparse.OptionParser(description='infinite Gaussian Mixture Model')
    parser.add_option('-D', '--data', help='data file name')
    parser.add_option('-O', '--out', help='output file name')
    parser.add_option('-V', '--vocab', help='vocabulary file name')
    parser.add_option('-a', '--alpha', type=float, default=1.0, help='concentration parameter for DP prior')
    parser.add_option('-L', '--Lambda', type=float, default=0.0, help='value for Inverse-Wishart scale matrix diagonal')
    parser.add_option('-P', '--pruning', type=int, default=100, help="maximum number of clusters induced")
    parser.add_option('-I', '--iter', type=int, default=100, help="number of Gibbs iterations")
    parser.add_option('-k', '--kappa', type=float, default=0.01, help="number of pseudo-observations")
    parser.add_option('-E', '--explicit', action='store_true', help="if set, then sample explicit cluster parameters")
    parser.add_option('-T', '--trace', help="name of the trace file")
    parser.add_option('-s', '--stats', action="store_true", help="when set then show number of clusters and cluster histogram in trace")
    parser.add_option('-e', '--evalscript', help="path to the evaluation script")
    parser.add_option('-g', '--gold', help="path to the goldstandard file to be used in evaluation script")
    parser.add_option('-r', '--result', help="path to the file where the evaluation results will be written")
    (args, posit) = parser.parse_args()
    
    data = np.load(args.data)
    mean = np.mean(data, axis=0)
    if args.vocab:
        vocab = open(args.vocab).read().split()
    else:
        vocab = map(str, range(data.shape[0]))
        
    if args.Lambda == 0:
        n, d = data.shape
        lambdaprior = np.zeros((d, d))
        for i in xrange(n):
            lambdaprior += np.outer(data[i], data[i])
        lambdaprior = sp.diag(sp.diag(lambdaprior) / n)
    else:
        d = data.shape[1]
        lambdaprior = np.identity(d, np.float) * args.Lambda

    con = Constants(data.shape[1], mean, args.alpha, lambdaprior, args.pruning, args.kappa)
    if args.explicit:
        state = IGMMState(vocab, data, con)
    else:
        state = IGMMStateIntegrated(vocab, data, con)
        
    if args.trace is None:
        tracef = sys.stdout
    else:
        tracef = open(args.trace, 'w')
    
    state.initialize()
    state.resampleParams()
        
    prior, baseprob, likelihood, likelihood_int = state.prior_prob(), state.param_probabilites(), state.likelihood(), state.likelihood_int()
    prob = prior + baseprob + likelihood
    prob_int = prior +likelihood_int
    
    elapsed = time.clock() - start
    
    tracef.write("iter\ttime\tprior\t")
    if args.explicit:
        tracef.write("params\t")
    tracef.write("llhood\ttotal\t")
    if args.explicit:
        tracef.write("total_int")
    if args.stats:
        tracef.write('\t# C\thistogram')
    tracef.write('\n')
    tracef.write( "> 0:\t" +  str(elapsed) + '\t' + str(round(prior)))
    if args.explicit:
        tracef.write('\t' + str(round(baseprob)) + '\t' + str(round(likelihood)))
    else:
        tracef.write('\t' + str(round(likelihood_int)))
    if args.explicit:
        tracef.write('\t' + str(round(prob)))
    tracef.write('\t' + str(round(prob_int)))
    if args.stats:
        tracef.write('\t' + str(state.numClusters()) + '\t' + state.histogram())
    tracef.write('\n')
    if args.trace is not None:
        tracef.close() 
        
    if args.evalscript is not None and args.gold is not None and args.result is not None:
        with open(args.out, 'w') as f:
            for i, item in enumerate(state.assignments):
                f.write(vocab[i] + '\t' + str(item) + '\n')
        parg = [args.evalscript, args.gold, args.out]
        p = subprocess.Popen(parg, stdout=subprocess.PIPE)
        res = p.communicate()[0]
        with open(args.result, 'w') as f:
            f.write(res) 
        
    for i in range(args.iter):
        state.resampleData()
        state.resampleParams()
        
        if (i + 1) % 1 == 0:
            prior, baseprob, likelihood, likelihood_int = state.prior_prob(), state.param_probabilites(), state.likelihood(), state.likelihood_int()
            prob = prior + baseprob + likelihood
            prob_int = prior +likelihood_int
            if args.trace is None:
                tracef = sys.stdout
            else:
                tracef = open(args.trace, 'a')
            elapsed = time.clock() - start
            tracef.write( "> " + str(i + 1) + ":\t" +  str(elapsed) + '\t' + str(round(prior)))
            if args.explicit:
                tracef.write('\t' + str(round(baseprob)) + '\t' + str(round(likelihood)))
            else:
                tracef.write('\t' + str(round(likelihood_int)))
            if args.explicit:
                tracef.write('\t' + str(round(prob)))
            tracef.write('\t' + str(round(prob_int)))
            if args.stats:
                tracef.write('\t' + str(state.numClusters()) + '\t' + state.histogram())
            tracef.write('\n')
            if args.trace is not None:
                tracef.close() 
                
            if args.evalscript is not None and args.gold is not None and args.result is not None:
                with open(args.out, 'w') as f:
                    for i, item in enumerate(state.assignments):
                        f.write(vocab[i] + '\t' + str(item) + '\n')
                parg = [args.evalscript, args.gold, args.out]
                p = subprocess.Popen(parg, stdout=subprocess.PIPE)
                res = p.communicate()[0]
                with open(args.result, 'a') as f:
                    f.write(res)
        
    with open(args.out, 'w') as f:
        for j, item in enumerate(state.assignments):
            f.write(vocab[j] + '\t' + str(item) + '\n')

