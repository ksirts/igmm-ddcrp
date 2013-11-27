#! /usr/bin/env python
'''
Created on Aug 9, 2013

@author: Kairit
'''
from __future__ import division
import math, random
import numpy as np
from numpy.linalg import slogdet, cholesky
from scipy.special import gammaln
import optparse
import sys, time, subprocess
import scipy as sp

from common import Constants, MultivariateStudentT
from common import State
from common import sampleIndex, logsumexp
      

        
class DDCRPState(State):
    
    def __init__(self, vocab, data, con, dist):
        super(DDCRPState, self).__init__(vocab, data, con)
        if dist is not None:
            #self.prior = -dist / con.a + math.log(con.alpha) * np.identity(self.n)
            self.prior = np.log(dist + con.alpha * np.identity(self.n))
        else:
            self.prior = np.zeros((self.n, self.n)) + math.log(con.alpha) * np.identity(self.n)
        self.follow = -1 * np.ones(self.n, dtype=np.int)
        self.sit_behind = [set() for _ in xrange(self.n)]
        
        std_nu = self.con.nu0 - self.d + 1
        std_S = (self.con.kappa0 + 1) / (self.con.kappa0 * std_nu) * self.con.lambda0
        self.mvStudentT = MultivariateStudentT(self.d, std_nu, self.con.mu0, std_S)
        
    def initialize(self, rand, initfn=None):
        super(DDCRPState, self).initialize()

        self.temps = np.zeros(self.d, np.float)
        self.tempss = np.zeros((self.d, self.d), np.float)
        self.probs = np.zeros(self.n)

        if initfn is None:
            # Randomly initialize who follows whom
            for i in xrange(self.n):
                if rand:
                    n = i+1 if self.con.seq else self.n # Sequential or non-sequential CRP
                    if random.random() < self.con.alpha / (i + self.con.alpha):
                        following = i
                    else:
                        following = random.randrange(n)
                else:
                    following = i
                self.follow[i] = following
                self.sit_behind[following].add(i)
                self.dd[i] = np.outer(self.data[i], self.data[i])
                
            # Incrementally merge all indices that belong to the same cluster
            tablemap = dict(zip(range(self.n), [set([i]) for i in range(self.n)]))
            for i, ind in enumerate(self.follow):
                tablemap[i] |= tablemap[ind]
                for item in tablemap[ind]:
                    tablemap[item] = tablemap[i]
                    
            # Assign data to clusters
            for i in tablemap:
                if self.assignments[i] == -1:
                    cluster = tablemap[i]
                    t = self.K
                    for j in cluster:
                        self.assignments[j] = t
                        self.s[t] += self.data[j]
                        self.ss[t] += self.dd[j]
                    self.counts[t] += len(cluster)
                    self.K += 1
        else:
            with open(initfn) as f:
                for line in f:
                    word, tag = line.split()
                    i = self.vocab.index(word)
                    tag = int(tag)
                    self.assignments[i] = tag
                    self.counts[tag] += 1
                    self.dd[i] = np.outer(self.data[i], self.data[i])
                    self.s[tag] += self.data[i]
                    self.ss[tag] += self.dd[i]
                    self.K = max(self.K, tag+1)
            
            for t in range(self.K):
                inds = self.getIndices(t)
                firstind = inds[0]
                for ind in inds:
                    self.follow[ind] = firstind
                    self.sit_behind[firstind].add(ind)
                  
        
    def followSet(self, i):
        ret = set()
        self.sitBehind(i, ret)
        ret.add(i)
        return ret
    
            
    def sitBehind(self, i, ret):
        if len(self.sit_behind[i]) == 0:
            return
        lenret = len(ret)
        ret.update(self.sit_behind[i])
        if len(ret) == lenret:
            return
        for j in self.sit_behind[i]:
            if i != j:
                self.sitBehind(j, ret)
            
    def addItem(self, i, newf, oldf, newt, oldt, followset, s, ss):    
             
        if newt == -1:
            newt = self.K
            self.K += 1
            n = len(followset)
            self.counts[oldt] -= n
            self.counts[newt] += n
            assert self.counts[oldt] >= 0
            assert self.counts[newt] > 0
            self.s[oldt] -= s
            self.s[newt] += s
            self.ss[oldt] -= ss
            self.ss[newt] += ss
            mu, precision = self.sampleNewParams(newt)
            self.mu[newt] = mu
            self.precision[newt] = precision
            precdet = slogdet(precision)[1]
            self.logdet[newt] = precdet
            paramprob = self.param_probs(newt)
            self.paramprobs[newt] = paramprob
        elif newt != oldt:
            n = len(followset)
            self.counts[oldt] -= n
            self.counts[newt] += n
            assert self.counts[oldt] >= 0
            assert self.counts[newt] > 0
            self.s[oldt] -= s
            self.s[newt] += s
            self.ss[oldt] -= ss
            self.ss[newt] += ss
            
        for j in followset:
            self.assignments[j] = newt
        self.follow[i] = newf
        self.sit_behind[newf].add(i)

        if newt != oldt:                
            oldll = self.mvNormalLL(s, ss, n, self.mu[oldt], self.precision[oldt], self.logdet[oldt])
            newll = self.mvNormalLL(s, ss, n, self.mu[newt], self.precision[newt], self.logdet[newt])
            self.cluster_likelihood[oldt] -= oldll
            self.cluster_likelihood[newt] += newll
            
            if self.counts[oldt] == 0:
                self.K -= 1
                self.updateData(oldt)
        
            
    def removeItem(self, i, followset, s, ss):
        f = self.follow[i]
        t = self.assignments[i]
        self.sit_behind[f].remove(i)
        self.follow[i] = i
        if f not in followset:
            for j in followset:
                self.assignments[j] = -1
        return t, f
        

    def resampleData(self):
        permuted = range(self.n)
        random.shuffle(permuted)
        for i in permuted:
            followset = self.followSet(i)
            s = self.temps
            ss = self.tempss
            s.fill(0.)
            ss.fill(0.)
            for j in followset:
                s += self.data[j]
                ss += self.dd[j]
            
            oldt, oldf, = self.removeItem(i, followset, s, ss)
            newt, newf = self.sample(i, oldt, oldf, followset, s, ss)
            self.addItem(i, newf, oldf, newt, oldt, followset, s, ss)              
               

    def sample(self, i, oldt, oldf, followset, s, ss):      
        llcache = np.zeros(self.K)
        n = self.n
        if self.con.seq:
            n = i + 1

        probs = self.probs[:n]
        probs.fill(0.)
        
        newll = 0

        for j in xrange(n):
            prior = self.prior[i,j]
            
            t = self.assignments[j]
            
            if t == -1:
                ll = newll
                if ll == 0:
                    ll = newll = self.integrateOverParameters(len(followset), s, ss)
            else:
                ll = llcache[t]
                if ll == 0:
                    if t == oldt and self.counts[t] == len(followset):
                        ll = self.integrateOverParameters(len(followset), s, ss)
                    else:
                        ll = self.mvNormalLL(s, ss, len(followset), self.mu[t], self.precision[t], self.logdet[t])
                    llcache[t] = ll

            probs[j]  = prior + ll                           

        normed = np.exp(probs - logsumexp(probs))
        #np.testing.assert_allclose(normed, normed2)
        ind = sampleIndex(normed)
        table = self.assignments[ind]
        return table, ind
        
    def updateData(self, t):
        assert self.counts[t] == 0
        if t < self.K:
            self.mu[t] = self.mu[self.K]
            self.precision[t] = self.precision[self.K]
            self.logdet[t] = self.logdet[self.K]
            self.counts[t] = self.counts[self.K]
            assert self.counts[t] > 0
            self.cluster_likelihood[t] = self.cluster_likelihood[self.K]
            self.s[t] = self.s[self.K]
            self.ss[t] = self.ss[self.K]
            self.paramprobs[t] = self.paramprobs[self.K]
            
            self.assignments = np.array([t if x==self.K else x for x in self.assignments])
                
        self.mu[self.K].fill(0.)
        self.precision[self.K].fill(0.)
        self.logdet[self.K] = 0.0
        self.counts[self.K] = 0
        self.cluster_likelihood[self.K] = 0
        self.s[self.K].fill(0.)
        self.ss[self.K].fill(0.)
        self.paramprobs[self.K] = 0.
                
    def prior_prob(self):
        prior = 0.0
        for i, j in enumerate(self.follow):
            prior += self.prior[i, j]
        return prior    
      
    
    def igmm_prior(self):
        prior =  self.K * math.log(self.con.alpha)
        for t in range(self.K):
            prior += gammaln(self.counts[t])
        return prior

    
class DDCRPStateIntegrated(DDCRPState):
    def initialize(self, rand, init):
        super(DDCRPStateIntegrated, self).initialize(rand, init)
        
        for t in range(self.K):
            #self.denom[t] = self.integrateOverParameters(self.counts[t], self.s[t], self.ss[t])
            n = self.counts[t]
            kappan = self.con.kappa0 + n
            mun = (self.con.kappa0 * self.con.mu0 + self.s[t]) / kappan
            #lambdan_part = self.con.lambda0 + self.ss[t] + self.con.kappa0_outermu0
            lambdan = self.con.lambda0 + self.ss[t] + self.con.kappa0_outermu0 - kappan * np.outer(mun, mun)
            # Don't remove the last part from cholesky, because when updating the composition, we would have to add it anyway
            # in order to replace it with the updated part
            #chol = cholesky(lambdan_part).T
            self.logdet[t] = slogdet(lambdan)[1]
            #self.cholesky[t] = chol
            
            
    def removeItem(self, i, followset, s, ss):
        f = self.follow[i]
        t = self.assignments[i]
        self.sit_behind[f].remove(i)
        self.follow[i] = i
        self.s[t] -= s
        self.ss[t] -= ss
        self.counts[t] -= len(followset)
        if self.counts[t] > 0:
            #self.denom[t] = self.integrateOverParameters(self.counts[t], self.s[t], self.ss[t])
            n = self.counts[t]
            kappan = self.con.kappa0 + n
            mun = (self.con.kappa0 * self.con.mu0 + self.s[t]) / kappan
            lambdan = self.con.lambda0 + self.ss[t] + self.con.kappa0_outermu0 - kappan * np.outer(mun, mun)
            self.logdet[t] = slogdet(lambdan)[1]
            #for ind in followset:
            #    choldowndate(self.cholesky[t], self.data[ind].copy())
        for j in followset:
            self.assignments[j] = -1
        if self.counts[t] == 0:
            self.removeCluster(t)
        return t, f
    
    def addItem(self, i, newf, oldf, newt, oldt, followset, s, ss):    
             
        if newt == -1:
            newt = self.K
            self.K += 1
            self.cholesky[newt] = cholesky(self.con.lambda0 + self.con.kappa0_outermu0).T
            
        n = len(followset)
        self.counts[newt] += n
        assert self.counts[oldt] >= 0
        assert self.counts[newt] > 0
        self.s[newt] += s
        self.ss[newt] += ss
        #self.denom[newt] = self.integrateOverParameters(self.counts[newt], self.s[newt], self.ss[newt])

        n = self.counts[newt]
        kappan = self.con.kappa0 + n
        mun = (self.con.kappa0 * self.con.mu0 + self.s[newt]) / kappan
        lambdan = self.con.lambda0 + self.ss[newt] + self.con.kappa0_outermu0 - kappan * np.outer(mun, mun)
        self.logdet[newt] = slogdet(lambdan)[1]
        #for ind in followset:
        #    cholupdate(self.cholesky[newt], self.data[ind].copy())
            
        for j in followset:
            self.assignments[j] = newt
        self.follow[i] = newf
        self.sit_behind[newf].add(i)
        
    def removeCluster(self, t):
        assert self.counts[t] == 0
        self.K -= 1
        if t != self.K:
            self.counts[t] = self.counts[self.K]
            assert self.counts[t] > 0
            self.s[t] = self.s[self.K]
            self.ss[t] = self.ss[self.K]
            #self.denom[t] = self.denom[self.K]
            #self.cholesky[t] = self.cholesky[self.K]
            self.logdet[t] = self.logdet[self.K]
            self.assignments = np.array([t if x==self.K else x for x in self.assignments])
                

        self.counts[self.K] = 0
        self.s[self.K].fill(0.)
        self.ss[self.K].fill(0.)
        #self.denom[self.K] = 0.
        #self.cholesky[self.K].fill(0.)
        self.logdet[self.K] = 0.
        
    def sample(self, i, oldt, oldf, followset, s, ss):      
        llcache = np.zeros(self.K)
        n = self.n
        if self.con.seq:
            n = i + 1

        probs = self.probs[:n]
        probs.fill(0.)
        
        newll = 0

        for j in xrange(n):
            prior = self.prior[i,j]
            
            t = self.assignments[j]
            
            if t == -1:
                ll = newll
                if ll == 0:
                    ll = newll = self.integrateOverParameters(len(followset), s, ss)
            else:
                ll = llcache[t]
                if ll == 0:
                    ll = self.posteriorPredictive(followset, s, ss, t)
                    llcache[t] = ll

            probs[j]  = prior + ll                           

        normed = np.exp(probs - logsumexp(probs))
        #np.testing.assert_allclose(normed, normed2)
        ind = sampleIndex(normed)
        table = self.assignments[ind]
        return table, ind
    
       
    def resampleParams(self):
        pass
       

if __name__ == '__main__':
    
    start = time.clock()
    parser = optparse.OptionParser(description='infinite Gaussian Mixture Model')
    parser.add_option('-D', '--data', help='data file name')
    parser.add_option('-O', '--out', default="hypothesis", help='output file name')
    parser.add_option('-V', '--vocab', help='vocabulary file name')
    parser.add_option('-a', '--alpha', type=float, default=1.0, help='concentration parameter for DP prior')
    parser.add_option('-L', '--Lambda', type=float, default=0.0, help='value for Inverse-Wishart scale matrix diagonal')
    parser.add_option('-P', '--pruning', type=int, default=100, help="maximum number of clusters induced")
    parser.add_option('-b', type=float, default=1.0, help="exponential distribution parameter for distance prior")
    parser.add_option('-I', '--iter', type=int, default=100, help="number of Gibbs iterations")
    parser.add_option('-k', '--kappa', type=float, default=0.1, help="number of pseudo-observations")
    parser.add_option('-d', '--dist', help='distance matrix file name')
    parser.add_option('-i', '--init', help='file with initialized data, one word per line, word and tag separated with tab')
    parser.add_option('-t', '--threshold', type=float, default=-10, help="sample only from the elements whose prior exceeds this threshold")
    parser.add_option('-S', '--seq', action="store_true", help="if set, then use sequential CRP")
    parser.add_option('-R', '--rand', action="store_true", help="if set then initialize the followings randomly, otherwise initialize everybody to follow itself")
    parser.add_option('-E', '--explicit', action='store_true', help="if set, then sample explicit cluster parameters")
    parser.add_option('-T', '--trace', help="name of the trace file")
    parser.add_option('-s', '--stats', action="store_true", help="when set then show number of clusters and cluster histogram in trace")
    parser.add_option('-e', '--evalscript', help="path to the evaluation script")
    parser.add_option('-g', '--gold', help="path to the goldstandard file to be used in evaluation script")
    parser.add_option('-r', '--result', help="path to the file where the evaluation results will be written")
    (args, opts) = parser.parse_args()
    
    print
    print "data file\t:", args.data
    print "output file\t:", args.out
    print "vocabulary file\t:", args.vocab
    print "distance file\t:", args.dist
    print "init file\t:", args.init
    print
    print "alpha\t\t=", args.alpha
    print "lambda\t\t=", args.Lambda
    print "pruning\t\t=", args.pruning
    print "iter\t\t=", args.iter
    print "prior th\t=", args.threshold
    print "b\t\t=", args.b
    print "kappa\t\t=", args.kappa
    print
    print "sequential\t:", args.seq
    print "random init\t:", args.rand
    print
    
    data = np.load(args.data)
    if args.vocab:
        vocab = open(args.vocab).read().split()
    else:
        vocab = map(str, range(data.shape[0]))

    mean = np.mean(data, axis=0)
    if args.dist is not None:
        dist = np.load(args.dist)
    else:
        dist = None
    
    pruning = args.pruning
    if pruning == -1:
        pruning = len(vocab)
        
    if args.Lambda == 0:
        n, d = data.shape
        lambdaprior = np.zeros((d, d))
        for i in xrange(n):
            lambdaprior += np.outer(data[i], data[i])
        lambdaprior = sp.diag(sp.diag(lambdaprior) / n)
    else:
        d = data.shape[1]
        lambdaprior = np.identity(d, np.float) * args.Lambda
    
    con = Constants(data.shape[1], mean, args.alpha, lambdaprior, pruning, args.kappa, args.b, args.threshold, args.seq)
    if args.explicit:
        state = DDCRPState(vocab, data, con, dist)
    else:
        state = DDCRPStateIntegrated(vocab, data, con, dist)
    
    if args.trace is None:
        tracef = sys.stdout
    else:
        tracef = open(args.trace,'w')
    
    state.initialize(args.rand, args.init)
    state.resampleParams()

    prior, baseprob, likelihood, likelihood_int = state.prior_prob(), state.param_probabilites(), state.likelihood(), state.likelihood_int()
    prob = prior + baseprob + likelihood
    prob_int = prior + likelihood_int
    igmm_prior = state.igmm_prior()
    igmm_prob = igmm_prior + baseprob + likelihood
    igmm_int_prob = igmm_prior +likelihood_int
    
    elapsed = time.clock() - start
    
    tracef.write("iter\ttime\tprior\t")
    if args.explicit:
        tracef.write("params\t")
    tracef.write("llhood\ttotal\t")
    if args.explicit:
        tracef.write("igmm\t")
    tracef.write("igmm_int")
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
    else:
        tracef.write('\t' + str(round(prob_int)))
    if args.explicit:
        tracef.write('\t' + str(round(igmm_prob)))
    tracef.write('\t' + str(round(igmm_int_prob)))
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
            prob_int = prior + likelihood_int
            igmm_prior = state.igmm_prior()
            igmm_prob = igmm_prior + baseprob + likelihood
            igmm_int_prob = igmm_prior + likelihood_int
            
            if args.trace is None:
                tracef = sys.stdout
            else:
                tracef = open(args.trace, 'a')
            elapsed = time.clock() - start
            tracef.write("> " + str(i+1) + ":\t" + str(elapsed) + '\t' + str(round(prior)))
            if args.explicit:
                tracef.write('\t' + str(round(baseprob)) + '\t' + str(round(likelihood)))
            else:
                tracef.write('\t' + str(round(likelihood_int)))
            if args.explicit:
                tracef.write('\t' + str(round(prob)))
            else:
                tracef.write('\t' + str(round(prob_int)))
            if args.explicit:
                tracef.write('\t' + str(round(igmm_prob)))
            tracef.write('\t' + str(round(igmm_int_prob)))
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

    

