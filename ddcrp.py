'''
Created on Aug 9, 2013

@author: Kairit
'''
from __future__ import division
import math, random
import numpy as np
from numpy.linalg import slogdet
from scipy.special import gammaln
import argparse
import sys
from scipy.misc import logsumexp
from collections import Counter

from common import Constants
from common import State
from common import sampleIndex
         

        
class DDCRPState(State):
    
    def __init__(self, vocab, data, con, dist):
        super(DDCRPState, self).__init__(vocab, data, con)
        if dist is not None:
            self.prior = -dist / con.a + math.log(con.alpha) * np.identity(self.n)
        else:
            self.prior = np.zeros((self.n, self.n)) + math.log(con.alpha) * np.identity(self.n)
        self.follow = -1 * np.ones(self.n, dtype=np.int)
        self.sit_behind = [set() for _ in xrange(self.n)]
        
    def initialize(self, rand, initfn=None):
        super(DDCRPState, self).initialize()

        self.temps = np.zeros((self.d, 1), np.float)
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
                item = self.data[i][None,:]
                self.dd[i] = np.dot(item.T, item)
                
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
                        item = self.data[j][:,None]
                        self.s[t] += item
                        self.ss[t] += np.dot(item, item.T)
                    self.counts[t] += len(cluster)
                    self.K += 1
        else:
            with open(initfn) as f:
                i = 0
                for line in f:
                    _, tag = line.split()
                    tag = int(tag)
                    self.assignments[i] = tag
                    self.counts[tag] += 1
                    self.integrated[i] = self.mvStudentT(self.data[i])
                    i += 1
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
            mu, precision = self.sampleNewParams(newt)
            self.mu[newt] = mu
            self.precision[newt] = precision
            precdet = slogdet(precision)[1]
            self.logdet[newt] = precdet
            paramprob = self.loginvWishartPdf(precision, precdet)
            paramprob += self.mvNormalLL(mu, np.dot(mu, mu.T), 1, self.con.mu0, self.con.kappa0 * precision, self.d * math.log(self.con.kappa0) + precdet)
            self.paramprobs[newt] = paramprob
            
        for j in followset:
            self.assignments[j] = newt
        self.follow[i] = newf
        self.sit_behind[newf].add(i)

        if newt != oldt:
            n = len(followset)
            self.counts[oldt] -= n
            self.counts[newt] += n
            assert self.counts[oldt] >= 0
            assert self.counts[newt] > 0
            self.s[oldt] -= s
            self.s[newt] += s
            self.ss[oldt] -= ss
            self.ss[newt] += ss
                
            oldll = self.mvNormalLL(s, ss, n, self.mu[oldt], self.precision[oldt], self.logdet[oldt])
            newll = self.mvNormalLL(s, ss, n, self.mu[newt], self.precision[newt], self.logdet[newt])
            self.cluster_likelihood[oldt] -= oldll
            self.cluster_likelihood[newt] += newll
            
            if self.counts[oldt] == 0:
                self.K -= 1
                self.updateData(oldt)
        '''    
        if newtable:
            table = self.K
            self.K += 1

        elif newtable:
            mu, precision = self.sampleNewParams(table)
            self.mu[table] = mu
            self.precision[table] = precision
            self.logdet[table] = slogdet(precision)[1]
            ll = self.mvNormalLL(self.s[table], self.ss[table], self.counts[table], mu, precision, self.logdet[table])
            self.cluster_likelihood[table] = ll
            self.paramprobs[table] = self.logWishartPdf(precision) + \
                    self.mvNormalLL(mu, np.dot(mu, mu.transpose()), 1, self.con.mu0, self.con.kappa0 * precision, self.d * math.log(self.con.kappa0) + self.logdet[table])
            ll = self.mvNormalLL(s, ss, len(followset), self.mu[oldtable], self.precision[oldtable], self.logdet[oldtable])
            self.cluster_likelihood[oldtable] -= ll
        elif table != oldtable:
            ll = self.mvNormalLL(s, ss, len(followset), self.mu[oldtable], self.precision[oldtable], self.logdet[oldtable])
            self.cluster_likelihood[oldtable] -= ll
            ll = self.mvNormalLL(s, ss, len(followset), self.mu[table], self.precision[table], self.logdet[table])
            self.cluster_likelihood[table] += ll
     '''   
        
            
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
        if not self.con.seq:
            random.shuffle(permuted)
        for i in permuted:
            followset = self.followSet(i)
            s = self.temps
            ss = self.tempss
            s.fill(0.)
            ss.fill(0.)
            for j in followset:
                s += self.data[j][:,None]
                ss += self.dd[j]
            
            oldt, oldf, = self.removeItem(i, followset, s, ss)
            newt, newf = self.sample(i, oldt, oldf, followset, s, ss)
            self.addItem(i, newf, oldf, newt, oldt, followset, s, ss)
                
                
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
               
    # 95% of cumulative time, including 40% inside here alone. can probably avoid reallocated memory, wonder if that would help.
    def sample(self, i, oldt, oldf, followset, s, ss):      
        llcache = np.zeros(self.K)
        n = self.n
        if self.con.seq:
            n = i + 1

        probs = self.probs[:n]
        probs.fill(0.)

        for j in xrange(n):
            prior = self.prior[i,j]
            
            t = self.assignments[j]
            #newtable = (j in followset and oldt not in followset)
            #jointable = (t != oldt)
            ll = 0
            tables = set()
            lls = Counter()
            for k, table in enumerate(self.assignments):
                if k not in followset:
                    l= self.mvNormalLL(self.data[k][:,None], self.dd[k], 1, self.mu[table], self.precision[table], self.logdet[table])
                    lls[table] += l
                    ll += l
                    if table not in tables:
                        tables.add(table)
                        param = self.loginvWishartPdf(self.precision[table], self.logdet[table])
                        param += self.mvNormalLL(self.mu[table], np.dot(self.mu[table], self.mu[table].T), 1, self.con.mu0, self.con.kappa0 * self.precision[table], self.d * math.log(self.con.kappa0) + self.logdet[table])
                        ll += param
            if t == -1:
                ll += self.integrateOverParameters(len(followset), s, ss)
            else:
                l = self.mvNormalLL(s, ss, len(followset), self.mu[t], self.precision[t], self.logdet[t])
                lls[t] += l
                ll += l
                if t not in tables:
                    param = self.loginvWishartPdf(self.precision[t], self.logdet[t])
                    param += self.mvNormalLL(self.mu[t], np.dot(self.mu[t], self.mu[t].T), 1, self.con.mu0, self.con.kappa0 * self.precision[t], self.d * math.log(self.con.kappa0) + self.logdet[t])
                    ll += param                           
            probs[j] = prior + ll
            '''
            jointable = (t != oldtable)
            if newtable and (self.K == self.con.pruningfactor):
                probs[j] = float('-inf')
            elif newtable:
                if newtableLL == 0.0:
                    newtableLL = self.integrateOverParameters(followset, s, ss) - oldll
                #import pdb; pdb.set_trace()
                probs[j] = newtableLL + self.prior[i,j]
            elif jointable:
                ll = llcache[t]
                if ll == 0:
                    #ll = self.likelihoodF_Jacob(followset, t)
                    #l = self.likelihoodF2(followset, t, s, ss)
                    newll = self.mvNormalLL(s, ss, len(followset), self.mu[t], self.precision[t], self.logdet[t])
                    ll = newll - oldll
                    if self.counts[oldtable] == len(followset):
                        ll -= self.paramprobs[oldtable]
                    llcache[t] = ll
                probs[j] = ll + self.prior[i,j]
            else:
                probs[j] = self.prior[i, j]
            '''
        normed = np.exp(probs - logsumexp(probs))
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
                
        
      

    def logprob(self):
        likelihood = self.cluster_likelihood.sum()
        prior = 0.0
        for i, j in enumerate(self.follow):
            prior += self.prior[i, j]
        baseprob = self.paramprobs.sum()
        return prior, baseprob, likelihood 
    
    def igmm_logprob(self):
        #self.resampleParams()
        prior =  self.K * math.log(self.con.alpha)
        assert not math.isinf(prior)
        for t in range(self.K):
            prior += gammaln(self.counts[t])
        baseprob = 0.0
        likelihood = 0.0
        for t in range(self.K):
            # Should we compute here posterior probability with updated hyperparameters?
            # But this does not conform with the generative story
            baseprob += self.logWishartPdf(self.precision[t])
            mu = self.mu[t][:,None]
            baseprob += self.mvNormalLL(mu, np.dot(mu, mu.T), 1, self.con.mu0, self.precision[t], self.logdet[t])
            likelihood += self.mvNormalLL(self.s[t], self.ss[t], self.counts[t], mu, self.precision[t], self.logdet[t])
        assert not math.isinf(baseprob)
        return prior, baseprob, likelihood
    
    def numClusters(self):
        return self.K
    
    def histogram(self):
        return ' '.join(map(str, self.counts[:self.K]))
       

if __name__ == '__main__':
    
    #random.seed(1)
    #np.random.seed(1)
    parser = argparse.ArgumentParser(description='infinite Gaussian Mixture Model')
    parser.add_argument('-D', '--data', help='data file name')
    parser.add_argument('-O', '--out', default="hypothesis", help='output file name')
    parser.add_argument('-V', '--vocab', help='vocabulary file name')
    parser.add_argument('-a', '--alpha', type=float, default=1.0, help='concentration parameter for DP prior')
    parser.add_argument('-L', '--Lambda', type=float, default=1.0, help='value for Inverse-Wishart scale matrix diagonal')
    parser.add_argument('-P', '--pruning', type=int, default=100, help="maximum number of clusters induced")
    parser.add_argument('-b', type=float, default=1.0, help="exponential distribution parameter for distance prior")
    parser.add_argument('-I', '--iter', type=int, default=100, help="number of Gibbs iterations")
    parser.add_argument('-k', '--kappa', type=float, default=0.1, help="number of pseudo-observations")
    parser.add_argument('-d', '--dist', help='distance matrix file name')
    parser.add_argument('-i', '--init', help='file with initialized data, one word per line, word and tag separated with tab')
    parser.add_argument('-t', '--threshold', type=float, default=-10, help="sample only from the elements whose prior exceeds this threshold")
    parser.add_argument('-S', '--seq', action="store_true", help="if set, then use sequential CRP")
    parser.add_argument('-R', '--rand', action="store_true", help="if set then initialize the followings randomly, otherwise initialize everybody to follow itself")
    args = parser.parse_args()
    
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
        vocab = range(data.shape[0])

    mean = np.mean(data, axis=0)[:,None]
    if args.dist is not None:
        dist = np.load(args.dist)
    else:
        dist = None
    
    pruning = args.pruning
    if pruning == -1:
        pruning = len(vocab)
    
    con = Constants(data.shape[1], mean, args.alpha, args.Lambda, pruning, args.kappa, args.b, args.threshold, args.seq)
    state = DDCRPState(vocab, data, con, dist)
    
    state.initialize(args.rand, args.init)
    state.resampleParams()

    prior, baseprob, likelihood = state.logprob()
    prob = prior + baseprob +  likelihood
    sys.stderr.write( "> iter 0:\t" +  str(round(prior)) + '\t' + str(round(baseprob)) + '\t' + str(round(likelihood)) + '\t' + str(round(prob)) + '\t' + str(state.numClusters()) + '\t' + state.histogram() + '\n')
    for i in range(args.iter):
        state.resampleData()
        state.resampleParams()
        if (i + 1) % 1 == 0:
            prior, baseprob, likelihood = state.logprob()
            prob = prior + baseprob + likelihood
            sys.stderr.write("> iter " + str(i+1) +":\t" + str(round(prior)) + '\t' + str(round(baseprob)) + '\t' + str(round(likelihood)) + '\t' + str(round(prob)) + '\t' + str(state.numClusters()) + '\t' + state.histogram() + '\n')

    with open(args.out, 'w') as f:
        for i, item in enumerate(state.assignments):
            f.write(vocab[i] + '\t' + str(item) + '\n')
            #f.write(str(i) + '\t' + str(item) + '\n')
    

