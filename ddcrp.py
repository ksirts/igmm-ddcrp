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
import optparse, cPickle
import sys, time, subprocess
from scipy.optimize import fmin_l_bfgs_b


from common import Constants, MultivariateStudentT
from common import State
from common import sampleIndex, logsumexp
      

        
class DDCRPState(State):
    
    def __init__(self, vocab, data, con, dist, featfn, mdist):
        super(DDCRPState, self).__init__(vocab, data, con)
        if dist is not None:
            self.dist = dist #* con.a + math.log(con.alpha) * np.identity(self.n)           
        else:
            self.prior = np.zeros((self.n, self.n)) + con.logalpha * np.identity(self.n)
        if mdist is not None:
            self.mdist = mdist
        else:
            self.mdist = np.zeros((self.n, self.n))
        self.prior = np.zeros((self.n, self.n))
        self.prepareSuffixes(featfn)
        self.featsetvals = np.zeros(len(self.featsets))
        
        std_nu = self.con.nu0 - self.d + 1
        std_S = (self.con.kappa0 + 1) / (self.con.kappa0 * std_nu) * self.con.lambda0
        self.mvStudentT = MultivariateStudentT(self.d, std_nu, self.con.mu0, std_S)
              
       
    def initialize(self, rand, initfn=None, follower=False):
        super(DDCRPState, self).initialize()

        self.temps = np.zeros(self.d, np.float)
        self.tempss = np.zeros((self.d, self.d), np.float)
        self.probs = np.zeros(self.n)
        self.follow = -1 * np.ones(self.n, dtype=np.int)
        self.sit_behind = [set() for _ in xrange(self.n)]
        self.assignments = -1 * np.ones(self.n, dtype=np.int)
        
        for i in xrange(self.n):
            self.dd[i] = np.outer(self.data[i], self.data[i])

        if initfn is None:
            # Randomly initialize who follows whom
            for i in xrange(self.n):
                if i % 100 == 0:
                    print i
                if rand:
                    n = i+1 if self.con.seq else self.n # Sequential or non-sequential CRP
                    if random.random() < self.con.alpha / (i + self.con.alpha):
                        following = i
                    else:
                        maxsim = min(max(self.dist[i]), 3)
                        inds = [j for j in range(self.n) if self.dist[i, j] >= maxsim]
                        if len(inds) == 0:
                            inds = range(self.n)
                        following = random.sample(inds, 1)[0]
                else:
                    following = i
                self.follow[i] = following
                self.sit_behind[following].add(i)
            
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
            if follower is False:
                clustering = {}
                with open(initfn) as f:
                    for line in f:
                        word, tag = line.split()
                        i = self.vocab.index(word)
                        tag = int(tag)
                        if tag not in clustering:
                            clustering[tag] = []
                        clustering[tag].append(i)
                    
                self.initFollowerStructure(clustering)
                
                tablemap = dict(zip(range(self.n), [set([i]) for i in range(self.n)]))
                for i, ind in enumerate(self.follow):
                    tablemap[i] |= tablemap[ind]
                    for item in tablemap[ind]:
                        tablemap[item] = tablemap[i]
                    
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
                print self.K
            else:
                #import pdb; pdb.set_trace()
                with open(initfn + '.follow') as f:
                    for line in f:
                        i, j = map(int, line.split())
                        self.follow[i] = j
                        self.sit_behind[j].add(i)

                with open(initfn) as f:
                    for line in f:
                        word, tag = line.split()
                        i = self.vocab.index(word)
                        tag = int(tag)
                        self.assignments[i] = tag
                        self.s[tag] += self.data[i]
                        self.ss[tag] += self.dd[i]
                        self.counts[tag] += 1
                        self.K = max(self.K, tag + 1)
            
            
    def initFollowerStructure(self, clustering):       
        for tag, cluster in clustering.iteritems():
            for ind in cluster:
                if len(cluster) == 1:
                    drawlist = list(cluster)
                else:
                    drawlist = list(set(cluster) - set([ind]))
                similarities = [self.dist[ind, i] for i in drawlist]
                maxsim = min(max(similarities), 3)
                samplelist = [i for i in drawlist if self.dist[ind, i] >= maxsim]
                if len(samplelist) == 0:
                    samplelist = drawlist
                f = random.sample(samplelist, 1)[0]
                self.follow[ind] = f
                self.sit_behind[f].add(ind)
                
    def initFollowerStructure2(self, clustering):
        for tag, cluster in clustering.iteritems():
            for ind in cluster:
                if ind == 161:
                    pass
                if len(cluster) == 1:
                    drawlist = list(cluster)
                else:
                    drawlist = list(set(cluster) - set([ind]))
                similarities = [self.dist[ind, i] for i in drawlist]
                maxsim = min(max(similarities), 3)
                while maxsim >= 0:
                    samplelist = [i for i in drawlist if self.dist[ind, i] >= maxsim]
                    samplelist2 = []
                    for item in samplelist:
                        fsetnum = self.features[ind, item]
                        fset = self.featsets[fsetnum]
                        for find in fset:
                            feat = self.featlist[find]
                            if feat.find('nomix') != -1:
                                samplelist2.append(item)
                                break
                    if len(samplelist2) > 0:
                        break
                    else:
                        maxsim -= 1
                if len(samplelist2) == 0:
                    pass
                f = random.sample(samplelist2, 1)[0]
                self.follow[ind] = f
                self.sit_behind[f].add(ind)
            
        
     
    def valueFunction(self, w=None, C=1):
        if w is None:
            w = self.w
        valnum = 0.0 # f value
        denom = 0.0  # f value
        grad = np.zeros(w.shape[0]) # f grad
        for i in range(len(self.featsets)):
            term = 0.0
            for f in self.featsets[i]:
                term += w[f]
            self.featsetvals[i] = term * self.con.a
        featsetvalsexp = np.exp(-self.featsetvals)
            
        for i, j in enumerate(self.follow):
            if i != j:
                valnum += self.featsetvals[self.features[i,j]]
                for f in self.featsets[self.features[i,j]]:
                    grad[f] += 1    # f grad
                num = np.zeros(w.shape[0])  # f grad
                denomterm = 0.0 # f grad
                for l, count in self.featcounts[i]:
                    fval = count * featsetvalsexp[l]
                    denomterm += fval
                    for f in self.featsets[l]:
                        num[f] += fval

                if denomterm < 0:
                    pass    
                denom += np.log(denomterm)    # f value

                grad -= num / denomterm # f grad

        val = valnum + denom
        print val
        #with open(self.con.out + ".train_weights", 'w') as f:
        #    for weight in w:
        #        f.write(str(weight) + '\n')
        #with open(self.con.out + ".gradients", 'w') as f:
        #    for g in grad:
        #        f.write(str(g) + '\n')
        return val, grad

    def valueFunction2(self, w=None):
        if w is None:
            w = self.w
        val = 0.0
        grad = np.zeros(w.shape[0])
        for i in range(len(self.featsets)):
            term = 0.0
            for f in self.featsets[i]:
                term += w[f]
            self.featsetvals[i] = term * self.con.a
        featsetvalsexp = np.exp(-self.featsetvals)

        clusters = {}
        for i, k in enumerate(self.assignments):
            clusters.setdefault(k, set())
            clusters[k].add(i)

        for i in range(self.n):
            k = self.assignments[i]
            firstval = 0.0
            firstnum = np.zeros(w.shape[0])
            firstdenom = 0.0
            for j in clusters[k]:
                if i != j:
                    featset = self.features[i,j]
                    featval = featsetvalsexp[featset]
                    assert(featval > 0)
                    firstval += featval
                    firstdenom += featval
                    for f in self.featsets[featset]:
                        firstnum[f] += featval
            val -= np.log(firstval)
            grad += firstnum / firstdenom
            
            secval = 0.0
            secnum = np.zeros(w.shape[0])
            for l, count in self.featcounts[i]:
                fval = count * featsetvalsexp[l]
                secval += fval
                for f in self.featsets[l]:
                    secnum[f] += fval
            
            val += np.log(secval)
            grad -= secnum / secval
        print val
        return val, grad

     
    def printWeights(self, fstem):
        sortedweights = sorted(zip(self.w, range(len(self.w))))     
        with open(fstem+".weights", 'w') as f:         
            for weight, ind in sortedweights:
                feat = self.featlist[ind]
                f.write(feat + '\t' + str(ind) + '\t' + str(weight) + '\n')
                
    def printConfiguration(self, fstem):
        total = 0.0
        with open(fstem + ".conf", 'w') as f:
            for i, j in enumerate(self.follow):
                f.write(str(i) + " " + str(j) + '\t' + self.vocab[i] + '\t' + self.vocab[j] + '\t' + str(-self.featsetvals[self.features[i,j]]) + '\n')
                total -= self.featsetvals[self.features[i,j]]
            f.write(str(total))
            
    
    def optimizeWeights(self, valf):
        print "optimize Start:", self.prior_prob()
        x, f, d = fmin_l_bfgs_b(valf, np.zeros(self.w.shape[0]), factr=10e11)
        self.w = x
        self.setParams()
        
        for i in xrange(self.n-1):
            for j in xrange(i+1, self.n):
                self.prior[i,j] = self.prior[i,j] = -self.featsetvals[self.features[i,j]]
        for i in xrange(self.n):
            self.prior[i,i] = self.con.logalpha
        for i in xrange(self.n):
            denom = np.log(np.exp(self.prior[i]).sum())
            self.prior[i] -= denom
        print "optimize End:", f, self.prior_prob()
        return x, f, d
    
    def readParams(self, w):
        self.w = w
        self.setParams()
        for i in xrange(self.n-1):
            for j in xrange(i+1, self.n):
                self.prior[i,j] = self.prior[i,j] = -self.featsetvals[self.features[i,j]]
        for i in xrange(self.n):
            self.prior[i,i] = self.con.logalpha
        for i in xrange(self.n):
            denom = np.log(np.exp(self.prior[i]).sum())
            self.prior[i] -= denom
                  
        
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
        for ind, i in enumerate(permuted):
            #if ind % 1000 == 0:
            #    print ind
            followset = self.followSet(i)
            s = self.temps
            ss = self.tempss
            s.fill(0.)
            ss.fill(0.)
            for j in followset:
                s += self.data[j]
                ss += self.dd[j]
            
            oldK = self.K
            oldt, oldf, = self.removeItem(i, followset, s, ss)
            newt, newf = self.sample(i, oldt, oldf, followset, s, ss)
            try:
                self.addItem(i, newf, oldf, newt, oldt, followset, s, ss)     
            except IndexError:
                print self.K
                self.addItem(i, newf, oldf, newt, oldt, followset, s, ss)
            if self.K > oldK:
                print self.K
            
    def storeMLLSolution(self, fn):
        with open(fn, 'w') as f:
            for t in range(self.K):
                indices = [i for i in range(self.n) if self.assignments[i] == t]
                for i in indices:
                    followset = self.followSet(i)
                    s = self.temps
                    ss = self.tempss
                    s.fill(0.)
                    ss.fill(0.)
                    for j in followset:
                        s += self.data[j]
                        ss += self.dd[j]
                    oldt, oldf, = self.removeItem(i, followset, s, ss)
                    maxllt, maxprior, maxtotal = self.findMaxLLSolution(i, followset, s, ss)
                    self.addItem(i, oldf, oldf, oldt, oldt, followset, s, ss)
                    
                    f.write(str(t) + '\t' + self.vocab[i] + '\t' + str(maxllt) + '\t:' + maxtotal[0] + '\t' + str(maxtotal[1]) + '\n')
                    for word, tag in maxprior:
                        f.write('\t' + word + '\t' + str(tag) + '\n')
                f.write('\n')
               

    def sample(self, i, oldt, oldf, followset, s, ss):      
        llcache = np.zeros(self.K)
        n = self.n
        if self.con.seq:
            n = i + 1

        probs = self.probs[:n]
        probs.fill(0.)
        
        newll = 0

        for j in xrange(n):
            if i == j:
                prior = self.con.logalpha
            else:
                prior = -self.featsetvals[self.features[i,j]]
            
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
        #featsetvalsexp = np.exp(-self.featsetvals)
        for i, j in enumerate(self.follow):
            prior += self.prior[i,j]
            '''
            if i == j:
                num += self.con.logalpha
            else:
                num += -self.featsetvals[self.features[i,j]]
            
            denomterm = 0.0
            for l, count in self.featcounts[i]:
                denomterm += count * featsetvalsexp[l]
            denomterm += self.con.logalpha
            denom += np.log(denomterm)
            '''
        #print num, denom, num - denom
        return prior    
      
    
    def igmm_prior(self):
        prior =  self.K * self.con.logalpha
        for t in range(self.K):
            prior += gammaln(self.counts[t])
        return prior
    
    def numFeatures(self):
        return len(self.featdict)

    
class DDCRPStateIntegrated(DDCRPState):
    def initialize(self, rand, init, follower):
        super(DDCRPStateIntegrated, self).initialize(rand, init, follower)
        
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
        
    def greedyClustering(self):
        #self.initialize(False, None, None)

        oldprob = self.prior_prob() + self.likelihood_int()
        while True:
            sorted_counts = sorted(zip(self.counts[:self.K], range(self.K)))
            sorted_clusters = map(lambda x: x[1], sorted_counts)
            for t in sorted_clusters:
                indices = self.getIndices(t)
                probs = []
                lls = []
                priors = []
                following = []
                i, j = None, None
                for k in xrange(self.K):
                    maxprior = float("-inf")
                    cluster_inds = self.getIndices(k)
                    for ind in indices:
                        for ind2 in cluster_inds:
                            if self.prior[ind, ind2] > maxprior:
                                maxprior = self.prior[ind, ind2]
                                i, j = ind, ind2
                    priors.append(maxprior)
                    following.append([i, j])
                            
                    if k != t:
                        ll = self.posteriorPredictive(len(indices), self.s[t], self.ss[t], k)
                    else:
                        ll = self.integrateOverParameters(len(indices), self.s[t], self.ss[t])
                    lls.append(ll)
                    probs.append(ll + maxprior)
                newt = probs.index(max(probs))
                i, j = following[newt]
                f = self.follow[i]
                prior = self.prior_prob() - self.prior[i, f] + self.prior[i,j]
                ll = self.likelihood_int() - self.integrateOverParameters(self.counts[t], self.s[t], self.ss[t], self.logdet[t]) \
                            + self.posteriorPredictive(self.counts[t], self.s[t], self.ss[t], t)
                prob = prior + ll
                if prob > oldprob:
                    if newt != t:
                        for ind in indices:
                            self.assignments[ind] = newt
                        self.counts[newt] += self.counts[t]
                        self.counts[t] = 0
                        self.s[newt] += self.s[t]
                        self.ss[newt] += self.ss[t]
                        self.K -= 1
                        f = self.follow[i]
                        self.follow[i] = j
                        self.sit_behind[j].add(i)
                        self.sit_behind[f].remove(i)
                        self.updateData(t)
                    else:
                        if f != j:
                            self.follow[i] = j
                            self.sit_behind[f].remove(i)
                            self.sit_behind[j].add(i)
                       
                    with open(self.con.out, 'w') as f:
                        for i, tag in enumerate(self.assignments):
                            f.write(self.vocab[i] + '\t' + str(tag) + '\n')
                    parg = ['/home/kairit/scripts/evaluate_pos.py', '-G', '/data/scratch/kairit/data/multext-east/english/eng_coarse_listed', '-P', self.con.out]
                    p = subprocess.Popen(parg, stdout=subprocess.PIPE)
                    res1 = '\t'.join(p.communicate()[0].strip().split()[1:])
                    parg = ['/home/kairit/scripts/evaluate_pos.py', '-G', '/data/scratch/kairit/data/multext-east/english/eng_fine_new_listed', '-P', self.con.out]
                    p = subprocess.Popen(parg, stdout=subprocess.PIPE)
                    res2 = '\t'.join(p.communicate()[0].strip().split()[1:])
                    print str(self.K) + '\t' + str(prob) + '\t' + res1 + '\t' + res2
                    oldprob = prob
                    break
            else:
                print "No improvments possible"
                break
                
            
        
        
    def sample(self, i, oldt, oldf, followset, s, ss):    
        if oldt == 26:
            pass
        #    import pdb; pdb.set_trace()
        llcache = np.zeros(self.K)
        n = self.n
        if self.con.seq:
            n = i + 1

        #probs = self.probs[:n]
        #probs.fill(0.)
        probs = []
        #inds = []
        
        newll = 0
        priors = []
        dist_priors = []
        
        for j in xrange(n):
            if i == j:
                prior = self.con.logalpha
            else:
                prior = -self.featsetvals[self.features[i,j]]
            priors.append(prior)
            distprior = -self.mdist[i,j]
            dist_priors.append(distprior)
            #if prior < self.con.priorth:
            #    continue
            #inds.append(j)
            t = self.assignments[j]
            
            if t == -1:
                ll = newll
                if ll == 0:
                    ll = newll = self.integrateOverParameters(len(followset), s, ss)
            else:
                ll = llcache[t]
                if ll == 0:
                    ll = self.posteriorPredictive(len(followset), s, ss, t)
                    llcache[t] = ll
            #probs[j]  = prior + ll   
            if oldt == 26:
                pass  
            prob = self.con.a * (prior + distprior) + ll
            probs.append(prob)    
            #probs.append(prior + ll)
        sorted_prior = sorted(zip(priors, range(n)), reverse=True) 
        sorted_dist_prior = sorted(zip(dist_priors, range(n)), reverse=True)
        probs = np.array(probs)              

        normed = np.exp(probs - logsumexp(probs))
        sorted_normed = sorted(zip(normed, range(n)), reverse=True)
        if oldt == 26:
            for score, ind_ in sorted_normed[:10]:
                print self.vocab[ind_], self.assignments[ind_]
            print
        #np.testing.assert_allclose(normed, normed2)
        ind = sampleIndex(normed)
        if oldt == 26:
            pass
        #ind = inds[ind]
        table = self.assignments[ind]
        return table, ind
    
    def findMaxLLSolution(self, i, followset, s, ss):
        likelihood = []
        for t in range(self.K):
            ll = self.posteriorPredictive(followset, s, ss, t)
            likelihood.append(ll)
        ll = self.integrateOverParameters(len(followset), s, ss)
        likelihood.append(ll)
        llres = likelihood.index(max(likelihood))
        
        priors = []
        for j in xrange(self.n):
            if i == j:
                prior = self.con.logalpha
            else:
                prior = -self.featsetvals[self.features[i,j]]
            priors.append(prior)
        maxprior = max(priors)
        priorres = [(self.vocab[j], self.assignments[j]) for j in range(self.n) if priors[j] == maxprior]
        
        maxtotal = []
        for j, t in enumerate(self.assignments):
            try:
                maxtotal.append(priors[j] + likelihood[t])
            except IndexError:
                import pdb; pdb.set_trace()
                print priors[j]
                print t, self.K, len(likelihood)
                print likelihood
                maxtotal.append(priors[j] + likelihood[t])
        totalind = maxtotal.index(max(maxtotal))
        return llres, priorres, (self.vocab[totalind], self.assignments[totalind])
        
    
    def printVariances(self, f):
        counts = self.counts[:self.K]
        counts = zip(counts, range(self.K))
        counts = sorted(counts, reverse=True)
        for count, t in counts:
            self.printPosteriorVariance(f, t)
        
            
                  
    def resampleParams(self):
        pass

if __name__ == '__main__':
    
    #random.seed(1)
    start = time.clock()
    parser = optparse.OptionParser(description='infinite Gaussian Mixture Model')
    parser.add_option('-D', '--data', help='data file name')
    parser.add_option('-O', '--out', help='output file name')
    parser.add_option('-V', '--vocab', help='vocabulary file name')
    parser.add_option('-a', '--alpha', type=float, default=1.0, help='concentration parameter for DP prior')
    parser.add_option('-L', '--Lambda', help='value for expected variance')
    parser.add_option('-P', '--pruning', type=int, default=1000, help="maximum number of clusters induced")
    parser.add_option('-b', type=float, default=1.0, help="exponential distribution parameter for distance prior")
    parser.add_option('-I', '--iter', type=int, default=100, help="number of Gibbs iterations")
    parser.add_option('-k', '--kappa', type=float, default=0.01, help="number of pseudo-observations")
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
    parser.add_option('-v', '--variance', help="File to write the posterior variances")
    parser.add_option('-n', '--nu', type=int, default=1, help='value to add to d to form the prior degrees-of-freedom')
    parser.add_option('-l', '--learn', type=int, default=-1, help='specifies the number of times the distance parameters should be learned')
    parser.add_option('-f', '--featfn', help="stem for pickled feature files")
    parser.add_option('-m', '--metric_dist', help="file containing euclidean distances between word embeddings")
    parser.add_option('-w', '--weights', help="weights file")
    (args, opts) = parser.parse_args()
    
    print
    print "data file\t:", args.data
    print "output file\t:", args.out
    print "vocabulary file\t:", args.vocab
    print "distance file\t:", args.dist
    print "init file\t:", args.init
    print "spacial dist\t", args.metric_dist
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

    if args.metric_dist is not None:
        mdist = np.load(args.metric_dist)
    else:
        mdist = None
    
    pruning = args.pruning
    if pruning == -1:
        pruning = len(vocab)
        
    if args.Lambda is None:
        n, d = data.shape
        ss = np.zeros((d, d))
        s = np.zeros(d)
        for i in xrange(n):
            ss += np.outer(data[i], data[i])
            s += data[i]
        lambdaprior = (ss - np.outer(s, s) / n) / (n - 1)
    else:
        d = data.shape[1]
        #lambdaprior = np.identity(d, np.float) * args.Lambda
        lambdaprior = np.load(args.Lambda) * (args.nu - 1)
    
    con = Constants(data.shape[1] + args.nu, mean, args.out, args.alpha, lambdaprior, pruning, args.kappa, args.b, args.threshold, args.seq)
    if args.explicit:
        state = DDCRPState(vocab, data, con, dist)
    else:
        state = DDCRPStateIntegrated(vocab, data, con, dist, args.featfn, mdist)
    
    if args.trace is None:
        tracef = sys.stdout
    else:
        tracef = open(args.trace,'w')      
    
    state.initialize(args.rand, args.init, True)
    state.resampleParams()
    if args.weights is not None:
        weights = []
        with open(args.weights) as f:
            for line in f:
                line = line.split()
                ind, weight = int(line[-2]), float(line[-1])
                if ind >= len(weights):
                    weights += (ind - len(weights) + 1) * [0]
                weights[ind] = weight
        weights = np.array(weights)
        state.readParams(weights)
    
    
    valfn = state.valueFunction
    if args.learn == -1:
        learn = args.iter + 1
    else:
        learn = args.learn
    learned = 0
    if learned < learn:
        w, val, d = state.optimizeWeights(valfn)
        learned += 1
        state.printWeights(args.out)
        state.printConfiguration(args.out)

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
    tracef.write("igmm_int\tval")
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
    #tracef.write('\t' + str(round(state.valueFunction()[0], 1)))
    if args.stats:
        tracef.write('\t' + str(state.numClusters()) + '\t' + state.histogram())
    tracef.write('\n')
    if args.trace is not None:
        tracef.close() 
    if args.out is not None:
        with open(args.out, 'w') as f:
            for i, item in enumerate(state.assignments):
                f.write(vocab[i] + '\t' + str(item) + '\n')
        if args.evalscript is not None and args.gold is not None:
            parg = [args.evalscript, '-G', args.gold, '-P', args.out]
            p = subprocess.Popen(parg, stdout=subprocess.PIPE)
            res = p.communicate()[0]
            with open(args.out + '.res', 'a') as f:
                f.write(res)
                
        
    for i in range(args.iter):
        #import pdb; pdb.set_trace()
        state.resampleData()
        state.resampleParams()
        if learned < learn:
            w, val, d = state.optimizeWeights(valfn)
            learned += 1
            state.printWeights(args.out)
            state.printConfiguration(args.out)
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
            #tracef.write('\t' + str(round(state.valueFunction()[0], 1)))
            if args.stats:
                tracef.write('\t' + str(state.numClusters()) + '\t' + state.histogram())
            tracef.write('\n')
            if args.trace is not None:
                tracef.close()
            
            if args.out is not None:
                with open(args.out, 'w') as f:
                    for i, item in enumerate(state.assignments):
                        f.write(vocab[i] + '\t' + str(item) + '\n')
                if args.evalscript is not None and args.gold is not None:
                    parg = [args.evalscript, '-G', args.gold, '-P', args.out]
                    p = subprocess.Popen(parg, stdout=subprocess.PIPE)
                    res = p.communicate()[0]
                    with open(args.out + '.res', 'a') as f:
                        f.write(res)
    if args.out is not None:
        with open(args.out, 'w') as f:
            for j, item in enumerate(state.assignments):
                f.write(vocab[j] + '\t' + str(item) + '\n')
            
    if args.variance is not None:
        #with open(args.variance, 'w') as f:
        #    state.printVariances(f)
        state.printVariances(args.variance)

    

