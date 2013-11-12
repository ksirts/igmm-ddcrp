'''
Created on Aug 9, 2013

@author: Kairit
'''
from __future__ import division
import math, random
import numpy as np
import numpy.random as npr
from numpy.linalg import inv, cholesky, slogdet
from scipy.stats import chi2
from scipy.special import gammaln
from scipy.misc import logsumexp
import argparse
import sys

from common import Constants
from common import State

def invwishartrand(nu, phi):
    return inv(wishartrand(nu, inv(phi)))

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
    return np.dot(chol, np.dot(foo, np.dot(foo.T, chol.T)))

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
        
def mylogsumexp(probs):
    probs = np.array(probs)
    a = np.amax(probs)
    return a + np.log(np.exp(probs - a).sum())

def distance(s1, s2):
    l1 = len(s1)
    l2 = len(s2)

    matrix = [range(l1 + 1)] * (l2 + 1)
    for zz in range(l2 + 1):
        matrix[zz] = range(zz,zz + l1 + 1)
    for zz in range(0,l2):
        for sz in range(0,l1):
            if s1[sz] == s2[zz]:
                matrix[zz+1][sz+1] = min(matrix[zz+1][sz] + 1, matrix[zz][sz+1] + 1, matrix[zz][sz])
            else:
                matrix[zz+1][sz+1] = min(matrix[zz+1][sz] + 1, matrix[zz][sz+1] + 1, matrix[zz][sz] + 1)
    return matrix[l2][l1]

def generateData(alpha, Lambda=None):
    means = []
    sigmas = []
        
    num_comp = 0
    dim = 2
    nu = dim
    Lambda = Lambda * np.identity(dim)
    counts = []
    for t in range(num_comp):
        sigma = invwishartrand(nu, Lambda)
        sigmas.append(sigma)
        mu = np.array(random.sample(range(60), 2))
        #mu = priormeans[t] * np.ones(dim)
        mean = npr.multivariate_normal(mu, sigma)
        means.append(mean)
    data = []
    labels = []
    for i in range(100):
        if random.random() < alpha / (i + alpha):
            t = num_comp
            num_comp += 1
            counts.append(1)
            sigma = invwishartrand(nu, Lambda)
            sigmas.append(sigma)
            mu = np.array(random.sample(range(60), 2))
            mean = npr.multivariate_normal(mu, sigma)
            means.append(mean)
            data.append(npr.multivariate_normal(means[t], sigmas[t]))
            labels.append(t)
        else:
            t = random.sample(labels, 1)[0]
            counts[t] += 1
            data.append(npr.multivariate_normal(means[t], sigmas[t]))
            labels.append(t)
    return np.array(data), labels, means, counts, sigmas

def logmvstprob(x, mu, nu, d, Lambda):
    diff = x - mu
    prob = gammaln((nu + d) / 2)
    prob -= gammaln(nu / 2)
    prob -= d / 2 * (math.log(nu) + math.log(math.pi))
    prob -= 0.5 * slogdet(Lambda)[1]
    try:
        prob -= (nu + d) / 2. * math.log(1 + 1. / nu * np.dot(np.dot(diff.T, inv(Lambda)), diff)[0][0])
    except ValueError:
        prob -= (nu + d) / 2. * math.log(1 + 1. / nu * np.dot(np.dot(diff.T, inv(Lambda)), diff)[0][0])
    return prob            

        
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

        self.temps = np.zeros(self.d, np.float)
        self.tempss = np.zeros((self.d, self.d), np.float)

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
                    word, tag = line.split()
                    tag = int(tag)
                    #assert self.vocab[i] == word
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
                
        self.changed = np.ones(self.con.pruningfactor)   
        self.probs = np.zeros(self.n)
        
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
        

    def getIndices(self, t):
        return [i for i in xrange(self.n) if self.assignments[i] == t]

    def getData(self, followset):
        return np.array([self.data[i] for i in followset])
    
    def removeItem(self, i, followset, s, ss):
        t = self.assignments[i] 
        '''
        n = len(followset)
        self.counts[t] -= n
        assert self.counts[t] >= 0
        '''
        f = self.follow[i]
        '''
        self.sit_behind[f].remove(i)        

        for ind in followset:
            self.assignments[ind] = -1
        self.s[t] -= s
        self.ss[t] -= ss
        '''
        return t, f
    
    def addItem(self, i, ind, oldind, table, oldtable, followset, s, ss):
        newtable = (ind in followset and oldind not in followset)
        if newtable:
            table = self.K
            self.K += 1
            
        self.follow[i] = ind
        self.sit_behind[oldind].remove(i)
        self.sit_behind[ind].add(i)
            
        if table != oldtable:
            n = len(followset)
            self.counts[oldtable] -= n
            self.counts[table] += n
            assert self.counts[oldtable] >= 0
            assert self.counts[table] > 0
            self.s[oldtable] -= s
            self.s[table] += s
            self.ss[oldtable] -= ss
            self.ss[table] += ss
            for ind in followset:
                self.assignments[ind] = table 
                
        if len(followset) > self.con.changeParams:
            mu, precision = self.sampleNewParams(table)
            self.mu[table] = mu
            self.precision[table] = precision
            self.logdet[table] = slogdet(precision)[1]
            ll = self.mvNormalLL(self.s[table], self.ss[table], self.counts[table], mu, precision, self.logdet[table])
            self.cluster_likelihood[table] = ll
            self.paramprobs[table] = self.logWishartPdf(precision) + \
                    self.mvNormalLL(mu, np.dot(mu, mu.transpose()), 1, self.con.mu0, self.con.kappa0 * precision, self.d * math.log(self.con.kappa0) + self.logdet[table])
            if self.counts[oldtable] > 0:
                mu, precision = self.sampleNewParams(oldtable)
                self.mu[oldtable] = mu
                self.precision[oldtable] = precision
                self.logdet[oldtable] = slogdet(precision)[1]
                ll = self.mvNormalLL(self.s[oldtable], self.ss[oldtable], self.counts[oldtable], mu, precision, self.logdet[oldtable])
                self.cluster_likelihood[oldtable] = ll
                self.paramprobs[oldtable] = self.logWishartPdf(precision) + \
                    self.mvNormalLL(mu, np.dot(mu, mu.transpose()), 1, self.con.mu0, self.con.kappa0 * precision, self.d * math.log(self.con.kappa0) + self.logdet[oldtable])
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
        
        if self.counts[oldtable] == 0:
            self.K -= 1
            self.updateData(oldtable)
        
    
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

    def likelihoodF2(self, followset, t, s, ss):
        ll = 0
        if t == -1:
            for i in followset:
                ll += self.integrated[i]
        else:
            ll = self.mvNormalLL(s, ss, len(followset), self.mu[t], self.precision[t], self.logdet[t])
        return ll

    def likelihoodF_Jacob(self, followset, t):
        ll = 0
        if t == -1:
            for i in followset:
                ll += self.integrated[i]
        else:
            need_update = []
            for i in followset:
                l = self.likelihood[t, i]
                if l != 0:
                    ll += l
                else:
                    need_update.append(i)
            
            ''' 
            This code allows you to compute approximately 45% more
            likelihood computations in the same time.  The reason is
            that when you are computing a lot of Gaussian
            probabilities, it's faster to just do the whole thing as a
            single matrix operation, rather than as a series of
            operations.  See the matrix multiplication below.
            
            But this is high overhead, because you have to create a
            matrix to store the difference of each observation from
            the mean. So we do it only when it's worth it: when there
            are 1000 such cases or more.
            
            I tested this threshold and it's better than 500 and 5000
            '''
            thresh = 1000
            if len(need_update) < thresh: #if only a few elements need an update, the matrix solution is too high overhead.
                for i in need_update:
                    l = self.mvNormalLL(self.data[i], self.mu[t], self.precision[t], self.logdet[t])
                    self.likelihood[t,i] = l
                    ll += l
            else:
                #store the update. this is apparently faster than the list comprehension.
                update_data = np.zeros((len(need_update),self.d))
                for item,i in enumerate(need_update):
                    update_data[item,:] = self.data[i]
                
                # compute the difference from the table mean
                update_data = np.matrix(update_data  - self.mu[t])

                # Gaussian stuff
                partialZ = self.d / 2 * math.log(2 * math.pi)
                Z = partialZ - 0.5 * self.logdet[t]

                # compute the PDF
                gaussian_update = -0.5 *np.multiply((update_data * self.precision[t]),update_data).sum(axis=1) - Z
                for item,i in enumerate(need_update):
                    self.likelihood[t,i] = gaussian_update[item]

                ll += gaussian_update.sum()

                you_dont_believe_it = False
                if you_dont_believe_it:
                    alt_l = self.mvNormalLL(self.data[i], self.mu[t], self.precision[t], self.logdet[t])
                    print alt_l, self.likelihood[t,i]
                    assert np.abs(alt_l - self.likelihood[t,i]) < 1e-4
        return ll
    
    def likelihoodF3(self, t, ss, s, count, blockll):
        n = self.counts[t]
        kappa = self.con.kappa0 + n
        mu = (self.con.kappa0 * self.con.mu0 + self.s[t]) / kappa
        nu = self.con.nu0 + n
        diff = self.s[t] / n - self.con.mu0
        newLambda = self.con.lambda0 + self.ss[t] - np.dot(self.s[t], self.s[t].T) / n + self.con.kappa0 * n / kappa * np.dot(diff, diff.T)
        precision = kappa * (nu - self.d + 1) / (kappa + 1) * inv(newLambda)
        logdet = slogdet(precision)[1]

        new = self.mvNormalLL(s, ss, count, mu, precision, logdet)
        ll = new - blockll
        return ll
    
    def resampleData(self):
        self.changed.fill(0)
        #permuted = range(self.n)
        #random.shuffle(permuted)
        for i in xrange(self.n):
            followset = self.followSet(i)
            s = self.temps
            ss = self.tempss
            s.fill(0.)
            ss.fill(0.)
            for ind in followset:
                s += self.data[ind]
                ss += self.dd[ind]
            s = s[:,None]
            
            oldtable, oldind, = self.removeItem(i, followset, s, ss)
            newind, newtable = self.sample(i, oldtable, oldind, followset, s, ss)
            if newind != oldind:
                self.addItem(i, newind, oldind, newtable, oldtable, followset, s, ss)
                #prob = self.igmm_logprob()
                #print i,  prob, sum(prob), self.K, self.histogram()
            '''
            if oldtable != newtable:
                self.changed[oldtable] = 1
                self.changed[newtable] = 1
            '''
                
    def integrateOverParameters(self, followset, s, ss):
#         ll = 0.0
#         s_ = np.zeros((self.d, 1), np.float)
#         ss_ = np.zeros((self.d, self.d), np.float)
#         mu_num = self.con.kappa0 * self.con.mu0 + s_
#         nu = self.con.nu0
#         kappa = self.con.kappa0
#         Lambda = 1 * self.con.lambda0
#         n = 0
#         for ind in followset:
#             x = self.data[ind][:,None]
#             nu_std = nu - self.d + 1
#             S_std = (kappa + 1) / (kappa * nu_std) * Lambda
#             ll += logmvstprob(x, mu_num / kappa, nu - self.d + 1, self.d, S_std)
#             nu += 1
#             kappa += 1
#             n += 1
#             s_ += x
#             ss_ += self.dd[ind]
#             mu_num += x
#             mean = s_ / n
#             diff = mean - self.con.mu0
#             Q = ss_ - np.dot(s_, s_.T) / n
#             Lambda = self.con.lambda0 + Q + self.con.kappa0 * n / kappa * np.dot(diff, diff.T) 
        
        n = len(followset)
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
    def sample(self, i, oldtable, oldind, followset, s, ss):      
        llcache = np.zeros(self.K+1)
        n = i+1 if self.con.seq else self.n
        #inds = [j for j in xrange(n) if self.prior[i, j] > self.con.priorth]
        
        #probs = self.probs[:len(inds)]
        probs = self.probs[:n]
        probs.fill(0.)
        oldll = self.mvNormalLL(s, ss, len(followset), self.mu[oldtable], self.precision[oldtable], self.logdet[oldtable])
        newtableLL = 0.0

        #for ind, j in enumerate(inds):
        for j in xrange(n):
            t = self.assignments[j]
            newtable = (j in followset and oldind not in followset) #this is faster inline
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

        normed = np.exp(probs - logsumexp(probs))
        ind = sampleIndex(normed)
        #ind = inds[indind]
        table = self.assignments[ind]
        #if table == -1:
        #    table = self.K
        return ind, table
    
    def updateData(self, t):
        assert self.counts[t] == 0
        if t < self.K:
            self.mu[t] = self.mu[self.K]
            self.precision[t] = self.precision[self.K]
            self.logdet[t] = self.logdet[self.K]
            self.counts[t] = self.counts[self.K]
            assert self.counts[t] > 0
            self.changed[t] = self.changed[self.K]
            self.likelihood[t] = self.likelihood[self.K]
            self.s[t] = self.s[self.K]
            self.ss[t] = self.ss[self.K]
            self.paramprobs[t] = self.paramprobs[self.K]
                
            self.mu[self.K].fill(0.)
            self.precision[self.K].fill(0.)
            self.logdet[self.K] = 0.0
            self.counts[self.K] = 0
            self.changed[self.K] = 0
            self.likelihood[self.K].fill(0.)
            self.s[self.K].fill(0.)
            self.ss[self.K].fill(0.)
            self.paramprobs[self.K] = 0.
                
            self.assignments = np.array([t if x==self.K else x for x in self.assignments])
      

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
    #npr.seed(1)
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
    
    vocab = open(args.vocab).read().split()
    data = np.load(args.data)

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
    #maxprob = prob
    sys.stderr.write( "> iter 0:\t" +  str(round(prior)) + '\t' + str(round(baseprob)) + '\t' + str(round(likelihood)) + '\t' + str(round(prob)) + '\t' + str(state.numClusters()) + '\t' + state.histogram() + '\n')
    for i in range(args.iter):
        state.resampleData()
        state.resampleParams()
        if (i + 1) % 1 == 0:
            prior, baseprob, likelihood = state.logprob()
            prob = prior + baseprob + likelihood
            #maxprob = max(prob, maxprob)
            sys.stderr.write("> iter " + str(i+1) +":\t" + str(round(prior)) + '\t' + str(round(baseprob)) + '\t' + str(round(likelihood)) + '\t' + str(round(prob)) + '\t' + str(state.numClusters()) + '\t' + state.histogram() + '\n')
    '''
    while prob < maxprob:
        i += 1
        state.resampleData()
        state.resampleParams()
        prior, baseprob, likelihood = state.logprob()
        prob = prior + baseprob + likelihood   
        sys.stderr.write("> iter " + str(i+1) +":\t" + str(round(prior)) + '\t' + str(round(baseprob)) + '\t' + str(round(likelihood)) + '\t' + str(round(prob)) + '\t' + str(round(maxprob)) + '\t' 
                         + str(state.numClusters()) + '\t' + state.histogram() + '\n')
    '''
    with open(args.out, 'w') as f:
        for i, item in enumerate(state.assignments):
            f.write(vocab[i] + '\t' + str(item) + '\n')
            #f.write(str(i) + '\t' + str(item) + '\n')
    
    '''
    print state.histogram()
    print round(sum(state.igmm_logprob()))
    means = state.mu[:state.K]
    for mean in means:
        print mean
    precision = state.precision[:state.K]
    for prec in precision:
        print inv(prec)
    '''
