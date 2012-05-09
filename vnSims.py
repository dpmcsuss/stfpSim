'''
Created on Mar 23, 2012

@author: dsussman
'''
from itertools import product, repeat
import Embed
import RandomGraph as rg
import networkx as nx
import numpy as np
import vertexNomination as vn
import cPickle as pickle
from joblib import Parallel, delayed

class errorfulSizeCond(object):
    rgg= []
    errFunc = lambda ep: 5000+50000/(np.sin(np.pi/4))*np.sin(ep*np.pi/2)
    epsRange = []
    
    mObserved = 10
    
    dRange = []
    kRange = []
    
    nmc = 2
    
    vnResults = {}
    mclustResults = {}
    kmeansResults = {}
    
    pickleFn = None
    
    block = None
    blockOI = None
    observed = None
    notObserved = None
    
    
    def __init__(self, rgg, errFunc, mObserved, epsRange, dRange, kRange):
        self.rgg = rgg
        self.errFunc = errFunc
        self.mObserved = mObserved
        self.epsRange = epsRange
        self.dRange = dRange
        self.kRange = kRange
        
        self.embed = Embed.Embed(np.max(dRange), matrix=Embed.adjacency_sparse)
        
    def start_mc(self):
        eps_d_pair = list(product(self.epsRange,self.dRange))
        self.vnResults = dict(zip(*[eps_d_pair, [[] for _ in eps_d_pair]]))
        self.mclustResults = dict(zip(*[eps_d_pair,[[] for _ in eps_d_pair]]))
        self.kmeansResults = dict(zip(*[eps_d_pair,[[] for _ in eps_d_pair]]))
        
        G = self.rgg.generate_graph()
        nnodes = G.number_of_nodes()
        self.block = np.array(nx.get_node_attributes(G, 'block').values())
        self.blockOI = [v for (l,v) in zip(self.block,np.arange(nnodes)) if l==0]
        self.observed = self.blockOI[0:self.mObserved]
        self.notObserved = self.blockOI[self.mObserved::]
        self.continue_mc(self.nmc)
    

    def continue_mc(self, nmc):
        for mc in xrange(self.nmc):
            print "Starting monte carlo "+repr(mc+1)
            self._do_mc()
            if self.pickleFn is not None:
                errFunc = self.errFunc
                self.errFunc = None
                pickle.dump(self, open(self.pickleFn,'wb'))
                self.errFunc = errFunc
    
    def continue_cm_ep(self,nmc):
        Parallel(n_jobs=3)(delayed(errorfulVN._do_mc)(self) for _ in xrange(nmc))

    def _do_mc(self):
        G = self.rgg.generate_graph()
        for eps in self.epsRange:
            Gerr = rg.get_errorful_subgraph(G, int(self.errFunc(eps)), eps)
            self.embed.embed(Gerr)
            for d in self.dRange:
                x = self.embed.get_scaled(d)
                
                vnRes = vn.vn_metrics(x, self.observed, self.notObserved)
                vnRes.run() 
                self.vnResults[(eps,d)].append(vnRes)
                
                mclustRes = vn.mclust_performance(x,self.block)
                mclustRes.run()
                self.mclustResults[(eps,d)].append(mclustRes)
                
                kmeansRes = vn.kmeans_performance(x, self.block, self.kRange)
                kmeansRes.run()
                self.kmeansResults[(eps,d)].append(kmeansRes)
                
                
class errorfulObs(object):
    pass

class errorfulPostVN(object):
    rgg= []
    post0 = lambda ep: ep
    post1 = lambda ep: ep**2
    epsRange = []
    
    mObserved = 10
    
    dRange = []
    kRange = []
    
    nmc = 2
    
    vnResults = {}
    mclustResults = {}
    kmeansResults = {}
    
    pickleFn = None
    
    block = None
    blockOI = None
    observed = None
    notObserved = None
    edgeProb = 0
    
    
    def __init__(self, rgg, post0, post1, mObserved, epsRange, dRange, kRange):
        self.post1 = post1
        self.post0 = post0
        self.rgg = rgg
        self.mObserved = mObserved
        self.epsRange = epsRange
        self.dRange = dRange
        self.kRange = kRange
        
        self.embed = Embed.Embed(np.max(dRange), matrix=Embed.adjacency_sparse)
        
    def start_mc(self):
        eps_d_pair = list(product(self.epsRange,self.dRange))
        self.vnResults = dict(zip(*[eps_d_pair, [[] for _ in eps_d_pair]]))
        self.mclustResults = dict(zip(*[eps_d_pair,[[] for _ in eps_d_pair]]))
        self.kmeansResults = dict(zip(*[eps_d_pair,[[] for _ in eps_d_pair]]))
        
        G = self.rgg.generate_graph()
        nnodes = G.number_of_nodes()
        self.block = np.array(nx.get_node_attributes(G, 'block').values())
        self.blockOI = [v for (l,v) in zip(self.block,np.arange(nnodes)) if l==0]
        self.observed = self.blockOI[0:self.mObserved]
        self.notObserved = self.blockOI[self.mObserved::]
        
        block_prob = self.rgg.block_prob
        nvec = self.rgg.nvec
        nnodes = np.sum(nvec)
        possible_edges = nnodes*(nnodes-1)
        self.edgeProb = (nvec.dot(block_prob).dot(nvec.T)-np.sum(np.diag(block_prob).dot(nvec)))/possible_edges
        
        self.continue_mc(self.nmc)
    

    def continue_mc(self, nmc):
        for mc in xrange(self.nmc):
            print "Starting monte carlo "+repr(mc+1)
            self._do_mc()
            if self.pickleFn is not None:
                try:
                    pickle.dump(self, open(self.pickleFn,'wb'))
                except Exception:
                    post1 = self.post1
                    post0 = self.post0
                    
                    self.post0 = None
                    self.post1 = None
                    pickle.dump(self, open(self.pickleFn,'wb'))
                    self.post0 = post0
                    self.post1 = post1
                    
    
    def continue_cm_ep(self,nmc):
        Parallel(n_jobs=3)(delayed(errorfulVN._do_mc)(self) for _ in xrange(nmc))

    def _do_mc(self):
        
        for eps in self.epsRange:
            block_prob = (1-self.post1(eps)-(1-self.post0(eps)))*self.rgg.block_prob+(1-self.post0(eps))
            
            rggErr = rg.SBMGenerator(block_prob,self.rgg.nvec)
            Gerr = rggErr.generate_graph()
            self.embed.embed(Gerr)
            for d in self.dRange:
                x = self.embed.get_scaled(d)
                
                vnRes = vn.vn_metrics(x, self.observed, self.notObserved)
                vnRes.run()     
                self.vnResults[(eps,d)].append(vnRes)
                
#                mclustRes = vn.mclust_performance(x,self.block)
#                mclustRes.run()
#                self.mclustResults[(eps,d)].append(mclustRes)
                
                kmeansRes = vn.kmeans_performance(x, self.block, self.kRange)
                kmeansRes.run()
                self.kmeansResults[(eps,d)].append(kmeansRes)
                
        
class errorfulSbmVN(object):
    rgg= []
    errFunc = lambda ep: 5000+50000/(np.sin(np.pi/4))*np.sin(ep*np.pi/2) #TODO: make this customized for each graph
    epsRange = []
    
    mObserved = 10
    
    dRange = []
    kRange = []
    
    nmc = 2
    
    vnResults = {}
    mclustResults = {}
    kmeansResults = {}
    
    pickleFn = None
    
    block = None
    blockOI = None
    observed = None
    notObserved = None
    edgeProb = 0
    
    
    def __init__(self, rgg, errFunc, mObserved, epsRange, dRange, kRange):
        self.rgg = rgg
        self.errFunc = errFunc
        self.mObserved = mObserved
        self.epsRange = epsRange
        self.dRange = dRange
        self.kRange = kRange
        
        self.embed = Embed.Embed(np.max(dRange), matrix=Embed.adjacency_sparse)
        
    def start_mc(self):
        eps_d_pair = list(product(self.epsRange,self.dRange))
        self.vnResults = dict(zip(*[eps_d_pair, [[] for _ in eps_d_pair]]))
        self.mclustResults = dict(zip(*[eps_d_pair,[[] for _ in eps_d_pair]]))
        self.kmeansResults = dict(zip(*[eps_d_pair,[[] for _ in eps_d_pair]]))
        
        G = self.rgg.generate_graph()
        nnodes = G.number_of_nodes()
        self.block = np.array(nx.get_node_attributes(G, 'block').values())
        self.blockOI = [v for (l,v) in zip(self.block,np.arange(nnodes)) if l==0]
        self.observed = self.blockOI[0:self.mObserved]
        self.notObserved = self.blockOI[self.mObserved::]
        
        block_prob = self.rgg.block_prob
        nvec = self.rgg.nvec
        nnodes = np.sum(nvec)
        possible_edges = nnodes*(nnodes-1)
        self.edgeProb = (nvec.dot(block_prob).dot(nvec.T)-np.sum(np.diag(block_prob).dot(nvec)))/possible_edges
        
        self.continue_mc(self.nmc)
    

    def continue_mc(self, nmc):
        for mc in xrange(self.nmc):
            print "Starting monte carlo "+repr(mc+1)
            self._do_mc()
            if self.pickleFn is not None:
                try:
                    pickle.dump(self, open(self.pickleFn,'wb'))
                except pickle.PicklingError:
                    errFunc = self.errFunc
                    self.errFunc = None
                    pickle.dump(self, open(self.pickleFn,'wb'))
                    self.errFunc = errFunc
    
    def continue_cm_ep(self,nmc):
        Parallel(n_jobs=3)(delayed(errorfulVN._do_mc)(self) for _ in xrange(nmc))

    def _do_mc(self):
        
        B = self.rgg.block_prob
        for eps in self.epsRange:
            block_prob = (self.errFunc(eps)/self.errFunc(1))*((1-eps)*B+eps*self.edgeProb)
            
            rggErr = rg.SBMGenerator(block_prob,self.rgg.nvec)
            Gerr = rggErr.generate_graph()
            self.embed.embed(Gerr)
            for d in self.dRange:
                x = self.embed.get_scaled(d)
                
                vnRes = vn.vn_metrics(x, self.observed, self.notObserved)
                vnRes.run()     
                self.vnResults[(eps,d)].append(vnRes)
                
#                mclustRes = vn.mclust_performance(x,self.block)
#                mclustRes.run()
#                self.mclustResults[(eps,d)].append(mclustRes)
                
                kmeansRes = vn.kmeans_performance(x, self.block, self.kRange)
                kmeansRes.run()
                self.kmeansResults[(eps,d)].append(kmeansRes)
                
def errFunc_default(ep):
    one = (5000+100000/(np.sin(np.pi/4))*np.sin(np.pi/2))
    return (5000+100000/(np.sin(np.pi/4))*np.sin(ep*np.pi/2))/one

def errFunc_cubeRoot(ep):
    return .01+.99*ep**(1.0/3.0) 


def nsrr_boxplot(errVN, d):
    nsrr = np.array([[vnR.nsrr for vnR in errVN.vnResults[(eps,d)]] for eps in errVN.epsRange])

if __name__=='__main__':
    K = 3
    block_prob = .05*np.ones((K,K))+.05*np.eye(K)
    rho = np.ones(K)/K
    n = 1000
    
    
    errFunc = lambda ep: 5000+100000/(np.sin(np.pi/4))*np.sin(ep*np.pi/2)
    errFuncN = lambda ep: (5000+100000/(np.sin(np.pi/4))*np.sin(ep*np.pi/2))/errFunc(1)*0.13326653306613229

    
    mObserved = 10
    
    epsRange = np.arange(0,1.05,.1)
    dRange = np.array([3,10])#np.array([1,2,3,4,5,10,15,20])
    kRange = np.array([3]) #np.arange(2,11)pl
    
    rgg = rg.SBMGenerator(block_prob, (rho*n).astype(int))
    edgeProb = rgg.nvec.dot(rgg.block_prob).dot(rgg.nvec.T)-np.sum(np.diag(rgg.block_prob).dot(rgg.nvec))
    
    
    #errVN = errorfulVN(rgg, errFunc, mObserved, epsRange, dRange, kRange)
    errVN = errorfulSbmVN(rgg, errFuncN, mObserved, epsRange, dRange, kRange)
    errVN.start_mc()
    #errVN.start_mc()