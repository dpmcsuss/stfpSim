import numpy as np
import networkx as nx
from numpy.random import random_integers
import scipy.stats as stats
from scipy.stats import norm, semicircular
from scipy.optimize import brentq


import giorgio as br
import Embed
from vertexNomination import mclust_performance as mclust
from RandomGraph import SBMGenerator

from matplotlib import pyplot as plt
from matplotlib import cm

from joblib import Parallel, delayed
import cPickle as pickle

class wigner(stats.distributions.semicircular_gen):
    def fitWig(self,data,weights=None):
        if weights is None:
            weights = np.ones_like(data)
        
        minR = np.max(np.abs(data[weights>0]))+.0000001
        R = brentq(lambda r: self._dll(r,data,weights),minR,2*minR)
        
        return R
        
    def _dll(self,r,s,w):
        """Returns the derivative of the log-likelihood for MLE computation"""

        return -2*np.sum(w)/r+np.sum(r*w/(r**2-s**2))
                

def EM(s, dinit, niter):
    n = s.shape[0]
    d = dinit
    z = np.concatenate((np.zeros(n-d),np.ones(d)))
    
    wig = wigner()
    wpdf = semicircular.pdf
    npdf = norm.pdf #semicircular.pdf
    for i in xrange(niter):
        gamma = np.sum(z)/n
        rhat = wig.fitWig(s,1-z)
        #np.sqrt(1.0/np.sum(1-z)*np.sum((1-z)*(s**2)))
        #rhat = np.max(s[z<1])*1.0001
        muhat = 1.0/np.sum(z)*np.sum(z*s)
        sighat = np.sqrt(1.0/np.sum(z)*np.sum(z*((s-muhat)**2)))
        
        ll = np.sum(np.log(gamma*npdf(s,muhat,sighat)+(1-gamma)*wpdf(s,0,rhat)))
        
        print 'R='+repr(rhat)+', sig='+repr(sighat)+', mu='+repr(muhat)
        #print repr(z)
        
        z = (gamma*npdf(s,muhat,sighat))/(gamma*npdf(s,muhat,sighat)+(1-gamma)*wpdf(s,0,rhat))
        
    return z,rhat,sighat,muhat
            


def get_subgraph_density(A, v):
    return sum(sum(A[v,:][:,v]))/(len(v)**2)

def get_results(rgg, eA,dRange):
    G = rgg.generate_graph();
    A = Embed.adjacency_matrix(G)
    n = G.number_of_nodes()
    
    m = max(int(np.sqrt(n)),50)
    
    rhoBS = np.array([get_subgraph_density(A,random_integers(0,n-1, m)) for _ in xrange(1000)])
    rhoHat = np.sort(.5-np.abs(.5-rhoBS))[950]
    
    sval = eA.embed(G).sval;
    
    mcr = [mclust(rggk.label,eA.get_scaled(d)) for d in dRange]
    return (sval, mcr,rhoHat)
    

    

def plot_svals(svals,nlist):
    dMax = svals.shape[2]
    color = cm.jet(np.linspace(0,255,dMax).astype(int))
    for d in xrange(svals.shape[2]):
        bp = plt.boxplot(svals[:,:,d].T, positions=nlist+10*d, notch=1, sym='+', vert=1, whis=1.5,hold=True,widths=10)
        plt.setp(bp['boxes'], color=color[d,:],linewidth=2)
        plt.setp(bp['whiskers'], color=color[d,:],linewidth=1)
        plt.setp(bp['medians'], color=color[d,:],linewidth=1)
        plt.setp(bp['fliers'], color=color[d,:],markersize=2)
    
    plt.xticks(nlist)
    plt.xlim((np.min(nlist), np.max(nlist)+dMax*12))

def posterior(sval, d, delta, sigP, sigW):
    dll = stats.poisson.logpmf(d, delta)
    Pll = sum(stats.chi2.logpdf(sval, sigP))
    
    
def run_mclust_sim():
    dRange = np.arange(1,15)
    eA = Embed.Embed(dim=max(dRange),matrix=Embed.marchette_matrix, directed=True)
    rgglist = [SBMGenerator(br.rgg.block_prob, (br.rgg.nvec/50)*k, directed=True) for k in np.arange(1,11)]
    nlist = np.array([np.sum(rggk.nvec) for rggk in rgglist])
    
    nmc = 10;
    res = np.array([Parallel(n_jobs=2)(delayed(get_results)(rggk,eA,dRange) for i in range(nmc))
                    for rggk in rgglist])
    sval = narray(res[:,:,0].tolist())
    mcr = res[:,:,1]
    rhoHat = res[:,:,2].astype(float)
    
    
    
    pickle.dump(res, open('/home/dsussman/Data/estimateDimResults.pickle','wb'))
    
def normW_vs_n(rgg,nrange,nmc):
    n0 = sum(rgg.nvec)
    nn = len(nrange)
    
    rhoQ = zeros((nn,nmc))
    sigQ = zeros((nn,nmc))
    normW = zeros((nn,nmc))
    
    
    for j in xrange(nn):
        n=nrange(j)
        nvec = (rgg.nvec*n)/n0
        tau = np.concatenate([np.ones(nvec[i])*i for i in xrange(len(nvec))]).astype(int)
        P = rgg.block_prob[tau,:][:,tau]
        sigP = sqrt(P*(1-P))
        
        for mc in xrange(nmc):
            A = (rand(n,n)<P).astype(float)
            W = A-P
            normW[j,nmc] = np.linalg.norm(W,2)
            
            rhoBS = np.array([get_subgraph_density(A,random_integers(0,n-1, m)) for _ in xrange(1000)])
        
            sigQ[j,mc] = float(sum(sigP >normW[j,nmc]/(2*sqrt(n))))/(n**2)

    

    
if __name__ == '__main__':
    run_mclust_sim();
    