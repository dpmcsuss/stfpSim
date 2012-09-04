'''
Created on Mar 28, 2012

@author: dsussman
'''
import csv
import networkx as nx
import Embed

from sklearn.cluster import k_means
from sklearn.metrics import adjusted_rand_score
from sklearn import cross_validation
from sklearn import neighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn import cross_validation
from sklearn.base import TransformerMixin

import vertexNomination as vn
from matplotlib import pyplot as plt
import numpy as np
import random

def boxplot(data,pos, c='black'):    
        
    width = np.min(pos[1:]-pos[:-1])*.7
    bp = plt.boxplot(data, positions=pos, notch=1, widths=width, sym='+', vert=1, whis=1.5)
    plt.setp(bp['boxes'], color=c)#,linewidth=2)
    plt.setp(bp['whiskers'], color=c)#,linewidth=1)
    plt.setp(bp['medians'], color=c)#,linewidth=1)
    plt.setp(bp['fliers'], color=c, marker='+')
    
    plt.xlim([np.min(pos)-width,np.max(pos)+width])

def getWikiGraph(edgeListFn, labelFn):
    '''
    Constructor
    '''
    csvreader =csv.reader(open(edgeListFn,'rt'),delimiter=' ')
    edgeList = [[int(u) for u in row] for row in csvreader]
    
    
    csvreader =csv.reader(open(labelFn,'rt'),delimiter=',')
    label = [[int(u) for u in row] for row in csvreader]

    G = nx.from_edgelist(edgeList)
    nx.set_node_attributes(G,'block',dict(label))
    
    return G

    

def get_embedding(G, d):
    eA = Embed.Embed(dim=d, matrix=Embed.adjacency_matrix)
    eL = Embed.Embed(dim=d, matrix=Embed.laplacian_matrix)

    eA.embed(G)
    eL.embed(G)
    
    return eA.get_scaled(d), eL.get_scaled(d)



def kmeans_analysis(G):
    block = nx.get_node_attributes(G,'block').values()  
    
    xA, xL = get_embedding(G,2)
    
    cA,kmA,_ = k_means(xA,2)
    cB,kmL,_ = k_means(xL,2)
    
#    plt.subplot(221); plt.scatter(xA[:,0],xA[:,1],c=block)
#    plt.subplot(222); plt.scatter(xA[:,0],xA[:,1],c=kmA)
#    plt.subplot(223); plt.scatter(xL[:,0],xL[:,1],c=block)
#    plt.subplot(224); plt.scatter(xL[:,0],xL[:,1],c=kmL)

    ax = plt.subplot(121); plt.scatter(xA[:,0],xA[:,1],c=block,marker='x')
    ax.set_aspect('equal','datalim')
    lim = plt.axis()
    a = cA[0,:]-cA[1,:]
    a = np.array([1, -a[0]/a[1]])
    b = np.mean(cA,axis=0)
    x = np.array([b+a,b-a])
    plt.plot(x[:,0],x[:,1],'k--',linewidth=1)
    plt.axis(lim)
    
    ax = plt.subplot(122); plt.scatter(xL[:,0],xL[:,1],c=block,marker='x')
    ax.set_aspect('equal','datalim')
    lim = plt.axis()
    a = cB[0,:]-cB[1,:]
    a = np.array([1, -a[0]/a[1]])
    b = np.mean(cB,axis=0)
    x = np.array([b+a,b-a])
    plt.plot(x[:,0],x[:,1],'k--',linewidth=1)
    plt.axis(lim)
    
    
    
    compare_results(block,kmA,kmL)
    
    _,kmA,_ = k_means(xA,5)
    _,kmL,_ = k_means(xL,5)
    
    print "ALL FIVE"
    num_diff = vn.num_diff_w_perms(block, kmA)
    ari = adjusted_rand_score(block,kmA)
    print "Adjacency: num error="+repr(num_diff)+" ari="+repr(ari)
    
    num_diff = vn.num_diff_w_perms(block, kmL)
    ari = adjusted_rand_score(block,kmL)
    print "Laplacian: num error="+repr(num_diff)+" ari="+repr(ari)
    
    

def compare_results(block,kmA,kmL):
    blockB = [[int(b==l) for b in block] for l in xrange(6)]
    for l in xrange(5):
        print "Block "+repr(l)+" results:"
        num_diff = vn.num_diff_w_perms(blockB[l], kmA)
        ari = adjusted_rand_score(blockB[l],kmA)
        print "Adjacency: num error="+repr(num_diff)+" ari="+repr(ari)
        
        num_diff = vn.num_diff_w_perms(blockB[l], kmL)
        ari = adjusted_rand_score(blockB[l],kmL)
        print "Laplacian: num error="+repr(num_diff)+" ari="+repr(ari)
        
def cv_subgraph(A,label,v):
    knn = KNeighborsClassifier(9)
    eA = Embed.Embed(10,matrix=Embed.self_matrix)
    loo = cross_validation.LeaveOneOut(len(v))
    return np.mean(cross_validation.cross_val_score(knn, eA.embed(A[v,:][:,v]).get_scaled(),label[v], cv=loo))

def exper_subgraph(A, label, nmc=400):
    n = A.shape[0]
    nrange = np.arange(100,1350,100)
    
    res = [[cv_subgraph(A,label,np.array(random.sample(np.arange(n),nr)))
            for mc in xrange(nmc)] for nr in nrange]
    return res
        
def cv_see_subset(x, label, fr,nmc):
    knn = KNeighborsClassifier(9)
    ss = cross_validation.ShuffleSplit(x.shape[0], n_iterations=nmc, test_fraction=fr)
    return cross_validation.cross_val_score(knn, x,label, cv=ss)
    
def exper_subset(x,label,nmc=400):
    fraction = np.linspace(.1,.9,17)
    res = [cv_see_subset(x[:,:10],label,fr,nmc) for fr in fraction]
    plt.figure()
    boxplot(res,fraction)
    return res
    
def exper_knn_by_k_and_d_loo(res = None):
    if res is None:
        wiki = getWikiGraph('/Users/dpmcsuss/Dropbox/Data/WikiGraph/agen.edgelist', '/Users/dpmcsuss/Dropbox/Data/WikiGraph/label.txt')
    
        eA = Embed.Embed(50,Embed.adjacency_matrix)
        X = eA.embed(wiki).get_scaled()
        
        Y = np.array(nx.get_node_attributes(wiki,'block').values())
        
        dRange = np.arange(1,51)
        kRange = np.arange(1,18,4)
        
        res = knn_by_k_and_d_loo(X,Y,dRange,kRange)
        
    
    plot_knn_by_k_and_d(res,dRange,kRange)
    
    return res

def knn_by_k_and_d_loo(X,label, dRange, kRange):
    """ kRange = arange(1,18,4)
    dRange = arange(51)"""
    knn_list = [KNeighborsClassifier(k) for k in kRange]
    loo = cross_validation.LeaveOneOut(X.shape[0])
    
    res = np.array([[1-np.mean(cross_validation.cross_val_score(knn,X[:,:d],label,cv=loo))
                for d in dRange] for knn in knn_list])
    
    # [plt.plot(dRange,res[i,:]) for i in xrange(len(kRange))]
    
    return res

def plot_knn_by_k_and_d(res, dRange, kRange):
    c = [(10.0-k)/10.0*np.array(plt.cm.jet(k*50+30)) for k in np.arange(len(kRange))]
    [plt.plot(dRange,res[k,:],color=c[k],linewidth=k+1) for k in np.arange(len(kRange))];
    plt.legend([r'$k='+repr(k)+r'$' for k in kRange])
    
    plt.xlabel(r'$d$ --- embedding dimension')
    plt.ylabel('classification error')
    plt.tight_layout()
    
def get_neighbor_class_mat(A,Y, normalize=False):
    nclass = np.max(list(set(Y)))+1
    feature = np.zeros((A.shape[0],nclass))
    
    for u,v in zip(*A.nonzero()):
        feature[u,Y[v]] += 1
        
    if normalize:
        suminv = np.array( [0 if f==0 else 1.0/f for f in np.sum(feature,1)])
        return np.diag(suminv).dot(feature)
    
    return feature
    
def get_neighbor_class(G):
    
    # graph is 1 indexed while everything else is 0 indexed ... be careful
    block = nx.get_node_attributes(G,'block')
    nclass = len(set(block.values()))
    
    feature = np.zeros((G.number_of_nodes(), nclass))
    
    for edge in G.edges_iter():
        feature[edge[0]-1, block[edge[1]]] += 1
        feature[edge[1]-1, block[edge[0]]] += 1
    
    return feature
    

def errbarSubset():
    res = np.load('/Users/dpmcsuss/Dropbox/Data/WikiGraph/subset_res.npy')
    sigSet = np.std(res,1)
    muSet = np.mean(res,1)
    frac = np.arange(.1,.91,.05)
    
    
    plt.errorbar(frac,muSet,yerr=sigSet, fmt='k-o',markersize=7)
    plt.xlabel('fraction class label observed')
    plt.xlim([0.05,.95])
    plt.xticks(frac, rotation=45)
    plt.ylabel('classification error')
    plt.plot([0,1400],[.688,.688],'--',linewidth=1)
    plt.tight_layout()

def errbarSubgraph():
    resG = np.load('/Users/dpmcsuss/Dropbox/Data/WikiGraph/subgraph_res.npy')
    muG = np.mean(resG,1)
    sigG = np.std(resG,1)

    plt.errorbar(frac,muSet,yerr=sigSet, fmt='k-o',markersize=7)
    plt.xlabel(r'$n$ --- number of vertices in subgraph')
    plt.xlim([0.05,.95])
    plt.xticks(frac, rotation=45)
    plt.ylabel('classification error')
    plt.plot([0,1400],[.688,.688],'--',linewidth=1)
    plt.tight_layout()


if __name__ == '__main__':
    wiki = getWikiGraph('/home/dsussman/Data/WikiGraph/agen.edgelist',
                        '/home/dsussman/Data/WikiGraph/label.txt')