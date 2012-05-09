'''
Created on Mar 28, 2012

@author: dsussman
'''
import csv
import networkx as nx
import Embed
from sklearn.cluster import k_means
from sklearn.metrics import adjusted_rand_score
import vertexNomination as vn
from matplotlib import pyplot as plt
import numpy as np

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


def analysis(G):
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

if __name__ == '__main__':
    wiki = getWikiGraph('/home/dsussman/Data/WikiGraph/agen.edgelist',
                                  '/home/dsussman/Data/WikiGraph/label.txt')