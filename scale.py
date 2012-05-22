import networkx as nx
import numpy as np

from metric import num_diff_w_perms
import Embed
from sklearn.metrics import adjusted_rand_score

from itertools import product
from joblib import delayed, Parallel

from rpy2.robjects import r
from rpy2.robjects import numpy2ri
numpy2ri.activate()


class EmbedPerformance:
    embed = None
    kRange = np.linspace(1,10,10)
    d = None
    scale  = None
    mclustModels = []
    
    mcr = []
    ari = []
    error = []
    kHat = []
    
    def __init__(self,embed, kRange, d, scale):
        self.embed = embed
        self.kRange = kRange
        self.dRagne = dRange
        self.scale = scale
        
    def add_result_for_graph(self,G):
        embed.embed(G);
        x=embed.get_embedding(self.d,self.scale)
        r.library('mclust')
        mcr = r.Mclust(x,G=self.kRange)
        self.mcr.append(mcr)
        self.khat.append(mcr['G'])
        
        trueLabel = nx.get_node_attribute(G,'block')
        predLabel = np.array(mcr['classification']).astype(int)-1
        
        self.ari.append(adjusted_rand_score(trueLabel,predLabel))
        self.error.append(num_diff_w_perms(trueLabel,predLabel))
    
    
def add_embed_performance_result(ep, G):
    ep.add_result_for_graph(G)
    
    
class SBMPerformance:
    embed_perf = []
    rgg = []
    
    def __init__(self,rgg, embed_perf):
        self.rgg = rgg
        self.embed_perf = embed_perf
        
    def add_mc_results(self,nmc):
        [add_all_results(sbmp, G) for G in rgg.iter_graph(nmc)]
        
        
def add_all_results(sbmp, G):
    Parallel(n_jobs=2)(delayed(add_embed_performance_result)(ep,G)
                       for ep in sbmp.embed_perf)
    
def get_embed_perf_list():
    kRange = np.arange(1,11,1,dtype=int) # [1,2,...,10]
    dRange = np.array([1,2,5,10])
    kTrue = np.array([2])
    scale = [True, False]
    
    embed = [Embed.Embed(np.max(dRange),Embed.adjacency_matrix),
             Embed.Embed(np.max(dRange),Embed.laplacian_matrix),
             Embed.Embed(np.max(dRange),Embed.marchette_matrix)]
    
    ep_list = [EmbedPerformance(e,kRange,d,s)
                for e,d,s in product(embed,d,scale)]
    ep_list.extend([[EmbedPerformance(e,kTrue,d,s)
                for e,d,s in product(embed,d,scale)]])
    
    
        