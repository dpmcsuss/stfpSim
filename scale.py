import networkx as nx
import numpy as np
import Embed
from metric import num_diff_w_perms
from sklearn.metrics import adjusted_rand_score

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
    
    def __init__(rgg, embed, kRange, d, scale):
        self.embed = embed
        self.kRange = kRange
        self.dRagne = dRange
        self.scale = scale
        
    def add_result_for_graph(G):
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
    
    def __init__(rgg, embed_perf):
        
        
def get_embed_perf_list():
    kRange = arange(1,10,10)
        
        