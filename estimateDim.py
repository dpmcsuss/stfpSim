import networkx as nx
import giorgio as br
import Embed
import numpy as np
from vertexNomination import mclust_performance as mclust
from RandomGraph import SBMGenerator
from matplotlib import pyplot as plt

dRange = np.arange(1,15)
eA = Embed.Embed(dim=max(dRange),matrix=Embed.marchette_matrix, directed=True)
rgglist = [SBMGenerator(br.rgg.block_prob, (br.rgg.nvec/50)*k, directed=True) for k in np.arange(1,11)]
nlist = np.array([np.sum(rggk.nvec) for rggk in rgglist])



res = np.array([[(eA.embed(G).sval,[mclust(rggk.label,eA.get_scaled(d)) for d in dRange])
                for G in rggk.iter_graph(1)]
                for rggk in rgglist])
svals = res[:,:,0]
mcr = res[:,:,1]


def plot_svals(svals):
    for d in xrange(10):
        bp = plt.boxplot(svals[:,:,d].T, positions=nlist+10*d, notch=1, sym='+', vert=1, whis=1.5,hold=True,widths=10)
        plt.setp(bp['boxes'], color='black',linewidth=2)
        plt.setp(bp['whiskers'], color='black',linewidth=1)
        plt.setp(bp['medians'], color='black',linewidth=1)
        plt.setp(bp['fliers'], color='black',markersize=2)
        
    plot(nlist, 2*sig*sqrt(nlist))
    plot(nlist, 2.3*sig*sqrt(nlist))