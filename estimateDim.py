import networkx as nx
import giorgio as br
import Embed
import numpy as np

from RandomGraph import SBMGenerator

eA = Embed.adjEmbed;
eA.dim = 10

rgglist = [SBMGenerator(br.rgg.block_prob, (br.rgg.nvec/50)*k, directed=True) for k in np.arange(1,11)]
    
nlist = np.array([np.sum(rggk.nvec) for rggk in rgglist])

svals = np.array([[eA.embed(G).sval for G in rggk.iter_graph(10)] for rggk in rgglist])

for d in xrange(10):
    
    bp = plot.boxplot(svals[d,:,:], positions=nlist+5*d, notch=1, sym='+', vert=1, whis=1.5,hold=True,widths=10)
    plot.setp(bp['boxes'], color='black',linewidth=2)
    plot.setp(bp['whiskers'], color='black',linewidth=1)
    plot.setp(bp['medians'], color='black',linewidth=1)
    plot.setp(bp['fliers'], color='black',markersize=2)