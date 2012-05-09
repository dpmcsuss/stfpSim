'''
Created on Feb 19, 2012

@author: dsussman
'''
import networkx as nx
import numpy as np
import RandomGraph as rg
import matplotlib.pyplot as plot
from itertools import permutations, izip, cycle
from scipy import stats
from sklearn import metrics
import Embed
from sklearn.grid_search import IterGrid
from sklearn.cluster import KMeans
import pickle
import dpplot
import adjacency
import hungarian
#from joblib import Parallel, delayed



class AffiliationMonteCarlo(object):
    nmc = 0
    nnodes = []
    dim = []
    scale = []
    matrix = []
    max_dim = 0
    rho = np.array([.5, .5])
    block_prob = np.array([[.3, .1],[.1,.2]])
    k=2
    p=.3
    q=.1
    
    rgg = rg.SBMGenerator(np.array([[.3, .1],[.1,.1]]),np.array([50, 50],dtype=int) )
    
    results = []
    
    def __init__(self, nmc, nnodes, dim, scale, matrix):
        """Constructor for AffiliationMonteCarlos
        
        Parameters
        ----------
        nmc -- integer denoting how many times to run the MC
        nnodes -- list of number of nodes to simulate
        scales -- list of scales. Possibilities: [True], [False], [True,False]
        dim -- list of embedding dimensions
        matrix -- dictionary of where the values are functions which return a matrix from
        
        This does a monte carlo simulation for each possible parameter value
        """
        self.nmc = nmc
        self.nnodes = nnodes
        self.dim = dim
        if isinstance(dim, np.int):
            self.max_dim = dim
        else: 
            self.max_dim = np.max(dim)
        self.scale =  scale
        self.matrix = matrix
    
    def get_embed_param(self, nnodes):
        param = list(IterGrid({'dim':self.dim,'scale':self.scale, 'matrix': self.matrix.keys()}))
        embed = self.matrix.copy()
        [embed.update({key:Embed.Embed(self.max_dim, self.matrix[key])}) for key in self.matrix]
        for p in param:
            p.update({'nnodes':nnodes, 'num_diff':np.zeros((self.nmc)),
                      'rand_idx':np.zeros((self.nmc)),
                      'embed':embed[p['matrix']]})
            
        return param
    
    def get_random_graph(self, nnodes):
        #return rg.affiliation_model(nnodes, self.k, self.p, self.q)
        return rg.SBM(self.rho.dot(nnodes).astype(int), self.block_prob, directed=False)
    
    def run_monte_carlo(self, pickle_fn=None, init=False):
        if init:
            self.results = []
        for nnodes in self.nnodes:
            print 'Running MC n='+repr(nnodes)
            embed_param = self.get_embed_param(nnodes)
            for mc in xrange(self.nmc):
                G = self.get_random_graph(nnodes)
                for epar in embed_param:
                    embed = epar['embed'] 
                    x = embed.embed(G)
                    x = embed.get_embedding(epar['dim'], epar['scale'])
                    
                    k_means = KMeans(init='k-means++', k=self.k, n_init=5)
                    pred = k_means.fit(x).labels_
                    epar['num_diff'][mc] = num_diff_w_perms(
                                                nx.get_node_attributes(G, 'block').values(), pred)
                    epar['rand_idx'][mc] = metrics.adjusted_rand_score(
                                                nx.get_node_attributes(G, 'block').values(), pred)
            [epar.pop('embed') for epar  in embed_param] # pop off the Embedding to save space
            self.results.extend(embed_param)
            if pickle_fn:
                pickle.dump(self, open(pickle_fn,'wb'))
                print 'Saved to '+pickle_fn
        return self.results
     
    def combine(self,amc):
        for r,rNew in zip(self.results,amc.results):
            r['num_diff'] = np.append(r['num_diff'], rNew['num_diff'])
            
        
    def plot_num_diff_vs_d(self, nnodes):
        if nnodes not in self.nnodes:
            print "Woahh: we didn't do that number of nodes"
            return
        if not self.results:
            return "Get some results first ... run_monte_carlo()"
        
        plot_bw = dpplot.plot_bw()
        
        
        results = [r for r in self.results if r['nnodes']==nnodes]
        params = IterGrid({'matrix':self.matrix.keys(), 'scale':self.scale})
        for p in params:
            data,dim = zip(*[(np.mean(r['num_diff']), r['dim'])
                            for r in results if r['matrix']==p['matrix'] and r['scale']==p['scale'] ])
            
            line_label = p['matrix']+(' (Scaled)' if p['scale'] else ' (Unscaled)') 
            #legend.append(p['matrix']+"; Scale:"+repr(p['scale']))
            plot_bw.plot(dim, data, label=line_label)
            
        plot.legend(loc='best')
        plot.ylabel("Mean Number Errors")
        plot.xlabel("Embedding Dimension")
        plot.show()
        
#        plot.boxplot(data, notch=1, sym='+', vert=1, whis=1.5)
#        plot.show()
#        for r in results:
#            print "d="+repr(r['dim'])+", matrix="+r['embed'].func_name+", scale="+repr(r['scale']),
#            print ": mean diff="+repr(np.mean(r['num_diff']))
        
        
    def plot_num_diff_vs_n(self, dim, matrix, scale):
        param = IterGrid({'dim':dim, 'matrix':matrix,'scale':scale})
        plot_bw = dpplot.plot_bw()
        
        for p in param:
            data,nnodes = zip(*[(np.mean(r['num_diff'])/r['nnodes'], r['nnodes'])
                               for r in self.results if r['matrix']==p['matrix'] 
                                                    and r['scale']==p['scale'] 
                                                    and r['dim']==p['dim']])
            line_label =  p['matrix']+(' (Scaled)' if p['scale'] else ' (Unscaled)')
            
            style = ('-' if p['matrix']=='Adjacency' else '--')
            color = ('b' if p['matrix']=='Adjacency' else 'r')
            marker = 's' if p['scale'] else 'd'
            markersize = 10
            plot.plot(nnodes, data, 
                      linestyle=style,marker=marker,
                      markersize=markersize, linewidth=2, color=color,label=line_label)
            
            #plot.plot(nnodes, data, label=line_label)
            
        plot.legend(loc='best')
        plot.ylabel("Percent Error")
        plot.xlabel(r'$n$ - Number of vertices')
        plot.show()
        
    def comp_vs_n(self,comp, dim, matrix, scale):
        """Each input is a pair"""
        for n in self.nnodes:
            data0 = [r['num_diff']/r['nnodes']
                               for r in self.results if r['matrix']==matrix[0] 
                                                    and r['scale']==scale[0] 
                                                    and r['dim']==dim[0]
                                                    and r['nnodes']==n][0]
            
            data1 = [r['num_diff']/r['nnodes']
                               for r in self.results if r['matrix']==matrix[1] 
                                                    and r['scale']==scale[1] 
                                                    and r['dim']==dim[1]
                                                    and r['nnodes']==n][0]
            
            stat,pval = comp(data0,data1)
            print "nnodes="+repr(n)+", stat="+repr(stat)+", pval="+repr(pval)
                                        
            
        
    def plot_num_diff_vs_n_ari(self, dim, matrix, scale):
        param = IterGrid({'dim':dim, 'matrix':matrix,'scale':scale})
        plot_bw = dpplot.plot_bw()
        
        for p in param:
            data,nnodes = zip(*[(np.mean(r['rand_idx'])/r['nnodes'], r['nnodes'])
                               for r in self.results if r['matrix']==p['matrix'] 
                                                    and r['scale']==p['scale'] 
                                                    and r['dim']==p['dim']])
            line_label =  p['matrix']+(' (Scaled)' if p['scale'] else ' (Unscaled)') + ' dim.: '+repr(p['dim'])
            
            
            
            #plot.plot(nnodes, data, label=line_label)
            
        plot.legend(loc='best')
        plot.ylabel("Percent Error")
        plot.xlabel(r'$n$ - Number of vertices')
        plot.show()
        
#    def plot_figure1(self, dim): 
#        for scale in [True]:
#            
#            for d in dim:
#                data,nnodes = zip(*[(np.mean(r['num_diff'])/r['nnodes'], r['nnodes'])
#                                   for r in self.results if r['matrix']=='Adjacency' 
#                                                        and r['scale']==scale
#                                                        and r['dim']==d])
#                label =  'R='+repr(d)
#                style = (':' if scale else '--')
#                color = ('r' if scale else 'b')
#                marker = 'o' if d==1 else (d, 1,0)
#                plot.plot(nnodes, data, 
#                          linestyle=style,marker=marker,
#                          markersize=15, linewidth=2, color='k',label=label)
#        plot.yscale('log')
#        plot.xlabel(r'$n$ - Number of vertices')
#        plot.ylabel('Percent Mis-assignment Error')
#        plot.legend(loc='best',prop={'size':'medium'})
#        plot.show()
#        
    def plot_figure1(self, dim,useColor=True): 
        for scale in [True]:
            
            for d in dim:
                data,nnodes = zip(*[(np.mean(r['num_diff'])/r['nnodes'], r['nnodes'])
                                   for r in self.results if r['matrix']=='Adjacency' 
                                                        and r['scale']==scale
                                                        and r['dim']==d])
                label = 'R='+repr(d)
                style = (':' if scale else '--')
                color = (('r' if scale else 'b') if useColor else 'k')
                marker = 'o' if d==1 else (d, 1,0)
                markersize = (7 if d==1 else 15)
                plot.plot(nnodes, data, 
                          linestyle=style,marker=marker,
                          markersize=markersize, linewidth=2, color='k',label=label)
        plot.yscale('log')
        plot.xlabel(r'$n$ - Number of vertices')
        plot.ylabel('Percent Mis-assignment Error')
        plot.legend(loc='best',prop={'size':'medium'})
        plot.show()

    def plot_figure1_boxplot(self, dim):
        for scale in [True]:
            
            for d in dim:
                data,nnodes = zip(*[(r['num_diff']/r['nnodes'], r['nnodes'])
                                   for r in self.results if r['matrix']=='Adjacency' 
                                                        and r['scale']==scale
                                                        and r['dim']==d])
                mean = np.array([np.median(datum) for datum in data])
                data = np.array(data)
                label = 'dim='+repr(d) #('Scaled' if scale else 'Unscaled')+
                style = (':' if scale else '--')
                color = ('r' if scale else 'b')
                marker = 'd' if d==1 else (d, 1,0)
                
                bp = plot.boxplot(data, positions=np.array(nnodes)+d, notch=1, sym='+', vert=1, whis=1.5,hold=True,widths=10)
                plot.setp(bp['boxes'], color='black',linewidth=2)
                plot.setp(bp['whiskers'], color='black',linewidth=1)
                plot.setp(bp['medians'], color='black',linewidth=1)
                plot.setp(bp['fliers'], color='black', marker=marker,markersize=2)
                plot.plot(nnodes, mean, 
                          linestyle=style,marker=marker,
                          markersize=15, linewidth=2, color='k',label=label)
                
             
        plot.yscale('log')#,linthreshy=(0,10**-4))
        plot.xlabel(r'$n$ - Number of vertices')
        plot.ylabel('Percent Mis-assignment Error')
        plot.legend(loc='best',prop={'size':'medium'})
        plot.show()
        
            
#           
#def _get_mc_results(self, rgg, amc):
#    embed_param = amc.get_embed_param(nnodes)
#    for G in rgg.
#        G = amc.get_random_graph(nnodes)
#        for epar in embed_param:
#            embed = epar['embed'] 
#            x = embed.embed(G)
#            x = embed.get_embedding(epar['dim'], epar['scale'])
#            
#            k_means = KMeans(init='k-means++', k=self.k, n_init=10)
#            pred = k_means.fit(x).labels_
#            epar['num_diff'][mc] = num_diff_w_perms(
#                                        nx.get_node_attributes(G, 'block').values(), pred)
#            epar['rand_idx'][mc] = metrics.adjusted_rand_score(
#                                        nx.get_node_attributes(G, 'block').values(), pred)
#    return results
    

def embedding_vs_dimension_performance():
    n = 50
    drange = np.arange(1,5)
    
    embed = [Embed.dot_product_embed,
         Embed.dot_product_embed_unscaled,
         Embed.normalized_laplacian_embed,
         Embed.normalized_laplacian_embed_scaled]
    
    nmc = 10   
    k = 2
    p = .5
    q = .1
    
    all_params = list(IterGrid({'d':drange, 'embed':embed}))
    [param.update({'num_diff':np.zeros(nmc),'rand_idx':np.zeros(nmc)}) for param in all_params]
    
    for mc in np.arange(nmc):
        print mc
        G = rg.affiliation_model(n, k, p, q)
        truth = nx.get_node_attributes(G, 'block').values()
        for param in all_params:
            pred = Embed.cluster_vertices_kmeans(G, param['embed'], param['d'], 2)
            param['num_diff'][mc] = num_diff_w_perms(truth, pred)
            param['rand_idx'][mc] = metrics.adjusted_rand_score(truth, pred)
    return all_params
    

def k_estimation_monte_carlo(rgg_list, nmc,altdim):
    results = []
    results_over = []
    
    for rgg in rgg_list:
        k = len(rgg.nvec)
        dim = np.linalg.matrix_rank(rgg.block_prob)
        embed = Embed.Embed(dim, Embed.adjacency_matrix)
        xi,lb,ub = zip(*[get_xi_bnd(G, embed,k) for G in rgg.iter_graph(nmc)])
        res_dict = rgg.get_param_dict()
        res_dict.update({'xi':xi,'xi_lb':lb, 'xi_ub':ub})
        results.append(res_dict)
        
        # Now try it for over-estimated dimension
        #dim *=2
        dim = altdim 
        embed.dim = dim
        xi,lb,ub = zip(*[get_xi_bnd(G, embed,k) for G in rgg.iter_graph(nmc)])
        res_dict = rgg.get_param_dict()
        res_dict.update({'xi':xi,'xi_lb':lb, 'xi_ub':ub})
        results_over.append(res_dict)
        
    plot_k_estimation_results(results, results_over)
    return results, results_over
    
def k_estimation_adj_monte_carlo(nnode, G, nmc,altdim):
    results = []
    results_over = []
    
    k = len(G.nvec)
    print k
    
    for n in nnode:
        G.n_nodes = n
        dim = np.linalg.matrix_rank(G.P)
        embed = Embed.Embed(dim, adjacency.Graph.get_adjacency)
        xi,lb,ub = zip(*[get_xi_bnd_adj(Gmc, embed,k) for Gmc in G.iter_mc(nmc,size_condition=True)])
        res_dict = {'nnodes':n, 'xi':xi,'xi_lb':lb, 'xi_ub':ub}
        results.append(res_dict)
        
        # Now try it for alternate dimension
        dim = altdim 
        embed.dim = altdim
        xi,lb,ub = zip(*[get_xi_bnd_adj(Gmc, embed,k) for Gmc in G.iter_mc(nmc,size_condition=True)])
        res_dict = {'nnodes':n, 'xi':xi,'xi_lb':lb, 'xi_ub':ub}
        results_over.append(res_dict)
        
    plot_k_estimation_results(results, results_over)
    return results, results_over

def get_xi_bnd_adj(G, embed,k):
    embed.embed(G,fast=False)
    x = embed.get_scaled()
    k_means = KMeans(init='k-means++', k=k+1, n_init=10)
    lb = np.log(k_means.fit(x).inertia_)/(2*np.log(G.n_nodes))
    k_means = KMeans(init='k-means++', k=k, n_init=10)
    xi = np.log(k_means.fit(x).inertia_)/(2*np.log(G.n_nodes))
    k_means = KMeans(init='k-means++', k=k-1, n_init=10)
    ub = np.log(k_means.fit(x).inertia_)/(2*np.log(G.n_nodes))
    return xi,lb,ub


def plot_k_estimation_results(results, results_over):
    line = cycle( ["--","-",":"])
    mean_xi = [np.mean(r['xi']) for r in results]
    mean_xi_ub = [np.mean(r['xi_ub']) for r in results]
    mean_xi_lb = [np.mean(r['xi_lb']) for r in results]
    if 'nvec' in results[0].keys():
        n = [np.sum(r['nvec']) for r in results]
    else:
        n = [r['nnodes'] for r in results]
    
    plot.subplot(1,2,1)

    [plot.plot(n,mx,color='k',linestyle=line.next(),marker='s',markersize=10) 
          for mx in [mean_xi_ub, mean_xi, mean_xi_lb]];
    plot.xlabel(r'$n$ - Number of vertices')
    plot.ylabel(r"$\log(\|\mathcal{C}_{K'}-X\|_F)/log(n)$ - Normalized Square Error")
    #plot.title(r'$R=\mathrm{rank}(M)$')
    
    mean_xi = [np.mean(r['xi']) for r in results_over]
    mean_xi_ub = [np.mean(r['xi_ub']) for r in results_over]
    mean_xi_lb = [np.mean(r['xi_lb']) for r in results_over]
    
    plot.subplot(1,2,2)
    [plot.plot(n,mx,color='k',linestyle=line.next(),marker='s',markersize=10) 
          for mx in [mean_xi_ub, mean_xi, mean_xi_lb]];
    plot.xlabel(r'$n$ - Number of vertices')
    plot.ylabel(r"$\log(\|\mathcal{C}_{K'}-X\|_F)/log(n)$ - Normalized Square Error")
    plot.legend([r"$K'=K-1$",r"$K'=K$",r"$K'=K+1$"],loc='best')
    
    plot.title(r'$R=2\mathrm{rank}(M)$')
    
#
#def plot_k_estimation_results(results, results_over):
#    pbw = dpplot.plot_bw()
#    mean_xi = [np.mean(r['xi']) for r in results]
#    mean_xi_ub = [np.mean(r['xi_ub']) for r in results]
#    mean_xi_lb = [np.mean(r['xi_lb']) for r in results]
#    n = [np.sum(r['nvec']) for r in results]
#    
#    plot.subplot(1,2,1)
#    [pbw.plot(n,mx,'') for mx in [mean_xi_lb, mean_xi, mean_xi_ub]];
#    plot.xlabel(r'$n$ - Number of vertices')
#    plot.ylabel(r'$\log(\|\mathcal{C}-X\|_F)/log(n)$')
#    plot.title(r'Embed to $\mathrm{rank}(M)$')
#    
#    pbw = dpplot.plot_bw()
#    mean_xi = [np.mean(r['xi']) for r in results_over]
#    mean_xi_ub = [np.mean(r['xi_ub']) for r in results_over]
#    mean_xi_lb = [np.mean(r['xi_lb']) for r in results_over]
#    
#    plot.subplot(1,2,2)
#    [pbw.plot(n,mx,'') for mx in [mean_xi_lb, mean_xi, mean_xi_ub]];
#    plot.xlabel(r'$n$ - Number of vertices')
#    plot.ylabel(r'$\log(\|\mathcal{C}-X\|_F)/\log(n)$')
#    plot.legend([r"$K'=K+1$",r"$K'=K$",r"$K'=K-1$"])
#    
#    plot.title(r'Embed to $2\mathrm{rank}(M)$')
#    


def get_xi_bnd(G, embed,k):
    embed.embed(G)
    x = embed.get_scaled()
    k_means = KMeans(init='k-means++', k=k+1, n_init=10)
    lb = np.log(k_means.fit(x).inertia_)/(2*np.log(G.number_of_nodes()))
    k_means = KMeans(init='k-means++', k=k, n_init=10)
    xi = np.log(k_means.fit(x).inertia_)/(2*np.log(G.number_of_nodes()))
    k_means = KMeans(init='k-means++', k=k-1, n_init=10)
    ub = np.log(k_means.fit(x).inertia_)/(2*np.log(G.number_of_nodes()))
    return xi,lb,ub
    
    
def simulate_affiliation_dpe():
    nrange = [400] #50*2**np.arange(3)
    drange = np.arange(1,5)
    
    embed = [Embed.dot_product_embed,
             Embed.dot_product_embed_unscaled,
             Embed.normalized_laplacian_embed,
             Embed.normalized_laplacian_embed_scaled]
    
    k = 2
    p = .15
    q = .1
    
    for n in nrange:
        G = rg.affiliation_model(n, k, p, q)
        for d in drange:
            print n*k,d,
            for e in embed:
                Embed.cluster_vertices_kmeans(G, e, d, k, 'kmeans')
                print num_diffs_w_perms_graph(G, 'block', 'kmeans'),
                
            print
    
    plot.matshow(nx.adj_matrix(G))
    plot.show()
    
def num_diff_w_perms(l1, l2):
    label1 = list(set(l1))
    label2 = list(set(l2))
    label  = label1
    n = len(l1)
    
#    cost = [[n-np.sum((l1==lab1)==(l2==lab2)) for lab1 in label1] for lab2 in label2]
#    
#    h = hungarian.Hungarian(cost)
#    try:
#        h.calculate()
#        return h.getTotalPotential()/2.0
#    except hungarian.HungarianError:
#        print 'Hungary lost this round'
#        return  
    
    min_diff = np.Inf
    for p in permutations(label):
        l1p = [p[label.index(l)] for l in l1]
        min_diff = min(min_diff,metrics.zero_one(l1p, l2))
    return min_diff    

    
def num_diffs_w_perms_graph(G, attr1, attr2):
    l1 = nx.get_node_attributes(G, attr1).values()
    l2 = nx.get_node_attributes(G, attr2).values()
    
    return num_diff_w_perms(l1, l2)

def plotEmbedComparison(G, e1, e2):
    e1.embed(G)
    e2.embed(G)
    x1 = e1.get_scaled(2)
    x2 = e2.get_scaled(2)
    
    x2p = Embed.procrustes(x1, x2)
    
    #block = nx.get_node_attributes(G, 'block').values()
    
    #plot.subplot(121);
    plot.scatter(x1[:,0],x1[:,1],c='r')
    #plot.subplot(122);
    plot.scatter(x2p[:,0],x2p[:,1],c='b')


def doniell_param():
    block_prob =np.array( [ [.10 , .13 , .11 , .06 , .09 , .15 , .08 , .20 , .12 , .13 ],
                            [.13 , .25 , .17 , .15 , .18 , .24 , .23 , .26 , .21 , .34 ],
                            [.11 , .17 , .13 , .09 , .12 , .18 , .13 , .22 , .15 , .20 ],
                            [.06 , .15 , .09 , .10 , .11 , .13 , .16 , .12 , .12 , .23 ],
                            [.09 , .18 , .12 , .11 , .13 , .17 , .17 , .18 , .15 , .25 ],
                            [.15 , .24 , .18 , .13 , .17 , .25 , .19 , .30 , .21 , .29 ],
                            [.08 , .23 , .13 , .16 , .17 , .19 , .26 , .16 , .18 , .37 ],
                            [.20 , .26 , .22 , .12 , .18 , .30 , .16 , .40 , .24 , .26 ],
                            [.12 , .21 , .15 , .12 , .15 , .21 , .18 , .24 , .18 , .27 ],
                            [.13 , .34 , .20 , .23 , .25 , .29 , .37 , .26 , .27 , .53 ] ] )
    rho = np.array([.09 , .08 , .10 , .11 , .09 , .08, .10 , .11 , .12 , .12 ])
    
    return (block_prob,rho)

if __name__ == '__main__':
#    block_prob = np.random.rand(5,5)
#    block_prob = np.triu(block_prob)+np.transpose(np.triu(block_prob,1))
#    block_prob = np.array([[ 0.5  ,  0.2  ,  0.15 ,  0.15 ],
#                           [ 0.2  ,  0.5  ,  0.15 ,  0.15 ],
#                           [ 0.15 ,  0.15 ,  0.475,  0.225],
#                           [ 0.15 ,  0.15 ,  0.225,  0.475]])
#        
#    matrices = {'Adjacency':Embed.adjacency_matrix, 'Laplacian':Embed.laplacian_matrix}
#
#    amc = AffiliationMonteCarlo(10, [400], np.arange(1,16), [True, False], matrices)
#    
#    amc.block_prob = block_probhttp://download.enthought.com/epd_7.2/epd-7.2-2-rh5-x86_64.sh
#    amc.rho = np.array([.3, .2, .3, .2])
#    amc.k = 4
#    amc.nmc = 1
#    
#    results = amc.run_monte_carlo()
#    amc.plot_num_diff_vs_d(400)
    block_prob,rho = doniell_param()
    
    #amc = pickle.load(open('/home/dsussman/Data/stfp_sims/donniel_sim_v0.1.pickle','rb'))
    amc = pickle.load(open('/home/dsussman/Data/stfp_sims/minh_example_v0.1.pickle','rb'))
    amc.nmc = 500
    amc.run_monte_carlo(init=True) 
    
    rgg_list = rgg_list = [rg.SBMGenerator(block_prob,np.array(rho*n).astype(int)) for n in np.arange(500,1100,100)]
    kEst,kEstAltDim = k_estimation_monte_carlo(rgg_list,1,10)
    plot_k_estimation_results(kEst,kEstAltDim)