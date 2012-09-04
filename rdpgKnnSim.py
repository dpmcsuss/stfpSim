import Embed
import adjacency
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import cross_validation
from matplotlib import pyplot as plt

def boxplot(data,pos, c='black'):    
        
    width = np.min(pos[1:]-pos[:-1])*.7
    bp = plt.boxplot(data, positions=pos, notch=1, widths=width, sym='+', vert=1, whis=1.5)
    plt.setp(bp['boxes'], color=c)#,linewidth=2)
    plt.setp(bp['whiskers'], color=c)#,linewidth=1)
    plt.setp(bp['medians'], color=c)#,linewidth=1)
    plt.setp(bp['fliers'], color=c, marker='+')
    
    plt.xlim([np.min(pos)-width,np.max(pos)+width])

def get_rdpg_sim_res(rdpg, embed, k):
    rdpg.generate_adjacency()
    Xhat = Embed.procrustes(rdpg.X, embed.embed(rdpg.Adj).get_scaled())
    #Y =  rdpg.block_assignment
    knn = KNeighborsClassifier(k)
    loo = cross_validation.LeaveOneOut(rdpg.n_nodes)
    
    X = rdpg.X
    Y = (X[:,0]>X[:,1]).astype(float)
    Xhat = Embed.procrustes(X,embed.embed(rdpg.Adj).get_scaled(),scale=False, center=False)
    
    
    result = np.zeros(1, dtype=[('X',np.float32),('Xhat',np.float32),
        ('gstarX',np.float32),('gstarXhat',np.float32),('sqerr',np.float32),('n',np.int)])
    
    result['X'] = 1-np.mean(cross_validation.cross_val_score(knn, X,Y, cv=loo))
    result['Xhat'] = 1-np.mean(cross_validation.cross_val_score(knn, Xhat,Y, cv=loo))
    result['gstarX'] = 1-np.mean(np.equal(X[:,0]>X[:,1],Y))
    result['gstarXhat'] = 1-np.mean(np.equal(Xhat[:,0]>Xhat[:,1],Y))
    result['sqerr'] = np.linalg.norm(X-Xhat,'fro')
    
    result['n'] = rdpg.n_nodes
    
    return result

def get_rdpg_sim_mc(rdpg,embed,k, mc=1):
    return np.concatenate([get_rdpg_sim_res(rdpg,embed,k) for _ in xrange(mc)])
    
a1 = np.array([1, 4, 2])
a2 = np.array([4, 1, 2])
dir1 = lambda n: np.random.dirichlet(a1,n)[:,:2]
dir2 = lambda n: np.random.dirichlet(a2,n)[:,:2]
n=1000

rdpg_n = lambda n: adjacency.RDPGraph(n,array([.5,.5]),[dir1,dir2])

def expLstar0_v1():
    nrange = np.arange(100,1600,100)
    eA = Embed.Embed(2,Embed.self_matrix)
    dir1 = lambda n: np.random.dirichlet(np.array(2*[1,1,1]),n)[:,:2]
    rdpg_n  = lambda n: adjacency.RDPGraph(n,np.array([1]),[dir1])
    res = np.concatenate([get_rdpg_sim_mc(rdpg_n(n),eA,5,500) for n in nrange]);
    np.save('/Users/dpmcsuss/Data/rdpgKnn_dir222_Lstar=0.npy',res)
    
    
def expLstar0_v2():
    nrange = np.arange(100,2100,100)
    eA = Embed.Embed(2,Embed.self_matrix)
    dir1 = lambda n: np.random.dirichlet(np.array(2*[1,1,1]),n)[:,:2]
    rdpg_n  = lambda n: adjacency.RDPGraph(n,np.array([1]),[dir1])
    res = np.concatenate([get_rdpg_sim_mc(rdpg_n(n),eA,int(np.sqrt(n)/4)*2+1,500) for n in nrange]);
    np.save('/Users/dpmcsuss/Data/rdpgKnn_dir222_Lstar=0_v2.npy',res)
    
    return res
    
    
def errorBarXvsXhat():
    res = np.load('/Users/dpmcsuss/Data/rdpgKnn_dir222_Lstar=0_v2.npy')
    nrange = np.arange(100,2100,100)
    muX = np.array([np.mean(res['X'][res['n']==n]) for n in nrange])
    muXhat = np.array([np.mean(res['Xhat'][res['n']==n]) for n in nrange])
    sigX = np.array([np.std(res['X'][res['n']==n]) for n in nrange])
    sigXhat = np.array([np.std(res['Xhat'][res['n']==n]) for n in nrange])
    plt.errorbar(nrange,muXhat,yerr=sigXhat, fmt='k-s')
    plt.errorbar(nrange,muX,yerr=sigX, fmt='r-.o')
    
    plt.yscale('log');
    plt.ylim([10**(-2.2),1]);
    plt.xlim([0,2100]);
    plt.plot([0,2100],[.5,.5],'--',linewidth=1)
    
    plt.ylabel(r'classification error')
    plt.xlabel(r'$n$ --- number of vertices')
    
    plt.legend((r'$\hat{\mathbf{X}}$',r'$\mathbf{X}$','Chance'))
    
    plt.tight_layout()