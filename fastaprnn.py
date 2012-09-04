
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors.base import _get_weights
from sklearn.utils.extmath import weighted_mode
from scipy import stats

import os
os.sys.path.extend([os.path.abspath('~/Packages/flann-1.7.1-src/src/python/')])
import pyflann
import numpy as np

class FAKNeighborsClassifier(KNeighborsClassifier):
    def __init__(self,n_neighbors=5,weights='uniform'):
        super(FAKNeighborsClassifier,self).__init__(n_neighbors=n_neighbors,weights=weights)
        
    def _fit(self,X):
        self.flann = pyflann.FLANN()
        self.flann.build_index(X)
    
    def kneighbors(self, X, n_neighbors=None, return_distance=True):
        if n_neighbors is None:
            n_neighbors = self.n_neighbors
        ind,dist = self.flann.nn_index(X,n_neighbors)

        if n_neighbors == 1:
            dist = dist[:, None]
            ind = ind[:, None]

        
        if return_distance:
            return dist,ind
        else:
            return ind

            
class FAKPseudoNeighbor(FAKNeighborsClassifier):
    def __init__(self,n_neighbors=5, weights='uniform'):
        super(FAKPseudoNeighbor,self).__init__(n_neighbors=n_neighbors,weights=weights)

    def fit(self,X,y,G):
        super(FAKPseudoNeighbor,self).fit(X,y)
        self._G = G

    def kneighbors(self, X, idx=None, n_neighbors=None, return_distance=True):
        if idx is None:
            return super(FAKPseudoNeighbor,self).kneighbors(X,n_neighbors,return_distance)
        
        if n_neighbors is None:
            n_neighbors = self.n_neighbors

        # Get the graph neighbors
        gneighbors = np.append(self._G[:,idx].nonzero()[0],idx)
        
        # Get enough neighbors so taht we are sure we'll have n_neighbors non graph neighbors
        ind,dist = self.flann.nn_index(X,n_neighbors+gneighbors.shape[0])
            
        #find the ones that aren't graph neighbors
        notGneighbor = np.equal(np.in1d(ind, gneighbors,True),False)
        
        ind = ind[notGneighbor][:n_neighbors]
        dist = dist[notGneighbor][:n_neighbors]
        
        if n_neighbors == 1:
            dist = dist[:, None]
            ind = ind[:, None]
        else:
            perm = np.random.permutation(n_neighbors)
            ind = ind[perm]
            dist = dist[perm]

        
        if return_distance:
            return dist,ind
        else:
            return ind

    def predict(self, X, idx=None):

        neigh_dist, neigh_ind = self.kneighbors(X,idx)
        pred_labels = self._y[neigh_ind]

        weights = _get_weights(neigh_dist, self.weights)

        if weights is None:
            mode, _ = stats.mode(pred_labels, axis=1)
        else:
            # Randomly permute the neighbors to tie-break randomly if necessary
            perm = np.random.permutation(n_neighbors)
            ind = ind[perm]
            mode, _ = weighted_mode(pred_labels,weights,axis)
            
        return mode.flatten().astype(np.int)
