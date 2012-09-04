from scipy.optimize import fmin_ncg
from scipy.spatial.distance import squareform
import numpy as np

def loglike(x, A):
    P = x.dot(x.T)
    P = squareform(P-diag(diag(P)))
    
    B = squareform(A)
    
    return np.