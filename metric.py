from itertools import permutations
import hungarian
import numpy as np

def num_diff_w_perms(l1, l2):
    label1 = list(set(l1))
    label2 = list(set(l2))
    label  = label1
    n = len(l1)
    
    cost = [[n-np.sum((l1==lab1)==(l2==lab2)) for lab1 in label1] for lab2 in label2]
    
    h = hungarian.Hungarian(cost)
    try:
        h.calculate()
        return h.getTotalPotential()/2.0
    except hungarian.HungarianError:
        print 'Hungary lost this round'
        min_diff = np.Inf
        for p in permutations(label):
            l1p = [p[label.index(l)] for l in l1]
            min_diff = min(min_diff,metrics.zero_one(l1p, l2))
        return min_diff    
        
