from matrix import build_matrix
import json
from matrix import integer_inequality
import numpy as np
from scipy.sparse import coo_matrix
from matplotlib import pyplot as plt
import scipy.sparse
from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg import eigs
import multiprocessing as mp
from functools import partial

def worker(q, instance=None):
    instance.random_rewiring(q)
    lam = instance.second_largest_eigenvalue(8, 1.2)/instance.degree
    return lam

def main(q_values, k_values, name, n, parallel=False):
    dictionary = {str(k):{str(q):[] for q in q_values} for k in k_values}
    for k in k_values:
        z = build_matrix('1d_ring_1000', np.array([1000, 1, 1]), (k/2))
        #print(z.all_indices_list)
        z.all_indices()
        z.one_int_index_tuples_and_adjacency()
        z.Laplacian_0()
        for q in q_values:
            # do the same thing n times
            if parallel:
                with mp.Pool(processes=n) as p:
                    lams=p.map(partial(worker, instance=z),[q]*n)
            else:
                lams = np.zeros(n)
                for i in range(n):
                    z.random_rewiring_undirected(q)
                    lams[i] = z.second_largest_eigenvalue_normalized(8, 1.2)
            #print('value of lams',lams)
            dictionary[str(k)][str(q)] = lams
        print('one k done.', k)
    # save dictionary in json
    print('done', dictionary)
    with open('./reproduce/' + name + '.txt', 'w') as outfile:
        json.dump(dictionary, outfile)


if __name__ == '__main__':
    exponent = np.arange(16)
    q_values = 10 ** (-exponent / 3)

    #q_values = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
    k_values = [20, 50, 100, 200, 400, 800]
    main(q_values, k_values, 'reproduce_1000_undirected_with_averaging', 10)
