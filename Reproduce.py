from matrix import build_matrix
import json
from matrix import integer_inequality
import numpy as np
from scipy.sparse import coo_matrix
from matplotlib import pyplot as plt
import scipy.sparse
from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg import eigs
from scipy.sparse import identity
import multiprocessing as mp
from functools import partial
from numba import njit

def slow_directed(L_0, k, q, N_tot):
    L_rnd = k/2*identity(N_tot)
    for i in range(N_tot):
        L_rnd[i] = build_matrix.fast_rewiring_directed_ith_row(L_0.getrow(i), N_tot, k, q)
    return L_rnd
        #with mp.Pool(processes=N_tot) as p:
         #   lams = p.map(partial(worker, L_0=z.L_0, k=k, N_tot=z.N_tot), [q] * n)


def worker(q, L_0=None, k=None, N_tot=None, directed=False):
    if directed:
        L_rnd=slow_directed(L_0, k, q, N_tot)
    else:
        L_rnd=build_matrix.fast_rewiring(L_0, k, q, N_tot)
    #instance.random_rewiring_undirected(q)
    lam = build_matrix.fast_second_largest(L_rnd, N_tot, directed=directed)
    return lam

@njit()
def numba_func(q, L_0, k, N_tot, n):
    lam = np.zeros(n)
    for i in range(n):
        L_rnd = build_matrix.fast_rewiring(L_0, k, q, N_tot)
        # instance.random_rewiring_undirected(q)
        lam[i] = build_matrix.fast_second_largest(L_rnd, N_tot)
    return np.mean(lam).tolist()

def main(q_values, k_values, name, n, parallel=False, numba=False, directed=False):
    dictionary = {str(k):{str(q):[] for q in q_values} for k in k_values}
    for k in k_values:
        z = build_matrix('1d_ring_1000', np.array([1000, 1, 1]), (k/2))
        #print(z.all_indices_list)
        z.all_indices()
        z.one_int_index_tuples_and_adjacency()
        z.Laplacian_0()
        for q in q_values:
            print('q', q)
            # do the same thing n times
            if parallel:
                with mp.Pool(processes=n) as p:
                    lams=p.map(partial(worker, L_0=z.L_0, k=k, N_tot=z.N_tot),[q]*n)
                lams = np.mean(np.array(lams)).tolist()
            elif numba:
                lams = numba_func(q, z.L_0, k, z.N_tot, n)
            else:
                lams = np.zeros(n)
                for i in range(n):
                    z.random_rewiring_undirected(q)
                    lams[i] = z.second_largest_eigenvalue_normalized(8, 1.2, directed=directed)
                lams=np.mean(lams).tolist()
            #print('value of lams',lams)
            dictionary[str(k)][str(q)] = lams
        print('one k done.', k)
    # save dictionary in json
    print('done', dictionary)
    with open('./reproduce/' + name + '.json', 'w') as outfile:
        json.dump(dictionary, outfile)


if __name__ == '__main__':
    exponent = np.arange(16)
    #q_values = 10 ** (-exponent / 3)

    q_values = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
    k_values = [20, 50, 100, 200, 400, 800]
    main(q_values, k_values, 'reproduce_1000_directed_with_averaging', 5, parallel=True, directed=True)
