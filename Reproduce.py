from matrix import build_matrix
import json
#from matrix import integer_inequality
import numpy as np
try:
    from mpi4py import MPI
except:
    pass
try:
    from mpi4py.futures import MPIPoolExecutor as PoolExecutor
except:
    from concurrent.futures import ProcessPoolExecutor as PoolExecutor
    print('Imported concurrent module instead of MPI.')
#from scipy.sparse import coo_matrix, csr_matrix, lil_matrix
#from matplotlib import pyplot as plt
#import scipy.sparse
#from scipy.sparse.linalg import eigsh
#from scipy.sparse.linalg import eigs
#from scipy.sparse import identity
#import multiprocessing as mp
#mp.set_start_method('spawn')
#mp.set_start_method('fork')
from functools import partial
#from numba import njit
from scipy.sparse import *
import time
import tqdm


def worker(q, L_0=None, k=None, N_tot=None, directed=False):
    if directed:
        rows, columns, values = find(L_0)
        new_rows = build_matrix.numba_fast_directed_rewiring(rows, columns, N_tot, k, q)
        L_rnd = csr_matrix((values, (new_rows, columns)), shape=(N_tot, N_tot))
    else:
        L_rnd=build_matrix.fast_rewiring_undirected(L_0, k, q, N_tot, save_mem=False)
    lam = build_matrix.fast_second_largest(L_rnd, N_tot, directed=directed)
    return lam

def main(q_values, r_0_values, filename, name, dimensions, directed=False):
    time1 = time.time()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    # set seed
    np.random.seed(rank)
    dictionary = {str(r_0):{str(q)[:6]:float() for q in q_values} for r_0 in r_0_values}
    # go through r_0 and q values
    for r_0 in r_0_values:
        time3=time.time()
        z = build_matrix(filename, dimensions, r_0)
        z.tuples=build_matrix.fast_all_indices(np.array(z.D_0), z.N)
        z.one_int_index_tuples_and_adjacency()
        z.Laplacian_0()
        for q in q_values:
            lam=worker(q, L_0 = z.L_0, k = z.k, N_tot = z.N_tot, directed = directed)
            dictionary[str(r_0)][str(q)] = lam
        time4 = time.time()
        print('{} seconds for previous r_0 {} in process {}: '.format(time4-time3, r_0, rank))
    # gather processes
    data = comm.gather(dictionary, root=0)
    if rank==0:
        print('gathered {} processes.'.format(len(data)))
        longdict={str(r_0):{str(q)[:6]:[] for q in q_values} for r_0 in r_0_values}
        shortdict={str(r_0):{str(q)[:6]:[] for q in q_values} for r_0 in r_0_values}
        for r_0 in r_0_values:
            for q in q_values:
                lams=[]
                for i in range(len(data)):
                    lams.append(data[i][str(r_0)][str(q)])
                longdict[str(r_0)][str(q)] = lams
                shortdict[str(r_0)][str(q)] = [np.mean(np.array(lams)), np.std(np.array(lams))]
        time2 = time.time()
        # save dictionary in json
        print('Done. Took {} s.'.format(time2-time1))
        with open(name + 'long' + '.json', 'w') as outfile:
            json.dump(longdict, outfile)
        with open(name + 'short' + '.json', 'w') as outfile:
            json.dump(shortdict, outfile)


if __name__ == '__main__':
    exponent = np.arange(16)
    #q_values = 10 ** (-exponent / 3)
    q_values = [1, 0.1, 0.01]#, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
    #q_values = [0.1]#[1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
    k_values = [10, 25, 50]#, 50, 100, 200, 400, 800]
    start = time.time()
    main(q_values, k_values, '1d_ring_1000','reproduce/test_2d', 5, np.array([1000, 1, 1]), parallel=True, directed=False)
    stop = time.time()
    print(stop-start)
