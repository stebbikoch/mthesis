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


def worker(q, L_0=None, k=None, N_tot=None, directed=False, function=None):
    try:
        comm = MPI.COMM_WORLD
        print("%d of %d" % (comm.Get_rank(), comm.Get_size()))
    except:
        pass
    start = time.time()
    # new seed
    np.random.seed()
    if directed:
        rows, columns, values = find(L_0)
        new_rows = function(rows, columns, N_tot, k, q)
        #new_rows = build_matrix.numba_fast_directed_rewiring(rows, columns, N_tot, k, q)
        L_rnd = csr_matrix((values, (new_rows, columns)), shape=(N_tot, N_tot))
    else:
        L_rnd=build_matrix.fast_rewiring_undirected(L_0, k, q, N_tot, save_mem=True)
    #instance.random_rewiring_undirected(q)
    lam = build_matrix.fast_second_largest(L_rnd, N_tot, directed=directed)
    end = time.time()
    #print('execution time worker:', end-start)
    return lam

def main(q_values, r_0_values, filename, name, n, dimensions, parallel=False, directed=False, processes=1):
    dictionary = {str(r_0):{str(q):[] for q in q_values} for r_0 in r_0_values}
    for r_0 in r_0_values:
        print('r_0: ', r_0)
        time3=time.time()
        z = build_matrix(filename, dimensions, r_0)
        #print(z.all_indices_list)
        z.tuples=build_matrix.fast_all_indices(np.array(z.D_0), z.N)
        z.one_int_index_tuples_and_adjacency()
        z.Laplacian_0()
        if directed:
            # call function to precompile
            rows, columns, values = find(z.L_0)
            build_matrix.numba_fast_directed_rewiring(rows, columns, z.N_tot, z.k, 0.001)
        else:
            build_matrix.fast_rewiring_undirected(z.L_0, z.k, 0.001, z.N_tot, save_mem=True)
        for q in q_values:
            print('q: ', q)
            time1 = time.time()
            # do the same thing n times
            if parallel:
                with PoolExecutor(max_workers=processes) as p:
                	results = p.map(partial(worker, L_0=z.L_0, k=z.k, N_tot=z.N_tot, directed=directed,
                                                 function=build_matrix.numba_fast_directed_rewiring), [q] * n)
                	print('do shutdown and join')
                	p.shutdown(wait=False)
                #results = []
                #for result in tqdm.tqdm(
                #        p.map(partial(worker, L_0=z.L_0, k=z.k, N_tot=z.N_tot, directed=directed,
                #                                 function=build_matrix.numba_fast_directed_rewiring), [q] * n),
                #        total=n):
                #    results.append(result)
                #p.shutdown()
                # p = mp.Pool()
                # results = []
                # for result in tqdm.tqdm(p.imap_unordered(partial(worker, L_0=z.L_0, k=z.k, N_tot=z.N_tot, directed=directed,
                #                        function=build_matrix.numba_fast_directed_rewiring), [q]*n), total=n):
                #     results.append(result)
                #with mp.Pool(processes=16) as p:
                 #   lams=p.map(partial(worker, L_0=z.L_0, k=z.k, N_tot=z.N_tot, directed=directed,
                  #                     function=build_matrix.numba_fast_directed_rewiring), [q]*n)
                    #p.close() # no more tasks
                    #p.join() # wrap up current tasks
                lams = np.array(list(results))
                #print(lams)
                lams = [np.mean(lams), np.std(lams)]
            else:
                lams = np.zeros(n)
                for i in range(n):
                    lams[i]=worker(q, z.L_0, k=k, N_tot=z.N_tot, directed=True)
                lams=np.mean(lams).tolist()
            time2=time.time()
            print('time for previous q: ', time2-time1)
            #print('value of lams',lams)
            dictionary[str(r_0)][str(q)] = lams
        time4 = time.time()
        print('time for previous r_0: ', time4-time3)
    # save dictionary in json
    print('done', dictionary)
    with open(name + '.json', 'w') as outfile:
        json.dump(dictionary, outfile)


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
