from matrix import build_matrix
import json
#from matrix import integer_inequality
import numpy as np
from mpi4py import MPI
from scipy.sparse import *
import time
import os
import tqdm
from sphere import fibonacci_sphere

def worker(q, L_0=None, k=None, N_tot=None, directed=False):
    if directed:
        rows, columns, values = find(L_0)
        new_rows = build_matrix.numba_fast_directed_rewiring(rows, columns, N_tot, k, q)
        L_rnd = csr_matrix((values, (new_rows, columns)), shape=(N_tot, N_tot))
    else:
        L_rnd=build_matrix.fast_rewiring_undirected(L_0, k, q, N_tot, save_mem=False)
    lam = build_matrix.fast_second_largest(L_rnd, N_tot, directed=directed, smallest=False)
    lam2 = build_matrix.fast_second_largest(L_rnd, N_tot, directed=directed, smallest=True)
    return lam, lam2

def main(q_values, r_0_values, name, dimensions, filename=None, sphere=False, directed=False):
    time1 = time.time()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    # set seed
    np.random.seed(rank)
    dictionary = {str(r_0):{'k':float(),'qs':{str(q):{'smallest':float(), 'second_largest':float()} for q in q_values}} for r_0 in r_0_values}
    # go through r_0 and q values
    for r_0 in r_0_values:
        time3=time.time()
        if sphere:
            z = fibonacci_sphere(np.prod(dimensions))
            z.wiring(r_0)
        else:
            z = build_matrix(filename, dimensions, r_0)
            z.tuples=build_matrix.fast_all_indices(np.array(z.D_0), z.N)
            z.one_int_index_tuples_and_adjacency()
            z.Laplacian_0()
        dictionary[str(r_0)]['k'] = z.k
        for q in q_values:
            lam, lam2 = worker(q, L_0 = z.L_0, k = z.k, N_tot = z.N_tot, directed = directed)
            dictionary[str(r_0)]['qs'][str(q)]['second_largest'] = lam
            dictionary[str(r_0)]['qs'][str(q)]['smallest'] = lam2
        time4 = time.time()
        print('{} seconds for previous r_0 {} in process {}: '.format(time4-time3, r_0, rank))
    # gather processes
    data = comm.gather(dictionary, root=0)
    if rank==0:
        print('gathered {} processes.'.format(len(data)))
        longdict={str(r_0):{'k':float(),'qs':{str(q):{'smallest':[],'second_largest':[]} for q in q_values}} for r_0 in r_0_values}
        shortdict={str(r_0):{'k':float(),'qs':{str(q):{'smallest':[],'second_largest':[]} for q in q_values}} for r_0 in r_0_values}
        for r_0 in r_0_values:
            longdict[str(r_0)]['k'] = dictionary[str(r_0)]['k']
            shortdict[str(r_0)]['k'] = dictionary[str(r_0)]['k']
            for q in q_values:
                lams=[]
                lams2=[]
                for i in range(len(data)):
                    lams.append(data[i][str(r_0)]['qs'][str(q)]['second_largest'])
                    lams2.append(data[i][str(r_0)]['qs'][str(q)]['smallest'])
                longdict[str(r_0)]['qs'][str(q)]['second_largest'] = lams
                longdict[str(r_0)]['qs'][str(q)]['smallest'] = lams2
                shortdict[str(r_0)]['qs'][str(q)]['second_largest'] = [np.mean(np.array(lams)), np.std(np.array(lams))]
                shortdict[str(r_0)]['qs'][str(q)]['smallest'] = [np.mean(np.array(lams2)), np.std(np.array(lams2))]
        time2 = time.time()
        # save dictionary in json
        print('Done. Took {} s.'.format(time2-time1))
        # remove result file, if exists
        filepathlong = name + 'long' + '.json'
        filepathshort = name + 'short' + '.json'
        if os.path.exists(filepathlong):
            os.remove(filepathlong)
        if os.path.exists(filepathshort):
            os.remove(filepathshort)
        with open(filepathlong, 'w') as outfile:
            json.dump(longdict, outfile)
        with open(filepathshort, 'w') as outfile:
            json.dump(shortdict, outfile)


if __name__ == '__main__':
    exponent = np.arange(16)
    q_values = 10 ** (-exponent / 3)
    #q_values = [0.7]
    #q_values = [1, 0.1, 0.01]#, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
    q_values = [0.9]#[1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
    r_0_values = [25]#[0.2, 1, 1.3]#[10, 25, 50]#, 50, 100, 200, 400, 800]
    main(q_values, r_0_values, 'reproduce/sphere_testing', np.array([100, 100, 1]),filename='2d_100_100_1',
         sphere=False, directed=False)
