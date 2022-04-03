from matrix import build_matrix, integer_inequality
import json
#from matrix import integer_inequality
import numpy as np
from mpi4py import MPI
from scipy.sparse import *
import time
import os
import tqdm
from sphere import fibonacci_sphere
import networkx as nx

def worker(q, L_0=None, k=None, N_tot=None, directed=False):
    if directed:
        rows, columns, values = find(L_0)
        new_rows = build_matrix.numba_fast_directed_rewiring(rows, columns, N_tot, k, q)
        L_rnd = csr_matrix((values, (new_rows, columns)), shape=(N_tot, N_tot))
    else:
        L_rnd=build_matrix.fast_rewiring_undirected(L_0, k, q, N_tot, save_mem=False)
    lam = build_matrix.arnoldi_eigenvalues(L_rnd, N_tot, directed=directed, smallest=False)
    lam2 = build_matrix.arnoldi_eigenvalues(L_rnd, N_tot, directed=directed, smallest=True)
    return lam, lam2

def main(q_values, r_0_values, name, dimensions, filename=None, sphere=False, randomsphere=False, eqsphere=False,
         directed=False, nxwatts_strogatz=False, pathclustering=False):
    time1 = time.time()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    # set seed
    np.random.seed(rank+100)
    dictionary = {str(r_0):{'k':float(),
                            'qs':{str(q):{'clustering':float(),
                                          'characterstic_path':float(),
                                          'smallest':float(),
                                          'second_largest':float(),
                                          'smallest_adjacency': float(),
                                          'second_largest_adjacency': float(),
                                          'k_max':float(),
                                          'k_min':float()} for q in q_values}} for r_0 in r_0_values}
    # go through r_0 and q values
    for r_0 in r_0_values:
        time3=time.time()
        if sphere:
            z = fibonacci_sphere(np.prod(dimensions), random=randomsphere, eq_partition=eqsphere)
            z.wiring(r_0)
        else:
            z = build_matrix(dimensions, r_0, d_function=filename)
            z.tuples=build_matrix.fast_all_indices(np.array(z.D_0), z.N)
            z.one_int_index_tuples_and_adjacency()
            z.Laplacian_0()
        dictionary[str(r_0)]['k'] = z.k
        for q in q_values:
            # use original watts strogatz algorithm (implemented through networkx, only for undirected case)
            if nxwatts_strogatz:
                G = nx.generators.random_graphs.watts_strogatz_graph(z.N_tot, z.k, q)
                L_rnd = -nx.linalg.laplacianmatrix.laplacian_matrix(G)
            elif directed:
                rows, columns, values = find(z.L_0)
                new_rows = build_matrix.numba_fast_directed_rewiring(rows, columns, z.N_tot, z.k, q)
                L_rnd = csr_matrix((values, (new_rows, columns)), shape=(z.N_tot, z.N_tot))
            else:
                L_rnd = build_matrix.fast_rewiring_undirected(z.L_0, z.k, q, z.N_tot, save_mem=False)
            # not normalized (only scaled) eigenvalues
            lam = build_matrix.arnoldi_eigenvalues(L_rnd, z.N_tot, directed=directed, smallest=False,
                                                   normalized=False)
            lam2 = build_matrix.arnoldi_eigenvalues(L_rnd, z.N_tot, directed=directed, smallest=True,
                                                    normalized=False)
            # adjacency eigenvalues
            lam_adj = build_matrix.arnoldi_eigenvalues(L_rnd, z.N_tot, directed=directed, smallest=False,
                                                   normalized=False, adjacency=True)
            lam2_adj = build_matrix.arnoldi_eigenvalues(L_rnd, z.N_tot, directed=directed, smallest=True,
                                                    normalized=False, adjacency=True)
            # max and min degrees
            min_degree = int(np.min(-L_rnd.diagonal()))
            max_degree = int(np.max(-L_rnd.diagonal()))
            # characteristic path length and clustering coefficient
            if pathclustering:
                G = nx.Graph(L_rnd)
                c = nx.average_clustering(G)
                l = nx.average_shortest_path_length(G)
            # write results in dictionary
                dictionary[str(r_0)]['qs'][str(q)]['clustering'] = c
                dictionary[str(r_0)]['qs'][str(q)]['characteristic_path'] = l
            dictionary[str(r_0)]['qs'][str(q)]['second_largest'] = lam
            dictionary[str(r_0)]['qs'][str(q)]['smallest'] = lam2
            dictionary[str(r_0)]['qs'][str(q)]['second_largest_adjacency'] = lam_adj
            dictionary[str(r_0)]['qs'][str(q)]['smallest_adjacency'] = lam2_adj
            dictionary[str(r_0)]['qs'][str(q)]['k_min'] = min_degree
            dictionary[str(r_0)]['qs'][str(q)]['k_max'] = max_degree
            print('q={} done'.format(q))
        time4 = time.time()
        print('{} seconds for previous r_0 {} in process {}: '.format(time4-time3, r_0, rank))
    # gather processes
    data = comm.gather(dictionary, root=0)
    if rank==0:
        print('gathered {} processes.'.format(len(data)))
        longdict={str(r_0):{'k':float(),
                            'qs':{str(q):{'clustering':[],
                                          'characteristic_path':[],
                                          'smallest':[],
                                          'second_largest':[],
                                          'smallest_adj': [],
                                          'second_largest_adj': [],
                                          'k_max':[],
                                          'k_min':[]} for q in q_values}} for r_0 in r_0_values}
        shortdict={str(r_0):{'k':float(),
                             'qs':{str(q):{'clustering':[],
                                           'characteristic_path':[],
                                           'smallest':[],
                                           'second_largest':[],
                                           'smallest_adj': [],
                                           'second_largest_adj': [],
                                           'k_max': float(),
                                           'k_min':float()} for q in q_values}} for r_0 in r_0_values}
        for r_0 in r_0_values:
            longdict[str(r_0)]['k'] = dictionary[str(r_0)]['k']
            shortdict[str(r_0)]['k'] = dictionary[str(r_0)]['k']
            for q in q_values:
                c_list = []
                l_list = []
                lams=[]
                lams2=[]
                lams_norm=[]
                lams2_norm=[]
                lams_adj = []
                lams2_adj = []
                k_mins=[]
                k_maxs=[]
                for i in range(len(data)):
                    if pathclustering:
                        c_list.append(data[i][str(r_0)]['qs'][str(q)]['clustering'])
                        l_list.append(data[i][str(r_0)]['qs'][str(q)]['characteristic_path'])
                    lams.append(data[i][str(r_0)]['qs'][str(q)]['second_largest'])
                    lams2.append(data[i][str(r_0)]['qs'][str(q)]['smallest'])
                    lams_adj.append(data[i][str(r_0)]['qs'][str(q)]['second_largest_adjacency'])
                    lams2_adj.append(data[i][str(r_0)]['qs'][str(q)]['smallest_adjacency'])
                    k_mins.append(data[i][str(r_0)]['qs'][str(q)]['k_min'])
                    k_maxs.append(data[i][str(r_0)]['qs'][str(q)]['k_max'])
                if pathclustering:
                    longdict[str(r_0)]['qs'][str(q)]['clustering'] = c_list
                    longdict[str(r_0)]['qs'][str(q)]['characteristic_path'] = l_list
                longdict[str(r_0)]['qs'][str(q)]['second_largest'] = lams
                longdict[str(r_0)]['qs'][str(q)]['smallest'] = lams2
                longdict[str(r_0)]['qs'][str(q)]['second_largest_adj'] = lams_adj
                longdict[str(r_0)]['qs'][str(q)]['smallest_adj'] = lams2_adj
                longdict[str(r_0)]['qs'][str(q)]['k_min'] = k_mins
                longdict[str(r_0)]['qs'][str(q)]['k_max'] = k_maxs
                if pathclustering:
                    shortdict[str(r_0)]['qs'][str(q)]['clustering'] = [np.mean(np.array(c_list)), np.std(np.array(c_list))]
                    shortdict[str(r_0)]['qs'][str(q)]['characteristic_path'] = [np.mean(np.array(l_list)),
                                                                            np.std(np.array(l_list))]
                shortdict[str(r_0)]['qs'][str(q)]['second_largest'] = [np.mean(np.array(lams)), np.std(np.array(lams))]
                shortdict[str(r_0)]['qs'][str(q)]['smallest'] = [np.mean(np.array(lams2)), np.std(np.array(lams2))]
                shortdict[str(r_0)]['qs'][str(q)]['second_largest_adj'] = [np.mean(np.array(lams_adj)),
                                                                            np.std(np.array(lams_adj))]
                shortdict[str(r_0)]['qs'][str(q)]['smallest_adj'] = [np.mean(np.array(lams2_adj)),
                                                                      np.std(np.array(lams2_adj))]
                shortdict[str(r_0)]['qs'][str(q)]['k_min'] = np.mean(np.array(k_mins))
                shortdict[str(r_0)]['qs'][str(q)]['k_max'] = np.mean(np.array(k_maxs))
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

def main2(q_values, r_0_values, N_values, name, dimensions, sphere=False, randomsphere=False, eqsphere=False,
         directed=False, nxwatts_strogatz=False, pathclustering=False, eucl=False):
    time1 = time.time()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    # set seed
    np.random.seed(rank+100)
    dictionary = {str(N):{'k':float(),
                            'qs':{str(q):{'clustering':float(),
                                          'characterstic_path':float(),
                                          'smallest':float(),
                                          'second_largest':float(),
                                          'smallest_adjacency': float(),
                                          'second_largest_adjacency': float(),
                                          'k_max':float(),
                                          'k_min':float()} for q in q_values}} for N in N_values}
    # go through r_0 and q values
    for N, r_0 in zip(N_values, r_0_values):
        time3=time.time()
        if sphere:
            dimensions = np.array([N, 1, 1])
            z = fibonacci_sphere(np.prod(dimensions), random=randomsphere, eq_partition=eqsphere)
            z.wiring(r_0)
        else:
            if dimensions[2] == 1 and dimensions[1]==1:
                dimensions = np.array([N, 1, 1])
            elif dimensions[2] == 1:
                dimensions = np.array([int(np.sqrt(N)), int(np.sqrt(N)), 1])
            else:
                dimensions = np.array([int(np.cbrt(N)), int(np.cbrt(N)), int(np.cbrt(N))])
            print(dimensions)
            x = integer_inequality(dimensions)
            x.all_numbers(400, d_given=[r_0], eucl=eucl)
            z = build_matrix(dimensions, r_0, data=x.data)
            z.tuples=build_matrix.fast_all_indices(np.array(z.D_0), z.N)
            z.one_int_index_tuples_and_adjacency()
            z.Laplacian_0()
        dictionary[str(N)]['k'] = z.k
        for q in q_values:
            # use original watts strogatz algorithm (implemented through networkx, only for undirected case)
            if nxwatts_strogatz:
                G = nx.generators.random_graphs.watts_strogatz_graph(z.N_tot, z.k, q)
                L_rnd = -nx.linalg.laplacianmatrix.laplacian_matrix(G)
            elif directed:
                rows, columns, values = find(z.L_0)
                new_rows = build_matrix.numba_fast_directed_rewiring(rows, columns, z.N_tot, z.k, q)
                L_rnd = csr_matrix((values, (new_rows, columns)), shape=(z.N_tot, z.N_tot))
            else:
                L_rnd = build_matrix.fast_rewiring_undirected(z.L_0, z.k, q, z.N_tot, save_mem=False)
            # not normalized (only scaled) eigenvalues
            lam = build_matrix.arnoldi_eigenvalues(L_rnd, z.N_tot, directed=directed, smallest=False,
                                                   normalized=False)
            lam2 = build_matrix.arnoldi_eigenvalues(L_rnd, z.N_tot, directed=directed, smallest=True,
                                                    normalized=False)
            # adjacency eigenvalues
            lam_adj = build_matrix.arnoldi_eigenvalues(L_rnd, z.N_tot, directed=directed, smallest=False,
                                                   normalized=False, adjacency=True)
            lam2_adj = build_matrix.arnoldi_eigenvalues(L_rnd, z.N_tot, directed=directed, smallest=True,
                                                    normalized=False, adjacency=True)
            # max and min degrees
            min_degree = int(np.min(-L_rnd.diagonal()))
            max_degree = int(np.max(-L_rnd.diagonal()))
            # characteristic path length and clustering coefficient
            if pathclustering:
                G = nx.Graph(L_rnd)
                c = nx.average_clustering(G)
                l = nx.average_shortest_path_length(G)
            # write results in dictionary
                dictionary[str(N)]['qs'][str(q)]['clustering'] = c
                dictionary[str(N)]['qs'][str(q)]['characteristic_path'] = l
            dictionary[str(N)]['qs'][str(q)]['second_largest'] = lam
            dictionary[str(N)]['qs'][str(q)]['smallest'] = lam2
            dictionary[str(N)]['qs'][str(q)]['second_largest_adjacency'] = lam_adj
            dictionary[str(N)]['qs'][str(q)]['smallest_adjacency'] = lam2_adj
            dictionary[str(N)]['qs'][str(q)]['k_min'] = min_degree
            dictionary[str(N)]['qs'][str(q)]['k_max'] = max_degree
            print('q={} done'.format(q))
        time4 = time.time()
        print('{} seconds for previous N {} in process {}: '.format(time4-time3, N, rank))
    # gather processes
    data = comm.gather(dictionary, root=0)
    if rank==0:
        print('gathered {} processes.'.format(len(data)))
        longdict={str(N):{'k':float(),
                            'qs':{str(q):{'clustering':[],
                                          'characteristic_path':[],
                                          'smallest':[],
                                          'second_largest':[],
                                          'smallest_adj': [],
                                          'second_largest_adj': [],
                                          'k_max':[],
                                          'k_min':[]} for q in q_values}} for N in N_values}
        shortdict={str(N):{'k':float(),
                             'qs':{str(q):{'clustering':[],
                                           'characteristic_path':[],
                                           'smallest':[],
                                           'second_largest':[],
                                           'smallest_adj': [],
                                           'second_largest_adj': [],
                                           'k_max': float(),
                                           'k_min':float()} for q in q_values}} for N in N_values}
        for N in N_values:
            longdict[str(N)]['k'] = dictionary[str(N)]['k']
            shortdict[str(N)]['k'] = dictionary[str(N)]['k']
            for q in q_values:
                c_list = []
                l_list = []
                lams=[]
                lams2=[]
                lams_adj = []
                lams2_adj = []
                k_mins=[]
                k_maxs=[]
                for i in range(len(data)):
                    if pathclustering:
                        c_list.append(data[i][str(N)]['qs'][str(q)]['clustering'])
                        l_list.append(data[i][str(N)]['qs'][str(q)]['characteristic_path'])
                    lams.append(data[i][str(N)]['qs'][str(q)]['second_largest'])
                    lams2.append(data[i][str(N)]['qs'][str(q)]['smallest'])
                    lams_adj.append(data[i][str(N)]['qs'][str(q)]['second_largest_adjacency'])
                    lams2_adj.append(data[i][str(N)]['qs'][str(q)]['smallest_adjacency'])
                    k_mins.append(data[i][str(N)]['qs'][str(q)]['k_min'])
                    k_maxs.append(data[i][str(N)]['qs'][str(q)]['k_max'])
                if pathclustering:
                    longdict[str(N)]['qs'][str(q)]['clustering'] = c_list
                    longdict[str(N)]['qs'][str(q)]['characteristic_path'] = l_list
                longdict[str(N)]['qs'][str(q)]['second_largest'] = lams
                longdict[str(N)]['qs'][str(q)]['smallest'] = lams2
                longdict[str(N)]['qs'][str(q)]['second_largest_adj'] = lams_adj
                longdict[str(N)]['qs'][str(q)]['smallest_adj'] = lams2_adj
                longdict[str(N)]['qs'][str(q)]['k_min'] = k_mins
                longdict[str(N)]['qs'][str(q)]['k_max'] = k_maxs
                if pathclustering:
                    shortdict[str(N)]['qs'][str(q)]['clustering'] = [np.mean(np.array(c_list)), np.std(np.array(c_list))]
                    shortdict[str(N)]['qs'][str(q)]['characteristic_path'] = [np.mean(np.array(l_list)),
                                                                            np.std(np.array(l_list))]
                shortdict[str(N)]['qs'][str(q)]['second_largest'] = [np.mean(np.array(lams)), np.std(np.array(lams))]
                shortdict[str(N)]['qs'][str(q)]['smallest'] = [np.mean(np.array(lams2)), np.std(np.array(lams2))]
                shortdict[str(N)]['qs'][str(q)]['second_largest_adj'] = [np.mean(np.array(lams_adj)),
                                                                            np.std(np.array(lams_adj))]
                shortdict[str(N)]['qs'][str(q)]['smallest_adj'] = [np.mean(np.array(lams2_adj)),
                                                                      np.std(np.array(lams2_adj))]
                shortdict[str(N)]['qs'][str(q)]['k_min'] = np.mean(np.array(k_mins))
                shortdict[str(N)]['qs'][str(q)]['k_max'] = np.mean(np.array(k_maxs))
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
    q_values = [0.01, 0.1, 0.3]
    #q_values = [1, 0.1, 0.01]#, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
    #q_values = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
    r_0_values = [1, 2, 3]#[5, 10, 25, 50, 100, 200, 400]
    N_values = [81, 484, 784]
    main2(q_values, r_0_values, N_values, 'results/2d_max_N_scaling', np.array([16, 16, 1]), directed=True)
