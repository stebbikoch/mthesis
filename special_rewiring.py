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
from matplotlib import pyplot as plt

def main(m_values, r_0_values, name, dimensions, filename=None, sphere=False, gaussian=False, mu=None, sigma=None):
    time1 = time.time()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    # set seed
    np.random.seed(rank)
    dictionary = {str(r_0):{'k':float(),'ms':{str(m):{'smallest':float(), 'second_largest':float()} for m in m_values}} for r_0 in r_0_values}
    # go through r_0 and q values
    for r_0 in r_0_values:
        time3=time.time()
        if sphere:
            z = fibonacci_sphere(np.prod(dimensions))
            z.wiring(r_0)
        else:
            z = build_matrix(dimensions, r_0, d_function=filename)
            z.tuples=build_matrix.fast_all_indices(np.array(z.D_0), z.N)
            z.one_int_index_tuples_and_adjacency()
            z.Laplacian_0()
        dictionary[str(r_0)]['k'] = z.k
        for m in m_values:
            #print(z.L_0.toarray())
            #print('regular')
            #plt.imshow(z.L_0.toarray(), vmin=0)
            #plt.show()
            if not gaussian:
                #print(m)
                z.special_rewiring_1d(m)
            else:
                z.gaussian_rewiring_1d(m, mu=mu, sigma=sigma)
            #print(z.L_rnd.diagonal())
            #print(z.L_rnd.toarray())
            #print('random')
            #plt.imshow(z.L_rnd.toarray(), vmin=0)
            #plt.show()
            lam = build_matrix.arnoldi_eigenvalues(z.L_rnd, z.N_tot, directed=False, smallest=False)
            lam2 = build_matrix.arnoldi_eigenvalues(z.L_rnd, z.N_tot, directed=False, smallest=True)
            dictionary[str(r_0)]['ms'][str(m)]['second_largest'] = lam
            dictionary[str(r_0)]['ms'][str(m)]['smallest'] = lam2
        time4 = time.time()
        print('{} seconds for previous r_0 {} in process {}: '.format(time4-time3, r_0, rank))
    # gather processes
    data = comm.gather(dictionary, root=0)
    if rank==0:
        print('gathered {} processes.'.format(len(data)))
        longdict={str(r_0):{'k':float(),'ms':{str(m):{'smallest':[],'second_largest':[]} for m in m_values}} for r_0 in r_0_values}
        shortdict={str(r_0):{'k':float(),'ms':{str(m):{'smallest':[],'second_largest':[]} for m in m_values}} for r_0 in r_0_values}
        for r_0 in r_0_values:
            longdict[str(r_0)]['k'] = dictionary[str(r_0)]['k']
            shortdict[str(r_0)]['k'] = dictionary[str(r_0)]['k']
            for m in m_values:
                lams=[]
                lams2=[]
                for i in range(len(data)):
                    lams.append(data[i][str(r_0)]['ms'][str(m)]['second_largest'])
                    lams2.append(data[i][str(r_0)]['ms'][str(m)]['smallest'])
                longdict[str(r_0)]['ms'][str(m)]['second_largest'] = lams
                longdict[str(r_0)]['ms'][str(m)]['smallest'] = lams2
                shortdict[str(r_0)]['ms'][str(m)]['second_largest'] = [np.mean(np.array(lams)), np.std(np.array(lams))]
                shortdict[str(r_0)]['ms'][str(m)]['smallest'] = [np.mean(np.array(lams2)), np.std(np.array(lams2))]
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
    n=int(10)
    ara = range(1,n)
    exponent = np.arange(1,16)
    q_values = 10 ** (-exponent / 3)
    exponent = np.arange(15).astype(float)
    m_values = (10**(-exponent/5)*1000).astype(int)
    m_values = np.append(m_values, 0)
    #print(m_values)
    r_0_values = [10, 25, 50, 100]
    main(m_values, r_0_values, 'reproduce/ring_delta_peak', np.array([1000, 1, 1]), filename='1d_1000',
         sphere=False, gaussian=False)