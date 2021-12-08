#!/bin/sh
import numpy as np
from Reproduce import main
import time
from mpi4py import MPI
#import sys
#print(sys.executable)
# build d_function thing
#x = integer_inequality(np.array([1000, 1, 1]))
#x.all_numbers(400)
#x.save_to_json('1d_ring_1000')
# do the rest
if __name__=='__main__':
    comm = MPI.COMM_WORLD
    print("%d of %d" % (comm.Get_rank(), comm.Get_size()))
    start = time.time()
    exponent = np.arange(16)
    q_values = 10 ** (-exponent / 3)
    q_values = [0.001, 0.1, 1]
    r_0 = [10]#[400, 200, 100, 50, 25, 10]
    main(q_values, r_0, '1d_ring_1000','results/ring_directed_100x',
         10,np.array([1000, 1, 1]),
         parallel=True, directed=True, processes=10)
    end = time.time()
    print('zeit ', end-start)
    print('okay')
