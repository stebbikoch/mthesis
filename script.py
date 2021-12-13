#!/bin/python3
import numpy as np
from Reproduce import main
import time
try:
    from mpi4py import MPI
except:
    pass
#import sys
#print(sys.executable)
# build d_function thing
#x = integer_inequality(np.array([1000, 1, 1]))
#x.all_numbers(400)
#x.save_to_json('1d_ring_1000')
# do the rest
if __name__=='__main__':
    try:
        comm = MPI.COMM_WORLD
        print("%d of %d" % (comm.Get_rank(), comm.Get_size()))
    except:
        pass
    start = time.time()
    exponent = np.arange(16)
    #q_values = 10 ** (-exponent / 3)
    q_values = [0.01, 0.1]#, 1]
    r_0 = [50]#[400, 200, 100, 50, 25, 10]
    main(q_values, r_0, '1d_ring_1000','results/ring_directed_3x',
         3,np.array([1000, 1, 1]),
         parallel=True, directed=True, processes=3)
    end = time.time()
    print('zeit ', end-start)
    print('okay')
