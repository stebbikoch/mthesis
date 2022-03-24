from matrix import build_matrix
from matrix import integer_inequality
import numpy as np
from scipy.sparse import coo_matrix
from matplotlib import pyplot as plt
import scipy.sparse
from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg import eigs
from Reproduce import main
import time
from multiprocessing import set_start_method
# build d_function thing
#x = integer_inequality(np.array([100, 100, 1]))
#x.all_numbers(49,d_given = [2,5,8,10,12,15,20,25, 30, 35, 40, 49])
#x.save_to_json('2d_100_100_1')
# do the rest
if __name__=='__main__':
     set_start_method('spawn')
     start = time.time()
     exponent = np.arange(16)
     q_values = 10 ** (-exponent / 3)
     q_values = [1]
     r_0_values = [400]
     main(q_values, r_0_values, '1d_ring_1000','../../home/results/ring_directed_test',
          5,np.array([1000, 1, 1]),
          parallel=True, directed=True)
     end = time.time()
     print('zeit ', end-start)