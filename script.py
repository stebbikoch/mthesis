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
# build d_function thing
x = integer_inequality(np.array([100, 100, 1]))
x.all_numbers(49,d_given = [2,5,8,10,12,15,20,25, 30, 35, 40, 49])
x.save_to_json('2d_100_100_1')
# do the rest
start = time.time()
exponent = np.arange(16)
q_values = 10 ** (-exponent / 3)
r_0 = [5, 10, 15, 20, 25, 45]
main(q_values, r_0, '2d_100_100_1','../../home/results/2d_nondirected',
     10,np.array([100, 100, 1]),
     parallel=True, directed=False)
end = time.time()
print('zeit ', end-start)