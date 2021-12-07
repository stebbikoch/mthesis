#from matrix import build_matrix
from matrix import integer_inequality
import numpy as np
#from scipy.sparse import coo_matrix
#from matplotlib import pyplot as plt
#import scipy.sparse
#from scipy.sparse.linalg import eigsh
#from scipy.sparse.linalg import eigs
from Reproduce import main
import time
# build d_function thing
if __name__=='__main__':
     #x = integer_inequality(np.array([50, 50, 50]))
     #x.all_numbers(24, d_given=[3, 5, 10, 15, 20])
     #x.save_to_json('3d_50_50_50')
     # do the rest
     start = time.time()
     exponent = np.arange(16)
     q_values = 10 ** (-exponent / 3)
     q_values = [1]
     r_0 = [3, 20]#[5, 10, 15, 20, 24]
     main(q_values, r_0, '3d_50_50_50','results/2d_nondirected',
          1,np.array([50, 50, 50]),
          parallel=True, directed=False)
     end = time.time()
     print('zeit ', end-start)