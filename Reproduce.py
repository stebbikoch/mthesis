from matrix import build_matrix
import json
from matrix import integer_inequality
import numpy as np
from scipy.sparse import coo_matrix
from matplotlib import pyplot as plt
import scipy.sparse
from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg import eigs

def main(q_values, k_values, name):
    dictq = {str(q):[] for q in q_values}
    dictionary = {str(k):dictq for k in k_values}
    for k in k_values:
        z = build_matrix('1d_ring_1001_by_1', 10001, 1, (k/2))
        z.all_indices()
        z.one_int_index_tuples()
        z.Laplacian_0()
        for q in q_values:
            z.random_rewiring(q)
            lam = z.second_largest_eigenvalue()
            dictionary[str(k)][str(q)].append(lam)
        print('one k done.', k)
    # save dictionary in json
    print('done', dictionary)
    with open('./reproduce/' + name + '.txt', 'w') as outfile:
        json.dump(dictionary, outfile)


if __name__ == '__main__':
    q_values = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    k_values = [20, 50, 100, 200, 400, 800]
    main(q_values, k_values, 'reproduce_1001_undirected_no_averaging')
