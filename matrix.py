from matplotlib import pyplot as plt
import numpy as np
import json
from scipy.sparse import coo_matrix
from scipy.sparse import identity
from scipy.sparse.linalg import eigsh, eigs
from scipy.sparse import *
from numba import njit, jit, prange
from multiprocessing import Pool, RawArray, Array, Process
import time
#from histogram_plots import gaussian

class integer_inequality:
    def __init__(self, N):
        self.d = len(N)
        self.N = N
        self.N_tot = np.prod(self.N)

    def max_distance(self, i, j, k):
        return max(abs(i),abs(j),abs(k))

    def eucl_distance(self, i, j, k):
        return (np.sqrt(i**2+j**2+k**2))

    def eucl_tightest(self, i, j, k):
        return np.sqrt(i**2 + j**2 + i*j)

    def d_list(self, d, eucl=False, tightest=False):
        """
        This function is supposed to find all (positive) indices, that satisfy the inequality d(i,j)<=d and returns them
        for a value of d. Also it counts the number of those tuples. For simplicity only allow positive numbers of nodes
        per dimension.
        :param d:
        :return:
        """
        if eucl is True:
            func = self.eucl_distance
        elif tightest is True:
            func = self.eucl_tightest
        else:
            func = self.max_distance
        degree = 0
        indices = []
        # lower limit
        lo_li = -self.N/2
        # upper limit
        up_li = self.N/2+1
        for i in range(int(lo_li[0]), int(up_li[0])):
            for j in range(int(lo_li[1]), int(up_li[1])):
                for k in range(int(lo_li[2]), int(up_li[2])):
                    if func(i,j,k) <= d and not (i==0 and j==0 and k==0):
                        degree += 1
                        indices.append([i,j,k])
        return degree, indices

    def all_numbers(self, d_max, d_given=None, eucl=False, tightest=False):
        self.numbers = [0]
        self.indices = []
        self.d_0 = []
        iterator = range(d_max+1)
        if d_given:
            iterator = d_given
        for d in iterator:
            degree, indices =self.d_list(d, eucl=eucl, tightest=tightest)
            if not degree == self.numbers[-1]:
                self.numbers.append(degree)
                self.indices.append(indices)
                self.d_0.append(d)
        self.numbers.pop(0)
        #plt.step(self.d_0, self.numbers, where='post')
        #plt.show()

    def save_to_json(self, name):
        with open('./d_functions/'+name+'.json', 'w') as outfile:
            json.dump([self.numbers, self.indices, self.d_0], outfile)

    @property
    def data(self):
        return [self.numbers, self.indices, self.d_0]

class build_matrix:
    def __init__(self, N, r_0, data=None, d_function=None):
        if data is None:
            try:
                f = open('../d_functions/'+d_function+'.json')
            except:
                f = open('d_functions/' +d_function + '.json')
            # returns JSON object as
            # a dictionary
            data = json.load(f)
        self.numbers = data[0]
        self.all_indices_list = data[1]
        self.d_0s = np.array(data[2])
        self.N = N
        self.N_tot = np.prod(N)
        #print(np.where(self.d_0s == r_0))
        self.important_index = np.where(self.d_0s == r_0)[0][-1]
        #print(self.all_indices_list, self.important_index)
        self.D_0 = self.all_indices_list[self.important_index]
        #print('len', len(self.D_0), len(self.D_0[0]))
        self.r_0 = r_0
        self.k = len(self.D_0)
        #print(self.D_0)

    @staticmethod
    @njit()
    def fast_all_indices(D_0, N):
        """
        This function translates the indices from point (0,0) around and adds them to a new complete list
        :return:
        """
        tuples = np.zeros((np.prod(N)*len(D_0),2, 3))
        counter = 0
        for i in range(N[0]):
            for j in range(N[1]):
                for k in range(N[2]):
                    for f in range(len(D_0)):
                        item = D_0[f]
                        tuples[counter] = np.array([[(0+i)%N[0],(0+j)%N[1], (0+k)%N[2]],
                                           [(item[0]+i)%N[0], (item[1]+j)%N[1], (item[2]+k)%N[2]]])
                        counter += 1
        return tuples

    def all_indices(self):
        self.tuples = build_matrix.fast_all_indices(np.array(self.D_0), self.N)
        print('done, tuples made')

    def one_int_index_tuples_and_adjacency(self):
        array = self.tuples
        left_is = array[:, 0, 0]
        left_js = array[:, 0, 1]
        left_ks = array[:, 0, 2]
        right_is = array[:, 1, 0]
        right_js = array[:, 1, 1]
        right_ks = array[:, 1, 2]
        self.left_p = self.i_j_k_to_p(left_is, left_js, left_ks)
        self.right_p = self.i_j_k_to_p(right_is, right_js, right_ks)
        #self.left_ps = np.append(left_p, right_p)
        #self.right_ps = np.append(right_p, left_p)
        self.A = coo_matrix((np.ones(len(self.left_p)), (self.left_p, self.right_p)), shape=(self.N_tot, self.N_tot))

    def i_j_k_to_p(self, i, j, k):
        return i+j*self.N[0]+k*self.N[0]*self.N[1]

    def p_to_i_j_k(self, p):
        k = int(p/(self.N[0]*self.N[1]))
        j = int((p-k*self.N[0]*self.N[1])/self.N[0])
        i = p-k*self.N[0]*self.N[1]-j*self.N[0]
        return i,j,k

    def Laplacian_0(self):
        """
        Build Laplacian from Adjacency Matrix.
        :return:
        """
        self.degree = self.numbers[self.important_index]
        self.L_0 = -self.degree*identity(self.N_tot) + self.A

    @staticmethod
    @njit
    def numba_fast_directed_rewiring(rows, columns, N_tot, k, q):
        """
        Gets indices of matrix elements of ordered Laplacian as rows and columns vectors. Return new rows and columns
        vectors, representing the new indices. Hopefully parallizable with numba. This functions columnwise.
        :param rows:
        :param columns:
        :param N_tot:
        :param k:
        :param q:
        :return:
        """
        # output array
        rows_new = np.zeros(len(rows))
        ## edges to be removed
        #M = np.random.binomial(int(k), q, size=N_tot)
        #print(M)
        for i in range(N_tot):
            # get indices of ones in column i
            indices = np.where(columns==i)
            row = rows[indices]
            # draw M now for correct k (in sphere case, not every node has same k)
            M = np.random.binomial(len(row)-1, q)
            # draw edges to be deleted and delete
            delete_indices = np.random.choice(len(row), size=M+1, replace=False)
            # delete diagonal index
            delete_indices = delete_indices[delete_indices!=np.where(row==i)[0]]
            #print(delete_indices)
            # remaining row with diagonal!
            row_remain = np.delete(row, delete_indices[:M])
            # rewire edges to new heads
            new_indices = np.arange(N_tot)
            np.random.shuffle(new_indices)
            for item in row_remain:
                new_indices = new_indices[new_indices != item]
            row[delete_indices[:M]] = new_indices[:M]
            rows_new[indices] = row
        return rows_new

    def random_rewiring_undirected(self, q):
        # pick number of rewiring edges
        k = self.numbers[self.important_index]
        edges = int(self.N_tot * k/ 2)
        M = np.random.binomial(edges, q)
        # delete edges
        del_edges_indices = np.random.choice(edges, size=M, replace=False)
        # find ones in upper triangle of matrix
        self.L_rnd = self.L_0.copy()
        ps, qs, values = find(triu(self.L_rnd, k=1))
        #print('edges, ps', edges, len(ps))
        ps_del, qs_del = ps[del_edges_indices], qs[del_edges_indices]
        # adjust degrees, add ones
        #print(self.L_rnd.toarray())
        self.L_rnd[np.append(ps_del, qs_del), np.append(qs_del, ps_del)] += -np.ones(2*len(ps_del))
        #print(self.L_rnd.toarray())
        self.L_rnd += csr_matrix((np.ones(2 * len(ps_del)), (np.append(ps_del, qs_del), np.append(ps_del, qs_del))),
                                 shape=(self.N_tot, self.N_tot))
        ps, qs = np.where((self.L_rnd.toarray() + np.tril(np.ones((self.N_tot, self.N_tot)))) == 0)
        add_edges_indices = np.random.choice(len(ps), size=M, replace=False)
        ps = ps[add_edges_indices]
        qs = qs[add_edges_indices]
        # new matrix to add to old
        self.L_rnd += coo_matrix((np.ones(2 * len(ps)), (np.append(ps, qs), np.append(qs, ps))), shape=(self.N_tot, self.N_tot))
        # adjust degree
        self.L_rnd += coo_matrix((-np.ones(2 * len(ps)), (np.append(ps, qs), np.append(ps, qs))), shape=(self.N_tot, self.N_tot))

    def special_rewiring_1d(self, m, directed=False):
        """
        Delete a number of next neighbor edges randomly and redistribute them as diagonal shortcut edges.
        :param m: number of next neighbor edges to be deleted
        :param directed:
        :return:
        """
        assert m<=self.N_tot, 'serious problem here!!!'
        # All together, there are N_tot edges to pick from
        ps_del = np.random.choice(self.N_tot, size=m, replace=False)
        qs_del = (ps_del+1)%self.N_tot
        self.L_rnd=self.L_0.copy()
        # adjust degrees, delete ones
        self.L_rnd[np.append(ps_del, qs_del), np.append(qs_del, ps_del)] += -np.ones(2 * len(ps_del))
        self.L_rnd += csr_matrix((np.ones(2 * len(ps_del)), (np.append(ps_del, qs_del), np.append(ps_del, qs_del))),
                            shape=(self.N_tot, self.N_tot))
        # redistribute
        ps_add = np.random.choice(int(self.N_tot), size=m, replace=False)
        qs_add = (ps_add+self.N_tot/2-1)%self.N_tot
        # new matrix to add to old
        self.L_rnd += coo_matrix((np.ones(2 * len(ps_add)), (np.append(ps_add, qs_add), np.append(qs_add, ps_add))),
                            shape=(self.N_tot, self.N_tot))
        # adjust degree
        self.L_rnd += coo_matrix((-np.ones(2 * len(ps_add)), (np.append(ps_add, qs_add), np.append(ps_add, qs_add))),
                            shape=(self.N_tot, self.N_tot))

    def gaussian_rewiring_1d(self, q, mu=300, sigma=10):
        if q==0:
            self.L_rnd=self.L_0.copy()
            return
        # pick number of rewiring edges
        edges = int(self.N_tot * self.k / 2)
        M = np.random.binomial(edges, q)
        # delete edges
        del_edges_indices = np.random.choice(edges, size=M, replace=False)
        # find ones in upper triangle of matrix
        self.L_rnd = self.L_0.copy()
        ps, qs, values = find(triu(self.L_rnd, k=1))
        # print('edges, ps', edges, len(ps))
        ps_del, qs_del = ps[del_edges_indices], qs[del_edges_indices]
        # adjust degrees, delete ones
        self.L_rnd[np.append(ps_del, qs_del), np.append(qs_del, ps_del)] += -np.ones(2 * len(ps_del))
        self.L_rnd += csr_matrix((np.ones(2 * len(ps_del)), (np.append(ps_del, qs_del), np.append(ps_del, qs_del))),
                            shape=(self.N_tot, self.N_tot))
        # now redistribute edges gaussian wise
        success = False
        while not success:
            lengthes = np.round(np.random.normal(loc=mu, scale=sigma, size=2*M)).astype(int)
            lengthes = lengthes[lengthes<500]
            assert len(lengthes)>=M, 'Length of lengthes variable is too short ({})!!'.format(len(lengthes))
            lengthes = lengthes[:M]
            histo = np.histogram(lengthes, range=(1,500), bins=500)[0]
            # more than 1000 in one bin? that's a problem!!
            if np.max(histo)<=self.N_tot:
                success=True
            else:
                print("turn another round.")
        #assert np.max(histo)<=self.N_tot, 'Redistribution of edges has problems, maximum larger than N_tot!!!'
        #print(histo)
        indices=np.where(histo!=0)[0]
        for i in range(len(indices)):
            length = indices[i]+1
            m = histo[indices[i]]
            #print('Length and number: {}, {}'.format(length, m))
            ps_add = np.random.choice(self.N_tot, size=m, replace=False)
            qs_add = (ps_add + length) % self.N_tot
            # new matrix to add to old
            self.L_rnd += coo_matrix(
                (np.ones(2*len(ps_add)), (np.append(ps_add, qs_add), np.append(qs_add, ps_add))),
                shape=(self.N_tot, self.N_tot))
            # adjust degree
            self.L_rnd += coo_matrix(
                (-np.ones(2*len(ps_add)), (np.append(ps_add, qs_add), np.append(ps_add, qs_add))),
                shape=(self.N_tot, self.N_tot))
        #plt.imshow(self.L_rnd.toarray(), vmin=-1, vmax=2)
        #plt.show()



    @staticmethod
    def fast_rewiring_undirected(L_0, k, q, N_tot, save_mem=False):
        # pick number of rewiring edges
        edges = int(N_tot * k / 2)
        M = np.random.binomial(edges, q)
        # delete edges
        del_edges_indices = np.random.choice(edges, size=M, replace=False)
        # find ones in upper triangle of matrix
        L_rnd = L_0.copy()
        ps, qs, values = find(triu(L_rnd, k=1))
        # print('edges, ps', edges, len(ps))
        ps_del, qs_del = ps[del_edges_indices], qs[del_edges_indices]
        # adjust degrees, delete ones
        L_rnd[np.append(ps_del, qs_del), np.append(qs_del, ps_del)] += -np.ones(2 * len(ps_del))
        L_rnd += csr_matrix((np.ones(2 * len(ps_del)), (np.append(ps_del, qs_del), np.append(ps_del, qs_del))),
                                 shape=(N_tot, N_tot))
        # now redistribute ones and adjust degrees again
        if save_mem:
            rows, columns, values = find(L_0)
            new_rows, new_columns = build_matrix.undirected_save_mem(rows, columns, M, N_tot)
            # add new ones
            L_rnd += csr_matrix((np.ones(2*len(new_rows)), (np.append(new_rows, new_columns),
                                                            np.append(new_columns, new_rows))), shape=(N_tot, N_tot))
            # adjust degrees
            L_rnd += csr_matrix((-np.ones(2*len(new_rows)), (np.append(new_rows, new_columns),
                                                            np.append(new_rows, new_columns))), shape=(N_tot, N_tot))

        else:
            ps, qs = np.where((L_rnd.toarray() + np.tril(np.ones((N_tot, N_tot)))) == 0)
            add_edges_indices = np.random.choice(len(ps), size=M, replace=False)
            ps = ps[add_edges_indices]
            qs = qs[add_edges_indices]
            # new matrix to add to old
            L_rnd += coo_matrix((np.ones(2 * len(ps)), (np.append(ps, qs), np.append(qs, ps))),
                                     shape=(N_tot, N_tot))
            # adjust degree
            L_rnd += coo_matrix((-np.ones(2 * len(ps)), (np.append(ps, qs), np.append(ps, qs))),
                                     shape=(N_tot, N_tot))
        return L_rnd

    @staticmethod
    @njit
    def undirected_save_mem(rows, columns, M, N_tot):
        # concatenate rows and colums in 2d array
        z = rows+N_tot*columns
        y = np.zeros(M)
        res_size = 0
        while res_size < M:
            # row index
            r = np.random.randint(N_tot)
            # column index, but not the same as r
            c = (r+np.random.randint(N_tot))%N_tot
            # total index
            i = r + N_tot*c
            if not i in z and not i in y:
                y[res_size] = i
            res_size +=1
        # convert y back to row and column indices
        columns_new = np.trunc(y/N_tot)
        rows_new = y-columns_new*N_tot
        return rows_new, columns_new




    def random_rewiring_stick_with_source(self, p):
        """
        Pick number from binomial(!) distribution. That is the number of nodes to be removed (added). Remove nodes
        arbitrarily out of existing set. Add nodes arbitrarily.
        Probability for M nodes to be picked if each node is picked with probability p is P(M)=(^n_p)*p**M*(1-p)**(n-M),
        where n is the total number of nodes.
        We want to pick a number from that distribution.
        :return:
        """
        self.q = p
        # pick number of rewiring edges
        degree = self.numbers[self.important_index]
        edges = int(self.N_tot * degree/ 2)
        M = np.random.binomial(edges, p)
        # delete edges
        del_edges_indices = np.random.choice(edges, size=M, replace=False)
        # this is a list containing tuples of indices
        band_low_part = tril(self.L_0, k=(self.r_0+1)-self.N_tot)
        upper_triangle = triu(self.L_0, k=1)
        band_up_part = upper_triangle - triu(upper_triangle, k=(self.r_0+1))
        band = band_low_part + band_up_part
        ps, qs, values = find(band)
        ps, qs = ps[del_edges_indices], qs[del_edges_indices]
        # sort with respect to ps
        args = np.argsort(ps)
        ps = ps[args]
        qs = qs[args]
        #print('sorted ps', ps, qs)
        # resort the qs
        for p in list(set(ps)):
            args = np.where(ps==p)
            new_qargs = np.argsort((qs[args]-p)%self.N_tot)
            #print('args', args,'new_args', new_qargs)
            qs[args] = qs[args][new_qargs]
        #print('sorted qs', qs)
        # go through array
        self.L_rnd = self.L_0
        for i in range(len(ps)):
            # delete edge and adjust degree
            self.L_rnd += coo_matrix(([-1, -1, 1], ([ps[i], qs[i], qs[i]], [qs[i], ps[i], qs[i]])),
                                   shape=(self.N_tot, self.N_tot))
            # look for free nodes in that row
            free_qs=np.where(self.L_rnd.getrow(ps[i]).toarray()[0]==0)
            # choose one of them
            new_q=np.random.choice(free_qs[0], size=1)[0]
            # rewire, adjust degree
            self.L_rnd += coo_matrix(([1,1,-1], ([ps[i], new_q, new_q], [new_q, ps[i], new_q])),
                                   shape=(self.N_tot, self.N_tot))



    def save_to_json(self, name, thing_to_dump):
        """
        Make sure thing_to_dump is a list or dictionary.
        :param name:
        :param thing_to_dump:
        :return:
        """
        with open('./matrices/'+name+'.txt', 'w') as outfile:
            json.dump(thing_to_dump, outfile)

    def second_largest_eigenvalue_normalized(self, numb, fact):
        D = diags(-1/self.L_rnd.diagonal())
        #print(D.toarray())
        eigenvalues, eigenvectors = eigsh(D*self.L_rnd + fact * identity(self.N_tot), k=numb, which='LM')
        second_largest = np.partition(eigenvalues.flatten(), -2)[-2]
        #print(eigenvalues)
        return second_largest-fact

    @staticmethod
    def arnoldi_eigenvalues(L_rnd, N_tot, directed=False, smallest=False, normalized=True, adjacency=False,
                            weighted=False, newman=False):
        L_rnd = L_rnd.copy()
        shift=2
        if smallest:
            if adjacency:
                shift = -0.5
            else:
                shift=0
        if directed:
            k=abs(L_rnd[0,0])
            if adjacency:
                L_rnd.setdiag(0)
            eigenvalues, eigenvectors = eigs(1/k * L_rnd + shift * identity(N_tot), k=4, ncv=30, which='LM')
            eigenvalues = np.real(eigenvalues)
        else:
            if normalized:
                D = diags(-1 / L_rnd.diagonal())
            elif weighted or newman:
                D = 1
            else:
                D = -1/np.mean(L_rnd.diagonal())
            if adjacency:
                L_rnd.setdiag(0)
            eigenvalues, eigenvectors = eigsh(D * L_rnd + shift * identity(N_tot), k=4, ncv=30, which='LM')
        #print('eigenvalues', eigenvalues-1.2)
        if smallest:
            output = np.partition(eigenvalues.flatten(), -2)[0]
        else:
            output = np.partition(eigenvalues.flatten(), -2)[-2]
        # print(eigenvalues)
        return output - shift

    def weighted_laplacian(self, L_rnd):
        """
        Defines an attribute L_rnd_weighted and asigns to it the weighted laplacian. The weighted Laplacian is calculated
        by elementwise multiplication of the Laplacian with a weight-matrix. Works for one dimension only.
        :return:
        """
        n = self.N_tot
        weight_values = [1 / i for i in range(1, int(n / 2 + 1))]
        weight_matrix = diags(weight_values, [i for i in range(1, int(n / 2 + 1))], shape=(n, n))
        weight_matrix += diags(weight_values, [-i for i in range(1, int(n / 2 + 1))], shape=(n, n))
        weight_values = weight_values[:-1]
        weight_matrix += diags(weight_values, [n - i for i in range(1, int(n / 2))], shape=(n, n))
        weight_matrix += diags(weight_values, [-n + i for i in range(1, int(n / 2))], shape=(n, n))
        A_weighted = csr_matrix(L_rnd.toarray()*weight_matrix.toarray())
        for i in range(n):
            A_weighted[i,i] = - np.sum(A_weighted[i,:])
        self.L_weighted = A_weighted
        return A_weighted

if __name__ == "__main__":
    x = integer_inequality(np.array([1000, 1, 1]))
    #x.all_numbers(49, d_given=[1.58, 3.32, 4.99, 6.6, 8.29, 10.78], tightest=True)
    x.all_numbers(500, d_given=[])
    x.save_to_json('1d_1000')
