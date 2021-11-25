from matplotlib import pyplot as plt
import numpy as np
import json
from scipy.sparse import coo_matrix
from scipy.sparse import identity
from scipy.sparse.linalg import eigsh, eigs
from scipy.sparse import *
from numba import njit, jit, prange
from multiprocessing import Pool, RawArray, Array, Process

class integer_inequality:
    def __init__(self, N):
        self.d = len(N)
        self.N = N
        self.N_tot = np.prod(self.N)

    def max_distance(self, i, j, k):
        return max(i,j,k)

    def eucl_distance(self, i, j, k):
        return (np.sqrt(i**2+j**2+k**2))

    def d_list(self, d, eucl=False):
        """
        This function is supposed to find all (positive) indices, that satisfy the inequality d(i,j)<=d and returns them
        for a value of d. Also it counts the number of those tuples. For simplicity only allow positive numbers of nodes
        per dimension.
        :param d:
        :return:
        """
        if eucl is True:
            func = self.eucl_distance
        else:
            func = self.max_distance
        degree = 0
        indices = []
        # lower limit
        lo_li = -np.minimum(d, self.N/2)
        # upper limit
        up_li = np.minimum(d+1, self.N/2+1)
        for i in range(int(lo_li[0]), int(up_li[0])):
            for j in range(int(lo_li[1]), int(up_li[1])):
                for k in range(int(lo_li[2]), int(up_li[2])):
                    if func(i,j,k) <= d and not (i==0 and j==0 and k==0):
                        degree += 1
                        indices.append([i,j,k])
        return degree, indices

    def all_numbers(self, d_max):
        self.numbers = [0]
        self.indices = []
        self.d_0 = []
        for d in range(d_max+1):
            degree, indices =self.d_list(d)
            if not degree == self.numbers[-1]:
                self.numbers.append(degree)
                self.indices.append(indices)
                self.d_0.append(d)
        self.numbers.pop(0)
        plt.step(self.d_0, self.numbers, where='post')
        plt.show()

    def save_to_json(self, name):
        with open('./d_functions/'+name+'.txt', 'w') as outfile:
            json.dump([self.numbers, self.indices, self.d_0], outfile)

class build_matrix:
    def __init__(self, d_function, N, r_0):
        self.d_function = d_function # string name of d_function
        f = open('./d_functions/'+self.d_function+'.txt', )
        # returns JSON object as
        # a dictionary
        data = json.load(f)
        self.numbers = data[0]
        self.all_indices_list = data[1]
        self.d_0s = np.array(data[2])
        self.N = N
        self.N_tot = np.prod(N)
        self.important_index = np.where(self.d_0s == r_0)[0][-1]
        self.D_0 = self.all_indices_list[self.important_index]
        self.r_0 = r_0
        #print(self.D_0)

    def all_indices(self):
        """
        This function translates the indices from point (0,0) around and adds them to a new complete list
        :return:
        """
        tuples = []
        for i in range(self.N[0]):
            for j in range(self.N[1]):
                for k in range(self.N[2]):
                    for item in self.D_0:
                        tuple = [[(0+i)%self.N[0],(0+j)%self.N[1], (0+k)%self.N[2]],
                                           [(item[0]+i)%self.N[0], (item[1]+j)%self.N[1], (item[2]+k)%self.N[2]]]
                        tuples.append(tuple)
        self.tuples = tuples
        #print('edges', len(self.tuples))

    def one_int_index_tuples_and_adjacency(self):
        array = np.array(self.tuples)
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
    def mp_worker_directed(i, rows=None, columns=None, M=None, rows_new=None, N_tot=None):
        """
        lsödkfj
        :return:
        """
        # get indices of ones in column i
        indices = np.where(columns == i)
        row = rows[indices]
        # print('input',row)
        # take out diagonal
        row_p = row[row != i]
        # draw edges to be deleted and delete
        # print(row_p, row)
        delete_indices = np.random.choice(len(row), size=min(2 * M[i], len(row)), replace=False)
        delete_indices = np.setdiff1d(delete_indices, np.where(row == i), assume_unique=True)
        row_remain = np.delete(row, delete_indices[:M[i]])
        new_indices = np.random.choice(N_tot, size=min(int(3 * M[i]), N_tot), replace=False)
        # add diagonal to existing row indices
        new_indices = np.setdiff1d(new_indices, row_remain, assume_unique=True)
        row[delete_indices[:M[i]]] = new_indices[:M[i]]
        rows_new[indices] = row

    # @staticmethod
    # def mp_directed_rewiring(rows, columns, N_tot, k, q):
    #     # input array
    #     rowss = RawArray('d',rows)
    #     print(rowss)
    #     columnss = RawArray('d',columns)
    #     new_rowss = Array('d',np.zeros(len(rows)))
    #     # edges to be removed
    #     Ms = RawArray('d',np.random.binomial(int(k), q, size=N_tot))
    #     p = Process(target=build_matrix.mp_worker_directed, np.arange(N_tot).tolist(),rows=rowss, columns=columnss, M=Ms,
    #                 new_rows=new_rowss, N_tot=N_tot)
    #     p.start()
    #     p.join()
    #     return new_rowss

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
        #print('rowlength',len(rows))
        # output array
        rows_new = np.zeros(len(rows))
        # edges to be removed
        M = np.random.binomial(int(k), q, size=N_tot)
        #print(M[5])
        for i in range(N_tot):
            # get indices of ones in column i
            indices = np.where(columns==i)
            row = rows[indices]
            #print('input',row)
            # draw edges to be deleted and delete
            #print(row_p, row)
            delete_indices = np.random.choice(len(row), size=M[i]+1, replace=False)
            #delete_indices = np.setdiff1d(delete_indices, np.where(row==i), assume_unique=True)
            # delete diagonal index
            delete_indices = delete_indices[delete_indices!=np.where(row==i)[0]]
            #print(delete_indices)
            # remaining row with diagonal!
            row_remain = np.delete(row, delete_indices[:M[i]])
            #print(row_remain)
            #row[delete_indices]=0
            # rewire edges to new heads
            #new_indices = np.random.choice(N_tot, size=min(int(3*M[i]), N_tot), replace=False)
            new_indices = np.arange(N_tot)
            np.random.shuffle(new_indices)
            #print(new_indices, row_remain)
            # add diagonal to existing row indices
            #new_indices = np.setdiff1d(new_indices, row_remain, assume_unique=True)
            for item in row_remain:
                new_indices = new_indices[new_indices != item]
            # if len(new_indices)<M[i]:
            #     print('shuffle all')
            #     new_indices = np.arange(N_tot)
            #     np.random.shuffle(new_indices)
            #     for item in row_remain:
            #         new_indices = new_indices[new_indices != item]
            #print(new_indices, 'm value',M[i])
            #np.random.shuffle(new_indices)
            #print(new_indices, type(new_indices))
            #print(delete_indices, new_indices[:M[i]], i)
            #print(M[i], row[delete_indices[:M[i]]])
            row[delete_indices[:M[i]]] = new_indices[:M[i]]
            #print('output',row)
            rows_new[indices] = row
        #print('new row length', len(rows_new), rows_new)
        return rows_new





    @staticmethod
    def fast_rewiring_directed_ith_row(L_0_i, N_tot, k, q, i):
        """
        Now we have to loop through all the rows.
        :param q:
        :return:
        """
        # new seed
        # np.random.seed()
        # edges to be removed
        M = np.random.binomial(int(k), q)
        # delete edges
        del_edges_indices = np.random.choice(np.arange(M), size=M, replace=False)
        L_rnd_i = L_0_i
        ps, qs, values = find(L_0_i)
        # delete diagonal element
        qs = qs[qs!=i]
        #print(qs)
        qs_del = qs[del_edges_indices]
        #print('deleted',qs_del)
        L_rnd_i[0,qs_del] = 0
        qs_not_del = np.setdiff1d(qs, qs_del)
        # add the diagonal
        qs_not_del = np.append(qs_not_del, i)
        #print('not deleted',qs_not_del)
        # create new edges
        new_indices = np.arange(N_tot)
        np.random.shuffle(new_indices)
        #print(new_indices)
        new_indices = np.setdiff1d(new_indices, qs_not_del, assume_unique=True)
        #print('rewired',new_indices)
        L_rnd_i[0,new_indices[:M]] = 1
        return L_rnd_i


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

    @staticmethod
    def fast_rewiring(L_0, k, q, N_tot):
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
        # adjust degrees, add ones
        # print(L_rnd.toarray())
        L_rnd[np.append(ps_del, qs_del), np.append(qs_del, ps_del)] += -np.ones(2 * len(ps_del))
        # print(L_rnd.toarray())
        L_rnd += csr_matrix((np.ones(2 * len(ps_del)), (np.append(ps_del, qs_del), np.append(ps_del, qs_del))),
                                 shape=(N_tot, N_tot))
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
    def fast_second_largest(L_rnd, N_tot, directed=False):
        if directed:
            k=abs(L_rnd[0,0])
            eigenvalues, eigenvectors = eigs(1/k * L_rnd + 1.2 * identity(N_tot), k=8, which='LM')
            eigenvalues = np.real(eigenvalues)
        else:
            D = diags(-1 / L_rnd.diagonal())
            # print(D.toarray())
            eigenvalues, eigenvectors = eigsh(D * L_rnd + 1.2 * identity(N_tot), k=8, which='LM')
        #print('eigenvalues', eigenvalues-1.2)
        second_largest = np.partition(eigenvalues.flatten(), -2)[-2]
        # print(eigenvalues)
        return second_largest - 1.2

if __name__ == "__main__":
    k=8
    q=0.5
    z = build_matrix('1d_ring_1000', np.array([100, 1, 1]), k/2)
    z.all_indices()
    z.one_int_index_tuples_and_adjacency()
    z.Laplacian_0()
    rows, columns, values = find(z.L_0)
    #new_rows = build_matrix.mp_directed_rewiring(rows, columns, z.N_tot, k, q)
    new_rows = build_matrix.numba_fast_directed_rewiring(rows, columns, z.N_tot, k, q)
    #L_rnd = csr_matrix((values, (new_rows, columns)), shape=(z.N_tot, z.N_tot))
    #L_rnd=lil_matrix((z.N_tot, z.N_tot))
    #for i in range(z.N_tot):
    #    L_rnd[i]=build_matrix.fast_rewiring_directed_ith_row(z.L_0.tolil().getrow(i), z.N_tot, k, q, i)
    #print(L_rnd.toarray())
    #lam = build_matrix.fast_second_largest(L_rnd, z.N_tot, directed=True)
    #print(lam)
    #z.random_rewiring_undirected(0.7)
    #lam = z.second_largest_eigenvalue_normalized(8, 1.2)
    #print(lam)
    #print(z.L_rnd.toarray())
    #print(z.L_0.toarray())

