from matplotlib import pyplot as plt
import numpy as np
import json
from scipy.sparse import coo_matrix
from scipy.sparse import identity
from scipy.sparse.linalg import eigsh
from scipy.sparse import *


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
        print(self.D_0)

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
        print('edges', len(self.tuples))

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

    def random_rewiring_undirected(self, q):
        # pick number of rewiring edges
        k = self.numbers[self.important_index]
        edges = int(self.N_tot * k/ 2)
        M = np.random.binomial(edges, q)
        # delete edges
        del_edges_indices = np.random.choice(edges, size=M, replace=False)
        # find ones in upper triangle of matrix
        self.L_rnd = self.L_0
        ps, qs, values = find(triu(self.L_rnd, k=1))
        ps_del, qs_del = ps[del_edges_indices], qs[del_edges_indices]
        # adjust degrees, add ones
        #print(self.L_rnd.toarray())
        self.L_rnd[np.append(ps_del, qs_del), np.append(qs_del, ps_del)] += -np.ones(2*len(ps_del))
        #print(self.L_rnd.toarray())
        self.L_rnd += csr_matrix((np.ones(2 * len(ps_del)), (np.append(ps_del, qs_del), np.append(ps_del, qs_del))),
                                 shape=(self.N_tot, self.N_tot))
        #self.L_rnd[np.append(ps_del, qs_del), np.append(ps_del, qs_del)] += np.ones(2*len(ps_del))
        #print(self.L_rnd.toarray())
        # only get zeros from upper triangle of matrix
        # counter=0
        # # change to lil format for better performance
        # self.L_rnd = self.L_rnd.tolil()
        # while counter<len(ps_del):
        #     p = np.random.randint(0,self.N_tot, size=1)[0]
        #     q = ((p + np.random.randint(0,self.N_tot-1, size=1))%self.N_tot)[0]
        #     print(p,q)
        #     if self.L_rnd[p,q]==0:
        #         self.L_rnd[[p,q],[q,p]]=1
        #         #print(self.L_rnd.toarray())
        #         #print(type(self.L_rnd))
        #         #print(p,q)
        #         self.L_rnd-=csr_matrix((np.array([1,1]),(np.array([q,p]),np.array([q,p]))), shape=(self.N_tot, self.N_tot))
        #         counter += 1
        #         if counter>200:
        #             print('more than 100')
        #             quit()
        #print('last matrix', self.L_rnd.toarray())
        ps, qs = np.where((self.L_rnd.toarray() + np.tril(np.ones((self.N_tot, self.N_tot)))) == 0)
        add_edges_indices = np.random.choice(len(ps), size=M, replace=False)
        ps = ps[add_edges_indices]
        qs = qs[add_edges_indices]
        # new matrix to add to old
        self.L_rnd += coo_matrix((np.ones(2 * len(ps)), (np.append(ps, qs), np.append(qs, ps))), shape=(self.N_tot, self.N_tot))
        # adjust degree
        self.L_rnd += coo_matrix((-np.ones(2 * len(ps)), (np.append(ps, qs), np.append(ps, qs))), shape=(self.N_tot, self.N_tot))

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
        shift = fact
        eigenvalues, eigenvectors = eigsh(D*self.L_rnd + shift * identity(self.N_tot), k=numb, which='LM')
        second_largest = np.partition(eigenvalues.flatten(), -2)[-2]
        print(eigenvalues)
        return second_largest-shift

if __name__ == "__main__":
    z = build_matrix('1d_ring_1000', np.array([1000, 1, 1]), 20)
    z.all_indices()
    z.one_int_index_tuples_and_adjacency()
    z.Laplacian_0()
    z.random_rewiring_undirected(0.7)
    lam = z.second_largest_eigenvalue_normalized(8, 1.2)
    print(lam)
    #print(z.L_rnd.toarray())
    #print(z.L_0.toarray())

