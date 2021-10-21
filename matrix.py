from matplotlib import pyplot as plt
import numpy as np
import json
from scipy.sparse import coo_matrix
from scipy.sparse import identity
from scipy.sparse.linalg import eigsh


class integer_inequality:
    def __init__(self, dimension, N_1, N_2):
        self.d = dimension
        self.N_1 = N_1
        self.N_2 = N_2
        self.n_max = N_1 * N_2

    def distance(self, i, j):
        if self.d==1:
            return i
        else:
            return i**2+j**2+i*j

    def number(self, d):
        """
        This function is supposed to find all (positive) indices, that satisfy the inequality d(i,j)<=d and returns them
        for a value of d. Also it counts the number of those tuples.
        :param d:
        :return:
        """
        number = 0
        indices = []
        #if len(self.indices) == d-1:
         #   number = self.numbers(d-1)
        #    indices = self.indices[d-1]
        for i in range(int(min(d**2+1, (self.N_1-1)/2+1))):
            for j in range(int(min(d**2+1, (self.N_2-1)/2+1))):
                #if (i,j) in indices:
                 #   pass
                if self.distance(i,-j) <= d and j>0 and i>0:
                    number += 2
                    indices.append((i,-j))
                if self.distance(i, j) <= d and not (j==0 and i==0):
                    number += 2
                    indices.append((i, j))
        return number, indices

    def all_numbers(self, d_max):
        self.numbers = [0]
        self.indices = []
        self.d_0 = []
        for d in range(d_max+1):
            #self.indices.append([])
            x=self.number(d)
            if not x[0] == self.numbers[-1]:
                self.numbers.append(x[0])
                self.indices.append(x[1])
                self.d_0.append(d)
            if x[0]>self.n_max:
                break
        self.numbers.pop(0)
        plt.step(self.d_0, self.numbers, where='post')
        #plt.hlines(9800,0,7203, linestyles='--')
        plt.show()

    def save_to_json(self, name):
        with open('./d_functions/'+name+'.txt', 'w') as outfile:
            json.dump([self.numbers, self.indices, self.d_0], outfile)

class build_matrix:
    def __init__(self, d_function, N_1, N_2, d_0):
        self.d_function = d_function # string name of d_function
        f = open('./d_functions/'+self.d_function+'.txt', )
        # returns JSON object as
        # a dictionary
        data = json.load(f)
        self.numbers = data[0]
        self.indices = data[1]
        self.d_0s = np.array(data[2])
        self.N_1 = N_1
        self.N_2 = N_2
        self.N = self.N_1*self.N_2
        self.important_index = np.where(self.d_0s <= d_0)[0][-1]

    def all_indices(self):
        """
        This function translates the indices from point (0,0) around and adds them to a new complete list
        :return:
        """
        indices_0 = self.indices[self.important_index]
        indices_1 = []
        for item in indices_0:
            item = [item[0]%self.N_1, item[1]%self.N_2]
            item = [-item[0] % self.N_1, -item[1] % self.N_2]
            indices_1.append(item)
        indices_tuples = []
        for i in range(self.N_1):
            for j in range(self.N_2):
                for item in indices_1:
                    tuple = [[(0+i)%self.N_1,(0+j)%self.N_2],
                                           [(item[0]+i)%self.N_1, (item[1]+j)%self.N_2]]
                    indices_tuples.append(tuple)
        self.index_tuples = indices_tuples
        #print(self.index_tuples)

    def one_int_index_tuples(self):
        # left side
        array = np.array(self.index_tuples)
        left_is = array[:, 0, 0]
        left_js = array[:, 0, 1]
        right_is = array[:, 1, 0]
        right_js = array[:, 1, 1]
        left_p = self.i_j_to_l(left_is, left_js)
        right_p = self.i_j_to_l(right_is, right_js)
        left_ps = np.append(left_p, right_p)
        right_ps = np.append(right_p, left_p)
        self.A = coo_matrix((np.ones(len(left_ps)), (left_ps, right_ps)), shape=(self.N, self.N))
        #print('kontrolllänge:', len(left_ps))

    def i_j_to_l(self, i, j):
        return i*self.N_2 + j

    def l_to_i_j(self, l):
        i = int(l/self.N_2)
        j = l - i*self.N_2
        return i,j

    def Laplacian_0(self):
        """
        Build Laplacian from Adjacency Matrix.
        :return:
        """
        degree = self.numbers[self.important_index]
        self.L_0 = -degree*identity(self.N) + self.A

    def random_rewiring(self, p):
        """
        Pick number from binomial(!) distribution. That is the number of nodes to be removed (added). Remove nodes
        arbitrarily out of existing set. Add nodes arbitrarily.
        Probability for M nodes to be picked if each node is picked with probability p is P(M)=(^n_p)*p**M*(1-p)**(n-M),
        where n is the total number of nodes.
        We want to pick a number from that distribution.
        :return:
        """
        # pick number of rewiring edges
        degree = self.numbers[self.important_index]
        edges = int(self.N * degree/ 2)
        M = np.random.binomial(edges, p)
        # delete edges
        del_edges_indices = np.random.choice(edges, size=M, replace=False)
        # this is a list containing tuples of indices
        edges_to_delete = np.array(self.index_tuples)[del_edges_indices]
        array = edges_to_delete
        # extract from this list of tuples a list of left and right i and j values
        left_is = array[:, 0, 0]
        left_js = array[:, 0, 1]
        right_is = array[:, 1, 0]
        right_js = array[:, 1, 1]
        left_p = self.i_j_to_l(left_is, left_js)
        right_p = self.i_j_to_l(right_is, right_js)
        # because of symmetry of matrix, we can exchange right and left indices
        left_ps = np.append(left_p, right_p)
        right_ps = np.append(right_p, left_p)
        # new matrix to add to old
        new_m = coo_matrix((-np.ones(len(left_ps)), (left_ps, right_ps)), shape=(self.N, self.N))
        self.L_rnd = self.L_0 + new_m
        # adjust degree
        new_m = coo_matrix((np.ones(len(left_ps)), (left_ps, left_ps)), shape=(self.N, self.N))
        self.L_rnd = self.L_rnd + new_m
        self.left_ps = left_ps
        self.right_ps = right_ps
        # add edges
        # only get zeros from upper triangle of matrix
        ps, qs = np.where((self.L_rnd.toarray()+np.tril(np.ones((self.N, self.N)))) == 0)
        add_edges_indices = np.random.choice(len(ps) , size=M, replace=False)
        ps = ps[add_edges_indices]
        qs = qs[add_edges_indices]
        self.M = M
        self.ps = ps
        self.qs = qs
        # new matrix to add to old
        new_m = coo_matrix((np.ones(2*len(ps)), (np.append(ps, qs),np.append(qs,ps))), shape=(self.N, self.N))
        self.L_rnd = self.L_rnd + new_m
        # adjust degree
        new_m = coo_matrix((-np.ones(2*len(ps)), (np.append(ps, qs),np.append(ps,qs))), shape=(self.N, self.N))
        self.L_rnd = self.L_rnd + new_m


    def save_to_json(self, name, thing_to_dump):
        """
        Make sure thing_to_dump is a list or dictionary.
        :param name:
        :param thing_to_dump:
        :return:
        """
        with open('./matrices/'+name+'.txt', 'w') as outfile:
            json.dump(thing_to_dump, outfile)

    def second_largest_eigenvalue(self):
        #self.random_rewiring(q)
        degree = self.numbers[self.important_index]
        eigenvalues, eigenvectors = eigsh(self.L_rnd + degree * identity(self.N), k=8, which='LM')
        second_largest = np.partition(eigenvalues.flatten(), -2)[-2]
        return second_largest-degree

if __name__ == "__main__":
    x = integer_inequality(2, 9, 9)
    x.all_numbers(3*8**2)
    #y=x.number(7203)
    #print(x.indices)
    #print(y[0])
    x.save_to_json('2dtorus_hexagonal_9_by_9')
    #print(zip(range(5), range(5)))
    # name = '2dtorus_tridiagonal'
    # f = open('./d_functions/'+name+'.txt', )
    # # returns JSON object as
    # # a dictionary
    # data = json.load(f)
    # plt.step(data[2], data[0], where='post')
    # plt.hlines(9800, 0, 7203, linestyles='--')
    # print(data[0][-1])
    # plt.show()
    #x = np.array([1,2,5,8,15,16, 20])
    #print(np.where(x<=15)[0][-1])
    #z = build_matrix('2dtorus_tridigonal', 99, 99, 200)
    #z.all_indices()
    #z.one_int_index_tuples()
    # z.Adjacency_0()
    #z.Laplacian_0()
    #z.random_rewiring(0.05)
    # print(z.L_0)
    #print('functioning')