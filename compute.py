import json
from matplotlib import pyplot as plt
import numpy as np

class compute_analytically:
    def __init__(self, filename_d_function, N, N_vec):
        f = open('./d_functions/'+filename_d_function+'.txt', )
        # returns JSON object as
        # a dictionary
        data = json.load(f)
        self.N=N
        self.k = data[0]
        self.d_0 = data[2]
        self.D = data[1]
        self.all_indices = np.array([[i,j] for i in range(int((N_vec[0]+1)/2))
                                     for j in range(int(-(N_vec[1]-3)/2), int((N_vec[1]+1)/2))])
        print(self.all_indices)
    def plot_d_function(self):
        plt.step(self.d_0, self.k, where='post')
        plt.show()

    def eival(self, p, d_0, q, ordered=False):
        index = np.where(np.array(self.d_0)<=d_0)[0][-1]
        D = np.array(self.D[index])
        k = self.k[index]
        #print('k', k)
        U = q*(1-k/(self.N-1))
        T = q*(k/(self.N-1))
        #print('l',len(D))
        lam_0 = -k + 2 * np.sum(np.cos(2*np.pi*np.dot(D,p)))
        #print('lam_0', lam_0)
        #print(2*np.pi*np.dot(D,p))
        #print('summe',np.sum(np.cos(2*np.pi*np.dot(D,p))))
        delta_lam = -2*(U+T)*np.sum(np.cos(2*np.pi*np.dot(D,p))) \
                    + 2*T*np.sum(np.cos(2*np.pi*np.dot(self.all_indices, p)))
        #print('summe2', np.sum(np.cos(2*np.pi*np.dot(self.all_indices, p))))
        #print('delta', delta_lam)
        if ordered is True:
            return lam_0
        else:
            return lam_0 + delta_lam

if __name__ == '__main__':
    z = compute_analytically('2dtorus_hexagonal_5_by_5', 25, [5,5])
    #print(z.eival(np.array([0, 0]), 4, 0.5))
    data_input = [[z.eival(np.array([(x-2)/5, (y-2)/5]), 4, 0.05) for x in range(5)] for y in range(5)]#plt.imshow(x.eival(np.array([x/9, y/9]), 20, 0.05) for x in range(9) for y in range(9))
    data_input1 = [[z.eival(np.array([(x - 2) / 5, (y - 2) / 5]), 4, 0.05, ordered=True) for x in range(5)] for y in
                  range(5)]  # plt.imshow(x.eival(np.array([x/9, y/9]), 20, 0.05) for x in range(9) for y in range(9))

    #print(z.eival(np.array([0,0]), 20, 0.05))
    fig, axs=plt.subplots(3)
    axs[0].imshow(data_input)
    axs[1].imshow(data_input1)
    axs[2].imshow(np.array(data_input)-np.array(data_input1))
    #z.plot_d_function()
    print(data_input)
    plt.show()
