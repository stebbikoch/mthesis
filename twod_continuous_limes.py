import numpy as np
from matplotlib import pyplot as plt
from scipy.special import jv
from analytics import *

class eigenvs():
    def __init__(self, radius=None, q=0):
        self.N = 100**2
        if radius:
            self.d_0 = radius
        else:
            self.d_0 = 10
        self.k = self.count(self.d_0)
        self.q = q
        self.Delta_2 = self.q*self.k/(self.N-1-(1-self.q)*self.k)
        self.Delta_1 = -self.q + self.q*self.Delta_2


    def count(self, d_0):
        number = 0
        self.d_list = []
        for i in range(-d_0, d_0+1):
            for j in range(-d_0, d_0+1):
                if i ** 2 + j ** 2  <= d_0 ** 2:
                    if not (i==0 and j==0):
                        number += 1
                        self.d_list.append([i,j])
        return number

    def calc_eigenvals_discrete(self, p_array):
        alpha = []
        for p in p_array:
            alpha.append(np.sum(np.exp(1j*2*np.pi*np.dot(np.array(self.d_list), p))))
        self.alpha_dis_k = np.array(alpha)/self.k
        output = np.array(self.alpha_dis_k)*(1+self.Delta_1-self.Delta_2) -self.k -self.Delta_2
        self.eigenvals_discrete = output/self.k
        return self.eigenvals_discrete
        #return output/self.k



    def calc_eigenvals(self,p):
        output = -self.k
        a= 1+ self.Delta_1-self.Delta_2
        output += (- self.q * self.k) / (self.N - 1 - (1 - self.q) * self.k)
        #self.other1 = a*(np.sin(2*np.pi*self.d_0*p)-2*np.pi*self.d_0*p*np.cos(2*np.pi*self.d_0*p))/self.k
        #self.other2 = (2*np.pi**2*p**3*self.N)/self.k
        #self.other3 = output/self.k*np.ones(len(p))
        self.alpha = 1/(2*np.pi*p**2)*jv(1, 2*np.pi*self.d_0*p)*2*np.pi*self.d_0*p#(np.sin(2*np.pi*self.d_0*p)/(2*np.pi*self.d_0*p)-np.cos(2*np.pi*self.d_0*p))
        self.alpha_k = self.alpha/self.k
        output += a * (self.alpha) - self.Delta_2
        self.eigenvals=output/self.k
        return self.eigenvals

    def plot_eigenvals(self, p):
        plt.scatter(p, self.eigenvals)
        #plt.yscale('symlog')
        plt.ylim(-0.5*1e16,0.5*1e16)

if __name__ == '__main__':
    #p = 10**(-np.arange(1, 151) / 50)
    n = 100
    Gamma = np.array([0,0,0])
    R = np.array([1,1,0])*0.5
    Y= np.array([0.5,0,0])
    p_array = np.array([Gamma + i/n * R for i in range(1, n+1)])
    p_array = np.append(p_array, np.array([R + i/n * (Y-R) for i in range(1, n+1)]), axis=0)
    p_array = np.append(p_array, np.array([Y + i/n*(Gamma-Y) for i in range(1,n)]), axis=0)
    #print(p_array)
    p_array_2 = np.array([np.linalg.norm(item) for item in p_array])
    #print(p_array_2)
    p_diff_for_plot = np.array([np.linalg.norm(p_array[i+1]-p_array[i]) for i in range(len(p_array)-1)])
    p_diff_for_plot = np.append( np.zeros(1), p_diff_for_plot)
    p_for_plot = np.array([np.sum(p_diff_for_plot[0:i]) for i in range(1, len(p_diff_for_plot)+1)])
    print(p_for_plot)
    fig = plt.figure()
    instance = analytics(1,1,10000)
    r_values=[10]
    for r in r_values:
        plt.plot(p_for_plot, instance.exact_alpha('2d_100_100_eucl', p_array, r, multi_p=True))
        #aha = eigenvs(radius=i)
        #eigenvals_discrete= aha.calc_eigenvals_discrete(p_array)
        #eigenvals1=aha.calc_eigenvals(p_array_2)
        ##eigenvals2=aha.calc_eigenvals(p_2)
        #plt.plot(p_for_plot, eigenvals1, label=r'$k=$'+str(aha.k)+', $r_0=$'+str(aha.d_0))
        #plt.plot(p_2, eigenvals2, 'o', label=r'$k=$' + str(aha.k) + ', $r_0=$' + str(aha.d_0))
        #plt.plot(p_for_plot, eigenvals_discrete, '--', label=r'$k=$'+str(aha.k)+', $r_0=$'+str(aha.d_0))
        #plt.plot(p, aha.other1, '--')
        #plt.plot(p, aha.other2, '--')
        #plt.plot(p, aha.other3, '--')
    plt.xlabel(r'Absolute value of reciprocal index $|p|$')
    plt.ylabel(r'Normalized mean-field eigenvalues $\lambda(|p|)$')
    #plt.xscale('log')
    plt.legend()
    #fig.savefig('continuous_eigenvalues.svg', format='svg', dpi=1000)
    plt.show()
