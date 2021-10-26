import numpy as np
from matplotlib import pyplot as plt

class eigenvs():
    def __init__(self):
        self.N = 100**3
        self.d_0 = 10
        self.k = self.count(self.d_0)
        self.q = 0
        self.T = -self.q**2*self.k/(self.N-1-(1-self.q)*self.k)
        self.U = 1-self.q+self.q*self.k/(self.N-1-(1-self.q)*self.k)


    def count(self, d_0):
        number = 0
        for i in range(-d_0, d_0+1):
            for j in range(-d_0, d_0+1):
                for k in range(-d_0, d_0+1):
                    if i**2+j**2+k**2 <= d_0**2:
                        number += 1
        return number

    def calc_alpha(self, p):
        prefactor=1/(2*self.N*p**3*np.pi**2)
        postfactor=np.sin(2*np.pi*p*self.d_0)-2*np.pi*self.d_0/self.N**(1/3)*np.cos(2*np.pi*p*self.d_0)
        self.alpha = prefactor*postfactor
        #return alpha

    def calc_beta(self, p):
        prefactor = 1/(2*self.N*p**3*np.pi**2)
        postfactor = np.sin(2 * np.pi * p * self.N**(1/3)) - 2 * np.pi * np.cos(2 * np.pi * p * self.N**(1/3))
        self.beta = prefactor*postfactor - self.alpha

    def calc(self, p):
        self.calc_alpha(p)
        self.calc_beta(p)
        self.eigenvals = (-self.k + (1+self.U-self.T)*self.alpha+self.T*self.beta)/self.k

    def controll(self,p):
        l=p*100
        self.controll_eigenvals=(-self.k - self.d_0/(np.pi*l**3*self.N**(1/3)))/self.k


    def plot_eigenvals(self, p):
        plt.scatter(p, self.eigenvals)
        #plt.yscale('symlog')
        plt.ylim(-0.5*1e16,0.5*1e16)

if __name__ == '__main__':
    aha = eigenvs()
    aha2=eigenvs()
    print(aha.k)
    p1 = np.arange(1,500)/500
    p2=np.arange(1,100)/100
    aha.calc(p1)
    aha2.calc(p2)
    aha.controll(p2)
    plt.scatter(p2, aha2.eigenvals)
    plt.yscale('symlog')
    plt.ylim(-1-1e-6)
    plt.plot(p1, aha.eigenvals)
    #plt.plot(p2, aha.controll_eigenvals)

    #plt.ylim(-0.5 * 1e16, 0.5 * 1e16)
    plt.show()

