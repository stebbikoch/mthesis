import numpy as np
from scipy.special import legendre as lg
from matplotlib import pyplot as plt
from legendrepolynomials import l_approx
import json

class analytics:
    def __init__(self,k, q, N):
        self.k = k
        self.q = q
        self.N = N

    def D_0_load(self, d_function):
        try:
            f = open('./d_functions/' + d_function + '.json', )
        except:
            f = open('./d_functions/' + d_function + '.txt', )
        # returns JSON object as
        # a dictionary
        data = json.load(f)
        self.numbers = data[0]
        self.all_indices_list = data[1]
        self.d_0s = np.array(data[2])

    def q0_eigenvalues(self,a=1, r=None):
        self.k=(2*r+1)**3-1
        index=np.where(self.d_0s==r)[0][0]
        d_vector = np.array(self.all_indices_list[index])
        output = -self.k+1/a*np.sum(np.exp(1j*np.pi*2*1/20*d_vector[:,0]))
        return np.real(output/self.k)

    @property
    def r_0(self):
        return self.k/2

    @property
    def Delta_2(self):
        return self.q * self.k / (self.N - 1 - (1 - self.q) * self.k)

    @property
    def Delta_1(self):
        return -self.q + self.q * self.Delta_2

    def second_lam_three_dim(self, q=None, r=None, smallest=False):
        self.q = q
        self.k = (2*r+1)**3-1
        if smallest:
            p_l=np.round(1.5*20/ (2 * r + 1))/20
            p = np.array([1e-9, 1e-9, p_l])
        else:
            p = np.array([1/20, 1e-9, 1e-9])
        alpha = np.prod(np.sin((2*r+1)*np.pi*p)/np.sin(np.pi*p))-1
        return (-self.k + (1+self.Delta_1-self.Delta_2)*alpha -self.Delta_2)/self.k

    def second_lam_two_dim(self):
        p = np.array([1e-9, 1 / np.sqrt(self.N)])
        alpha = np.prod(np.sin((2 * self.r_0 + 1) * np.pi * p) / np.sin(np.pi * p)) - 1
        return -self.k + (1 + self.Delta_1 - self.Delta_2) * alpha - self.Delta_2

    def second_lam_one_dim(self, q=None, k=None):
        if q:
            self.q = q
        if k:
            self.k=k
        p = 1/self.N
        alpha = np.sin((2 * self.r_0 + 1) * np.pi * p) / np.sin(np.pi * p) - 1
        return -self.k + (1 + self.Delta_1 - self.Delta_2) * alpha - self.Delta_2

    def second_lam_sphere(self, q, r_0):
        self.q = q
        theta_0 = 2*np.arcsin(r_0/2)
        # average k (ideally)
        self.k = self.N * (1-np.cos(theta_0))/2
        #print('k: ', self.k)
        alpha = np.pi*np.sin(theta_0)**2*1/(4*np.pi)*self.N
        #Delta1 = 1-(q*2*np.pi*(1-np.cos(theta_0))+q*np.pi*(1-np.cos(theta_0))**2)*self.N/(4*np.pi)
        #Delta2 = q*(1-np.cos(theta_0)**2)/4*self.N/(4*np.pi)
        out = -1 + (1-q + (q**2-q)/(1/np.sin(theta_0/2)**2-(1-q)))*np.cos(theta_0/2)**2
        return out#(-self.k + (1+self.Delta_1-self.Delta_2) * alpha)/self.k

    def smallest_lam_sphere(self, q, r_0):
        self.q = q
        theta_0 = 2*np.arcsin(r_0/2)
        x = np.cos(theta_0)
        self.k = self.N * (1-np.cos(theta_0))/2
        l = int(l_approx(x))
        alpha = []
        for i in range(max(2,l-2), l+2):
            alpha.append((lg(i-1)(x)-lg(i+1)(x))/(2*l+1))
        alpha = min(alpha)
        #alpha = -np.cos(theta_0)
        #print('k: {}'.format(self.k))
        out = (-self.k + (1+self.Delta_1-self.Delta_2) * 2 * np.pi *alpha*self.N/(4*np.pi))/self.k
        return out

    def special_rewiring_lam(self, r_0, m):
        k = 2*r_0
        lam = []
        for l in range(-499, 500):
            if not l==0:
                lam.append(-k -1 + np.sin((k+1) *l* np.pi/self.N) / np.sin(np.pi * l/self.N) + 2*(np.cos(2*np.pi*l*(self.N/2-1)/self.N)
                  - np.cos(2*np.pi*l/self.N))*m/self.N)
        output = max(lam)
        return output/k

    def special_rewiring_lam_gaussian(self, r_0, q):
        self.q = q
        self.k=2*r_0
        lam = []
        for l in range(-499, 500):
            if not l == 0:
                p=l/self.N
                alpha = np.sin((2 * self.r_0 + 1) * np.pi * p) / np.sin(np.pi * p) - 1
                extra=self.q*self.k*np.exp(-2*np.pi**2*p**2*30**2)*np.cos(2*np.pi*p*300)
                lam.append(-self.k + (1 + self.Delta_1 - self.Delta_2) * alpha - self.Delta_2-extra)
        output=max(lam)
        return output/self.k


if __name__=='__main__':
    p=1e-9
    r=2
    p = np.round(1.5*20/ (2 * r + 1))/20
    x=np.sin((2 * r + 1) * np.pi * p) / np.sin(np.pi * p)
    print(x)
    # x = analytics(1,1,8000)
    # x.D_0_load('3d_20_20_20')
    # for r_0 in [2,3,5,6,8,9]:
    #     a=x.q0_eigenvalues(r=r_0)
    #     b=x.second_lam_three_dim(q=0,r=r_0)
    #     print('{} vs {}'.format(a,b))
    #    print(x.smallest_lam_sphere(q, r_0))
    #    print(x.k)
    #x = np.arange(100)/100*np.pi
    #plt.plot(x, np.pi*np.sin(x)**2 )
    #plt.show()



