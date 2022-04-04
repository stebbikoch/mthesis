import numpy as np
from scipy.special import legendre as lg
from scipy.special import jv
from matplotlib import pyplot as plt
from legendrepolynomials import l_approx
import json
from matrix import build_matrix

class analytics:
    def __init__(self,k, q, N):
        self.k = k
        self.q = q
        self.N = N

    def laplacian_approx(self, k, N, largest=True):
        p = k/(N-1)
        q = 1-p
        def approx2(p, q, N):
            n = N
            output = p * n + np.sqrt(2 * p * q * n * np.log(n)) - np.sqrt(p * q * n / 8 / np.log(n) * np.log(np.log(n)))
            return output
        if largest:
            return - approx2(p, q, N)/k
        else:
            p = q
            q = 1 - p
            return -(N - approx2(p,q, N))/k

    def D_0_load(self, d_function):
        try:
            f = open('./d_functions/' + d_function + '.json', )
        except:
            f = open('../d_functions/' + d_function + '.json', )
        # returns JSON object as
        # a dictionary
        data = json.load(f)
        self.numbers = data[0]
        self.all_indices_list = data[1]
        self.d_0s = np.array(data[2])

    def k_out(self, d_function, r):
        self.D_0_load(d_function)
        index = np.where(self.d_0s == r)[0][0]
        d_vector = np.array(self.all_indices_list[index])
        return len(d_vector)

    def exact_alpha(self, d_function, p, r, smallest=False, multi_p=False, threed=False):
        self.D_0_load(d_function)
        index=np.where(self.d_0s==r)[0][0]
        d_vector = np.array(self.all_indices_list[index])
        self.k = len(d_vector[:,0])
        if smallest:
            alpha = []
            limit=int(5.1356 * 100 / (2 * np.pi * r)+1.5)
            if threed:
                limit = int(5.763/(2*np.pi*r)*20+1.5)
            for i in range(limit):
                for j in range(limit):
                    if threed:
                        for k in range(limit):
                            p = np.array([i / 20, j / 20, k/20])
                            alpha.append(np.real(np.sum(np.exp(1j * np.pi * 2 * np.dot(p, d_vector.T)))))
                    else:
                        p = np.array([i / 100, j / 100, 0])
                        alpha.append(np.real(np.sum(np.exp(1j*np.pi*2*np.dot(p,d_vector.T)))))
            #print(alpha)
            return min(alpha)
        elif multi_p:
            out = np.zeros(len(p))
            for i in range(len(p)):
                out[i] = np.real(np.sum(np.exp(1j*np.pi*2*np.dot(p[i],d_vector.T))))
            return out/len(d_vector)
        else:
            return np.sum(np.exp(1j*np.pi*2*np.dot(p,d_vector.T)))

    def exact_eigens(self, q, r, hexagonal=True):
        self.q=q
        if hexagonal:
            alpha= self.exact_alpha('2d_32_32_hexagonal', 1,r, smallest=True)
        else:
            alpha = self.exact_alpha('2d_32_32_square', 1, r, smallest=True)
        print('real k', self.k)
        return (-self.k+(1+self.Delta_1-self.Delta_2)*alpha-self.Delta_2)/self.k


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

    def lam_two_dim_eucl(self, q=None, r=None, smallest=False, exact=False, real_k=False):
        self.q=q
        self.k=np.pi*r**2-1
        if real_k:
            self.exact_alpha('2d_100_100_eucl', 1/100, r)
        if smallest:
            p=1/100*round(5.1356*100/(2*np.pi*r))
        else:
            p = 1/100
        alpha = r/p*jv(1,2*np.pi*r*p)
        if exact:
            if smallest:
                alpha = self.exact_alpha('2d_100_100_eucl', p, r, smallest=True)
            else:
                alpha = self.exact_alpha('2d_100_100_eucl', np.array([p,0,0]), r)
                assert np.imag(alpha) < 1e-4, 'imaginery part should be zero but is {}'.format(np.imag(alpha))
                alpha=np.real(alpha)
        #print('compare', self.k, alpha)
        out = -self.k+(1+self.Delta_1-self.Delta_2)*alpha-self.Delta_2
        return out/self.k

    def lam_three_dim_eucl(self, q=None, r=None, smallest=False, exact=False, real_k=False):
        self.q=q
        self.k=4/3*np.pi*r**3-1
        if real_k:
            self.exact_alpha('3d_20_20_20_eucl', 1/20, r)
        p = 1/20
        alpha = (np.sin(2*np.pi*r*p)-2*np.pi*r*p*np.cos(2*np.pi*r*p))/(2*np.pi**2*p**3)
        if exact:
            if smallest:
                alpha = self.exact_alpha('3d_20_20_20_eucl', p, r, smallest=True, threed=True)
            else:
                alpha = self.exact_alpha('3d_20_20_20_eucl', np.array([p, 0, 0]), r)
                alpha = np.real(alpha)
        out = -self.k+(1+self.Delta_1-self.Delta_2)*alpha-self.Delta_2
        return out/self.k

    def second_lam_three_dim(self, q=None, r=None, smallest=False):
        self.q = q
        self.k = (2*r+1)**3-1
        n=np.cbrt(self.N)
        if smallest:
            p_l=np.round(1.5*n/ (2 * r + 1))/n
            p = p_l#np.array([1e-9, 1e-9, p_l])
        else:
            p = 1/n#np.array([1/n, 1e-9, 1e-9])
        alpha = (2*r+1)**2*np.sin((2*r+1)*np.pi*p)/np.sin(np.pi*p)-1
        return (-self.k + (1+self.Delta_1-self.Delta_2)*alpha -self.Delta_2)/self.k

    def second_lam_arb_dim(self, d, q, dim=3):
        return -1 + (1 - q + (q-1)*q*d/(1-d*(1-q))) * (np.sin(np.pi*d**(1/dim))/(np.pi*d**(1/dim)))

    def second_lam_two_dim(self, q=None, r=None, n=None, smallest=False):
        self.q = q
        self.k = (2*r+1)**2-1
        if n:
            self.n=n
        else:
            self.n=np.sqrt(self.N)
        if smallest:
            p_l=np.round(1.5*self.n/ (2 * r + 1))/self.n
            p = p_l
        else:
            p = 1 / self.n
        if smallest:
            alpha = min([(2*r+1)*(np.sin((2 * r + 1) * np.pi * i) / np.sin(np.pi * i)) - 1 for i in [p-2/n, p-1/n, p, p+1/n, p+2/n]])
        else:
            alpha = (2 * r + 1) * (np.sin((2 * r + 1) * np.pi * p) / np.sin(np.pi * p)) - 1
        return (-self.k + (1 + self.Delta_1 - self.Delta_2) * alpha - self.Delta_2)/self.k

    def second_lam_one_dim(self, q=None, k=None, smallest=True):
        if q:
            self.q = q
        if k:
            self.k=k
        p = 1/self.N
        if smallest:
            p = int(round(1.5*self.N/2*(k+1)+1)/self.N)
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

    def smallest_lam_sphere(self, q, r_0, limes=False):
        self.q = q
        theta_0 = 2*np.arcsin(r_0/2)
        x = np.cos(theta_0)
        self.k = self.N * (1-np.cos(theta_0))/2
        l = int(l_approx(x))
        alpha = []
        for i in range(max(2,l-2), l+2):
            alpha.append((lg(i-1)(x)-lg(i+1)(x))/(2*l+1))
        alpha = min(alpha)
        if limes:
            alpha = 1/8*(np.cos(theta_0)-1)
        #alpha = -np.cos(theta_0)
        #print('k: {}'.format(self.k))
        out = (-self.k + (1+self.Delta_1-self.Delta_2) * 2 * np.pi *alpha*self.N/(4*np.pi))/self.k
        return out

    def special_rewiring_lam(self, r_0, m, smallest=False):
        k = 2*r_0
        lam = []
        range1=range(1, 10)
        if smallest:
            l_approx=int(round(1.5*self.N/(k+1)+1))
            range1=range(min(l_approx-10,1),l_approx+10)
        for l in range1:
            lam.append(-k -1 + np.sin((k+1) *l* np.pi/self.N) / np.sin(np.pi * l/self.N) + 2*(np.cos(2*np.pi*l*(self.N/2-1)/self.N)
              - np.cos(2*np.pi*l/self.N))*m/self.N)
        output = max(lam)
        if smallest:
            output = min(lam)
        return output/k

    def special_rewiring_lam_gaussian(self, r_0, q, mu=300, sigma=30, smallest=False):
        self.q = q
        self.k=2*r_0
        lam = []
        range1 = range(1, 10)
        if smallest:
            l_approx = int(round(1.5 * self.N / (self.k + 1) + 1))
            range1 = range(1, l_approx + 20)
        for l in range1:
            if not l == 0:
                p=l/self.N
                alpha = np.sin((2 * self.r_0 + 1) * np.pi * p) / np.sin(np.pi * p) - 1
                extra=self.q*self.k*np.exp(-2*np.pi**2*p**2*sigma**2)*np.cos(2*np.pi*p*mu)
                lam.append(-self.k + (1 - q) * alpha+extra)
        output=max(lam)
        #index = np.where(np.array(lam)==max(lam))[0]
        if smallest:
            output=min(lam)
            #index = np.where(np.array(lam)==min(lam))[0]
        #print('l={}'.format(index - 499))
        return output/self.k


if __name__=='__main__':
    x=analytics(1,1,10000)
    a=x.exact_alpha('2d_100_100_eucl', 1, 15, smallest=True)
    print(a)



