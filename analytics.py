import numpy as np

class analytics:
    def __init__(self,k, q, N):
        self.k=k
        self.q = q
        self.N = N

    @property
    def r_0(self):
        return self.k/2

    @property
    def Delta_2(self):
        return self.q * self.k / (self.N - 1 - (1 - self.q) * self.k)

    @property
    def  Delta_1(self):
        return -self.q + self.q * self.Delta_2

    def second_lam_three_dim(self):
        p = np.array([1e-9, 1e-9, 1/np.sqrt[3](self.N)])
        alpha = np.prod(np.sin((2*self.r_0+1)*np.pi*p)/np.sin(np.pi*p))-1
        return -self.k + (1+self.Delta_1-self.Delta_2)*alpha -self.Delta_2

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

