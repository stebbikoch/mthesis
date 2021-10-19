# plots
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import SymLogNorm
class Torus:
    def __init__(self, N_1, N_2, q):
        self.N_1 = N_1
        self.N_2 = N_2
        self.x = np.arange(N_1)/N_1
        self.y = np.arange(N_1)/N_2
        self.A = q/((N_1*N_2-1)/6-1)

    def f_lambda(self, x, y):
        output = -6 + 2 * np.cos(2*np.pi*x) + 2 * np.cos(2*np.pi*y) + 2 * np.cos(2*np.pi*(x+y))
        return output

    def delta_lambda(self, x, y):
        output = np.sin(np.pi*((self.N_1+1)/2*x)) * np.sin(np.pi*((self.N_2+1)/2*y)) \
        * np.sin(np.pi*((self.N_1-1)/2*x+(self.N_2-1)/2*y)) \
        /np.cos(np.pi*x) / np.cos(np.pi*y)*self.A
        return output

    def total(self, x, y):
        output = self.f_lambda(x,y)+self.delta_lambda(x,y)
        return output

def main():
    x = Torus(49,49,0.9)
    eigenvalues = np.array([[x.total(a,b) for a in x.x] for b in x.y])
    eigenvalues2 = np.array([[x.f_lambda(a, b) for a in x.x] for b in x.y])
    print(np.max(eigenvalues))
    fig = plt.figure()
    ax1 = fig.add_subplot(1,3,1)
    im = ax1.imshow(eigenvalues, vmax=0, vmin=-10, cmap='jet')#, norm=SymLogNorm(5))#, interpolation='bilinear')
    ax2 = fig.add_subplot(1,3,2)
    ax2.imshow(eigenvalues2, vmax=0, vmin=-10,cmap='jet')#, norm=SymLogNorm(5))
    ax3 = fig.add_subplot(1,3,3)
    ax3.imshow(-eigenvalues2+eigenvalues, vmax=0, vmin=-10,cmap='jet')
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.3, 0.01, 0.4])
    fig.colorbar(im, cax=cbar_ax)

    fig.show()
    #plt.colorbar()

if __name__ == "__main__":
    main()
