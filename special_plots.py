from matplotlib import pyplot as plt
import numpy as np

def lam(m, l=2,k=20,n=1000):
    output = -k -1 + np.sin((k+1)/n*l*np.pi)/np.sin(l*np.pi/n)+m/n*((-1)**l-np.cos(2*np.pi*l/n))
    return output

if __name__=="__main__":
    ms=np.arange(1,500)
    plt.plot(ms, lam(ms))
    plt.show()
    # ls = np.arange(1,10)
    # print(ls)
    # print(lam(2))
    # plt.plot(ls,lam(ls))
    # plt.show()