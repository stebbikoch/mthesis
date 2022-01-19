from matplotlib import pyplot as plt
import numpy as np

def n(k, N_tot, q):
    """
    number of edges with distance d after rewiring
    :param d:
    :return:
    """
    edges = int(N_tot * k / 2)
    M = np.random.binomial(edges, q)
    d_out = np.zeros(int(N_tot/2))
    d_out[:int(k/2)] = N_tot
    for i in range(M):
        # delete edges
        index = np.random.randint(k/2)
        d_out[index]+=-1
        # add edges
        repeat = True
        while repeat:
            index = int(round(np.random.normal(loc=300, scale=10)))
            if index <= N_tot/2:
                d_out[index]+=1
                repeat=False
    return d_out

def gaussian(x, mu, sig):
    return 1/(sig*np.sqrt(2*np.pi))*np.exp(-(x - mu)**2 / (2 * sig**2))

def n_ave(k, N_tot, q, x):
    out=np.zeros(len(x))
    for i in range(len(x)):
        if x[i]<=int(k/2):
            out[i]=N_tot-q*N_tot
        else:
            out[i] = q*N_tot*k/2*gaussian(x[i], 300, 10)
    return out

if __name__=="__main__":
    print(gaussian(5, 300, 10))
    fig = plt.figure(figsize=(6.4, 0.5*4.8))
    data=n(100, 1000, 0.2)
    print(len(data))
    plt.bar(np.arange(1,501),data, width=1, alpha=0.7, label='one instance after rewiring')
    print(len)
    x = np.arange(1,1000)/1000*501
    plt.plot(x, n_ave(100, 1000, 0.2, x), color='orange', linestyle='--', label=r"approximate averaged probability density"
                            + "\n" + "after rewiring")
    y = np.append(1000*np.ones(int(100/2)), np.zeros(int(501-100/2)))
    plt.plot(np.arange(501),y, color='purple', linestyle='--', label='edge histogram before rewiring')
    plt.xlabel('edge-length')
    plt.ylabel('# of edges')
    plt.legend()
    plt.tight_layout()
    #plt.show()
    plt.savefig('figures/1000ring_gauss_histogram.svg', format='svg', dpi=1000)
