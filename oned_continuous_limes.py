from matplotlib import pyplot as plt
import numpy as np


def f(p, k):
    output = np.sin((k+1)*p*np.pi)/np.sin(p*np.pi)-1
    return output/k

def g(p, k):
    output = np.sin(np.pi * k * p)/(np.pi*p)
    return output/k

fig =  plt.figure(figsize=(6,4))
n = 5000
p_array = np.arange(n)/(2*n)+1e-5
for k in [10, 20, 50, 100, 200, 400]:
    line, = plt.plot(p_array, f(p_array, k), linewidth=1, label=r'$k=$'+str(k))
    line2, =plt.plot(p_array, g(p_array, k), linestyle='--', color=line.get_color())
plt.xlabel(r'Reciprocal index $p$')
plt.ylabel(r'Value of sum, respectively integral')
legend1 = plt.legend([line, line2], [r"Discrete sum $f_k(p)$", r"Integral $g_k(p)$"], loc=(0.3,0.82 ))
plt.legend()#loc=(0.8, 0.8))
plt.gca().add_artist(legend1)
plt.show()
