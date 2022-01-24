from matplotlib import pyplot as plt
import numpy as np


def f(p, k):
    output = np.sin((k+1)*p*np.pi)/np.sin(p*np.pi)-1
    return output/k

def g(p, k):
    output = np.sin(np.pi * k * p)/(np.pi*p)
    return output/k

fig =  plt.figure(figsize=(6.4, 2.4))
n = 5000
p_array = np.arange(n)/(2*n)+1e-5
for k in [10, 50, 200]:
    line, = plt.plot(p_array, f(p_array, k), linewidth=1, label=r'$k=$'+str(k))
    line2, =plt.plot(p_array, g(p_array, k), linestyle='--', color=line.get_color())
plt.xlabel(r'Reciprocal index $p$')
plt.ylabel(r'$y$')
legend1 = plt.legend([line, line2], [r"Discrete sum $f_k(p)$", r"Integral $g_k(p)$"], loc='upper center')
plt.legend(loc='upper right')#loc=(0.8, 0.8))
plt.gca().add_artist(legend1)
plt.tight_layout()
fig.savefig('figures/continuous_alpha_1d.svg', format='svg', dpi=1000)
plt.show()
