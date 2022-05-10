import numpy as np
from matplotlib import pyplot as plt
def F(s, d=1):
    return np.sin(np.pi*s**(1/d))/(np.pi*s**(1/d))
def f(q,s, d=1, notexact=False, N=1000):
    if notexact:
        F_val = 1-s
    else:
        F_val = F(s, d=d)
    epsilon = 3/4 * q * (1-s*(1-q))/(s*(1-s)*(1-q)) * 1/F_val
    rho = np.sqrt(3)/6
    return np.pi*epsilon**2/N * rho

fig, ((ax3, ax1), (ax2, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(6.4, 4.8), sharex=True, sharey=True)
n=100
string = 'viridis'
q_arr = np.arange(n)/(n)
s_arr = (np.arange(n)/n)
Z = np.array([[f(q, s, notexact=True, N=4000) for q in q_arr] for s in s_arr])
Z1 = np.array([[f(q, s, d=2) for q in q_arr] for s in s_arr])
Z2 = np.array([[f(q, s, notexact=True) for q in q_arr] for s in s_arr])
Z3 = np.array([[f(q, s, d=3) for q in q_arr] for s in s_arr])
cs1=ax1.contour(q_arr, s_arr, Z, levels=(0.01, 0.1, 0.5, 1,2), cmap=string)
cs2=ax2.contour(q_arr, s_arr, Z1, levels=(0.01, 0.1,  0.5, 1,2), cmap=string)
cs3=ax3.contour(q_arr, s_arr, Z2, levels=(0.01,0.1,  0.5, 1,2), cmap=string)
cs4=ax4.contour(q_arr, s_arr, Z3, levels=(0.01,0.1,  0.5, 1,2), cmap=string)
ax2.set_xlabel(r"$q$")
ax4.set_xlabel(r"$q$")
ax2.set_ylabel(r"$s$")
ax3.set_ylabel(r"$s$")
for i,j, label in zip((ax3, ax1, ax2, ax4),(cs3, cs1, cs2, cs4),("a", "b", "c", "d")):
    i.clabel(j, inline=True, manual=True)
    i.set_title(label, loc='left', fontweight='bold', fontsize=10)
plt.tight_layout()
#plt.xscale("log")
plt.xlim(left=0)
plt.savefig('../figures/contourlines.svg', format='svg', dpi=1000, bbox_inches='tight')
plt.show()