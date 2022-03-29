from matplotlib import pyplot as plt
import json
import numpy as np
import statistics as st
from analytics import analytics

f1 = open('/run/user/1000/gvfs/sftp:host=taurus.hrsk.tu-dresden.de,user=s8583916/home/h3/s8583916/home/scratch/'+
          's8583916-stefans_ws/mthesis/results/fibbonaccisphereshort.json')
f2 = open('/run/user/1000/gvfs/sftp:host=taurus.hrsk.tu-dresden.de,user=s8583916/home/h3/s8583916/home/scratch/'+
          's8583916-stefans_ws/mthesis/results/fibbonaccisphere_dirshort.json')
# returns JSON object as
# a dictionary
instance1 = analytics(1,1,1000)
data1 = json.load(f1)
data2 = json.load(f2)
exponent = np.arange(16)
q_values_1 = 10 ** (-exponent / 3)
q_values_2 = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
q_values_3 = [0.6, 0.7, 0.8, 0.9, 1.0]
exponent = np.arange(201)
q_values = 10**(-exponent/40)
r_values = r_0_values = [0.283, 0.447, 0.894, 1.789]#[0.283, 0.447, 0.632, 0.894, 1.265, 1.789]
fig, ((ax, ax1), (ax2, ax3), (ax4, ax5)) = plt.subplots(nrows=3, ncols=2, figsize=(6.4, 7.2))
for r_0 in r_values:
    k=data1[str(r_0)]['k']
    # second largest
    line0, = ax.plot(q_values, [instance1.second_lam_sphere(q=q, r_0=r_0) for q in q_values],
                      label=r'$r_0={}~~(\^k={:.2f})$'.format(r_0, k), zorder=2, linewidth=1)
    # undirected
    line2 = ax.errorbar(q_values_1, [data1[str(r_0)]['qs'][str(q)]['second_largest'][0] for q in q_values_1],
                         yerr=[data1[str(r_0)]['qs'][str(q)]['second_largest'][1] for q in q_values_1], fmt='x',
                         color=line0.get_color(),
                         markerfacecolor='none', capsize=2)
    # directed
    ax.errorbar(q_values_1, [data2[str(r_0)]['qs'][str(q)]['second_largest'][0] for q in q_values_1],
                yerr=[data2[str(r_0)]['qs'][str(q)]['second_largest'][1] for q in q_values_1], fmt='o',
                color=line0.get_color(),
                markerfacecolor='none', capsize=2)
    # smallest
    #ax1.plot(q_values, [instance1.smallest_lam_sphere(q=q, r_0=r_0) for q in q_values],
     #                label=r'$r_0={}~~(\^k={:.2f})$'.format(r_0, k), zorder=2, linewidth=1)
    if r_0 in [0.283, 0.447, 0.894, 1.789]:
        # undirected
        ax1.errorbar(q_values_1, [data1[str(r_0)]['qs'][str(q)]['smallest'][0] for q in q_values_1],
                            yerr=[data1[str(r_0)]['qs'][str(q)]['smallest'][1] for q in q_values_1], fmt='x',
                            color=line0.get_color(),
                            markerfacecolor='none', capsize=2)
        # directed
        ax1.errorbar(q_values_1, [data2[str(r_0)]['qs'][str(q)]['smallest'][0] for q in q_values_1],
                     yerr=[data2[str(r_0)]['qs'][str(q)]['smallest'][1] for q in q_values_1], fmt='o',
                     color=line0.get_color(),
                     markerfacecolor='none', capsize=2)
    # laplacian p->1
    ax2.errorbar(q_values_2, [data1[str(r_0)]['qs'][str(q)]['second_largest'][0] for q in q_values_2],
                         yerr=[data1[str(r_0)]['qs'][str(q)]['second_largest'][1] for q in q_values_2], fmt='x',
                         color=line0.get_color(),
                      markerfacecolor='none', capsize=2)
    ax2.errorbar(q_values_2, [data1[str(r_0)]['qs'][str(q)]['smallest'][0] for q in q_values_2],
                 yerr=[data1[str(r_0)]['qs'][str(q)]['smallest'][1] for q in q_values_2], fmt='x',
                 color=line0.get_color(),
                 markerfacecolor='none', capsize=2)
    # undirected laplacian approx
    yd = instance1.laplacian_approx(k, 1000)
    yu = instance1.laplacian_approx(k, 1000, largest=False)
    line4=ax2.hlines(yd, 0.1, 1.1, color=line0.get_color(), linewidth=1)
    line5=ax2.hlines(yu, 0.1, 1.1, color=line0.get_color(), linewidth=1)
    ax2.set_xlim(0.15, 1.05)
    # directed laplacian p->1
    ax3.errorbar(q_values_3, [data2[str(r_0)]['qs'][str(q)]['second_largest'][0] for q in q_values_3],
                 yerr=[data1[str(r_0)]['qs'][str(q)]['second_largest'][1] for q in q_values_3], fmt='o',
                 color=line0.get_color(),
                 markerfacecolor='none', capsize=2)
    ax3.errorbar(q_values_3, [data2[str(r_0)]['qs'][str(q)]['smallest'][0] for q in q_values_3],
                 yerr=[data1[str(r_0)]['qs'][str(q)]['smallest'][1] for q in q_values_3], fmt='o',
                 color=line0.get_color(),
                 markerfacecolor='none', capsize=2)
    # undirected laplacian approx
    yd =  np.sqrt(1 / k - 1 / 1000) - 1
    yu =  - np.sqrt(1 / k - 1 / 1000) - 1
    ax3.hlines(yd, 0.5, 1.1, color=line0.get_color(), linewidth=1)
    ax3.hlines(yu, 0.5, 1.1, color=line0.get_color(), linewidth=1)
    ax3.set_xlim(0.55, 1.05)
    # undirected adjacency
    ax4.errorbar(q_values_2, [data1[str(r_0)]['qs'][str(q)]['second_largest_adj'][0] for q in q_values_2],
                 yerr=[data1[str(r_0)]['qs'][str(q)]['second_largest_adj'][1] for q in q_values_2], fmt='x',
                 color=line0.get_color(),
                 markerfacecolor='none', capsize=2)
    ax4.errorbar(q_values_2, [data1[str(r_0)]['qs'][str(q)]['smallest_adj'][0] for q in q_values_2],
                 yerr=[data1[str(r_0)]['qs'][str(q)]['smallest_adj'][1] for q in q_values_2], fmt='x',
                 color=line0.get_color(),
                 markerfacecolor='none', capsize=2)
    yd = 2 * np.sqrt(1 / k - 1 / 1000)
    yu = - 2 * np.sqrt(1 / k - 1 / 1000)
    line4 = ax4.hlines(yd, 0.1, 1.1, color=line0.get_color(), linewidth=1)
    line5 = ax4.hlines(yu, 0.1, 1.1, color=line0.get_color(), linewidth=1)
    ax4.set_xlim(0.15, 1.05)
#plt.yscale('symlog', linthreshy=0.0001)
#ax.set_ylim(-1, -0.001)
ax.set_yscale('symlog', linthresh=0.0001)
ax.set_xscale('log')
ax1.set_xscale('log')
#ax1.set_yscale('symlog')
#ax1.set_yticks([-1, -1.2, -2])
titles = ['a', 'b', 'c', 'd', 'e']
ylabels = [r'Re $\lambda_2(L)$', r'Re $\lambda_-(L)$', r'$\lambda_2(L)$, $\lambda_-(L)$',
           r'$\lambda_2(L)$, $\lambda_-(L)$', r'$\lambda_2(A)$, $\lambda_-(A)$']
xlabels = ['$q$']*5
for ax, label, ylabel, xlabel in zip((ax, ax1, ax2, ax3, ax4), titles, ylabels, xlabels):
    ax.grid()
    ax.set_title(label, loc='left', fontweight='bold', fontsize=12)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
#plt.xlim(0.4,1.1)
#plt.ylim(-1, -0.5)
ax5.set_xlabel(r'topological randomness $q$')
#legend1 = plt.legend([line0, line2, line3, line4, line5], [r"Analytical prediction", r"Numerical results undirected",
 #                                                          r"Numerical results directed", r"Wigner semi-circle directed"
  #                                                         , "Wigner semi-circle undirected"], loc=3)
#plt.legend(loc=7)
#plt.gca().add_artist(legend1)
#plt.grid(zorder=1)
#plt.legend()
plt.tight_layout()
#axs[1].set_yscale('symlog')
#axs[1].set_xscale('log')
#plt.savefig('figures/sphere_largest_1000_eq_unaveraged.svg', format='svg', dpi=1000)
plt.show()