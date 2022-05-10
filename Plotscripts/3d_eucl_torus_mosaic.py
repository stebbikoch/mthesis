from matplotlib import pyplot as plt
import json
import numpy as np
import statistics as st
from analytics import analytics
import matplotlib as mpl
mpl.rcParams['lines.markersize'] = 4.5


f1 = open('/run/user/1000/gvfs/sftp:host=taurus.hrsk.tu-dresden.de,user=s8583916/home/h3/s8583916/home/scratch/'+
          's8583916-stefans_ws/mthesis/results/3deuclshort.json')
f2 = open('/run/user/1000/gvfs/sftp:host=taurus.hrsk.tu-dresden.de,user=s8583916/home/h3/s8583916/home/scratch/'+
          's8583916-stefans_ws/mthesis/results/3d_eucldirshort.json')
f3 = open('/run/user/1000/gvfs/sftp:host=taurus.hrsk.tu-dresden.de,user=s8583916/home/h3/s8583916/home/scratch/'+
          's8583916-stefans_ws/mthesis/results/N_scaling3deuclundirshort.json')
f4 = open('/run/user/1000/gvfs/sftp:host=taurus.hrsk.tu-dresden.de,user=s8583916/home/h3/s8583916/home/scratch/'+
          's8583916-stefans_ws/mthesis/results/N_scaling3deucldirshort.json')
cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
#f4 = open('/run/user/1000/gvfs/sftp:host=taurus.hrsk.tu-dresden.de,user=s8583916/home/h3/s8583916/home/scratch/'+
 #         's8583916-stefans_ws/mthesis/results/N_scaling3dmaxdirshort.json')
# returns JSON object as
# a dictionary
instance1 = analytics(0,0,4096)
data1 = json.load(f1)
data2 = json.load(f2)
data3 = json.load(f3)
data4 = json.load(f4)
exponent = np.arange(16)
q_values_1 = 10 ** (-exponent / 3)
q_values_2 = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
q_values_3 = [0.6, 0.7, 0.8, 0.9, 1.0]
exponent = np.arange(201)
q_values = 10**(-exponent/40)
r_values = r_0_values = [1.9, 4.3, 5.5, 7.9]
fig, ((ax, ax1), (ax2, ax3), (ax4, ax5)) = plt.subplots(nrows=3, ncols=2, figsize=(6.4, 7.2))
#ax1.hlines(-1-1/8, 0, 1, color='k', linewidth=1)
for r_0, k_about in zip(r_values,['26', '340', '738', '2102']):
    k=data1[str(r_0)]['k']
    # second largest
    line0, = ax.plot(q_values, [instance1.lam_three_dim_eucl(q=q, r=r_0) for q in q_values],
                      label=r'$k = {}$'.format(k_about), linewidth=1)
    # undirected
    line2 = ax.errorbar(q_values_1, [data1[str(r_0)]['qs'][str(q)]['second_largest'][0] for q in q_values_1],
                         yerr=[data1[str(r_0)]['qs'][str(q)]['second_largest'][1] for q in q_values_1], fmt='x',
                         color=line0.get_color(),
                         markerfacecolor='none', capsize=2)
    # directed
    line3 = ax.errorbar(q_values_1, [data2[str(r_0)]['qs'][str(q)]['second_largest'][0] for q in q_values_1],
                yerr=[data2[str(r_0)]['qs'][str(q)]['second_largest'][1] for q in q_values_1], fmt='o',
                color=line0.get_color(),
                markerfacecolor='none', capsize=2)
    # smallest
    ax1.plot(q_values, instance1.lam_three_dim_eucl(q=q_values, r=r_0, real_k=True, exact=True, smallest=True),
             color=line0.get_color(), linewidth=1)
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
    yd = instance1.laplacian_approx(k, 4096)
    yu = instance1.laplacian_approx(k, 4096, largest=False)
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
    yd =  np.sqrt(1 / k - 1 / 4096) - 1
    yu =  - np.sqrt(1 / k - 1 / 4096) - 1
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
    yd = 2 * np.sqrt(1 / k - 1 / 4096)
    yu = - 2 * np.sqrt(1 / k - 1 / 4096)
    line4 = ax4.hlines(yd, 0.1, 1.1, color=line0.get_color(), linewidth=1, label=r'$k \approx {}$'.format(k_about))
    line5 = ax4.hlines(yu, 0.1, 1.1, color=line0.get_color(), linewidth=1)
    ax4.set_xlim(0.15, 1.05)
N_values = [64, 216, 512, 1331, 2197, 3375, 4096]
for q, color in zip([0.001, 0.01, 0.1, 0.3], cycle[4:8]):
    # scaling with network-size N
    instance2 = analytics(1, 1, 1234)
    y1 = instance2.second_lam_arb_dim(0.1, q, eucl=True)
    #y = instance2.second_lam_three_dim(q=q, r=3)
    line5=ax5.hlines(y1, 0, 4200, color=color)
    ax5.set_xlim(0, 4200)
    #line5=ax5.hlines(y, 0, 3500, color=color)
    ax5.errorbar(N_values, [data3[str(N)]['qs'][str(q)]['second_largest'][0] for N in N_values],
                     yerr=[data3[str(N)]['qs'][str(q)]['second_largest'][1] for N in N_values], fmt='x',
                     markerfacecolor='none', capsize=2, label=r'$q = {}$'.format(q), color=line5.get_color())
    ax5.errorbar(N_values, [data4[str(N)]['qs'][str(q)]['second_largest'][0] for N in N_values],
                 yerr=[data4[str(N)]['qs'][str(q)]['second_largest'][1] for N in N_values], fmt='o',
                 color=line5.get_color(),
                 markerfacecolor='none', capsize=2)
#plt.yscale('symlog', linthreshy=0.0001)
#ax.set_ylim(-1, -0.001)
ax.set_yscale('symlog', linthresh=0.0001)
ax.set_xscale('log')
ax1.set_xscale('log')
ax1.set_ylim(ymin=-1.6)
#ax1.set_yscale('symlog')
#ax1.set_yticks([-1, -1.2, -2])
titles = ['a', 'b', 'c', 'd', 'e', 'f']
titles2 = [r'$L_\mathrm{dir/undir},~ \lambda_2$', r'$L_\mathrm{dir/undir},~ \lambda_-$',
           r'$L_\mathrm{undir},~ \lambda_2 ~\mathrm{and}~ \lambda_-,~ p \sim 1$',
           r'$L_\mathrm{dir},~ \lambda_2 ~\mathrm{and}~ \lambda_-,~ p \sim 1$',
           r'$A_\mathrm{undir},~ \lambda_2 ~\mathrm{ and }~ \lambda_-,~ p \sim 1$',
           r'$L_\mathrm{dir/undir},~ \lambda_2,~ k/N\approx\mathrm{const.}$']
ylabels = [r'Re $\lambda_2$', r'Re $\lambda_-$', r'$\lambda_2$, $\lambda_-$',
           r'Re $\lambda_2$, Re $\lambda_-$', r'$\lambda_2$, $\lambda_-$', r'Re $\lambda_2$']
xlabels = ['topological randomness $q$']*6
for axe, label, title, ylabel, xlabel in zip((ax, ax1, ax2, ax3, ax4, ax5), titles, titles2, ylabels, xlabels):
    axe.set_title(label, loc='left', fontweight='bold', fontsize=10)
    axe.set_title(title, loc='center', fontweight='bold', fontsize=10)
    axe.set_ylabel(ylabel)
    axe.set_xlabel(xlabel)
#plt.xlim(0.4,1.1)
#plt.ylim(-1, -0.5)
ax5.set_xlabel(r'network-size $N$')
#ax4.legend([line0, line2, line3], ['anal. prediction', 'num. result undirected', 'num. result directed'], bbox_to_anchor=(0, -0.9), loc='lower left', ncol=1)
plt.tight_layout()
#plt.figure(figsize=(6.4, 8))
#ax.add_artist(ax.legend([line0], ['proximate \n'
 #                                 'mean-field'], loc='upper left', bbox_to_anchor=(0, 0.9)))
ax.legend(bbox_to_anchor=(0, 1.15), loc='lower left', ncol=4)
ax4.legend([line0, line2, line3], ['anal. prediction','num. result undirected', 'num. result directed'], bbox_to_anchor=(0, -0.85), loc='lower left', ncol=1)
ax5.legend(loc='lower left', bbox_to_anchor=(0, -0.72), ncol=2, columnspacing=1)
#legend1 = plt.legend([line0, line2, line3, line4, line5], [r"Analytical prediction", r"Numerical results undirected",
 #                                                          r"Numerical results directed", r"Wigner semi-circle directed"
  #                                                         , "Wigner semi-circle undirected"], loc=3)
#plt.legend(loc=7)
#plt.gca().add_artist(legend1)
#plt.grid(zorder=1)
#plt.legend()
#axs[1].set_yscale('symlog')
#axs[1].set_xscale('log')
plt.savefig('../figures/3d_torus_mosaic_eucl.svg', format='svg', dpi=1000, bbox_inches='tight')
plt.show()