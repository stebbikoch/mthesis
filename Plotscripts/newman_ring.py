from matplotlib import pyplot as plt
import json
import numpy as np
import statistics as st
from analytics import analytics


f = open('../results/newman_ring_undirected.jsonshort.json')
# returns JSON object as
# a dictionary
instance = analytics(1,1,1000)
data = json.load(f)
exponent = np.arange(16)
exponent1 = np.arange(51)
q_values = 10 ** (-exponent / 3)
q_values1 = 10 ** (-exponent1 / 10)
r_0_values = [10,25,100,400]
fig, (ax, ax1) = plt.subplots(ncols=2, figsize=(6.4, 3))
for r_0 in r_0_values:
    line1, =ax.plot(q_values1, [instance.lam_one_dim_additive(r_0, q) for q in q_values1], label=r'$k_0=$'+str(2*r_0),
                    linewidth=1)#, color=c[index])
    #ax.plot(q_values1, 1/q_values1)
    line2 =ax.errorbar(q_values, [data[str(r_0)]['qs'][str(q)]['second_largest'][0] for q in q_values],
                       yerr=[data[str(r_0)]['qs'][str(q)]['second_largest'][1] for q in q_values],
                     fmt='x', markerfacecolor='none', capsize=2, color=line1.get_color())
    line4, = ax1.plot(q_values1, [instance.lam_one_dim_additive(r_0, q, smallest=True) for q in q_values1], label=r'k=' + str(2 * r_0), linewidth=1)
    line3 = ax1.errorbar(q_values, [data[str(r_0)]['qs'][str(q)]['smallest'][0] for q in q_values],
                        yerr=[data[str(r_0)]['qs'][str(q)]['smallest'][1] for q in q_values],
                        fmt='x', markerfacecolor='none', capsize=2, color=line1.get_color())


ax.set_yscale('symlog', linthresh=0.0001)
ax.set_ylim(bottom=-1.5)
#ax1.set_yscale('symlog', linthresh=0.0001)
for a, label in zip([ax, ax1],['a', 'b']):
    a.set_xscale('log')
    a.set_xlabel(r'topological randomness $q$')
    a.set_title(label, loc='left', fontweight='bold', fontsize=10)
#plt.xlim(0.47,1.03)
#plt.yticks([-1, -0.9, -0.8, -0.7, -0.6],[-1.0, -0.9, -0.8, -0.7, -0.6])
ax1.set_ylim(top=-0.75, bottom=-1.6)
ax.set_ylabel(r'$\lambda_2$')
ax1.set_ylabel(r'$\lambda_-$')
plt.tight_layout()
legend1=ax1.legend([line1, line2], [r"anal. prediction", r"num. result undirected"], loc='upper left')
ax1.add_artist(legend1)
ax.legend(bbox_to_anchor=(0, 1.15), loc='lower left', ncol=4)
plt.savefig('../figures/newman_ring.svg', format='svg', dpi=1000, bbox_inches='tight')
plt.show()