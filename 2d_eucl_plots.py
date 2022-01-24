from matplotlib import pyplot as plt
import json
import numpy as np
import statistics as st
from analytics import analytics

#f1 = open('./reproduce/reproduce_1001_undirected_analytics.txt', )
f3 = open('/run/user/1000/gvfs/sftp:host=taurus.hrsk.tu-dresden.de,user=s8583916/home/h3/s8583916/home/scratch/s8583916-stefans_ws/mthesis/results/2d_undirected_10000_eucl_100xshort.json')
#f2 = open('/run/user/1000/gvfs/sftp:host=taurus.hrsr.tu-dresden.de,user=s8583916/home/h3/s8583916/home/scratch/'+
 #         's8583916-stefans_ws/mthesis/results/1000ring_sl_directed_Q_100xshort.json')
#f2 = open('./reproduce/reproduce_1000_directed_with_averaging_test_big_q.json', )
#f3 = open('./reproduce/reproduce_1000_undirected_with_averaging_test_big_q.json', )
# returns JSON object as
# a dictionary
instance = analytics(1,1,10000)
#data2 = json.load(f2)
data3=json.load(f3)
fig = plt.figure()
exponent = np.arange(16)
q_values_1 = 10 ** (-exponent / 3)
#q_values_1 = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
#q_values_1 = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
exponent = np.arange(51)
q_values = 10**(-exponent/10)
r_values = r_0 = [3, 10, 15, 25, 35, 45]
c=['b', 'g', 'r', 'c', 'm', 'y', 'r', 'w']
#index=0
#r = 50
#q=1.0
#print(data2[str(r)][str(q)])
instance.D_0_load('2d_100_100_eucl')
for r in r_values:
    line1, =plt.plot(q_values, instance.lam_two_dim_eucl(q=q_values, r=r, real_k=True, exact=True),
                     label=r'$r_0={}~(k\approx{})$'.format(r, round(np.pi*r**2-1)))#, color=c[index])
    line6, = plt.plot(q_values, [instance.lam_two_dim_eucl(q=q, r=r, real_k=True) for q in q_values], color=line1.get_color(), linestyle='--')
    #line2, = plt.plot(q_values, [instance.second_lam_two_dim(q=q, r=r, smallest=True) for q in q_values], label=r'$r_0={}~(k={})$'.format(r, (2*r+1)**2-1))#, color=c[index])
    #line2, = plt.plot(q_values, [instance.second_lam_two_dim(q=q, r=r, smallest=True) for q in q_values])#, color=line1.get_color()))#, color=line1.get_color())
    #line4 =plt.errorbar(q_values_1, [data3[str(r)]['qs'][str(q)]['smallest'][0] for q in q_values_1],
     #                   yerr=[data3[str(r)]['qs'][str(q)]['smallest'][1] for q in q_values_1],  fmt='o',color=line1.get_color(), markerfacecolor='none', capsize=5)#, color=c[index])
    line4 = plt.errorbar(q_values_1, [data3[str(r)]['qs'][str(q)]['second_largest'][0] for q in q_values_1],
                         yerr=[data3[str(r)]['qs'][str(q)]['second_largest'][1] for q in q_values_1], fmt='o', color=line1.get_color(),
                      markerfacecolor='none', capsize=5)
    k = np.pi*r**2-1
    k = data3[str(r)]['k']
    yu = 2 * np.sqrt(1 / k - 1 / 10000) - 1
    yu2 = -2 * np.sqrt(1 / k - 1 / 10000) - 1
    line5 = plt.hlines(yu, 0.5, 1.5, linewidth=0.4, color=line1.get_color(), linestyles='--')
    #line5 = plt.hlines(yu2, 0.5, 1.5, linewidth=0.4, color=line1.get_color(), linestyles='--')
plt.yscale('symlog', linthresh=0.0001)
#plt.set_yticks([-0.99, -0.1, 0.11])
plt.xscale('log')
#plt.xlim(0.47,1.03)
plt.ylim(top=-0.001)
#plt.ylim(-1.05, -0.35)
#plt.yticrs([-1, -0.9, -0.8, -0.7, -0.6],[-1.0, -0.9, -0.8, -0.7, -0.6])
plt.xlabel(r'Small-world parameter $q$')
plt.ylabel(r'Value of normalized second largest eigenvalue')
legend1 = plt.legend([line1, line6, line4, line5], [r"Analytical prediction", r"Proximate analytical prediction", r'Numerical results undirected',
                                             r'Wigner semi-circle prediction undirected', r'Wigner semi-circle prediction directed'],
                     loc='upper left')
plt.legend(loc='upper right')
plt.gca().add_artist(legend1)
plt.grid()
plt.tight_layout()
#axs[1].set_yscale('symlog')
#axs[1].set_xscale('log')
plt.savefig('figures/10000_2d_eucl_largest.svg', format='svg', dpi=1000)
plt.show()