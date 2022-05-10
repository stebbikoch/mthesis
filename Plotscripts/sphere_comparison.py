from matplotlib import pyplot as plt
import json
import numpy as np
import statistics as st
from analytics import analytics
import matplotlib as mpl
mpl.rcParams['lines.markersize'] = 4.5
f1 = open('/run/user/1000/gvfs/sftp:host=taurus.hrsk.tu-dresden.de,user=s8583916/home/h3/s8583916/home/scratch/'+
          's8583916-stefans_ws/mthesis/results/fibbonaccisphereshort.json')
f2 = open('/run/user/1000/gvfs/sftp:host=taurus.hrsk.tu-dresden.de,user=s8583916/home/h3/s8583916/home/scratch/'+
          's8583916-stefans_ws/mthesis/results/fibbonaccisphere_dirshort.json')
f3 = open('/run/user/1000/gvfs/sftp:host=taurus.hrsk.tu-dresden.de,user=s8583916/home/h3/s8583916/home/scratch/'+
          's8583916-stefans_ws/mthesis/results/equal-areasphereshort.json')
f4 = open('/run/user/1000/gvfs/sftp:host=taurus.hrsk.tu-dresden.de,user=s8583916/home/h3/s8583916/home/scratch/'+
          's8583916-stefans_ws/mthesis/results/equal-areasphere_dirshort.json')
f5 = open('/run/user/1000/gvfs/sftp:host=taurus.hrsk.tu-dresden.de,user=s8583916/home/h3/s8583916/home/scratch/'+
          's8583916-stefans_ws/mthesis/results/randomsphereshort.json')
f6 = open('/run/user/1000/gvfs/sftp:host=taurus.hrsk.tu-dresden.de,user=s8583916/home/h3/s8583916/home/scratch/'+
          's8583916-stefans_ws/mthesis/results/randomsphere_dirshort.json')
# returns JSON object as
# a dictionary
instance1 = analytics(1,1,1000)
data1 = json.load(f1)
data2 = json.load(f2)
data3 = json.load(f3)
data4 = json.load(f4)
data5 = json.load(f5)
data6 = json.load(f6)
exponent = np.arange(16)
q_values_1 = 10 ** (-exponent / 3)
#exponent = np.arange(8)
#q_values_1 = 10 ** (-exponent / 1.5)
exponent = np.arange(201)
q_values = 10**(-exponent/40)
r_values = r_0_values = [0.283, 0.447, 0.894, 1.789]#[0.283, 0.447, 0.632, 0.894, 1.265, 1.789]
fig, ((ax, ax1), (ax2, ax3), (ax4, ax5)) = plt.subplots(nrows=3, ncols=2, figsize=(6.4, 7.2), sharex=True)
for r_0,item in zip(r_values,('20', '50', '200', '800')):
    for a, b, d1, d2 in zip([ax, ax2, ax4], [ax1, ax3, ax5], [data1, data3, data5], [data2, data4, data6]):
        k=data1[str(r_0)]['k']
        # second largest
        line0, = a.plot(q_values, [instance1.second_lam_sphere(q=q, r_0=r_0) for q in q_values],zorder=2, linewidth=1)
                          #label=r'$r_0={}~~(\^k={:.2f})$'.format(r_0, k),
        # undirected
        line2 = a.errorbar(q_values_1, [data1[str(r_0)]['qs'][str(q)]['second_largest'][0] for q in q_values_1],
                             yerr=[d1[str(r_0)]['qs'][str(q)]['second_largest'][1] for q in q_values_1], fmt='x',
                             color=line0.get_color(),
                             markerfacecolor='none', capsize=2, label=r'$k \approx $'+ item)
        # directed
        a.errorbar(q_values_1, [d2[str(r_0)]['qs'][str(q)]['second_largest'][0] for q in q_values_1],
                    yerr=[d2[str(r_0)]['qs'][str(q)]['second_largest'][1] for q in q_values_1], fmt='o',
                    color=line0.get_color(),
                    markerfacecolor='none', capsize=2)
        # smallest
        b.plot(q_values, [instance1.smallest_lam_sphere(q=q, r_0=r_0) for q in q_values],
             zorder=2, linewidth=1, color=line0.get_color())
        # undirected
        line3 = b.errorbar(q_values_1, [d1[str(r_0)]['qs'][str(q)]['smallest'][0] for q in q_values_1],
                            yerr=[d1[str(r_0)]['qs'][str(q)]['smallest'][1] for q in q_values_1], fmt='x',
                            color=line0.get_color(),
                            markerfacecolor='none', capsize=2, label=r'$k \approx $'+ item)
        # directed
        line4 = b.errorbar(q_values_1, [d2[str(r_0)]['qs'][str(q)]['smallest'][0] for q in q_values_1],
                     yerr=[d2[str(r_0)]['qs'][str(q)]['smallest'][1] for q in q_values_1], fmt='o',
                     color=line0.get_color(),
                     markerfacecolor='none', capsize=2)
for ax, label, label2 in zip((ax, ax2, ax4),('a', 'c', 'e'),(r'fibbonacci, $\lambda_2$', r'equal-area, $\lambda_2$', r'random, $\lambda_2$')):
    #ax.grid()
    ax.set_yscale('symlog', linthresh=0.01)
    ax.set_xscale('log')
    ax.set_title(label, loc='left', fontweight='bold', fontsize=10)
    ax.set_title(label2, loc='center', fontsize=10)
    ax.set_ylabel(r'Re $\lambda_2$')
ax4.set_xlabel('$q$')
ax5.set_xlabel(r'$q$')
for ax, label, label2 in zip((ax1, ax3, ax5),('b', 'd', 'f'),(r'fibbonacci, $\lambda_-$', r'equal-area, $\lambda_-$', r'random, $\lambda_-$')):
    #ax.grid()
    ax.set_xscale('log')
    ax.set_ylim(-2, -1)
    ax.set_title(label, loc='left', fontweight='bold', fontsize=10)
    ax.set_title(label2, loc='center', fontsize=10)
    ax.set_ylabel(r'Re $\lambda_-$')
ax1.legend(loc='lower left')
ax5.legend([line0, line3, line4], ['mean-field', 'undirected','directed'], loc='lower left')
plt.tight_layout()
plt.savefig('../figures/sphere_comparison.svg', format='svg', dpi=1000)
plt.show()