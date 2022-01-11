from matplotlib import pyplot as plt
import json
import numpy as np
import statistics as st
from analytics import analytics

#f1 = open('./reproduce/reproduce_1001_undirected_analytics.txt', )
f3 = open('/run/user/1000/gvfs/sftp:host=taurus.hrsk.tu-dresden.de,user=s8583916/home/h3/s8583916/home/scratch/'+
          's8583916-stefans_ws/mthesis/results/1000ring_sl_undirected_Q_100xshort.json')
f2 = open('/run/user/1000/gvfs/sftp:host=taurus.hrsk.tu-dresden.de,user=s8583916/home/h3/s8583916/home/scratch/'+
          's8583916-stefans_ws/mthesis/results/1000ring_sl_directed_Q_100xshort.json')
#f2 = open('./reproduce/reproduce_1000_directed_with_averaging_test_big_q.json', )
#f3 = open('./reproduce/reproduce_1000_undirected_with_averaging_test_big_q.json', )
# returns JSON object as
# a dictionary
instance = analytics(1,1,1000)
data2 = json.load(f2)
data3=json.load(f3)
fig = plt.figure()
exponent = np.arange(16)
q_values_1 = 10 ** (-exponent / 3)
#q_values_1 = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
#q_values_1 = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
exponent = np.arange(51)
q_values = 10**(-exponent/10)
q_values_1 = [1, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5]
k_values = [10, 25, 50, 100, 200, 400]#[20, 50, 100, 200, 400, 800]
c=['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
#index=0
#k = 50
#q=1.0
#print(data2[str(k)][str(q)])
for k in k_values:
    line1, =plt.plot(q_values, [instance.second_lam_one_dim(q=q, k=2*k)/(2*k) for q in q_values], label=r'k='+str(2*k), alpha=0.4)#, color=c[index])
    line2 =plt.errorbar(q_values_1, [data2[str(k)]['qs'][str(q)][0] for q in q_values_1], yerr=[data2[str(k)]['qs'][str(q)][1] for q in q_values_1],  fmt='o',color=line1.get_color(), markerfacecolor='none', capsize=10)#, color=c[index])
    line3 = plt.errorbar(q_values_1, [data3[str(k)][str(q)][0] for q in q_values_1], yerr=[data3[str(k)][str(q)][1] for q in q_values_1], fmt='x', color=line1.get_color(),
                      markerfacecolor='none', capsize=5)
    line4 = plt.axhline(y=2*np.sqrt(1/(2*k)-1/1000)-1, linestyle='--',color=line1.get_color(), alpha=0.4)
    line5 = plt.axhline(y=np.sqrt(1/(2*k)-1/1000)-1,linestyle='-.',color=line1.get_color(), alpha=0.4)
plt.yscale('symlog', linthreshy=0.0001)
#plt.xscale('log')
plt.xlim(0.47,1.03)
plt.ylim(-1.05, -0.35)
plt.yticks([-1, -0.9, -0.8, -0.7, -0.6],[-1.0, -0.9, -0.8, -0.7, -0.6])
plt.xlabel(r'Small-world parameter $q$')
plt.ylabel(r'Value of normalized second largest eigenvalue')
legend1 = plt.legend([line1, line2, line3, line4, line5], [r"Analytical prediction", r"Numerical results directed", r'Numerical results undirected', r'Wigner semi-circle prediction undirected', r'Wigner semi-circle prediction directed'], loc=9)
plt.legend(loc=1)
plt.gca().add_artist(legend1)
plt.grid()
plt.tight_layout()
#axs[1].set_yscale('symlog')
#axs[1].set_xscale('log')
plt.savefig('figures/1000ring_Q_100x.svg', format='svg', dpi=1000)
#plt.show()