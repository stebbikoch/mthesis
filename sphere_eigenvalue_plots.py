from matplotlib import pyplot as plt
import json
import numpy as np
import statistics as st
from analytics import analytics

#f1 = open('./reproduce/reproduce_1001_undirected_analytics.txt', )
#f3 = open('/run/user/1000/gvfs/sftp:host=taurus.hrsk.tu-dresden.de,user=s8583916/home/h3/s8583916/home/scratch/'+
 #         's8583916-stefans_ws/mthesis/results/1000ring_sl_undirected_100xshort.json')
f = open('results/test_sphere_directed.jsonshort.json')
#f2 = open('results/test_sphere_smallest5000.jsonshort.json')
#f2 = open('./reproduce/reproduce_1000_directed_with_averaging_test_big_q.json', )
#f3 = open('./reproduce/reproduce_1000_undirected_with_averaging_test_big_q.json', )
# returns JSON object as
# a dictionary
instance = analytics(1,1,5000)
instance1 = analytics(1,1,1000)
data = json.load(f)
#data2 = json.load(f2)
#data3=json.load(f3)
fig = plt.figure()
exponent = np.arange(16)
q_values_1 = 10 ** (-exponent / 3)
#q_values_1 = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
#q_values_1 = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
exponent = np.arange(201)
q_values = 10**(-exponent/40)
#q_values = [0]
r_values = [0.2, 0.3, 0.5, 1, 1.3, 1.7]#[20, 50, 100, 200, 400, 800]
c=['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
#index=0
#k = 50
#q=1.0
#print(data2[str(k)][str(q)])
#print([data2[str(0.2)]['qs'][str(q)] for q in q_values_1])
for r_0 in r_values:
    #print(data2[str(r_0)]['k'])
    line0, = plt.plot(q_values, [instance1.second_lam_sphere(q=q, r_0=r_0) for q in q_values],
                      label=r'r_0=' + str(r_0))
    #line1, =plt.plot(q_values, [instance.smallest_lam_sphere(q=q, r_0=r_0) for q in q_values], label=r'r_0='+str(r_0), color=line0.get_color())#, color=c[index])
    #line2 =plt.scatter(q_values_1, [data2[str(r_0)]['qs'][str(q)][0] for q in q_values_1],color=line1.get_color())#, color=c[index])
    line3 = plt.scatter(q_values_1, [data[str(r_0)]['qs'][str(q)][0] for q in q_values_1], color=line0.get_color())
    #line3 = plt.errorbar(q_values_1, [data3[str(k)][str(q)][0] for q in q_values_1], yerr=[data3[str(k)][str(q)][1] for q in q_values_1], fmt='x', color=line1.get_color(),
     #                 markerfacecolor='none', capsize=5)
plt.yscale('symlog', linthreshy=0.0001)
plt.xscale('log')
#plt.xlim(0.4,1.1)
#plt.ylim(-1, -0.5)
plt.xlabel(r'Small-world parameter $q$')
plt.ylabel(r'Value of normalized second largest eigenvalue')
#legend1 = plt.legend([line1, line2], [r"Analytical prediction", r"Numerical results undirected"])
#plt.legend(loc=7)
#plt.gca().add_artist(legend1)
plt.grid()
plt.legend()
plt.tight_layout()
#axs[1].set_yscale('symlog')
#axs[1].set_xscale('log')
#plt.savefig('figures/1000ring_100x.svg', format='svg', dpi=1000)
plt.show()