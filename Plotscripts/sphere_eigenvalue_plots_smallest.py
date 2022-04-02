from matplotlib import pyplot as plt
import json
import numpy as np
import statistics as st
from analytics import analytics

#f1 = open('./reproduce/reproduce_1001_undirected_analytics.txt', )
#f3 = open('/run/user/1000/gvfs/sftp:host=taurus.hrsk.tu-dresden.de,user=s8583916/home/h3/s8583916/home/scratch/'+
 #         's8583916-stefans_ws/mthesis/results/1000ring_sl_undirected_100xshort.json')
#f = open('/run/user/1000/gvfs/sftp:host=taurus.hrsk.tu-dresden.de,user=s8583916/home/h3/s8583916/home/scratch/'+
#          's8583916-stefans_ws/mthesis/results/test_sphere_second_largest1000.jsonshort.json')
#f = open('/run/user/1000/gvfs/sftp:host=taurus.hrsk.tu-dresden.de,user=s8583916/home/h3/s8583916/home/scratch/'+
 #         's8583916-stefans_ws/mthesis/results/sphere_new_r0_1000_undirected.jsonshort.json')
f = open('../results/fibbo_sphere_1000short.json')
f1=open('../results/random_sphere_1000short.json')
f2=open('../results/eq_sphere_1000short.json')
#f2 = open('/run/user/1000/gvfs/sftp:host=taurus.hrsk.tu-dresden.de,user=s8583916/home/h3/s8583916/home/scratch/'+
 #         's8583916-stefans_ws/mthesis/results/sphere_new_r0_1000_undirected.jsonshort.json')
#f2 = open('results/test_sphere_smallest5000.jsonshort.json')
#f2 = open('./reproduce/reproduce_1000_directed_with_averaging_test_big_q.json', )
#f3 = open('./reproduce/reproduce_1000_undirected_with_averaging_test_big_q.json', )
# returns JSON object as
# a dictionary
#instance = analytics(1,1,5000)
instance1 = analytics(1,1,1000)
data = json.load(f)
data1 = json.load(f1)
data2= json.load(f2)
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
r_values = [0.283, 0.447, 0.632, 0.894, 1.265, 1.789]#[0.2, 0.3, 0.5, 1, 1.3, 1.7]#[20, 50, 100, 200, 400, 800]
r_values =  [0.2, 0.4, 0.6, 0.8, 1, 1.3]
c=['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
#index=0
#k = 50
#q=1.0
#print(data2[str(k)][str(q)])
#print([data2[str(0.2)]['qs'][str(q)] for q in q_values_1])
k_list=[]
for r_0 in r_values:
    k=data[str(r_0)]['k']
    r=np.sqrt((k+1)*np.sqrt(3)/(2*np.pi))
    r= round(r, 2)
    print(r)
    #,
                      #label=r'$r_0={}~~(\^k={:.2f})$'.format(r_0, k), zorder=1)
    line0, = plt.plot(q_values, instance1.exact_eigens(q_values, r),
                   label=r'$r_0={}~~(\^k={:.2f})$'.format(r_0, k), zorder=1)
    line, = plt.plot(q_values, instance1.smallest_lam_sphere(q_values, r_0), color=line0.get_color(), alpha=0.5)
    #line0, =plt.plot(q_values, [instance1.smallest_lam_sphere(q=q, r_0=r_0) for q in q_values], label=r'r_0='+str(r_0))# color=line0.get_color())#, color=c[index])
    #line2 =plt.scatter(q_values_1, [data2[str(r_0)]['qs'][str(q)]['second_largest'][0] for q in q_values_1],color=line0.get_color(), zorder=2)#, color=c[index])
    #line3 = plt.scatter(q_values_1, [data[str(r_0)]['qs'][str(q)]['second_largest'][0] for q in q_values_1], color=line0.get_color(), zorder=2)
    line2 = plt.errorbar(q_values_1, [data[str(r_0)]['qs'][str(q)]['smallest'][0] for q in q_values_1],
                         yerr=[data[str(r_0)]['qs'][str(q)]['smallest'][1] for q in q_values_1], fmt='x',
                         color=line0.get_color(),
                         markerfacecolor='none', capsize=5)
    line3 = plt.errorbar(q_values_1, [data1[str(r_0)]['qs'][str(q)]['smallest'][0] for q in q_values_1],
                         yerr=[data1[str(r_0)]['qs'][str(q)]['smallest'][1] for q in q_values_1], fmt='o',
                         color=line0.get_color(),
                         markerfacecolor='none', capsize=5)
    line4 = plt.errorbar(q_values_1, [data2[str(r_0)]['qs'][str(q)]['smallest'][0] for q in q_values_1],
                         yerr=[data2[str(r_0)]['qs'][str(q)]['smallest'][1] for q in q_values_1], fmt='+',
                         color=line0.get_color(),
                         markerfacecolor='none', capsize=5)
    #line3 = plt.errorbar(q_values_1, [data[str(r_0)]['qs'][str(q)]['second_largest'][0] for q in q_values_1],
     ##                    yerr=[data[str(r_0)]['qs'][str(q)]['second_largest'][1] for q in q_values_1], fmt='o',
       #                  color=line0.get_color(),
        #              markerfacecolor='none', capsize=5)
    yd = -2 * np.sqrt(1 / k - 1 / 1000) - 1
    yu = -np.sqrt(1 / k - 1 / 10000) - 1
    line5=plt.hlines(yd, 0.5, 1.5, linewidth=0.4, color=line0.get_color(), linestyles='--')
    #line5=plt.hlines(yu, 0.5, 1.5, linewidth=0.4, color=line0.get_color(), linestyles='-.')
plt.yscale('symlog', linthreshy=0.0001)
plt.yticks([-1, -1.1, -1.2, -1.3, -1.4, -1.5, -1.6, -1.7],[-1.0, -1.1, -1.2, -1.3, -1.4, -1.5, -1.6, -1.7])
plt.xscale('log')
#plt.xlim(0.4,1.1)
#plt.ylim(-1, -0.5)
plt.xlabel(r'Small-world parameter $q$')
plt.ylabel(r'Value of normalized second largest eigenvalue')
legend1 = plt.legend([line0, line, line2, line3, line4, line5], [r"Analytical prediction closed packed hexagonal"+ "\n" +"lattice on torus with eucledian norm",
                                                                 "Analytical prediction sphere",
                                                           r"Numerical results undirected fibbonaci sphere grid",
                                                                 "... random sphere grid",
                                                                 "... equal partitioning sphere grid"
                                                           , "Wigner semi-circle undirected"], loc='lower right')
plt.legend(loc='upper left')
plt.gca().add_artist(legend1)
plt.grid()
plt.tight_layout()
#axs[1].set_yscale('symlog')
#axs[1].set_xscale('log')
plt.savefig('figures/sphere_smallest_1000_unaveraged.svg', format='svg', dpi=1000)
plt.show()