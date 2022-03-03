from matplotlib import pyplot as plt
import json
import numpy as np
import statistics as st
from analytics import analytics

#f1 = open('./reproduce/reproduce_1001_undirected_analytics.txt', )
#f2 = open('../results/wattsstrogatz_undir_bothshort.json')
f2 = open('../results/wattsstrogatz_undir_both_ws_grabows_mistakeshort.json')
#f3 = open('../results/wattsstrogatz_undir_both_ws_originalshort.json')
f3 = open('../results/wattsstrogatz_undir_both_ws_original_grabows_mistakeshort.json')
#f2 = open('./reproduce/reproduce_1000_directed_with_averaging_test_big_q.json', )
#f3 = open('./reproduce/reproduce_1000_undirected_with_averaging_test_big_q.json', )
# returns JSON object as
# a dictionary
only_upper=True
data2 = json.load(f2)
data3=json.load(f3)
fig = plt.figure()
r_values = [10, 25, 50, 100, 200, 400]
c=['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
for k in r_values:
    line1, = plt.plot(0,0)
    line2 =plt.errorbar(1, data2[str(k)]['qs'][str(1)]['second_largest'][0],
                        yerr=data2[str(k)]['qs'][str(1)]['second_largest'][1],  fmt='o', color=line1.get_color(), markerfacecolor='none', capsize=10)
    line3 = plt.errorbar(3, data2[str(k)]['qs'][str(1)]['second_largest_norm'][0],
                         yerr=data2[str(k)]['qs'][str(1)]['second_largest_norm'][1],  fmt='o',color=line1.get_color(), markerfacecolor='none', capsize=10, label='k='+str(2*k))
    line5 = plt.errorbar(1, data2[str(k)]['qs'][str(1)]['smallest'][0],
                         yerr=data2[str(k)]['qs'][str(1)]['smallest'][1], fmt='o', color=line1.get_color(), markerfacecolor='none',
                         capsize=10)
    line6 = plt.errorbar(3, data2[str(k)]['qs'][str(1)]['smallest_norm'][0],
                         yerr=data2[str(k)]['qs'][str(1)]['smallest_norm'][1], fmt='o', color=line1.get_color(),
                         markerfacecolor='none', capsize=10)
    line8 = plt.errorbar(2, data3[str(k)]['qs'][str(1)]['second_largest'][0],
                         yerr=data3[str(k)]['qs'][str(1)]['second_largest'][1], fmt='o', color=line1.get_color(),
                         markerfacecolor='none', capsize=10)
    line9 = plt.errorbar(4, data3[str(k)]['qs'][str(1)]['second_largest_norm'][0],
                         yerr=data3[str(k)]['qs'][str(1)]['second_largest_norm'][1], fmt='o', color=line1.get_color(),
                         markerfacecolor='none', capsize=10)
    line10 = plt.errorbar(2, data3[str(k)]['qs'][str(1)]['smallest'][0],
                         yerr=data3[str(k)]['qs'][str(1)]['smallest'][1], fmt='o', color=line1.get_color(),
                         markerfacecolor='none',
                         capsize=10)
    line11 = plt.errorbar(4, data3[str(k)]['qs'][str(1)]['smallest_norm'][0],
                         yerr=data3[str(k)]['qs'][str(1)]['smallest_norm'][1], fmt='o', color=line1.get_color(),
                         markerfacecolor='none', capsize=10)
    line4 = plt.hlines(2*np.sqrt(1/(2*k)-1/1000)-1, 1, 4.2, linestyle='--',color=line1.get_color(), alpha=0.4)
    line7 = plt.hlines(-2 * np.sqrt(1 / (2 * k) - 1 / 1000) - 1, 2, 4.2,  linestyle='--', color=line1.get_color(), alpha=0.4)
    #line7 = plt.hlines(-data2[str(k)]['qs'][str(1)]['k_max']/(2*k), 0.8, 2.2, linestyle='-.', color=line1.get_color(), alpha=0.4)
    #line7 = plt.hlines(-data2[str(k)]['qs'][str(1)]['k_min'] / (2 * k), 0.8, 2.2, linestyle='-.', color=line1.get_color(),
     #                   alpha=0.4)
    kx=2*k
    p = kx/999
    q = 1-p
    def approx(p,q):
        if p > 0.92:
            m = 10
        elif p > 0.7:
            m = 5
        else:
            m = 1
        output = p*1000*m + m*np.sqrt(2*p*1000*q*1000/999*np.log(1000/m))
        return output/m
    def approx2(p,q):
        n=1000
        output = p*n + np.sqrt(2*p*q*n*np.log(n)) - np.sqrt(p*q*n/8/np.log(n)*np.log(np.log(n)))
        return output
    line8 = plt.hlines(- approx2(p,q)/kx, 0.8, 2, linestyle='-',
                       color=line1.get_color(),
                       alpha=0.4)
    p = q
    q = 1 - p
    line7 = plt.hlines(-(1000 - approx2(p,q))/kx, 0.8, 2, linestyle='-',
                       color=line1.get_color(),
                       alpha=0.4)
    #plt.hlines(yu, 0.5, 1.5, linewidth=0.4, color=line1.get_color(), linestyles='--')
    #line4 = plt.axhline(y=, linestyle='--', color=line1.get_color(), alpha=0.4)

if only_upper:
    plt.yscale('symlog', linthresh=0.01)
    plt.yticks([-0.3, -0.4, -0.5, -0.6, -0.7, -0.8, -0.9, -1], [-0.3, -0.4, -0.5, -0.6, -0.7, -0.8, -0.9, -1])
    plt.ylim(-1.02, -0.2)
#else:
    #plt.ylim(ymax=-0.1)
    #plt.yticks([-i/10 for i in range(2, 20)], [-i/10 for i in range(2, 20)])
plt.xticks([1,2,3,4],['generalized \n scaled','original \n scaled', 'generalized \n normalized', 'original \n normalized'])
#plt.xticks([1.5, 2.5],['scaled laplacian', 'normalized laplacian'])
plt.xlim(0.5, 4.5)
#plt.xscale('log')
#plt.xlim(0.47,1.03)
#plt.yticks([-1, -0.9, -0.8, -0.7, -0.6],[-1.0, -0.9, -0.8, -0.7, -0.6])
#plt.xlabel(r'Small-world parameter $q$')
#plt.ylabel(r'Value of normalized second largest eigenvalue')
legend1 = plt.legend([line2, line7, line8], [r"Numerical results undirected", r'Wigner semi-circle prediction undirected', r'Prediction from estimation on degree distribution'], loc='upper left')
plt.legend(loc='upper right')
plt.gca().add_artist(legend1)
plt.grid()
plt.tight_layout()
#axs[1].set_yscale('symlog')
#axs[1].set_xscale('log')
if only_upper:
    plt.savefig('../figures/investigation_upper_grabows_mistake.svg', format='svg', dpi=1000)
#else:
 #   plt.savefig('../figures/investigation_all.svg', format='svg', dpi=1000)
plt.show()