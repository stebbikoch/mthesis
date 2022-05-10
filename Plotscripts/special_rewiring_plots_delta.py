from matplotlib import pyplot as plt
import json
import numpy as np
import statistics as st
from analytics import analytics


f = open('../reproduce/ring_specialshort.json')
mu=300
sigma=30
# returns JSON object as
# a dictionary
instance = analytics(1,1,1000)
data = json.load(f)
#data3=json.load(f3)
n=int(10)
ara = range(1,n)
fac=300
end=0.2*3*fac
exponent = np.arange(end, 15*fac+1)
q_values = 10**(-exponent/(3*fac))
exponent = np.arange(1,16)
m_values = 10 ** (-exponent / 3)
exponent = np.arange(15).astype(float)
m_values = (10**(-exponent/5)*1000).astype(int)
#m_values = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.2, 0.4]#[0,2,5,10,15,20,25,30,40,50]+[int(item/n*1000) for item in ara]
exponent = np.arange(1001).astype(float)
m_values_1 = (10**(-exponent/(1000/3))*1000)
#m_values_1 = np.append(m_values_1, np.arange(10)/10)
r_0_values = [10, 25, 50, 100]#, 200, 400]#[20, 50, 100, 200, 400, 800]
c=['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
#index=0
#k = 50
#q=1.0
#print(data2[str(k)][str(q)])
fig, (ax, ax1) = plt.subplots(ncols=2, figsize=(6.4, 3))
for r_0 in r_0_values:
    line1, =ax.plot(m_values_1, [instance.special_rewiring_lam(r_0, m) for m in m_values_1], label=r'k='+str(2*r_0))#, color=c[index])
    line2 =ax.errorbar(m_values, [data[str(r_0)]['ms'][str(m)]['second_largest'][0] for m in m_values], yerr=[data[str(r_0)]['ms'][str(m)]['second_largest'][1] for m in m_values],  color=line1.get_color(), fmt='o', markerfacecolor='none', capsize=10)#, color=c[index])
    line1, = ax1.plot(m_values_1, [instance.special_rewiring_lam(r_0, m, smallest=True) for m in m_values_1], label=r'k='+str(2*r_0))#, color=line1.get_color())
    line2 = ax1.errorbar(m_values, [data[str(r_0)]['ms'][str(m)]['smallest'][0] for m in m_values],
                        yerr=[data[str(r_0)]['ms'][str(m)]['smallest'][1] for m in m_values],
                       color=line1.get_color(), fmt='o', markerfacecolor='none', capsize=10)

#plt.yscale('symlog', linthreshy=0.0001)
for a, label in zip([ax, ax1],['a', 'b']):
    a.set_xscale('log')
    a.set_xlabel(r'moved edges $m$')
    a.set_title(label, loc='left', fontweight='bold', fontsize=10)
#plt.xlim(0.47,1.03)
#plt.xlim(None, 0.75)
#plt.yticks([-1, -0.9, -0.8, -0.7, -0.6],[-1.0, -0.9, -0.8, -0.7, -0.6])
ax.set_ylabel(r'$\lambda_2$')
ax1.set_ylabel(r'$\lambda_-$')
plt.tight_layout()
ax.legend(bbox_to_anchor=(0, 1.15), loc='lower left', ncol=4)
ax1.legend([line1, line2], [r"anal. prediction", r"num. result undirected"], loc='lower left')


#axs[1].set_yscale('symlog')
#axs[1].set_xscale('log')
plt.savefig('../figures/1000ring_special_largest.svg', format='svg', dpi=1000, bbox_inches='tight')
plt.show()