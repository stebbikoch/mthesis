from matplotlib import pyplot as plt
import json
import numpy as np
import statistics as st
from analytics import analytics


f = open('reproduce/ring_gaussianshort.json')
mu=300
sigma=30
# returns JSON object as
# a dictionary
instance = analytics(1,1,1000)
data = json.load(f)
#data3=json.load(f3)
fig = plt.figure()
n=int(10)
ara = range(1,n)
fac=300
end=0.2*3*fac
exponent = np.arange(end, 15*fac+1)
q_values = 10**(-exponent/(3*fac))
exponent = np.arange(1,16)
m_values = 10 ** (-exponent / 3)
#m_values = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.2, 0.4]#[0,2,5,10,15,20,25,30,40,50]+[int(item/n*1000) for item in ara]
m_values_1 = np.arange(1000).tolist()
r_0_values = [10, 25, 50, 100]#, 200, 400]#[20, 50, 100, 200, 400, 800]
c=['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
#index=0
#k = 50
#q=1.0
#print(data2[str(k)][str(q)])
for r_0 in r_0_values:
    line1, =plt.plot(q_values, [instance.special_rewiring_lam_gaussian(r_0, m, mu=mu, sigma=sigma) for m in q_values], label=r'k='+str(2*r_0))#, color=c[index])
    line2 =plt.errorbar(m_values, [data[str(r_0)]['ms'][str(m)]['second_largest'][0] for m in m_values], yerr=[data[str(r_0)]['ms'][str(m)]['second_largest'][1] for m in m_values],  color=line1.get_color(), fmt='o', markerfacecolor='none', capsize=10)#, color=c[index])
    #line1, = plt.plot(q_values, [instance.special_rewiring_lam_gaussian(r_0, m, mu=mu, sigma=sigma, smallest=True) for m in q_values], label=r'k='+str(2*r_0))#, color=line1.get_color())
    #line2 = plt.errorbar(m_values, [data[str(r_0)]['ms'][str(m)]['smallest'][0] for m in m_values],
     #                    yerr=[data[str(r_0)]['ms'][str(m)]['smallest'][1] for m in m_values],
      #                 color=line1.get_color(), fmt='o', markerfacecolor='none', capsize=10)

#plt.yscale('symlog', linthreshy=0.0001)
plt.xscale('log')
#plt.xlim(0.47,1.03)
plt.xlim(None, 0.75)
#plt.yticks([-1, -0.9, -0.8, -0.7, -0.6],[-1.0, -0.9, -0.8, -0.7, -0.6])
plt.xlabel(r'Small-world parameter $q$')
plt.ylabel(r'Value of normalized second largest eigenvalue')
legend1 = plt.legend([line1, line2], [r"Analytical prediction", r"Numerical results undirected"], loc='lower center')
plt.legend(loc='lower left')
plt.gca().add_artist(legend1)
plt.grid()
plt.tight_layout()
#axs[1].set_yscale('symlog')
#axs[1].set_xscale('log')
plt.savefig('figures/1000ring_gaussian_unaveraged_largest.svg', format='svg', dpi=1000)
plt.show()