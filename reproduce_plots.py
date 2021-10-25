from matplotlib import pyplot as plt
import json
import numpy as np
import statistics as st

f1 = open('./reproduce/reproduce_1001_undirected_analytics.txt', )
f2 = open('./reproduce/reproduce_1001_undirected_with_averaging.txt', )
# returns JSON object as
# a dictionary
data1 = json.load(f1)
data2 = json.load(f2)

fig, axs = plt.subplots(1)
exponent = np.arange(16)
q_values_1 = 10 ** (-exponent / 3)
exponent = np.arange(51)
q_values = 10**(-exponent/10)
k_values = [20, 50, 100, 200, 400, 800]
c=['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
index=0
k = 50
q=1.0
#print(data2[str(k)][str(q)])
for k in k_values:
    axs.plot(q_values, [data1[str(k)][str(q)] for q in q_values])#, color=c[index])
    axs.plot(q_values_1, [sum(data2[str(k)][str(q)])/10 for q in q_values_1], 'o')#, color=c[index])
axs.set_yscale('symlog', linthreshy=0.0001)
axs.set_xscale('log')
#axs[1].set_yscale('symlog')
#axs[1].set_xscale('log')
plt.show()