from matplotlib import pyplot as plt
import json

f1 = open('./reproduce/reproduce_1001_undirected_analytics.txt', )
f2 = open('./reproduce/reproduce_1001_undirected_no_averaging.txt', )
# returns JSON object as
# a dictionary
data1 = json.load(f1)
data2 = json.load(f2)

fig, axs = plt.subplots(1)
q_values = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
k_values = [20, 50, 100, 200, 400, 800]
c=['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
index=0
for k in k_values:
    axs.plot(q_values, [data1[str(k)][str(q)] for q in q_values])#, color=c[index])
    axs.plot(q_values, [data2[str(k)][str(q)] for q in q_values], 'o')#, color=c[index])
axs.set_yscale('symlog', linthreshy=0.0001)
axs.set_xscale('log')
#axs[1].set_yscale('symlog')
#axs[1].set_xscale('log')
plt.show()