from matplotlib import pyplot as plt
import json
import numpy as np
from analytics import analytics

f1 = open('/run/user/1000/gvfs/sftp:host=taurus.hrsk.tu-dresden.de,user=s8583916/home/h3/s8583916/home/scratch/s8583916-stefans_ws/mthesis/results/1d_clusteringshort.json' )
f2 = open('/run/user/1000/gvfs/sftp:host=taurus.hrsk.tu-dresden.de,user=s8583916/home/h3/s8583916/home/scratch/s8583916-stefans_ws/mthesis/results/2d_clusteringshort.json' )
f3 = open('/run/user/1000/gvfs/sftp:host=taurus.hrsk.tu-dresden.de,user=s8583916/home/h3/s8583916/home/scratch/s8583916-stefans_ws/mthesis/results/3d_clusteringshort.json')
data1=json.load(f1)
data2=json.load(f2)
data3=json.load(f3)
fig, ((ax, ax1), (ax2, ax3)) = plt.subplots(nrows=2, ncols=2)
exponent = np.arange(16)
q_values = 10 ** (-exponent / 3)
# colors
a = '#1f77b4'
b = '#ff7f0e'
line1 = ax.errorbar(q_values, [data1["13"]['qs'][str(q)]['clustering'][0] for q in q_values],
                             yerr=[data1["13"]['qs'][str(q)]['clustering'][1] for q in q_values], fmt='o',
                          markerfacecolor='none', capsize=5, label='Clustering coefficient')
ax01 = ax.twinx()
line2 = ax01.errorbar(q_values, [data1["13"]['qs'][str(q)]['characteristic_path'][0] for q in q_values],
                             yerr=[data1["13"]['qs'][str(q)]['characteristic_path'][1] for q in q_values], fmt='x',
              color= b,
                          markerfacecolor='none', capsize=5, label='Characteristic path-length')
ax1.errorbar(q_values, [data2["2"]['qs'][str(q)]['clustering'][0] for q in q_values],
                             yerr=[data2["2"]['qs'][str(q)]['clustering'][1] for q in q_values], fmt='o',
                          markerfacecolor='none', capsize=5)
ax11 = ax1.twinx()
ax11.errorbar(q_values, [data2["2"]['qs'][str(q)]['characteristic_path'][0] for q in q_values],
                             yerr=[data2["2"]['qs'][str(q)]['characteristic_path'][1] for q in q_values], fmt='x',
              color= b,
                          markerfacecolor='none', capsize=5)
ax2.errorbar(q_values, [data3["1"]['qs'][str(q)]['clustering'][0] for q in q_values],
                             yerr=[data3["1"]['qs'][str(q)]['clustering'][1] for q in q_values], fmt='o',
                          markerfacecolor='none', capsize=5)
ax21 = ax2.twinx()
ax21.errorbar(q_values, [data3["1"]['qs'][str(q)]['characteristic_path'][0] for q in q_values],
                             yerr=[data3["1"]['qs'][str(q)]['characteristic_path'][1] for q in q_values], fmt='x',
              color= b,
                          markerfacecolor='none', capsize=5)
#plt.gca().add_artist(legend)
#fig.legend([line1, line2],
 #                        [r"Clustering coefficient", r'Characteristic path-length'],
  #                       bbox_to_anchor=(1.3,1), loc='upper right')
titles = [r'$1D$ torus', r'$2D$ torus ($\infty$-norm)', r'$3D$ torus ($\infty$-norm)',
          r'fibbonacci sphere-network']
count = 0
for i in (ax, ax1, ax2):
    i.tick_params(axis='y', colors=a)
    i.yaxis.label.set_color(a)
    i.set_xscale('log')
    i.set_xlabel(r'$q$')
    i.set_ylabel('Clustering coefficient')
    i.grid()
    i.set_title(titles[count])
    count += 1
for i in (ax01, ax11, ax21):
    i.tick_params(axis='y', colors=b)
    i.yaxis.label.set_color(b)
    i.set_ylabel('Characteristic path-length')
    i.spines['right'].set_color(b)
    i.spines['left'].set_color(a)
fig.tight_layout()
plt.show()