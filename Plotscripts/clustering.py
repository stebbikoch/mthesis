from matplotlib import pyplot as plt
import json
import numpy as np
from analytics import analytics

f1 = open('/run/user/1000/gvfs/sftp:host=taurus.hrsk.tu-dresden.de,user=s8583916/home/h3/s8583916/home/scratch/s8583916-stefans_ws/mthesis/results/1d_clusteringshort.json' )
f11 = open('/run/user/1000/gvfs/sftp:host=taurus.hrsk.tu-dresden.de,user=s8583916/home/h3/s8583916/home/scratch/s8583916-stefans_ws/mthesis/results/1d_clustering_dirshort.json')
f2 = open('/run/user/1000/gvfs/sftp:host=taurus.hrsk.tu-dresden.de,user=s8583916/home/h3/s8583916/home/scratch/s8583916-stefans_ws/mthesis/results/2d_clusteringshort.json' )
f3 = open('/run/user/1000/gvfs/sftp:host=taurus.hrsk.tu-dresden.de,user=s8583916/home/h3/s8583916/home/scratch/s8583916-stefans_ws/mthesis/results/3d_clusteringshort.json')
f31 = open('/run/user/1000/gvfs/sftp:host=taurus.hrsk.tu-dresden.de,user=s8583916/home/h3/s8583916/home/scratch/s8583916-stefans_ws/mthesis/results/3d_clustering_dirshort.json')
f4 = open('/run/user/1000/gvfs/sftp:host=taurus.hrsk.tu-dresden.de,user=s8583916/home/h3/s8583916/home/scratch/s8583916-stefans_ws/mthesis/results/sphere_clustering.jsonshort.json')
data1=json.load(f1)
data11=json.load(f11)
data2=json.load(f2)
data3=json.load(f3)
data31=json.load(f31)
data4=json.load(f4)
fig, ((ax, ax1), (ax2, ax3)) = plt.subplots(nrows=2, ncols=2, figsize=(6.4, 4.8))
exponent = np.arange(16)
q_values = 10 ** (-exponent / 3)
# colors
a = '#1f77b4'
b = '#ff7f0e'
# one dimension
line1 = ax.errorbar(q_values, [data1["13"]['qs'][str(q)]['clustering'][0] for q in q_values],
                             yerr=[data1["13"]['qs'][str(q)]['clustering'][1] for q in q_values], fmt='x',
                          markerfacecolor='none', label='Clustering coefficient', color=a)
line11 = ax.errorbar(q_values, [data11["13"]['qs'][str(q)]['clustering'][0] for q in q_values],
                             yerr=[data11["13"]['qs'][str(q)]['clustering'][1] for q in q_values], fmt='o',
                          markerfacecolor='none', capsize=5, label='Clustering coefficient', color=a)
ax01 = ax.twinx()
line2 = ax01.errorbar(q_values, [data1["13"]['qs'][str(q)]['characteristic_path'][0] for q in q_values],
                             yerr=[data1["13"]['qs'][str(q)]['characteristic_path'][1] for q in q_values], fmt='x',
              color= b,
                          markerfacecolor='none', capsize=5, label='Characteristic path-length')
line2 = ax01.errorbar(q_values, [data11["13"]['qs'][str(q)]['characteristic_path'][0] for q in q_values],
                             yerr=[data11["13"]['qs'][str(q)]['characteristic_path'][1] for q in q_values], fmt='o',
              color= b,
                          markerfacecolor='none', capsize=5, label='Characteristic path-length')
# two dimensions
ax1.errorbar(q_values, [data2["2"]['qs'][str(q)]['clustering'][0] for q in q_values],
                             yerr=[data2["2"]['qs'][str(q)]['clustering'][1] for q in q_values], fmt='x',
                          markerfacecolor='none', capsize=5, color=a)
ax11 = ax1.twinx()
ax11.errorbar(q_values, [data2["2"]['qs'][str(q)]['characteristic_path'][0] for q in q_values],
                             yerr=[data2["2"]['qs'][str(q)]['characteristic_path'][1] for q in q_values], fmt='x',
              color= b,
                          markerfacecolor='none', capsize=5)
# three dimensions
ax2.errorbar(q_values, [data3["1"]['qs'][str(q)]['clustering'][0] for q in q_values],
                             yerr=[data3["1"]['qs'][str(q)]['clustering'][1] for q in q_values], fmt='x',
                          markerfacecolor='none', capsize=5, color=a, label='undirected')
ax2.errorbar(q_values, [data31["1"]['qs'][str(q)]['clustering'][0] for q in q_values],
                             yerr=[data31["1"]['qs'][str(q)]['clustering'][1] for q in q_values], fmt='o',
                          markerfacecolor='none', capsize=5, color=a, label='directed')
ax21 = ax2.twinx()
ax21.errorbar(q_values, [data3["1"]['qs'][str(q)]['characteristic_path'][0] for q in q_values],
                             yerr=[data3["1"]['qs'][str(q)]['characteristic_path'][1] for q in q_values], fmt='x',
              color= b,
                          markerfacecolor='none', capsize=5)
ax21.errorbar(q_values, [data31["1"]['qs'][str(q)]['characteristic_path'][0] for q in q_values],
                             yerr=[data31["1"]['qs'][str(q)]['characteristic_path'][1] for q in q_values], fmt='o',
              color= b,
                          markerfacecolor='none', capsize=5)
# sphere network
ax3.errorbar(q_values, [data4["0.165"]['qs'][str(q)]['clustering'][0] for q in q_values],
                             yerr=[data4["0.165"]['qs'][str(q)]['clustering'][1] for q in q_values], fmt='x',
                          markerfacecolor='none', capsize=5, color=a)
ax31 = ax3.twinx()
ax31.errorbar(q_values, [data4["0.165"]['qs'][str(q)]['characteristic_path'][0] for q in q_values],
                             yerr=[data4["0.165"]['qs'][str(q)]['characteristic_path'][1] for q in q_values], fmt='x',
              color= b,
                          markerfacecolor='none', capsize=5)
#plt.gca().add_artist(legend)
#fig.legend([line1, line2],
 #                        [r"Clustering coefficient", r'Characteristic path-length'],
  #                       bbox_to_anchor=(1.3,1), loc='upper right')
#titles = [r'$1D$ torus', r'$2D$ torus ($\infty$-norm)', r'$3D$ torus ($\infty$-norm)',
 #         r'fibbonacci sphere-network']
ax.set_ylabel('Clust. coefficient')
for ax, label in zip((ax, ax1, ax2, ax3), ['a', 'b', 'c', 'd']):
    ax.tick_params(axis='y', colors=a)
    ax.yaxis.label.set_color(a)
    ax.set_xscale('log')
    ax.grid()
    #ax.text(0.0, 1.0, 'a', fontweight = 'bold')
    ax.set_title(label, loc='left', fontweight='bold', fontsize=12)
    #bbox = ax.get_tightbbox(fig.canvas.get_renderer())
    #ax.text(bbox.x0, bbox.y1, label, fontsize=12, fontweight="bold", va="top", ha="left",
     #       transform=None)
ax31.set_ylabel('Char. path-length')
ax3.set_xlabel(r'topological randomness $q$')
ax2.legend(loc='lower left')
for i in (ax01, ax11, ax21, ax31):
    i.tick_params(axis='y', colors=b)
    i.yaxis.label.set_color(b)
    i.spines['right'].set_color(b)
    i.spines['left'].set_color(a)
fig.tight_layout()
plt.savefig('../figures/clustering_pathlength.svg', format='svg', dpi=1000)
plt.show()