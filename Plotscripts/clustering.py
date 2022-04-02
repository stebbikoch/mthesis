from matplotlib import pyplot as plt
import json
import numpy as np
import matplotlib as mpl
from analytics import analytics
#plt.rcParams["text.latex.preamble"].join([
 #       r"\usepackage{amsmath}"
#])

mpl.rcParams['lines.markersize'] = 4.5
f1 = open('/run/user/1000/gvfs/sftp:host=taurus.hrsk.tu-dresden.de,user=s8583916/home/h3/s8583916/home/scratch/s8583916-stefans_ws/mthesis/results/1d_clusteringshort.json' )
f11 = open('/run/user/1000/gvfs/sftp:host=taurus.hrsk.tu-dresden.de,user=s8583916/home/h3/s8583916/home/scratch/s8583916-stefans_ws/mthesis/results/1d_clustering_dirshort.json')
f2 = open('/run/user/1000/gvfs/sftp:host=taurus.hrsk.tu-dresden.de,user=s8583916/home/h3/s8583916/home/scratch/s8583916-stefans_ws/mthesis/results/2d_clusteringshort.json' )
f21 = open('/run/user/1000/gvfs/sftp:host=taurus.hrsk.tu-dresden.de,user=s8583916/home/h3/s8583916/home/scratch/s8583916-stefans_ws/mthesis/results/2d_clustering_dirshort.json' )
f3 = open('/run/user/1000/gvfs/sftp:host=taurus.hrsk.tu-dresden.de,user=s8583916/home/h3/s8583916/home/scratch/s8583916-stefans_ws/mthesis/results/3d_clusteringshort.json')
f31 = open('/run/user/1000/gvfs/sftp:host=taurus.hrsk.tu-dresden.de,user=s8583916/home/h3/s8583916/home/scratch/s8583916-stefans_ws/mthesis/results/3d_clustering_dirshort.json')
f4 = open('/run/user/1000/gvfs/sftp:host=taurus.hrsk.tu-dresden.de,user=s8583916/home/h3/s8583916/home/scratch/s8583916-stefans_ws/mthesis/results/sphere_clustering.jsonshort.json')
f41 = open('/run/user/1000/gvfs/sftp:host=taurus.hrsk.tu-dresden.de,user=s8583916/home/h3/s8583916/home/scratch/s8583916-stefans_ws/mthesis/results/sphere_clustering_dir.jsonshort.json')
data1=json.load(f1)
data11=json.load(f11)
data2=json.load(f2)
data21=json.load(f21)
data3=json.load(f3)
data31=json.load(f31)
data4=json.load(f4)
data41=json.load(f41)
fig, ((ax, ax1), (ax2, ax3)) = plt.subplots(nrows=2, ncols=2, figsize=(6.4, 4.8), sharey=True, sharex=True)
exponent = np.arange(16)
q_values = 10 ** (-exponent / 3)
# colors
a = '#1f77b4'
b = '#ff7f0e'
# one dimension
line1 = ax.errorbar(q_values, [data1["13"]['qs'][str(q)]['clustering'][0] for q in q_values],
                             yerr=[data1["13"]['qs'][str(q)]['clustering'][1] for q in q_values], fmt='x',
                          markerfacecolor='none', label='Clust. coef. undir.', color=a)
line11 = ax.errorbar(q_values, [data11["13"]['qs'][str(q)]['clustering'][0] for q in q_values],
                             yerr=[data11["13"]['qs'][str(q)]['clustering'][1] for q in q_values], fmt='o',
                          markerfacecolor='none', capsize=2, label='Clust. coef. dir.', color=a)
ax01 = ax.twinx()
line2 = ax01.errorbar(q_values, [data1["13"]['qs'][str(q)]['characteristic_path'][0] for q in q_values],
                             yerr=[data1["13"]['qs'][str(q)]['characteristic_path'][1] for q in q_values], fmt='x',
              color= b,
                          markerfacecolor='none', capsize=2, label='Char. p.-l. undir.')
line3 = ax01.errorbar(q_values, [data11["13"]['qs'][str(q)]['characteristic_path'][0] for q in q_values],
                             yerr=[data11["13"]['qs'][str(q)]['characteristic_path'][1] for q in q_values], fmt='o',
              color= b,
                          markerfacecolor='none', capsize=2, label='Char. p.-l. dir.')
# two dimensions
ax1.errorbar(q_values, [data2["2"]['qs'][str(q)]['clustering'][0] for q in q_values],
                             yerr=[data2["2"]['qs'][str(q)]['clustering'][1] for q in q_values], fmt='x',
                          markerfacecolor='none', capsize=2, color=a)
ax1.errorbar(q_values, [data21["2"]['qs'][str(q)]['clustering'][0] for q in q_values],
                             yerr=[data21["2"]['qs'][str(q)]['clustering'][1] for q in q_values], fmt='o',
                          markerfacecolor='none', capsize=2, color=a)
ax11 = ax1.twinx()
ax11.errorbar(q_values, [data2["2"]['qs'][str(q)]['characteristic_path'][0] for q in q_values],
                             yerr=[data2["2"]['qs'][str(q)]['characteristic_path'][1] for q in q_values], fmt='x',
              color= b,
                          markerfacecolor='none', capsize=2)
ax11.errorbar(q_values, [data21["2"]['qs'][str(q)]['characteristic_path'][0] for q in q_values],
                             yerr=[data21["2"]['qs'][str(q)]['characteristic_path'][1] for q in q_values], fmt='o',
              color= b,
                          markerfacecolor='none', capsize=2)
# three dimensions
line1=ax2.errorbar(q_values, [data3["1"]['qs'][str(q)]['clustering'][0] for q in q_values],
                             yerr=[data3["1"]['qs'][str(q)]['clustering'][1] for q in q_values], fmt='x',
                          markerfacecolor='none', capsize=2, color=a, label='Clust. undir.')
line2=ax2.errorbar(q_values, [data31["1"]['qs'][str(q)]['clustering'][0] for q in q_values],
                             yerr=[data31["1"]['qs'][str(q)]['clustering'][1] for q in q_values], fmt='o',
                          markerfacecolor='none', capsize=2, color=a, label='Clust. dir.')
ax21 = ax2.twinx()
line3=ax21.errorbar(q_values, [data3["1"]['qs'][str(q)]['characteristic_path'][0] for q in q_values],
                             yerr=[data3["1"]['qs'][str(q)]['characteristic_path'][1] for q in q_values], fmt='x',
              color= b,
                          markerfacecolor='none', capsize=2, label='Char. undir.')
line4=ax21.errorbar(q_values, [data31["1"]['qs'][str(q)]['characteristic_path'][0] for q in q_values],
                             yerr=[data31["1"]['qs'][str(q)]['characteristic_path'][1] for q in q_values], fmt='o',
              color= b,
                          markerfacecolor='none', capsize=2, label='Char. dir.')
# sphere network
ax3.errorbar(q_values, [data4["0.165"]['qs'][str(q)]['clustering'][0] for q in q_values],
                             yerr=[data4["0.165"]['qs'][str(q)]['clustering'][1] for q in q_values], fmt='x',
                          markerfacecolor='none', capsize=2, color=a)
ax3.errorbar(q_values, [data41["0.165"]['qs'][str(q)]['clustering'][0] for q in q_values],
                             yerr=[data41["0.165"]['qs'][str(q)]['clustering'][1] for q in q_values], fmt='o',
                          markerfacecolor='none', capsize=2, color=a)
ax31 = ax3.twinx()
ax31.errorbar(q_values, [data4["0.165"]['qs'][str(q)]['characteristic_path'][0] for q in q_values],
                             yerr=[data4["0.165"]['qs'][str(q)]['characteristic_path'][1] for q in q_values], fmt='x',
              color= b,
                          markerfacecolor='none', capsize=2)
ax31.errorbar(q_values, [data41["0.165"]['qs'][str(q)]['characteristic_path'][0] for q in q_values],
                             yerr=[data41["0.165"]['qs'][str(q)]['characteristic_path'][1] for q in q_values], fmt='o',
              color= b,
                          markerfacecolor='none', capsize=2)
#plt.gca().add_artist(legend)
#fig.legend([line1, line2],
 #                        [r"Clustering coefficient", r'Characteristic path-length'],
  #                       bbox_to_anchor=(1.3,1), loc='upper right')
#titles = [r'$1D$ torus', r'$2D$ torus ($\infty$-norm)', r'$3D$ torus ($\infty$-norm)',
 #         r'fibbonacci sphere-network']
ax.set_ylabel(r'Clust. coefficient $C$')
ax2.set_ylabel(r'Clust. coefficient $C$')
for axe, label, label1 in zip((ax, ax1, ax2, ax3), ['a', 'b', 'c', 'd'], [r'$1D$ ring', r'$2D$ torus', r'$3D$ torus', 'sphere']):
    axe.tick_params(axis='y', colors=a)
    axe.yaxis.label.set_color(a)
    axe.set_xscale('log')
    #äaxe.grid()
    #ax.text(0.0, 1.0, 'a', fontweight = 'bold')
    axe.set_title(label, loc='left', fontweight='bold', fontsize=10)
    axe.set_title(label1, loc='center', fontsize=10)
    #bbox = ax.get_tightbbox(fig.canvas.get_renderer())
    #ax.text(bbox.x0, bbox.y1, label, fontsize=12, fontweight="bold", va="top", ha="left",
     #       transform=None)
ax11.set_ylabel(r'Char. path-length $L$')
ax31.set_ylabel(r'Char. path-length $L$')
#ax2.legend([line1, line2, line3, line4], [r'$C_\mathrm{undir}(q)$', r'$C_\mathrm{dir}(q)$',
 #                                         r'$L_{\mathrm{undir}}(q)$', r'$L_\mathrm{dir}(q)$'], loc='lower left')
#ax2.legend([line1, line2], ['asdf', 'adf'], loc='center left')
for i in (ax01, ax11, ax21, ax31):
    i.tick_params(axis='y', colors=b)
    i.yaxis.label.set_color(b)
    i.spines['right'].set_color(b)
    i.spines['left'].set_color(a)
ax2.set_xlabel(r'topological randomness $q$')
ax3.set_xlabel(r'topological randomness $q$')
fig.tight_layout()
ax.legend([line1, line2, line3, line4], [r'$C_\mathrm{undir}(q)$', r'$C_\mathrm{dir}(q)$',
                                          r'$L_{\mathrm{undir}}(q)$', r'$L_\mathrm{dir}(q)$'],bbox_to_anchor=(0, 1.15), loc='lower left', ncol=4)
plt.savefig('../figures/clustering_pathlength.svg', format='svg', dpi=1000, bbox_inches='tight')
plt.show()