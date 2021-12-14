# !/bin/python
from mpi4py import MPI
import json
import numpy as np
#comm = MPI.COMM_WORLD
#print("%d of %d" % (comm.Get_rank(), comm.Get_size()))
#f1 = open('/run/user/1000/gvfs/sftp:host=taurus.hrsk.tu-dresden.de,user=s8583916/home/h3/s8583916/home/scratch/s8583916-stefans_ws/mthesis/results/1000ring_undirected_100xshort.json')
#data = json.load(f1)
#print(data)
k = 20
N = 1000
lamdir = np.sqrt(1/k-1/N)-1
lamundir = 2*np.sqrt(1/k-1/N)-1
print('directed {}, undirected{}'.format(lamdir, lamundir))