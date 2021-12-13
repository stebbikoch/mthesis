# !/bin/python
from mpi4py import MPI
import json
comm = MPI.COMM_WORLD
print("%d of %d" % (comm.Get_rank(), comm.Get_size()))
f1 = open('/run/user/1000/gvfs/sftp:host=taurus.hrsk.tu-dresden.de,user=s8583916/home/h3/s8583916/home/scratch/s8583916-stefans_ws/mthesis/results/testresult.json')
data = json.load(f1)
print(data)