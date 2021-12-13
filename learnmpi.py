from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# do some calculation
np.random.seed(4*rank)
a = np.random.randint(comm.Get_size(), size=4)
data = comm.gather(a, root=0)
if rank==0:
    print('Hello World')
    print('I gathered this: ', data)
    print(type(data))