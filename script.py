#!/bin/sh
import numpy as np
from Reproduce import main
import time
import sys
print(sys.executable)
# build d_function thing
#x = integer_inequality(np.array([1000, 1, 1]))
#x.all_numbers(400)
#x.save_to_json('1d_ring_1000')
# do the rest
start = time.time()
exponent = np.arange(16)
q_values = 10 ** (-exponent / 3)
#q_values = [0.01, 0.1, 1]
r_0 = [400, 200, 100, 50, 25, 10]
main(q_values, r_0, '1d_ring_1000','results/ring_directed_1000_100x',
     100,np.array([1000, 1, 1]),
     parallel=True, directed=True)
end = time.time()
print('zeit ', end-start)
print('okay')