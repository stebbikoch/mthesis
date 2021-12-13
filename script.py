#!/bin/python3
import numpy as np
from Reproduce import main

if __name__=='__main__':
    exponent = np.arange(16)
    q_values = 10 ** (-exponent / 3)
    q_values = [0.001, 0.1, 1]
    r_0 = [10]#[400, 200, 100, 50, 25, 10]
    main(q_values, r_0, '1d_ring_1000','results/ring_directed_100x',
         np.array([1000, 1, 1]),
         directed=True)
