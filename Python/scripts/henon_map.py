# Filename: henon_map.py
# Author: Sean Mitchell
# short script to generate a text file with the henon map

import numpy as np

length = 10000

x_i = np.zeros(length)
y_i = np.zeros(length)

x_i[0] = 0
y_i[0] = 0

beta = 0.3
alpha = 1.4

for i in range(1,length):
    x_i[i] = 1 - alpha * x_i[i-1]**2 + y_i[i-1];
    y_i[i] = beta * x_i[i - 1];

np.savetxt('../../datasets/data/henon_map_x.txt', x_i)
np.savetxt('../../datasets/data/henon_map_y.txt', y_i)