# Filename: logistic_map.py
# Author: Sean Mitchell
# short script to generate a text file with the logistics map

import numpy as np

length = 10000
a = 3.95

y_i = np.zeros(length)
y_i[0] = 0.7920

for j in range(1,length):
    y_i[j] = (a * y_i[j-1]) * (1 -  y_i[j-1])

#y_i_temp = y_i.reshape(120,120)
#y_i_shaped = np.transpose(y_i_temp)

np.savetxt('../../../datasets/logistic_map_raw.txt', y_i)
#np.savetxt('../../datasets/logistic_map_shaped.txt', y_i_shaped)