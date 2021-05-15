# New shit
#k = [0:1599];                % length of time series
#y_i = zeros(1,length(k));   % Pre-allocation
#y_i(1) = 0.7920;  % initial condition

# Solving the iterated solution
#for i = 2:length(y_i)
#    y_i(i) = (3.900142000000020 * y_i(i-1)) * (1 - y_i(i-1));
#end
# y_i = y_i';
#d2 = y_i;

import numpy as np

length = 1600
y_i = np.zeros(length)
y_i[0] = 0.7920

for j in range(1,length):
    y_i[j] = (3.900142000000020 * y_i[j-1]) * (1 -  y_i[j-1])

a = y_i.reshape(40,40)
y_i_shaped = np.transpose(a)

np.savetxt('../data/logistic_map_raw.txt', y_i)
np.savetxt('../data/logistic_map_shaped.txt', y_i_shaped)