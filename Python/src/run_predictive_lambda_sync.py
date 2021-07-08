from rc import *
import matplotlib.pyplot as plt
import random
import math 

def sm_sweep(a, res_size):
    length = res_size * res_size

    y_i = np.zeros(length)
    y_i[0] = random.uniform(0.1, 0.9)

    for j in range(1,length):
        y_i[j] = a * 2*y_i[j-1] % 1

    y_i_temp = y_i.reshape(res_size,res_size)
    y_i_shaped = np.transpose(y_i_temp)

    return y_i_shaped

def lm_sweep(res_size):
    length = 5000

    y_i = np.zeros(length)
    y_i[0] = random.uniform(0.1, 0.9)
    #a = 3.718 #lambda = 0.343
    #a = 3.9899999999999576 # lambda = 0.541
    #a = 3.72 #0.3526
    a = 3.99
    for j in range(1,length):
        y_i[j] = (a * y_i[j-1]) * (1 -  y_i[j-1])

    return y_i

res_size = 256
lm_data = lm_sweep(res_size)
new_rc = RC(res_size,0.1)
#new_rc.Load_Reservoir_Data('../../datasets/logistic_map_shaped.txt')
new_rc.Load_Data('../../datasets/MackeyGlass_t17.txt')
new_rc.rc_data  = sm_sweep(4.2, res_size)
#new_rc.data = lm_data
new_rc.Generate_Reservoir()
new_rc.Train(1000)
new_rc.Run_Generative(1000)
new_rc.Compute_MSE(1000)

silent_run = True
new_rc.Plots()