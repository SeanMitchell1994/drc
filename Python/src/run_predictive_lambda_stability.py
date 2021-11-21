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

res_size = 120
lm_data = lm_sweep(res_size)
new_rc = RC(res_size,0.3)
#new_rc.Load_Reservoir_Data('../../datasets/logistic_map_shaped.txt')
#new_rc.Load_Data('../../datasets/MackeyGlass_t17.txt')
new_rc.Load_Data('../../datasets/lorenz_x.txt')
new_rc.rc_data  = sm_sweep(2.0, res_size)
#new_rc.data = lm_data
new_rc.Generate_Reservoir()
new_rc.Train(2000)
#new_rc.Run_Generative_Stability(4000,3000)
new_rc.Run_Predictive_Stability(2000, 1000)
new_rc.Compute_MSE(1000)

plt.figure(1)
plt.plot( new_rc.data[new_rc.train_len+1:new_rc.train_len+new_rc.test_len+1], 'r', linewidth=1 )
plt.plot( new_rc.Y.T, '--b', linewidth=1 )
plt.axvline(x=1000, color='k', linestyle='--', linewidth=1)
plt.title('Optimal DSN Open Loop Stability')
plt.xlabel('Step')
plt.ylabel('Value')
plt.legend(['Target signal', 'Free-running predicted signal','Signal Stop'])
plt.show()
#silent_run = True
#new_rc.Plots()