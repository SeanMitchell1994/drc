import matplotlib.pyplot as plt
import numpy as np
import random

# Local imports
import common
from rc import *

def sm_sweep(a, res_size):
    length = res_size * res_size

    y_i = np.zeros(length)
    #y_i[0] = random.uniform(0.01, 0.09)
    y_i[0] = 0.4

    for j in range(1,length):
        y_i[j] = a * 2*y_i[j-1] % 1

    y_i_temp = y_i.reshape(res_size,res_size)
    y_i_shaped = np.transpose(y_i_temp)

    return y_i_shaped

def main():
    param = 0
    iterate = 0.01
    res_size = 40
    mse_list = []
    param_list = []

    while param <= 4:
        new_rc = RC(40,0.3)
        #new_rc.Load_Reservoir_Data('../../datasets/logistic_map_shaped.txt')
        new_rc.rc_data  = sm_sweep(param, res_size)
        new_rc.Load_Data('../../datasets/lorenz_x.txt')
        #new_rc.Load_Data('../../datasets/MackeyGlass_t17.txt')
        new_rc.Generate_Reservoir()
        new_rc.Train(4000)
        new_rc.Run_Predictive(1000)
        #new_rc.Run_Generative(1000)
        new_rc.Compute_MSE(1000)
        mse = new_rc.Get_MSE()

        mse_list.append(mse)
        param_list.append(param)
        param = param + iterate

    stddev = np.std(mse_list)
    print("std dev: ", stddev)

    mean = np.mean(mse_list)
    print("mean MSE: ", mean)

    min = np.amin(mse_list)
    print("minimum: ", min)

    plt.figure(1).clear()
    plt.plot( param_list,mse_list, linewidth=1 )
    plt.title('Shift Map Parameter Sweep')
    plt.xlabel('a parameter')
    plt.ylabel('MSE')
    #plt.ylim([40,300])
    plt.show()

if __name__ == "__main__":
    main()