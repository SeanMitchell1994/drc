import matplotlib.pyplot as plt
import numpy as np
import random

# Local imports
import common
from rc import *

def sm_sweep(a, res_size, ic):
    length = res_size * res_size

    y_i = np.zeros(length)
    #y_i[0] = random.uniform(0.01, 0.09)
    y_i[0] = ic

    for j in range(1,length):
        y_i[j] = a * 2*y_i[j-1] % 1

    y_i_temp = y_i.reshape(res_size,res_size)
    y_i_shaped = np.transpose(y_i_temp)

    return y_i_shaped

    
def lm_sweep(a, res_size, ic):
    length = res_size * res_size

    y_i = np.zeros(length)
    y_i[0] = ic

    for j in range(1,length):
        y_i[j] = (a * y_i[j-1]) * (1 -  y_i[j-1])

    y_i_temp = y_i.reshape(res_size,res_size)
    y_i_shaped = np.transpose(y_i_temp)

    return y_i_shaped

def main():
    param = 3.600
    #param_max = 4
    run_cur = 0
    run_end = 1000
    iterate = 0.01
    res_size = 120
    learning_rate = 0.3
    training_length = 4000
    test_length = 1000
    sm_ic = 0
    mse_list = []
    param_list = []

    while run_cur <= run_end:
        sm_ic = random.uniform(0.001, 0.009)
        new_rc = RC(res_size,learning_rate)
        new_rc.rc_data  = lm_sweep(param, res_size, sm_ic)
        new_rc.Load_Data('../../datasets/lorenz_x.txt')
        new_rc.Generate_Reservoir()
        new_rc.Train(training_length)
        new_rc.Run_Predictive(test_length)
        new_rc.Compute_MSE(test_length)
        mse = new_rc.Get_MSE()

        mse_list.append(mse)
        param_list.append(sm_ic)
        run_cur = run_cur + 1

    stddev = np.std(mse_list)
    print("std dev: ", stddev)

    mean = np.mean(mse_list)
    print("mean MSE: ", mean)

    min = np.amin(mse_list)
    print("minimum: ", min)

    #print(mse_list)
    #z1 = np.polyfit(param_list, mse_list, 1)
    #p = np.poly1d(z1)

    plt.scatter( param_list,mse_list, linewidth=1 )
    #plt.imshow(mse_list, cmap="hot", interpolation="nearest")
    plt.xlabel('SM IC')
    plt.ylabel('DSN MSE')
    plt.show()

if __name__ == "__main__":
    main()