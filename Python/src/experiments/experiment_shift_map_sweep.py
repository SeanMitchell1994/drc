import matplotlib.pyplot as plt
import random

# Local imports
import common
from rc import *

def sm_sweep(a, res_size):
    length = res_size * res_size

    y_i = np.zeros(length)
    y_i[0] = random.uniform(0.01, 0.09)
    #y_i[0] = 0.7

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

    while param <= 2:
        new_rc = RC(40,0.35)
        #new_rc.Load_Reservoir_Data('../../datasets/logistic_map_shaped.txt')
        new_rc.rc_data  = sm_sweep(param, res_size)
        new_rc.Load_Data('../../datasets/MackeyGlass_t17.txt')
        new_rc.Generate_Reservoir()
        new_rc.Train(2000)
        new_rc.Run_Predictive(1000)
        new_rc.Compute_MSE(500)
        mse = new_rc.Get_MSE()

        mse_list.append(mse)
        param_list.append(param)
        param = param + iterate

    plt.figure(1).clear()
    plt.plot( param_list,mse_list, linewidth=1 )
    plt.title('Shift Map Parameter Sweep')
    plt.xlabel('a parameter')
    plt.ylabel('MSE')
    plt.show()

if __name__ == "__main__":
    main()