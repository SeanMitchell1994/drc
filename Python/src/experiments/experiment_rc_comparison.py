# Local imports
import common
from rc import *
from esn import *

import matplotlib.pyplot as plt

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

def main():
    res_size = 40
    learning_rate = 0.3
    param = 2.0
    sm_ic = 0.5
    mse_dsn = 0
    mse_esn = 0
    silent_run = True

    new_rc = RC(res_size,learning_rate)
    #new_rc.Load_Reservoir_Data('../../datasets/logistic_map_shaped.txt')
    new_rc.rc_data  = sm_sweep(param, res_size, sm_ic)
    new_rc.Load_Data('../../datasets/lorenz_x.txt')
    new_rc.Generate_Reservoir()
    new_rc.Train(2000)
    new_rc.Run_Predictive(2000)
    new_rc.Compute_MSE(2000)
    new_rc.Plots(silent_run)
    mse_dsn = new_rc.Get_MSE()

    new_rc2 = ESN(res_size,learning_rate)
    new_rc2.Load_Data('../../datasets/lorenz_x.txt')
    new_rc2.Generate_Reservoir()
    new_rc2.Train(2000)
    new_rc2.Run_Predictive(2000)
    new_rc2.Compute_MSE(2000)
    new_rc2.Plots(silent_run)
    mse_esn = new_rc2.Get_MSE()

    improvement = (mse_esn - mse_dsn) * 100
    print("% improvement: ", improvement)

if __name__ == "__main__":
    main()