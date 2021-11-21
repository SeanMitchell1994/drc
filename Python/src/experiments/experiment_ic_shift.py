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
    res_size = 120
    learning_rate = 0.3
    param = 2.0
    sm_ic = 0.5
    mse_dsn = 0
    mse_esn = 0
    silent_run = True

    new_rc = RC(res_size,learning_rate)
    new_rc.rc_data  = sm_sweep(param, res_size, sm_ic)
    new_rc.Load_Data('../../datasets/lorenz_x.txt')
    new_rc.Generate_Reservoir()
    new_rc.Train(3000)
    new_rc.Run_Predictive(500)
    new_rc.Compute_MSE(500)
    #new_rc.Plots()
    mse_dsn = new_rc.Get_MSE()

    new_rc2 = ESN(res_size,learning_rate)
    new_rc2.rc_data  = sm_sweep(param, res_size, sm_ic)
    new_rc2.Load_Data('../../datasets/lorenz_x.txt')
    new_rc2.Generate_Reservoir()
    new_rc2.Train(3000)
    #new_rc2.Load_Data('E:\School\Graduate\Research\Code\MATLAB\datasets\lorenz_x_ic_shift.txt')
    new_rc2.Run_Predictive(500)
    new_rc2.Compute_MSE(500)
    #new_rc2.Plots()
    mse_esn = new_rc2.Get_MSE()

    print("MSE (no shift): ", mse_dsn)
    print("MSE (shift): ", mse_esn)
    diff = (abs(mse_esn - mse_dsn)/((mse_esn + mse_dsn)/2) * 100)
    print("mean % difference: ", diff)

    plt.figure(1)
    plt.subplot(121)
    plt.plot( new_rc.data[new_rc.train_len+1:new_rc.train_len+new_rc.test_len+1], 'r', linewidth=2 )
    plt.plot( new_rc.Y.T, '--b', linewidth=2 )
    plt.title('Optimal DSN Lorenz x time series Prediction')
    plt.xlabel('Step')
    plt.ylabel('Value')
    plt.legend(['Target signal', 'Free-running predicted signal'])
    plt.subplot(122)
    plt.plot( new_rc2.data[new_rc2.train_len+1:new_rc2.train_len+new_rc2.test_len+1], 'r', linewidth=2 )
    plt.plot( new_rc2.Y.T, '--g' , linewidth=2)
    plt.title('ESN Lorenz x time series Prediction')
    plt.xlabel('Step')
    plt.ylabel('Value')
    plt.legend(['Target signal', 'Free-running predicted signal'])
    plt.show()

if __name__ == "__main__":
    main()