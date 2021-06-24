from rc import *
import matplotlib.pyplot as plt
import numpy as np
#from multiprocessing import Process, Pool

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
    param = 3.8
    iterate = 0.001
    res_size = 40
    ic = 0.001
    mse_list = []
    param_list = []
    while ic <= 1:
        new_rc = RC(res_size,0.35)
        #new_rc.Load_Reservoir_Data('../../datasets/logistic_map_shaped.txt')
        new_rc.rc_data  = lm_sweep(param,res_size,ic)
        new_rc.Load_Data('../../datasets/MackeyGlass_t17.txt')
        new_rc.Generate_Reservoir()
        new_rc.Train(2000)
        new_rc.Run_Predictive(1000)
        new_rc.Compute_MSE(500)
        mse = new_rc.Get_MSE()

        mse_list.append(mse)
        param_list.append(ic)
        ic = ic + iterate
    
    plt.figure(1).clear()
    plt.plot( param_list,mse_list, linewidth=1 )
    plt.title('Logistics Map Parameter Sweep')
    plt.xlabel('Neuron Count')
    plt.ylabel('MSE')
    plt.show()

if __name__ == "__main__":
    main()