# System imports
import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Process, Pool
#from scipy.signal import savgol_filter
import random

# Local imports
import common
from rc import *

def lm_sweep(a, res_size, ic):
    length = res_size * res_size

    y_i = np.zeros(length)
    y_i[0] = ic

    for j in range(1,length):
        y_i[j] = (a * y_i[j-1]) * (1 -  y_i[j-1])

    y_i_temp = y_i.reshape(res_size,res_size)
    y_i_shaped = np.transpose(y_i_temp)

    return y_i_shaped

def run_exp(param):
        res_size = 40
        #param = 3.82842712
        #param = 3.44948974

        new_rc = RC(res_size,0.3)
        ic = random.uniform(0.001, 1)
        new_rc.rc_data  = lm_sweep(param, res_size, ic)
        new_rc.Load_Data('../../datasets/lorenz_x.txt')
        new_rc.Generate_Reservoir()
        new_rc.Train(4000)
        new_rc.Run_Predictive(1000)
        new_rc.Compute_MSE(1000)
        mse = new_rc.Get_MSE()
        return (ic,mse)

def main():
    param = 0
    param_max = 225
    iterate = 0.01
    res_size = 120
    values = []
    while param <= param_max:
        values.append(param)
        param = param + iterate

    pool = Pool(processes=8)
    param_list,mse_list = zip(*pool.map(run_exp, values))

    stddev = np.std(mse_list)
    print("std dev: ", stddev)

    mean = np.mean(mse_list)
    print("mean MSE: ", mean)

    min = np.amin(mse_list)
    print("minimum: ", min)

    plt.figure(figsize=(10, 8), dpi=100).clear()
    plt.scatter( param_list,mse_list, marker='o', s=2 )
    #plt.plot( param_list,p(param_list),'r--', linewidth=1)
    #plt.plot(param_list[mse_list.index(min)],mse_list[mse_list.index(min)],'ro') 
    plt.title('Logisitics Map Mean Point')
    plt.xlabel('Iteration (a)')
    plt.ylabel('Point MSE')
    #plt.axvline(x=0.5, color='k', linestyle='--', linewidth=1)
    #plt.axvspan(0, 0.5, alpha=0.1, color='green')
    #plt.axvspan(0.5, 4, alpha=0.1, color='red')
    plt.show()

if __name__ == "__main__":
    main()