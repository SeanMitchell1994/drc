# System imports
import matplotlib.pyplot as plt
import numpy as np
import random
import math 
from multiprocessing import Process, Pool

# Local imports
import common
from rc import *

def sm_sweep(a, res_size):
    length = res_size * res_size

    y_i = np.zeros(length)
    y_i[0] = random.uniform(0.1, 0.9)

    for j in range(1,length):
        y_i[j] = a * 2*y_i[j-1] % 1

    y_i_temp = y_i.reshape(res_size,res_size)
    y_i_shaped = np.transpose(y_i_temp)

    return y_i_shaped

def lm_sweep(a, res_size,length):
    #length = 10000

    y_i = np.zeros(length)
    y_i[0] = random.uniform(0.1, 0.9)
    for j in range(1,length):
        y_i[j] = (a * y_i[j-1]) * (1 -  y_i[j-1])

    return y_i

def Lyapunov_Exponent_sm(iterate):
    result = []
    lambdas = []
    r_list = []

    xmin = 0
    xmax = 2

    rvalues = np.arange(xmin, xmax, iterate)

    for r in rvalues:
        x = 0.1
        result = []
        for t in range(1000):
            x = r*2* (x % 1)
            result.append(np.log(abs(2*r)))
        lambdas.append(np.mean(result))
        r_list.append(r)

    return lambdas, r_list

def Lyapunov_Exponent_lm(iterate):
    result = []
    lambdas = []
    r_list = []

    xmin = 2
    xmax = 4

    rvalues = np.arange(xmin, xmax, iterate)

    for r in rvalues:
        x = 0.1
        result = []
        for t in range(1000):
            x = r * x * (1 - x)
            result.append(np.log(abs(r - 2*r*x)))
        lambdas.append(np.mean(result))
        r_list.append(r)

    return lambdas, r_list

def Match_Parameter(value, r_list, lya_list):
    allowed_error = 0.0001
    index = 0
    lya_exp = 0
    for i in range(len(r_list)):
        if abs(r_list[i] - value) <= allowed_error:
            index = i
            lya_exp = lya_list[index]
    return index, lya_exp

def Match_Lya_Exp(lya_exp, r_list, lya_list):
    allowed_error = 0.01
    index = 0
    #lya_exp = 0
    r_val = 0
    for i in range(len(r_list)):
        if abs(lya_list[i] - lya_exp) <= allowed_error:
            index = i
            r_val = r_list[index]
    return index, r_val

def main():
    # Variables we need
    param = 0
    iterate = 0.01
    res_size = 256
    r = 3.448
    learning_rate = 0.3
    #lambda_01 = 0
    #lambda_02 = 0
    length = 10000

    # Strucures we need
    mse_list = []
    param_list = []
    lambdas_sm = []
    r_list_sm = []
    lambdas_lm = []
    r_list_lm = []

    # Setting up the data and finding our lyapunov exponent match
    lm_data = lm_sweep(r, res_size, length)
    lambdas_lm,r_list_lm = Lyapunov_Exponent_lm(0.0001)
    lambdas_sm,r_list_sm = Lyapunov_Exponent_sm(0.0001)

    match_index, match_lya = Match_Parameter(r, r_list_lm, lambdas_lm)
    match_index_2, match_r = Match_Lya_Exp(match_lya, r_list_sm, lambdas_sm)

    # Sweeping the shfit map
    while param <= 3:
        new_rc = RC(res_size,learning_rate)
        #new_rc.Load_Reservoir_Data('../../datasets/logistic_map_shaped.txt')
        new_rc.rc_data  = sm_sweep(param, res_size)
        new_rc.data = lm_data
        #new_rc.Load_Data('../../datasets/logistic_map_raw.txt')
        new_rc.Generate_Reservoir()
        new_rc.Train(500)
        new_rc.Run_Generative(2000)
        new_rc.Compute_MSE(2000)
        mse = new_rc.Get_MSE()

        mse_list.append(mse)
        param_list.append(param)
        param = param + iterate

    print(mse_list.index(min(mse_list)))
    print(mse_list[mse_list.index(min(mse_list))])

    stddev = np.std(mse_list)
    print("std dev: ", stddev)

    # Plotting
    plt.figure(1).clear()
    plt.plot( param_list,mse_list, linewidth=1 )
    plt.plot(param_list[mse_list.index(min(mse_list))],mse_list[mse_list.index(min(mse_list))],'ro') 
    plt.title('Shift Map Parameter Sweep')
    plt.xlabel('Sweep parameter (a)')
    plt.ylabel('MSE')
    plt.axvline(x=r_list_sm[match_index_2], color='k', linestyle='--', linewidth=1)
    plt.legend(['MSE','Minimum','Lambda_01 = Lambda_02'],loc="upper left")
    plt.show()

if __name__ == "__main__":
    main()