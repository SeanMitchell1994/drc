import matplotlib.pyplot as plt
import numpy as np
import random
from multiprocessing import Process, Pool
from numba import jit
import pandas as pd 

# Local imports
import common
from rc import *
from logger import *

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

def run_exp(param):
    sub_iterates = 0
    max_sub_iterates = 100
    res_size = 32
    learning_rate = 0.3
    training_length = 500
    test_length = 5000
    mse_temp = 0
    metrics = [max_sub_iterates, res_size, learning_rate, training_length, test_length]

    while sub_iterates <= max_sub_iterates:
        new_rc = RC(res_size,learning_rate)
        sm_ic = random.uniform(0.001, 8)
        new_rc.Load_Reservoir_Function(sm_sweep(param, res_size, sm_ic))
        #new_rc.Load_Data('../../../datasets/lorenz_x.txt')
        #new_rc.Load_Data('../../datasets/MackeyGlass_t17.txt')
        #new_rc.Load_Data('../../../datasets/chua_x.txt')
        new_rc.Load_Data('../../../datasets/logistic_map_raw.txt')
        new_rc.Generate_Reservoir()
        new_rc.Train(training_length)
        new_rc.Run_Predictive(test_length)
        #new_rc.Run_Generative(test_length)
        new_rc.Compute_MSE(test_length)
        mse_temp = new_rc.Get_MSE() + mse_temp
        sub_iterates = sub_iterates + 1

    mse = mse_temp/max_sub_iterates
    return (param,mse)

def main():
    param = 0
    param_max = 2
    iterate = 0.01
    #sub_iterates = 0
    #max_sub_iterates = 10
    res_size = 8
    learning_rate = 0.3
    training_length = 500
    test_length = 5000
    #mse_temp = 0
    sm_ic = 0.4
    values = []
    mse_list = []
    param_list = []

    logger = Logger()
    print("Starting monte carlo")

    while param <= param_max:
        if not((float(param) % 2).is_integer() or (float(param) % 2 - 0.5).is_integer()):
            values.append(param)
        param = param + iterate

    pool = Pool()
    param_list,mse_list = zip(*pool.map(run_exp, values))

    stddev = np.std(mse_list)
    print("std dev: ", stddev)

    mean = np.mean(mse_list)
    print("mean MSE: ", mean)

    min = np.amin(mse_list)
    print("minimum: ", min)

    z1 = np.polyfit(param_list, mse_list, 9)
    p = np.poly1d(z1)

    df = pd.DataFrame(param_list,mse_list) 
    logger.Save_Data(df, "dsn_sweep")

    print("Done!")

    plt.figure(figsize=(14, 8), dpi=100).clear()
    plt.plot( param_list,mse_list, linewidth=0.5 )
    plt.plot( param_list,p(param_list),'r--', linewidth=1)
    plt.plot(param_list[mse_list.index(min)],mse_list[mse_list.index(min)],'ro') 
    plt.title('Shift Map Parameter Sweep (Lorenz x time series)')
    plt.xlabel('Sweep Parameter (a)')
    plt.ylabel('DSN MSE')
    plt.axvline(x=0.5, color='k', linestyle='--', linewidth=1)
    plt.axvline(x=1.193, color='k', linestyle='--', linewidth=1)
    plt.axvspan(0, 0.5, alpha=0.1, color='green')
    #plt.axvspan(0.5, 4, alpha=0.1, color='red')
    plt.axvspan(0.5, 1.193, alpha=0.1, color='red')
    plt.axvspan(1.193, 2, alpha=0.1, color='blue')

    leg = plt.legend(['DSN MSE','DSN MSE Trendline','Minimum MSE','Shift Map LE Zero-Crossing','Negative Entropy','Positive Entropy'],bbox_to_anchor=(1,0.895), loc="center left", numpoints=1)
    leg.get_frame().set_edgecolor('black')
    side_text = plt.figtext(0.913, 0.5, 
        'Reservoir Size: '+str(res_size)+'\n'
        +'Learning Rate: ' + str(learning_rate) +'\n' 
        +'Iterate: '+str(iterate) +'\n'
        +'Training Length: ' + str(training_length) +'\n' 
        +'Test Length: ' + str(test_length) +'\n'
        +'Shift Map IC: ' + str(sm_ic) +'\n'
        +'Min MSE: ' + "{:e}".format(min) +'\n'
        +'Mean MSE: ' + "{:e}".format(mean) +'\n'
        +'MSE std dev: ' + "{:e}".format(stddev)
        , bbox=dict(facecolor='white'))
    #plt.ylim([40,300])
    #plt.savefig('out.png', bbox_extra_artists=(leg,), bbox_inches='tight', dpi=100)
    logger.Save_Fig(plt,leg,"shift_map_sweep")
    #plt.show()

if __name__ == "__main__":
    main()