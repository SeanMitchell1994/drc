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

def main():
    param = 0
    param_max = 4
    iterate = 0.01
    res_size = 120
    learning_rate = 0.3
    training_length = 4000
    test_length = 1000
    sm_ic = 0.4
    mse_list = []
    param_list = []

    while param <= param_max:
        new_rc = RC(res_size,learning_rate)
        #new_rc.Load_Reservoir_Data('../../datasets/logistic_map_shaped.txt')
        new_rc.rc_data  = sm_sweep(param, res_size, sm_ic)
        new_rc.Load_Data('../../datasets/lorenz_x.txt')
        #new_rc.Load_Data('../../datasets/MackeyGlass_t17.txt')
        #new_rc.Load_Data('../../datasets/logistic_map_raw.txt')
        new_rc.Generate_Reservoir()
        new_rc.Train(training_length)
        new_rc.Run_Predictive(test_length)
        #new_rc.Run_Generative(1000)
        new_rc.Compute_MSE(test_length)
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

    z1 = np.polyfit(param_list, mse_list, 1)
    p = np.poly1d(z1)

    plt.figure(figsize=(10, 8), dpi=100).clear()
    plt.plot( param_list,mse_list, linewidth=1 )
    plt.plot( param_list,p(param_list),'r--', linewidth=1)
    plt.plot(param_list[mse_list.index(min)],mse_list[mse_list.index(min)],'ro') 
    plt.title('Shift Map Parameter Sweep (Lorenz x time series)')
    plt.xlabel('Sweep Parameter (a)')
    plt.ylabel('DSN MSE')
    plt.axvline(x=0.5, color='k', linestyle='--', linewidth=1)
    plt.axvspan(0, 0.5, alpha=0.1, color='green')
    plt.axvspan(0.5, 4, alpha=0.1, color='red')

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
    plt.show()
if __name__ == "__main__":
    main()