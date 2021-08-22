import matplotlib.pyplot as plt
import math
import random

# Local imports
import common
from rc import *

def sm_sweep(a, res_size):
    length = res_size * res_size

    y_i = np.zeros(length)
    #y_i[0] = random.uniform(0.01, 0.09)
    y_i[0] = 0.2

    for j in range(1,length):
        y_i[j] = a * 2*y_i[j-1] % 1

    y_i_temp = y_i.reshape(res_size,res_size)
    y_i_shaped = np.transpose(y_i_temp)

    return y_i_shaped

def Sinewave(length):
    y_i = np.zeros(length)
    y_i[0] = random.uniform(0.01, 0.09)
    a = random.uniform(0.01, 1)
    delay = random.randint(-5,5)
    noise = random.randint(-10,10)
    multipath = 3

    for j in range(1,length):
        for i in range(1,multipath):
            y_i[j] = y_i[j] + (a*math.sin(j/100 - delay) + noise)

    return y_i

def main():

    # simulation parameters
    training_length = 5
    mse_list = []
    param_list = []
    increment = 1
    sub_runs = 100
    runs = 15

    for i in range(runs):
        avg_mse = 0
        
        for j in range(sub_runs):

            # RC setup
            param = 2.2
            iterate = 0.01
            res_size = 40

            new_rc = RC(40,0.3)
            new_rc.rc_data  = sm_sweep(param, res_size)
            new_rc.data = Sinewave(10000)
            new_rc.Generate_Reservoir()

            # RC training
            for k in range(1,training_length):
                new_rc.data = Sinewave(10000)
                new_rc.Train(5000)

            # RC run
            new_rc.data = Sinewave(10000)
            #new_rc.Run_Generative(1000)
            new_rc.Run_Predictive(500)
            new_rc.Compute_MSE(500)
            mse = new_rc.Get_MSE()
            avg_mse = avg_mse + mse

        # Metrics
        avg_mse = avg_mse/runs
        mse_list.append(avg_mse)
        param_list.append(training_length)
        training_length = training_length + increment

    # Trendline computation
    z1 = np.polyfit(param_list, mse_list, 1)
    p = np.poly1d(z1)

    # Plotting
    plt.figure(1).clear()
    plt.plot( param_list,mse_list, linewidth=1 )            # Data
    plt.plot( param_list,p(param_list),'r--', linewidth=1)  # Trendline
    plt.title('Training Length Sweep')
    plt.xlabel('Training Length')
    plt.ylabel('MSE')
    plt.show()

if __name__ == "__main__":
    main()