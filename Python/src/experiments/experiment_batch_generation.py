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
    y_i[0] = 0.3

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
    multipath = 7

    for j in range(1,length):
        for i in range(1,multipath):
            y_i[j] = y_i[j] + (a*math.sin(j/100 - delay) + noise)

    #print(y_i)
    #np.savetxt('E:\School\Graduate\Research\Code\MATLAB\datasets/sinewave.txt', y_i)

    return y_i


def main():
    param = 4.2
    iterate = 0.01
    res_size = 40
    #Sinewave(10001)
    new_rc = RC(40,0.3)
    #new_rc.Load_Reservoir_Data('../../datasets/logistic_map_shaped.txt')
    new_rc.rc_data  = sm_sweep(param, res_size)
    #new_rc.data  = Sinewave(2000)
    new_rc.data = Sinewave(10000)
    new_rc.Generate_Reservoir()

    for i in range(1,10):
        new_rc.data = Sinewave(10000)
        new_rc.Train(5000)

    new_rc.data = Sinewave(10000)
    new_rc.Train(1000)
    new_rc.Run_Generative(1000)
    new_rc.Compute_MSE(1000)
    mse = new_rc.Get_MSE()

    new_rc.Plots()

if __name__ == "__main__":
    main()