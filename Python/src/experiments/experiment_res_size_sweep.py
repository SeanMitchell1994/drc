import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Process, Pool

# Local imports
import common
from rc import *

def lm_sweep(a, res_size):
    length = res_size * res_size

    y_i = np.zeros(length)
    y_i[0] = 0.7920

    for j in range(1,length):
        y_i[j] = (a * y_i[j-1]) * (1 -  y_i[j-1])

    y_i_temp = y_i.reshape(res_size,res_size)
    y_i_shaped = np.transpose(y_i_temp)

    return y_i_shaped


def main():
    param = 3.5
    iterate = 2
    res_size = 40
    mse_list = []
    param_list = []
    while res_size <= 1600:
        new_rc = RC(res_size,0.35)
        #new_rc.Load_Reservoir_Data('../../datasets/logistic_map_shaped.txt')
        new_rc.rc_data  = lm_sweep(param,res_size)
        new_rc.Load_Data('../../datasets/MackeyGlass_t17.txt')
        new_rc.Generate_Reservoir()
        new_rc.Train(2000)
        new_rc.Run_Predictive(1000)
        new_rc.Compute_MSE(500)
        mse = new_rc.Get_MSE()

        mse_list.append(mse)
        param_list.append(res_size)
        res_size = res_size * iterate
    
    plt.figure(1).clear()
    plt.plot( param_list,mse_list, linewidth=1 )
    plt.title('Logistics Map Parameter Sweep')
    plt.xlabel('Neuron Count')
    plt.ylabel('MSE')

    # values for fixed points from https://mathworld.wolfram.com/LogisticMap.html
    #xcoords = [3, 3.82842712, 3.44948974, 3.73817237, 3.62655316, 3.54409035, 3.70164076, 3.56440726, 3.939200000003982, 3.774900000003636,3.685500000003447, 3.872300000003841,3.866200000003828,3.859600000003814,3.801700000003692,3.755000000003594,3.751400000003586,3.711700000003502,3.724800000003530,3.640300000003351, 3.640300000003351,3.605200000003277]
    #even_fp = [3.540650000010093, 3.568640000010277, 3.571900000010298]
    #odd_fp = [3, 3.436550000009411, 3.5635300000102433, 3.569950000010285, 3.570210000010287, 3.570730000010290]
    #for xc in even_fp:
    #    plt.axvline(x=xc, color='r', linestyle='--', linewidth=1)
    #for yc in odd_fp:
    #    plt.axvline(x=yc, color='g', linestyle='--', linewidth=1)
    #plt.legend(['MSE','Even Fixed Points', 'Odd Fixed Points'],loc="lower left")
    plt.show()

if __name__ == "__main__":
    main()