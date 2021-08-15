import matplotlib.pyplot as plt

# Local imports
import common
from rc import *

def tm_sweep(x):
    length = 1600
    mu = x
    y_i = np.zeros(length)
    y_i[0] = 1; 

    for i in range(1,length):
        if (y_i[i - 1] < 0.5):
            y_i[i] = mu * y_i[i - 1]
        elif (0.5 <= y_i[i - 1]):
            y_i[i] = mu * 1 - y_i[i - 1]

    y_i_temp = y_i.reshape(40,40)
    y_i_shaped = np.transpose(y_i_temp)

    return y_i_shaped

def main():
    param = 1
    iterate = 0.001
    mse_list = []
    param_list = []

    while param <= 2:
        new_rc = RC(40,0.3)
        #new_rc.Load_Reservoir_Data('../../datasets/logistic_map_shaped.txt')
        new_rc.rc_data  = tm_sweep(param)
        new_rc.Load_Data('../../datasets/MackeyGlass_t17.txt')
        new_rc.Generate_Reservoir()
        new_rc.Train(4000)
        #new_rc.Run_Predictive(1000)
        new_rc.Run_Generative(1000)
        new_rc.Compute_MSE(1000)
        mse = new_rc.Get_MSE()

        mse_list.append(mse)
        param_list.append(param)
        param = param + iterate


    stddev = np.std(mse_list)
    print("std dev: ", stddev)

    mean = np.mean(mse_list)
    print("mean MSE: ", mean)

    plt.figure(1).clear()
    plt.plot( param_list,mse_list, linewidth=1 )
    plt.title('Tent Map Parameter Sweep')
    plt.xlabel('mu')
    plt.ylabel('MSE')

    #xcoords = [0,1/9,2/9,3/9,6/9,7/9,8/9]
    #for xc in xcoords:
    #    plt.axvline(x=xc, color='k', linestyle='--', linewidth=1)
    #plt.legend(['MSE','Fixed Points'],loc="upper left")
    plt.show()

if __name__ == "__main__":
    main()