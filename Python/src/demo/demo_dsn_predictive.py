# Local imports
import common
from rc import *
from logger import *

def shift_map(a, res_size, ic):
    length = res_size * res_size

    y_i = np.zeros(length)
    y_i[0] = ic

    for j in range(1,length):
        y_i[j] = a * 2*y_i[j-1] % 1

    y_i_temp = y_i.reshape(res_size,res_size)
    y_i_shaped = np.transpose(y_i_temp)

    return y_i_shaped

def main():

    res_size = 120
    learning_rate = 0.3
    training_length = 1000
    test_length = 1000
    sm_ic = 0.4
    sm_slope = 2

    res_fcn = shift_map(sm_slope, res_size, sm_ic)
    dataset = '../../../datasets/lorenz_x.txt'
    # values = []
    # mse_list = []
    # param_list = []

    new_rc = RC(res_size, learning_rate)
    #new_rc.Load_Reservoir_Data('../../../datasets/logistic_map_shaped.txt')
    new_rc.Load_Reservoir_Function(res_fcn)
    new_rc.Load_Data(dataset)
    new_rc.Generate_Reservoir()
    new_rc.Train(training_length)
    new_rc.Run_Predictive(test_length)
    new_rc.Compute_MSE(test_length)
    #new_rc.Save_Metrics()

    print(new_rc.Get_MSE())
    silent_run = False
    new_rc.Plots(silent_run)

if __name__ == "__main__":
    main()