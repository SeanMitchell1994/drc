from rc import *
import matplotlib.pyplot as plt


def lm_sweep(a):
    length = 1600
    #a = 3.900142000000020

    y_i = np.zeros(length)
    y_i[0] = 0.7920

    for j in range(1,length):
        y_i[j] = (a * y_i[j-1]) * (1 -  y_i[j-1])

    y_i_temp = y_i.reshape(40,40)
    y_i_shaped = np.transpose(y_i_temp)

    return y_i_shaped

def main():
    param = 3
    iterate = 0.001
    mse_list = []
    param_list = []
    while param <= 4.0:
        new_rc = RC(40,0.35)
        #new_rc.Load_Reservoir_Data('../../datasets/logistic_map_shaped.txt')
        new_rc.rc_data  = lm_sweep(param)
        new_rc.Load_Data('../../datasets/MackeyGlass_t17.txt')
        new_rc.Generate_Reservoir()
        new_rc.Train(2000)
        new_rc.Run_Predictive(1000)
        mse = new_rc.Get_MSE(500)

        mse_list.append(mse)
        param_list.append(param)
        param = param + iterate

    plt.figure(1).clear()
    plt.plot( param_list,mse_list, linewidth=1 )
    plt.title('Logistics Map Parameter Sweep')
    plt.xlabel('a')
    plt.ylabel('MSE')

    # values for fixed points from https://mathworld.wolfram.com/LogisticMap.html
    xcoords = [3, 3.82842712, 3.44948974, 3.73817237, 3.62655316, 3.54409035, 3.70164076, 3.56440726]
    for xc in xcoords:
        plt.axvline(x=xc, color='k', linestyle='--', linewidth=1)
    plt.legend(['MSE','Fixed Points'],loc="lower left")
    plt.show()

if __name__ == "__main__":
    main()