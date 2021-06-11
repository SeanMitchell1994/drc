from rc import *
import matplotlib.pyplot as plt


def tm_sweep(x):
    length = 1600
    mu = 2
    y_i = np.zeros(length)
    y_i[0] = x; 

    for i in range(1,length):
        if (y_i[i - 1] < 0.5):
            y_i[i] = mu * y_i[i - 1]
        elif (0.5 <= y_i[i - 1]):
            y_i[i] = mu * 1 - y_i[i - 1]

    y_i_temp = y_i.reshape(40,40)
    y_i_shaped = np.transpose(y_i_temp)
    #plt.figure(1).clear()
    #plt.plot(y_i_shaped, linewidth=1 )
    #plt.show()
    return y_i_shaped



def main():
    param = 0.001
    iterate = 0.01
    mse_list = []
    param_list = []
    #tm_sweep()
    while param <= 2:
        new_rc = RC(40,0.35)
        #new_rc.Load_Reservoir_Data('../../datasets/logistic_map_shaped.txt')
        new_rc.rc_data  = tm_sweep(param)
        new_rc.Load_Data('../../datasets/MackeyGlass_t17.txt')
        new_rc.Generate_Reservoir()
        new_rc.Train(2000)
        new_rc.Run_Predictive(1000)
        new_rc.Compute_MSE(500)
        mse = new_rc.Get_MSE()

        mse_list.append(mse)
        param_list.append(param)
        param = param + iterate

    plt.figure(1).clear()
    plt.plot( param_list,mse_list, linewidth=1 )
    plt.title('Tent Map Parameter Sweep')
    plt.xlabel('x')
    plt.ylabel('MSE')

    xcoords = [0,1/9,2/9,3/9,6/9,7/9,8/9]#, 1.41]#, 1.8393]
    for xc in xcoords:
        plt.axvline(x=xc, color='k', linestyle='--', linewidth=1)
    plt.legend(['MSE','Fixed Points'],loc="upper left")
    plt.show()

if __name__ == "__main__":
    main()