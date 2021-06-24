from rc import *
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d
import scipy.interpolate as interp
import random

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

def bifurt(r, a):
    length = 1.0/a + 1.0

    while r < 4:
        y_i = np.zeros(int(length))
        y_i[0] = random.uniform(0, 1)

        for i in range(1,int(length)):
            y_i[i] = (r * y_i[i-1]) * (1 -  y_i[i-1])
        r = r + a

        np.savetxt('../../datasets/orbit_2.txt', y_i)

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
        new_rc.Compute_MSE(500)
        mse = new_rc.Get_MSE()

        mse_list.append(mse)
        param_list.append(param)
        param = param + iterate

    bifurt(3,iterate)
    z = np.loadtxt('../../datasets/orbit_2.txt')

    plotx,ploty, = np.meshgrid(np.linspace(np.min(z),np.max(z),10),\
                           np.linspace(np.min(param_list),np.max(param_list),10))
    plotz = interp.griddata((z,param_list),mse_list,(plotx,ploty),method='linear')
    scale_factor = 10000
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(plotx,ploty,plotz*scale_factor,cmap='YlGnBu_r',vmin=np.nanmin(z), vmax=np.nanmax(z)) 
    ax.set_xlabel('orbit diagram')
    ax.set_ylabel('r')
    ax.set_zlabel('MSE (scaled)')
    plt.show()

if __name__ == "__main__":
    main()