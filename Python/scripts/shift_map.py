import numpy as np
import random

def sm_sweep(a, res_size):
    length = res_size * res_size

    y_i = np.zeros(length)
    #y_i[0] = random.uniform(0.1, 0.9)
    y_i[0] = 0.4

    for j in range(1,length):
        y_i[j] = a * 2*y_i[j-1] % 1

    y_i_temp = y_i.reshape(res_size,res_size)
    y_i_shaped = np.transpose(y_i_temp)

    np.savetxt('../../datasets/shift_map_raw.txt', y_i)
    np.savetxt('../../datasets/shift_map_shaped.txt', y_i_shaped)
    return y_i_shaped


def main():
    param = 24
    iterate = 0.001
    res_size = 40
    ic = 0.001

    sm_sweep(param, res_size)

if __name__ == "__main__":
    main()