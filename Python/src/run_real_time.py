from rc import *
import numpy as np

import math
import random
import time
import matplotlib.pyplot as plt

def Sinewave(timestep):
    x = math.sin(timestep/10)# + random.randint(-3,3) + random.randint(-2,2) + random.randint(-1,1)
    return x

def main():
    # Setting up the RC
    new_rc = RC(40,0.35)
    new_rc.Load_Reservoir_Data('../../datasets/logistic_map_shaped.txt')
    new_rc.Load_Data('../../datasets/MackeyGlass_t17.txt')
    new_rc.Generate_Reservoir()
    new_rc.Train(2000)

    # Main loop
    b = []
    d2 = np.loadtxt('../../datasets/MackeyGlass_t17.txt')

    for j in range(0,300):

        # Generate sinewave for real time testing
        a = Sinewave(j)
        b.append(a)

        # Put the data into the RC to test
        new_rc.data = b
        new_rc.train_len = 0

        # Test
        new_rc.Run_Predictive(j)
        new_rc.Get_MSE(500)

        # plot some signals
        plt.figure(1).clear()
        plt.plot( new_rc.data[new_rc.train_len+1:new_rc.train_len+new_rc.test_len+1], 'g' )
        plt.plot( new_rc.Y.T, 'b' )
        plt.title('Target and generated signals $y(n)$ starting at $n=0$')
        plt.legend(['Target signal', 'Free-running predicted signal'])
    
        # Live plotting
        plt.pause(0.01)

    plt.show()

if __name__ == "__main__":
    main()