# Local imports
import common
from rc import *

# System includes
import numpy as np
import math
import random
import time
import matplotlib.pyplot as plt
import socket

b = []

def run_rc(rc, a, steps):
    # Generate sinewave for real time testing
    a = float(a.decode('utf_8'))
    b.append(a)

    # Put the data into the RC to test
    rc.data = b
    rc.train_len = 0
    #rc.Train(1 + steps)

    # Test
    #if (steps < 25):
    #    rc.Run_Predictive(steps)
    #else:
    #    rc.Run_Generative(steps)
    rc.Run_Predictive(steps)
    rc.Compute_MSE(1 + steps)
    mse = rc.Get_MSE()
    return mse

def main():
    HOST = '127.0.0.1'  # The server's hostname or IP address
    PORT = 65432        # The port used by the server

    steps = 0
    max_steps = 250

    avg_mse = 0

    # Setting up the RC
    new_rc = RC(40,0.3)
    new_rc.Load_Reservoir_Data('../../datasets/shift_map_shaped.txt')
    new_rc.Load_Data('../../datasets/MackeyGlass_t17.txt')
    new_rc.Generate_Reservoir()
    new_rc.Train(1000)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        while True:
            if (steps > max_steps - 1): break
            #s.sendall(b'Hello, world\n')
            #time.sleep(1)
            data = s.recv(128)
            avg_mse = avg_mse + run_rc(new_rc, data, steps)
            steps = steps + 1

    true_avg = avg_mse/max_steps
    print("True average mse: ", true_avg)
    new_rc.Plots()

if __name__ == "__main__":
    main()