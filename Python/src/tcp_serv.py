import socket
# generate random floating point values
from random import seed
from random import random
# seed random number generator
import time

# System includes
import numpy as np
import math
import random
import matplotlib.pyplot as plt

seed(1)

def rng():
    value = random()
    return value

def Sinewave(timestep):
    noise = random.uniform(0.01, 0.09)
    x = math.sin(timestep/10) + noise
    return x

def main():
    HOST = '127.0.0.1'  # Standard loopback interface address (localhost)
    PORT = 65432        # Port to listen on (non-privileged ports are > 1023)

    cur_step = 1
    last_step = 0
    max_steps = 500

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        conn, addr = s.accept()
        with conn:
            print('Connected by', addr)
            for j in range(0,max_steps):
                data = str(Sinewave(j)).encode('utf_8')
                conn.sendall(data)
                time.sleep(0.1)
            
if __name__ == "__main__":
    main()