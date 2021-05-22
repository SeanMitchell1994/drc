# Filename: waveform_sim.py
# Author: Sean Mitchell
# short script to generate a real time waveform

import math
import random
import matplotlib.pyplot as plt

def Sinewave(timestep):
    x = math.sin(timestep) + random.randint(-3,3) + random.randint(-2,2) + random.randint(-1,1)
    return x

def main():
    b = []
    for i in range(1,100):
        a = Sinewave(i)
        b.append(a)

    plt.plot(b)
    plt.show()

if __name__ == "__main__":
    main()
