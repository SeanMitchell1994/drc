import numpy as np
from scipy import linalg 
# numpy.linalg is also an option for even fewer dependencies

class Reservoir:
    def __init__(self, reservoir_size):
        # === Runtime environment parameters ===
        self.reservoir_size = reservoir_size    # Size of reservoir
        self.in_size = 1                        # Dimension(s) of input
        self.out_size = self.in_size            # Dimension(s) of output

        # === Data structures ===
        self.rc_data = 0                        # Dataset for the reservoir core

        # === Member variables ===
        self.Win = 0                            # Weights mapping the input data -> reservoir
        self.W = 0                              # Weights mapping the interal connections of the reservoir
        self.rhoW = 0                           # Spectral radius

    def Load_Reservoir_Data(self, data):
        print('Loading reservoir core data...', end='')
        self.rc_data = np.loadtxt(data)
        print('Done!')

    def Load_Reservoir_Function(self, fcn):
        #print('Loading reservoir core function...', end='')
        self.rc_data = fcn
        #print('Done!')

    def Generate_Reservoir(self):
        # generate the ESN reservoir
        #print('Generating reservoir...',end='')

        #print('Generating reservoir')
        np.random.seed(42)
        self.Win = (np.random.rand(self.reservoir_size,1 + self.in_size) - 0.5) * 1
        self.W = np.random.rand(self.reservoir_size, self.reservoir_size) - 0.5
        rcd_t = np.transpose(self.rc_data)
        self.W = np.dot(self.W, rcd_t)
        #print('Done!')

        # normalizing and setting spectral radius (correct, slow):
        #print('Computing spectral radius...',end='')
        self.rhoW = max(abs(linalg.eig(self.W)[0]))
        self.W *= 1.25 / self.rhoW
        #print('Done!')