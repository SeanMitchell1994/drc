import os
import errno
import numpy as np
import random
import matplotlib.pyplot as plt
import math
from scipy import linalg 
import datetime
import time
# numpy.linalg is also an option for even fewer dependencies

class RC:
    def __init__(self, reservoir_size, leak):
        #print("Creating instance of RC...", end='')

        # === Runtime environment parameters ===
        self.reservoir_size = reservoir_size    # Size of reservoir
        self.leak = leak                        # leaking/learning rate
        self.train_len = 0                      # Length of training phase
        self.test_len = 0                       # Length of testing phase
        self.init_len = 100                     # Index delay between end of training and start of testing
        self.in_size = 1                        # Dimension(s) of input
        self.out_size = self.in_size            # Dimension(s) of output

        # === Data structures ===
        self.data = 0
        self.rc_data = 0                        # Dataset for the reservoir core
        self.test_data = 0                      # Data we train against
        self.training_data = 0                  # Data we test against

        # === Member variables ===
        self.Win = 0                            # Weights mapping the input data -> reservoir
        self.W = 0                              # Weights mapping the interal connections of the reservoir
        self.rhoW = 0                           # Spectral radius
        self.X = 0                              # Weights of the readout layer
        self.x = 0                              # State equation of the reservoir
        self.Yt = 0                             # Feedforward input data matrix for use in linear regression

        # === Error variables ===
        self.error_len = 0                      # What subset of data do we check against for our error rate?
        self.mse = 0                            # Our actual error rate

        #print("Done!")

        self.Output_Init()

    def Load_Reservoir_Data(self, data):
        #print('Loading reservoir core data...', end='')
        self.rc_data = np.loadtxt(data)
        #print('Done!')

    def Load_Reservoir_Function(self, fcn):
        self.rc_data = fcn

    def Load_Data(self, data):
        #print('Loading data...', end='')
        self.data = np.loadtxt(data)
        #print('Done!')

    def Load_Test_Data(self, data):
        #print('Loading test data...', end='')
        self.test_data = np.loadtxt(data)
        #print('Done!')

    def Load_Training_Data(self, data):
        #print('Loading training data...', end='')
        self.training_data = np.loadtxt(data)
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

    def Train(self, train_length):
        #print('Training the reservoir computer...',end='')
        self.train_len = train_length

        # allocated memory for the design (collected states) matrix
        self.X = np.zeros((1 + self.in_size + self.reservoir_size, self.train_len - self.init_len))
        # set the corresponding target matrix directly
        self.Yt = self.data[None,self.init_len+1:self.train_len+1] 

        # run the reservoir with the data and collect X
        self.x = np.zeros((self.reservoir_size,1))
        for t in range(self.train_len):
            u = self.data[t]
            self.x = (1-self.leak)*self.x + self.leak*np.tanh( np.dot( self.Win, np.vstack((1,u)) ) + np.dot( self.W, self.x ) )
            if t >= self.init_len:
                self.X[:,t-self.init_len] = np.vstack((1,u,self.x))[:,0]
            
        # train the output by ridge regression
        reg = 1e-8  # regularization coefficient
        # direct equations from texts:
        #X_T = X.T
        #Wout = np.dot( np.dot(Yt,X_T), linalg.inv( np.dot(X,X_T) + \
        #    reg*np.eye(1+inSize+resSize) ) )
        # using scipy.linalg.solve:
        self.Wout = linalg.solve( np.dot(self.X,self.X.T) + reg*np.eye(1+self.in_size + self.reservoir_size), 
            np.dot(self.X,self.Yt.T) ).T
        #print('Done!')

    def Run_Generative(self, test_len):
        print('Running RC in generative mode...', end='')
        self.test_len = test_len
        # run the trained ESN in a generative mode. no need to initialize here, 
        # because x is initialized with training data and we continue from there.
        self.Y = np.zeros((self.out_size,self.test_len))
        u = self.data[self.train_len]
        for t in range(self.test_len):
            self.x = (1-self.leak)*self.x + self.leak*np.tanh( np.dot( self.Win, np.vstack((1,u)) ) + np.dot( self.W, self.x ) )
            y = np.dot( self.Wout, np.vstack((1,u,self.x)) )
            self.Y[:,t] = y

            # anti-windup
            #gain = 0.01
            # Generative mode
            u = y #- (0.9*y)
            #u = y + random.uniform(-0.05, 0.05) # Noise prediction
        print('Done!')

    def Run_Generative_Stability(self, test_len, stop_t):
        print('Running RC in generative mode (stability test)...', end='')
        self.test_len = test_len
        # run the trained ESN in a generative mode. no need to initialize here, 
        # because x is initialized with training data and we continue from there.
        self.Y = np.zeros((self.out_size,self.test_len))
        u = self.data[self.train_len]
        for t in range(self.test_len):
            self.x = (1-self.leak)*self.x + self.leak*np.tanh( np.dot( self.Win, np.vstack((1,u)) ) + np.dot( self.W, self.x ) )
            y = np.dot( self.Wout, np.vstack((1,u,self.x)) )
            self.Y[:,t] = y

            # Generative mode
            if (t < stop_t):
                u = y
            else:
                y = 0
        print('Done!')

    def Run_Predictive_Stability(self, test_len, stop_t):
        print('Running RC in predictive mode (stability test)...',end='')
        self.test_len = test_len
        self.Y = np.zeros((self.out_size,self.test_len))
        u = self.data[self.train_len]
        for t in range(self.test_len):
            self.x = (1-self.leak)*self.x + self.leak*np.tanh( np.dot( self.Win, np.vstack((1,u)) ) + np.dot( self.W, self.x ) ) 
            y = np.dot( self.Wout, np.vstack((1,u,self.x)) )
            self.Y[:,t] = y
   
            # Predictive mode
            if (t < stop_t):
                u = self.data[self.train_len+t+1] # Ideal (non-noise) prediction
            else:
                u = 0
        print('Done!')

    def Run_Predictive(self, test_len):
        #print('Running RC in predictive mode...',end='')
        self.test_len = test_len
        self.Y = np.zeros((self.out_size,self.test_len))
        u = self.data[self.train_len]
        for t in range(self.test_len):
            self.x = (1-self.leak)*self.x + self.leak*np.tanh( np.dot( self.Win, np.vstack((1,u)) ) + np.dot( self.W, self.x ) ) 
            y = np.dot( self.Wout, np.vstack((1,u,self.x)) )
            self.Y[:,t] = y
   
            # Predictive mode
            u = self.data[self.train_len+t+1] # Ideal (non-noise) prediction
            #u = self.data[self.train_len+t+1] + random.uniform(-0.1, 0.1) # Noise prediction
        #print('Done!')

    def Compute_MSE(self, error_len):
        #print('Computing Mean Square Error (MSE)...',end='')
        self.error_len = error_len

        # compute MSE for the first errorLen time steps
        self.mse = sum( np.square( self.data[self.train_len+1:self.train_len+self.error_len+1] - 
            self.Y[0,0:self.error_len] ) ) / self.error_len
        #print('Done!')
        #print('MSE = ' + str( self.mse ))

    def Output_Init(self):
        # Checks if the output path exists and makes it if it doesn't
        try:
            os.makedirs("../../run/")
        except FileExistsError:
            # path already exists, move on
            pass

        try:
            os.makedirs("../../run/output")
        except FileExistsError:
            # path already exists, move on
            pass
    
        ts = time.time()
        self.st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H-%M-%S')
        try:
            os.makedirs("../../run/output/%s" % self.st)
        except FileExistsError:
            # path already exists, move on
            pass

        try:
            os.makedirs("../../run/output/%s/metrics" % self.st)
        except FileExistsError:
            # path already exists, move on
            pass

        try:
            os.makedirs("../../run/output/%s/plots" % self.st)
        except FileExistsError:
            # path already exists, move on
            pass

        try:
            os.makedirs("../../run/output/%s/data" % self.st)
        except FileExistsError:
            # path already exists, move on
            pass

    def Plots(self,silent=False):

        self.Output_Init()

        # plot some of it
        plt.figure(10).clear()
        plt.plot(self.data[:1000])
        plt.title('A sample of data')
        plt.savefig('../../run/output/%s/plots/data_sample.png' % self.st, bbox_inches='tight')

        # plot some signals
        plt.figure(1).clear()
        plt.plot( self.data[self.train_len+1:self.train_len+self.test_len+1], 'g' )
        plt.plot( self.Y.T, 'b' )
        plt.title('Target and generated signals $y(n)$ starting at $n=0$')
        plt.legend(['Target signal', 'Free-running predicted signal'])
        #xc1 = [600]
        #for xc in xc1:
        #    plt.axvline(x=xc, color='k', linestyle='--', linewidth=1)
        plt.savefig('../../run/output/%s/plots/target_predicted_signal.png' % self.st, bbox_inches='tight')

        plt.figure(2).clear()
        plt.plot( self.X[0:20,0:200].T )
        plt.title(r'Some reservoir activations $\mathbf{x}(n)$')
        plt.savefig('../../run/output/%s/plots/reservoir_activations.png' % self.st, bbox_inches='tight')

        plt.figure(3).clear()
        plt.bar( np.arange(1+self.in_size+self.reservoir_size), self.Wout[0].T )
        plt.title(r'Output weights $\mathbf{W}^{out}$')
        plt.savefig('../../run/output/%s/plots/output_weights.png' % self.st, bbox_inches='tight')

        if (silent == False):
            plt.show()

    def Save_Metrics(self):
        dirname = os.path.dirname(__file__)
        filename = '../../run/output/%s/metrics/metrics.txt' % self.st
        f = open(filename, "w+")

        s = []
        s.append("Reservoir Size: " + str(self.reservoir_size))
        s.append("Learning rate: " + str(self.leak))
        s.append("Training Length: " + str(self.train_len))
        s.append("Testing Length: " + str(self.test_len))
        s.append("Init Length: " + str(self.init_len))
        s.append("In Size: " + str(self.in_size))
        s.append("Out Size: " + str(self.out_size))         
        s.append("rho_W: " + str(self.rhoW))
        s.append("Error Length: " + str(self.error_len))
        s.append("mse: " + str(self.mse))

        for metric in s:
            f.write(metric)
            f.write("\n")

    # === Accessor Functions ===
    def Get_MSE(self):
        # Accessor function to get MSE
        return self.mse