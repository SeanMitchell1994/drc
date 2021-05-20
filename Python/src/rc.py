import os
import errno
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg 
# numpy.linalg is also an option for even fewer dependencies

class RC:
    def __init__(self, reservoir_size, leak):
        self.reservoir_size = reservoir_size
        self.leak = leak    # leaking rate
        self.train_len = 0
        self.test_len = 0
        self.init_len = 100
        self.in_size = 1
        self.out_size = self.in_size

        self.data = 0
        self.rc_data = 0
        self.test_data = 0

        self.Win = 0
        self.W = 0
        self.rhoW = 0
        self. X = 0
        self.x = 0
        self. Yt = 0

        self.error_len = 0
        self.mse = 0

    def Load_Reservoir_Data(self, data):
        print('Loading data...')
        self.rc_data = np.loadtxt(data)
        print('Done')

    def Load_Data(self, data):
        print('Loading data...')
        self.data = np.loadtxt(data)
        print('Done')

    def Load_Test_Data(self, data):
        print('Loading data...')
        self.test_data = np.loadtxt(data)
        print('Done')

    def Generate_Reservoir(self):
        # generate the ESN reservoir
        #inSize = outSize = 1
        #resSize = 1000
        #a = 0.3 # leaking rate
        print('Generating reservoir')
        np.random.seed(42)
        self.Win = (np.random.rand(self.reservoir_size,1 + self.in_size) - 0.5) * 1
        self.W = np.random.rand(self.reservoir_size, self.reservoir_size) - 0.5
        rcd_t = np.transpose(self.rc_data)
        self.W = np.dot(self.W, rcd_t)

        # normalizing and setting spectral radius (correct, slow):
        print('Computing spectral radius...')
        self.rhoW = max(abs(linalg.eig(self.W)[0]))
        print('Done')
        self.W *= 1.25 / self.rhoW

    def Train(self, train_length):
        print('Training the reservoir computer...')
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
        print('Done')

    def Run_Generative(self, test_len):
        print('Running RC in generative mode...')
        self.test_len = test_len
        # run the trained ESN in a generative mode. no need to initialize here, 
        # because x is initialized with training data and we continue from there.
        self.Y = np.zeros((self.out_size,self.test_len))
        u = self.data[self.train_len]
        for t in range(self.test_len):
            self.x = (1-self.leak)*self.x + self.leak*np.tanh( np.dot( self.Win, np.vstack((1,u)) ) + np.dot( self.W, self.x ) )
            y = np.dot( self.Wout, np.vstack((1,u,self.x)) )
            self.Y[:,t] = y
            # generative mode:
            u = y
            ## this would be a predictive mode:
            #u = data[trainLen+t+1] 
        print('Done')

    def Run_Predictive(self, test_len):
        print('Running RC in predictive mode...')
        self.test_len = test_len
        # run the trained ESN in a generative mode. no need to initialize here, 
        # because x is initialized with training data and we continue from there.
        self.Y = np.zeros((self.out_size,self.test_len))
        u = self.data[self.train_len]
        for t in range(self.test_len):
            self.x = (1-self.leak)*self.x + self.leak*np.tanh( np.dot( self.Win, np.vstack((1,u)) ) + np.dot( self.W, self.x ) )
            y = np.dot( self.Wout, np.vstack((1,u,self.x)) )
            self.Y[:,t] = y
            # generative mode:
            #u = y
            ## this would be a predictive mode:
            u = self.data[self.train_len+t+1] 
        print('Done')

    def Get_MSE(self, error_len):
        print('Computing Mean Square Error (MSE)')
        self.error_len = error_len

        # compute MSE for the first errorLen time steps
        #errorLen = 500
        self.mse = sum( np.square( self.data[self.train_len+1:self.train_len+self.error_len+1] - 
            self.Y[0,0:self.error_len] ) ) / self.error_len
        print('MSE = ' + str( self.mse ))
        print('Done')

    def Output_Init(self):
        # Checks if the output path exists and makes it if it doesn't
        try:
            os.makedirs("../output/")
        except FileExistsError:
            # path already exists, move on
            pass

    def Plots(self,silent=False):

        self.Output_Init()

        # plot some of it
        plt.figure(10).clear()
        plt.plot(self.data[:1000])
        plt.title('A sample of data')
        plt.savefig('../output/data_sample.png', bbox_inches='tight')

        # plot some signals
        plt.figure(1).clear()
        plt.plot( self.data[self.train_len+1:self.train_len+self.test_len+1], 'g' )
        plt.plot( self.Y.T, 'b' )
        plt.title('Target and generated signals $y(n)$ starting at $n=0$')
        plt.legend(['Target signal', 'Free-running predicted signal'])
        plt.savefig('../output/target_predicted_signal.png', bbox_inches='tight')

        plt.figure(2).clear()
        plt.plot( self.X[0:20,0:200].T )
        plt.title(r'Some reservoir activations $\mathbf{x}(n)$')
        plt.savefig('../output/reservoir_activations.png', bbox_inches='tight')

        plt.figure(3).clear()
        plt.bar( np.arange(1+self.in_size+self.reservoir_size), self.Wout[0].T )
        plt.title(r'Output weights $\mathbf{W}^{out}$')
        plt.savefig('../output/output_weights.png', bbox_inches='tight')

        if (silent == False):
            plt.show()