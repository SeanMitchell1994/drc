# ===============================================
# demo_dsn_generative.py
#
# small demo of the DSN running a time series
# prediction using a shift map as r(t)
# ===============================================

# === global imports ===
import pandas

# === Local imports ===
import common
from rc import *
from logger import *

def shift_map(a, res_size, ic):
    length = res_size * res_size

    y_i = np.zeros(length)
    y_i[0] = ic

    for j in range(1,length):
        y_i[j] = a * 2*y_i[j-1] % 1

    y_i_temp = y_i.reshape(res_size,res_size)
    y_i_shaped = np.transpose(y_i_temp)

    return y_i_shaped

def main():

    # === Runtime variables ===
    res_size = 120              # Size of the reservoir (nxn matrix)
    learning_rate = 0.3         # learning rate/neural leak/transient rsp
    training_length = 1000      # Length of data we train on
    test_length = 1000          # length of data we test on
    sm_ic = 0.4                 # IC for shift map
    sm_slope = 2                # Slope of shift map

    # === Data variables ===
    res_fcn = shift_map(sm_slope, res_size, sm_ic)
    dataset = '../../../datasets/lorenz_x.txt'

    # Logging object that collects and saves outputs, metrics, plots, etc
    # This is optional and you can run without
    logger = Logger()

    # === RC stuff ===
    # create new RC object
    new_rc = RC(res_size, learning_rate)

    # Load the reservoir with complexity from the res. function
    new_rc.Load_Reservoir_Function(res_fcn)

    # Load the data we want to train and test with
    new_rc.Load_Data(dataset)

    # Actually generate the reservoir now that we've set up all the variables
    new_rc.Generate_Reservoir()

    # Train it
    new_rc.Train(training_length)

    # Test it
    new_rc.Run_Generative(test_length)

    # How accurate were we?
    new_rc.Compute_MSE(test_length)
    #new_rc.Save_Metrics()

    df = pandas.DataFrame(res_fcn) 
    logger.Save_Data(df, "reservoir_function")

    # Produce plots
    print(new_rc.Get_MSE())
    silent_run = False
    new_rc.Plots(silent_run)

if __name__ == "__main__":
    main()