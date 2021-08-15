from rc import *
import matplotlib.pyplot as plt

new_rc = RC(40,0.1)
new_rc.Load_Reservoir_Data('../../datasets/shift_map_shaped.txt')
new_rc.Load_Data('../../datasets/MackeyGlass_t17.txt')
new_rc.Generate_Reservoir()
new_rc.Train(2000)
new_rc.Run_Generative(1000)
new_rc.Compute_MSE(1000)

silent_run = True
new_rc.Plots()