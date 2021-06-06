from rc import *
import matplotlib.pyplot as plt

new_rc = RC(40,0.35)
new_rc.Load_Reservoir_Data('../../datasets/logistic_map_shaped.txt')
new_rc.Load_Data('../../datasets/MackeyGlass_t17.txt')
new_rc.Generate_Reservoir()
new_rc.Train(2000)
new_rc.Run_Predictive(1000)
new_rc.Get_MSE(500)

silent_run = True
new_rc.Plots()