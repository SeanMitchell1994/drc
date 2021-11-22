from rc import *
from logger import *
import matplotlib.pyplot as plt

new_rc = RC(120,0.3)
new_rc.Load_Reservoir_Data('../../../datasets/logistic_map_shaped.txt')
new_rc.Load_Data('../../../datasets/MackeyGlass_t17.txt')
new_rc.Generate_Reservoir()
new_rc.Train(4000)
new_rc.Run_Predictive(1000)
new_rc.Compute_MSE(1000)
#new_rc.Save_Metrics()

print(new_rc.Get_MSE())
silent_run = False
new_rc.Plots(silent_run)