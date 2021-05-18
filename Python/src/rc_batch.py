from rc import *
import matplotlib.pyplot as plt

leak = 0.001
dleak = 0.001
runs = 1000
leakage = []
error = []
for i in range(runs):
    if leak > 0.999:
        break

    rc = RC(40,leak)
    rc.Load_Reservoir_Data('../data/logistic_map_shaped.txt')
    rc.Load_Data('../data/MackeyGlass_t17.txt')
    rc.Generate_Reservoir()
    rc.Train(2000)
    rc.Run_Generative(1000)
    rc.Get_MSE(500)

    leakage.append(rc.leak)
    error.append(rc.mse)

    leak = leak + dleak

min_err = error[0]
min_index = 0
for i in range(1,len(error)): 
    if error[i]<min_err: 
        min_err=error[i] 
        min_index = i

min_leak = leakage[min_index]
print("\nminimum leak: ", min_leak)
print("minimum mse: ", min_err)

plt.plot(leakage,error)
plt.title("MSE for a given leakage");
plt.xlabel("Leaking rate");
plt.ylabel("MSE");
plt.show()