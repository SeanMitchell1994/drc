# System imports
import matplotlib.pyplot as plt
import numpy as np
import random
import math 
from multiprocessing import Process, Pool

def lm_sweep(a, res_size):
    length = 10000

    y_i = np.zeros(length)
    y_i[0] = random.uniform(0.01, 0.09)
    #a = 3.718 #lambda = 0.343
    #a = 3.937 # lambda = 0.541
    for j in range(1,length):
        y_i[j] = (a * y_i[j-1]) * (1 -  y_i[j-1])

    return y_i

def Lyapunov_Exponent_lm(iterate):
    result = []
    lambdas = []
    r_list = []

    xmin = 2
    xmax = 4

    rvalues = np.arange(xmin, xmax, iterate)

    for r in rvalues:
        x = 0.1
        result = []
        for t in range(1000):
            x = r * x * (1 - x)
            result.append(np.log(abs(r - 2*r*x)))
        lambdas.append(np.mean(result))
        r_list.append(r)

    return lambdas, r_list

def Lyapunov_Exponent_sm(iterate):
    result = []
    lambdas = []
    r_list = []

    xmin = 0
    xmax = 2

    rvalues = np.arange(xmin, xmax, iterate)

    for r in rvalues:
        x = 0.1
        result = []
        for t in range(1000):
            x = r*2* (x % 1)
            result.append(np.log(abs(2*r)))
        lambdas.append(np.mean(result))
        r_list.append(r)

    return lambdas, r_list

def Match_Parameter(value, r_list, lya_list):
    allowed_error = 0.0001
    index = 0
    lya_exp = 0
    for i in range(len(r_list)):
        if abs(r_list[i] - value) <= allowed_error:
            index = i
            lya_exp = lya_list[index]
    return index, lya_exp

def Match_Lya_Exp(lya_exp, r_list, lya_list):
    allowed_error = 0.01
    index = 0
    #lya_exp = 0
    r_val = 0
    for i in range(len(lya_list)):
        if abs(lya_list[i] - lya_exp) <= allowed_error:
            index = i
            r_val = r_list[index]
    return index, r_val

def main():
    param = 0
    iterate = 0.01
    res_size = 40
    r = 3.2
    lambda_01 = 0
    lambda_02 = 0
    length = 10000

    mse_list = []
    param_list = []
    lambdas_sm = []
    r_list_sm = []
    lambdas_lm = []
    r_list_lm = []
    lmb_hold = []

    #lm_data = lm_sweep(param, res_size)
    lambdas_lm,r_list_lm = Lyapunov_Exponent_lm(iterate)
    lambdas_sm,r_list_sm = Lyapunov_Exponent_sm(iterate)

    index = 0
    index_2 = 0
    allowed_error = 0.00001
    for i in range(len(lambdas_sm)):
        for j in range(len(lambdas_lm)):
            if abs(lambdas_sm[i] - lambdas_lm[j]) <= allowed_error:
                index = i
                index_2 = j

    match_index, match_lya = Match_Parameter(r, r_list_lm, lambdas_lm)
    print(match_index, match_lya)
    match_index_2, match_r = Match_Lya_Exp(match_lya, r_list_sm, lambdas_sm)
    print(match_index_2, match_r)
    print("break")

if __name__ == "__main__":
    main()