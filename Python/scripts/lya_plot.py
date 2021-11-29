import matplotlib.pyplot as plt
import numpy as np

def main():
    sm_lya_path = '../../../datasets/lya/sm_lya.txt'
    lm_lya_path = '../../../datasets/lya/lm_lya.txt'
    lorenz_x_lya_path = '../../../datasets/lya/lorenz_x_lya.txt'
    r_lya_path = '../../../datasets/lya/r_lya.txt'

    sm_lya = np.loadtxt(sm_lya_path)
    lm_lya = np.loadtxt(lm_lya_path)
    lorenz_x_lya = np.loadtxt(lorenz_x_lya_path)
    r_lya = np.loadtxt(r_lya_path)

    plt.figure(1).clear()
    plt.plot(r_lya,sm_lya,'b')
    plt.plot(r_lya,lm_lya,'g')
    plt.plot(r_lya, lorenz_x_lya[0:4001],'r')
    #plt.fill_between(r_lya, sm_lya, lorenz_x_lya[0:4001],where=(sm_lya >= lorenz_x_lya[0:4001]), color='C1', alpha=0.3, interpolate=True)
    plt.axhline(y=0.0, color='k', linestyle='--', linewidth=0.5)
    plt.axvline(x=0.5, color='k', linestyle='--', linewidth=1)
    plt.axvline(x=1.125, color='k', linestyle='--', linewidth=1)
    plt.axvspan(0, 0.5, alpha=0.1, color='green')
    plt.axvspan(0.5, 1.125, alpha=0.1, color='red')
    plt.axvspan(1.125, 4, alpha=0.1, color='blue')
    plt.title('Comparsion of Lyapunov Exponents')
    plt.xlabel('Sweep Parameter (a)')
    plt.ylabel('Lambda')
    plt.legend(['Shift Map LE','Logistics Map LE','Lorenz x LE'])
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()