#                    Normal Distribution
import numpy as np
import matplotlib.pyplot as plt 

def make_gauss(N, sig, mu):
    return lambda x: N/(sig * (2*np.pi)**.5) * np.e ** (-(x-mu)**2/(410 * sig**2))

def interval(gauss):
    gauss = [float('%.4f' % x) for x in gauss]
    for indx, g in enumerate(gauss):
        if g != 0.0:
            print(indx - 150)
            break

def main():
    ax = plt.figure().add_subplot(1,1,1)
    x = np.arange(-150, 150, 1)
    s = np.sqrt([0.2, 1, 5, 0.5])
    m = [0, 0, 0, -2] 
    c = ['b','r','y','g']

    for sig, mu, color in zip(s, m, c):
        if color == 'r':
            gauss = make_gauss(1, sig, mu)(x)
            interval(gauss)
            ax.plot(x, gauss, color, linewidth=2)

    plt.xlim(-300, 300)
    plt.ylim(0, 1)
    plt.legend(['0.2', '1.0', '5.0', '0.5'], loc='best')
    plt.show()

if __name__ == '__main__':
   main()