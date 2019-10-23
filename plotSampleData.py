import numpy as np
import os
from glob import glob
import matplotlib.pyplot as plt


def main():
    """Plots the input data specified in filename variable and prints its target data"""
    inputsPath = 'data/inputs/'
    targetsPath = 'data/targets/'

    xLoaded = []
    yLoaded = []

    filename = '9_427.csv'

    xData = open(os.path.join(os.getcwd(), inputsPath, filename), 'r').readlines()
    xLoaded.append([float(item.strip('[]\r\n')) for item in xData])
    yData = open(os.path.join(os.getcwd(), targetsPath, filename), 'r').readlines()
    yLoaded.append(yData)

    print(xLoaded[0])
    print(yLoaded[0])

    fig, ax = plt.subplots()
    ax.set_aspect(2)
    plt.plot(np.linspace(0, 30, 7500), xLoaded[0][:], linewidth=1)
    ax.set(xlabel='time (s)', ylabel='voltage (mV)',
           title='Input Signal')

    fig.savefig(str(filename) + ".pdf")
    plt.show()

    return None


# starts main if file called as script (rather than imported)
if __name__ == "__main__":
    main()
