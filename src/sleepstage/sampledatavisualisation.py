import numpy as np
import os
import matplotlib.pyplot as plt
from ..utils.utils import setup_directory


def main():
    """Plots the input data specified in the filename variable and prints its target data"""
    inputs_path, targets_path, _ = setup_directory()

    x_loaded = []
    y_loaded = []

    filename = '9_427.csv'

    x_data = open(os.path.join(
        os.getcwd(), inputs_path, filename), 'r').readlines()
    x_loaded.append([float(item.strip('[]\r\n')) for item in x_data])
    y_data = open(os.path.join(
        os.getcwd(), targets_path, filename), 'r').readlines()
    y_loaded.append(y_data)

    print(x_loaded[0])
    print(y_loaded[0])

    fig, ax = plt.subplots()
    ax.set_aspect(2)
    plt.plot(np.linspace(0, 30, 7500), x_loaded[0][:], linewidth=1)
    ax.set(xlabel='time (s)', ylabel='voltage (mV)',
           title='Input Signal')

    fig.savefig(str(filename) + ".pdf")
    plt.show()

    return None


# starts main if file called as script (rather than imported)
if __name__ == "__main__":
    main()
