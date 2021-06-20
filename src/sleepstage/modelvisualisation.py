import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
from scipy.fftpack import fft
from keras.models import load_model


def main():
    model = load_model(os.path.join('models', 'example-trained-model.h5'))

    # look at model summary to see shapes between layers etc.
    model.summary()

    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)

    # summarize filter shapes
    for layer in model.layers:
        # skips non-convolutional layers
        if 'conv' not in layer.name:
            continue
        # get filter weights
        filters, biases = layer.get_weights()
        print(layer.name, filters.shape)

    # Number of sample points in filters
    N1 = 125
    N2 = 50
    N3 = 10
    # sample spacing
    T = 1.0 / 250.0

    # visualise the filters in the first layer (conv layers are layers 0, 5 and 10)
    filters1, biases1 = model.layers[0].get_weights()
    for i in range(20):
        f = filters1[:, :, i]
        x = np.linspace(0.0, N1*T, N1)
        plt.subplot(10, 2, (i+1))
        plt.ylim(-0.2, 0.2)
        plt.plot(x, f, color='xkcd:azure')
    fig = plt.figure()

    for i in range(20):
        y = filters1[:, :, i]
        x = np.linspace(0.0, N1*T, N1)
        yf = fft(y)
        xf = np.linspace(0.0, 1.0/(2.0*T), N1//2)
        yAmp = 2.0/N1 * np.abs(yf[0:N1//2])
        plt.subplot(10, 2, (i+1))
        plt.plot(xf, yAmp, color='xkcd:azure')
        plt.xlim(0, 30)
    plt.suptitle('Fourier Transforms of the Filters in the First Layer')

    # plot a specific filter and its fourier transform
    filterNumber = 7   # change to the particular filter you want to visualise

    x = np.linspace(0.0, N1*T, N1)
    f = filters1[:, :, filterNumber]
    fig, ax = plt.subplots()
    plt.subplot(2, 1, 1)
    ax.tick_params(labelsize=20)
    plt.plot(x, f, color='xkcd:azure')
    plt.xlabel('Time (s)', fontsize=18)
    plt.ylabel('Weights', fontsize=18)

    y = filters1[:, :, filterNumber]
    yf = fft(y)
    xf = np.linspace(0.0, 1.0/(2.0*T), N1//2)
    yAmp = 2.0/N1 * np.abs(yf[0:N1//2])
    plt.subplot(2, 1, 2)
    ax.tick_params(labelsize=20)
    plt.plot(xf, yAmp, color='xkcd:azure')
    plt.xlim(0, 30)
    plt.xlabel('Frequency (Hz)', fontsize=18)
    plt.ylabel('Amplitude', fontsize=18)
    # plt.suptitle('Single First Layer Filter with Fourier Transform', fontsize=16)

    # plot heat map to visually show the fourier transform
    fig, ax = plt.subplots()
    plt.subplot(2, 1, 1)
    plt.plot(f, color='xkcd:azure')
    plt.xlabel('Points')
    plt.ylabel('Weights')

    plt.subplot(2, 1, 2)
    df = pd.DataFrame(data=yAmp, index=np.around(xf, 0))
    df = df.transpose()
    ax = sn.heatmap(df, cmap='Blues', cbar=True)
    plt.xlim(0, 15)
    plt.suptitle('Single First Layer Filter with Frequency Heat Map')

    plt.figure()
    # plot specific filters from the second layer
    filters2, biases2 = model.layers[5].get_weights()
    for i in range(40):
        f = filters2[:, filterNumber, i]
        plt.subplot(10, 4, (i+1))
        plt.plot(f, color='xkcd:azure')
    plt.figure()

    for i in range(40):
        y = filters2[:, filterNumber, i]
        x = np.linspace(0.0, N2*T, N2)
        yf = fft(y)
        xf = np.linspace(0.0, 1.0/(2.0*T), N2//2)
        yAmp = 2.0/N2 * np.abs(yf[0:N2//2])
        plt.subplot(10, 4, (i+1))
        plt.plot(xf, yAmp, color='xkcd:azure')
        plt.xlim(0, 30)
    plt.show()


# starts main if file called as script (rather than imported)
if __name__ == "__main__":
    main()
