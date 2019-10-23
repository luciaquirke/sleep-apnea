import os

import numpy as np
import keras
from keras.utils import to_categorical


class FoldGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, inputsPath, targetsPath, filenames, batch_size=32, shuffle=True):
        'Initialization'
        self.inputsPath = inputsPath
        self.targetsPath = targetsPath
        self.filenames = filenames
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.filenames) / self.batch_size))

    def __getitem__(self, batch_num):
        'Generate one batch of data'
        # Generate indexes of the batch
        next_batch_indices = self.indices[batch_num*self.batch_size:(batch_num+1)*self.batch_size]

        # Find list of IDs
        next_batch_filenames = [self.filenames[i] for i in next_batch_indices]

        # Generate data
        X, y = self.__data_generation(next_batch_filenames)

        return X, y

    def on_epoch_end(self):
        'Updates indices after each epoch'
        self.indices = np.arange(len(self.filenames))
        if self.shuffle == True:
            np.random.shuffle(self.indices)

    def __data_generation(self, filenames):
        'Generates data containing batch_size samples'
        xLoaded = []
        yLoaded = []

        for filename in filenames:
            xData = open(os.path.join(os.getcwd(), self.inputsPath, filename), 'r').readlines()
            xLoaded.append([float(item.strip('[]\r\n')) for item in xData])

            if len(xData) != 7500:
                print("bad data")
                print(xData)
                print(filename)

            yData = open(os.path.join(os.getcwd(), self.targetsPath, filename), 'r').readlines()
            yLoaded.append(yData)

        xLoaded = np.array(xLoaded)
        xLoaded = xLoaded[..., np.newaxis]
        if xLoaded.shape == (32, 1):
            print("generation failed - wrong shape")
            print(filename)
            print(len(np.array(xData)))

        yLoaded = np.array(yLoaded)
        yLoaded = to_categorical(yLoaded)

        return xLoaded, yLoaded
