import os
import numpy as np
import keras
from keras.utils import to_categorical


class FoldGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, inputs_path, targets_path, filenames, batch_size=32, shuffle=True):
        'Initialization'
        self.inputs_path = inputs_path
        self.targets_path = targets_path
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
        next_batch_indices = self.indices[batch_num *
                                          self.batch_size:(batch_num+1)*self.batch_size]

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
        x_loaded = []
        y_loaded = []

        for filename in filenames:
            x_data = open(os.path.join(
                os.getcwd(), self.inputs_path, filename), 'r').readlines()
            x_loaded.append([float(item.strip('[]\r\n')) for item in x_data])

            if len(x_data) != 7500:
                print("bad data")
                print(x_data)
                print(filename)

            y_data = open(os.path.join(
                os.getcwd(), self.targets_path, filename), 'r').readlines()
            y_loaded.append(y_data)

        x_loaded = np.array(x_loaded)
        x_loaded = x_loaded[..., np.newaxis]
        if x_loaded.shape == (32, 1):
            print("generation failed - wrong shape")
            print(filename)
            print(len(np.array(x_data)))

        y_loaded = np.array(y_loaded)
        y_loaded = to_categorical(y_loaded)

        return x_loaded, y_loaded
