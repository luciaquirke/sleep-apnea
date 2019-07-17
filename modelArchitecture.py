import numpy as np
import pandas as pd
import os
import sklearn

from sklearn.model_selection import train_test_split

""" from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D """

#change to current OS
operatingSystem = 'windows'

if operatingSystem == 'linux':
    inputsPath = '/data/inputs/'
    targetsPath = '/data/targets/'
else:
    inputsPath = '\\data\\inputs\\'
    targetsPath = '\\data\\targets\\'

def load_file(filepath):
	dataframe = pd.read_csv(filepath, header=None, delim_whitespace=True)
	return dataframe.values

# load a list of files into a 3D array of [samples, timesteps, features]
xLoaded = list()
yLoaded = []

for root, dirs, files in os.walk('.' + inputsPath):
    for fileName in files:
        xData = load_file(os.getcwd() + inputsPath + fileName)
        xLoaded.append(xData)
        yData = load_file(os.getcwd() + targetsPath + fileName)
        yLoaded.append(yData)

# stack group so that features are the 3rd dimension
X = np.stack(xLoaded, axis = 0) 
# Y is simply an array of data
Y = yLoaded

print(X.shape)
print(len(Y))

xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size = 0.3)
print(xTrain.shape)
print(len(yTrain))
print(xTest.shape)
print(len(yTest))

verbose, epochs, batch_size = 0, 10, 32
#CNN layers
""" model = Sequential()
model.add(Conv1D(filters=128, kernel_size=7, activation='relu', input_shape=(7500,1)))
model.add(Conv1D(filters=128, kernel_size=7, activation='relu')
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=256, kernel_size=7, activation='relu')
model.add(Conv1D(filters=256, kernel_size=7, activation='relu')
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(100, activation='relu'))    #Not sure about 100 - size of output of dense layer
model.add(Dense(2, activation='softmax')) """

#Ideas for improvement: 
#   Add dropout
#   Add more layers
#   Experiment with kernel size
#   Experiment with number of filters/features

