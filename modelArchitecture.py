import numpy as np
import pandas as pd
import os
import sklearn
import keras

from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.utils import to_categorical

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

print("Loading Data...")

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
print(Y)

Y = to_categorical(Y)
X = np.array(X)
Y = np.array(Y)
Y = Y.reshape(-1, 2)
print(Y)
print(X.shape)


#Use to check the balance of classes in the data
# ones = 0
# for event in Y:
#     if event == 1:
#         ones+=1

# print(((ones/len(Y))*100), "%")

xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size = 0.3)

print("Data Ready")

def evaluate_model(xTrain, yTrain, xTest, yTest):
    verbose, epochs, batch_size = 1, 10, 32
    #CNN layers
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=7, activation='relu', input_shape=(7500,1)))
    model.add(Conv1D(filters=64, kernel_size=7, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=128, kernel_size=7, activation='relu'))
    model.add(Conv1D(filters=128, kernel_size=7, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))    #Not sure about 100 - size of output of dense layer
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(xTrain, yTrain, epochs=epochs, batch_size=batch_size, verbose=verbose)
    _, accuracy = model.evaluate(xTest, yTest, batch_size=batch_size, verbose=0)
    return accuracy

score = evaluate_model(xTrain, yTrain, xTest, yTest)
score = score * 100.0
print(score, "%")

#Ideas for improvement: 
#   Add dropout
#   Add more layers
#   Experiment with kernel size
#   Experiment with number of filters/features

