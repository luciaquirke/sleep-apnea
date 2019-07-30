import numpy as np
import pandas as pd
import os
import sklearn
import keras
import keras.backend as k
import seaborn as sn
import matplotlib.pyplot as plt
import h5py

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Activation
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.normalization import BatchNormalization
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping

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

#Use to check the balance of classes in the data
# ones = 0
# for event in Y:
#     if event == 1:
#         ones+=1

# print(((ones/len(Y))*100), "%")

Y = np.array(Y)

Y = to_categorical(Y)
Y = Y.reshape(-1, 2)

xShuffle, yShuffle = shuffle(X, Y, random_state = 2)

print(X.shape)
print(Y.shape)

xTrain, xTest, yTrain, yTest = train_test_split(xShuffle, yShuffle, test_size = 0.2)

print("Data Ready")

verbose, epochs, batch_size = 1, 10, 32
#class_weights = {0: 0.3, 1: 0.7}
#CNN layers
model = Sequential()

model.add(Conv1D(filters=40, kernel_size=250, input_shape=(7500,1)))
model.add(BatchNormalization())
model.add(Activation('elu'))
model.add(MaxPooling1D(pool_size=15))

model.add(Conv1D(filters=40, kernel_size=75))
model.add(BatchNormalization())
model.add(Activation('elu'))
model.add(MaxPooling1D(pool_size=5))

model.add(Conv1D(filters=60, kernel_size=10))
model.add(BatchNormalization())
model.add(Activation('elu'))
model.add(MaxPooling1D(pool_size=5))

model.add(Flatten())
model.add(Dense(15, activation='elu')) 
model.add(Dropout(0.5))

model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(xTrain, yTrain, epochs=epochs, batch_size=batch_size, verbose=verbose)
_, accuracy = model.evaluate(xTest, yTest, batch_size=batch_size, verbose=0)
yPred = model.predict(xTest)

#Calculate accuracy as a percentage
accuracy = accuracy * 100.0
print(accuracy, "%")

#Generate confusion matrix
matrix = confusion_matrix(yTest.argmax(axis=1), yPred.argmax(axis=1))
print(np.matrix(matrix))

#Generate classification report
target_names = ['non-apnea', 'apnea']
print(classification_report(yTest.argmax(axis=1), yPred.argmax(axis=1), target_names=target_names))

print(model.summary())

#Save the model. Change the name depending on the date/model
model.save('model_30_07_19_2.h5')
print('Model Saved')


