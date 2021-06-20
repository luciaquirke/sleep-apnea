import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import time

from datetime import date

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from scipy.stats import norm

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

# name of model for saving, specified by date and time of creation
name = 'model-' + str(date.today()) + '-' + \
    str(time.localtime().tm_hour) + '-' + str(time.localtime().tm_min)

# loads data from a single .csv file


def load_file(filepath):
    dataframe = pd.read_csv(filepath, header=None, delim_whitespace=True)
    return dataframe.values


# change to current OS
operatingSystem = 'windows'

if operatingSystem is 'linux' or operatingSystem is 'macOS':
    inputsPath = '/data/inputs/'
    targetsPath = '/data/targets/'
else:
    inputsPath = '\\data\\inputs\\'
    targetsPath = '\\data\\targets\\'

xLoaded = list()
yLoaded = []

print("Loading Data...")

# load input and target .csv files
for root, dirs, files in os.walk('.' + inputsPath):
    for fileName in files:
        xData = load_file(os.getcwd() + inputsPath + fileName)
        xLoaded.append(xData)
        yData = load_file(os.getcwd() + targetsPath + fileName)
        yLoaded.append(yData)

# data loaded in
X = np.stack(xLoaded, axis=0)
Y = yLoaded

# use to check the balance of classes in the data
ones = 0
for event in Y:
    if event == 1:
        ones += 1

print(((ones/len(Y))*100), "%")

Y = np.array(Y)

# change targets to one hot encoded form
Y = to_categorical(Y)
Y = Y.reshape(-1, 2)

# shuffle data before split
xShuffle, yShuffle = shuffle(X, Y, random_state=2)

# split inputs and targets into train and test sets
xTrain, xTest, yTrain, yTest = train_test_split(
    xShuffle, yShuffle, test_size=0.2)

print("Data Ready")

verbose, epochs, batch_size = 1, 50, 32

# initilialise model
model = Sequential()

# Convolutional Layer 1
model.add(Conv1D(filters=20, kernel_size=125, input_shape=(7500, 1)))
model.add(BatchNormalization())
model.add(Activation('elu'))
model.add(MaxPooling1D(pool_size=10))
model.add(Dropout(0.3))

# Convolutional Layer 2
model.add(Conv1D(filters=40, kernel_size=50))
model.add(BatchNormalization())
model.add(Activation('elu'))
model.add(MaxPooling1D(pool_size=5))
model.add(Dropout(0.3))

# Convolutional Layer 3
model.add(Conv1D(filters=60, kernel_size=10))
model.add(BatchNormalization())
model.add(Activation('elu'))
model.add(MaxPooling1D(pool_size=5))
model.add(Dropout(0.3))

# Fully Connected Layer 1
model.add(Flatten())
model.add(Dense(10, activation='elu'))
model.add(Dropout(0.3))

# Fully Connected Layer 2
model.add(Dense(2, activation='softmax'))

# configure model for training
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

# initialise early stopping callback
es = EarlyStopping(monitor='val_acc', mode='max', patience=10,
                   verbose=1, restore_best_weights=True)

# train model
history = model.fit(xTrain, yTrain, epochs=epochs, batch_size=batch_size,
                    verbose=verbose, validation_split=0.1, callbacks=[es])

# test model and return accuracy
_, accuracy = model.evaluate(xTest, yTest, batch_size=batch_size, verbose=0)

# find predictions model makes for test set
yPred = model.predict(xTest)

# calculate accuracy as a percentage
accuracy = accuracy * 100.0
print('Accuracy =', accuracy, "%")

# generate confusion matrix
matrix = confusion_matrix(yTest.argmax(axis=1), yPred.argmax(axis=1))
print('Confusion Matrix:')
print(np.matrix(matrix))

# calculate d'
tp, fn, fp, tn = matrix.ravel()
dprime = norm.ppf(tn/(tn+fp)) - norm.ppf(fn/(tp+fn))
print('dPrime =', dprime)

# generate classification report
target_names = ['non-apnea', 'apnea']
print('Classification Report:')
print(classification_report(yTest.argmax(axis=1),
      yPred.argmax(axis=1), target_names=target_names))

# access the accuracy and loss values found throughout training
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

# plot accuracy throughout training (validation and training)
plt.plot(epochs, acc, color='xkcd:azure')
plt.plot(epochs, val_acc, color='xkcd:green')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['Training', 'Validation'], loc='upper left')
plt.figure()

# plot loss throughout training (validation and training)
plt.plot(epochs, loss, color='xkcd:azure')
plt.plot(epochs, val_loss, color='xkcd:green')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Training', 'Validation'], loc='upper left')
plt.figure()

# plot accuracy throughout training (just training)
plt.plot(epochs, acc, 'b')
plt.title('Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.figure()

# plot loss throughout training (just training)
plt.plot(epochs, loss, 'b')
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

# save the model
model.save(name + '.h5')
print('Model Saved')
