import numpy as np
import os
from glob import glob
import matplotlib.pyplot as plt
import h5py

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from scipy.stats import norm

import keras
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

from dataGenerator import DataGenerator

inputsPath = 'data/inputs/'
targetsPath = 'data/5-shot-targets/'

print("Loading Data...")

files = list(glob(os.path.join(inputsPath, "**", "*.csv"), recursive=True))
files = [os.path.basename(filename) for filename in files]

print("{:} data-points found".format(len(files)))

# check the balance of classes in the data

# countClasses = [0]*7
#
# for event in Y:
#     print(event)
#     countClasses(defaultdict[event])

#print(((ones/len(Y))*100), "%")

trainFiles, testFiles = train_test_split(files, test_size=0.2)
trainFiles, validationFiles = train_test_split(trainFiles, test_size=0.1)

# generates training/test/validation data in batches
trainGenerator = DataGenerator(inputsPath, targetsPath, trainFiles)
validationGenerator = DataGenerator(inputsPath, targetsPath, validationFiles)
testGenerator = DataGenerator(inputsPath, targetsPath, testFiles)

verbose, epochs = 1, 100
# CNN layers
model = Sequential()

model.add(Conv1D(filters=20, kernel_size=125, input_shape=(7500, 1)))
model.add(BatchNormalization())
model.add(Activation('elu'))
model.add(MaxPooling1D(pool_size=10))
model.add(Dropout(0.3))

model.add(Conv1D(filters=40, kernel_size=50))
model.add(BatchNormalization())
model.add(Activation('elu'))
model.add(MaxPooling1D(pool_size=5))
model.add(Dropout(0.3))

model.add(Conv1D(filters=60, kernel_size=10))
model.add(BatchNormalization())
model.add(Activation('elu'))
model.add(MaxPooling1D(pool_size=5))
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(15, activation='elu')) 
model.add(Dropout(0.3))

model.add(Dense(9, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

es = EarlyStopping(monitor='val_acc', mode='max', patience=2, verbose=1, restore_best_weights=True)

history = model.fit_generator(trainGenerator, validation_data=validationGenerator, epochs=epochs, verbose=verbose, callbacks=[es])
_, accuracy = model.evaluate_generator(testGenerator, verbose=0)

xTest = []
yTest = []
testGenerator.on_epoch_end()
for i in range(len(testGenerator)):
    xBatch, yBatch = testGenerator[i]
    xTest.extend(xBatch)
    yTest.extend(yBatch)

xTest = np.array(xTest)
yTest = np.array(yTest)

yPred = model.predict_generator(xTest, steps=32)

# Calculate accuracy as a percentage
accuracy = accuracy * 100.0
print('Accuracy =', accuracy, "%")

# Generate confusion matrix
matrix = confusion_matrix(yTest.argmax(axis=1), yPred.argmax(axis=1))
print('Confusion Matrix:')
print(np.matrix(matrix))

# Calculate d' from testing
tp, fn, fp, tn = matrix.ravel()
dprime = norm.ppf(tp/(tp+fn)) - norm.ppf(tn/(tn+fp))
print('dPrime =', dprime)

# Generate classification report
target_names = ['non-apnea', 'apnea']
print('Classification Report:')
print(classification_report(yTest.argmax(axis=1), yPred.argmax(axis=1), target_names=target_names))

acc = history.history['acc']
loss = history.history['loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo')
plt.title('Training accuracy')
plt.xlabel('Training Epochs')
plt.ylabel('Accuracy')
plt.figure()

plt.plot(epochs, loss, 'bo')
plt.title('Training loss')
plt.xlabel('Training Epochs')
plt.ylabel('Loss')
plt.show()

print(model.summary())

# Save the model. Change the name depending on the date/model
model.save('model_01_08_19_3.h5')
print('Model Saved')
