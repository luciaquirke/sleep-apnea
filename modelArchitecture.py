from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

model = Sequential()
model.add(Conv1D(filters=128, kernel_size=7, activation='relu', input_shape=(7500,1)))
model.add(Conv1D(filters=128, kernel_size=7, activation='relu')
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=256, kernel_size=7, activation='relu')
model.add(Conv1D(filters=256, kernel_size=7, activation='relu')
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(100, activation='relu'))    #Not sure about 100 - size of output of dense layer
model.add(Dense(2, activation='softmax'))

#Ideas for improvement: 
#   Add dropout
#   Add more layers
#   Experiment with kernel size
#   Experiment with number of filters/features

