import numpy as np
import os
import datetime
from glob import glob
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
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
from keras.callbacks import EarlyStopping

from datagenerator import DataGenerator


def main(inputs_path='data/inputs/', targets_path='data/5-shot-targets/', save_confusion=True):
    """Convolutional Neural Network Creation and Evaluation. Uses data collected from the Physiobank data set and stored
    in the specified input and target folders"""

    print("Running...")

    model = define_model()

    train_files, test_files, validation_files = get_input_file_names(
        inputs_path)

    # generate training/test/validation data in batches
    train_generator = DataGenerator(inputs_path, targets_path, train_files)
    validation_generator = DataGenerator(
        inputs_path, targets_path, validation_files)
    test_generator = DataGenerator(inputs_path, targets_path, test_files)

    # run model training and evaluation
    es = EarlyStopping(monitor='val_acc', mode='max',
                       patience=5, verbose=1, restore_best_weights=True)
    history = model.fit_generator(train_generator, validation_data=validation_generator, epochs=100, verbose=1,
                                  callbacks=[es])
    _, accuracy = model.evaluate_generator(test_generator, verbose=0)

    # create test set and targets
    x_test, y_test = [], []
    test_generator.on_epoch_end()
    for i in range(len(test_generator)):
        x_batch, y_batch = test_generator[i]
        x_test.extend(x_batch)
        y_test.extend(y_batch)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    y_prediction = model.predict(x_test)

    save_model(model)

    evaluate_model(model, history, accuracy, y_test,
                   y_prediction, save_confusion)

    return None


def get_input_file_names(inputs_path):
    """"Generates separate lists of train, test, and validation file names from files in specified file paths"""
    print("Loading Data...")

    files = list(
        glob(os.path.join(inputs_path, "**", "*.csv"), recursive=True))
    files = [os.path.basename(filename) for filename in files]

    print("{:} data-points found".format(len(files)))

    train_file_names, test_file_names = train_test_split(files, test_size=0.2)
    train_file_names, validation_file_names = train_test_split(
        train_file_names, test_size=0.1)

    return train_file_names, test_file_names, validation_file_names


def define_model():
    """Defines CNN layers"""
    model = Sequential()

    model.add(Conv1D(filters=20, kernel_size=80, input_shape=(7500, 1)))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(MaxPooling1D(pool_size=10))
    model.add(Dropout(0.3))

    model.add(Conv1D(filters=40, kernel_size=20))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(MaxPooling1D(pool_size=5))
    model.add(Dropout(0.3))

    model.add(Conv1D(filters=60, kernel_size=10))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(MaxPooling1D(pool_size=5))
    model.add(Dropout(0.3))

    model.add(Conv1D(filters=80, kernel_size=4))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(MaxPooling1D(pool_size=5))
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(25, activation='elu'))
    model.add(Dropout(0.3))

    model.add(Dense(6, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    return model


def save_model(model):
    """Saves the learned model architecture and hyperparameters to an HDF5 file which can be loaded with Keras's
    load_model function"""
    now = datetime.datetime.now()
    title = now.strftime("%Y-%m-%d_%H%M")
    model.save('model_' + title)
    print('Model saved to working directory')

    return None


def evaluate_model(model, history, accuracy, y_test, y_prediction, save_confusion):
    """Generates all performance metrics"""
    # print accuracy as a percentage
    percent_accuracy = accuracy * 100.0
    print('Accuracy =', percent_accuracy, "%")

    # print confusion matrix
    matrix = confusion_matrix(y_test.argmax(
        axis=1), y_prediction.argmax(axis=1))
    print('Confusion Matrix:')
    print(np.matrix(matrix))

    # save confusion matrix
    if save_confusion:
        squeezed_confusion_matrix = np.squeeze(np.asarray(matrix))
        now = datetime.datetime.now()
        title = now.strftime("%Y-%m-%d_%H%M-%S")
        np.savetxt(title + '.csv', squeezed_confusion_matrix,
                   delimiter=',', fmt='%d')
        print("Confusion matrix saved to working directory")

    # print classification report
    target_names = ['1', '2', '3', 'REM', 'awake']
    print('Classification Report:')
    print(classification_report(y_test.argmax(axis=1),
          y_prediction.argmax(axis=1), target_names=target_names))

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

    return None


# starts main if file called as script (rather than imported)
if __name__ == "__main__":
    main()
