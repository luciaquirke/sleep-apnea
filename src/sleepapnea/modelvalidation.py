import numpy as np
import pandas as pd
import os

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold

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

from ..utils.utils import setup_directory


def main() -> None:
    inputs_path, targets_path, _ = setup_directory(classifier='2shot')

    # Set random seed
    seed = 7
    np.random.seed(seed)

    # load a list of files into a 3D array of [samples, timesteps, features]
    x_loaded = list()
    y_loaded = []

    print("Loading Data...")

    # load input and target .csv files
    for root, dirs, files in os.walk(os.path.join(inputs_path)):
        for file_name in files:
            x_data = load_file(os.path.join(inputs_path, file_name))
            x_loaded.append(x_data)
            y_data = load_file(os.path.join(targets_path, file_name))
            y_loaded.append(y_data)

    # data loaded in
    X = np.stack(x_loaded, axis=0)
    Y = y_loaded
    Y = np.array(Y)

    # change targets to one hot encoded form
    Y = to_categorical(Y)
    Y = Y.reshape(-1, 2)

    # create 10 splits within the data
    k_fold = KFold(n_splits=10, shuffle=True, random_state=seed)

    # initialise lists for saving results
    accuracy_list = list()
    cm_list = list()
    cr_list = list()
    dp_list = list()

    verbose, epochs, batch_size = 0, 30, 32
    fold = 1

    for train, test in k_fold.split(X, Y):
        print('Fold: ', fold)

        # Create model
        model = Sequential()

        # Convolutional Layer 2
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

        # train model with early stopping call back
        es = EarlyStopping(monitor='val_acc', mode='max',
                           patience=5, verbose=1, restore_best_weights=True)
        model.fit(X[train], Y[train], epochs=epochs, batch_size=batch_size,
                  verbose=verbose, validation_split=0.1, callbacks=[es])

        # test model
        accuracy = model.evaluate(
            X[test], Y[test], batch_size=batch_size, verbose=0)
        y_pred = model.predict(X[test])

        print('Accuracy =', accuracy[1]*100, "%")
        accuracy_list.append(accuracy[1]*100)

        # generate confusion matrix
        matrix = confusion_matrix(
            Y[test].argmax(axis=1), y_pred.argmax(axis=1))
        print('Confusion Matrix:')
        print(np.matrix(matrix))
        cm_list.append(matrix)

        # calculate d'
        tp, fn, fp, tn = matrix.ravel()
        dprime = norm.ppf(tn/(tn+fp)) - norm.ppf(fn/(tp+fn))
        print('dPrime =', dprime)
        dp_list.append(dprime)

        # generate classification report
        target_names = ['non-apnea', 'apnea']
        print('Classification Report:')
        cr = classification_report(Y[test].argmax(
            axis=1), y_pred.argmax(axis=1), target_names=target_names)
        print(cr)
        cr_list.append(cr)

        # save each model
        model_name = 'model' + str(fold) + '.h5'
        model.save(model_name)

        fold = fold + 1

    print('Mean accuracy = ', np.mean(accuracy_list),
          'Standard Deviation =', np.std(accuracy_list))
    print('Mean dPrime = ', np.mean(dp_list),
          'Standard Deviation =', np.std(dp_list))


# loads data from a single .csv file
def load_file(filepath):
    dataframe = pd.read_csv(filepath, header=None, delim_whitespace=True)
    return dataframe.values


if __name__ == "__main__":
    main()
