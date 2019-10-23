import numpy as np
import os
import datetime
from glob import glob
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold

from keras.callbacks import EarlyStopping

from foldGenerator import FoldGenerator
from modelArchitecture import define_model


def get_input_filenames(inputs_file_path):
    """Returns list of data filenames by recursive search of inputs path"""
    print("Loading Data...")

    files = list(glob(os.path.join(inputs_file_path, "**", "*.csv"), recursive=True))
    files = [os.path.basename(filename) for filename in files]

    print("{:} data-points found".format(len(files)))

    return files


def main():
    """Runs model training with 10-fold data and saves model and confusion matrices"""
    inputs_path = 'data/inputs/'
    targets_path = 'data/5-shot-targets/'
    files = get_input_filenames(inputs_path)

    cv = KFold(n_splits=10, shuffle=True, random_state=3)

    for train_index, test_index in cv.split(files):

        # creates validation/training data split
        validation_index = train_index[0:917]  # hardcoded to 10% of the data
        train_index = train_index[917:-1]

        # generates training/test/validation data in batches
        train_generator = FoldGenerator(inputs_path, targets_path, files[train_index[0]:train_index[-1]])
        validation_generator = FoldGenerator(inputs_path, targets_path, files[validation_index[0]:validation_index[-1]])
        test_generator = FoldGenerator(inputs_path, targets_path, files[test_index[0]:test_index[-1]])

        # defines model architecture and set training parameters
        model = define_model()

        # model training
        es = EarlyStopping(monitor='val_acc', mode='max', patience=5, verbose=1, restore_best_weights=True)
        history = model.fit_generator(train_generator, validation_data=validation_generator, epochs=100, verbose=1,
                                      callbacks=[es])
        _, accuracy = model.evaluate_generator(test_generator, verbose=0)

        # saves the model, named by time/date
        model.save('model_' + title)
        print('Model Saved')

        # tests model performance
        x_test = []
        y_test = []
        test_generator.on_epoch_end()
        for i in range(len(test_generator)):
            x_batch, y_batch = test_generator[i]
            x_test.extend(x_batch)
            y_test.extend(y_batch)

        x_test = np.array(x_test)
        y_test = np.array(y_test)

        y_pred = model.predict(x_test)

        # generates confusion matrix
        matrix = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
        print('Confusion Matrix:')
        print(np.matrix(matrix))

        # writes each confusion matrix to a csv file
        squeezed_confusion_matrix = np.squeeze(np.asarray(matrix))
        now = datetime.datetime.now()
        title = now.strftime("%Y-%m-%d_%H%M-%S")
        np.savetxt(title + '.csv', squeezed_confusion_matrix, delimiter=',', fmt='%d')
        print('Confusion matrix saved to working directory')

        return None


# starts main if file called as script (rather than imported)
if __name__ == "__main__":
    main()


