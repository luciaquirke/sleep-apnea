import numpy as np
from pathlib import Path
import wfdb
import os
from collections import defaultdict
from sklearn import preprocessing
from typing import Tuple
import subprocess
from os import fspath


def main() -> None:
    """Loads relevant data from PhysioBank using wfdb package specified in documentation and saves it to folders"""

    annotation_dict = defaultdict(lambda: 'error', {
        '1': [1, 0, 0, 0],
        '2': [0, 1, 0, 0],
        '3': [0, 0, 1, 0],
        '4': [0, 0, 1, 0],
        'R': [0, 0, 0, 1],
    })

    classes = defaultdict(lambda: '6', {
        '1000': '1',
        '0100': '2',
        '0010': '3',
        '0001': '4',
        '0000': '5'
    })

    class_count = defaultdict(lambda: 0, {
        '1000': 0,
        '0100': 0,
        '0010': 0,
        '0001': 0,
        '0000': 0
    })

    inputs_path, targets_path, data_path = folder_setup()

    if not os.path.exists(data_path):
        get_physionet_data()

    record_list = wfdb.get_record_list('slpdb')

    for record_index, record in enumerate(record_list):
        # retrieve the annotations and data
        record_path = os.path.join(data_path, record)
        inputs, annotations = preprocess_data(record_path)

        # initialise each 5 shot classification target (N1, N2, N3, REM, and wake) as 0000
        target = [[0]*4 for _ in range(len(annotations))]

        # annotate each input and write its data and annotation to separate files
        for annotation_index in range(len(annotations)):
            labels = annotations[annotation_index].split(' ')
            for label in labels:
                if annotation_dict[label] != 'error':
                    target[annotation_index] = annotation_dict[label]

            # write each input to a csv file, named by record number and input number
            record_name = str(record_index) + '_' + \
                str(annotation_index) + '.csv'

            with open(os.path.join(inputs_path, record_name), 'w') as filehandler:
                filehandler.write("\n".join(str(num)
                                  for num in inputs[annotation_index]))

            with open(os.path.join(targets_path, record_name), "w") as fileHandler:
                event_class = ''.join([str(v)
                                       for v in target[annotation_index]])
                class_count[event_class] += 1
                event_class = classes[event_class]

                fileHandler.write(event_class)

        print("\r {:2d}/{:2d}".format(record_index +
              1, len(record_list)), end=" ")

    # class count statistic
    for key, value in class_count.items():
        print("\n")
        print(key, value)


def preprocess_data(record_path: str):
    # retrieve signal and annotation from memory
    annotations = wfdb.rdann(record_path,
                             extension='st', summarize_labels=True).aux_note
    signal = wfdb.rdrecord(record_path, channels=[2]).p_signal

    signal = preprocessing.scale(signal)

    # remove unannotated epochs (30 second input segments) from the start of the record and split into inputs
    starting_index = int(
        (len(signal) / 7500) - len(annotations))*7500
    signal = signal[starting_index:]
    inputs = np.split(signal, len(annotations))

    return inputs, annotations


def folder_setup(classifier='5shot') -> Tuple[str, str, str]:
    """Specifies the directory structures and pathways, creating any missing directories"""
    data_path = Path(
        'data/physionet.org/files/slpdb/1.0.0')
    inputs_path = Path('data/inputs/')
    targets_path = Path('data/5-shot-targets/')

    if classifier == '2shot':
        targets_path = Path('data/targets/')

    data_path.mkdir(parents=True, exist_ok=True)
    inputs_path.mkdir(parents=True, exist_ok=True)
    targets_path.mkdir(parents=True, exist_ok=True)

    print('Directories set up!')
    return inputs_path.absolute(), targets_path.absolute(), data_path.absolute()


def get_physionet_data() -> None:
    print('Retrieving physionet data, please allow a few minutes...')
    subprocess.call(['sh', 'src/scripts/get-physionet-data.sh'])
    print('Data retrieved successfully')


# starts main if file called as script (rather than imported)
if __name__ == "__main__":
    main()
