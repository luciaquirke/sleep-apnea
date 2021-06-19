import numpy as np
from pathlib import Path
import wfdb
import os
from collections import defaultdict
from sklearn import preprocessing
from typing import Tuple
import requests
import wget
import subprocess


def main() -> None:
    """Loads relevant data from PhysioBank using wfdb package specified in documentation and saves it to folders"""

    annotation_dict = defaultdict(lambda: 5, {
        '1': 0,
        '2': 1,
        '3': 2,
        '4': 2,
        'R': 3
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
        print('not')
        get_physionet_data()

    record_list = wfdb.get_record_list('slpdb')

    for record_index, record in enumerate(record_list):
        # read the annotations and data then create a record
        annsamp = wfdb.rdann(str(data_path / record),
                             extension='st', summarize_labels=True)
        signal = wfdb.rdrecord(str(data_path / record), channels=[2])
        physical_signal = signal.p_signal
        physical_signal = preprocessing.scale(physical_signal)

        # remove unannotated epochs (30 second input segments) from the start of the record and split into inputs
        number_annotations = len(annsamp.aux_note)
        starting_index = int(
            (len(physical_signal) / 7500) - number_annotations)*7500
        physical_signal = physical_signal[starting_index:]
        inputs = np.split(physical_signal, number_annotations)

        # generate each 5 shot classification target as 0000: N1, N2, N3, REM, and wake
        target = [[0]*4 for _ in range(number_annotations)]

        # annotate each input and write its data and annotation to separate files
        for annotation_index in range(number_annotations):
            labels = annsamp.aux_note[annotation_index].split(' ')
            for label in labels:
                if annotation_dict[label] != 5:
                    target[annotation_index][annotation_dict[label]] = 1

            # write each input to a csv file, named by record number and input number
            with open(str(inputs_path / (str(record_index) + '_' + str(annotation_index) + '.csv')),
                      'w') as filehandler:
                filehandler.write("\n".join(str(num)
                                  for num in inputs[annotation_index]))

            with open(str(targets_path / (str(record_index) + '_' + str(annotation_index) + ".csv")), "w") as \
                    fileHandler:
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


def folder_setup() -> Tuple[str, str, str]:
    """Specifies the folder structures and pathways, creating any missing folders"""
    data_path = Path(
        'data/physionet.org/files/slpdb/1.0.0')
    inputs_path = Path('data/inputs/')
    targets_path = Path('data/5-shot-targets/')

    data_path.mkdir(parents=True, exist_ok=True)
    inputs_path.mkdir(parents=True, exist_ok=True)
    targets_path.mkdir(parents=True, exist_ok=True)

    return inputs_path.absolute(), targets_path.absolute(), data_path.absolute()


def get_physionet_data():
    print('Retrieving physionet data, please allow a few minutes...')
    subprocess.call(['sh', 'src/scripts/get-physionet-data.sh'])
    print('Data retrieved successfully')


# starts main if file called as script (rather than imported)
if __name__ == "__main__":
    main()
