import numpy as np
from pathlib import Path
import wfdb
from sklearn import preprocessing
from typing import Tuple
import subprocess


def get_physionet_data() -> None:
    print('Retrieving physionet data, please allow a few minutes...')
    subprocess.call(['sh', 'src/scripts/get-physionet-data.sh'])
    print('Data retrieved successfully')


def setup_directory(classifier='5shot') -> Tuple[str, str, str]:
    """Specifies the directory structures and pathways, creating any missing directories"""
    data_path = Path(
        'data/physionet.org/files/slpdb/1.0.0')

    if classifier == '2shot':
        targets_path = Path('data/targets/')
        inputs_path = Path('data/inputs')
    else:
        inputs_path = Path('data/5-shot-inputs/')
        targets_path = Path('data/5-shot-targets/')

    data_path.mkdir(parents=True, exist_ok=True)
    inputs_path.mkdir(parents=True, exist_ok=True)
    targets_path.mkdir(parents=True, exist_ok=True)

    print('Directories set up!')
    return inputs_path.absolute(), targets_path.absolute(), data_path.absolute()


def get_record_data(record_path: str):
    # retrieve signal and annotation from memory
    annotations = wfdb.rdann(record_path,
                             extension='st', summarize_labels=True).aux_note
    signal = wfdb.rdrecord(record_path, channels=[2]).p_signal

    # standardise signal
    signal = preprocessing.scale(signal)

    # remove unannotated epochs (30 second input segments) from the start of the record and split into epochs
    number_epochs = int(len(signal)/7500)
    starting_index = (number_epochs - len(annotations))*7500
    signal = signal[starting_index:]
    epochs = np.split(signal, len(annotations))

    return epochs, annotations
