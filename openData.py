# NOTES
# print(annsamp.__dict__)        #can be used to list attributes and values of object
# fs = sig.fs                    #can be used to find sampling rate of 250 Hz
# most signals from c4-al of brain; several are taken at other locations

import numpy as np
import os
import wfdb
import csv
from collections import defaultdict
from sklearn import preprocessing

# change to current OS
operatingSystem = 'macOS'

if operatingSystem is 'linux' or operatingSystem is 'macOS':
    dataPath = '/data/mit-bih-polysomnographic-database-1.0.0/'
    inputsPath = '/data/inputs/'
    targetsPath = '/data/5-shot-targets/'
else:
    dataPath = '\\data\\mit-bih-polysomnographic-database-1.0.0\\'
    inputsPath = '\\data\\inputs\\'
    targetsPath = "\\data\\5-shot-targets\\"

recordList = wfdb.get_record_list('slpdb')

annotationDict = defaultdict(lambda: 4, {
    # 'H': 0,
    # 'HA': 0,
    # 'OA': 0,
    # 'CA': 0,
    # 'CAA': 0,
    # 'X': 0,
    '1': 0,
    '2': 1,
    '3': 2,
    '4': 2,
    'R': 3
})

# classes = defaultdict(lambda: '8', { # 9 shot
#     '01000': '0',
#     '00100': '1',
#     '00010': '2',
#     '00001': '3',
#     '11000': '4',
#     '10100': '5',
#     '10010': '6',
#     '10001': '7',
#     '00000': '8'
# })

classes = defaultdict(lambda: '6', {
    '1000': '1',
    '0100': '2',
    '0010': '3',
    '0001': '4',
    '0000': '5'
})

for recordIndex, record in enumerate(recordList):

    # read the annotations and data then create a record
    annsamp = wfdb.rdann(os.getcwd() + dataPath + record, extension='st', summarize_labels=True)
    sig = wfdb.rdrecord(os.getcwd() + dataPath + record, channels=[2])
    actualPSignal = sig.p_signal
    actualPSignal = preprocessing.scale(actualPSignal)

    # remove unannotated epochs from start of record and split into epochs
    numberAnnotations = len(annsamp.aux_note)
    startingIndex = int((len(actualPSignal) / 7500) - numberAnnotations)*7500
    actualPSignal = actualPSignal[startingIndex:]
    epochs = np.split(actualPSignal, numberAnnotations)

    # 5 shot classification target: N1, N2, N3, REM, and wake
    target = [[0]*4 for _ in range(numberAnnotations)]

    # annotate each epoch and write its data and annotation to separate files
    for annotationIndex in range(numberAnnotations):
        labels = annsamp.aux_note[annotationIndex].split(' ')
        for label in labels:
            if annotationDict[label] is not 4:
                target[annotationIndex][annotationDict[label]] = 1

        with open(os.getcwd() + targetsPath + str(recordIndex) + '_' + str(annotationIndex) + ".csv", "w") as fileHandler:
            eventClass = ''.join([str(v) for v in target[annotationIndex]])
            eventClass = classes[eventClass]
            fileHandler.write(eventClass)

    print("\r {:2d}/{:2d}".format(recordIndex + 1, len(recordList)), end=" ")
