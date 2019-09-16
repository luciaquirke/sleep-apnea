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

annotationDict = defaultdict(lambda: 5, {
    'H': 0,
    'HA': 0,
    'OA': 0,
    'CA': 0,
    'CAA': 0,
    'X': 0,
    '1': 1,
    '2': 2,
    '3': 3,
    '4': 3,
    'R': 4
})

classes = defaultdict(lambda: '8', {
    '01000': '0',
    '00100': '1',
    '00010': '2',
    '00001': '3',
    '11000': '4',
    '10100': '5',
    '10010': '6',
    '10001': '7',
    '00000': '8'
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

    # 5 shot classification target: first digit is apnea, next four are N1, N2, N3, and REM
    target = [[0]*5 for _ in range(numberAnnotations)]

    # annotate each epoch and write its data and annotation to separate files
    for annotationIndex in range(numberAnnotations):
        labels = annsamp.aux_note[annotationIndex].split(' ')
        for label in labels:
            if annotationDict[label] is not 5:
                target[annotationIndex][annotationDict[label]] = 1

        # write each epoch to a csv file, named by record number and epoch number
        with open(os.getcwd() + inputsPath + str(recordIndex + 1) + '_' + str(annotationIndex + 1) + '.csv', 'w') as fileHandler:
            csvWriter = csv.writer(fileHandler, delimiter=' ')
            csvWriter.writerows(epochs[annotationIndex])

        # write each target value to a csv file, named by record number and epoch number
        with open(os.getcwd() + targetsPath + str(recordIndex + 1) + '_' + str(annotationIndex + 1) + ".csv", "w") as fileHandler:
            eventClass = ''.join([str(v) for v in target[annotationIndex]])
            eventClass = classes[eventClass]
            fileHandler.write(eventClass)
