# NOTES
# print(annsamp.__dict__)        #can be used to list attributes and values of object
# fs = sig.fs                    #can be used to find sampling rate of 250 Hz

import numpy as np
import os
import wfdb
import csv
from collections import defaultdict
from sklearn import preprocessing

# change to current OS
operatingSystem = 'macOS'

if operatingSystem is 'linux' or 'macOS':
    dataPath = '/data/mit-bih-polysomnographic-database-1.0.0/'
    inputsPath = '/data/inputs/'
    targetsPath = '/data/6-shot-targets/'
else:
    dataPath = '\\data\\mit-bih-polysomnographic-database-1.0.0\\'
    inputsPath = '\\data\\inputs\\'
    targetsPath = "\\data\\6-shot-targets\\"

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

for recordNumber, record in enumerate(recordList):

    # read the annotations
    annsamp = wfdb.rdann(os.getcwd() + dataPath + record, extension='st', summarize_labels=True)

    # read the data, create a Record
    sig = wfdb.rdrecord(os.getcwd() + dataPath + record, channels=[2])
    actualPSignal = sig.p_signal

    # remove unannotated epochs from start of record
    numberEpochs = len(actualPSignal) / 7500
    numberAnnotations = len(annsamp.aux_note)

    if numberEpochs != numberAnnotations:
        startingIndex = int(numberEpochs - numberAnnotations)
        actualPSignal = actualPSignal[startingIndex * 7500:]

    # standardise signal
    actualPSignal = preprocessing.scale(actualPSignal)

    # 5 shot classification target: first digit is apnea, next four are N1, N2, N3, and REM
    target = np.zeros((numberAnnotations, 5)).astype(int)

    for i, note in enumerate(annsamp.aux_note):
        labels = note.split(' ')
        for label in labels:
            if annotationDict[label] is not 5:
                target[i, annotationDict[label]] = 1

    # write each epoch to a csv file, named by record number and epoch number
    trimmedNumberEpochs = len(actualPSignal) / 7500
    epochs = np.split(actualPSignal, trimmedNumberEpochs)

    for epochIndex, epoch in enumerate(epochs):
        # save input and output arrays as csv files
        with open(os.getcwd() + inputsPath + str(recordNumber) + '_' + str(epochIndex) + '.csv', 'w') as filehandler:
            csvWriter = csv.writer(filehandler, delimiter=' ')
            csvWriter.writerows(epoch)

            # write target values to csv files, named by record number and epoch number
        with open(os.getcwd() + targetsPath + str(recordNumber) + '_' + str(epochIndex) + ".csv", "w") as filehandler:
            csvWriter = csv.writer(filehandler, delimiter=' ')
            csvWriter.writerow(str(target[(epochIndex - 1), :]))
