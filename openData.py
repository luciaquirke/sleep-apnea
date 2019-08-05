# NOTES
# print(annsamp.__dict__)        #can be used to list attributes and values of object
# fs = sig.fs                    #can be used to find sampling rate of 250 Hz

from IPython.display import display
import numpy as np
import os
import wfdb
import csv
from sklearn import preprocessing

# change to current OS
operatingSystem = 'macOS'

if operatingSystem == 'linux' or operatingSystem == 'macOS':
    dataPath = '/data/mit-bih-polysomnographic-database-1.0.0/'
    inputsPath = '/data/inputs/'
    targetsPath = '/data/targets/'
else:
    dataPath = '\\data\\mit-bih-polysomnographic-database-1.0.0\\'
    inputsPath = '\\data\\inputs\\'
    targetsPath = "\\data\\targets\\"
    
recordList = wfdb.get_record_list('slpdb')
apneaLabels = np.array(['H', 'HA', 'OA', 'CA', 'CAA', 'X']) # apnea classifiers
sleepStageLabels = np.array(['W', '1', '2', '3', '4', '5']) # sleep stages

inputArray = np.empty((0,1))
outputArray = np.empty((0,1))
recordNumber = 1

for record in recordList:

    # Read the annotations
    annsamp = wfdb.rdann(os.getcwd() + dataPath + record, extension='st', summarize_labels=True)
    
    # Read the data, create a Record
    sig = wfdb.rdrecord(os.getcwd() + dataPath + record, channels=[2]) 
    
    actualPSignal = sig.p_signal
    
    # binarize data: 1 for a sleep apnea event in the epoch
    for idx, event in enumerate(annsamp.aux_note): 
        for x in apneaLabels:
            if x in event:
                annsamp.aux_note[idx] = 1
                break
            else:
                annsamp.aux_note[idx] = 0

    # remove unannotated epochs from start of record
    numberEpochs = len(actualPSignal)/7500
    if numberEpochs != len(annsamp.aux_note):
        startingIndex = int(numberEpochs - len(annsamp.aux_note))
        actualPSignal = actualPSignal[startingIndex*7500:]

    # standardise signal
    actualPSignal = preprocessing.scale(actualPSignal)

    # write each epoch to a csv file, named by record number and epoch number
    trimmedNumberEpochs = len(actualPSignal)/7500
    print(trimmedNumberEpochs)
    epochs = np.split(actualPSignal, trimmedNumberEpochs)
    index = 1

    for epoch in epochs:
        # save input and output arrays as csv files
        with open(os.getcwd() + inputsPath + str(recordNumber) + '_' + str(index) + '.csv', 'w') as filehandler:
            csvWriter = csv.writer(filehandler, delimiter=' ')
            csvWriter.writerows(epoch) 

            # write target values to csv files, named by record number and epoch number
        with open(os.getcwd() + targetsPath + str(recordNumber) + '_' + str(index) + ".csv", "w") as filehandler:
            csvWriter = csv.writer(filehandler, delimiter=' ')
            csvWriter.writerow(str(annsamp.aux_note[index-1]))
        
        index = index + 1

    recordNumber = recordNumber + 1