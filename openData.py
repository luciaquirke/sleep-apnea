#NOTES
#print(annsamp.__dict__)        #can be used to list attributes and values of object
#fs = sig.fs                    #can be used to find sampling rate of 250 Hz

from IPython.display import display
import numpy as np
import os
import wfdb
import csv

operatingSystem = 'linux'

if operatingSystem == 'linux':
    dataPath = '/data/mit-bih-polysomnographic-database-1.0.0/'
    inputsPath = '/data/inputs/'
else:
    dataPath = '\\data\\mit-bih-polysomnographic-database-1.0.0\\'
    inputsPath = '\\data\\inputs\\'
    
recordList = wfdb.get_record_list('slpdb')
apneaLabels = np.array(['H', 'HA', 'OA', 'CA', 'CAA', 'X'])
inputArray = np.empty((0,1))
outputArray = np.empty((0,1))
recordNumber = 1

for record in recordList:
    print('new record')

    #Read the annotations 
    annsamp = wfdb.rdann(os.getcwd() + dataPath + record, extension='st', summarize_labels=True)
    
    #Read the data, create a Record
    sig = wfdb.rdrecord(os.getcwd() + dataPath + record, channels=[2]) 
    
    actualPSignal = sig.p_signal
    
    #binarize data: 1 for a sleep apnea event in the epoch
    for idx, event in enumerate(annsamp.aux_note): 
        for x in apneaLabels:
            if x in event:
                annsamp.aux_note[idx] = 1
                break
            else:
                annsamp.aux_note[idx] = 0

    #remove unannotated epochs from start of record
    numberEpochs = len(actualPSignal)/7500
    if numberEpochs != len(annsamp.aux_note):
        startingIndex = int(numberEpochs - len(annsamp.aux_note))
        actualPSignal = actualPSignal[startingIndex*7500:]

    #add all epochs to an array, row by row
    trimmedNumberEpochs = len(actualPSignal)/7500
    epochs = np.split(actualPSignal, trimmedNumberEpochs)
    index = 1

    for epoch in epochs:

        #save input and output arrays as csv files
        with open(os.getcwd() + inputsPath + str(recordNumber) + '_' + str(index) + '.csv','w') as filehandler: #/data/inputs/
            csvWriter = csv.writer(filehandler,delimiter=' ')
            csvWriter.writerows(epoch)
        
        index = index + 1

        #with open("/home/anaconda3/envs/sleep-apnea/data/targets/" + str(index) + ".csv","w") as filehandler:
        #    csvWriter = csv.writer(filehandler,delimiter=' ')
        #    csvWriter.writerows(outputArray)
    
    recordNumber = recordNumber + 1