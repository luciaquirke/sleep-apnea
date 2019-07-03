from IPython.display import display
import numpy as np
import os
import shutil
import wfdb

recordList = wfdb.get_record_list('slpdb')
apneaLabels = np.array(['H', 'HA', 'OA', 'CA', 'CAA', 'X'])
allData = []
index = 0

for record in recordList:
    print(record)
    #Read the annotations 
    annsamp = wfdb.rdann(os.getcwd() + '\\data\\mit-bih-polysomnographic-database-1.0.0\\' + record, extension='st', summarize_labels=True)
    
    #Read the data, create a Record
    sig = wfdb.rdrecord(os.getcwd() + '\\data\\mit-bih-polysomnographic-database-1.0.0\\' + record, channels=[2])

    startAnn = annsamp.sample[0]

    actualPSignal = sig.p_signal
    print(actualPSignal)
    fs = sig.fs #sampling rate is 250 Hz

    dataList = []


    for idx, event in enumerate(annsamp.aux_note):
        for x in apneaLabels:
            if x in event:
                annsamp.aux_note[idx] = 1
                break
            else:
                annsamp.aux_note[idx] = 0

    numberEpochs = len(actualPSignal)/7500
    signalEpochs = np.split(actualPSignal, numberEpochs)
    if numberEpochs != len(annsamp.aux_note):
        print('Annotations:', len(annsamp.aux_note))
        print('Epochs:', numberEpochs)

    #ERROR: annsamp.aux_note[i] is causing a 'list index out of range' error. It does not happen for every 
    #record, it occurs first at record number 7 (slp14). CAUSE: This record has 720 epochs and only 714 annotations. 
    #From ploting the data it seems that these missing annotations are at the start of the record. Also occurs for
    #slp16, slp37, slp59, slp61 and slp66.

    # i = 0
    # for epoch in signalEpochs:
    #     dataList.append([[epoch], [annsamp.aux_note[i]], [index]])
    #     i+=1
    #     index+=1
    # print(i)
