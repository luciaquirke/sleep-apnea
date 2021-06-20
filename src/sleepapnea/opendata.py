# NOTES
# print(annsamp.__dict__)        #can be used to list attributes and values of object
# fs = sig.fs                    #can be used to find sampling rate of 250 Hz

import numpy as np
import os
import wfdb
import csv
from ..utils.utils import get_record_data, setup_directory, get_physionet_data


def main() -> None:
    inputs_path, targets_path, data_path = setup_directory(classifier='2shot')

    if not os.path.exists(data_path):
        get_physionet_data()

    record_list = wfdb.get_record_list('slpdb')
    # apnea_labels = np.array(['H', 'HA', 'OA', 'CA', 'CAA', 'X'])
    apnea_labels = ['H', 'HA', 'OA', 'CA', 'CAA', 'X']

    for record_number, record in enumerate(record_list):
        epochs, annotations = get_record_data(os.path.join(data_path, record))

        # binarize data: 1 for a sleep apnea event, 0 for a non-apnoea event
        for idx, event in enumerate(annotations):
            for x in apnea_labels:
                if x in event:
                    annotations[idx] = 1
                    break
                else:
                    annotations[idx] = 0

        for epoch_number, epoch in enumerate(epochs):
            # write input and target data to files
            record_name = get_record_name(record_number, epoch_number)

            # save input and output arrays as csv files
            with open(os.path.join(inputs_path, record_name), 'w') as filehandler:
                csv_writer = csv.writer(filehandler, delimiter=' ')
                csv_writer.writerows(epoch)

            # write target values to csv files, named by record number and epoch number
            with open(os.path.join(targets_path, record_name), "w") as filehandler:
                csv_writer = csv.writer(filehandler, delimiter=' ')
                csv_writer.writerow(str(annotations[epoch_number]))


if __name__ == "__main__":
    main()


def get_record_name(record_number, epoch_number) -> str:
    return str(record_number+1) + '_' + \
        str(epoch_number+1) + '.csv'
