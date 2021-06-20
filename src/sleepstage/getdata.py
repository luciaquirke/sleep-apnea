import wfdb
import os
from collections import defaultdict
from ..utils.utils import get_record_data, setup_directory, get_physionet_data


def get_record_name(record_index: int, annotation_index: int) -> str:
    return str(record_index) + '_' + \
        str(annotation_index) + '.csv'


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

    inputs_path, targets_path, data_path = setup_directory()

    if not os.path.exists(data_path):
        get_physionet_data()

    record_list = wfdb.get_record_list('slpdb')

    for record_index, record in enumerate(record_list):
        epochs, annotations = get_record_data(os.path.join(data_path, record))

        for annotation_index in range(len(annotations)):
            # annotations may have several labels but only one is relevant
            labels = annotations[annotation_index].split(' ')
            target = [0, 0, 0, 0]
            for label in labels:
                if annotation_dict[label] != 'error':
                    target = annotation_dict[label]

            # write input and target data to files
            record_name = get_record_name(record_index, annotation_index)

            with open(os.path.join(inputs_path, record_name), 'w') as filehandler:
                filehandler.write("\n".join(str(num)
                                            for num in epochs[annotation_index]))

            with open(os.path.join(targets_path, record_name), "w") as fileHandler:
                event_class = ''.join([str(v)
                                       for v in target])
                class_count[event_class] += 1

                event_class = classes[event_class]
                fileHandler.write(event_class)

        print("\r {:2d}/{:2d}".format(record_index +
              1, len(record_list)), end=" ")

    # class count statistic
    for key, value in class_count.items():
        print("\n")
        print(key, value)


# starts main if file called as script (rather than imported)
if __name__ == "__main__":
    main()
