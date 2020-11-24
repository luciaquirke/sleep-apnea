# sleep-apnea

Convolutional neural network which takes as input 30 second single-channel EEG signals and produces as output a sleep stage classification. Possible classifications are N1, N2, N3, REM, and W (awake). Includes implementation, validation, and example trained model.

### Prerequisites

* NumPy
* Pandas
* scikit-learn
* WFDB
* TensorFlow
* Keras
* h5py
* Matplotlib

### Data

All models were trained using the open-access [MIT-BIH Polysomnographic Database](https://physionet.org/content/slpdb/1.0.0/). 
