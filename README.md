# sleep-apnea

Two packages:
- A convolutional neural network (CNN) which takes as input 30-second single-channel EEG signals and produces as output a sleep apnea classification.
- A CNN trained on the same data which classifies sleep stage. Possible classifications are N1, N2, N3, REM, and W (awake).

Both packages include implementation, validation, and an example trained model.

### Requirements

- Python 3.5-3.8 (you can use [pyenv](https://github.com/pyenv/pyenv) to manage different python versions).
- Training data from [Physionet](https://physionet.org/content/slpdb/1.0.0/).

### Setup

This project is best set up in a virtual environment to isolate package and language versions.

In the terminal, navigate into the project directory and create a new virtual environment using a compatible (3.5-3.8) version of Python:

```
python3.8 -m venv env
```

Activate virtual environment:

```
source env/bin/activate
```

Install required packages:

```
python -m pip install -r requirements.txt
```

You also need wget if you want to automate the downloading of Physionet data:

```
brew install wget
```

You're all set! Run `python3 -m src.sleepapnea.getdata` to retrieve and preprocess physionet data, and `python3 -m src.sleepapnea.modelarchitecture` to train your first model.
