# Social Robot Simulator

This is a social robot simulator. The robot will read your facial features and response to your faicial expression.

The robot has 3 major features:
- Have Eye Contact
- Mirror the facial expression
- Response to the user's general mood

## Installation

1. Use Anaconda to install the packages for running the program
```shell script
conda create -name ENVNAME
conda activate ENVNAME
conda install -c conda-forge opencv # Install opencv
conda install -c conda-forge dlib
```

2. Run the robot simulation using Python3
```shell script
python3 main.py
```

## Credit
This project use `dlib` and `opencv` for the computer vision function
