# Deep Learning with Tensorflow, Keras and Python

Software for classifying hand-written digits with a neural network, trained with the mnist dataset [Info](https://keras.io/api/datasets/mnist/).

Simply run the `main.py` file.

- `build_and_train_mnist_model` will prepare the [mnist](https://keras.io/api/datasets/mnist/) dataset, create a Sequential-Model and train it with the given data. The model will be saved into the project directory, after the first execution the training isn't necessary anymore.
- `application_gui` will open an graphical user interface where you can draw digits on a canvas and predict them within the created model.

## Usage

Run `main.py`

## Requirements

- Anaconda with Python 3.8 [Download](https://www.anaconda.com/download/)
- Tensorflow 2.3+ for Python [Installation](https://www.tensorflow.org/install/)
- pyqt5, opencv-python, numpy, scipy and matplotlib for Python
