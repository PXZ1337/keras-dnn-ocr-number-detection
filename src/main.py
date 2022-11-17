import sys

from PyQt5 import QtWidgets


from dnn.DeepNeuralNetwork import DeepNeuralNetwork
from cnn.ConvNeuralNetwork import ConvNeuralNetwork
from ApplicationUi import ApplicationUi

def application_gui() -> int:
    app = QtWidgets.QApplication(sys.argv)
    main_window = ApplicationUi()
    main_window.show()
    sys.exit(app.exec_())

def build_and_train_dnn_model():
    mnist_dnn = DeepNeuralNetwork(
        num_features=784,
        num_targets=10,
        learning_rate=0.0005,
        train_batch_size=256,
        train_epochs=30
    )

    mnist_dnn.train()

def build_and_train_cnn_model():
    mnist_cnn = ConvNeuralNetwork(
        train_batch_size=256,
        train_epochs=30
    )

    mnist_cnn.train()

if __name__ == "__main__":
    build_and_train_dnn_model()
    build_and_train_cnn_model()
    application_gui()
