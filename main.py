import sys

from PyQt5 import QtWidgets

from src import DeepNeuralNetwork, ApplicationUi

def application_gui() -> int:
    app = QtWidgets.QApplication(sys.argv)
    main_window = ApplicationUi()
    main_window.show()
    sys.exit(app.exec_())

def build_and_train_mnist_model(): 
    mnist_deep_neuronal_network = DeepNeuralNetwork(
        num_features=784,
        num_targets=10,
        learning_rate=0.0005,
        train_batch_size=256,
        train_epochs=15
    )

    mnist_deep_neuronal_network.train()

if __name__ == "__main__":
    build_and_train_mnist_model()
    application_gui()
    