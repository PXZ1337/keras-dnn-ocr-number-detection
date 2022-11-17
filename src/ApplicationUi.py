import os
import matplotlib.pyplot as plt

from PyQt5 import QtCore
from PyQt5 import QtGui
from PyQt5 import QtWidgets
from PyQt5 import uic

from config import GUI_DIR, IMAGE_DIR
from Shape import Shapes
from Painter import Painter
from ImagePreprocesser import ImagePreprocesser
from dnn.DeepNeuralNetwork import load_mnist_model
from cnn.ConvNeuralNetwork import load_mnist_cnn_model

# Load the UI File
GUI_MODEL = os.path.join(GUI_DIR, "GUI.ui")
FORM, BASE = uic.loadUiType(GUI_MODEL)

class ApplicationUi(BASE, FORM):
    DrawingShapes = Shapes()
    PaintPanel = 0
    IsPainting = False
    ShapeNum = 0

    def __init__(self):
        # Set up main window and widgets
        super().__init__()
        self.setupUi(self)
        self.setObjectName("Rig Helper")
        self.PaintPanel = Painter(self)
        self.PaintPanel.close()
        self.DrawingFrame.insertWidget(0, self.PaintPanel)
        self.DrawingFrame.setCurrentWidget(self.PaintPanel)
        self.default_image_path = self.getImageByName(-1)

        self.preparePredictionMapForDNN()
        self.preparePredictionMapForCNN()

        self.Clear_Button.clicked.connect(self.clear)
        self.Predict_Button.clicked.connect(self.predict)
        self.dnn_model = load_mnist_model()
        self.cnn_model = load_mnist_cnn_model()

    def preparePredictionMapForDNN(self):
        self.headline_dnn = QtWidgets.QLabel(self)
        self.headline_dnn.setText("Prediction DNN")
        self.headline_dnn.setGeometry(QtCore.QRect(460, 40, 280, 30))

        self.label_dnn = QtWidgets.QLabel(self)
        self.label_dnn.setGeometry(QtCore.QRect(460, 70, 280, 280))

        self.pixmap_dnn = QtGui.QPixmap(self.default_image_path)
        self.label_dnn.setPixmap(self.pixmap_dnn)

    def preparePredictionMapForCNN(self):
        self.headline_cnn = QtWidgets.QLabel(self)
        self.headline_cnn.setText("Prediction CNN")
        self.headline_cnn.setGeometry(QtCore.QRect(780, 40, 280, 30))

        self.label_cnn = QtWidgets.QLabel(self)
        self.label_cnn.setGeometry(QtCore.QRect(780, 70, 280, 280))

        self.pixmap_cnn = QtGui.QPixmap(self.default_image_path)
        self.label_cnn.setPixmap(self.pixmap_cnn)

    def clear(self):
        self.DrawingShapes = Shapes()
        self.PaintPanel.repaint()  # type: ignore

        self.pixmap_dnn = QtGui.QPixmap(self.default_image_path)
        self.label_dnn.setPixmap(self.pixmap_dnn)

        self.pixmap_cnn = QtGui.QPixmap(self.default_image_path)
        self.label_cnn.setPixmap(self.pixmap_cnn)

    def predict(self):
        image_preprocesser = ImagePreprocesser(self.DrawingFrame.grab())
        image = image_preprocesser.process()

        # Predict dnn
        self.pixmap_dnn = QtGui.QPixmap(self.getImageByName(self.dnn_model.nn_predict(image)))
        self.label_dnn.setPixmap(self.pixmap_dnn)

        # Predict cnn
        self.pixmap_cnn = QtGui.QPixmap(self.getImageByName(self.cnn_model.nn_predict(image)))
        self.label_cnn.setPixmap(self.pixmap_cnn)

    def getImageByName(self, name: str) -> str:
        return os.path.join(IMAGE_DIR, str(name) + ".png")


