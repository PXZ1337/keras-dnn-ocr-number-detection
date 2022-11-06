import sys, os
import matplotlib.pyplot as plt

from PyQt5 import QtCore
from PyQt5 import QtGui
from PyQt5 import QtWidgets
from PyQt5 import uic

from .config import GUI_DIR, IMAGE_DIR
from .Shape import Shapes
from .Painter import Painter
from .ImagePreprocesser import ImagePreprocesser
from .dnn.mnist import load_mnist_model

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
        
        # Set up Label for on hold picture
        self.label = QtWidgets.QLabel(self)
        self.label.setGeometry(QtCore.QRect(460, 70, 280, 280))
        default_image_path = os.path.join(IMAGE_DIR, str(-1) + ".png")
        self.pixmap = QtGui.QPixmap(default_image_path)
        self.label.setPixmap(self.pixmap)

        self.Clear_Button.clicked.connect(self.ClearSlate)
        self.Predict_Button.clicked.connect(self.PredictNumber)
        self.model = load_mnist_model()

    def ClearSlate(self):
        self.DrawingShapes = Shapes()
        self.PaintPanel.repaint()  # type: ignore
        default_image_path = os.path.join(IMAGE_DIR, str(-1) + ".png")
        self.pixmap = QtGui.QPixmap(default_image_path)
        self.label.setPixmap(self.pixmap)

    def PredictNumber(self):
        image_preprocesser = ImagePreprocesser(self.DrawingFrame.grab())
        image = image_preprocesser.process()
        y_pred_class_idx = self.model.nn_predict(image)
        image_file_path = os.path.join(
            IMAGE_DIR, str(y_pred_class_idx) + ".png"
        )
        print(image_file_path)
        self.pixmap = QtGui.QPixmap(image_file_path)
        self.label.setPixmap(self.pixmap)