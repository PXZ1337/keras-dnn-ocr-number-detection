from PyQt5 import QtCore
from PyQt5 import QtGui
from PyQt5 import QtWidgets
from .Position import Position

class Painter(QtWidgets.QWidget):
    ParentLink = 0
    MouseLoc = Position(0, 0)
    LastPos = Position(0, 0)

    def __init__(self, parent):
        super().__init__()
        self.ParentLink = parent
        self.MouseLoc = Position(0, 0)
        self.LastPos = Position(0, 0)

    def mousePressEvent(self, event=None):
        self.ParentLink.IsPainting = True 
        self.ParentLink.ShapeNum += 1 
        self.LastPos = Position(0, 0)

    def mouseMoveEvent(self, event=None):
        if self.ParentLink.IsPainting is True:  
            self.MouseLoc = Position(event.x(), event.y())
            if (self.LastPos.x != self.MouseLoc.x) and (
                self.LastPos.y != self.MouseLoc.y
            ):
                self.LastPos = Position(event.x(), event.y())
                self.ParentLink.DrawingShapes.NewShape(
                    self.LastPos, self.ParentLink.ShapeNum
                )  
            self.repaint()

    def mouseReleaseEvent(self, event=None):
        if self.ParentLink.IsPainting is True:  
            self.ParentLink.IsPainting = False  

    def paintEvent(self, event):
        painter = QtGui.QPainter()
        painter.begin(self)
        self.drawLines(event, painter)
        painter.end()

    def drawLines(self, event, painter):
        for i in range(self.ParentLink.DrawingShapes.NumberOfShapes() - 1):  
            T = self.ParentLink.DrawingShapes.GetShape(i)  
            T1 = self.ParentLink.DrawingShapes.GetShape(i + 1)  
            if T.number == T1.number:
                pen = QtGui.QPen(QtGui.QColor(0, 0, 0), 7, QtCore.Qt.SolidLine)
                painter.setPen(pen)
                painter.drawLine(
                    T.location.x, T.location.y, T1.location.x, T1.location.y
                )
