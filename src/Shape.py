from typing import Any
from Position import Position

class Shape:
    location = Position()
    number = 0

    def __init__(self, L: Any, S: Any):
        self.location = L
        self.number = S


class Shapes:
    shapes: list = []

    def __init__(self):
        self.shapes = []

    def NumberOfShapes(self):
        return len(self.shapes)

    def NewShape(self, L: Any, S: Any):
        shape = Shape(L, S)
        self.shapes.append(shape)

    def GetShape(self, Index) -> Any:
        return self.shapes[Index]
