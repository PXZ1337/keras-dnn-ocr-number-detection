import os

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import center_of_mass

from.config import IMAGE_DIR, MNIST_IMAGE_WIDTH, MNIST_IMAGE_HEIGHT

import cv2


"""
This class implements the required steps which were also applied to the images in the MNIST dataset. 
This is necessary to reach the best possible result for the prediction.
"""
class ImagePreprocesser: 
    image_path: str = os.path.join(IMAGE_DIR, "temp_image.jpg")
    image: np.ndarray = None

    # Save image as greyscaled
    def __init__(self, pixmap: Any):
        pixmap.save(self.image_path)
        self.image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)

    def process(self, debug: bool = False):
        if self.image is not None:
            self.__resize()
            self.__normalize()
            self.__center()

            if debug:
                self.preview()

            return self.image

    def preview(self):
        plt.imshow(self.image)
        plt.show()

    def __resize(self):
        self.image = cv2.resize(self.image, (MNIST_IMAGE_WIDTH, MNIST_IMAGE_HEIGHT))

    # Invert the colors (dark background, bright digits)
    def __normalize(self):
        _, self.image = cv2.threshold(self.image, 127, 255, cv2.THRESH_BINARY_INV)

    # Center the painted
    def __center(self):
        cy, cx = center_of_mass(self.image)
        rows, cols = self.image.shape
        shiftx = np.round(cols / 2 - cx).astype(int)
        shifty = np.round(rows / 2 - cy).astype(int)
        M = np.array([[1, 0, shiftx], [0, 1, shifty]]).astype(np.float32)
        self.image = cv2.warpAffine(self.image, M, (cols, rows))


