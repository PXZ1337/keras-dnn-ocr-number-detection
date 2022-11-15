from typing import Tuple

import numpy as np

from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

class MnistDataset:
    def __init__(self):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        num_features = x_train.shape[1]*x_train.shape[2]

        # Preprocess x data
        self.x_train = x_train.reshape(-1, num_features).astype(np.float32)
        self.x_test = x_test.reshape(-1, num_features).astype(np.float32)

        # Normalize (0:1)
        self.x_train = self.x_train / 255.0
        self.x_test = self.x_test / 255.0
        
        # Dataset attributes (size,width,height,depth)
        self.train_size = self.x_train.shape[0]
        self.test_size = self.x_test.shape[0]
        self.width = self.x_train.shape[1]
        self.image_shape = (num_features)
        self.num_classes = len(np.unique(y_test))

        # Preprocess y data
        self.y_train = to_categorical(y_train, num_classes=self.num_classes, dtype=np.float32)
        self.y_test = to_categorical(y_test, num_classes=self.num_classes, dtype=np.float32)

    def get_train_set(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.x_train, self.y_train

    def get_test_set(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.x_test, self.y_test

    def get_dataset(self) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        return (self.x_train, self.x_test), (self.y_train, self.y_test)

    def __str__(self) -> str:
        return f"Shape: {self.image_shape}, TrainSize: {self.train_size}, TestSize: {self.test_size}, NumClasses: {self.num_classes}"