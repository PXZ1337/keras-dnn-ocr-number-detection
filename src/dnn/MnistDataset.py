from typing import Tuple

import numpy as np

from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class MnistDataset:
    def __init__(self):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
         # Create additional images for training
         # Reshape to rank 4
        self.x_train = np.expand_dims(x_train, axis=-1).astype(np.float32)
        self.y_train = np.expand_dims(x_train, axis=-1).astype(np.float32)

        # Data attributes
        self.train_size = x_train.shape[0]
        self.test_size = x_test.shape[0]

        self.data_augmentation(size=5000)

        num_features = x_train.shape[1]*x_train.shape[2]
        self.num_classes = len(np.unique(y_test))

        # Preprocess x data
        self.x_train = x_train.reshape(-1, num_features)
        self.x_test = x_test.reshape(-1, num_features)

        # Normalize (0:1)
        self.x_train = self.x_train / 255.0
        self.x_test = self.x_test / 255.0

        # Preprocess y data
        self.y_train = to_categorical(y_train, num_classes=self.num_classes, dtype=np.float32)
        self.y_test = to_categorical(y_test, num_classes=self.num_classes, dtype=np.float32)

        self.image_shape = (num_features)

    def get_train_set(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.x_train, self.y_train

    def get_test_set(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.x_test, self.y_test

    def get_dataset(self) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        return (self.x_train, self.x_test), (self.y_train, self.y_test)

    def data_augmentation(self, size: int = 5_000) -> None:
        data_generator = ImageDataGenerator(
            rotation_range=5,
            width_shift_range=0.08,
            height_shift_range=0.08,
            fill_mode='nearest'
        )

        data_generator.fit(self.x_train, augment=True)
        rand_image_idx = np.random.randint(self.train_size, size=size)
        x_augmented = self.x_train[rand_image_idx].copy()
        x_augmented = data_generator.flow(
            x=x_augmented,
            y=np.zeros(size),
            batch_size=size,
            shuffle=False
        ).next()[0]

        self.x_train = np.concatenate((self.x_train, x_augmented))
        y_augmented = self.y_train[rand_image_idx].copy()
        self.y_train = np.concatenate((self.y_train, y_augmented))
        self.train_size = self.x_train.shape[0]

    def __str__(self) -> str:
        return f"Shape: {self.image_shape}, TrainSize: {self.train_size}, TestSize: {self.test_size}, NumClasses: {self.num_classes}"


if __name__ == '__main__':
    mnist = MnistDataset()
    print(mnist)
