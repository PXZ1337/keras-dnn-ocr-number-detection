import numpy as np

from typing import Tuple
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers.experimental.preprocessing import RandomRotation, RandomTranslation, RandomZoom, Rescaling
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

np.random.seed(0)
tf.random.set_seed(0)

class MnistTfDataset:
    def __init__(self, validation_size: float = 0.33) -> None:
        self.num_classes = 10
        self.batch_size = 128

        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        # Split the dataset
        x_train, x_val, y_train, y_val = train_test_split(
            x_train, y_train, test_size=validation_size
        )

         # Preprocess x data
        self.x_train = np.expand_dims(x_train, axis=-1).astype(np.float32)
        self.x_test = np.expand_dims(x_test, axis=-1).astype(np.float32)
        self.x_val = np.expand_dims(x_val, axis=-1).astype(np.float32)

        # Preprocess y data
        self.y_train = to_categorical(y_train, num_classes=self.num_classes)
        self.y_test = to_categorical(y_test, num_classes=self.num_classes)
        self.y_val = to_categorical(y_val, num_classes=self.num_classes)

        # Dataset attributes
        self.train_size = self.x_train.shape[0]
        self.test_size = self.x_test.shape[0]
        self.val_size = self.x_val.shape[0]
        self.width = self.x_train.shape[1]
        self.height = self.x_train.shape[2]
        self.depth = self.x_train.shape[3]
        self.img_shape = (self.width, self.height, self.depth)

        # tf.data Datasets
        self.train_dataset = tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train))
        self.test_dataset = tf.data.Dataset.from_tensor_slices((self.x_test, self.y_test))
        self.val_dataset = tf.data.Dataset.from_tensor_slices((self.x_val, self.y_val))
        self.train_dataset = self._prepare_dataset(self.train_dataset, shuffle=True, augment=True)
        self.test_dataset = self._prepare_dataset(self.test_dataset)
        self.val_dataset = self._prepare_dataset(self.val_dataset)

    def get_train_set(self) -> tf.data.Dataset:
        return self.train_dataset

    def get_test_set(self) -> tf.data.Dataset:
        return self.test_dataset

    def get_val_set(self) -> tf.data.Dataset:
        return self.val_dataset

    @staticmethod
    def _build_preprocessing() -> Sequential:
        model = Sequential()

        # Rescale x/255 => x * 1/255
        model.add(Rescaling(scale=1./255, offset=0.0))

        return model

    @staticmethod
    def _build_data_augmentation() -> Sequential:
        model = Sequential()

        model.add(RandomRotation(factor=0.08))
        # 2px 2/24 ~ 0,8
        model.add(RandomTranslation(height_factor=0.08, width_factor=0.08))
        model.add(RandomZoom(height_factor=0.08, width_factor=0.08))

        return model

    def _prepare_dataset(
        self,
        dataset: tf.data.Dataset,
        shuffle: bool = False,
        augment: bool = False
    ) -> tf.data.Dataset:
        preprocessing_model = self._build_preprocessing()
        dataset = dataset.map(
            map_func = lambda x, y: (preprocessing_model(x, training=False), y),
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )

        if shuffle:
            dataset = dataset.shuffle(buffer_size=1_000)

        dataset = dataset.batch(batch_size=self.batch_size)

        if augment:
            data_augmentation_model = self._build_data_augmentation()
            dataset = dataset.map(
                map_func = lambda x, y: (data_augmentation_model(x, training=False), y),
                num_parallel_calls=tf.data.experimental.AUTOTUNE
            )

        return dataset.prefetch(buffer_size=self.batch_size)

    def __str__(self) -> str:
        return f"Shape: {self.img_shape}, Train: {self.train_size}, Test: {self.test_size}, Validation: {self.val_size}"
