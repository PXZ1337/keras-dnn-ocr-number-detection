import os

import tensorflow as tf
import keras.backend as K
import numpy as np

from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Flatten, Dense, Activation
from tensorflow.keras.optimizers import Adam

from .MnistTfDataset import MnistTfDataset

FILE_PATH = os.path.abspath(__file__)
PROJECT_DIR = os.path.dirname(os.path.dirname(FILE_PATH))
MODEL_FILE_PATH = os.path.join(PROJECT_DIR, "ressources", "cnn")
MODEL_LOG_DIR = os.path.join(MODEL_FILE_PATH, "mnist_cnn")

if not os.path.exists(MODEL_LOG_DIR):
    os.mkdir(MODEL_LOG_DIR)

class ConvNeuralNetwork:
    dataset = MnistTfDataset()

    def __init__(self, train_batch_size: int = 128, train_epochs: int = 15, model: Model = None):
        self.num_targets = 10
        self.learning_rate = 0.0005
        self.train_batch_size = train_batch_size
        self.epochs = train_epochs

        if model is not None:
            self.model = model

    def build_model(self):
        input_img = Input(shape=self.dataset.img_shape)

        x = Conv2D(filters=32, kernel_size=3, padding="same")(input_img)
        x = Activation("relu")(x)
        x = Conv2D(filters=32, kernel_size=3, padding="same")(x)
        x = Activation("relu")(x)
        x = MaxPool2D()(x)

        x = Conv2D(filters=60, kernel_size=5, padding="same")(x)
        x = Activation("relu")(x)
        x = Conv2D(filters=60, kernel_size=5, padding="same")(x)
        x = Activation("relu")(x)
        x = MaxPool2D()(x)

        x = Flatten()(x)
        x = Dense(units=self.num_targets)(x)
        y_pred = Activation("softmax")(x)

        self.model = Model(inputs=[input_img], outputs=[y_pred])

        self.model.summary()

    def train(self):
        train_set = self.dataset.get_train_set()
        validation_set = self.dataset.get_val_set()
        test_set = self.dataset.get_test_set()

        self.build_model()
        self.model.compile(
            loss="categorical_crossentropy",
            optimizer=Adam(learning_rate=self.learning_rate),
            metrics=["accuracy"]
        )

        tb_callback = TensorBoard(
            log_dir=MODEL_LOG_DIR, histogram_freq=1, write_graph=True
        )

        self.model.fit(
            train_set,
            batch_size=self.train_batch_size,
            epochs=self.epochs,
            verbose=1,
            validation_data=validation_set,
            callbacks=[tb_callback],
        )

        scores = self.model.evaluate(test_set)
        print(f"Scores: {scores}")

        self.model.save(filepath=MODEL_FILE_PATH)

    def nn_predict(self, image: np.ndarray = None):
        if image is not None:
            image = np.expand_dims(image.reshape(1, image.shape[0], image.shape[1]), axis=-1).astype(np.float32)
            y_pred = self.model.predict(image)[0]
            y_pred_idx = np.argmax(y_pred, axis=0)
            return y_pred_idx

        return -1

def load_mnist_cnn_model():
    if os.path.exists(MODEL_FILE_PATH):
        return ConvNeuralNetwork(model=load_model(MODEL_FILE_PATH))
    else:
        raise FileNotFoundError("Model not found - Please create the model first!")
