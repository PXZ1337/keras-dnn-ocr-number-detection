
import numpy as np

import os
import keras.backend as K

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

from .MnistDataset import MnistDataset

FILE_PATH = os.path.abspath(__file__)
PROJECT_DIR = os.path.dirname(os.path.dirname(FILE_PATH))
MODEL_FILE_PATH = os.path.join(PROJECT_DIR, "ressources", "dnn")

class DeepNeuralNetwork:
    dataset = MnistDataset()

    def __init__(self, num_features: int = 784, num_targets: int = 10, learning_rate: float = 0.0005, train_batch_size: int = 128, train_epochs: int = 15, model: Sequential = None):
        self.num_features = num_features
        self.num_targets = num_targets
        self.learning_rate = learning_rate
        self.train_batch_size = train_batch_size
        self.epochs = train_epochs

        if model is not None:
            self.__model = model
            self.num_features = model.input_shape[1]
            self.num_targets = model.output_shape[1]
            self.learning_rate = K.eval(model.optimizer.lr)

    def build_model(self):
        self.__model = Sequential([
            Dense(
                units=512,
                input_shape=(self.num_features,),
            ),
            Activation("relu"),
            Dense(
                units=256,
            ),
            Activation("relu"),
            Dense(
                units=128,
            ),
            Activation("relu"),
            Dense(
                units=self.num_targets,
            ),
            Activation("Softmax")
        ])
        self.__model.summary()

    def train(self):
        (x_train, x_test), (y_train, y_test) = self.dataset.get_dataset()

        self.build_model()
        self.__model.compile(
            loss="categorical_crossentropy",
            optimizer=Adam(learning_rate=self.learning_rate),
            metrics=["accuracy"]
        )

        self.__model.fit(
            x=x_train,
            y=y_train,
            batch_size=self.train_batch_size,
            epochs=self.epochs,
            verbose=1,
            validation_data=(x_test,y_test)
        )

        scores = self.__model.evaluate(x=x_test, y=y_test)
        print(f"Scores: {scores}")

        self.__model.save(filepath=MODEL_FILE_PATH)

    def nn_predict(self, image: np.ndarray = None):
        if image is not None:
            y_pred = self.__model.predict(image.reshape(1, self.num_features))[0]
            y_pred_idx = np.argmax(y_pred, axis=0)
            return y_pred_idx

        return -1

def load_mnist_model():
    if os.path.exists(MODEL_FILE_PATH):
        return DeepNeuralNetwork(model=load_model(MODEL_FILE_PATH))
    else:
        print(MODEL_FILE_PATH)
        raise FileNotFoundError("Model not found - Please create the model first!")
