
import numpy as np

import os
import keras.backend as K

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.initializers import Constant
from tensorflow.keras.initializers import TruncatedNormal

from ..config import MODEL_FILE_PATH, MODEL_WEIGHTS_FILE_PATH

class MnistDeepNeuronalNetwork:
    num_features: int = 0
    num_targets: int = 0
    learning_rate: int = 0.0
    train_batch_size: int = 0
    epochs: int = 0
    __model: Sequential

    kernel_initializer = TruncatedNormal(mean=0.0, stddev=0.01)
    bias_initializer = Constant(value=0.0)

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

    def prepare_dataset(self) -> tuple[tuple, tuple]:
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        print(f"Shapes before (x_train, x_test), (y_train, y_test): {x_train}, {x_test}, {y_train}, {y_test}")

        # Convert shape from (60000, 28, 28) to (60000, 10)
        # We want to convert the input to an ndarray with x records and 28x28pixels (x,786)
        x_train = x_train.reshape(-1, self.num_features).astype(np.float32)
        x_test = x_test.reshape(-1, self.num_features).astype(np.float32)

        # We want to convert the output to an ndarray with num_targesxnum_targes entries. 
        # Each entry of the vector represents the class by it's index
        y_train = to_categorical(y_train, num_classes=self.num_targets, dtype=np.float32)
        y_test = to_categorical(y_test, num_classes=self.num_targets, dtype=np.float32)

        print(f"Shapes after (x_train, x_test), (y_train, y_test): {x_train}, {x_test}, {y_train}, {y_test}")
        
        return (x_train, x_test), (y_train, y_test) 

    def build_model(self):
        self.__model = Sequential([
            Dense(
                units=500,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                input_shape=(self.num_features,),
            ),
            Activation("relu"),
            Dense(
                units=250,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
            ),
            Activation("relu"),
            Dense(
                units=self.num_targets,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
            ),
            Activation("Softmax")
        ])
        self.__model.summary()

    def train(self):
        (x_train, x_test), (y_train, y_test) = self.prepare_dataset()
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

        self.__model.save_weights(filepath=MODEL_WEIGHTS_FILE_PATH)
        self.__model.save(filepath=MODEL_FILE_PATH)

    def nn_predict(self, image: np.ndarray = None):
        if image is not None:
            y_pred = self.__model.predict(image.reshape(1, self.num_features))[0]
            y_pred_idx = np.argmax(y_pred, axis=0)
            return y_pred_idx

        return -1

def load_mnist_model():
    if os.path.exists(MODEL_FILE_PATH):
        return MnistDeepNeuronalNetwork(model=load_model(MODEL_FILE_PATH))
    else:
        print(MODEL_FILE_PATH)
        raise FileNotFoundError("Model not found - Please create the model first!")