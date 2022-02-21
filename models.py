from utils import *

class mnist_cnn(Model):
    def __init__(self):
        super(mnist_cnn, self).__init__()
        self.model = self.build_model()

    def build_model(self):
        model = tf.keras.models.Sequential([
            keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
            keras.layers.MaxPool2D((2, 2)),
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.MaxPool2D((2, 2)),
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.Flatten(),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(10)
        ])
        
        return model

    def call(self, inputs):
        return self.model(inputs)

    def predict_classes(self, inputs):
        return self.model.predict_classes(inputs)


class paper_mnist(Model):
    def __init__(self):
        super(paper_mnist, self).__init__()
        self.model = self.build_model()

    def build_model(self):
        model = tf.keras.models.Sequential([
            keras.layers.Conv2D(32, (5, 5), activation='relu', padding='same', input_shape=(28, 28, 1)),
            keras.layers.MaxPool2D((2, 2)),
            keras.layers.Conv2D(64, (5, 5), activation='relu'),
            keras.layers.MaxPool2D((2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(1024, activation='relu'),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(10)
        ])
        
        return model

    def call(self, inputs):
        return self.model(inputs)

    def predict_classes(self, inputs):
        return self.model.predict_classes(inputs)