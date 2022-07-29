from utils import *
from tensorflow.keras.applications.resnet50 import ResNet50

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
            keras.layers.Dense(1024),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(10),

        ])
        return model

    def call(self, inputs):
        return self.model(inputs)

    def predict_classes(self, inputs):
        return self.model.predict_classes(inputs)

# 출처: https://www.kaggle.com/code/kutaykutlu/resnet50-transfer-learning-cifar-10-beginner/notebook
class resnet50_cifar10(Model):
    
    def __init__(self):
        super(resnet50_cifar10, self).__init__()
        self.model = self.build_model()

    def build_model(self):
        model = tf.keras.models.Sequential([
            keras.layers.UpSampling2D(size=(7,7), input_shape=(32, 32, 3)),
            keras.applications.resnet.ResNet50(include_top=False,
                                               weights='imagenet'),
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Flatten(),
            keras.layers.Dense(1024, activation="relu"),
            keras.layers.Dense(512, activation="relu"),
            keras.layers.Dense(10, activation="softmax"),
        ])
        return model

    def call(self, inputs):
        return self.model(inputs)

    def predict_classes(self, inputs):
        return self.model.predict_classes(inputs)
