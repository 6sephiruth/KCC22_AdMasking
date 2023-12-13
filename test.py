import pickle
import tensorflow as tf
from tensorflow import keras


targeted_cw_data = pickle.load(open(f'./dataset/targeted_cw/paper_mnist-0_1','rb'))

print(targeted_cw_data[0].shape)
# dataset = tf.keras.datasets.mnist

# (x_train, y_train), (x_test, y_test) = dataset.load_data()

# x_train = x_train.reshape((60000, 28, 28, 1))
# x_test = x_test.reshape((10000, 28, 28, 1))

# # 이미지를 0~1의 범위로 낮추기 위한 Normalization
# x_train, x_test = x_train / 255.0, x_test / 255.0
