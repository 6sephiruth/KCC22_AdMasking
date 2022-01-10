import argparse
import os
import yaml
import pickle
import time

from tqdm import trange

import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, Sequential
from keras.callbacks import ModelCheckpoint

import numpy as np

from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.tf2.attacks.carlini_wagner_l2 import carlini_wagner_l2
from cleverhans.tf2.attacks.projected_gradient_descent import projected_gradient_descent
from cleverhans.tf2.attacks.momentum_iterative_method import momentum_iterative_method

from attack_method import *
from models import *
from dataset_process import *
from restore_ad import *

def mkdir(dir_names):
    for d in dir_names:
        if not os.path.exists(d):
            os.mkdir(d)

def exists(pathname):
    return os.path.exists(pathname)

def mnist_data():
    
    dataset = tf.keras.datasets.mnist
        
    (x_train, y_train), (x_test, y_test) = dataset.load_data()

    x_train = x_train.reshape((60000, 28, 28, 1))
    x_test = x_test.reshape((10000, 28, 28, 1))

    # 이미지를 0~1의 범위로 낮추기 위한 Normalization
    x_train, x_test = x_train / 255.0, x_test / 255.0

    return (x_train, y_train), (x_test, y_test)

def cifar10_data():
    
    dataset = tf.keras.datasets.cifar10

    (x_train, y_train), (x_test, y_test) = dataset.load_data()
    
    x_train = x_train.reshape((50000, 32, 32, 3))
    x_test = x_test.reshape((10000, 32, 32, 3))

    # # MIN, MAX normalization
    # x_train = (x_train - np.min(x_train))/ (np.max(x_train) - np.min(x_train))
    # x_test = (x_test - np.min(x_test))/ (np.max(x_test) - np.min(x_test))

    # 이미지를 0~1의 범위로 낮추기 위한 Normalization
    x_train, x_test = x_train / 255.0, x_test / 255.0

    return (x_train, y_train), (x_test, y_test)

def neuron_activation_analyze(model, data):

    total_activation = np.empty((10000,0))

    for each_layer in range(len(model.model.layers)-1):
        
        intermediate_layer_model = tf.keras.Model(inputs=model.model.input, outputs=model.model.layers[each_layer].output)
        intermediate_output = intermediate_layer_model(data)
        intermediate_output = np.reshape(intermediate_output, (len(intermediate_output), -1))    

        total_activation = np.append(total_activation, intermediate_output, axis=1)

    non_activation_position = np.where(total_activation <= 0)
    activation_position = np.where(total_activation > 0)

    total_activation[non_activation_position] = 0
    total_activation[activation_position] = 1

    total_activation = np.sum(total_activation, axis=0)
    sort_total_activation = np.sort(total_activation)
    
    top_5_activation = sort_total_activation[int(len(sort_total_activation)-np.around((len(sort_total_activation)/100*5))):]
    top_10_activation = sort_total_activation[int(len(sort_total_activation)-np.around((len(sort_total_activation)/100*10))):]
    top_20_activation = sort_total_activation[int(len(sort_total_activation)-np.around((len(sort_total_activation)/100*20))):]
    
    top_5_position_activation = np.where(top_5_activation[0] <= total_activation)
    top_10_position_activation = np.where(top_10_activation[0] <= total_activation)
    top_20_position_activation = np.where(top_20_activation[0] <= total_activation)

    top_5_result = np.zeros_like(total_activation)
    top_5_result[top_5_position_activation] = 1
    top_5_result = top_5_result[:44944]
    top_5_result = np.reshape(top_5_result, (212, 212))

    top_10_result = np.zeros_like(total_activation)
    top_10_result[top_10_position_activation] = 1
    top_10_result = np.reshape(top_10_result, (1,-1))
    top_10_result = top_10_result[:44944]
    top_10_result = np.reshape(top_10_result, (212, 212))


    top_20_result = np.zeros_like(total_activation)
    top_20_result[top_20_position_activation] = 1
    top_20_result = np.reshape(top_20_result, (1,-1))
    top_20_result = top_20_result[:44944]
    top_20_result = np.reshape(top_20_result, (212, 212))


    print(top_5_result.shape)
    plt.imshow(top_5_result)
    plt.savefig("./333.png")