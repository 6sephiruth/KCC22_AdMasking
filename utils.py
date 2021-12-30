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
