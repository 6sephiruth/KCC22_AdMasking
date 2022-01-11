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

def neuron_activation_analyze(model, data, num1, num2):

    total_activation = np.empty((len(data),0))

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
    copy_top_5_result = top_5_result

    top_5_result = top_5_result[:44950]   ################## 원래 크기 44992
    top_5_result = np.reshape(top_5_result, (50, 899))

    top_10_result = np.zeros_like(total_activation)
    top_10_result[top_10_position_activation] = 1
    copy_top_10_result = top_10_result

    top_10_result = top_10_result[:44950]
    top_10_result = np.reshape(top_10_result, (50, 899))

    top_20_result = np.zeros_like(total_activation)
    top_20_result[top_20_position_activation] = 1
    copy_top_20_result = top_20_result

    top_20_result = top_20_result[:44950]
    top_20_result = np.reshape(top_20_result, (50, 899))


    plt.imshow(top_5_result)
    plt.axis('off')
    plt.savefig("./img/top5_{}-{}.png".format(num1, num2))
    plt.cla()

    plt.imshow(top_10_result)
    plt.axis('off')
    plt.savefig("./img/top10_{}-{}.png".format(num1, num2))
    plt.cla()

    plt.imshow(top_20_result)
    plt.axis('off')
    plt.savefig("./img/top20_{}-{}.png".format(num1, num2))
    plt.cla()

    arrange_result = np.empty((3, len(copy_top_5_result)))

    arrange_result[0] = copy_top_5_result
    arrange_result[1] = copy_top_10_result
    arrange_result[2] = copy_top_20_result

    pickle.dump(arrange_result, open(f'./dataset/targeted_analysis/{num1}-{num2}','wb'))

def model_weight_analysis(analysis_num, model):

    total_data = np.empty((10,3,44992))

    for i in range(10):
    
        total_data[i] = pickle.load(open(f'./dataset/targeted_analysis/{i}-{analysis_num}','rb'))

    adversarial_activation_position = np.zeros((3,44992))

    for i in range(10):

        for j in range(3):

            if i == analysis_num:    
                break
            
            position = np.where(total_data[i][j] == 1)

            adversarial_activation_position[j][position] = 1 

    top_5_result = adversarial_activation_position[0] - total_data[analysis_num][0]
    position = np.where(top_5_result != 1)
    top_5_result[position] = 0

    top_10_result = adversarial_activation_position[1] - total_data[analysis_num][1]
    position = np.where(top_10_result != 1)
    top_10_result[position] = 0

    top_20_result = adversarial_activation_position[2] - total_data[analysis_num][2]
    position = np.where(top_20_result != 1)
    top_20_result[position] = 0


    # print(np.sum(top_5_result))
    # print(np.sum(top_10_result))
    # print(np.sum(top_20_result))
    # model.model.layers[7].weights[0][0] = 0

    # weight0 = np.zeros((64, 10))


    # weights = np.array(weight0)
    # model.model.layers[7].set_weights(weights)

    # weight0 = np.zeros((3, 3, 1, 32))
    # weight1 = np.zeros((32))
    # weight2 = np.ones((3,3,32,64))
    # weight3 = np.zeros((64))
    # weight4 = np.zeros((3,3,64,64))
    # weight5 = np.zeros((64))
    # weight6 = np.ones((1024,64))
    # weight7 = np.zeros((64))
    # weight8 = np.zeros((64,10))
    # weight9 = np.zeros((10))

    # weights = np.array([weight0, weight1, weight2, weight3, weight4, weight5, weight6, weight7, weight8, weight9])

    # # 5. 웨이트 적용하기
    # model.model.set_weights(weights)

    print(model.model.get_weights()[1])

    # for i in range(10):
    #     print(model.model.get_weights()[i].shape)



    # print(model.model.layers[7].get_weights()[0].shape)


    # print(model.model.layers[7].weights[0][0][0])