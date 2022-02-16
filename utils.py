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

    # 여기서 data는 pickle.load(open(f'./dataset/targeted_cw/0-1','rb'))

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
    
    top_1_activation = sort_total_activation[int(len(sort_total_activation)-np.around((len(sort_total_activation)/100*1))):]
    top_3_activation = sort_total_activation[int(len(sort_total_activation)-np.around((len(sort_total_activation)/100*3))):]
    top_5_activation = sort_total_activation[int(len(sort_total_activation)-np.around((len(sort_total_activation)/100*5))):]
    top_10_activation = sort_total_activation[int(len(sort_total_activation)-np.around((len(sort_total_activation)/100*10))):]
    top_15_activation = sort_total_activation[int(len(sort_total_activation)-np.around((len(sort_total_activation)/100*15))):]
    top_20_activation = sort_total_activation[int(len(sort_total_activation)-np.around((len(sort_total_activation)/100*20))):]

    top_1_position_activation = np.where(top_1_activation[0] <= total_activation)
    top_3_position_activation = np.where(top_3_activation[0] <= total_activation)
    top_5_position_activation = np.where(top_5_activation[0] <= total_activation)
    top_10_position_activation = np.where(top_10_activation[0] <= total_activation)
    top_15_position_activation = np.where(top_15_activation[0] <= total_activation)
    top_20_position_activation = np.where(top_20_activation[0] <= total_activation)

    top_1_result = np.zeros_like(total_activation)
    top_1_result[top_1_position_activation] = 1
    copy_top_1_result = top_1_result

    top_3_result = np.zeros_like(total_activation)
    top_3_result[top_3_position_activation] = 1
    copy_top_3_result = top_3_result

    top_5_result = np.zeros_like(total_activation)
    top_5_result[top_5_position_activation] = 1
    copy_top_5_result = top_5_result

    top_10_result = np.zeros_like(total_activation)
    top_10_result[top_10_position_activation] = 1
    copy_top_10_result = top_10_result

    top_15_result = np.zeros_like(total_activation)
    top_15_result[top_15_position_activation] = 1
    copy_top_15_result = top_15_result

    top_20_result = np.zeros_like(total_activation)
    top_20_result[top_20_position_activation] = 1
    copy_top_20_result = top_20_result

    arrange_result = np.empty((6, len(copy_top_1_result)))

    arrange_result[0] = copy_top_1_result
    arrange_result[1] = copy_top_3_result
    arrange_result[2] = copy_top_5_result
    arrange_result[3] = copy_top_10_result
    arrange_result[4] = copy_top_15_result
    arrange_result[5] = copy_top_20_result

    # pickle.dump(arrange_result, open(f'./dataset/targeted_analysis/{num1}-{num2}','wb'))
    #### pickle.dump(arrange_result, open(f'./dataset/targeted_half_analysis/{num1}-{num2}','wb'))

def model_weight_analysis(analysis_num, model, dataset):

    # 여기서 dataset는 pickle.load(open(f'./dataset/targeted_cw/0-1','rb'))

    total_data = np.empty((10,6,44992))

    for i in range(10):
    
        total_data[i] = pickle.load(open(f'./dataset/targeted_analysis/{analysis_num}-{i}','rb'))
        # total_data[i] = pickle.load(open(f'./dataset/targeted_half_analysis/{analysis_num}-{i}','rb'))

    # 상위 1% 3% 5% 10% 15% 20% Activation 저장
    adversarial_activation_position = np.zeros((6,44992))

    for i in range(10):

        for j in range(6):

            if i == analysis_num:    
                break
            
            position = np.where(total_data[i][j] == 1)

            adversarial_activation_position[j][position] = 1 

    top_1_result = adversarial_activation_position[0] - total_data[analysis_num][0]
    position = np.where(top_1_result != 1)
    top_1_result[position] = 0

    top_3_result = adversarial_activation_position[1] - total_data[analysis_num][1]
    position = np.where(top_3_result != 1)
    top_3_result[position] = 0

    top_5_result = adversarial_activation_position[2] - total_data[analysis_num][2]
    position = np.where(top_5_result != 1)
    top_5_result[position] = 0

    top_10_result = adversarial_activation_position[3] - total_data[analysis_num][3]
    position = np.where(top_10_result != 1)
    top_10_result[position] = 0

    top_15_result = adversarial_activation_position[4] - total_data[analysis_num][4]
    position = np.where(top_15_result != 1)
    top_15_result[position] = 0

    top_20_result = adversarial_activation_position[5] - total_data[analysis_num][5]
    position = np.where(top_20_result != 1)
    top_20_result[position] = 0

    ####################################################################
    cp_weight0 = model.model.get_weights()[0]
    cp_weight1 = model.model.get_weights()[1]
    cp_weight2 = model.model.get_weights()[2]
    cp_weight3 = model.model.get_weights()[3]
    cp_weight4 = model.model.get_weights()[4]
    cp_weight5 = model.model.get_weights()[5]
    cp_weight6 = model.model.get_weights()[6]
    cp_weight7 = model.model.get_weights()[7]
    cp_weight8 = model.model.get_weights()[8]
    cp_weight9 = model.model.get_weights()[9]

    intermediate_layer_model = tf.keras.Model(inputs=model.model.input, outputs=model.model.layers[0].output)

    layer_1_output_1 = np.array(intermediate_layer_model(dataset))
    layer_1_output_3 = np.array(intermediate_layer_model(dataset))
    layer_1_output_5 = np.array(intermediate_layer_model(dataset))
    layer_1_output_10 = np.array(intermediate_layer_model(dataset))
    layer_1_output_15 = np.array(intermediate_layer_model(dataset))
    layer_1_output_20 = np.array(intermediate_layer_model(dataset))

    adversarial_actvation_position_1 = np.reshape(top_1_result[:25088], (28, 28, 32))
    adversarial_actvation_position_3 = np.reshape(top_3_result[:25088], (28, 28, 32))
    adversarial_actvation_position_5 = np.reshape(top_5_result[:25088], (28, 28, 32))
    adversarial_actvation_position_10 = np.reshape(top_10_result[:25088], (28, 28, 32))
    adversarial_actvation_position_15 = np.reshape(top_15_result[:25088], (28, 28, 32))
    adversarial_actvation_position_20 = np.reshape(top_20_result[:25088], (28, 28, 32))

    adversarial_actvation_position_position_1 = np.where(adversarial_actvation_position_1 == 1)
    adversarial_actvation_position_position_3 = np.where(adversarial_actvation_position_3 == 1)
    adversarial_actvation_position_position_5 = np.where(adversarial_actvation_position_5 == 1)
    adversarial_actvation_position_position_10 = np.where(adversarial_actvation_position_10 == 1)
    adversarial_actvation_position_position_15 = np.where(adversarial_actvation_position_15 == 1)
    adversarial_actvation_position_position_20 = np.where(adversarial_actvation_position_20 == 1)

    for dataset_count in range(len(dataset)):
    
        layer_1_output_1[dataset_count][adversarial_actvation_position_position_1] = 0
        layer_1_output_3[dataset_count][adversarial_actvation_position_position_3] = 0
        layer_1_output_5[dataset_count][adversarial_actvation_position_position_5] = 0
        layer_1_output_10[dataset_count][adversarial_actvation_position_position_10] = 0
        layer_1_output_15[dataset_count][adversarial_actvation_position_position_15] = 0
        layer_1_output_20[dataset_count][adversarial_actvation_position_position_20] = 0

    top_1_result = top_1_result[25088:]
    top_3_result = top_3_result[25088:]
    top_5_result = top_5_result[25088:]
    top_10_result = top_10_result[25088:]
    top_15_result = top_15_result[25088:]
    top_20_result = top_20_result[25088:]

    #-------------------------------------------------------------------------------------------------
    model_layer_2 = tf.keras.models.Sequential([
        tf.keras.layers.MaxPool2D((2, 2), input_shape=(28, 28, 32))
    ])

    layer_2_output_1 = model_layer_2.predict(layer_1_output_1)
    layer_2_output_3 = model_layer_2.predict(layer_1_output_3)
    layer_2_output_5 = model_layer_2.predict(layer_1_output_5)
    layer_2_output_10 = model_layer_2.predict(layer_1_output_10)
    layer_2_output_15 = model_layer_2.predict(layer_1_output_15)
    layer_2_output_20 = model_layer_2.predict(layer_1_output_20)

    adversarial_actvation_position_1 = np.reshape(top_1_result[:6272], (14, 14, 32))
    adversarial_actvation_position_3 = np.reshape(top_3_result[:6272], (14, 14, 32))
    adversarial_actvation_position_5 = np.reshape(top_5_result[:6272], (14, 14, 32))
    adversarial_actvation_position_10 = np.reshape(top_10_result[:6272], (14, 14, 32))
    adversarial_actvation_position_15 = np.reshape(top_15_result[:6272], (14, 14, 32))
    adversarial_actvation_position_20 = np.reshape(top_20_result[:6272], (14, 14, 32))

    adversarial_actvation_position_position_1 = np.where(adversarial_actvation_position_1 == 1)
    adversarial_actvation_position_position_3 = np.where(adversarial_actvation_position_3 == 1)
    adversarial_actvation_position_position_5 = np.where(adversarial_actvation_position_5 == 1)
    adversarial_actvation_position_position_10 = np.where(adversarial_actvation_position_10 == 1)
    adversarial_actvation_position_position_15 = np.where(adversarial_actvation_position_15 == 1)
    adversarial_actvation_position_position_20 = np.where(adversarial_actvation_position_20 == 1)

    for dataset_count in range(len(dataset)):
        
        layer_2_output_1[dataset_count][adversarial_actvation_position_position_1] = 0
        layer_2_output_3[dataset_count][adversarial_actvation_position_position_3] = 0
        layer_2_output_5[dataset_count][adversarial_actvation_position_position_5] = 0
        layer_2_output_10[dataset_count][adversarial_actvation_position_position_10] = 0
        layer_2_output_15[dataset_count][adversarial_actvation_position_position_15] = 0
        layer_2_output_20[dataset_count][adversarial_actvation_position_position_20] = 0

    top_1_result = top_1_result[6272:]
    top_3_result = top_3_result[6272:]
    top_5_result = top_5_result[6272:]
    top_10_result = top_10_result[6272:]
    top_15_result = top_15_result[6272:]
    top_20_result = top_20_result[6272:]

    #--------------------------------------------------------------------------------------------------------
    model_layer_3 = tf.keras.models.Sequential([
        keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(14,14,32))
    ])

    model_hidden_3_weight = [cp_weight2, cp_weight3]
    model_layer_3.set_weights(model_hidden_3_weight)

    layer_3_output_1 = model_layer_3.predict(layer_2_output_1)
    layer_3_output_3 = model_layer_3.predict(layer_2_output_3)
    layer_3_output_5 = model_layer_3.predict(layer_2_output_5)
    layer_3_output_10 = model_layer_3.predict(layer_2_output_10)
    layer_3_output_15 = model_layer_3.predict(layer_2_output_15)
    layer_3_output_20 = model_layer_3.predict(layer_2_output_20)

    adversarial_actvation_position_1 = np.reshape(top_1_result[:9216], (12, 12, 64))
    adversarial_actvation_position_3 = np.reshape(top_3_result[:9216], (12, 12, 64))
    adversarial_actvation_position_5 = np.reshape(top_5_result[:9216], (12, 12, 64))
    adversarial_actvation_position_10 = np.reshape(top_10_result[:9216], (12, 12, 64))
    adversarial_actvation_position_15 = np.reshape(top_15_result[:9216], (12, 12, 64))
    adversarial_actvation_position_20 = np.reshape(top_20_result[:9216], (12, 12, 64))

    adversarial_actvation_position_position_1 = np.where(adversarial_actvation_position_1 == 1)
    adversarial_actvation_position_position_3 = np.where(adversarial_actvation_position_3 == 1)
    adversarial_actvation_position_position_5 = np.where(adversarial_actvation_position_5 == 1)
    adversarial_actvation_position_position_10 = np.where(adversarial_actvation_position_10 == 1)
    adversarial_actvation_position_position_15 = np.where(adversarial_actvation_position_15 == 1)
    adversarial_actvation_position_position_20 = np.where(adversarial_actvation_position_20 == 1)

    for dataset_count in range(len(dataset)):
        
        layer_3_output_1[dataset_count][adversarial_actvation_position_position_1] = 0
        layer_3_output_3[dataset_count][adversarial_actvation_position_position_3] = 0
        layer_3_output_5[dataset_count][adversarial_actvation_position_position_5] = 0
        layer_3_output_10[dataset_count][adversarial_actvation_position_position_10] = 0
        layer_3_output_15[dataset_count][adversarial_actvation_position_position_15] = 0
        layer_3_output_20[dataset_count][adversarial_actvation_position_position_20] = 0

    top_1_result = top_1_result[9216:]
    top_3_result = top_3_result[9216:]
    top_5_result = top_5_result[9216:]
    top_10_result = top_10_result[9216:]
    top_15_result = top_15_result[9216:]
    top_20_result = top_20_result[9216:]

    #--------------------------------------------------------------------------------------------------------
    model_layer_4 = tf.keras.models.Sequential([
        tf.keras.layers.MaxPool2D((2, 2), input_shape=(12, 12, 64))
    ])

    layer_4_output_1 = model_layer_4.predict(layer_3_output_1)
    layer_4_output_3 = model_layer_4.predict(layer_3_output_3)
    layer_4_output_5 = model_layer_4.predict(layer_3_output_5)
    layer_4_output_10 = model_layer_4.predict(layer_3_output_10)
    layer_4_output_15 = model_layer_4.predict(layer_3_output_15)
    layer_4_output_20 = model_layer_4.predict(layer_3_output_20)

    adversarial_actvation_position_1 = np.reshape(top_1_result[:2304], (6, 6, 64))
    adversarial_actvation_position_3 = np.reshape(top_3_result[:2304], (6, 6, 64))
    adversarial_actvation_position_5 = np.reshape(top_5_result[:2304], (6, 6, 64))
    adversarial_actvation_position_10 = np.reshape(top_10_result[:2304], (6, 6, 64))
    adversarial_actvation_position_15 = np.reshape(top_15_result[:2304], (6, 6, 64))
    adversarial_actvation_position_20 = np.reshape(top_20_result[:2304], (6, 6, 64))

    adversarial_actvation_position_position_1 = np.where(adversarial_actvation_position_1 == 1)
    adversarial_actvation_position_position_3 = np.where(adversarial_actvation_position_3 == 1)
    adversarial_actvation_position_position_5 = np.where(adversarial_actvation_position_5 == 1)
    adversarial_actvation_position_position_10 = np.where(adversarial_actvation_position_10 == 1)
    adversarial_actvation_position_position_15 = np.where(adversarial_actvation_position_15 == 1)
    adversarial_actvation_position_position_20 = np.where(adversarial_actvation_position_20 == 1)

    for dataset_count in range(len(dataset)):
        
        layer_4_output_1[dataset_count][adversarial_actvation_position_position_1] = 0
        layer_4_output_3[dataset_count][adversarial_actvation_position_position_3] = 0
        layer_4_output_5[dataset_count][adversarial_actvation_position_position_5] = 0
        layer_4_output_10[dataset_count][adversarial_actvation_position_position_10] = 0
        layer_4_output_15[dataset_count][adversarial_actvation_position_position_15] = 0
        layer_4_output_20[dataset_count][adversarial_actvation_position_position_20] = 0

    top_1_result = top_1_result[2304:]
    top_3_result = top_3_result[2304:]
    top_5_result = top_5_result[2304:]
    top_10_result = top_10_result[2304:]
    top_15_result = top_15_result[2304:]
    top_20_result = top_20_result[2304:]

    #--------------------------------------------------------------------------------------------------------
    model_layer_5 = tf.keras.models.Sequential([
        keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(6,6,64))
    ])

    model_hidden_5_weight = [cp_weight4, cp_weight5]
    model_layer_5.set_weights(model_hidden_5_weight)

    layer_5_output_1 = model_layer_5.predict(layer_4_output_1)
    layer_5_output_3 = model_layer_5.predict(layer_4_output_3)
    layer_5_output_5 = model_layer_5.predict(layer_4_output_5)
    layer_5_output_10 = model_layer_5.predict(layer_4_output_10)
    layer_5_output_15 = model_layer_5.predict(layer_4_output_15)
    layer_5_output_20 = model_layer_5.predict(layer_4_output_20)

    adversarial_actvation_position_1 = np.reshape(top_1_result[:1024], (4, 4, 64))
    adversarial_actvation_position_3 = np.reshape(top_3_result[:1024], (4, 4, 64))
    adversarial_actvation_position_5 = np.reshape(top_5_result[:1024], (4, 4, 64))
    adversarial_actvation_position_10 = np.reshape(top_10_result[:1024], (4, 4, 64))
    adversarial_actvation_position_15 = np.reshape(top_15_result[:1024], (4, 4, 64))
    adversarial_actvation_position_20 = np.reshape(top_20_result[:1024], (4, 4, 64))

    adversarial_actvation_position_position_1 = np.where(adversarial_actvation_position_1 == 1)
    adversarial_actvation_position_position_3 = np.where(adversarial_actvation_position_3 == 1)
    adversarial_actvation_position_position_5 = np.where(adversarial_actvation_position_5 == 1)
    adversarial_actvation_position_position_10 = np.where(adversarial_actvation_position_10 == 1)
    adversarial_actvation_position_position_15 = np.where(adversarial_actvation_position_15 == 1)
    adversarial_actvation_position_position_20 = np.where(adversarial_actvation_position_20 == 1)

    for dataset_count in range(len(dataset)):
        
        layer_5_output_1[dataset_count][adversarial_actvation_position_position_1] = 0
        layer_5_output_3[dataset_count][adversarial_actvation_position_position_3] = 0
        layer_5_output_5[dataset_count][adversarial_actvation_position_position_5] = 0
        layer_5_output_10[dataset_count][adversarial_actvation_position_position_10] = 0
        layer_5_output_15[dataset_count][adversarial_actvation_position_position_15] = 0
        layer_5_output_20[dataset_count][adversarial_actvation_position_position_20] = 0

    top_1_result = top_1_result[1024:]
    top_3_result = top_3_result[1024:]
    top_5_result = top_5_result[1024:]
    top_10_result = top_10_result[1024:]
    top_15_result = top_15_result[1024:]
    top_20_result = top_20_result[1024:]

    #--------------------------------------------------------------------------------------------------------
    model_layer_6 = tf.keras.models.Sequential([
        keras.layers.Flatten(input_shape=(4,4,64))
    ])

    layer_6_output_1 = model_layer_6.predict(layer_5_output_1)
    layer_6_output_3 = model_layer_6.predict(layer_5_output_3)
    layer_6_output_5 = model_layer_6.predict(layer_5_output_5)
    layer_6_output_10 = model_layer_6.predict(layer_5_output_10)
    layer_6_output_15 = model_layer_6.predict(layer_5_output_15)
    layer_6_output_20 = model_layer_6.predict(layer_5_output_20)

    #--------------------------------------------------------------------------------------------------------
    model_layer_7 = tf.keras.models.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(1024,))
    ])

    model_hidden_7_weight = [cp_weight6, cp_weight7]
    model_layer_7.set_weights(model_hidden_7_weight)

    layer_7_output_1 = model_layer_7.predict(layer_6_output_1)
    layer_7_output_3 = model_layer_7.predict(layer_6_output_3)
    layer_7_output_5 = model_layer_7.predict(layer_6_output_5)
    layer_7_output_10 = model_layer_7.predict(layer_6_output_10)
    layer_7_output_15 = model_layer_7.predict(layer_6_output_15)
    layer_7_output_20 = model_layer_7.predict(layer_6_output_20)

    adversarial_actvation_position_1 = np.reshape(top_1_result[:64], 64)
    adversarial_actvation_position_3 = np.reshape(top_3_result[:64], 64)
    adversarial_actvation_position_5 = np.reshape(top_5_result[:64], 64)
    adversarial_actvation_position_10 = np.reshape(top_10_result[:64], 64)
    adversarial_actvation_position_15 = np.reshape(top_15_result[:64], 64)
    adversarial_actvation_position_20 = np.reshape(top_20_result[:64], 64)

    adversarial_actvation_position_position_1 = np.where(adversarial_actvation_position_1 == 1)
    adversarial_actvation_position_position_3 = np.where(adversarial_actvation_position_3 == 1)
    adversarial_actvation_position_position_5 = np.where(adversarial_actvation_position_5 == 1)
    adversarial_actvation_position_position_10 = np.where(adversarial_actvation_position_10 == 1)
    adversarial_actvation_position_position_15 = np.where(adversarial_actvation_position_15 == 1)
    adversarial_actvation_position_position_20 = np.where(adversarial_actvation_position_20 == 1)

    for dataset_count in range(len(dataset)):
        
        layer_7_output_1[dataset_count][adversarial_actvation_position_position_1] = 0
        layer_7_output_3[dataset_count][adversarial_actvation_position_position_3] = 0
        layer_7_output_5[dataset_count][adversarial_actvation_position_position_5] = 0
        layer_7_output_10[dataset_count][adversarial_actvation_position_position_10] = 0
        layer_7_output_15[dataset_count][adversarial_actvation_position_position_15] = 0
        layer_7_output_20[dataset_count][adversarial_actvation_position_position_20] = 0

    top_1_result = top_1_result[64:]
    top_3_result = top_3_result[64:]
    top_5_result = top_5_result[64:]
    top_10_result = top_10_result[64:]
    top_15_result = top_15_result[64:]
    top_20_result = top_20_result[64:]

    #--------------------------------------------------------------------------------------------------------
    model_layer_8 = tf.keras.models.Sequential([
        keras.layers.Dense(10, activation='relu', input_shape=(64,))
    ])

    model_hidden_8_weight = [cp_weight8, cp_weight9]
    model_layer_8.set_weights(model_hidden_8_weight)

    layer_8_output_1 = model_layer_8.predict(layer_7_output_1)
    layer_8_output_3 = model_layer_8.predict(layer_7_output_3)
    layer_8_output_5 = model_layer_8.predict(layer_7_output_5)
    layer_8_output_10 = model_layer_8.predict(layer_7_output_10)
    layer_8_output_15 = model_layer_8.predict(layer_7_output_15)
    layer_8_output_20 = model_layer_8.predict(layer_7_output_20)

    layer_8_output_1 = tf.nn.softmax(layer_8_output_1)
    layer_8_output_3 = tf.nn.softmax(layer_8_output_3)
    layer_8_output_5 = tf.nn.softmax(layer_8_output_5)
    layer_8_output_10 = tf.nn.softmax(layer_8_output_10)
    layer_8_output_15 = tf.nn.softmax(layer_8_output_15)
    layer_8_output_20 = tf.nn.softmax(layer_8_output_20)

    layer_8_output_1 = np.argmax(layer_8_output_1, axis=1)
    layer_8_output_3 = np.argmax(layer_8_output_3, axis=1)
    layer_8_output_5 = np.argmax(layer_8_output_5, axis=1)
    layer_8_output_10 = np.argmax(layer_8_output_10, axis=1)
    layer_8_output_15 = np.argmax(layer_8_output_15, axis=1)
    layer_8_output_20 = np.argmax(layer_8_output_20, axis=1)

    # k0 = np.where(layer_8_output_1 == analysis_num)[0]
    # k1 = np.where(layer_8_output_3 == analysis_num)[0]
    # k2 = np.where(layer_8_output_5 == analysis_num)[0]
    # k3 = np.where(layer_8_output_10 == analysis_num)[0]
    # k4 = np.where(layer_8_output_15 == analysis_num)[0]
    # k5 = np.where(layer_8_output_20 == analysis_num)[0]

    # print("1%  {: .2f}".format(len(k0) / len(dataset)*100))
    # print("3%  {: .2f}".format(len(k1) / len(dataset)*100))
    # print("5%  {: .2f}".format(len(k2) / len(dataset)*100))
    # print("10%  {: .2f}".format(len(k3) / len(dataset)*100))
    # print("15%  {: .2f}".format(len(k4) / len(dataset)*100))
    # print("20%  {: .2f}".format(len(k5) / len(dataset)*100))
    print("---------------------------------------------------")
    for i in range(100):
        print(layer_8_output_20[i])
    print("---------------------------------------------------")


#========================================================================================================
def model_compress(analysis_num, model, dataset):

    adversarial_activation_position = np.zeros((6,44992))
    normal_activation_position = np.zeros((6,44992))

    for i in range(10):
    
        for j in range(10):

            for k in range(6):

                data = pickle.load(open(f'./dataset/targeted_analysis/{i}-{j}','rb'))
                position = np.where(data[k] == 1)

                if i == j:
                    normal_activation_position[k][position] = 1 
                else:   
                    adversarial_activation_position[k][position] = 1 

    top_1_result = adversarial_activation_position[0] - normal_activation_position[0]
    position = np.where(top_1_result != 1)
    top_1_result[position] = 0

    top_3_result = adversarial_activation_position[1] - normal_activation_position[1]
    position = np.where(top_3_result != 1)
    top_3_result[position] = 0

    top_5_result = adversarial_activation_position[2] - normal_activation_position[2]
    position = np.where(top_5_result != 1)
    top_5_result[position] = 0

    top_10_result = adversarial_activation_position[3] - normal_activation_position[3]
    position = np.where(top_10_result != 1)
    top_10_result[position] = 0

    top_15_result = adversarial_activation_position[4] - normal_activation_position[4]
    position = np.where(top_15_result != 1)
    top_15_result[position] = 0

    top_20_result = adversarial_activation_position[5] - normal_activation_position[5]
    position = np.where(top_20_result != 1)
    top_20_result[position] = 0


    ####################################################################
    cp_weight0 = model.model.get_weights()[0]
    cp_weight1 = model.model.get_weights()[1]
    cp_weight2 = model.model.get_weights()[2]
    cp_weight3 = model.model.get_weights()[3]
    cp_weight4 = model.model.get_weights()[4]
    cp_weight5 = model.model.get_weights()[5]
    cp_weight6 = model.model.get_weights()[6]
    cp_weight7 = model.model.get_weights()[7]
    cp_weight8 = model.model.get_weights()[8]
    cp_weight9 = model.model.get_weights()[9]

    intermediate_layer_model = tf.keras.Model(inputs=model.model.input, outputs=model.model.layers[0].output)

    layer_1_output_1 = np.array(intermediate_layer_model(dataset))
    layer_1_output_3 = np.array(intermediate_layer_model(dataset))
    layer_1_output_5 = np.array(intermediate_layer_model(dataset))
    layer_1_output_10 = np.array(intermediate_layer_model(dataset))
    layer_1_output_15 = np.array(intermediate_layer_model(dataset))
    layer_1_output_20 = np.array(intermediate_layer_model(dataset))

    adversarial_actvation_position_1 = np.reshape(top_1_result[:25088], (28, 28, 32))
    adversarial_actvation_position_3 = np.reshape(top_3_result[:25088], (28, 28, 32))
    adversarial_actvation_position_5 = np.reshape(top_5_result[:25088], (28, 28, 32))
    adversarial_actvation_position_10 = np.reshape(top_10_result[:25088], (28, 28, 32))
    adversarial_actvation_position_15 = np.reshape(top_15_result[:25088], (28, 28, 32))
    adversarial_actvation_position_20 = np.reshape(top_20_result[:25088], (28, 28, 32))

    adversarial_actvation_position_position_1 = np.where(adversarial_actvation_position_1 == 1)
    adversarial_actvation_position_position_3 = np.where(adversarial_actvation_position_3 == 1)
    adversarial_actvation_position_position_5 = np.where(adversarial_actvation_position_5 == 1)
    adversarial_actvation_position_position_10 = np.where(adversarial_actvation_position_10 == 1)
    adversarial_actvation_position_position_15 = np.where(adversarial_actvation_position_15 == 1)
    adversarial_actvation_position_position_20 = np.where(adversarial_actvation_position_20 == 1)

    for dataset_count in range(len(dataset)):
    
        layer_1_output_1[dataset_count][adversarial_actvation_position_position_1] = 0
        layer_1_output_3[dataset_count][adversarial_actvation_position_position_3] = 0
        layer_1_output_5[dataset_count][adversarial_actvation_position_position_5] = 0
        layer_1_output_10[dataset_count][adversarial_actvation_position_position_10] = 0
        layer_1_output_15[dataset_count][adversarial_actvation_position_position_15] = 0
        layer_1_output_20[dataset_count][adversarial_actvation_position_position_20] = 0

    top_1_result = top_1_result[25088:]
    top_3_result = top_3_result[25088:]
    top_5_result = top_5_result[25088:]
    top_10_result = top_10_result[25088:]
    top_15_result = top_15_result[25088:]
    top_20_result = top_20_result[25088:]

    #-------------------------------------------------------------------------------------------------
    model_layer_2 = tf.keras.models.Sequential([
        tf.keras.layers.MaxPool2D((2, 2), input_shape=(28, 28, 32))
    ])

    layer_2_output_1 = model_layer_2.predict(layer_1_output_1)
    layer_2_output_3 = model_layer_2.predict(layer_1_output_3)
    layer_2_output_5 = model_layer_2.predict(layer_1_output_5)
    layer_2_output_10 = model_layer_2.predict(layer_1_output_10)
    layer_2_output_15 = model_layer_2.predict(layer_1_output_15)
    layer_2_output_20 = model_layer_2.predict(layer_1_output_20)

    adversarial_actvation_position_1 = np.reshape(top_1_result[:6272], (14, 14, 32))
    adversarial_actvation_position_3 = np.reshape(top_3_result[:6272], (14, 14, 32))
    adversarial_actvation_position_5 = np.reshape(top_5_result[:6272], (14, 14, 32))
    adversarial_actvation_position_10 = np.reshape(top_10_result[:6272], (14, 14, 32))
    adversarial_actvation_position_15 = np.reshape(top_15_result[:6272], (14, 14, 32))
    adversarial_actvation_position_20 = np.reshape(top_20_result[:6272], (14, 14, 32))

    adversarial_actvation_position_position_1 = np.where(adversarial_actvation_position_1 == 1)
    adversarial_actvation_position_position_3 = np.where(adversarial_actvation_position_3 == 1)
    adversarial_actvation_position_position_5 = np.where(adversarial_actvation_position_5 == 1)
    adversarial_actvation_position_position_10 = np.where(adversarial_actvation_position_10 == 1)
    adversarial_actvation_position_position_15 = np.where(adversarial_actvation_position_15 == 1)
    adversarial_actvation_position_position_20 = np.where(adversarial_actvation_position_20 == 1)

    for dataset_count in range(len(dataset)):
        
        layer_2_output_1[dataset_count][adversarial_actvation_position_position_1] = 0
        layer_2_output_3[dataset_count][adversarial_actvation_position_position_3] = 0
        layer_2_output_5[dataset_count][adversarial_actvation_position_position_5] = 0
        layer_2_output_10[dataset_count][adversarial_actvation_position_position_10] = 0
        layer_2_output_15[dataset_count][adversarial_actvation_position_position_15] = 0
        layer_2_output_20[dataset_count][adversarial_actvation_position_position_20] = 0

    top_1_result = top_1_result[6272:]
    top_3_result = top_3_result[6272:]
    top_5_result = top_5_result[6272:]
    top_10_result = top_10_result[6272:]
    top_15_result = top_15_result[6272:]
    top_20_result = top_20_result[6272:]

    #--------------------------------------------------------------------------------------------------------
    model_layer_3 = tf.keras.models.Sequential([
        keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(14,14,32))
    ])

    model_hidden_3_weight = [cp_weight2, cp_weight3]
    model_layer_3.set_weights(model_hidden_3_weight)

    layer_3_output_1 = model_layer_3.predict(layer_2_output_1)
    layer_3_output_3 = model_layer_3.predict(layer_2_output_3)
    layer_3_output_5 = model_layer_3.predict(layer_2_output_5)
    layer_3_output_10 = model_layer_3.predict(layer_2_output_10)
    layer_3_output_15 = model_layer_3.predict(layer_2_output_15)
    layer_3_output_20 = model_layer_3.predict(layer_2_output_20)

    adversarial_actvation_position_1 = np.reshape(top_1_result[:9216], (12, 12, 64))
    adversarial_actvation_position_3 = np.reshape(top_3_result[:9216], (12, 12, 64))
    adversarial_actvation_position_5 = np.reshape(top_5_result[:9216], (12, 12, 64))
    adversarial_actvation_position_10 = np.reshape(top_10_result[:9216], (12, 12, 64))
    adversarial_actvation_position_15 = np.reshape(top_15_result[:9216], (12, 12, 64))
    adversarial_actvation_position_20 = np.reshape(top_20_result[:9216], (12, 12, 64))

    adversarial_actvation_position_position_1 = np.where(adversarial_actvation_position_1 == 1)
    adversarial_actvation_position_position_3 = np.where(adversarial_actvation_position_3 == 1)
    adversarial_actvation_position_position_5 = np.where(adversarial_actvation_position_5 == 1)
    adversarial_actvation_position_position_10 = np.where(adversarial_actvation_position_10 == 1)
    adversarial_actvation_position_position_15 = np.where(adversarial_actvation_position_15 == 1)
    adversarial_actvation_position_position_20 = np.where(adversarial_actvation_position_20 == 1)

    for dataset_count in range(len(dataset)):
        
        layer_3_output_1[dataset_count][adversarial_actvation_position_position_1] = 0
        layer_3_output_3[dataset_count][adversarial_actvation_position_position_3] = 0
        layer_3_output_5[dataset_count][adversarial_actvation_position_position_5] = 0
        layer_3_output_10[dataset_count][adversarial_actvation_position_position_10] = 0
        layer_3_output_15[dataset_count][adversarial_actvation_position_position_15] = 0
        layer_3_output_20[dataset_count][adversarial_actvation_position_position_20] = 0

    top_1_result = top_1_result[9216:]
    top_3_result = top_3_result[9216:]
    top_5_result = top_5_result[9216:]
    top_10_result = top_10_result[9216:]
    top_15_result = top_15_result[9216:]
    top_20_result = top_20_result[9216:]

    #--------------------------------------------------------------------------------------------------------
    model_layer_4 = tf.keras.models.Sequential([
        tf.keras.layers.MaxPool2D((2, 2), input_shape=(12, 12, 64))
    ])

    layer_4_output_1 = model_layer_4.predict(layer_3_output_1)
    layer_4_output_3 = model_layer_4.predict(layer_3_output_3)
    layer_4_output_5 = model_layer_4.predict(layer_3_output_5)
    layer_4_output_10 = model_layer_4.predict(layer_3_output_10)
    layer_4_output_15 = model_layer_4.predict(layer_3_output_15)
    layer_4_output_20 = model_layer_4.predict(layer_3_output_20)

    adversarial_actvation_position_1 = np.reshape(top_1_result[:2304], (6, 6, 64))
    adversarial_actvation_position_3 = np.reshape(top_3_result[:2304], (6, 6, 64))
    adversarial_actvation_position_5 = np.reshape(top_5_result[:2304], (6, 6, 64))
    adversarial_actvation_position_10 = np.reshape(top_10_result[:2304], (6, 6, 64))
    adversarial_actvation_position_15 = np.reshape(top_15_result[:2304], (6, 6, 64))
    adversarial_actvation_position_20 = np.reshape(top_20_result[:2304], (6, 6, 64))

    adversarial_actvation_position_position_1 = np.where(adversarial_actvation_position_1 == 1)
    adversarial_actvation_position_position_3 = np.where(adversarial_actvation_position_3 == 1)
    adversarial_actvation_position_position_5 = np.where(adversarial_actvation_position_5 == 1)
    adversarial_actvation_position_position_10 = np.where(adversarial_actvation_position_10 == 1)
    adversarial_actvation_position_position_15 = np.where(adversarial_actvation_position_15 == 1)
    adversarial_actvation_position_position_20 = np.where(adversarial_actvation_position_20 == 1)

    for dataset_count in range(len(dataset)):
        
        layer_4_output_1[dataset_count][adversarial_actvation_position_position_1] = 0
        layer_4_output_3[dataset_count][adversarial_actvation_position_position_3] = 0
        layer_4_output_5[dataset_count][adversarial_actvation_position_position_5] = 0
        layer_4_output_10[dataset_count][adversarial_actvation_position_position_10] = 0
        layer_4_output_15[dataset_count][adversarial_actvation_position_position_15] = 0
        layer_4_output_20[dataset_count][adversarial_actvation_position_position_20] = 0

    top_1_result = top_1_result[2304:]
    top_3_result = top_3_result[2304:]
    top_5_result = top_5_result[2304:]
    top_10_result = top_10_result[2304:]
    top_15_result = top_15_result[2304:]
    top_20_result = top_20_result[2304:]

    #--------------------------------------------------------------------------------------------------------
    model_layer_5 = tf.keras.models.Sequential([
        keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(6,6,64))
    ])

    model_hidden_5_weight = [cp_weight4, cp_weight5]
    model_layer_5.set_weights(model_hidden_5_weight)

    layer_5_output_1 = model_layer_5.predict(layer_4_output_1)
    layer_5_output_3 = model_layer_5.predict(layer_4_output_3)
    layer_5_output_5 = model_layer_5.predict(layer_4_output_5)
    layer_5_output_10 = model_layer_5.predict(layer_4_output_10)
    layer_5_output_15 = model_layer_5.predict(layer_4_output_15)
    layer_5_output_20 = model_layer_5.predict(layer_4_output_20)

    adversarial_actvation_position_1 = np.reshape(top_1_result[:1024], (4, 4, 64))
    adversarial_actvation_position_3 = np.reshape(top_3_result[:1024], (4, 4, 64))
    adversarial_actvation_position_5 = np.reshape(top_5_result[:1024], (4, 4, 64))
    adversarial_actvation_position_10 = np.reshape(top_10_result[:1024], (4, 4, 64))
    adversarial_actvation_position_15 = np.reshape(top_15_result[:1024], (4, 4, 64))
    adversarial_actvation_position_20 = np.reshape(top_20_result[:1024], (4, 4, 64))

    adversarial_actvation_position_position_1 = np.where(adversarial_actvation_position_1 == 1)
    adversarial_actvation_position_position_3 = np.where(adversarial_actvation_position_3 == 1)
    adversarial_actvation_position_position_5 = np.where(adversarial_actvation_position_5 == 1)
    adversarial_actvation_position_position_10 = np.where(adversarial_actvation_position_10 == 1)
    adversarial_actvation_position_position_15 = np.where(adversarial_actvation_position_15 == 1)
    adversarial_actvation_position_position_20 = np.where(adversarial_actvation_position_20 == 1)

    for dataset_count in range(len(dataset)):
        
        layer_5_output_1[dataset_count][adversarial_actvation_position_position_1] = 0
        layer_5_output_3[dataset_count][adversarial_actvation_position_position_3] = 0
        layer_5_output_5[dataset_count][adversarial_actvation_position_position_5] = 0
        layer_5_output_10[dataset_count][adversarial_actvation_position_position_10] = 0
        layer_5_output_15[dataset_count][adversarial_actvation_position_position_15] = 0
        layer_5_output_20[dataset_count][adversarial_actvation_position_position_20] = 0

    top_1_result = top_1_result[1024:]
    top_3_result = top_3_result[1024:]
    top_5_result = top_5_result[1024:]
    top_10_result = top_10_result[1024:]
    top_15_result = top_15_result[1024:]
    top_20_result = top_20_result[1024:]

    #--------------------------------------------------------------------------------------------------------
    model_layer_6 = tf.keras.models.Sequential([
        keras.layers.Flatten(input_shape=(4,4,64))
    ])

    layer_6_output_1 = model_layer_6.predict(layer_5_output_1)
    layer_6_output_3 = model_layer_6.predict(layer_5_output_3)
    layer_6_output_5 = model_layer_6.predict(layer_5_output_5)
    layer_6_output_10 = model_layer_6.predict(layer_5_output_10)
    layer_6_output_15 = model_layer_6.predict(layer_5_output_15)
    layer_6_output_20 = model_layer_6.predict(layer_5_output_20)

    #--------------------------------------------------------------------------------------------------------
    model_layer_7 = tf.keras.models.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(1024,))
    ])

    model_hidden_7_weight = [cp_weight6, cp_weight7]
    model_layer_7.set_weights(model_hidden_7_weight)

    layer_7_output_1 = model_layer_7.predict(layer_6_output_1)
    layer_7_output_3 = model_layer_7.predict(layer_6_output_3)
    layer_7_output_5 = model_layer_7.predict(layer_6_output_5)
    layer_7_output_10 = model_layer_7.predict(layer_6_output_10)
    layer_7_output_15 = model_layer_7.predict(layer_6_output_15)
    layer_7_output_20 = model_layer_7.predict(layer_6_output_20)

    adversarial_actvation_position_1 = np.reshape(top_1_result[:64], 64)
    adversarial_actvation_position_3 = np.reshape(top_3_result[:64], 64)
    adversarial_actvation_position_5 = np.reshape(top_5_result[:64], 64)
    adversarial_actvation_position_10 = np.reshape(top_10_result[:64], 64)
    adversarial_actvation_position_15 = np.reshape(top_15_result[:64], 64)
    adversarial_actvation_position_20 = np.reshape(top_20_result[:64], 64)

    adversarial_actvation_position_position_1 = np.where(adversarial_actvation_position_1 == 1)
    adversarial_actvation_position_position_3 = np.where(adversarial_actvation_position_3 == 1)
    adversarial_actvation_position_position_5 = np.where(adversarial_actvation_position_5 == 1)
    adversarial_actvation_position_position_10 = np.where(adversarial_actvation_position_10 == 1)
    adversarial_actvation_position_position_15 = np.where(adversarial_actvation_position_15 == 1)
    adversarial_actvation_position_position_20 = np.where(adversarial_actvation_position_20 == 1)

    for dataset_count in range(len(dataset)):
        
        layer_7_output_1[dataset_count][adversarial_actvation_position_position_1] = 0
        layer_7_output_3[dataset_count][adversarial_actvation_position_position_3] = 0
        layer_7_output_5[dataset_count][adversarial_actvation_position_position_5] = 0
        layer_7_output_10[dataset_count][adversarial_actvation_position_position_10] = 0
        layer_7_output_15[dataset_count][adversarial_actvation_position_position_15] = 0
        layer_7_output_20[dataset_count][adversarial_actvation_position_position_20] = 0

    top_1_result = top_1_result[64:]
    top_3_result = top_3_result[64:]
    top_5_result = top_5_result[64:]
    top_10_result = top_10_result[64:]
    top_15_result = top_15_result[64:]
    top_20_result = top_20_result[64:]

    #--------------------------------------------------------------------------------------------------------
    model_layer_8 = tf.keras.models.Sequential([
        keras.layers.Dense(10, activation='relu', input_shape=(64,))
    ])

    model_hidden_8_weight = [cp_weight8, cp_weight9]
    model_layer_8.set_weights(model_hidden_8_weight)

    layer_8_output_1 = model_layer_8.predict(layer_7_output_1)
    layer_8_output_3 = model_layer_8.predict(layer_7_output_3)
    layer_8_output_5 = model_layer_8.predict(layer_7_output_5)
    layer_8_output_10 = model_layer_8.predict(layer_7_output_10)
    layer_8_output_15 = model_layer_8.predict(layer_7_output_15)
    layer_8_output_20 = model_layer_8.predict(layer_7_output_20)

    layer_8_output_1 = tf.nn.softmax(layer_8_output_1)
    layer_8_output_3 = tf.nn.softmax(layer_8_output_3)
    layer_8_output_5 = tf.nn.softmax(layer_8_output_5)
    layer_8_output_10 = tf.nn.softmax(layer_8_output_10)
    layer_8_output_15 = tf.nn.softmax(layer_8_output_15)
    layer_8_output_20 = tf.nn.softmax(layer_8_output_20)

    layer_8_output_1 = np.argmax(layer_8_output_1, axis=1)
    layer_8_output_3 = np.argmax(layer_8_output_3, axis=1)
    layer_8_output_5 = np.argmax(layer_8_output_5, axis=1)
    layer_8_output_10 = np.argmax(layer_8_output_10, axis=1)
    layer_8_output_15 = np.argmax(layer_8_output_15, axis=1)
    layer_8_output_20 = np.argmax(layer_8_output_20, axis=1)

    k0 = np.where(layer_8_output_1 == analysis_num)[0]
    k1 = np.where(layer_8_output_3 == analysis_num)[0]
    k2 = np.where(layer_8_output_5 == analysis_num)[0]
    k3 = np.where(layer_8_output_10 == analysis_num)[0]
    k4 = np.where(layer_8_output_15 == analysis_num)[0]
    k5 = np.where(layer_8_output_20 == analysis_num)[0]

    print("1%  {: .2f}".format(len(k0) / len(dataset)*100))
    print("3%  {: .2f}".format(len(k1) / len(dataset)*100))
    print("5%  {: .2f}".format(len(k2) / len(dataset)*100))
    print("10%  {: .2f}".format(len(k3) / len(dataset)*100))
    print("15%  {: .2f}".format(len(k4) / len(dataset)*100))
    print("20%  {: .2f}".format(len(k5) / len(dataset)*100))
