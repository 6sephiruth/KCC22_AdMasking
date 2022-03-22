import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
import pickle
from tensorflow import keras
import time

def normal_prunning(model, origin_data, origin_label):

    origin_pred = model.predict(origin_data)
    origin_pred = tf.nn.softmax(origin_pred)
    origin_pred = np.argmax(origin_pred, axis=1)

    origin_position = np.where(origin_pred == origin_label)[0]

    common_origin_data = origin_data[origin_position]

    origin_total_activation = np.empty((len(common_origin_data),0))

    # for each_layer in range(len(model.model.layers)-1):
    for each_layer in range(len(model.layers)-1):
        
        # intermediate_layer_model = tf.keras.Model(inputs=model.model.input, outputs=model.model.layers[each_layer].output)
        intermediate_layer_model = tf.keras.Model(inputs=model.input, outputs=model.layers[each_layer].output)

        intermediate_origin_output = intermediate_layer_model(common_origin_data)

        intermediate_origin_output = np.reshape(intermediate_origin_output, (len(intermediate_origin_output), -1))
        
        origin_total_activation = np.append(origin_total_activation, intermediate_origin_output, axis=1)

    origin_total_activation = np.where(origin_total_activation <= 0, 0, 1)

    total_activation = np.sum(origin_total_activation, axis=0)
    print(total_activation.shape)


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

    pickle.dump(arrange_result, open(f'./model/paper_mnist/dataset/pre_pruning/{origin_label}-{origin_label}','wb'))



def top_adversarial_actvation_select(model, origin_data, adver_data, origin_label, adver_label):
    # 방법 2

    origin_pred = model.predict(origin_data)
    origin_pred = tf.nn.softmax(origin_pred)
    origin_pred = np.argmax(origin_pred, axis=1)

    adver_pred = model.predict(adver_data)
    adver_pred = tf.nn.softmax(adver_pred)
    adver_pred = np.argmax(adver_pred, axis=1)

    origin_position = np.where(origin_pred == origin_label)[0]
    adv_position = np.where(adver_pred == adver_label)[0]
    
    common_data = list(set(origin_position) & set(adv_position))

    common_origin_data = origin_data[common_data]
    common_adver_data = adver_data[common_data]

    origin_total_activation = np.empty((len(common_data),0))
    adver_total_activation = np.empty((len(common_data),0))

    for each_layer in range(len(model.layers)-1):
        
        intermediate_layer_model = tf.keras.Model(inputs=model.input, outputs=model.layers[each_layer].output)

        intermediate_origin_output = intermediate_layer_model(common_origin_data)
        intermediate_adver_output = intermediate_layer_model(common_adver_data)

        intermediate_origin_output = np.reshape(intermediate_origin_output, (len(intermediate_origin_output), -1))
        intermediate_adver_output = np.reshape(intermediate_adver_output, (len(intermediate_adver_output), -1))
        
        origin_total_activation = np.append(origin_total_activation, intermediate_origin_output, axis=1)
        adver_total_activation = np.append(adver_total_activation, intermediate_adver_output, axis=1)

    origin_total_activation = np.where(origin_total_activation <= 0, 0, 1)
    adver_total_activation = np.where(adver_total_activation <= 0, 0, 1)

    find_adver_activation = adver_total_activation - origin_total_activation

    position = np.where(find_adver_activation != 1)
    find_adver_activation[position] = 0


    total_activation = np.sum(find_adver_activation, axis=0)


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

    pickle.dump(arrange_result, open(f'./model/cifar-10/dataset/pre_pruning/{origin_label}-{adver_label}','wb'))

def mnist_model_compress(analysis_num ,model, dataset):
# def mnist_model_compress(model, dataset, y_full_data):

    normal_activation_position = np.zeros((6,42048))
    adversarial_activation_position = np.zeros((6,42048))

    for i in range(10):
    
        for j in range(10):

            for k in range(6):
                data = pickle.load(open(f'./model/paper_mnist/dataset/pre_pruning/{i}-{j}','rb'))
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

    model_hidden_weight = [cp_weight0, cp_weight1]
    intermediate_layer_model.set_weights(model_hidden_weight)
    
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
        keras.layers.Conv2D(64, (5, 5), activation='relu', input_shape=(14,14,32))
    ])

    model_hidden_3_weight = [cp_weight2, cp_weight3]
    model_layer_3.set_weights(model_hidden_3_weight)

    layer_3_output_1 = model_layer_3.predict(layer_2_output_1)
    layer_3_output_3 = model_layer_3.predict(layer_2_output_3)
    layer_3_output_5 = model_layer_3.predict(layer_2_output_5)
    layer_3_output_10 = model_layer_3.predict(layer_2_output_10)
    layer_3_output_15 = model_layer_3.predict(layer_2_output_15)
    layer_3_output_20 = model_layer_3.predict(layer_2_output_20)

    adversarial_actvation_position_1 = np.reshape(top_1_result[:6400], (10, 10, 64))
    adversarial_actvation_position_3 = np.reshape(top_3_result[:6400], (10, 10, 64))
    adversarial_actvation_position_5 = np.reshape(top_5_result[:6400], (10, 10, 64))
    adversarial_actvation_position_10 = np.reshape(top_10_result[:6400], (10, 10, 64))
    adversarial_actvation_position_15 = np.reshape(top_15_result[:6400], (10, 10, 64))
    adversarial_actvation_position_20 = np.reshape(top_20_result[:6400], (10, 10, 64))

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

    top_1_result = top_1_result[6400:]
    top_3_result = top_3_result[6400:]
    top_5_result = top_5_result[6400:]
    top_10_result = top_10_result[6400:]
    top_15_result = top_15_result[6400:]
    top_20_result = top_20_result[6400:]

    #--------------------------------------------------------------------------------------------------------
    model_layer_4 = tf.keras.models.Sequential([
        tf.keras.layers.MaxPool2D((2, 2), input_shape=(10, 10, 64))
    ])

    layer_4_output_1 = model_layer_4.predict(layer_3_output_1)
    layer_4_output_3 = model_layer_4.predict(layer_3_output_3)
    layer_4_output_5 = model_layer_4.predict(layer_3_output_5)
    layer_4_output_10 = model_layer_4.predict(layer_3_output_10)
    layer_4_output_15 = model_layer_4.predict(layer_3_output_15)
    layer_4_output_20 = model_layer_4.predict(layer_3_output_20)

    adversarial_actvation_position_1 = np.reshape(top_1_result[:1600], (5, 5, 64))
    adversarial_actvation_position_3 = np.reshape(top_3_result[:1600], (5, 5, 64))
    adversarial_actvation_position_5 = np.reshape(top_5_result[:1600], (5, 5, 64))
    adversarial_actvation_position_10 = np.reshape(top_10_result[:1600], (5, 5, 64))
    adversarial_actvation_position_15 = np.reshape(top_15_result[:1600], (5, 5, 64))
    adversarial_actvation_position_20 = np.reshape(top_20_result[:1600], (5, 5, 64))

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

    top_1_result = top_1_result[1600:]
    top_3_result = top_3_result[1600:]
    top_5_result = top_5_result[1600:]
    top_10_result = top_10_result[1600:]
    top_15_result = top_15_result[1600:]
    top_20_result = top_20_result[1600:]

    #--------------------------------------------------------------------------------------------------------
    model_layer_6 = tf.keras.models.Sequential([
        keras.layers.Flatten(input_shape=(5,5,64))
    ])

    layer_6_output_1 = model_layer_6.predict(layer_4_output_1)
    layer_6_output_3 = model_layer_6.predict(layer_4_output_3)
    layer_6_output_5 = model_layer_6.predict(layer_4_output_5)
    layer_6_output_10 = model_layer_6.predict(layer_4_output_10)
    layer_6_output_15 = model_layer_6.predict(layer_4_output_15)
    layer_6_output_20 = model_layer_6.predict(layer_4_output_20)

    top_1_result = top_1_result[1600:]
    top_3_result = top_3_result[1600:]
    top_5_result = top_5_result[1600:]
    top_10_result = top_10_result[1600:]
    top_15_result = top_15_result[1600:]
    top_20_result = top_20_result[1600:]

    #--------------------------------------------------------------------------------------------------------
    model_layer_7 = tf.keras.models.Sequential([
        keras.layers.Dense(1024, activation='relu', input_shape=(1600,))
    ])

    model_hidden_7_weight = [cp_weight4, cp_weight5]
    model_layer_7.set_weights(model_hidden_7_weight)

    layer_7_output_1 = model_layer_7.predict(layer_6_output_1)
    layer_7_output_3 = model_layer_7.predict(layer_6_output_3)
    layer_7_output_5 = model_layer_7.predict(layer_6_output_5)
    layer_7_output_10 = model_layer_7.predict(layer_6_output_10)
    layer_7_output_15 = model_layer_7.predict(layer_6_output_15)
    layer_7_output_20 = model_layer_7.predict(layer_6_output_20)

    adversarial_actvation_position_1 = np.reshape(top_1_result[:1024], 1024)
    adversarial_actvation_position_3 = np.reshape(top_3_result[:1024], 1024)
    adversarial_actvation_position_5 = np.reshape(top_5_result[:1024], 1024)
    adversarial_actvation_position_10 = np.reshape(top_10_result[:1024], 1024)
    adversarial_actvation_position_15 = np.reshape(top_15_result[:1024], 1024)
    adversarial_actvation_position_20 = np.reshape(top_20_result[:1024], 1024)

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

    top_1_result = top_1_result[1024:]
    top_3_result = top_3_result[1024:]
    top_5_result = top_5_result[1024:]
    top_10_result = top_10_result[1024:]
    top_15_result = top_15_result[1024:]
    top_20_result = top_20_result[1024:]

    #--------------------------------------------------------------------------------------------------------

    #--------------------------------------------------------------------------------------------------------
    model_layer_8 = tf.keras.models.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(1024,))
    ])

    model_hidden_8_weight = [cp_weight6, cp_weight7]
    model_layer_8.set_weights(model_hidden_8_weight)

    layer_8_output_1 = model_layer_8.predict(layer_7_output_1)
    layer_8_output_3 = model_layer_8.predict(layer_7_output_3)
    layer_8_output_5 = model_layer_8.predict(layer_7_output_5)
    layer_8_output_10 = model_layer_8.predict(layer_7_output_10)
    layer_8_output_15 = model_layer_8.predict(layer_7_output_15)
    layer_8_output_20 = model_layer_8.predict(layer_7_output_20)

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
        
        layer_8_output_1[dataset_count][adversarial_actvation_position_position_1] = 0
        layer_8_output_3[dataset_count][adversarial_actvation_position_position_3] = 0
        layer_8_output_5[dataset_count][adversarial_actvation_position_position_5] = 0
        layer_8_output_10[dataset_count][adversarial_actvation_position_position_10] = 0
        layer_8_output_15[dataset_count][adversarial_actvation_position_position_15] = 0
        layer_8_output_20[dataset_count][adversarial_actvation_position_position_20] = 0

    top_1_result = top_1_result[64:]
    top_3_result = top_3_result[64:]
    top_5_result = top_5_result[64:]
    top_10_result = top_10_result[64:]
    top_15_result = top_15_result[64:]
    top_20_result = top_20_result[64:]


    #-------------------------------------------------------------------------------------------------------------------------
    model_layer_9 = tf.keras.models.Sequential([
        keras.layers.Dense(10, activation='relu', input_shape=(64,))
    ])

    model_hidden_9_weight = [cp_weight8, cp_weight9]
    model_layer_9.set_weights(model_hidden_9_weight)

    layer_9_output_1 = model_layer_9.predict(layer_8_output_1)
    layer_9_output_3 = model_layer_9.predict(layer_8_output_3)
    layer_9_output_5 = model_layer_9.predict(layer_8_output_5)
    layer_9_output_10 = model_layer_9.predict(layer_8_output_10)
    layer_9_output_15 = model_layer_9.predict(layer_8_output_15)
    layer_9_output_20 = model_layer_9.predict(layer_8_output_20)

    layer_9_output_1 = tf.nn.softmax(layer_9_output_1)
    layer_9_output_3 = tf.nn.softmax(layer_9_output_3)
    layer_9_output_5 = tf.nn.softmax(layer_9_output_5)
    layer_9_output_10 = tf.nn.softmax(layer_9_output_10)
    layer_9_output_15 = tf.nn.softmax(layer_9_output_15)
    layer_9_output_20 = tf.nn.softmax(layer_9_output_20)

    layer_9_output_1 = np.argmax(layer_9_output_1, axis=1)
    layer_9_output_3 = np.argmax(layer_9_output_3, axis=1)
    layer_9_output_5 = np.argmax(layer_9_output_5, axis=1)
    layer_9_output_10 = np.argmax(layer_9_output_10, axis=1)
    layer_9_output_15 = np.argmax(layer_9_output_15, axis=1)
    layer_9_output_20 = np.argmax(layer_9_output_20, axis=1)

    # print(layer_9_output_1[:150])
    # time.sleep(1)
    # print(layer_9_output_3[:150])
    # time.sleep(1)
    # print(layer_9_output_5[:150])
    # time.sleep(1)
    # print(layer_9_output_10[:150])
    # time.sleep(1)
    # print(layer_9_output_15[:150])
    # time.sleep(1)
    # print(layer_9_output_20[:150])

    k0 = np.where(layer_9_output_1 == analysis_num)[0]
    k1 = np.where(layer_9_output_3 == analysis_num)[0]
    k2 = np.where(layer_9_output_5 == analysis_num)[0]
    k3 = np.where(layer_9_output_10 == analysis_num)[0]
    k4 = np.where(layer_9_output_15 == analysis_num)[0]
    k5 = np.where(layer_9_output_20 == analysis_num)[0]

    print("1%  {: .2f}".format(len(k0) / len(dataset)*100))
    print("3%  {: .2f}".format(len(k1) / len(dataset)*100))
    print("5%  {: .2f}".format(len(k2) / len(dataset)*100))
    print("10%  {: .2f}".format(len(k3) / len(dataset)*100))
    print("15%  {: .2f}".format(len(k4) / len(dataset)*100))
    print("20%  {: .2f}".format(len(k5) / len(dataset)*100))

    # from sklearn.metrics import accuracy_score
    # accuracy_1 = accuracy_score(layer_9_output_1, y_full_data)
    # accuracy_3 = accuracy_score(layer_9_output_3, y_full_data)
    # accuracy_5 = accuracy_score(layer_9_output_5, y_full_data)
    # accuracy_10 = accuracy_score(layer_9_output_10, y_full_data)
    # accuracy_15 = accuracy_score(layer_9_output_15, y_full_data)
    # accuracy_20 = accuracy_score(layer_9_output_20, y_full_data)

    # print(accuracy_1)
    # print(accuracy_3)
    # print(accuracy_5)
    # print(accuracy_10)
    # print(accuracy_15)
    # print(accuracy_20)
