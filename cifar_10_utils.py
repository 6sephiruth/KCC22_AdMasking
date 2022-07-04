from re import X
import numpy as np
import tensorflow as tf
import pickle
from tensorflow import keras

from tensorflow.keras import layers

from tensorflow.keras.regularizers import l2

import time

def cifar_10_neuron_activation_analyze(model, data, num1, num2):

    # 여기서 data는 pickle.load(open(f'./dataset/targeted_cw/0-1','rb'))

    total_activation = np.empty((len(data),0))

    for each_layer in range(len(model.layers)-1):

        intermediate_layer_model = tf.keras.Model(inputs=model.input, outputs=model.layers[each_layer].output)
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

    pickle.dump(arrange_result, open(f'./model/cifar-10/targeted_analysis/{num1}-{num2}','wb'))
    #### pickle.dump(arrange_result, open(f'./dataset/targeted_half_analysis/{num1}-{num2}','wb'))

# 압축
def cifar_model_unifying(analysis_num ,model, dataset):
# def cifar_model_unifying(model, dataset, y_full_data):

    model_activation_count = 0

    # model의 전체 actvation 개수를 구한다. ※ 마지막 출력 layer 제외
    for each_layer in range(len(model.layers)-1):

        output_shape = model.layers[each_layer].output.shape

        # CNN으로 채널이 포함된 것, ex) (None, 32, 32, 3)
        if len(output_shape) == 4:
            model_activation_count += output_shape[1] * output_shape[2] * output_shape[3]
        # DNN으로 채널이 미 포함 된 것, ex) (None, 64)
        elif len(output_shape) == 2:
            model_activation_count += output_shape[1]

    adversarial_activation_position = np.zeros((6, model_activation_count))
    normal_activation_position = np.zeros((6, model_activation_count))

    for i in range(10):

        for j in range(10):

            for k in range(6):

                data = pickle.load(open(f'./model/cifar-10/targeted_analysis/{i}-{j}','rb'))
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


    ############################################################################

    ####################################################################
    cp_weight0 = model.get_weights()[0]
    cp_weight1 = model.get_weights()[1]
    cp_weight2 = model.get_weights()[2]
    cp_weight3 = model.get_weights()[3]
    cp_weight4 = model.get_weights()[4]
    cp_weight5 = model.get_weights()[5]
    ####################################################################

    intermediate_layer_model = tf.keras.Model(inputs=model.input, outputs=model.layers[0].output)

    layer_1_output_1 = np.array(intermediate_layer_model(dataset))
    layer_1_output_3 = np.array(intermediate_layer_model(dataset))
    layer_1_output_5 = np.array(intermediate_layer_model(dataset))
    layer_1_output_10 = np.array(intermediate_layer_model(dataset))
    layer_1_output_15 = np.array(intermediate_layer_model(dataset))
    layer_1_output_20 = np.array(intermediate_layer_model(dataset))

    # adversarial_actvation_position_1 = np.reshape(top_1_result[:3072], (32, 32, 3))
    # adversarial_actvation_position_3 = np.reshape(top_3_result[:3072], (32, 32, 3))
    # adversarial_actvation_position_5 = np.reshape(top_5_result[:3072], (32, 32, 3))
    # adversarial_actvation_position_10 = np.reshape(top_10_result[:3072], (32, 32, 3))
    # adversarial_actvation_position_15 = np.reshape(top_15_result[:3072], (32, 32, 3))
    # adversarial_actvation_position_20 = np.reshape(top_20_result[:3072], (32, 32, 3))

    # adversarial_actvation_position_position_1 = np.where(adversarial_actvation_position_1 == 1)
    # adversarial_actvation_position_position_3 = np.where(adversarial_actvation_position_3 == 1)
    # adversarial_actvation_position_position_5 = np.where(adversarial_actvation_position_5 == 1)
    # adversarial_actvation_position_position_10 = np.where(adversarial_actvation_position_10 == 1)
    # adversarial_actvation_position_position_15 = np.where(adversarial_actvation_position_15 == 1)
    # adversarial_actvation_position_position_20 = np.where(adversarial_actvation_position_20 == 1)

    # for dataset_count in range(len(dataset)):

    #     layer_1_output_1[dataset_count][adversarial_actvation_position_position_1] = 0
    #     layer_1_output_3[dataset_count][adversarial_actvation_position_position_3] = 0
    #     layer_1_output_5[dataset_count][adversarial_actvation_position_position_5] = 0
    #     layer_1_output_10[dataset_count][adversarial_actvation_position_position_10] = 0
    #     layer_1_output_15[dataset_count][adversarial_actvation_position_position_15] = 0
    #     layer_1_output_20[dataset_count][adversarial_actvation_position_position_20] = 0

    top_1_result = top_1_result[3072:]
    top_3_result = top_3_result[3072:]
    top_5_result = top_5_result[3072:]
    top_10_result = top_10_result[3072:]
    top_15_result = top_15_result[3072:]
    top_20_result = top_20_result[3072:]

    #-------------------------------------------------------------------------------------------------
    model_layer_2 = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, kernel_size=(3, 3), padding="same", kernel_initializer="he_normal", kernel_regularizer=l2(1e-4), input_shape=(32,32,3))
    ])

    model_hidden_2_weight = [cp_weight0, cp_weight1]
    model_layer_2.set_weights(model_hidden_2_weight)

    layer_2_output_1 = model_layer_2.predict(layer_1_output_1)
    layer_2_output_3 = model_layer_2.predict(layer_1_output_3)
    layer_2_output_5 = model_layer_2.predict(layer_1_output_5)
    layer_2_output_10 = model_layer_2.predict(layer_1_output_10)
    layer_2_output_15 = model_layer_2.predict(layer_1_output_15)
    layer_2_output_20 = model_layer_2.predict(layer_1_output_20)

    adversarial_actvation_position_1 = np.reshape(top_1_result[:16384], (32, 32, 16))
    adversarial_actvation_position_3 = np.reshape(top_3_result[:16384], (32, 32, 16))
    adversarial_actvation_position_5 = np.reshape(top_5_result[:16384], (32, 32, 16))
    adversarial_actvation_position_10 = np.reshape(top_10_result[:16384], (32, 32, 16))
    adversarial_actvation_position_15 = np.reshape(top_15_result[:16384], (32, 32, 16))
    adversarial_actvation_position_20 = np.reshape(top_20_result[:16384], (32, 32, 16))

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

    top_1_result = top_1_result[16384:]
    top_3_result = top_3_result[16384:]
    top_5_result = top_5_result[16384:]
    top_10_result = top_10_result[16384:]
    top_15_result = top_15_result[16384:]
    top_20_result = top_20_result[16384:]

    #--------------------------------------------------------------------------------------------------------
    model_layer_3 = tf.keras.models.Sequential([
        keras.layers.BatchNormalization(input_shape=(32,32,16))
    ])

    model_hidden_3_weight = [cp_weight2, cp_weight3, cp_weight4, cp_weight5]
    model_layer_3.set_weights(model_hidden_3_weight)

    layer_3_output_1 = model_layer_3.predict(layer_2_output_1)
    layer_3_output_3 = model_layer_3.predict(layer_2_output_3)
    layer_3_output_5 = model_layer_3.predict(layer_2_output_5)
    layer_3_output_10 = model_layer_3.predict(layer_2_output_10)
    layer_3_output_15 = model_layer_3.predict(layer_2_output_15)
    layer_3_output_20 = model_layer_3.predict(layer_2_output_20)

    adversarial_actvation_position_1 = np.reshape(top_1_result[:16384], (32, 32, 16))
    adversarial_actvation_position_3 = np.reshape(top_3_result[:16384], (32, 32, 16))
    adversarial_actvation_position_5 = np.reshape(top_5_result[:16384], (32, 32, 16))
    adversarial_actvation_position_10 = np.reshape(top_10_result[:16384], (32, 32, 16))
    adversarial_actvation_position_15 = np.reshape(top_15_result[:16384], (32, 32, 16))
    adversarial_actvation_position_20 = np.reshape(top_20_result[:16384], (32, 32, 16))

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

    top_1_result = top_1_result[16384:]
    top_3_result = top_3_result[16384:]
    top_5_result = top_5_result[16384:]
    top_10_result = top_10_result[16384:]
    top_15_result = top_15_result[16384:]
    top_20_result = top_20_result[16384:]

    #--------------------------------------------------------------------------------------------------------
    model_layer_4 = tf.keras.models.Sequential([
        keras.layers.Activation(tf.nn.relu , input_shape=(32,32,16))
    ])

    layer_4_output_1 = model_layer_4.predict(layer_3_output_1)
    layer_4_output_3 = model_layer_4.predict(layer_3_output_3)
    layer_4_output_5 = model_layer_4.predict(layer_3_output_5)
    layer_4_output_10 = model_layer_4.predict(layer_3_output_10)
    layer_4_output_15 = model_layer_4.predict(layer_3_output_15)
    layer_4_output_20 = model_layer_4.predict(layer_3_output_20)

    x_1 = layer_4_output_1
    x_3 = layer_4_output_3
    x_5 = layer_4_output_5
    x_10 = layer_4_output_10
    x_15 = layer_4_output_15
    x_20 = layer_4_output_20

    layer_output_1 = layer_4_output_1
    layer_output_3 = layer_4_output_3
    layer_output_5 = layer_4_output_5
    layer_output_10 = layer_4_output_10
    layer_output_15 = layer_4_output_15
    layer_output_20 = layer_4_output_20

    # 규민 테스트
    top_1_result = top_1_result[16384:]
    top_3_result = top_3_result[16384:]
    top_5_result = top_5_result[16384:]
    top_10_result = top_10_result[16384:]
    top_15_result = top_15_result[16384:]
    top_20_result = top_20_result[16384:]


    #-------------------------------------------------------------------------------------------------
    # for문 시작

    n = 9
    channels = [16, 32, 64]
    hidden_weight_count = 6

    for c in channels:
        for i in range(n):

            subsampling = i == 0 and c > 16
            strides = (2, 2) if subsampling else (1, 1)


            print("{}----{}".format(c, i))
            first_input_channel = c
            if c == 16:
                input_shape_value = 32
                first_input_shape_value = 32
            elif c == 32:
                input_shape_value = 16
                first_input_shape_value = 16
            elif c == 64:
                input_shape_value = 8
                first_input_shape_value = 8



            if c == 32 and i == 0:
                first_input_shape_value = 32
                first_input_channel = 16

            if c == 64 and i == 0:
                first_input_shape_value = 16
                first_input_channel = 32


            model_layer = tf.keras.models.Sequential([
                tf.keras.layers.Conv2D(c, kernel_size=(3, 3), padding="same", strides=strides, kernel_initializer="he_normal", kernel_regularizer=l2(1e-4), input_shape=(first_input_shape_value ,first_input_shape_value, first_input_channel))
            ])

            model_hidden_weight = [model.get_weights()[hidden_weight_count], model.get_weights()[hidden_weight_count + 1]]
            model_layer.set_weights(model_hidden_weight)
            hidden_weight_count += 2

            layer_output_1 = model_layer.predict(layer_output_1)
            layer_output_3 = model_layer.predict(layer_output_3)
            layer_output_5 = model_layer.predict(layer_output_5)
            layer_output_10 = model_layer.predict(layer_output_10)
            layer_output_15 = model_layer.predict(layer_output_15)
            layer_output_20 = model_layer.predict(layer_output_20)

            adversarial_actvation_position_1 = np.reshape(top_1_result[:input_shape_value*input_shape_value*c], (input_shape_value, input_shape_value, c))
            adversarial_actvation_position_3 = np.reshape(top_3_result[:input_shape_value*input_shape_value*c], (input_shape_value, input_shape_value, c))
            adversarial_actvation_position_5 = np.reshape(top_5_result[:input_shape_value*input_shape_value*c], (input_shape_value, input_shape_value, c))
            adversarial_actvation_position_10 = np.reshape(top_10_result[:input_shape_value*input_shape_value*c], (input_shape_value, input_shape_value, c))
            adversarial_actvation_position_15 = np.reshape(top_15_result[:input_shape_value*input_shape_value*c], (input_shape_value, input_shape_value, c))
            adversarial_actvation_position_20 = np.reshape(top_20_result[:input_shape_value*input_shape_value*c], (input_shape_value, input_shape_value, c))

            adversarial_actvation_position_position_1 = np.where(adversarial_actvation_position_1 == 1)
            adversarial_actvation_position_position_3 = np.where(adversarial_actvation_position_3 == 1)
            adversarial_actvation_position_position_5 = np.where(adversarial_actvation_position_5 == 1)
            adversarial_actvation_position_position_10 = np.where(adversarial_actvation_position_10 == 1)
            adversarial_actvation_position_position_15 = np.where(adversarial_actvation_position_15 == 1)
            adversarial_actvation_position_position_20 = np.where(adversarial_actvation_position_20 == 1)

            for dataset_count in range(len(dataset)):
                layer_output_1[dataset_count][adversarial_actvation_position_position_1] = 0
                layer_output_3[dataset_count][adversarial_actvation_position_position_3] = 0
                layer_output_5[dataset_count][adversarial_actvation_position_position_5] = 0
                layer_output_10[dataset_count][adversarial_actvation_position_position_10] = 0
                layer_output_15[dataset_count][adversarial_actvation_position_position_15] = 0
                layer_output_20[dataset_count][adversarial_actvation_position_position_20] = 0

            top_1_result = top_1_result[input_shape_value*input_shape_value*c:]
            top_3_result = top_3_result[input_shape_value*input_shape_value*c:]
            top_5_result = top_5_result[input_shape_value*input_shape_value*c:]
            top_10_result = top_10_result[input_shape_value*input_shape_value*c:]
            top_15_result = top_15_result[input_shape_value*input_shape_value*c:]
            top_20_result = top_20_result[input_shape_value*input_shape_value*c:]

            ##############
            model_layer = tf.keras.models.Sequential([
                tf.keras.layers.BatchNormalization(input_shape=(input_shape_value ,input_shape_value,c))
            ])

            model_hidden_weight = [model.get_weights()[hidden_weight_count], model.get_weights()[hidden_weight_count + 1], model.get_weights()[hidden_weight_count + 2], model.get_weights()[hidden_weight_count + 3]]
            model_layer.set_weights(model_hidden_weight)

            hidden_weight_count += 4

            layer_output_1 = model_layer.predict(layer_output_1)
            layer_output_3 = model_layer.predict(layer_output_3)
            layer_output_5 = model_layer.predict(layer_output_5)
            layer_output_10 = model_layer.predict(layer_output_10)
            layer_output_15 = model_layer.predict(layer_output_15)
            layer_output_20 = model_layer.predict(layer_output_20)

            adversarial_actvation_position_1 = np.reshape(top_1_result[:input_shape_value*input_shape_value*c], (input_shape_value, input_shape_value, c))
            adversarial_actvation_position_3 = np.reshape(top_3_result[:input_shape_value*input_shape_value*c], (input_shape_value, input_shape_value, c))
            adversarial_actvation_position_5 = np.reshape(top_5_result[:input_shape_value*input_shape_value*c], (input_shape_value, input_shape_value, c))
            adversarial_actvation_position_10 = np.reshape(top_10_result[:input_shape_value*input_shape_value*c], (input_shape_value, input_shape_value, c))
            adversarial_actvation_position_15 = np.reshape(top_15_result[:input_shape_value*input_shape_value*c], (input_shape_value, input_shape_value, c))
            adversarial_actvation_position_20 = np.reshape(top_20_result[:input_shape_value*input_shape_value*c], (input_shape_value, input_shape_value, c))

            adversarial_actvation_position_position_1 = np.where(adversarial_actvation_position_1 == 1)
            adversarial_actvation_position_position_3 = np.where(adversarial_actvation_position_3 == 1)
            adversarial_actvation_position_position_5 = np.where(adversarial_actvation_position_5 == 1)
            adversarial_actvation_position_position_10 = np.where(adversarial_actvation_position_10 == 1)
            adversarial_actvation_position_position_15 = np.where(adversarial_actvation_position_15 == 1)
            adversarial_actvation_position_position_20 = np.where(adversarial_actvation_position_20 == 1)

            for dataset_count in range(len(dataset)):

                layer_output_1[dataset_count][adversarial_actvation_position_position_1] = 0
                layer_output_3[dataset_count][adversarial_actvation_position_position_3] = 0
                layer_output_5[dataset_count][adversarial_actvation_position_position_5] = 0
                layer_output_10[dataset_count][adversarial_actvation_position_position_10] = 0
                layer_output_15[dataset_count][adversarial_actvation_position_position_15] = 0
                layer_output_20[dataset_count][adversarial_actvation_position_position_20] = 0

            top_1_result = top_1_result[input_shape_value*input_shape_value*c:]
            top_3_result = top_3_result[input_shape_value*input_shape_value*c:]
            top_5_result = top_5_result[input_shape_value*input_shape_value*c:]
            top_10_result = top_10_result[input_shape_value*input_shape_value*c:]
            top_15_result = top_15_result[input_shape_value*input_shape_value*c:]
            top_20_result = top_20_result[input_shape_value*input_shape_value*c:]

            model_layer = tf.keras.models.Sequential([
                keras.layers.Activation(tf.nn.relu , input_shape=(input_shape_value ,input_shape_value,c))
            ])

            layer_output_1 = model_layer.predict(layer_output_1)
            layer_output_3 = model_layer.predict(layer_output_3)
            layer_output_5 = model_layer.predict(layer_output_5)
            layer_output_10 = model_layer.predict(layer_output_10)
            layer_output_15 = model_layer.predict(layer_output_15)
            layer_output_20 = model_layer.predict(layer_output_20)

            # 규민 테스트
            top_1_result = top_1_result[input_shape_value * input_shape_value * c:]
            top_3_result = top_3_result[input_shape_value * input_shape_value * c:]
            top_5_result = top_5_result[input_shape_value * input_shape_value * c:]
            top_10_result = top_10_result[input_shape_value * input_shape_value * c:]
            top_15_result = top_15_result[input_shape_value * input_shape_value * c:]
            top_20_result = top_20_result[input_shape_value * input_shape_value * c:]

            model_layer = tf.keras.models.Sequential([
                tf.keras.layers.Conv2D(c, kernel_size=(3, 3), padding="same", kernel_initializer="he_normal", kernel_regularizer=l2(1e-4), input_shape=(input_shape_value ,input_shape_value,c))
            ])

            model_hidden_weight = [ model.get_weights()[hidden_weight_count], model.get_weights()[hidden_weight_count + 1]]

            model_layer.set_weights(model_hidden_weight)

            hidden_weight_count += 2

            layer_output_1 = model_layer.predict(layer_output_1)

            layer_output_3 = model_layer.predict(layer_output_3)

            layer_output_5 = model_layer.predict(layer_output_5)

            layer_output_10 = model_layer.predict(layer_output_10)

            layer_output_15 = model_layer.predict(layer_output_15)

            layer_output_20 = model_layer.predict(layer_output_20)

            adversarial_actvation_position_1 = np.reshape(top_1_result[:input_shape_value*input_shape_value*c], (input_shape_value, input_shape_value, c))
            adversarial_actvation_position_3 = np.reshape(top_3_result[:input_shape_value*input_shape_value*c], (input_shape_value, input_shape_value, c))
            adversarial_actvation_position_5 = np.reshape(top_5_result[:input_shape_value*input_shape_value*c], (input_shape_value, input_shape_value, c))
            adversarial_actvation_position_10 = np.reshape(top_10_result[:input_shape_value*input_shape_value*c], (input_shape_value, input_shape_value, c))
            adversarial_actvation_position_15 = np.reshape(top_15_result[:input_shape_value*input_shape_value*c], (input_shape_value, input_shape_value, c))
            adversarial_actvation_position_20 = np.reshape(top_20_result[:input_shape_value*input_shape_value*c], (input_shape_value, input_shape_value, c))

            adversarial_actvation_position_position_1 = np.where(adversarial_actvation_position_1 == 1)
            adversarial_actvation_position_position_3 = np.where(adversarial_actvation_position_3 == 1)
            adversarial_actvation_position_position_5 = np.where(adversarial_actvation_position_5 == 1)
            adversarial_actvation_position_position_10 = np.where(adversarial_actvation_position_10 == 1)
            adversarial_actvation_position_position_15 = np.where(adversarial_actvation_position_15 == 1)
            adversarial_actvation_position_position_20 = np.where(adversarial_actvation_position_20 == 1)

            for dataset_count in range(len(dataset)):

                layer_output_1[dataset_count][adversarial_actvation_position_position_1] = 0
                layer_output_3[dataset_count][adversarial_actvation_position_position_3] = 0
                layer_output_5[dataset_count][adversarial_actvation_position_position_5] = 0
                layer_output_10[dataset_count][adversarial_actvation_position_position_10] = 0
                layer_output_15[dataset_count][adversarial_actvation_position_position_15] = 0
                layer_output_20[dataset_count][adversarial_actvation_position_position_20] = 0

            top_1_result = top_1_result[input_shape_value*input_shape_value*c:]
            top_3_result = top_3_result[input_shape_value*input_shape_value*c:]
            top_5_result = top_5_result[input_shape_value*input_shape_value*c:]
            top_10_result = top_10_result[input_shape_value*input_shape_value*c:]
            top_15_result = top_15_result[input_shape_value*input_shape_value*c:]
            top_20_result = top_20_result[input_shape_value*input_shape_value*c:]


            ##############
            if subsampling:

                if c == 32:
                    cc = 16
                if c == 64:
                    cc = 32

                model_layer = tf.keras.models.Sequential([
                    tf.keras.layers.Conv2D(c, kernel_size=(1, 1), strides=(2, 2), padding="same", kernel_initializer="he_normal", kernel_regularizer=l2(1e-4), input_shape=(input_shape_value ,input_shape_value,cc))
                ])


                model_hidden_weight = [model.get_weights()[hidden_weight_count], model.get_weights()[hidden_weight_count + 1]]
                model_layer.set_weights(model_hidden_weight)

                hidden_weight_count += 2

                layer_output_1 = model_layer.predict(x_1)
                layer_output_3 = model_layer.predict(x_3)
                layer_output_5 = model_layer.predict(x_5)
                layer_output_10 = model_layer.predict(x_10)
                layer_output_15 = model_layer.predict(x_15)
                layer_output_20 = model_layer.predict(x_20)

                x_1 = layer_output_1
                x_3 = layer_output_3
                x_5 = layer_output_5
                x_10 = layer_output_10
                x_15 = layer_output_15
                x_20 = layer_output_20

                adversarial_actvation_position_1 = np.reshape(top_1_result[:input_shape_value*input_shape_value*c], (input_shape_value, input_shape_value, c))
                adversarial_actvation_position_3 = np.reshape(top_3_result[:input_shape_value*input_shape_value*c], (input_shape_value, input_shape_value, c))
                adversarial_actvation_position_5 = np.reshape(top_5_result[:input_shape_value*input_shape_value*c], (input_shape_value, input_shape_value, c))
                adversarial_actvation_position_10 = np.reshape(top_10_result[:input_shape_value*input_shape_value*c], (input_shape_value, input_shape_value, c))
                adversarial_actvation_position_15 = np.reshape(top_15_result[:input_shape_value*input_shape_value*c], (input_shape_value, input_shape_value, c))
                adversarial_actvation_position_20 = np.reshape(top_20_result[:input_shape_value*input_shape_value*c], (input_shape_value, input_shape_value, c))

                adversarial_actvation_position_position_1 = np.where(adversarial_actvation_position_1 == 1)
                adversarial_actvation_position_position_3 = np.where(adversarial_actvation_position_3 == 1)
                adversarial_actvation_position_position_5 = np.where(adversarial_actvation_position_5 == 1)
                adversarial_actvation_position_position_10 = np.where(adversarial_actvation_position_10 == 1)
                adversarial_actvation_position_position_15 = np.where(adversarial_actvation_position_15 == 1)
                adversarial_actvation_position_position_20 = np.where(adversarial_actvation_position_20 == 1)

                for dataset_count in range(len(dataset)):

                    layer_output_1[dataset_count][adversarial_actvation_position_position_1] = 0
                    layer_output_3[dataset_count][adversarial_actvation_position_position_3] = 0
                    layer_output_5[dataset_count][adversarial_actvation_position_position_5] = 0
                    layer_output_10[dataset_count][adversarial_actvation_position_position_10] = 0
                    layer_output_15[dataset_count][adversarial_actvation_position_position_15] = 0
                    layer_output_20[dataset_count][adversarial_actvation_position_position_20] = 0

                top_1_result = top_1_result[input_shape_value*input_shape_value*c:]
                top_3_result = top_3_result[input_shape_value*input_shape_value*c:]
                top_5_result = top_5_result[input_shape_value*input_shape_value*c:]
                top_10_result = top_10_result[input_shape_value*input_shape_value*c:]
                top_15_result = top_15_result[input_shape_value*input_shape_value*c:]
                top_20_result = top_20_result[input_shape_value*input_shape_value*c:]

            model_layer = tf.keras.models.Sequential([
                tf.keras.layers.BatchNormalization(input_shape=(input_shape_value ,input_shape_value,c))
            ])

            model_hidden_weight = [model.get_weights()[hidden_weight_count], model.get_weights()[hidden_weight_count + 1], model.get_weights()[hidden_weight_count + 2], model.get_weights()[hidden_weight_count + 3]]
            model_layer.set_weights(model_hidden_weight)
            hidden_weight_count += 4

            layer_output_1 = model_layer.predict(layer_output_1)
            layer_output_3 = model_layer.predict(layer_output_3)
            layer_output_5 = model_layer.predict(layer_output_5)
            layer_output_10 = model_layer.predict(layer_output_10)
            layer_output_15 = model_layer.predict(layer_output_15)
            layer_output_20 = model_layer.predict(layer_output_20)

            adversarial_actvation_position_1 = np.reshape(top_1_result[:input_shape_value*input_shape_value*c], (input_shape_value, input_shape_value, c))
            adversarial_actvation_position_3 = np.reshape(top_3_result[:input_shape_value*input_shape_value*c], (input_shape_value, input_shape_value, c))
            adversarial_actvation_position_5 = np.reshape(top_5_result[:input_shape_value*input_shape_value*c], (input_shape_value, input_shape_value, c))
            adversarial_actvation_position_10 = np.reshape(top_10_result[:input_shape_value*input_shape_value*c], (input_shape_value, input_shape_value, c))
            adversarial_actvation_position_15 = np.reshape(top_15_result[:input_shape_value*input_shape_value*c], (input_shape_value, input_shape_value, c))
            adversarial_actvation_position_20 = np.reshape(top_20_result[:input_shape_value*input_shape_value*c], (input_shape_value, input_shape_value, c))

            adversarial_actvation_position_position_1 = np.where(adversarial_actvation_position_1 == 1)
            adversarial_actvation_position_position_3 = np.where(adversarial_actvation_position_3 == 1)
            adversarial_actvation_position_position_5 = np.where(adversarial_actvation_position_5 == 1)
            adversarial_actvation_position_position_10 = np.where(adversarial_actvation_position_10 == 1)
            adversarial_actvation_position_position_15 = np.where(adversarial_actvation_position_15 == 1)
            adversarial_actvation_position_position_20 = np.where(adversarial_actvation_position_20 == 1)

            for dataset_count in range(len(dataset)):

                layer_output_1[dataset_count][adversarial_actvation_position_position_1] = 0
                layer_output_3[dataset_count][adversarial_actvation_position_position_3] = 0
                layer_output_5[dataset_count][adversarial_actvation_position_position_5] = 0
                layer_output_10[dataset_count][adversarial_actvation_position_position_10] = 0
                layer_output_15[dataset_count][adversarial_actvation_position_position_15] = 0
                layer_output_20[dataset_count][adversarial_actvation_position_position_20] = 0

            top_1_result = top_1_result[input_shape_value*input_shape_value*c:]
            top_3_result = top_3_result[input_shape_value*input_shape_value*c:]
            top_5_result = top_5_result[input_shape_value*input_shape_value*c:]
            top_10_result = top_10_result[input_shape_value*input_shape_value*c:]
            top_15_result = top_15_result[input_shape_value*input_shape_value*c:]
            top_20_result = top_20_result[input_shape_value*input_shape_value*c:]

            y_1 = layer_output_1
            y_3 = layer_output_3
            y_5 = layer_output_5
            y_10 = layer_output_10
            y_15 = layer_output_15
            y_20 = layer_output_20

            layer_output_1 = keras.layers.Add()([x_1, y_1])
            layer_output_3 = keras.layers.Add()([x_3, y_3])
            layer_output_5 = keras.layers.Add()([x_5, y_5])
            layer_output_10 = keras.layers.Add()([x_10, y_10])
            layer_output_15 = keras.layers.Add()([x_15, y_15])
            layer_output_20 = keras.layers.Add()([x_20, y_20])

            # 규민 테스트
            top_1_result = top_1_result[input_shape_value * input_shape_value * c:]
            top_3_result = top_3_result[input_shape_value * input_shape_value * c:]
            top_5_result = top_5_result[input_shape_value * input_shape_value * c:]
            top_10_result = top_10_result[input_shape_value * input_shape_value * c:]
            top_15_result = top_15_result[input_shape_value * input_shape_value * c:]
            top_20_result = top_20_result[input_shape_value * input_shape_value * c:]

            model_layer = tf.keras.models.Sequential([
                keras.layers.Activation(tf.nn.relu , input_shape=(input_shape_value, input_shape_value, c))
            ])

            layer_output_1 = model_layer.predict(layer_output_1)
            layer_output_3 = model_layer.predict(layer_output_3)
            layer_output_5 = model_layer.predict(layer_output_5)
            layer_output_10 = model_layer.predict(layer_output_10)
            layer_output_15 = model_layer.predict(layer_output_15)
            layer_output_20 = model_layer.predict(layer_output_20)

            x_1 = layer_output_1
            x_3 = layer_output_3
            x_5 = layer_output_5
            x_10 = layer_output_10
            x_15 = layer_output_15
            x_20 = layer_output_20

            # 규민 테스트
            top_1_result = top_1_result[input_shape_value * input_shape_value * c:]
            top_3_result = top_3_result[input_shape_value * input_shape_value * c:]
            top_5_result = top_5_result[input_shape_value * input_shape_value * c:]
            top_10_result = top_10_result[input_shape_value * input_shape_value * c:]
            top_15_result = top_15_result[input_shape_value * input_shape_value * c:]
            top_20_result = top_20_result[input_shape_value * input_shape_value * c:]


    model_layer = tf.keras.models.Sequential([
        tf.keras.layers.GlobalAveragePooling2D(input_shape=(input_shape_value ,input_shape_value,c))
    ])

    layer_output_1 = model_layer.predict(layer_output_1)
    layer_output_3 = model_layer.predict(layer_output_3)
    layer_output_5 = model_layer.predict(layer_output_5)
    layer_output_10 = model_layer.predict(layer_output_10)
    layer_output_15 = model_layer.predict(layer_output_15)
    layer_output_20 = model_layer.predict(layer_output_20)

    # 규민 테스트
    top_1_result = top_1_result[input_shape_value * input_shape_value * c:]
    top_3_result = top_3_result[input_shape_value * input_shape_value * c:]
    top_5_result = top_5_result[input_shape_value * input_shape_value * c:]
    top_10_result = top_10_result[input_shape_value * input_shape_value * c:]
    top_15_result = top_15_result[input_shape_value * input_shape_value * c:]
    top_20_result = top_20_result[input_shape_value * input_shape_value * c:]

    model_layer = tf.keras.models.Sequential([
        keras.layers.Flatten(input_shape=(64,))
    ])

    layer_output_1 = model_layer.predict(layer_output_1)
    layer_output_3 = model_layer.predict(layer_output_3)
    layer_output_5 = model_layer.predict(layer_output_5)
    layer_output_10 = model_layer.predict(layer_output_10)
    layer_output_15 = model_layer.predict(layer_output_15)
    layer_output_20 = model_layer.predict(layer_output_20)

    # 규민 테스트
    top_1_result = top_1_result[64:]
    top_3_result = top_3_result[64:]
    top_5_result = top_5_result[64:]
    top_10_result = top_10_result[64:]
    top_15_result = top_15_result[64:]
    top_20_result = top_20_result[64:]


    model_layer = tf.keras.models.Sequential([
        keras.layers.Dense(10, activation='relu', input_shape=(64,))
    ])

    model_hidden_weight = [model.get_weights()[hidden_weight_count], model.get_weights()[hidden_weight_count + 1]]
    model_layer.set_weights(model_hidden_weight)

    layer_output_1 = model_layer.predict(layer_output_1)
    layer_output_3 = model_layer.predict(layer_output_3)
    layer_output_5 = model_layer.predict(layer_output_5)
    layer_output_10 = model_layer.predict(layer_output_10)
    layer_output_15 = model_layer.predict(layer_output_15)
    layer_output_20 = model_layer.predict(layer_output_20)


    layer_output_1 = tf.nn.softmax(layer_output_1)
    layer_output_3 = tf.nn.softmax(layer_output_3)
    layer_output_5 = tf.nn.softmax(layer_output_5)
    layer_output_10 = tf.nn.softmax(layer_output_10)
    layer_output_15 = tf.nn.softmax(layer_output_15)
    layer_output_20 = tf.nn.softmax(layer_output_20)

    layer_output_1 = np.argmax(layer_output_1, axis=1)
    layer_output_3 = np.argmax(layer_output_3, axis=1)
    layer_output_5 = np.argmax(layer_output_5, axis=1)
    layer_output_10 = np.argmax(layer_output_10, axis=1)
    layer_output_15 = np.argmax(layer_output_15, axis=1)
    layer_output_20 = np.argmax(layer_output_20, axis=1)

    k0 = np.where(layer_output_1 == analysis_num)[0]
    k1 = np.where(layer_output_3 == analysis_num)[0]
    k2 = np.where(layer_output_5 == analysis_num)[0]
    k3 = np.where(layer_output_10 == analysis_num)[0]
    k4 = np.where(layer_output_15 == analysis_num)[0]
    k5 = np.where(layer_output_20 == analysis_num)[0]

    print("1%  {: .2f}".format(len(k0) / len(dataset)*100))
    print("3%  {: .2f}".format(len(k1) / len(dataset)*100))
    print("5%  {: .2f}".format(len(k2) / len(dataset)*100))
    print("10%  {: .2f}".format(len(k3) / len(dataset)*100))
    print("15%  {: .2f}".format(len(k4) / len(dataset)*100))
    print("20%  {: .2f}".format(len(k5) / len(dataset)*100))

    # from sklearn.metrics import accuracy_score
    # accuracy_1 = accuracy_score(layer_output_1, y_full_data)
    # accuracy_3 = accuracy_score(layer_output_3, y_full_data)
    # accuracy_5 = accuracy_score(layer_output_5, y_full_data)
    # accuracy_10 = accuracy_score(layer_output_10, y_full_data)
    # accuracy_15 = accuracy_score(layer_output_15, y_full_data)
    # accuracy_20 = accuracy_score(layer_output_20, y_full_data)

    # print(accuracy_1)
    # print(accuracy_3)
    # print(accuracy_5)
    # print(accuracy_10)
    # print(accuracy_15)
    # print(accuracy_20)
    # print("-----------------------")
    # print(top_1_result.shape)