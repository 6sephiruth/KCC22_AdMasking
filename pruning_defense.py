import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

def top_adversarial_actvation_select(model, origin_data, adver_data, origin_label, adver_label):
    # 555 적대적
    print(origin_data.shape)
    print(adver_data.shape)

    origin_pred = model.predict(origin_data)
    origin_pred = tf.nn.softmax(origin_pred)
    origin_pred = np.argmax(origin_pred, axis=1)

    adver_pred = model.predict(adver_data)
    adver_pred = tf.nn.softmax(adver_pred)
    adver_pred = np.argmax(adver_pred, axis=1)

    print(adver_pred)

    origin_position = np.where(origin_pred == origin_label)[0]
    adv_position = np.where(adver_pred == adver_label)[0]

    print(origin_position.shape)
    print(adv_position.shape)
    
    common = list(set(origin_position) & set(adv_position))

    kk = origin_data[common]
    kk2 = adver_data[common]

    plt.imshow(kk2[8])
    plt.savefig('4.png')

    # total_activation = np.empty((len(data),0))

    # for each_layer in range(len(model.model.layers)-1):
        
    #     intermediate_layer_model = tf.keras.Model(inputs=model.model.input, outputs=model.model.layers[each_layer].output)
    #     intermediate_output = intermediate_layer_model(data)
    #     intermediate_output = np.reshape(intermediate_output, (len(intermediate_output), -1))    
        
    #     total_activation = np.append(total_activation, intermediate_output, axis=1)


    # non_activation_position = np.where(total_activation <= 0)
    # activation_position = np.where(total_activation > 0)

    # total_activation[non_activation_position] = 0
    # total_activation[activation_position] = 1

    # total_activation = np.sum(total_activation, axis=0)
