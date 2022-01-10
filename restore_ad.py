from numpy.lib.function_base import extract
from utils import *
from itertools import permutations


def check_tensor_ad(model, origin_data, origin_label):
    
    ad_data = pickle.load(open(f'./dataset/fgsm/0.1_test','rb'))
    ad_label = pickle.load(open(f'./dataset/fgsm/0.1_label','rb'))

    ad_list = np.where(ad_label == 1)[0]

    extract_origin_label = origin_label[ad_list]

    extract_origin = origin_data[ad_list]
    extract_ad = ad_data[ad_list]

    purturbation_data = extract_ad - extract_origin
    purturbation_data = np.reshape(purturbation_data, (purturbation_data.shape[0],-1))

    max_purturbation = np.max(purturbation_data, axis=1)

    # saving_ad_to_normal = np.zeros_like(extract_ad)
    saving_ad_to_normal = []
    saving_ad_to_normal_label = []

    for adversarial_count in range(extract_ad.shape[0]):

        print("{} 시작".format(adversarial_count))
        position = list(np.where(purturbation_data[adversarial_count] == max_purturbation[adversarial_count])[0])

        adversarial_to_normal = np.reshape(ad_data[ad_list][adversarial_count] , -1)


        for list_position in position:

            adversarial_to_normal[list_position] = 0

        adversarial_to_normal = np.reshape(adversarial_to_normal, (28, 28, 1))

        pred = model.predict(tf.expand_dims(adversarial_to_normal, 0))
        pred = np.argmax(pred)

        
        if pred == extract_origin_label[adversarial_count]:

            saving_ad_to_normal.append(adversarial_to_normal)
            saving_ad_to_normal_label.append(adversarial_count)

    saving_ad_to_normal = np.array(saving_ad_to_normal)
    saving_ad_to_normal_label = np.array(saving_ad_to_normal_label)

    pickle.dump(saving_ad_to_normal, open(f'./dataset/recover_data','wb'))
    pickle.dump(saving_ad_to_normal_label, open(f'./dataset/recover_label','wb'))


        # for combination_count in range(len(position)):

        #     print("{} 번째에 {}".format(len(position), combination_count))

        #     for combination_list in list(permutations(position, combination_count + 1)):

        #         adversarial_to_normal = np.reshape(ad_data[ad_list][adversarial_count] , -1)

        #         for list_position in list(combination_list):

        #             adversarial_to_normal[list_position] = 0

        #         adversarial_to_normal = np.reshape(adversarial_to_normal, (28, 28, 1))


        #         pred = model.predict(tf.expand_dims(adversarial_to_normal, 0))
        #         pred = np.argmax(pred)


        #         if pred == extract_origin_label[adversarial_count]:
        #             print("★★★★★★★★★★★★★★★★★★★★★★★★★★★★★")
        #             print("{} 번째 찾았다. ---------".format(adversarial_count))

        #             plt.imshow(adversarial_to_normal)
        #             plt.savefig("./img/{}_{}_{}.png".format(adversarial_count, combination_count, combination_list))
        #             plt.close()


        #             saving_ad_to_normal[adversarial_count] = adversarial_to_normal

        #             pickle.dump(saving_ad_to_normal, open(f'./dataset/fgsm/0.1_recover','wb'))
        #             break

        #         # else:
        #         #     adversarial_to_normal = ad_data[ad_list][adversarial_count]

        #     if pred == extract_origin_label[adversarial_count]:
        #         break


def neuron_check(model):

    ad_data = pickle.load(open(f'./dataset/fgsm/0.1_test','rb'))
    ad_label = pickle.load(open(f'./dataset/fgsm/0.1_label','rb'))

    ad_list = np.where(ad_label == 1)[0]
    extract_ad = ad_data[ad_list]

    recover_data = pickle.load(open(f'./dataset/recover_data','rb'))
    recover_label = pickle.load(open(f'./dataset/recover_label','rb'))

    select_ad = extract_ad[recover_label]

    # print(recover_data.shape)
    # print(recover_label.shape)

    # print(select_ad.shape)


    # 5. 은닉층의 출력 확인하기

    layer_0_output = tf.keras.Model(inputs=model.model.input, outputs=model.model.layers[0].output)(recover_data)

    layer_0_list = np.zeros_like(layer_0_output[0])

    for i in range(len(layer_0_output)):

        layer_position = np.where(layer_0_output[i] == 0)
        layer_0_list[layer_position] += 1

    
    # kk = np.where(layer_0_output[0] == 0)
    # layer_0_list[kk] += 1
    # print(layer_0_list)

    # print(intermediate_output[0])
    # position_where = np.where(intermediate_output[0] == 0)
    # print(position_where)