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

    saving_ad_to_normal = np.zeros_like(extract_ad)

    for adversarial_count in range(extract_ad.shape[0]):
        print("{} 시작".format(adversarial_count))
        position = list(np.where(purturbation_data[adversarial_count] == max_purturbation[adversarial_count])[0])


        for combination_count in range(len(position)):
            print("{} 번째에 {}".format(len(position), combination_count))
            for combination_list in list(permutations(position, combination_count + 1)):

                adversarial_to_normal = np.reshape(ad_data[ad_list][adversarial_count] , -1)

                for list_position in list(combination_list):

                    adversarial_to_normal[list_position] = 0

                adversarial_to_normal = np.reshape(adversarial_to_normal, (28, 28, 1))

                # plt.imshow(adversarial_to_normal)
                # plt.savefig("./img/{}_{}_{}.png".format(adversarial_count, combination_count, combination_list))
                # plt.close()

                pred = model.predict(tf.expand_dims(adversarial_to_normal, 0))
                pred = np.argmax(pred)


                if pred == extract_origin_label[adversarial_count]:
                    print("★★★★★★★★★★★★★★★★★★★★★★★★★★★★★")
                    print("{} 번째 찾았다. ---------".format(adversarial_count))

                    plt.imshow(adversarial_to_normal)
                    plt.savefig("./img/{}_{}_{}.png".format(adversarial_count, combination_count, combination_list))
                    plt.close()


                    saving_ad_to_normal[adversarial_count] = adversarial_to_normal

                    pickle.dump(saving_ad_to_normal, open(f'./dataset/fgsm/0.1_recover','wb'))
                    break

                # else:
                #     adversarial_to_normal = ad_data[ad_list][adversarial_count]

            if pred == extract_origin_label[adversarial_count]:
                break




def ffds(model, origin_data, origin_label):
    
    ad_data = pickle.load(open(f'./dataset/fgsm/0.1_test','rb'))
    ad_label = pickle.load(open(f'./dataset/fgsm/0.1_label','rb'))

    ad_list = np.where(ad_label == 1)[0]

    extract_origin_label = origin_label[ad_list]

    extract_origin = origin_data[ad_list]
    extract_ad = ad_data[ad_list]

    num = 0

    pred = model.predict(tf.expand_dims(extract_origin[num], 0))
    pred = np.argmax(pred)
    print("정상 {}".format(pred))

    pred = model.predict(tf.expand_dims(extract_ad[num], 0))
    pred = np.argmax(pred)
    print("적대적 {}".format(pred))

    recover = pickle.load(open(f'./dataset/fgsm/0.1_recover','rb'))

    pred = model.predict(tf.expand_dims(recover[num], 0))
    pred = np.argmax(pred)
    print("복구 {}".format(pred))


        # if pred == extract_origin_label[adversarial_count]:
        #     break


    #             # reshape_extract_ad = np.reshape(extract_ad[adversarial_count], (28, 28, 1))

    #             # plt.imshow(reshape_extract_ad)
    #             # plt.savefig("./img/{}_{}_{}.png".format(adversarial_count, combination_count, combination_list))

    #             pred = model.predict(tf.expand_dims(reshape_extract_ad, 0))
    #             pred = np.argmax(pred)

    #             if pred == extract_origin_label[adversarial_count]:
    #                 restore_adversarial[adversarial_count] = reshape_extract_ad

    #                 print(adversarial_count)
    #                 print("변경되었다.")
    #                 break

    #             else:
    #                 extract_ad[adversarial_count] = temporary_adversarial

    #         if pred == extract_origin_label[adversarial_count]:
    #             break
