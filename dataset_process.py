# from utils import *

#from src.attacks import cw
from cleverhans.tf2.attacks.carlini_wagner_l2 import carlini_wagner_l2

import pickle
import numpy as np
from tqdm import trange

import tensorflow as tf
import time

def make_origin_data(model, x_data, y_data):

    for label_class in range(10):

        particular_dataset = x_data[np.where(y_data == label_class)]
        pickle.dump(particular_dataset, open(f'./dataset/origin_data/{model.name}-{label_class}','wb'))


def make_targeted_cw(model, x_data, y_data):

    print("cw_adversarial data 생성 중")

    if model.name == 'paper_mnist':
        cw_make_batch_size = 100
    elif model.name == 'resnet50_cifar10':
        cw_make_batch_size = 1

    for label_class in trange(10):

        particular_dataset = x_data[np.where(y_data == label_class)]

        for ad_target_class in trange(10):
            adversarial_dataset = []

            if label_class != ad_target_class:

                # cw 데이터 생성에 메모리 한계가 있어,100개씩 끊어서 생성

                start_position = 0
                end_point = cw_make_batch_size

                while True:

                    part_particular_data = particular_dataset[start_position:end_point]
                    part_particular_data = tf.cast(part_particular_data, tf.float32)

                    target = np.empty([len(part_particular_data)])
                    target[0: len(part_particular_data)] = ad_target_class

                    adver_data = carlini_wagner_l2(model, part_particular_data, y=target, targeted=True)

                    for each_ad_data in range(len(adver_data)):
                        adversarial_dataset.append(adver_data[each_ad_data])

                    start_position += cw_make_batch_size
                    end_point += cw_make_batch_size

                    if len(particular_dataset) < start_position:
                        break

                adversarial_dataset = np.array(adversarial_dataset)
                pickle.dump(adversarial_dataset, open(f'./dataset/targeted_cw/{model.name}-{label_class}_{ad_target_class}','wb'))








    # for targeted_num in trange(10):

    #     adversarial_dataset = []

    #     for dataset_count in trange(len(particular_dataset)):

    #         img = particular_dataset[dataset_count]

    #         adversarial_data = targeted_cw(model, img, targeted_num)

    #         pred_adv_data = model.predict(tf.expand_dims(adversarial_data, 0))
    #         pred_adv_data = np.argmax(pred_adv_data)

    #         if targeted_num == pred_adv_data:
    #             adversarial_dataset.append(adversarial_data)

    #     adversarial_dataset = np.array(adversarial_dataset)
    #     pickle.dump(adversarial_dataset, open(f'./dataset/targeted_cw/{targeted_class}-{targeted_num}','wb'))

# if __name__ == '__main__':

# def make_untargeted_cw(model, dataset):

#     ### untargeted_CW 만들기
#     if exists(f'./dataset/untargeted_cw/test') and exists(f'./dataset/untargeted_cw/label'):
#         attack_test = pickle.load(open(f'./dataset/untargeted_cw/test','rb'))
#         attack_label = pickle.load(open(f'./dataset/untargeted_cw/label','rb'))

#     else:
#         attack_test, attack_label = [], []

#         for i in trange(len(dataset)):

#             adv_data = eval('untargeted_cw')(model, dataset[i]) # (28, 28, 1)
#             attack_test.append(adv_data)

#             pred_adv_data = model.predict(tf.expand_dims(adv_data, 0))
#             pred_adv_data = np.argmax(pred_adv_data)


#             pred_original = model.predict(tf.expand_dims(dataset[i], 0))
#             pred_original = np.argmax(pred_original)

#             if pred_original != pred_adv_data:
#                 attack_label.append(1)
#             else:
#                 attack_label.append(0)

#         attack_test, attack_label = np.array(attack_test), np.array(attack_label)

#         pickle.dump(attack_test, open(f'./dataset/untargeted_cw/test','wb'))
#         pickle.dump(attack_label, open(f'./dataset/untargeted_cw/label','wb'))

#     return attack_test, attack_label
