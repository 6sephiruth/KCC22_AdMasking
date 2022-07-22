# from utils import *

#from src.attacks import cw
from cleverhans.tf2.attacks.carlini_wagner_l2 import carlini_wagner_l2

import pickle
import numpy as np
from tqdm import trange

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
import tensorflow as tf
import time
def make_targeted_cw(model, x_data, y_data):

    for label_class in trange(10):

        particular_dataset = x_data[np.where(y_data != label_class)]

        for targeted_class in trange(10):

            if label_class != targeted_class:

                adversarial_dataset = []

                start_position = 0
                end_point = 100

                while(len(particular_dataset) > start_position):

                    part_particular_data = particular_dataset[start_position:end_point]

                    target = np.ones(len(particular_dataset))
                    print(target)
                    where = np.where(target == 1.)
                    
                    target[where] = i
                    print(target)
                    exit()
                    #adversarial_dataset.append(carlini_wagner_l2(model, particular_dataset, y=target, targeted=True))







                # targeted_class = tf.expand_dims(tf.convert_to_tensor(targeted_class, dtype=tf.int64), 0)
                # print(targeted_class)
                # print(particular_dataset.shape)












                # ad_data = carlini_wagner_l2(model, particular_dataset, y=targeted_class, targeted=True)
                # pickle.dump(ad_data, open(f'./dataset/targeted_cw/{label_class}-{targeted_class}','wb'))
                time.sleep(1)

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

