from utils import *


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

def make_targeted_cw(particular_class, model, x_data, y_data):

    particular_dataset = x_data[np.where(y_data == particular_class)]

    for targeted_num in trange(10):

        adversarial_dataset = []

        for dataset_count in trange(len(particular_dataset)):

            img = particular_dataset[dataset_count]         

            adversarial_data = targeted_cw(model, img, targeted_num)

            pred_adv_data = model.predict(tf.expand_dims(adversarial_data, 0))
            pred_adv_data = np.argmax(pred_adv_data)

            if targeted_num == pred_adv_data:
                adversarial_dataset.append(adversarial_data)
                
        adversarial_dataset = np.array(adversarial_dataset)
        pickle.dump(adversarial_dataset, open(f'./dataset/targeted_cw/{particular_class}-{targeted_num}','wb'))
