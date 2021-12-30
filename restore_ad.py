from utils import *

def check_tensor_ad(model, origin_data, origin_label):
    
    ad_data = pickle.load(open(f'./dataset/fgsm/0.1_test','rb'))
    ad_check = pickle.load(open(f'./dataset/fgsm/0.1_label','rb'))

    only_ad_list = np.where(ad_check == 1)[0]

    extract_origin = origin_data[only_ad_list]
    extract_ad = ad_data[only_ad_list]


    purturbation_data = extract_ad - extract_origin
    purturbation_data = np.reshape(purturbation_data, (purturbation_data.shape[0],-1))

    # print(purturbation_data[0])
    # print("-------")
    # print(np.sort(purturbation_data)[0])
    kkk = np.where(purturbation_data[0] == np.max(purturbation_data[0]))[0]

    for i in range(len(kkk)):
        point_position = kkk[i]

        xx = point_position/28
        yy = point_position%28

        kk = np.reshape(extract_ad[0], (784))
        kk[point_position] = 0

        ff = np.reshape(kk, (28, 28, 1))

        pred_adv_data = model.predict(tf.expand_dims(extract_ad[0], 0))
        pred_adv_data = np.argmax(pred_adv_data)
        print(pred_adv_data)

        pred_adv_data = model.predict(tf.expand_dims(ff, 0))
        pred_adv_data = np.argmax(pred_adv_data)
        print(pred_adv_data)
        time.sleep(4)

