
attack_test, attack_label = [], []

for i in trange(len(x_test)):
    
    adv_data = eval('untargeted_fgsm')(model, x_test[i], 0.1) # (28, 28, 1)
    attack_test.append(adv_data)

    pred_adv_data = model.predict(tf.expand_dims(adv_data, 0))
    pred_adv_data = np.argmax(pred_adv_data)

    if y_test[i] != pred_adv_data:
        attack_label.append(1)
    else:
        attack_label.append(0)

attack_test, attack_label = np.array(attack_test), np.array(attack_label)

pickle.dump(attack_test, open(f'./dataset/fgsm/0.1_test','wb'))
pickle.dump(attack_label, open(f'./dataset/fgsm/0.1_label','wb'))
