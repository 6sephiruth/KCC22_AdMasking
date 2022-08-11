#from dataset_process import make_targeted_cw
# from utils import neuron_activation_analyze

from utils import *
#from attack_method import *
from dataset_process import *

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--params', dest='params')
    args = parser.parse_args()

    # fix of random seed value
    seed = 0
    tf.random.set_seed(seed)
    np.random.seed(seed)

    with open(f'./{args.params}', 'r') as f:
        params_loaded = yaml.safe_load(f)
    # designate gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = params_loaded['gpu_num']
    # enable memory growth
    physical_devices = tf.config.list_physical_devices('GPU')
    for d in physical_devices:
        tf.config.experimental.set_memory_growth(d, True)
    os.environ['TF_DETERMINISTIC_OPS'] = '0'

    # load params
    DATASET = params_loaded['dataset']
    MODEL_NAME = params_loaded['model_name']
    ATTACK_METHOD = params_loaded['attack_method']
    # ATTACK_EPS = params_loaded['attack_eps']

    datadir = ['model', 'model/' + MODEL_NAME, 'dataset', 'dataset/' + ATTACK_METHOD, 'dataset/' +'origin_data','result']
    mkdir(datadir)

    # dataset load
    if DATASET == 'mnist':
        train, test = mnist_data()
    elif DATASET == 'cifar10':
        train, test = cifar10_data()
        # loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    x_train, y_train = train
    x_test, y_test = test

    # load of DL models
    model = eval(params_loaded['model_name'])()

    checkpoint_path = f'model/{MODEL_NAME}'

    # save of model training & model load
    if exists(f'model/{MODEL_NAME}/saved_model.pb'):
        model = tf.keras.models.load_model(checkpoint_path)
    else:
        # MNIST 학습 checkpoint
        checkpoint = ModelCheckpoint(checkpoint_path,
                                    save_best_only=True,
                                    save_weights_only=True,
                                    monitor='val_accuracy',
                                    verbose=1)
        if DATASET == 'mnist':
            loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            model.compile(optimizer='adam',
                        loss=loss_fn,
                        metrics=['accuracy'])
            model.fit(x_train, y_train, epochs=10, shuffle=True, validation_data=(x_test, y_test), callbacks=[checkpoint],)

        elif DATASET == 'cifar10':

            model.compile(optimizer='SGD',
                loss='sparse_categorical_crossentropy',
                metrics = ['accuracy'])

            model.fit(x_train, y_train, epochs=10, shuffle=True, validation_data=(x_test, y_test), callbacks=[checkpoint], batch_size=64)

        model.save(checkpoint_path)
        model = tf.keras.models.load_model(checkpoint_path)

    model.trainable = False

    # origin & targeted attack 데이터 생성
    if not os.path.isfile(f'./dataset/targeted_cw/{model.name}-0_1'):

        make_origin_data(model, x_test, y_test)
        make_targeted_cw(model, x_test, y_test)

    # for normal_label in range(10):
    #     for attack_label in range(10):
    #         if normal_label == attack_label:
    #             dataset = pickle.load(open(f'./dataset/origin_data/{model.name}-{normal_label}','rb'))
    #         else:
    #             dataset = pickle.load(open(f'./dataset/targeted_cw/{model.name}-{normal_label}_{attack_label}','rb'))

    #         print("{}  {} 데이터셋 개수 {}:".format(normal_label, attack_label, len(dataset)))
    #         pred = model.predict(dataset)
    #         pred = np.argmax(pred, axis=1)
    #         print(pred[:100])

    #         pred_count = len(np.where(pred == attack_label)[0])
    #         print("{}  {} Acc {:.2f}:".format(normal_label, attack_label, pred_count/len(dataset)*100))


    # activation 저장
    if not os.path.isfile(f'./dataset/{model.name}-normal'):
        normal_activation = load_normal_activation(model)
        adver_activation = load_adver_activation(model)
    else:
        normal_activation = pickle.load(open(f'./dataset/{model.name}-normal','rb'))
        adver_activation = pickle.load(open(f'./dataset/{model.name}-adver','rb'))

    #activation_compare(normal_activation, adver_activation, 1)

    k_per = np.arange(1.00, 1.02, 0.01)

    # for N_LABEL in range(10):
    N_LABEL = 6
    for A_LABEL in range(10):

        if N_LABEL == A_LABEL:
            experiment_dataset = pickle.load(open(f'./dataset/origin_data/{model.name}-{N_LABEL}','rb'))
            out = f'./result/{model.name}-{N_LABEL}.tsv'
        else:
            experiment_dataset = pickle.load(open(f'./dataset/targeted_cw/{model.name}-{N_LABEL}_{A_LABEL}','rb'))
            out = f'./result/{model.name}-{N_LABEL}_{A_LABEL}.tsv'

        # for k_percent in range(99):
        for k_percent in k_per:

            #k_percent += 1

            # 10% 라고 가정
            masking_activation = activation_compare(normal_activation, adver_activation, k_percent)
            temp_masking_activation = masking_activation

            dataset_count = len(experiment_dataset)
            correct_count = 0
            attack_count = 0

            # 각 레이블당 판별 갯수
            cnt_labels = [0]*10

            if not os.path.exists(out):
                cols = ['Percent', 'Attack', 'Defense', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
                with open(out, 'w') as f:
                    f.write('\t'.join(cols))
                    f.write('\n')

            for each_data in experiment_dataset:
                masking_activation = temp_masking_activation

                if DATASET == 'mnist':
                    input_activation = np.reshape(each_data, (1,28,28,1))
                elif DATASET == 'cifar10':
                    input_activation = np.reshape(each_data, (1,32,32,3))

                for each_layer in model.model.layers:
                    #print(masking_activation.shape)
                    if len(masking_activation) == 0:
                        hidden_activation = each_layer(input_activation)
                    else:
                        hidden_activation = each_layer(input_activation)
                        hidden_activation = np.array(hidden_activation)
                        hidden_activation_shape = hidden_activation.shape
                        hidden_activation = np.reshape(hidden_activation, -1)

                        temp_masking_position = masking_activation[:len(hidden_activation)]

                        hidden_activation[np.where(temp_masking_position == 1)] = 0

                        input_activation = np.reshape(hidden_activation, hidden_activation_shape)

                        masking_activation = masking_activation[len(hidden_activation):]

                result_max = np.argmax(hidden_activation, axis=1)

                if result_max == N_LABEL:
                    correct_count += 1
                elif result_max == A_LABEL:
                    attack_count += 1

                # 레이블별 카운트
                cnt_labels[int(result_max)] += 1

            with open(out,'a') as f:
                f.write('\t'.join(map(lambda s: f'{s:.2f}',
                                      [k_percent,
                                       attack_count/dataset_count*100,
                                       correct_count/dataset_count*100]
                                    )))
                f.write('\t')
                f.write('\t'.join(map(str, cnt_labels)))

                """
                f.write('\t'.join(
                        [str(k_percent)+'%',
                         '  Attack: '+str(attack_count/dataset_count*100),
                         ' Defense: ' + str(correct_count/dataset_count*100)
                        ]) + '\t')
                f.write('\t'.join(
                            map(lambda i: f' {i[0]+1} : {i[1]}', enumerate(cnt_labels))
                            )
                        )
                """
                #f.write(' 0 : '+str(label_0), ' 1 : '+str(label_1), ' 2 : '+str(label_2), ' 3 : '+str(label_3), ' 4 : '+str(label_4), ' 5 : '+str(label_5), ' 6 : '+str(label_6), ' 7 : '+str(label_7), ' 8 : '+str(label_8), ' 9 : '+str(label_9)]) + '\t')
                # f.write('\t'.join(['0:'+str(label_0), '1:'+str(label_1), '2:'+str(label_2), '3:'+str(label_3), '4:'+str(label_4), '5:'+str(label_5), '6:'+str(label_6), '7:'+str(label_7), '8:'+str(label_8), '9:'+str(label_9) ]) + '\t')

                f.write('\n')

if __name__ == '__main__':
    main()



""" unused code
    # weights = model.model.get_weights()
    # for i,w in enumerate(weights):
    #     print(w.shape)
    #     print(i)

    #     if i >= 6:
    #         break

    # ### at inference time
    # masks = None
    # y_pred = model(x_test)
    # y_pred_adv = model(x_adv)

    # y_pred_mask = model.call_with_mask(x_test,masks)
    # y_pred_mask_adv = model.call_with_mask(x_adv,masks)

    # exit()




    # print(model.model.summary())
    # print(model.model.get_weights()[0].shape)
    # print(model.model.get_weights()[1].shape)
    # print(model.model.get_weights()[2].shape)
    # print(model.model.get_weights()[3].shape)

    # intermediate_layer_model = tf.keras.Model(inputs=model.model.input, outputs=model.model.layers[0].output)
    # layer_1_output_1 = np.array(intermediate_layer_model(dataset))


    # get_1st_layer_output = K.function([model.layers[0].input],
    #                                 [model.layers[1].output])
    # layer_output = get_1st_layer_output([X])

    # print(model.model.layers[0].name) # 이름 얻는 방법
    # print(model.model.layers[1].name)
    # print(model.model.layers[2].name)

    # a = model.model.get_layer("dense_2").get_weights()
    # print("-----")
    # print(a)
    # print(a[0].shape)
    # print(a[1].shape)
    # print(a[2].shape)




    # ##########################

    # result = threshold_activation(model, input_dataset)

    # print(activation_dataset.shape)
    # print(result.shape)
    # print(result.shape)


    # for layers in model.model.layers:
    #     a = layers.get_weights()[0]
    #     print(a)
    #     print(a.shape)
    #     #print(a.shape)
    #     time.sleep(4)
    # print(model.model.summary())
    # intermediate_layer_model = tf.keras.Model(inputs=model.model.input, outputs=model.model.layers[1].output)
    # intermediate_output = intermediate_layer_model(x_test)

    # intermediate_output =  intermediate_output.numpy()
    # print(type(intermediate_output))
    # print(intermediate_output.reshape((len(intermediate_output), -1)).shape)



    # # targeted attack 데이터 생성
    # if not os.path.isfile('./dataset/targeted_cw/{model.name}-0_0'):
    #     make_targeted_cw(model, x_test, y_test)

#from pruning_defense import *

# for i in range(10):

#     for j in range(10):
#         if i == j:
#             particular_data_position = np.where(y_test == i)
#             particular_data = x_test[particular_data_position]

#             normal_prunning(model, particular_data, i)
        # else:
        #     particular_data_position = np.where(y_test == i)
        #     particular_data = x_test[particular_data_position]

        #     adver_data = pickle.load(open(f'./model/paper_mnist/dataset/targeted_cw/{i}-{j}','rb'))

        #     top_adversarial_activation_select(model, particular_data, adver_data, i, j)

# for analysis_num in range(10):
#     for j in range(10):
#         print("----------------------")
#         print(analysis_num, j)
#         x_data = pickle.load(open(f'./model/paper_mnist/dataset/perfect_targeted_cw/{analysis_num}-{j}','rb'))

#         # pp = model.predict(x_data)
#         # pp = tf.nn.softmax(pp)
#         # pp = np.argmax(pp, axis=1)
#         # print(pp[:200])
#         # time.sleep(1)
#         mnist_model_compress(analysis_num ,model, x_data)
#         print()





# for i in range(10):
#     for j in range(10):
#         dataset = pickle.load(open(f'./model/paper_mnist/dataset/perfect_targeted_cw/{i}-{j}','rb'))
#         print("{}-------{}".format(i,j))
#         mnist_model_compress(i ,model, dataset)
#         print()



# 아래는 원래 주석 없었음.
# x_data = pickle.load(open(f'./model/paper_mnist/dataset/perfect_targeted_cw/x_full_data','rb'))
# y_data = pickle.load(open(f'./model/paper_mnist/dataset/perfect_targeted_cw/y_full_data','rb'))

# mnist_model_compress(model, x_data[90000:], y_data[90000:])
"""