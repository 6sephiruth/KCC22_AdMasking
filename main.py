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

    datadir = ['model', 'model/' + MODEL_NAME, 'dataset', 'dataset/' + ATTACK_METHOD, 'dataset/' +'origin_data']
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
                                    monitor='val_acc',
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
    # if not os.path.isfile(f'./dataset/targeted_cw/{model.name}-0_1'):
        
    #     make_origin_data(model, x_test, y_test)
    #     make_targeted_cw(model, x_test, y_test)

    # print(model.model.summary())
    # print(model.model.get_weights()[0].shape)
    # print(model.model.get_weights()[1].shape)
    # print(model.model.get_weights()[2].shape)
    # print(model.model.get_weights()[3].shape)

    # intermediate_layer_model = tf.keras.Model(inputs=model.model.input, outputs=model.model.layers[0].output)
    # layer_1_output_1 = np.array(intermediate_layer_model(dataset))


    from keras.models import Model 




    # get_1st_layer_output = K.function([model.layers[0].input],
    #                                 [model.layers[1].output])
    # layer_output = get_1st_layer_output([X])
    
    # print(model.model.layers[0].name) # 이름 얻는 방법
    # print(model.model.layers[1].name)
    # print(model.model.layers[2].name)

    a = model.model.get_layer("dense_2").get_weights()
    print("-----")
    print(a)
    print(a[0].shape)
    print(a[1].shape)
    print(a[2].shape)




    ##########################
    def load_normal_activation():
        total_normal_activation = []

        for each_label in range(10):

            each_normal_input = pickle.load(open(f'./dataset/origin_data/{model.name}-{each_label}','rb'))

            each_threshold_actvation = threshold_activation(model, each_normal_input)

            for i in range(len(each_normal_input)):
                total_normal_activation.append(each_threshold_actvation[i])

        total_normal_activation = np.array(total_normal_activation)



    def load_adver_activation():
        total_adver_activation = []

        for each_label in range(10):
            
            temporarily_actvation = None

            for each_attack_label in range(10):

                if each_label != each_attack_label:
            
                    each_adver_input = pickle.load(open(f'./dataset/targeted_cw/{model.name}-{each_label}_{each_attack_label}','rb'))

                    each_threshold_actvation = threshold_activation(model, each_adver_input)
                    
                    if temporarily_actvation == None:
                        temporarily_actvation = each_threshold_actvation
                    else:
                        temporarily_actvation += each_threshold_actvation

            each_threshold_actvation[np.where(each_threshold_actvation > 0)] = 1

            for i in range(len(each_normal_input)):
                total_adver_activation.append(each_threshold_actvation[i])

        total_adver_activation = np.array(total_adver_activation)









    #     normal_activation = 


    # result = threshold_activation(model, input_dataset)

    # print(actvation_dataset.shape)
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

if __name__ == '__main__':
    main()



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

        #     top_adversarial_actvation_select(model, particular_data, adver_data, i, j)

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
