#from dataset_process import make_targeted_cw
# from utils import neuron_activation_analyze

from utils import *
#from attack_method import *

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

    datadir = ['model', 'model/' + MODEL_NAME, 'dataset', 'dataset/' + ATTACK_METHOD]
    mkdir(datadir)

    # dataset load
    if DATASET == 'mnist':
        train, test = mnist_data()
    elif DATASET == 'cifar10':
        train, test = cifar10_data()
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

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

        # elif DATASET == 'cifar10':

        model.save(checkpoint_path)
        model = tf.keras.models.load_model(checkpoint_path)

    model.trainable = False

    from dataset_process import make_targeted_cw
    make_targeted_cw(model, x_test, y_test)


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
