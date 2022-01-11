from os import sched_getaffinity
from dataset_process import make_targeted_cw
from utils import neuron_activation_analyze

from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--params', dest='params')
args = parser.parse_args()

with open(f'./{args.params}', 'r') as f:
    params_loaded = yaml.safe_load(f)

# designate gpu
os.environ['CUDA_VISIBLE_DEVICES'] = params_loaded['gpu_num']

seed = 0
tf.random.set_seed(seed)
np.random.seed(seed)

# enable memory growth
physical_devices = tf.config.list_physical_devices('GPU')

for d in physical_devices:
    tf.config.experimental.set_memory_growth(d, True)

os.environ['TF_DETERMINISTIC_OPS'] = '0'

ATTACK_METHOD = params_loaded['attack_method']
DATASET = params_loaded['dataset']

datadir = ['model', 'model/' + DATASET, 'dataset', 'dataset/' + ATTACK_METHOD]
mkdir(datadir)

ATTACK_EPS = params_loaded['attack_eps']

# dataset load
if DATASET == 'mnist':
    train, test = mnist_data()
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

elif DATASET == 'cifar10':
    train, test = cifar10_data()
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
x_train, y_train = train
x_test, y_test = test

model = eval(params_loaded['model_name'])()

checkpoint_path = f'model/{DATASET}'

if exists(f'model/{DATASET}/saved_model.pb'):

    model = tf.keras.models.load_model(checkpoint_path)

else:

    # MNIST 학습 checkpoint
    checkpoint = ModelCheckpoint(checkpoint_path, 
                                save_best_only=True, 
                                save_weights_only=True, 
                                monitor='val_loss',
                                verbose=1)
    if DATASET == 'mnist':

        model.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

        model.fit(x_train, y_train, epochs=10, shuffle=True, validation_data=(x_test, y_test), callbacks=[checkpoint],)
    
    # if DATASET == 'cifar10':

    #     model.compile(optimizer='adam',
    #                 loss='sparse_categorical_crossentropy',
    #                 metrics=['accuracy'])

    #     model.fit(x_train, y_train, epochs=40, shuffle=True, validation_data=(x_test, y_test), callbacks=[checkpoint],)

    model.save(checkpoint_path)
    model = tf.keras.models.load_model(checkpoint_path)

model.trainable = False

# print(model.evaluate(x_test, y_test))

# change_cw_data = pickle.load(open(f'./dataset/cw_specific/change_data','rb'))
# change_cw_data = pickle.load(open(f'./dataset/cw_specific/0','rb'))
# print(change_cw_data.shape)


# check_tensor_ad(model, x_test, y_test)


# neuron_check(model)


# make_targeted_cw(2, model, x_test, y_test)


########## 오늘 저녁에 그대로 코드 실행#########################################
########## 
# x_test = pickle.load(open(f'./dataset/targeted_cw/0-1','rb'))
# neuron_activation_analyze(model, x_test, 0, 1)
##########
##########



model_weight_analysis(0, model)











# ad_label = pickle.load(open(f'./dataset/fgsm/0.1_label','rb'))

# ad_list = np.where(ad_label == 1)[0]

# extract_origin_label = y_test[ad_list]

# extract_origin = x_test[ad_list]
# extract_ad = ad_data[ad_list]

# num = 33

# pred = model.predict(tf.expand_dims(extract_origin[num], 0))
# pred = np.argmax(pred)
# print("정상 {}".format(pred))

# pred = model.predict(tf.expand_dims(extract_ad[num], 0))
# pred = np.argmax(pred)
# print("적대적 {}".format(pred))

# recover = pickle.load(open(f'./dataset/fgsm/0.1_recover','rb'))

# pred = model.predict(tf.expand_dims(recover[num], 0))
# pred = np.argmax(pred)
# print("복구 {}".format(pred))

# plt.imshow(recover[num])
# plt.savefig("./img/{}.png".format(num))