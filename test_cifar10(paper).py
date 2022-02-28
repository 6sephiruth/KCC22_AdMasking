import tensorflow as tf
import numpy as np

# seeding randomness
tf.set_random_seed(451760341)
np.random.seed(216105420)

# Setting up training parameters
max_num_training_steps = 80000
num_output_steps = 100
num_summary_steps = 100
num_checkpoint_steps = 1000
step_size_schedule = [[0, 0.1], [40000, 0.01], [60000, 0.001]]
weight_decay = 0.0002
data_path = "cifar10_data",
momentum = 0.9
batch_size = 128

# Setting up the data and the model
raw_cifar = cifar10_input.CIFAR10Data(data_path)
global_step = tf.contrib.framework.get_or_create_global_step()
model = Model(mode='train')

tf.contrib.