from utils import *

def untargeted_cw(model, img):
    
    img = tf.expand_dims(img, 0)
    img = tf.cast(img, tf.float32)

    cw_data = carlini_wagner_l2(model, img)

    return cw_data[0]

def targeted_cw(model, img, target):
    
    target = tf.expand_dims(tf.convert_to_tensor(target, dtype=tf.int64), 0)

    img = tf.expand_dims(img, 0)
    img = tf.cast(img, tf.float32)

    cw_data = carlini_wagner_l2(model, img, y=target, targeted=True)

    return cw_data[0]