from utils import *

def untargeted_fgsm(model, img, eps):
    """
    untargeted FGSM의 적대적 예제 생성 함수

    :model: 학습된 인공지능 모델.
            공격자는 인공지능 모델의 모든 파라미터 값을 알고있음.
    :img:   적대적 예제로 바꾸고자 하는 이미지 데이터
    :eps:   적대적 예제에 포함될 noise 크기 결정.
            eps가 크면 클 수록, 적대적 공격은 성공률이 높지만,
            적대적 예제의 시각적 표현이 높아지는 단점이 있음.
    :return: tensor 형태의 적대적 예제

    """

    img = tf.expand_dims(img, 0)

    fgsm_data = fast_gradient_method(model, img, eps, np.inf)

    return fgsm_data[0]

def untargeted_cw(model, img):
    
    img = tf.expand_dims(img, 0)
    img = tf.cast(img, tf.float32)

    cw_data = carlini_wagner_l2(model, img)

    return cw_data[0]

# def targeted_cw(model, img, target):
    
#     target = tf.expand_dims(tf.convert_to_tensor(target, dtype=tf.int64), 0)

#     img = tf.expand_dims(img, 0)
#     img = tf.cast(img, tf.float32)

#     cw_data = carlini_wagner_l2(model, img, y=target, targeted=True)

#     return cw_data[0]

def targeted_cw(model, img, target):
    
    cw_data = carlini_wagner_l2(model, img, y=target, targeted=True)

    return cw_data