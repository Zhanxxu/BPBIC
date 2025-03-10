#import tensorflow.keras.backend as K

#def DiceLoss(targets, inputs, smooth=1e-6):
    # flatten label and prediction tensors
#    inputs = K.flatten(inputs)
#    targets = K.flatten(targets)

#    intersection = K.sum(K.dot(targets, inputs))
#    dice = (2 * intersection + smooth) / (K.sum(targets) + K.sum(inputs) + smooth)
#    return 1 - dice

import tensorflow as tf
tf.compat.v1.enable_eager_execution()

def DiceLoss(n_classes, logits, label, smooth=1.e-5):
    epsilon = 1.e-6
    alpha = 2.0   # 这个是dice coe的系数，
    y_true = tf.one_hot(label, n_classes)
    softmax_prob = tf.nn.softmax(logits)
    print("{}".format(softmax_prob.numpy()))
    y_pred = tf.clip_by_value(softmax_prob, epsilon, 1. - epsilon)

    y_pred_mask = tf.multiply(y_pred, y_true)
    common = tf.multiply((tf.ones_like(y_true) - y_pred_mask), y_pred_mask)
    nominator = tf.multiply(tf.multiply(common, y_true), alpha) + smooth
    denominator = common + y_true + smooth
    dice_coe = tf.divide(nominator, denominator)
    return tf.reduce_mean(tf.reduce_max(1 - dice_coe, axis=-1))