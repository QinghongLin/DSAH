import numpy as np
import tensorflow as tf

def H_Loss(hash_code, W, P, B,
           PIC, batch_size):
    hash_code = tf.compat.v1.nn.l2_normalize(hash_code, axis=1)

    # Pairwise Information Content
    P = tf.reshape(P, [batch_size, batch_size])
    if PIC == 1:
        A = -tf.math.log(P)
    elif PIC == 2:
        A = -tf.math.log(1 - P)
    else:
        A = 1

    loss_1 = tf.reduce_mean(A * tf.square(tf.matmul(hash_code, tf.transpose(hash_code)) - W))

    # Quantitative Loss
    quantitative_loss = tf.compat.v1.reduce_mean(tf.square(B - hash_code))

    return loss_1, quantitative_loss