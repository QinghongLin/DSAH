import numpy as np
import tensorflow as tf
from utils.Encoder import Vgg

def compute_S(H):
    # Given an Hash codes Z_{nxl}, \
    #	compute their cosine similarity S_{nxn}.
    H = tf.compat.v1.nn.l2_normalize(H, axis=1)
    S = tf.compat.v1.matmul(H, tf.compat.v1.matrix_transpose(H))
    S = tf.reshape(S, [-1, 1])
    return S

def compute_P(H, temperature=1):
    # Compute the relatively similarity P_{nxn}
    P = tf.nn.softmax(H / temperature, axis=0)
    return P

def update_W_AND(H, W0, gamma):
    # Update the similarity matrix W0_{nxn}
    H_T = np.transpose(H)
    S = np.matmul(H, H_T)

    mu = np.mean( S[W0 >= 0] )
    std = np.std( S[W0 >= 0], ddof=1 )
    threshold = mu + gamma * std

    W0[S >= threshold] = 1
    percent = (W0>0).sum() / (W0.shape[0] * W0.shape[1])
    return W0, percent, threshold

def Inference(input_image, train_mode, codelen, temperature=1):
    vgg_encoder = Vgg('/public/dataset/vgg19.npy', codelen=codelen)
    vgg_encoder.build(input_image, train_mode=train_mode)

    hash_code = vgg_encoder.fc9

    S_ij = compute_S(hash_code)
    P_ij = compute_P(S_ij, temperature)

    return hash_code, S_ij, P_ij