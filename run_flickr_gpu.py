import os
import tqdm
import scipy
import argparse
import numpy as np
import scipy.io as scio
import tensorflow as tf
from utils.loss import H_Loss
from utils.logger import setup_logger
from utils.dataloader import data_iterator
from utils.Update import Inference, update_W_AND
from utils.metric import eval_map

def parse_args():
    """parse the model arguments"""
    parser_arg = argparse.ArgumentParser()
    # basic setting
    parser_arg.add_argument('--num_gpus', type=int, default='0', help='Number of GPU')
    parser_arg.add_argument('--version', type=str, default='debug', help='Record the version of this times')
    parser_arg.add_argument('--codelen', type=int, default=32, help='Length of binary code expected to be generated')
    parser_arg.add_argument('--batch_size', type=int, default=50, help='batch size')

    # initial similarity matrix
    parser_arg.add_argument('--W', type=str, default='S_K1_500_K2_500_I', help='Similarity Matrix')

    # training hyperparameter
    parser_arg.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser_arg.add_argument('--round', type=int, default=1, help='How many round of the AND')
    parser_arg.add_argument('--max_epoch', type=int, default=30, help='Maximum number of iterations for each round')
    parser_arg.add_argument('--save_epoch', type=int, default=99, help='How many iterations to save')

    # DSAH components
    parser_arg.add_argument('--PIC', type=int, default=1, help='whether to assign the pairwise weights by PIC')
    parser_arg.add_argument('--AND', type=int, default=1, help='whether to update the similarity by AND')

    # DSAH loss
    parser_arg.add_argument('--alpha', type=float, default=1, help='Similarity loss term parameters')
    parser_arg.add_argument('--beta', type=float, default=0, help='Quantitative loss term parameters')

    # DSAH hyperparameter
    parser_arg.add_argument('--temp', type=float, default=1,
                            help='temperature factor for PIC')
    parser_arg.add_argument('--gamma', type=float, default=0,
                            help='\mu + \gamma * \sigma')
    return parser_arg.parse_args()

args = parse_args()
version = args.version
num_gpus = args.num_gpus
codelen = args.codelen
batch_size = args.batch_size
W = args.W
learning_rate = args.lr
round = args.round
max_epoch = args.max_epoch
save_epoch = args.save_epoch
PIC = args.PIC
AND = args.AND
alpha = args.alpha
beta = args.beta
temp = args.temp
gamma = args.gamma

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = str(num_gpus)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# data path & saving path
data_path = '/public/dataset/flickr25k/FLICKR25K.mat'
save_path = '/lqh/DeepUnsupervisedHashing/DSAH/data/output/flickr25k/' + version
logfile_dir = '/lqh/DeepUnsupervisedHashing/DSAH/data/log/flickr25k/'
logfile_name = version + '.log'

logger = setup_logger(logfile_name, logs_dir=logfile_dir, also_stdout=False)
logger.info('dataset: flickr')
for arg in vars(args):
    print(arg, getattr(args, arg))
    logger.info(arg + ' : '+ str(getattr(args, arg)))
print()
logger.info('')

# data preparation
print('0. loading the dataset')
train_data = scio.loadmat(data_path)['train_data']
# todo: please providing the similarity matrix for training
W = scio.loadmat('/lqh/DeepUnsupervisedHashing/W/flickr25k/' + W + '.mat')['S']

num_of_train = train_data.shape[-1]
total_batch = int(np.floor(num_of_train / batch_size))
img224 = []

# Data preprocessing
for i in tqdm.tqdm(range(num_of_train)):
    t = train_data[:, :, :, i]
    if train_data.shape[0] != 224:
        image = scipy.misc.imresize(t, [224, 224])
        img224.append(image)
    else:
        img224.append(t)
img224 = np.array(img224)
del train_data

# Construct the graph
graph = tf.Graph()
config = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=True)
config.gpu_options.allow_growth = True

with graph.as_default(), tf.device('/gpu:' + str(num_gpus)):
    tower_grads = []
    train_mode = tf.compat.v1.placeholder(tf.bool)
    alpha_s = tf.compat.v1.placeholder(tf.float32, shape=[])
    beta_s = tf.compat.v1.placeholder(tf.float32, shape=[])
    eta_s = tf.compat.v1.placeholder(tf.float32, shape=[])
    B_s = tf.compat.v1.placeholder(tf.float32, [None, codelen])
    W_s = tf.compat.v1.placeholder(tf.float32, [None, None])
    input = tf.compat.v1.placeholder(tf.float32, [None, 224, 224, 3], name='input')

    learning_rate_s = tf.compat.v1.placeholder(tf.float32, shape=[])
    global_step = tf.compat.v1.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
    learning_rate_decay = tf.compat.v1.train.exponential_decay(learning_rate_s, global_step, 200, 0.99, staircase=True)
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate_decay, epsilon=1.0)

    with tf.compat.v1.variable_scope(tf.compat.v1.get_variable_scope()):
        with tf.compat.v1.device('/gpu:' + str(num_gpus)):
            with tf.name_scope('tower_0') as scope:
                H_code, S, P = Inference(input, train_mode=train_mode, codelen=codelen, temperature=temp)
                loss_1, quantitative_loss = \
                    H_Loss(hash_code=H_code, W=W_s, P=P, B=B_s, PIC=PIC, batch_size=batch_size)
                loss = alpha_s * loss_1 + beta_s * quantitative_loss

                grads = optimizer.compute_gradients(loss)
                train_g = optimizer.apply_gradients(grads, global_step=global_step)

    sess = tf.compat.v1.InteractiveSession(graph=graph, config=config)
    sess.run(tf.compat.v1.global_variables_initializer())

def Save_binary_code(data_path, save_path, batch_size):
    mat = scio.loadmat(data_path)
    dataset = mat['data_set']
    train = mat['train_data']
    test = mat['test_data']
    del mat

    dataset_images = []  # This is for dataset
    train_images = []  # This is for train
    test_images = []
    dataset_H = []
    train_H = []
    test_H = []

    for i in tqdm.tqdm(range(dataset.shape[-1])):
        t = dataset[:, :, :, i]
        if dataset.shape[0] != 224:
            image = scipy.misc.imresize(t, [224, 224])
            dataset_images.append(image)
        else:
            dataset_images.append(t)
    dataset_images = np.array(dataset_images)
    del dataset

    for i in tqdm.tqdm(range(train.shape[-1])):
        t = train[:, :, :, i]
        if train.shape[0] != 224:
            image = scipy.misc.imresize(t, [224, 224])
            train_images.append(image)
        else:
            train_images.append(t)
    train_images = np.array(train_images)
    del train

    for i in tqdm.tqdm(range(test.shape[-1])):
        t = test[:, :, :, i]
        if test.shape[0] != 224:
            image = scipy.misc.imresize(t, [224, 224])
            test_images.append(image)
        else:
            test_images.append(t)
    test_images = np.array(test_images)
    del test

    print('generate binary code...')
    for i in tqdm.tqdm(range(0, len(dataset_images), batch_size)):
        batch_image = dataset_images[i:i + batch_size]
        H_all = sess.run(H_code, feed_dict={input: batch_image, train_mode: False})
        if i == 0:
            dataset_H = H_all
        else:
            dataset_H = np.concatenate((dataset_H, H_all), axis=0)
    del dataset_images

    for i in tqdm.tqdm(range(0, len(train_images), batch_size)):
        batch_image = train_images[i:i + batch_size]
        H_all = sess.run(H_code, feed_dict={input: batch_image, train_mode: False})
        if i == 0:
            train_H = H_all
        else:
            train_H = np.concatenate((train_H, H_all), axis=0)
    del train_images

    for i in tqdm.tqdm(range(0, len(test_images), batch_size)):
        batch_image = test_images[i:i + batch_size]
        H_test = sess.run(H_code, feed_dict={input: batch_image, train_mode: False})
        if i == 0:
            test_H = H_test
        else:
            test_H = np.concatenate((test_H, H_test), axis=0)
    del test_images

    print('save code...')
    np.savez(save_path + '.npz', dataset=dataset_H, train=train_H, test=test_H)
    print('save done!')

# 1.Train Encoder parameters
for r in range(round):
    epoch = 0
    while epoch < max_epoch:
        # Update network
        print('Training the network...')
        print("Round: {0}, epoch: {1}".format(r, epoch))
        mean_loss = []
        mean_loss_1 = []
        mean_loss_quantitative = []
        iter_ = data_iterator(img224, batch_size)
        for i in tqdm.tqdm(range(total_batch)):
            next_batch, next_idx = iter_.__next__()
            W_batch = W[next_idx, :][:, next_idx]
            B_batch = np.sign(sess.run(H_code, feed_dict={input: next_batch, train_mode: False}))
            loss_, loss_1_, quantitative_loss_, S_, P_, H_batch, _ = sess.run(
                [loss, loss_1, quantitative_loss, S, P, H_code, train_g],
                feed_dict={input: next_batch, W_s: W_batch, B_s: B_batch,
                           alpha_s: alpha, beta_s: beta,
                           learning_rate_s: learning_rate, train_mode: True})
            mean_loss.append(loss_)
            mean_loss_1.append(alpha * loss_1_)
            mean_loss_quantitative.append(beta * quantitative_loss_)
            print("total_mean_loss:{:.4f}, mean_loss_1:{:.4f}, quantitative_loss:{:.4f},".
                  format(np.mean(mean_loss), np.mean(mean_loss_1), np.mean(mean_loss_quantitative)) )

        logger.info("Round: {0}, epoch: {1}".format(r, epoch))
        logger.info("mean_loss: %f" % np.mean(mean_loss))
        logger.info("mean_loss_1: %f" % np.mean(mean_loss_1))
        logger.info("mean_loss_quantitative: %f" % np.mean(mean_loss_quantitative))
        logger.info('')

        if (epoch + 1) % save_epoch == 0:
            temp_save_path = save_path + '_' + str(r) + '_epoch_' + str(epoch + 1)
            Save_binary_code(data_path, temp_save_path, 50)
        epoch += 1

    # Saving the model for each round
    print('Saving the final model...')
    temp_save_path = save_path + '_' + str(r) + '_epoch_' + str(max_epoch)
    Save_binary_code(data_path, temp_save_path, 50)

    # Update the similarity matrix by AND
    if AND == 1:
        print("Update the Similarity via Adaptive Similarity Learning...")
        H_total = np.load(temp_save_path + '.npz')['train']
        W, percent, mu = update_W_AND(H_total, W, gamma)
        print("The percentage of postive similarity: %f" % percent + "%")
        logger.info("The percentage of postive similarity: %f" % percent + "%")
        print("The threshold of mu as: " + str(mu))
        logger.info("The threshold of mu as: " + str(mu))

# 3.Evaluating
print('Evaluating...')
map_1000, map_5000, map_all = eval_map(temp_save_path, dataset='flickr')
print('codelen =', codelen, ', map:{:.4f}, map_1000:{:.4f}, map_5000:{:.4f}'.
      format(map_all, map_1000, map_5000))
logger.info('codelen:' + str(codelen) + ' map:{:.4f}, map_1000:{:.4f}, map_5000:{:.4f}'.
            format(map_all, map_1000, map_5000))