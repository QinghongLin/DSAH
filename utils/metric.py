import numpy as np
import h5py
import scipy.io as scio

def mean_average_precision(qB, rB, query_L, retrieval_L, k):
    database_code = rB
    validation_code = qB
    database_labels = retrieval_L
    validation_labels = query_L
    query_num = validation_code.shape[0]

    sim = np.dot(database_code, validation_code.T)
    ids = np.argsort(-sim, axis=0)
    APx = []

    for i in range(query_num):
        label = validation_labels[i, :]
        label[label == 0] = -1
        idx = ids[:, i]
        imatch = np.sum(database_labels[idx[0:k], :] == label, axis=1) > 0
        relevant_num = np.sum(imatch)
        Lx = np.cumsum(imatch)
        Px = Lx.astype(float) / np.arange(1, k + 1, 1)
        if relevant_num != 0:
            APx.append(np.sum(Px * imatch) / relevant_num)

    return np.mean(np.array(APx))

def calc_map_k(qB, rB, query_L, retrieval_L, k):
    num_query = query_L.shape[0]
    map = 0
    for iter in range(num_query):
        gnd2 = (np.dot(query_L[iter, :], retrieval_L.transpose()) > 0).astype(np.float32)
        hamm = calc_hammingDist(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd = gnd2[ind[0:k]]
        tsum = np.sum(gnd)
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, int(tsum))
        tindex = np.asarray(np.where(gnd == 1)) + 1.0
        map = map + np.mean(count / (tindex))
    map = map / num_query
    return map

def calc_map(qB, rB, query_L, retrieval_L):
    num_query = query_L.shape[0]
    map = 0
    for iter in range(num_query):
        gnd2 = (np.dot(query_L[iter, :], retrieval_L.transpose()) > 0).astype(np.float32)
        hamm = calc_hammingDist(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd = gnd2[ind[:]]
        tsum = np.sum(gnd)
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, int(tsum))
        tindex = np.asarray(np.where(gnd == 1)) + 1.0
        map = map + np.mean(count / (tindex))
    map = map / num_query
    return map

def precision_top_n(qB, rB, query_L, retrieval_L, k):
    num_query = query_L.shape[0]
    pre = 0
    for iter in range(num_query):
        gnd2 = (np.dot(query_L[iter, :], retrieval_L.transpose()) > 0).astype(np.float32)
        hamm = calc_hammingDist(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd = gnd2[ind[0:k]]
        tsum = np.sum(gnd)
        pre = pre + tsum/k
    pre = pre / num_query
    return pre

def recall_top_n(qB, rB, query_L, retrieval_L, k):
    num_query = query_L.shape[0]
    recall = 0
    for iter in range(num_query):
        gnd2 = (np.dot(query_L[iter, :], retrieval_L.transpose()) > 0).astype(np.float32)
        same_class = np.sum(gnd2)
        hamm = calc_hammingDist(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd = gnd2[ind[0:k]]
        tsum = np.sum(gnd)
        recall = recall + tsum/same_class
    recall = recall / num_query
    return recall

def calc_hammingDist(B1, B2):
    q = B2.shape[1]
    distH = 0.5 * (q - np.dot(B1,B2.transpose()))
    return distH

def one_hot_label(single_label):
    num_label = np.max(single_label)+1
    num_samples = single_label.size
    one_hot_label = np.zeros([num_samples, num_label], int)
    for i in range(num_samples):
        one_hot_label[i, single_label[i][0]] = 1
    return one_hot_label

def eval_map(save_path, dataset):
    assert dataset in ['cifar', 'flickr', 'nus']
    if dataset == 'cifar':
        mat = scio.loadmat('/public/dataset/cifar-10/cifar-10.mat')
        dataset_L = one_hot_label(mat['dataset_L'])
        test_L = one_hot_label(mat['test_L'])
    elif dataset == 'flickr':
        mat = scio.loadmat('/public/dataset/flickr25k/FLICKR25K.mat')
        dataset_L = mat['dataset_L']
        test_L = mat['test_L']
    elif dataset == 'nus':
        mat = h5py.File('/public/dataset/nus-wide/nus-wide.mat', mode='r')
        dataset_L = np.array(mat['dataset_L']).T
        test_L = np.array(mat['test_L']).T
    del mat

    hash_code = np.load(save_path + '.npz')
    dataset = np.sign(hash_code['dataset'])
    test = np.sign(hash_code['test'])

    map_1000 = calc_map_k(test, dataset, test_L, dataset_L, 1000)
    map_5000 = calc_map_k(test, dataset, test_L, dataset_L, 5000)
    map_all = calc_map(test, dataset, test_L, dataset_L)
    return map_1000, map_5000, map_all

    # result = precision_top_n(test, dataset, test_L, dataset_L, 100)
    # return result

if __name__ == 'main':
    pass